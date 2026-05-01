#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Vendored notice:
# - This file is derived from upstream SeedVR/SeedVR2 inference code (Apache-2.0).
#   See: https://github.com/ByteDance-Seed/SeedVR
# - SPDX-License-Identifier remains Apache-2.0 for compliance; NVIDIA modifications are covered by the
#   SPDX-FileCopyrightText line(s) above.
# - Modifications from vendored original:
#   1. Removed `from __future__ import annotations` (project convention — Python 3.12 native hints).
#   2. Extracted `cut_videos` from nested function inside `generation_loop` to module level.
"""
SeedVR2 long-video inference via sliding windows + overlap stitching.

Wraps the upstream SeedVR2 inference code with windowed processing
for long videos: https://github.com/ByteDance-Seed/SeedVR

This script is intentionally standalone: it does NOT modify the original inference scripts.
Instead, it imports them as "variants" and reuses their:
- configure_runner(sp_size)
- generation_step(runner, text_embeds_dict, cond_latents)

Design goals:
- Handle long videos without decoding / processing the full T frames at once.
- Keep overlap alignment stable (so blending does not produce ghosting).
- Fail-fast per video: the first failed window stops that video, but the batch continues.
- Write failures to a single log (failures.log); do not crash the whole batch for partial failures.

Decoding:
- Uses torchvision.io.VideoReader in streaming mode to build windows in frame-space.
  This avoids seek-based decoding misalignment that can cause overlap "ghosting".

Stitching:
- Default: blend overlap frames.
- Use --no_blend_overlap to disable blending (overlap frames are dropped from the later window).
"""

import argparse
import datetime
import gc
import importlib.util
import logging
import os
import shutil
import sys
import traceback
from fractions import Fraction
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import av
import mediapy
import numpy as np
import torch

# Ensure SeedVR is importable without relying on global PYTHONPATH.
_SEEDVR_ROOT = str(os.getenv("SEEDVR_ROOT", "")).strip()
if (
    importlib.util.find_spec("common") is None
    and _SEEDVR_ROOT
    and os.path.isdir(_SEEDVR_ROOT)
    and _SEEDVR_ROOT not in sys.path
):
    sys.path.insert(0, _SEEDVR_ROOT)

from common.distributed import get_device  # noqa: E402
from common.distributed.advanced import (  # noqa: E402
    get_data_parallel_rank,
    get_data_parallel_world_size,
    get_sequence_parallel_rank,
    get_sequence_parallel_world_size,
)
from common.partition import partition_by_groups, partition_by_size  # noqa: E402
from common.seed import set_seed  # noqa: E402
from data.image.transforms.divisible_crop import DivisibleCrop  # noqa: E402
from data.image.transforms.na_resize import NaResize  # noqa: E402
from data.video.transforms.rearrange import Rearrange  # noqa: E402
from einops import rearrange  # noqa: E402
from torchvision.io import VideoReader, read_image, write_video  # noqa: E402
from torchvision.transforms import Compose, Lambda, Normalize  # noqa: E402
from tqdm import tqdm  # noqa: E402

logger = logging.getLogger(__name__)


def is_image_file(filename: str) -> bool:
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    return os.path.splitext(filename.lower())[1] in image_exts


def _is_cuda_oom(err: BaseException) -> bool:
    msg = str(err).lower()
    return isinstance(err, RuntimeError) and (
        "cuda out of memory" in msg
        or "out of memory" in msg
        or ("cublas" in msg and "alloc" in msg)
        or ("memory" in msg and "allocation" in msg)
    )


def _cleanup_cuda():
    gc.collect()
    try:
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    except Exception:
        pass


def _is_cuda_oom_any(err: BaseException) -> bool:
    """Return True if any exception in the chain looks like a CUDA OOM."""
    cur: Optional[BaseException] = err
    seen = set()
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if _is_cuda_oom(cur):
            return True
        cur = getattr(cur, "__cause__", None) or getattr(cur, "__context__", None)
    return False


def _blend_overlap(prev_tail: torch.Tensor, curr_head: torch.Tensor) -> torch.Tensor:
    """
    Blend overlap frames linearly.
    prev_tail, curr_head: (T, C, H, W), float32 in [0,1]
    """
    t = int(prev_tail.shape[0])
    if t <= 0:
        return curr_head
    w = torch.linspace(0.0, 1.0, steps=t, device=prev_tail.device).view(t, 1, 1, 1)
    return prev_tail * (1.0 - w) + curr_head * w


def _stitch_segments(segments: List[torch.Tensor], overlap_frames: int, *, blend: bool) -> torch.Tensor:
    """
    segments: list of (T, C, H, W) float32 in [0,1]
    Returns concatenated (T_total, C, H, W)

    If blend=False, we drop the overlap frames from the later segment (simple stitching).
    """
    if not segments:
        return torch.empty(0)
    if overlap_frames <= 0:
        return torch.cat(segments, dim=0)
    out = segments[0]
    for seg in segments[1:]:
        ov = min(overlap_frames, out.shape[0], seg.shape[0])
        if ov <= 0:
            out = torch.cat([out, seg], dim=0)
            continue
        if blend:
            blended = _blend_overlap(out[-ov:], seg[:ov])
            out = torch.cat([out[:-ov], blended, seg[ov:]], dim=0)
        else:
            out = torch.cat([out, seg[ov:]], dim=0)
    return out


def _iter_windows_by_streaming(
    path: str,
    *,
    window_frames: int,
    overlap_frames: int,
) -> Iterable[Tuple[int, int, torch.Tensor]]:
    """Yield windows by sequential decode so overlaps share the exact same frames.

    Yields: (window_index, start_frame_index, video_TCHW_uint8)
    """
    if window_frames <= 0:
        raise ValueError("window_frames must be > 0")
    overlap_frames = max(0, int(overlap_frames))
    stride = max(1, int(window_frames - overlap_frames))

    vr = VideoReader(path, "video")
    buf: List[torch.Tensor] = []

    def _stack(frames_hwc: List[torch.Tensor]) -> torch.Tensor:
        # VideoReader returns HWC uint8; convert to TCHW uint8.
        if not frames_hwc:
            return torch.empty((0, 3, 1, 1), dtype=torch.uint8)
        f0 = frames_hwc[0]
        if f0.ndim == 3 and f0.shape[-1] in (1, 3, 4):  # HWC
            frames_chw = [f[..., :3].permute(2, 0, 1).contiguous() for f in frames_hwc]
        elif f0.ndim == 3 and f0.shape[0] in (1, 3, 4):  # CHW
            frames_chw = [f[:3].contiguous() for f in frames_hwc]
        else:
            raise ValueError(f"Unexpected frame tensor shape from VideoReader: {tuple(f0.shape)}")
        return torch.stack(frames_chw, dim=0)

    for item in vr:
        buf.append(item["data"])
        if len(buf) >= window_frames:
            break

    if not buf:
        yield 0, 0, torch.empty((0, 3, 1, 1), dtype=torch.uint8)
        return

    win_i = 0
    start_idx = 0
    yield win_i, start_idx, _stack(buf)
    win_i += 1

    overlap_keep = overlap_frames
    while True:
        prefix = buf[-overlap_keep:] if overlap_keep > 0 else []
        new_frames: List[torch.Tensor] = []
        for item in vr:
            new_frames.append(item["data"])
            if len(new_frames) >= stride:
                break
        if not new_frames:
            break
        buf = prefix + new_frames
        start_idx = start_idx + stride
        yield win_i, start_idx, _stack(buf)
        win_i += 1


def _write_output(sample_tchw_01: torch.Tensor, filename: str, fps: float):
    sample = rearrange(sample_tchw_01, "t c h w -> t h w c")
    sample = (sample * 255.0).round().clamp(0, 255).to(torch.uint8)
    if torch.is_tensor(sample):
        sample = sample.detach().cpu().numpy()
    sample = np.asarray(sample, dtype=np.uint8)
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    if sample.shape[0] == 1:
        mediapy.write_image(filename, sample.squeeze(0))
    else:
        try:
            mediapy.write_video(filename, sample, fps=float(fps))
        except Exception:
            write_video(filename, torch.from_numpy(sample), fps=float(fps))


def _safe_stem(path: str) -> str:
    base = os.path.basename(path)
    stem = os.path.splitext(base)[0]
    keep = []
    for ch in stem:
        if ch.isalnum() or ch in ("-", "_", ".", " "):
            keep.append(ch)
        else:
            keep.append("_")
    stem = "".join(keep).strip().replace(os.sep, "_")
    return stem or "video"


def _write_video_streaming_from_segments_u8(
    segment_paths: List[str],
    out_path: str,
    *,
    fps: float,
    overlap_frames: int,
    blend: bool,
) -> None:
    if not segment_paths:
        raise RuntimeError("No segments to stitch.")

    overlap_frames = max(0, int(overlap_frames))

    def _load_u8(path: str) -> torch.Tensor:
        t = torch.load(path, map_location="cpu")
        if not torch.is_tensor(t):
            raise TypeError(f"Expected a torch Tensor in {path}, got {type(t)}")
        if t.dtype != torch.uint8:
            raise TypeError(f"Expected uint8 tensor in {path}, got {t.dtype}")
        if t.ndim != 4:
            raise ValueError(f"Expected (T,C,H,W) in {path}, got shape {tuple(t.shape)}")
        return t.contiguous()

    def _write_u8_tchw(container, stream, tchw_u8: torch.Tensor) -> None:
        if tchw_u8.numel() == 0:
            return
        thwc = tchw_u8.permute(0, 2, 3, 1).contiguous().cpu().numpy()
        thwc = np.asarray(thwc, dtype=np.uint8)
        for frame in thwc:
            vf = av.VideoFrame.from_ndarray(frame, format="rgb24")
            for pkt in stream.encode(vf):
                container.mux(pkt)

    def _blend_u8(prev_tail_u8: torch.Tensor, curr_head_u8: torch.Tensor) -> torch.Tensor:
        t = min(int(prev_tail_u8.shape[0]), int(curr_head_u8.shape[0]))
        if t <= 0:
            return torch.empty((0,) + tuple(prev_tail_u8.shape[1:]), dtype=torch.uint8)
        a = prev_tail_u8[-t:].float().div(255.0)
        b = curr_head_u8[:t].float().div(255.0)
        blended = _blend_overlap(a, b)
        return (blended * 255.0).round().clamp(0, 255).to(torch.uint8)

    first = _load_u8(segment_paths[0])
    h = int(first.shape[2])
    w = int(first.shape[3])

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    container = av.open(out_path, mode="w")
    try:

        def _mk_stream(codec_name: str):
            s = container.add_stream(codec_name, rate=Fraction(fps).limit_denominator(1000))
            if codec_name == "h264_nvenc":
                s.options = {"preset": "p4", "tune": "hq"}
            elif codec_name == "mpeg4":
                s.options = {"qscale": "5"}
            return s

        def _nvenc_error(e: Exception) -> bool:
            msg = str(e).lower()
            return any(
                k in msg
                for k in (
                    "h264_nvenc",
                    "libnvidia-encode.so",
                    "nvenc",
                    "avcodec_open2",
                    "operation not permitted",
                    "minimum required nvidia driver",
                )
            )

        def _probe_nvenc_available() -> bool:
            """
            Best-effort probe for NVENC availability.

            IMPORTANT:
            - Do NOT probe by encoding on the real output stream. Feeding a dummy frame and/or flushing
              (encode(None)) can finalize the encoder and/or consume PTS, leading to EOFError or corrupt output.
            - Instead, probe with a temporary throwaway container.
            """
            probe_path = f"{out_path}.nvenc_probe.mp4"
            try:
                # Remove any previous probe file.
                try:
                    os.remove(probe_path)
                except Exception:
                    pass

                c = av.open(probe_path, mode="w")
                try:
                    s = c.add_stream("h264_nvenc", rate=Fraction(fps).limit_denominator(1000))
                    s.width = w
                    s.height = h
                    s.pix_fmt = "yuv420p"
                    s.options = {"preset": "p4", "tune": "hq"}

                    test = np.zeros((h, w, 3), dtype=np.uint8)
                    vf = av.VideoFrame.from_ndarray(test, format="rgb24")
                    for pkt in s.encode(vf):
                        c.mux(pkt)
                    for pkt in s.encode(None):
                        c.mux(pkt)
                finally:
                    try:
                        c.close()
                    except Exception:
                        pass

                return True
            except Exception as e:
                if _nvenc_error(e):
                    return False
                # Treat any other probe error as "no nvenc" as well; we'll fall back to mpeg4.
                return False
            finally:
                # Probe file is not needed; ignore any cleanup errors.
                try:
                    os.remove(probe_path)
                except Exception:
                    pass

        try:
            if not _probe_nvenc_available():
                raise RuntimeError("NVENC probe failed")
            stream = _mk_stream("h264_nvenc")
        except Exception as e:
            if _nvenc_error(e):
                logger.warning("h264_nvenc not available or failed at encode; falling back to mpeg4. %s", e)
                stream = _mk_stream("mpeg4")
            else:
                raise
        stream.width = w
        stream.height = h
        stream.pix_fmt = "yuv420p"

        if overlap_frames <= 0:
            for p in segment_paths:
                seg = _load_u8(p)
                _write_u8_tchw(container, stream, seg)
        else:
            if not blend:
                seg0 = first
                _write_u8_tchw(container, stream, seg0)
                for p in segment_paths[1:]:
                    seg = _load_u8(p)
                    ov = min(overlap_frames, int(seg.shape[0]))
                    _write_u8_tchw(container, stream, seg[ov:])
            else:
                seg0 = first
                if int(seg0.shape[0]) > overlap_frames:
                    _write_u8_tchw(container, stream, seg0[:-overlap_frames])
                    prev_tail = seg0[-overlap_frames:]
                else:
                    prev_tail = seg0

                for p in segment_paths[1:]:
                    seg = _load_u8(p)
                    ov = min(overlap_frames, int(seg.shape[0]), int(prev_tail.shape[0]))
                    head = seg[:ov]

                    if int(prev_tail.shape[0]) > ov:
                        _write_u8_tchw(container, stream, prev_tail[:-ov])
                    blended_u8 = _blend_u8(prev_tail, head)
                    _write_u8_tchw(container, stream, blended_u8)

                    rest = seg[ov:]
                    if int(rest.shape[0]) > overlap_frames:
                        _write_u8_tchw(container, stream, rest[:-overlap_frames])
                        prev_tail = rest[-overlap_frames:]
                    else:
                        prev_tail = rest

                _write_u8_tchw(container, stream, prev_tail)

        for pkt in stream.encode(None):
            container.mux(pkt)
    finally:
        container.close()


def _resolve_variant_module(variant: str):
    mapping = {
        "seedvr2_3b": ("projects.inference_seedvr2_3b", "inference_seedvr2_3b.py"),
        "seedvr2_7b": ("projects.inference_seedvr2_7b", "inference_seedvr2_7b.py"),
        "seedvr2_7b_sharp": ("projects.inference_seedvr2_7b_sharp", "inference_seedvr2_7b_sharp.py"),
    }
    if variant not in mapping:
        raise ValueError(f"Unknown --variant: {variant}. Choose one of: {', '.join(mapping.keys())}")

    module_name, rel_py = mapping[variant]
    this_dir = os.path.dirname(os.path.abspath(__file__))
    py_path = os.path.join(this_dir, rel_py)
    seedvr_root = str(os.getenv("SEEDVR_ROOT", "")).strip()

    # Some upstream SeedVR scripts check for optional resources via relative paths at import-time
    # (e.g. "./projects/video_diffusion_sr/color_fix.py"). Import them from the SeedVR repo root
    # when available so those checks behave as intended.
    old_cwd = os.getcwd()
    try:
        if seedvr_root and os.path.isdir(seedvr_root):
            os.chdir(seedvr_root)

        if os.path.isfile(py_path):
            unique_name = f"_seedvr_variant_{variant}"
            spec = importlib.util.spec_from_file_location(unique_name, py_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Failed to load variant module spec for {variant} from {py_path}")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod

        return __import__(module_name, fromlist=["*"])
    finally:
        try:
            os.chdir(old_cwd)
        except Exception:
            pass


def configure_runner(sp_size: int, *, variant: str = "seedvr2_3b"):
    module = _resolve_variant_module(str(variant))
    # Upstream SeedVR variant scripts assume they are executed from the SeedVR repo root
    # (they load configs like "./configs_7b/main.yaml"). When we run from a different cwd
    # (e.g. /workspace), those relative paths break. Fix by temporarily switching cwd.
    seedvr_root = str(os.getenv("SEEDVR_ROOT", "")).strip()
    old_cwd = os.getcwd()
    try:
        if seedvr_root and os.path.isdir(seedvr_root):
            os.chdir(seedvr_root)
        runner = module.configure_runner(int(sp_size))
    finally:
        try:
            os.chdir(old_cwd)
        except Exception:
            pass
    setattr(runner, "_seedvr_window_variant", module)
    return runner


def generation_step(runner, text_embeds_dict, cond_latents):
    module = getattr(runner, "_seedvr_window_variant", None)
    if module is None:
        raise RuntimeError("Runner is missing '_seedvr_window_variant'.")
    return module.generation_step(runner, text_embeds_dict, cond_latents=cond_latents)


def cut_videos(videos, sp_size):
    """Pad video length to be compatible with sequence parallelism."""
    t = videos.size(1)
    if t == 1:
        return videos
    if t <= 4 * sp_size:
        padding = [videos[:, -1].unsqueeze(1)] * (4 * sp_size - t + 1)
        padding = torch.cat(padding, dim=1)
        videos = torch.cat([videos, padding], dim=1)
        return videos
    if (t - 1) % (4 * sp_size) == 0:
        return videos
    padding = [videos[:, -1].unsqueeze(1)] * (4 * sp_size - ((t - 1) % (4 * sp_size)))
    padding = torch.cat(padding, dim=1)
    videos = torch.cat([videos, padding], dim=1)
    return videos


def generation_loop(
    runner,
    video_path: str = "./test_videos",
    output_dir: str = "./results",
    output_path: Optional[str] = None,
    tmp_dir: Optional[str] = None,
    batch_size: int = 1,
    cfg_scale: float = 1.0,
    cfg_rescale: float = 0.0,
    sample_steps: int = 1,
    seed: int = 666,
    res_h: int = 720,
    res_w: int = 1280,
    sp_size: int = 1,
    out_fps: Optional[float] = None,
    window_frames: int = 128,
    overlap_frames: int = 64,
    no_blend_overlap: bool = False,
    variant: str = "seedvr2_3b",
):
    module = getattr(runner, "_seedvr_window_variant", None)
    if module is None:
        raise RuntimeError("Runner is missing '_seedvr_window_variant'.")

    os.makedirs(output_dir, exist_ok=True)
    failure_log_path = os.path.join(output_dir, "failures.log")
    tgt_path = output_dir

    runner.config.diffusion.cfg.scale = cfg_scale
    runner.config.diffusion.cfg.rescale = cfg_rescale
    runner.config.diffusion.timesteps.sampling.steps = sample_steps
    runner.configure_diffusion()

    set_seed(seed, same_across_ranks=True)

    if os.path.isdir(video_path):
        if output_path:
            raise ValueError("--output_path requires --video_path to be a single file (not a directory).")
        video_root = video_path
        video_list_for_prompts = os.listdir(video_root)
    else:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"--video_path not found: {video_path}")
        video_root = os.path.dirname(video_path) or "."
        video_list_for_prompts = [os.path.basename(video_path)]

    original_videos = []
    for f in video_list_for_prompts:
        if is_image_file(f) or f.lower().endswith(".mp4"):
            original_videos.append(f)
    print(f"Total prompts to be generated: {len(original_videos)}")

    original_videos_group = partition_by_groups(
        original_videos,
        get_data_parallel_world_size() // get_sequence_parallel_world_size(),
    )
    original_videos_local = original_videos_group[get_data_parallel_rank() // get_sequence_parallel_world_size()]
    original_videos_local = partition_by_size(original_videos_local, batch_size)

    def _extract_text_embeds():
        def _find_asset(name: str) -> Path:
            # Prefer cwd (matches upstream behavior), but fall back to SEEDVR_ROOT
            # so this runner can be executed from other working directories.
            candidates: list[Path] = [Path.cwd() / name]
            seedvr_root = str(os.getenv("SEEDVR_ROOT", "")).strip()
            if seedvr_root:
                candidates.append(Path(seedvr_root) / name)
            for p in candidates:
                if p.exists():
                    return p
            raise FileNotFoundError(
                f"Expected {name} in current working directory or $SEEDVR_ROOT. "
                f"cwd={Path.cwd()} SEEDVR_ROOT={seedvr_root!r}"
            )

        pos_path = _find_asset("pos_emb.pt")
        neg_path = _find_asset("neg_emb.pt")
        text_pos_embeds = torch.load(str(pos_path), map_location="cpu")
        text_neg_embeds = torch.load(str(neg_path), map_location="cpu")
        return {"texts_pos": [text_pos_embeds], "texts_neg": [text_neg_embeds]}

    positive_prompts_embeds = []
    for _ in tqdm(original_videos_local):
        positive_prompts_embeds.append(_extract_text_embeds())
    gc.collect()
    torch.cuda.empty_cache()

    video_transform = Compose(
        [
            NaResize(
                resolution=(res_h * res_w) ** 0.5,
                mode="area",
                downsample_only=False,
            ),
            Lambda(lambda x: torch.clamp(x, 0.0, 1.0)),
            DivisibleCrop((16, 16)),
            Normalize(0.5, 0.5),
            Rearrange("t c h w -> c t h w"),
        ]
    )

    def _infer_fps(src_path: str, out_fps: float | None) -> float:
        if out_fps is not None:
            return float(out_fps)
        try:
            vr = VideoReader(src_path, "video")
            md = vr.get_metadata()
            fps = None
            if isinstance(md, dict):
                v = md.get("video", None)
                if isinstance(v, dict):
                    fps = v.get("fps", None)
            if isinstance(fps, (list, tuple)) and fps:
                fps = fps[0]
            if fps is not None and float(fps) > 0:
                return float(fps)
        except Exception:
            pass
        try:
            with av.open(src_path) as c:
                if c.streams.video:
                    st = c.streams.video[0]
                    rate = st.average_rate or st.base_rate
                    if rate is not None:
                        return float(rate)
        except Exception:
            pass
        return 30.0

    for videos, text_embeds in tqdm(zip(original_videos_local, positive_prompts_embeds)):
        for i, emb in enumerate(text_embeds["texts_pos"]):
            text_embeds["texts_pos"][i] = emb.to(get_device())
        for i, emb in enumerate(text_embeds["texts_neg"]):
            text_embeds["texts_neg"][i] = emb.to(get_device())

        for video in videos:
            src_path = os.path.abspath(os.path.join(video_root, video))
            if output_path:
                out_file = os.path.abspath(str(output_path))
            else:
                out_file = os.path.join(tgt_path, os.path.basename(video))

            try:
                if is_image_file(video):
                    if sp_size > 1:
                        raise ValueError("Sp size should be set to 1 for image inputs!")
                    img = read_image(src_path).unsqueeze(0) / 255.0
                    cond = video_transform(img.to(get_device()))
                    cond_cut = cut_videos(cond, sp_size)

                    runner.dit.to("cpu")
                    runner.vae.to(get_device())
                    cond_latents = runner.vae_encode([cond_cut])
                    runner.vae.to("cpu")
                    runner.dit.to(get_device())

                    set_seed(seed, same_across_ranks=True)
                    samples = generation_step(runner, text_embeds, cond_latents=cond_latents)
                    runner.dit.to("cpu")

                    if get_sequence_parallel_rank() == 0:
                        sample = samples[0].to("cpu")
                        out_01 = sample.clip(-1, 1).mul_(0.5).add_(0.5).float()
                        _write_output(out_01, out_file, fps=float(out_fps or 24.0))
                    _cleanup_cuda()
                    continue

                save_fps = _infer_fps(src_path, out_fps)
                tmp_root = os.path.abspath(str(tmp_dir)) if tmp_dir else os.path.join(tgt_path, "_tmp_window_segments")
                tmp_dir = os.path.join(
                    tmp_root,
                    f"{_safe_stem(video)}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_pid{os.getpid()}",
                )
                segment_paths: List[str] = []
                if get_sequence_parallel_rank() == 0:
                    os.makedirs(tmp_dir, exist_ok=True)
                window_iter = _iter_windows_by_streaming(
                    src_path, window_frames=int(window_frames), overlap_frames=int(overlap_frames)
                )
                for win_i, start_frame, win_tchw_u8 in window_iter:
                    if win_tchw_u8.numel() == 0 or (hasattr(win_tchw_u8, "shape") and win_tchw_u8.shape[0] == 0):
                        raise ValueError(
                            f"decode_failed: got 0 frames for {src_path} window starting at frame {start_frame}"
                        )
                    win = win_tchw_u8 / 255.0
                    cond = video_transform(win.to(get_device()))
                    ori_len = int(cond.size(1))
                    cond_cut = cut_videos(cond, sp_size)

                    runner.dit.to("cpu")
                    runner.vae.to(get_device())
                    cond_latents = runner.vae_encode([cond_cut])
                    runner.vae.to("cpu")
                    runner.dit.to(get_device())

                    set_seed(seed, same_across_ranks=True)
                    samples = generation_step(runner, text_embeds, cond_latents=cond_latents)
                    runner.dit.to("cpu")

                    if get_sequence_parallel_rank() == 0:
                        sample = samples[0]
                        if ori_len < sample.shape[0]:
                            sample = sample[:ori_len]
                        if getattr(module, "use_colorfix", False):
                            inp_tchw = rearrange(cond, "c t h w -> t c h w")
                            sample = module.wavelet_reconstruction(
                                sample.to("cpu"), inp_tchw[: sample.size(0)].to("cpu")
                            )
                        else:
                            sample = sample.to("cpu")
                        out_01 = sample.clip(-1, 1).mul_(0.5).add_(0.5).float()
                        out_u8 = (out_01 * 255.0).round().clamp(0, 255).to(torch.uint8)
                        seg_path = os.path.join(tmp_dir, f"seg_{win_i:06d}_start{start_frame:09d}.pt")
                        torch.save(out_u8.contiguous().cpu(), seg_path)
                        segment_paths.append(seg_path)

                    _cleanup_cuda()

                if get_sequence_parallel_rank() == 0:
                    if not segment_paths:
                        raise RuntimeError(f"No successful windows for {src_path} (see failures.log)")
                    _write_video_streaming_from_segments_u8(
                        segment_paths,
                        out_file,
                        fps=float(save_fps),
                        overlap_frames=int(overlap_frames),
                        blend=not bool(no_blend_overlap),
                    )
                    shutil.rmtree(tmp_dir, ignore_errors=True)
                    # Best-effort cleanup of the temp root if it's now empty.
                    try:
                        if tmp_root and os.path.isdir(tmp_root) and not os.listdir(tmp_root):
                            os.rmdir(tmp_root)
                    except Exception:
                        pass

            except Exception as e:
                # In single-file mode we re-raise; still try to clean temp segments if they were created.
                try:
                    if "tmp_dir" in locals() and isinstance(tmp_dir, str):
                        shutil.rmtree(tmp_dir, ignore_errors=True)
                except Exception:
                    pass
                if not os.path.isdir(video_path):
                    raise
                if get_sequence_parallel_rank() == 0:
                    kind = "OOM" if _is_cuda_oom_any(e) else "ERROR"
                    os.makedirs(os.path.dirname(failure_log_path) or ".", exist_ok=True)
                    with open(failure_log_path, "a", encoding="utf-8") as f:
                        f.write(f"[{datetime.datetime.now().isoformat()}] {kind}: {src_path}\n")
                        f.write(f"{str(e)}\n")
                        f.write(traceback.format_exc())
                        f.write("\n")
                    try:
                        if "tmp_dir" in locals() and isinstance(tmp_dir, str):
                            shutil.rmtree(tmp_dir, ignore_errors=True)
                    except Exception:
                        pass
                _cleanup_cuda()
                continue

        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant", type=str, default="seedvr2_3b", choices=["seedvr2_3b", "seedvr2_7b", "seedvr2_7b_sharp"]
    )
    parser.add_argument("--video_path", type=str, default="./test_videos")
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument(
        "--output_path",
        type=str,
        default="",
        help="Optional: write output to this exact file path (single-file mode only).",
    )
    parser.add_argument(
        "--tmp_dir",
        type=str,
        default="",
        help="Optional: directory root for window segment temp files (defaults to <output_dir>/_tmp_window_segments).",
    )
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--res_h", type=int, default=720)
    parser.add_argument("--res_w", type=int, default=1280)
    parser.add_argument("--sp_size", type=int, default=1)
    parser.add_argument("--out_fps", type=float, default=None)
    parser.add_argument("--window_frames", type=int, default=128, help="Frames per window.")
    parser.add_argument("--overlap_frames", type=int, default=64, help="Overlap frames between windows.")
    parser.add_argument(
        "--no_blend_overlap", action="store_true", help="Disable overlap blending (overlap frames will be dropped)."
    )
    args = parser.parse_args()

    runner = configure_runner(args.sp_size, variant=args.variant)
    args.output_path = args.output_path.strip()
    args.tmp_dir = args.tmp_dir.strip()
    if not args.output_path:
        args.output_path = None
    if not args.tmp_dir:
        args.tmp_dir = None
    generation_loop(runner, **vars(args))
