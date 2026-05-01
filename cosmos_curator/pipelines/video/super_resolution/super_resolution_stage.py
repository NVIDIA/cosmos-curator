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

"""SeedVR2 video super-resolution pipeline stage (Curator adapter).

Applies diffusion-based video super-resolution to each clip using a windowed
inference approach from the upstream SeedVR2 project. The windowed approach
processes videos in overlapping segments to bound GPU memory usage, then
stitches the segments back together with linear blending.

Core SR inference logic (windowing, stitching, ``generation_step``,
``cut_videos``) is vendored from upstream — see
``inference_seedvr2_window.py`` in this package.  This module is the thin
Curator adapter that bridges that logic with the pipeline framework.

Why a separate adapter instead of calling the vendored code directly?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
In the Curator pipeline, clips flow between stages as **in-memory byte
arrays** (``SplitPipeTask`` -> ``Video`` -> ``Clip`` -> ``clip.encoded_data``).
Upstream chunking/splitting stages hand off clips as bytes; downstream
stages (captioning, filtering, …) expect to receive bytes.  The vendored
code, by contrast, reads/writes video **files on disk** via
``torchvision.io.VideoReader`` and ``mediapy.write_video``.

Rather than round-tripping through temp files (write bytes -> read file ->
process -> write file -> read bytes), this adapter stays in-memory:

- ``_decode_video_to_frames`` decodes clip bytes via PyAV into a frame tensor.
- ``_iter_windows_from_frames`` slices the already-decoded tensor into
  overlapping windows (the vendored ``_iter_windows_by_streaming`` streams
  from a file path, which doesn't apply here).
- ``_encode_frames_to_mp4`` encodes the upscaled tensor back to bytes.

The per-window inference loop itself (VAE encode, CPU offloading, DiT
forward pass, colorfix, denormalize) mirrors the vendored
``generation_loop`` but calls ``generation_step`` and ``cut_videos``
directly from the vendored module.

Everything else is Curator-specific plumbing:

- ``SuperResolutionStage`` implements ``CuratorStage`` so it can be
  scheduled by Ray via ``run_pipeline()``.
- ``process_data`` iterates the pipeline's data model
  (``SplitPipeTask.videos[].clips[]``), with per-clip error handling so
  one failure doesn't kill the batch.
- ``_load_text_embeds`` finds ``pos_emb.pt`` / ``neg_emb.pt`` in the
  HuggingFace model weights directory (managed by ``model_utils``),
  whereas the vendored code looks in cwd / ``$SEEDVR_ROOT``.
- ``_configure_diffusion`` applies ``SuperResolutionConfig`` overrides.
"""

import io
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from loguru import logger

from cosmos_curator.core.interfaces.model_interface import ModelInterface
from cosmos_curator.core.interfaces.stage_interface import CuratorStage, CuratorStageResource
from cosmos_curator.core.utils.data.bytes_transport import bytes_to_numpy
from cosmos_curator.core.utils.model import model_utils
from cosmos_curator.models.seedvr2 import SeedVR2
from cosmos_curator.pipelines.video.super_resolution.super_resolution_builders import SuperResolutionConfig
from cosmos_curator.pipelines.video.utils.data_model import Clip, SplitPipeTask

if TYPE_CHECKING:
    import torch


def _iter_windows_from_frames(
    frames_tchw_u8: "torch.Tensor",
    *,
    window_frames: int,
    overlap_frames: int,
) -> list[tuple[int, int, "torch.Tensor"]]:
    """Split a frame tensor into overlapping windows.

    Args:
        frames_tchw_u8: Video frames as (T, C, H, W) uint8 tensor.
        window_frames: Number of frames per window.
        overlap_frames: Number of overlapping frames between windows.

    Returns:
        List of (window_index, start_frame_index, window_tchw_u8) tuples.

    """
    total_frames = frames_tchw_u8.shape[0]
    if total_frames == 0:
        return []

    overlap_frames = max(0, min(overlap_frames, window_frames - 1))
    stride = max(1, window_frames - overlap_frames)

    windows: list[tuple[int, int, torch.Tensor]] = []
    win_i = 0
    start = 0
    while start < total_frames:
        end = min(start + window_frames, total_frames)
        windows.append((win_i, start, frames_tchw_u8[start:end]))
        win_i += 1
        start += stride
        if end >= total_frames:
            break

    return windows


def _decode_video_to_frames(video_bytes: npt.NDArray[np.uint8]) -> tuple["torch.Tensor", float]:
    """Decode MP4 bytes into a frame tensor and extract FPS.

    Args:
        video_bytes: Raw MP4 video bytes as numpy array.

    Returns:
        Tuple of (frames_tchw_u8, fps) where frames are (T, C, H, W) uint8 torch tensor.

    """
    import av  # noqa: PLC0415
    import torch as _torch  # noqa: PLC0415

    with av.open(io.BytesIO(bytes(video_bytes))) as container:
        video_stream = container.streams.video[0]
        rate = video_stream.average_rate or video_stream.base_rate
        fps = float(rate) if rate else 30.0

        frames: list[_torch.Tensor] = []
        for frame in container.decode(video=0):  # type: ignore[union-attr]
            arr = frame.to_ndarray(format="rgb24")
            t = _torch.from_numpy(arr).permute(2, 0, 1)  # HWC -> CHW
            frames.append(t)

    if not frames:
        return _torch.empty((0, 3, 1, 1), dtype=_torch.uint8), fps
    return _torch.stack(frames, dim=0), fps


def _encode_frames_to_mp4(
    frames_tchw_u8: "torch.Tensor",
    fps: float,
) -> npt.NDArray[np.uint8]:
    """Encode a frame tensor to MP4 bytes.

    Args:
        frames_tchw_u8: Frames as (T, C, H, W) uint8 tensor.
        fps: Output frame rate.

    Returns:
        MP4 video bytes as numpy array.

    """
    from fractions import Fraction  # noqa: PLC0415

    import av  # noqa: PLC0415

    buf = io.BytesIO()
    with av.open(buf, mode="w", format="mp4") as container:
        stream = container.add_stream("mpeg4", rate=Fraction(fps).limit_denominator(1000))
        stream.width = int(frames_tchw_u8.shape[3])
        stream.height = int(frames_tchw_u8.shape[2])
        stream.pix_fmt = "yuv420p"
        stream.options = {"qscale": "5"}

        thwc = frames_tchw_u8.permute(0, 2, 3, 1).contiguous().cpu().numpy()
        for frame_arr in thwc:
            vf = av.VideoFrame.from_ndarray(np.asarray(frame_arr, dtype=np.uint8), format="rgb24")
            for pkt in stream.encode(vf):
                container.mux(pkt)

        for pkt in stream.encode(None):
            container.mux(pkt)

    return bytes_to_numpy(buf.getvalue())


class SuperResolutionStage(CuratorStage):
    """Pipeline stage that applies SeedVR2 video super-resolution to clips.

    For each clip in each task, the stage:
    1. Decodes the clip's encoded video data into frames
    2. Splits frames into overlapping windows to bound GPU memory
    3. Runs SeedVR2 diffusion SR on each window (with VAE/DiT CPU offloading)
    4. Stitches windows back together with linear blending
    5. Re-encodes the upscaled frames and replaces the clip's encoded_data
    """

    def __init__(self, config: SuperResolutionConfig) -> None:
        """Initialize the super-resolution stage.

        Args:
            config: Super-resolution configuration.

        """
        self._config = config
        self._model = SeedVR2(variant=config.variant, sp_size=config.sp_size)
        self._text_embeds: dict[str, list[Any]] | None = None

    @property
    def resources(self) -> CuratorStageResource:
        """Return the resource requirements for this stage."""
        return CuratorStageResource(cpus=1.0, gpus=1.0)

    @property
    def model(self) -> ModelInterface | None:
        """Return the SeedVR2 model interface."""
        return self._model

    def stage_setup(self) -> None:
        """Set up the model and pre-load text embeddings."""
        self._model.setup()
        self._text_embeds = self._load_text_embeds()
        self._configure_diffusion()

    def _configure_diffusion(self) -> None:
        """Apply diffusion config overrides from the stage config."""
        runner = self._model.runner
        runner.config.diffusion.cfg.scale = self._config.cfg_scale
        runner.config.diffusion.cfg.rescale = self._config.cfg_rescale
        runner.config.diffusion.timesteps.sampling.steps = self._config.sample_steps
        runner.configure_diffusion()

    def _load_text_embeds(self) -> dict[str, list[Any]]:
        """Load pre-computed positive/negative text embeddings.

        SeedVR2 uses fixed text embeddings (pos_emb.pt / neg_emb.pt) rather than
        per-prompt encoding.

        Returns:
            Dict with 'texts_pos' and 'texts_neg' lists of tensors.

        """
        import torch as _torch  # noqa: PLC0415
        from common.distributed import get_device  # type: ignore[import-not-found]  # noqa: PLC0415

        model_dir = model_utils.get_local_dir_for_weights_name(self._model.model_id_names[0])

        def _find_asset(name: str) -> Path:
            p = Path(model_dir) / name
            if p.exists():
                return p
            msg = f"Expected {name} in model weights directory: {model_dir}"
            raise FileNotFoundError(msg)

        pos_embeds = _torch.load(str(_find_asset("pos_emb.pt")), map_location="cpu", weights_only=True)
        neg_embeds = _torch.load(str(_find_asset("neg_emb.pt")), map_location="cpu", weights_only=True)

        device = get_device()
        return {
            "texts_pos": [pos_embeds.to(device)],
            "texts_neg": [neg_embeds.to(device)],
        }

    def process_data(self, tasks: list[SplitPipeTask]) -> list[SplitPipeTask] | None:  # type: ignore[override]
        """Apply super-resolution to all clips in each task.

        Args:
            tasks: List of pipeline tasks containing video clips to upscale.

        Returns:
            The tasks with clip encoded_data replaced by upscaled video bytes.

        """
        for task in tasks:
            for video in task.videos:
                for clip in video.clips:
                    try:
                        self._upscale_clip(clip)
                    except Exception:  # noqa: BLE001
                        logger.exception(f"SR failed for clip {clip.uuid} in {video.input_video}")
                        clip.errors["super_resolution"] = "SR inference failed"
        return tasks

    def _upscale_clip(self, clip: Clip) -> None:  # noqa: PLR0915
        """Run windowed SR inference on a single clip.

        Args:
            clip: The clip whose encoded_data will be upscaled in place.

        """
        import torch as _torch  # noqa: PLC0415
        from common.distributed import get_device  # type: ignore[import-not-found]  # noqa: PLC0415
        from common.seed import set_seed  # type: ignore[import-not-found]  # noqa: PLC0415
        from data.image.transforms.divisible_crop import (  # type: ignore[import-not-found]  # noqa: PLC0415
            DivisibleCrop,
        )
        from data.image.transforms.na_resize import NaResize  # type: ignore[import-not-found]  # noqa: PLC0415
        from data.video.transforms.rearrange import Rearrange  # type: ignore[import-not-found]  # noqa: PLC0415
        from einops import rearrange  # type: ignore[import-not-found, import-untyped]  # noqa: PLC0415
        from torchvision.transforms import (  # type: ignore[import-not-found, import-untyped]  # noqa: PLC0415
            Compose,
            Lambda,
            Normalize,
        )

        from cosmos_curator.pipelines.video.super_resolution.inference_seedvr2_window import (  # noqa: PLC0415
            _cleanup_cuda,
            _stitch_segments,
            cut_videos,
            generation_step,
        )

        cfg = self._config

        clip_bytes = clip.encoded_data.resolve()
        if clip_bytes is None:
            return

        frames_tchw_u8, src_fps = _decode_video_to_frames(clip_bytes)
        if frames_tchw_u8.shape[0] == 0:
            logger.warning(f"Clip {clip.uuid}: decoded 0 frames, skipping SR")
            return

        out_fps = cfg.out_fps if cfg.out_fps is not None else src_fps

        video_transform = Compose(
            [
                NaResize(
                    resolution=(cfg.target_height * cfg.target_width) ** 0.5,
                    mode="area",
                    downsample_only=False,
                ),
                Lambda(lambda x: _torch.clamp(x, 0.0, 1.0)),
                DivisibleCrop((16, 16)),
                Normalize(0.5, 0.5),
                Rearrange("t c h w -> c t h w"),
            ]
        )

        runner = self._model.runner
        module = self._model.variant_module
        device = get_device()
        sp_size = cfg.sp_size

        windows = _iter_windows_from_frames(
            frames_tchw_u8,
            window_frames=cfg.window_frames,
            overlap_frames=cfg.overlap_frames,
        )

        output_segments: list[_torch.Tensor] = []

        for _win_i, _start_frame, win_tchw_u8 in windows:
            win = win_tchw_u8.float() / 255.0
            cond = video_transform(win.to(device))
            ori_len = int(cond.size(1))
            cond_cut = cut_videos(cond, sp_size)  # type: ignore[no-untyped-call]

            runner.dit.to("cpu")
            runner.vae.to(device)
            cond_latents = runner.vae_encode([cond_cut])
            runner.vae.to("cpu")
            runner.dit.to(device)

            set_seed(cfg.seed, same_across_ranks=True)
            samples = generation_step(runner, self._text_embeds, cond_latents=cond_latents)  # type: ignore[no-untyped-call]
            runner.dit.to("cpu")

            sample = samples[0]
            if ori_len < sample.shape[0]:
                sample = sample[:ori_len]

            if getattr(module, "use_colorfix", False):
                inp_tchw = rearrange(cond, "c t h w -> t c h w")
                sample = module.wavelet_reconstruction(sample.to("cpu"), inp_tchw[: sample.size(0)].to("cpu"))
            else:
                sample = sample.to("cpu")

            out_01 = sample.clip(-1, 1).mul_(0.5).add_(0.5).float()
            output_segments.append(out_01)

            _cleanup_cuda()  # type: ignore[no-untyped-call]

        if not output_segments:
            logger.warning(f"Clip {clip.uuid}: no SR segments produced")
            return

        stitched_01 = _stitch_segments(output_segments, cfg.overlap_frames, blend=cfg.blend_overlap)
        stitched_u8 = (stitched_01 * 255.0).round().clamp(0, 255).to(_torch.uint8)

        upscaled_bytes = _encode_frames_to_mp4(stitched_u8, out_fps)
        clip.encoded_data = upscaled_bytes  # type: ignore[assignment]

        if self._config.verbose:
            logger.info(
                f"Clip {clip.uuid}: SR {frames_tchw_u8.shape[0]} frames "
                f"-> {stitched_u8.shape[0]} frames at {stitched_u8.shape[2]}x{stitched_u8.shape[3]}"
            )
