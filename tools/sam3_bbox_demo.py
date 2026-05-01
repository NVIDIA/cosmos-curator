#!/usr/bin/env python3
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

r"""SAM3 bounding-box overlay demo.

Runs SAM3 on every Nth frame of a video and writes an annotated output video
with per-prompt bounding boxes and mask overlays.

Usage (inside the sam3 pixi env):
    pixi run -e sam3 python tools/sam3_bbox_demo.py \
        --video /config/videos/toothless.mp4 \
        --prompts "a dragon" "a viking" \
        --output /config/output/toothless_tracked.mp4 \
        --fps 10

Run via cosmos-curator local launch:
    micromamba run -n cosmos-curator cosmos-curator local launch --curator-path . -- \
        "pixi run -e sam3 python tools/sam3_bbox_demo.py \
         --video /config/videos/toothless.mp4 \
         --prompts 'a dragon' 'a viking' \
         --output /config/output/toothless_tracked.mp4 \
         --fps 10"
"""

import argparse
import collections
import json
import os
import pathlib

import cv2
import numpy as np
import torch
from loguru import logger

from cosmos_curator.models.sam3 import SAM3Model
from cosmos_curator.pipelines.video.tracking.visualization import Detection, draw_frame


def _log_memory(label: str = "") -> None:
    """Log current and peak GPU/CPU memory usage."""
    prefix = f"[memory {label}] " if label else "[memory] "
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        peak = torch.cuda.max_memory_allocated() / 1024**3
        logger.info(f"{prefix}GPU: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved, {peak:.2f} GB peak")
    try:
        import psutil  # noqa: PLC0415 — lazy import so the tool still runs if psutil is unavailable

        proc = psutil.Process(os.getpid())
        rss = proc.memory_info().rss / 1024**3
        logger.info(f"{prefix}CPU RSS: {rss:.2f} GB")
    except ImportError:
        pass


def _load_model(config_overrides: dict[str, object] | None = None) -> SAM3Model:
    """Load and return the SAM3 model with optional config overrides."""
    logger.info("Loading SAM3 model...")
    sam3 = SAM3Model()
    sam3.setup(config_overrides=config_overrides)
    logger.info("SAM3 ready.")
    return sam3


def _open_video(path: pathlib.Path) -> tuple[cv2.VideoCapture, float, int, int, int]:
    """Open a video file and return (cap, fps, width, height, total_frames)."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        msg = f"Cannot open {path}"
        raise FileNotFoundError(msg)
    fps: float = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return cap, fps, w, h, total


def _infer_frame(
    sam3: SAM3Model,
    session: object,
    bgr: np.ndarray,
    prompts: list[str],
) -> list[Detection]:
    """Run SAM3 on one BGR frame (streaming mode) and return Detections."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    inputs = sam3.processor(images=rgb, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to("cuda", dtype=torch.bfloat16)
    model_out = sam3.model(inference_session=session, frame=pixel_values)
    processed = sam3.processor.postprocess_outputs(session, model_out, original_sizes=inputs["original_sizes"])
    return _postprocess_to_detections(processed, prompts)


def _read_frames(
    path: pathlib.Path,
    step: int,
    max_src_frames: int,
) -> tuple[list[np.ndarray], list[np.ndarray], list[int], float, int, int]:
    """Read video frames, subsampling every *step*-th frame.

    Returns:
        (rgb_frames, bgr_frames, source_indices, src_fps, width, height)
        rgb_frames: list of H x W x 3 uint8 RGB arrays (for SAM3).
        bgr_frames: list of H x W x 3 uint8 BGR arrays (for drawing/writing).
        source_indices: source video frame index for each sampled frame.

    """
    cap, src_fps, w, h, total = _open_video(path)
    rgb_frames: list[np.ndarray] = []
    bgr_frames: list[np.ndarray] = []
    source_indices: list[int] = []
    frame_idx = 0
    limit = min(max_src_frames, total)
    while frame_idx < limit:
        ret, bgr = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            rgb_frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            bgr_frames.append(bgr)
            source_indices.append(frame_idx)
        frame_idx += 1
    cap.release()
    logger.info(
        f"Read {len(rgb_frames)} frames from {total} source frames (step={step}, src_fps={src_fps:.1f}, {w}x{h})"
    )
    return rgb_frames, bgr_frames, source_indices, src_fps, w, h


def _postprocess_to_detections(
    processed: dict,
    prompts: list[str],
) -> list[Detection]:
    """Convert postprocess_outputs dict to a list of Detection objects."""
    obj_ids: list[int] = processed["object_ids"].tolist()
    masks = processed["masks"]
    boxes = processed["boxes"]
    p2o: dict[str, list[int]] = processed["prompt_to_obj_ids"]

    detections: list[Detection] = []
    for prompt in prompts:
        for oid in p2o.get(prompt, []):
            if oid not in obj_ids:
                continue
            idx = obj_ids.index(oid)
            mask_np = masks[idx].cpu().numpy()
            if mask_np.any():
                detections.append(
                    Detection(
                        prompt=prompt,
                        object_id=oid,
                        box_xyxy=boxes[idx].tolist(),
                        mask=mask_np,
                    )
                )
    return detections


def _build_config_overrides(args: argparse.Namespace) -> dict[str, object] | None:
    """Collect non-None Sam3VideoConfig overrides from CLI args."""
    _CONFIG_ARGS = [
        "score_threshold_detection",
        "det_nms_thresh",
        "new_det_thresh",
        "fill_hole_area",
        "recondition_every_nth_frame",
        "recondition_on_trk_masks",
        "high_conf_thresh",
        "high_iou_thresh",
    ]
    overrides = {}
    for key in _CONFIG_ARGS:
        val = getattr(args, key, None)
        if val is not None:
            overrides[key] = val
    return overrides or None


def run(args: argparse.Namespace) -> None:  # noqa: PLR0915 — CLI entry point; splitting hides the linear read of the SAM3 loop
    """Run SAM3 on the input video and write an annotated output video."""
    sam3 = _load_model(_build_config_overrides(args))
    cap, src_fps, w, h, total_frames = _open_video(args.video)
    step = max(1, round(src_fps / args.fps))
    out_fps = src_fps / step

    logger.info(f"Source: {total_frames} frames @ {src_fps:.1f} fps  ({w}x{h})")
    logger.info(f"Output: {out_fps:.1f} fps  (every {step} source frames)")
    logger.info(f"Prompts: {args.prompts}")

    out_path = pathlib.Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (w, h))

    reset_every = int(args.session_reset_s * src_fps) if args.session_reset_s else 0

    # Per-object trajectory: object_id → list of (cx, cy) centre points
    trails: dict[int, list[tuple[int, int]]] = collections.defaultdict(list)
    # Raw detection log: list of per-frame records for JSON dump
    raw_log: list[dict] = []

    def _new_session() -> object:
        sess = sam3.processor.init_video_session(
            inference_device="cuda", video_storage_device="cpu", dtype=torch.bfloat16
        )
        for prompt in args.prompts:
            sess = sam3.processor.add_text_prompt(sess, prompt)
        return sess

    _log_memory("after model load")
    torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        session = _new_session()
        frame_idx = 0
        tracked = 0
        max_src_frames = int(args.duration_s * src_fps) if args.duration_s else total_frames
        while frame_idx < max_src_frames:
            ret, bgr = cap.read()
            if not ret:
                break
            if reset_every and frame_idx > 0 and frame_idx % reset_every == 0:
                session = _new_session()
                trails.clear()
                logger.info(f"  reset session at frame {frame_idx} ({frame_idx / src_fps:.1f}s)")
            if frame_idx % step == 0:
                detections = _infer_frame(sam3, session, bgr, args.prompts)
                for det in detections:
                    trails[det.object_id].append(det.center)
                time_s = round(frame_idx / src_fps, 3) if src_fps > 0 else 0.0
                raw_log.append(
                    {
                        "frame_idx": frame_idx,
                        "time_s": time_s,
                        "detections": [det.to_json_dict() for det in detections],
                    }
                )
                writer.write(
                    draw_frame(
                        bgr,
                        detections,
                        args.prompts,
                        trails,
                        draw_trails=args.trails,
                        current_time_s=time_s,
                    )
                )
                tracked += 1
                if tracked % 50 == 0:
                    det_summary = [(d.prompt, d.object_id) for d in detections]
                    logger.info(f"  frame {frame_idx}/{total_frames}  detected={det_summary}")
                    _log_memory(f"frame {frame_idx}")
            frame_idx += 1

    cap.release()
    writer.release()
    _log_memory("final")
    logger.info(f"Wrote {tracked} annotated frames to {out_path}")

    json_path = out_path.with_suffix(".json")
    json_path.write_text(json.dumps(raw_log, indent=2))
    logger.info(f"Wrote raw detections to {json_path}")


def _run_preloaded(args: argparse.Namespace) -> None:  # noqa: PLR0915 — CLI entry point; splitting hides the linear read of the SAM3 loop
    """Run SAM3 in pre-loaded mode with hotstart heuristics enabled."""
    sam3 = _load_model(_build_config_overrides(args))
    cap, src_fps, w, h, total_frames = _open_video(args.video)
    cap.release()

    step = max(1, round(src_fps / args.fps))
    out_fps = src_fps / step
    max_src_frames = int(args.duration_s * src_fps) if args.duration_s else total_frames

    logger.info(f"[preloaded] Source: {total_frames} frames @ {src_fps:.1f} fps  ({w}x{h})")
    logger.info(f"[preloaded] Output: {out_fps:.1f} fps  (every {step} source frames)")
    logger.info(f"[preloaded] Prompts: {args.prompts}")

    rgb_frames, bgr_frames, source_indices, _, _, _ = _read_frames(
        args.video,
        step,
        max_src_frames,
    )
    if not rgb_frames:
        logger.warning("No frames read — nothing to process.")
        return

    out_path = pathlib.Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (w, h))

    trails: dict[int, list[tuple[int, int]]] = collections.defaultdict(list)
    raw_log: list[dict] = []

    chunk_size = int(args.session_reset_s * args.fps) if args.session_reset_s else len(rgb_frames)
    chunk_size = max(1, chunk_size)
    n_chunks = (len(rgb_frames) + chunk_size - 1) // chunk_size
    logger.info(f"[preloaded] {len(rgb_frames)} sampled frames, chunk_size={chunk_size}, chunks={n_chunks}")

    _log_memory("after model load")
    torch.cuda.reset_peak_memory_stats()

    written = 0
    with torch.no_grad():
        session = None
        for chunk_idx in range(n_chunks):
            start = chunk_idx * chunk_size
            end = min(start + chunk_size, len(rgb_frames))
            chunk_rgb = rgb_frames[start:end]
            chunk_bgr = bgr_frames[start:end]
            chunk_src_indices = source_indices[start:end]

            if session is not None:
                del session
                torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            session = sam3.processor.init_video_session(
                video=chunk_rgb,
                inference_device="cuda",
                video_storage_device="cpu",
                dtype=torch.bfloat16,
            )
            for prompt in args.prompts:
                sam3.processor.add_text_prompt(session, prompt)

            logger.info(
                f"[preloaded] chunk {chunk_idx + 1}/{n_chunks}: "
                f"frames {chunk_src_indices[0]}-{chunk_src_indices[-1]} "
                f"({len(chunk_rgb)} sampled frames)"
            )

            for model_outputs in sam3.model.propagate_in_video_iterator(
                inference_session=session,
                show_progress_bar=False,
            ):
                processed = sam3.processor.postprocess_outputs(session, model_outputs)
                local_idx = model_outputs.frame_idx
                if local_idx >= len(chunk_bgr):
                    continue
                bgr = chunk_bgr[local_idx]
                src_idx = chunk_src_indices[local_idx]

                detections = _postprocess_to_detections(processed, args.prompts)
                for det in detections:
                    trails[det.object_id].append(det.center)
                time_s = round(src_idx / src_fps, 3) if src_fps > 0 else 0.0
                raw_log.append(
                    {
                        "frame_idx": src_idx,
                        "time_s": time_s,
                        "detections": [det.to_json_dict() for det in detections],
                    }
                )
                writer.write(
                    draw_frame(
                        bgr,
                        detections,
                        args.prompts,
                        trails,
                        draw_trails=args.trails,
                        current_time_s=time_s,
                    )
                )
                written += 1
                if written % 50 == 0:
                    det_summary = [(d.prompt, d.object_id) for d in detections]
                    logger.info(f"  frame {src_idx}/{total_frames}  detected={det_summary}")
                    _log_memory(f"frame {src_idx}")

            _log_memory(f"chunk {chunk_idx + 1}/{n_chunks} done")
            trails.clear()

    writer.release()
    _log_memory("final")
    logger.info(f"[preloaded] Wrote {written} annotated frames to {out_path}")

    json_path = out_path.with_suffix(".json")
    json_path.write_text(json.dumps(raw_log, indent=2))
    logger.info(f"[preloaded] Wrote raw detections to {json_path}")


def main() -> None:
    """Parse arguments and run the demo."""
    parser = argparse.ArgumentParser(
        description="SAM3 bounding-box overlay demo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--video", type=pathlib.Path, required=True, help="Input video file")
    parser.add_argument("--prompts", nargs="+", required=True, help="Text prompts to track")
    parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path("tracked.mp4"), help="Output video path")
    parser.add_argument("--fps", type=float, default=10.0, help="Output frame rate")
    parser.add_argument("--duration-s", type=float, default=None, help="Max seconds of source video to process")
    parser.add_argument(
        "--session-reset-s",
        type=float,
        default=10.0,
        help="Re-initialise the SAM3 session every N seconds (0 to disable). "
        "Prevents stale object state accumulating across clip boundaries.",
    )
    parser.add_argument(
        "--preloaded",
        action="store_true",
        default=False,
        help="Use pre-loaded video inference instead of streaming. "
        "Enables SAM3's hotstart heuristics (phantom removal, duplicate suppression, "
        "occlusion handling) for higher quality tracks. Uses more CPU RAM since all "
        "chunk frames are loaded upfront. --session-reset-s controls chunk length "
        "(0 = load entire video at once).",
    )
    parser.add_argument(
        "--trails",
        action="store_true",
        default=False,
        help="Draw trajectory trails showing each object's movement path.",
    )

    cfg = parser.add_argument_group("Sam3VideoConfig overrides (SAM3-native quality tuning)")
    cfg.add_argument(
        "--score-threshold-detection",
        type=float,
        default=None,
        help="Detection confidence threshold (default: 0.5, range: 0.0-1.0). "
        "RAISE (e.g. 0.6-0.7) to suppress low-confidence false positives — fewer phantom boxes. "
        "LOWER to detect more objects at the risk of more noise.",
    )
    cfg.add_argument(
        "--det-nms-thresh",
        type=float,
        default=None,
        help="IoU threshold for detection NMS (default: 0.1, range: 0.0-1.0). "
        "RAISE (e.g. 0.3-0.5) to keep more overlapping detections (crowded scenes). "
        "LOWER to aggressively merge nearby boxes and reduce duplicates.",
    )
    cfg.add_argument(
        "--new-det-thresh",
        type=float,
        default=None,
        help="Confidence threshold for adding a detection as a new tracked object (default: 0.7, range: 0.0-1.0). "
        "RAISE (e.g. 0.8-0.9) to be stricter about starting new tracks — fewer phantom objects. "
        "LOWER to pick up faint or partially-occluded objects sooner.",
    )
    cfg.add_argument(
        "--fill-hole-area",
        type=int,
        default=None,
        help="Min pixel area for filling mask holes and removing small sprinkles (default: 16, range: 0+). "
        "RAISE (e.g. 64-256) to clean up fragmented masks that cause bbox jitter. "
        "LOWER for finer mask detail at the cost of more noise. Set 0 to disable.",
    )
    cfg.add_argument(
        "--recondition-every-nth-frame",
        type=int,
        default=None,
        help="Re-anchor tracked masks against fresh detections every N frames (default: 16, range: 0+). "
        "LOWER (e.g. 4-8) to correct tracker drift more often — biggest lever for stationary "
        "object flicker. RAISE for less compute overhead. Set 0 to disable reconditioning entirely.",
    )
    cfg.add_argument(
        "--recondition-on-trk-masks",
        type=bool,
        default=None,
        help="Use tracked masks (True) or detection masks (False) for reconditioning (default: True). "
        "Set FALSE when tracked masks drift on stationary objects — uses the detector's fresh "
        "mask instead. Set TRUE when the tracker is more accurate than the detector.",
    )
    cfg.add_argument(
        "--high-conf-thresh",
        type=float,
        default=None,
        help="Min detection confidence required to recondition a tracklet (default: 0.8, range: 0.0-1.0). "
        "LOWER (e.g. 0.5-0.6) to allow reconditioning from weaker detections — helps "
        "stationary objects with lower detection scores. RAISE to only trust high-confidence corrections.",
    )
    cfg.add_argument(
        "--high-iou-thresh",
        type=float,
        default=None,
        help="Min IoU between detection and track required for reconditioning (default: 0.8, range: 0.0-1.0). "
        "LOWER (e.g. 0.4-0.5) to allow correction even when the tracked mask has drifted "
        "significantly from the detection. RAISE to only correct closely-matching tracks.",
    )
    args = parser.parse_args()
    if args.preloaded:
        _run_preloaded(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
