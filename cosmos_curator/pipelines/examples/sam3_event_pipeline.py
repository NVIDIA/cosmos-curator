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

r"""Example pipeline: SAM3 object tracking + per-event VLM captioning on pre-split clips.

Runs *only* the two new tracking stages on a directory of short (recommended
≤ 30 s) mp4 clips. Intended for quick iteration on the SAM3 config and the
per-event prompt without going through the full splitting pipeline.

Example usage:

.. code-block:: bash

    cosmos-curator local launch --curator-path . -- \\
        python -m cosmos_curator.pipelines.examples.sam3_event_pipeline \\
            --input-dir /config/clips_30s \\
            --output-dir /config/output/sam3_events \\
            --sam3-prompts "a car" "a pedestrian" "a traffic light" \\
            --event-caption-prompt "Describe each driving-relevant event in order."

Outputs written per input ``<name>.mp4``:

- ``<output-dir>/<name>/instances.json``
- ``<output-dir>/<name>/objects.json``
- ``<output-dir>/<name>/events.json`` (if ``--enable-event-captioning``)
- ``<output-dir>/<name>/tracked.mp4`` (annotated video; always on for this example)
"""

import argparse
import json
import pathlib
import uuid
from typing import Any

import cv2  # type: ignore[import-untyped]
from loguru import logger

from cosmos_curator.core.interfaces.pipeline_interface import run_pipeline
from cosmos_curator.core.interfaces.stage_interface import (
    CuratorStage,
    CuratorStageResource,
    CuratorStageSpec,
)
from cosmos_curator.core.utils.data.bytes_transport import bytes_to_numpy
from cosmos_curator.pipelines.video.captioning.per_event_caption_stage import PerEventCaptionStage
from cosmos_curator.pipelines.video.captioning.per_event_cli_args import (
    add_event_caption_args,
    resolve_event_caption_prompt,
)
from cosmos_curator.pipelines.video.tracking.cli_args import add_sam3_args
from cosmos_curator.pipelines.video.tracking.sam3_bbox_stage import SAM3QualityConfig
from cosmos_curator.pipelines.video.tracking.serialization import (
    sam3_events_envelope,
    sam3_instances_envelope,
    sam3_objects_envelope,
)
from cosmos_curator.pipelines.video.tracking.tracking_builders import (
    SAM3TrackingConfig,
    build_sam3_tracking_stages,
)
from cosmos_curator.pipelines.video.utils.data_model import Clip, SplitPipeTask, Video

_RECOMMENDED_MAX_CLIP_DURATION_S = 30.0


def _probe_mp4(path: pathlib.Path) -> tuple[float, float]:
    """Return ``(duration_seconds, fps)`` using OpenCV."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        msg = f"Cannot open {path}"
        raise RuntimeError(msg)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    cap.release()
    return (frames / fps if fps else 0.0), float(fps)


def _make_task(path: pathlib.Path) -> SplitPipeTask:
    """Build a ``SplitPipeTask`` with one ``Video`` / one ``Clip`` from an mp4 file."""
    duration, _fps = _probe_mp4(path)
    if duration > _RECOMMENDED_MAX_CLIP_DURATION_S:
        logger.warning(
            f"[sam3_event_pipeline] {path.name}: duration {duration:.1f}s exceeds "
            f"recommended {_RECOMMENDED_MAX_CLIP_DURATION_S}s — SAM3 may OOM without "
            f"--sam3-max-clip-duration-s override"
        )
    mp4_bytes = path.read_bytes()
    # Deterministic UUID derived from filename so outputs are reproducible.
    clip_uuid = uuid.uuid5(uuid.NAMESPACE_URL, f"sam3-event-example::{path.name}")
    clip = Clip(
        uuid=clip_uuid,
        source_video=str(path),
        span=(0.0, duration),
        encoded_data=bytes_to_numpy(mp4_bytes),  # type: ignore[arg-type]
    )
    video = Video(input_video=path, clips=[clip])
    return SplitPipeTask(session_id=str(path), videos=[video])


def _load_tasks(input_dir: pathlib.Path) -> list[SplitPipeTask]:
    """Build one ``SplitPipeTask`` per ``*.mp4`` file in ``input_dir``."""
    tasks: list[SplitPipeTask] = []
    for path in sorted(input_dir.glob("*.mp4")):
        try:
            tasks.append(_make_task(path))
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[sam3_event_pipeline] skipping {path}: {exc}")
    logger.info(f"[sam3_event_pipeline] loaded {len(tasks)} clips from {input_dir}")
    return tasks


class _WriteSam3OutputsStage(CuratorStage):
    """Output stage — writes ``instances/objects/events.json`` and ``tracked.mp4`` per clip."""

    def __init__(self, output_dir: pathlib.Path) -> None:
        self._output_dir = output_dir

    @property
    def resources(self) -> CuratorStageResource:
        """CPU-only output stage."""
        return CuratorStageResource(cpus=1.0)

    def stage_setup(self) -> None:
        """Ensure the output directory exists."""
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def _write_clip(self, clip: Clip) -> None:
        stem = pathlib.Path(clip.source_video).stem
        clip_dir = self._output_dir / stem
        clip_dir.mkdir(parents=True, exist_ok=True)

        def _write_json(name: str, payload: dict[str, Any]) -> None:
            (clip_dir / name).write_text(json.dumps(payload, indent=2))

        if clip.sam3_instances is not None:
            _write_json("instances.json", sam3_instances_envelope(clip.sam3_instances))
        if clip.sam3_objects_by_frame is not None:
            _write_json("objects.json", sam3_objects_envelope(clip.sam3_objects_by_frame))
        if clip.sam3_events is not None:
            _write_json("events.json", sam3_events_envelope(clip.sam3_events))

        annotated = clip.sam3_annotated_video.resolve()
        if annotated is not None:
            out_path = clip_dir / "tracked.mp4"
            out_path.write_bytes(bytes(annotated.tobytes()))

        logger.info(f"[sam3_event_pipeline] wrote outputs for {stem} → {clip_dir}")

    def process_data(self, tasks: list[SplitPipeTask]) -> list[SplitPipeTask] | None:  # type: ignore[override]
        """Write SAM3 outputs for every clip of every task."""
        for task in tasks:
            for video in task.videos:
                for clip in video.clips:
                    self._write_clip(clip)
        return tasks


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run SAM3 + per-event VLM captioning on pre-split mp4 clips.",
    )
    parser.add_argument("--input-dir", type=pathlib.Path, required=True, help="Directory of input mp4 clips.")
    parser.add_argument("--output-dir", type=pathlib.Path, required=True, help="Directory for outputs.")
    add_sam3_args(
        parser,
        include_enable_flag=False,
        include_write_annotated_flag=False,
        sam3_prompts_required=True,
    )
    add_event_caption_args(parser)
    parser.add_argument("--verbose", action="store_true", default=False)
    return parser


def _assemble_stages(args: argparse.Namespace) -> list[CuratorStage | CuratorStageSpec]:
    stages: list[CuratorStage | CuratorStageSpec] = []
    stages.extend(
        build_sam3_tracking_stages(
            SAM3TrackingConfig(
                prompts=list(args.sam3_prompts),
                target_fps=args.sam3_target_fps,
                max_clip_duration_s=args.sam3_max_clip_duration_s,
                session_reset_s=args.sam3_session_reset_s,
                quality=SAM3QualityConfig(
                    score_threshold_detection=args.sam3_score_threshold_detection,
                    det_nms_thresh=args.sam3_det_nms_thresh,
                    new_det_thresh=args.sam3_new_det_thresh,
                    fill_hole_area=args.sam3_fill_hole_area,
                    recondition_every_nth_frame=args.sam3_recondition_every_nth_frame,
                    recondition_on_trk_masks=args.sam3_recondition_on_trk_masks,
                    high_conf_thresh=args.sam3_high_conf_thresh,
                    high_iou_thresh=args.sam3_high_iou_thresh,
                ),
                # Annotation is always on for this example — visual output is the point.
                write_annotated_video=True,
                draw_trails=args.sam3_annotated_video_trails,
                verbose=args.verbose,
            )
        )
    )
    if args.enable_event_captioning:
        caption_stage = PerEventCaptionStage(
            backend=args.event_caption_backend,
            prompt_text=resolve_event_caption_prompt(args),
            qwen_variant=args.event_caption_qwen_variant,
            qwen_sampling_fps=args.event_caption_qwen_sampling_fps,
            qwen_fp8=args.event_caption_qwen_fp8,
            qwen_temperature=args.event_caption_qwen_temperature,
            qwen_top_p=args.event_caption_qwen_top_p,
            qwen_top_k=args.event_caption_qwen_top_k,
            gemini_model_name=args.event_caption_gemini_model_name,
            gemini_video_fps=args.event_caption_gemini_fps,
            gemini_media_resolution=args.event_caption_gemini_media_resolution,
            gemini_thinking_budget=args.event_caption_gemini_thinking_budget,
            gemini_max_output_tokens=args.event_caption_gemini_max_output_tokens,
            verbose=args.verbose,
        )
        # Pin Gemini to 1 worker x 1 slot so we don't exceed its per-minute
        # quota; Qwen is local and runs at default concurrency.
        if args.event_caption_backend == "gemini":
            stages.append(
                CuratorStageSpec(
                    caption_stage,
                    num_workers_per_node=1,
                    slots_per_actor=1,
                )
            )
        else:
            stages.append(caption_stage)
    stages.append(_WriteSam3OutputsStage(args.output_dir))
    return stages


def main() -> None:
    """Run the SAM3-event example pipeline."""
    args = _build_parser().parse_args()
    if not args.input_dir.is_dir():
        msg = f"--input-dir does not exist or is not a directory: {args.input_dir}"
        raise SystemExit(msg)

    tasks = _load_tasks(args.input_dir)
    if not tasks:
        logger.warning(f"[sam3_event_pipeline] no mp4 clips found under {args.input_dir}; nothing to do")
        return

    stages = _assemble_stages(args)
    run_pipeline(tasks, stages)
    logger.info("sam3_event_pipeline completed")


if __name__ == "__main__":
    main()
