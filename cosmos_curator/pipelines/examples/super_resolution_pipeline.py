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

r"""Example pipeline: SeedVR2 video super-resolution.

A standalone pipeline for testing the super-resolution stage end-to-end.
Reads videos from an input directory, applies SeedVR2 SR, and writes
upscaled videos to an output directory.

Usage::

    cosmos-curator local launch --curator-path . -- pixi run --as-is -e seedvr python -m \
        cosmos_curator.pipelines.examples.super_resolution_pipeline \
        --input-dir /data/test_videos \
        --output-dir /data/sr_output \
        --sr-variant seedvr2_7b \
        --sr-target-height 720 \
        --sr-target-width 1280
"""

import argparse
from pathlib import Path
from uuid import uuid4

from loguru import logger

from cosmos_curator.core.interfaces.pipeline_interface import run_pipeline
from cosmos_curator.core.interfaces.stage_interface import (
    CuratorStage,
    CuratorStageResource,
    CuratorStageSpec,
    PipelineTask,
)
from cosmos_curator.core.utils.data.bytes_transport import bytes_to_numpy
from cosmos_curator.pipelines.video.super_resolution.super_resolution_builders import (
    SuperResolutionConfig,
    build_super_resolution_stages,
)
from cosmos_curator.pipelines.video.utils.data_model import Clip, SplitPipeTask, Video

_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"}


def _discover_videos(input_dir: str) -> list[Path]:
    """Find all video files in input directory.

    Args:
        input_dir: Path to directory containing video files.

    Returns:
        Sorted list of video file paths.

    """
    input_path = Path(input_dir)
    if not input_path.is_dir():
        msg = f"Input directory does not exist: {input_dir}"
        raise FileNotFoundError(msg)
    videos = [p for p in input_path.iterdir() if p.suffix.lower() in _VIDEO_EXTENSIONS]
    return sorted(videos)


class _VideoReadStage(CuratorStage):
    """Read video files from disk into task encoded_data."""

    @property
    def resources(self) -> CuratorStageResource:
        return CuratorStageResource(cpus=1.0, gpus=0.0)

    def process_data(self, tasks: list[SplitPipeTask]) -> list[SplitPipeTask] | None:  # type: ignore[override]
        for task in tasks:
            video = task.video
            video_path = Path(str(video.input_video))
            if not video_path.is_file():
                logger.warning(f"Video file not found: {video_path}")
                continue
            raw_bytes = video_path.read_bytes()
            video.encoded_data = bytes_to_numpy(raw_bytes)  # type: ignore[assignment]
            clip = video.clips[0]
            clip.encoded_data = bytes_to_numpy(raw_bytes)  # type: ignore[assignment]
            logger.info(f"Read {len(raw_bytes)} bytes from {video_path}")
        return tasks


class _VideoWriteStage(CuratorStage):
    """Write upscaled clip data back to disk."""

    def __init__(self, output_dir: str) -> None:
        self._output_dir = output_dir

    @property
    def resources(self) -> CuratorStageResource:
        return CuratorStageResource(cpus=1.0, gpus=0.0)

    def process_data(self, tasks: list[SplitPipeTask]) -> list[SplitPipeTask] | None:  # type: ignore[override]
        out_dir = Path(self._output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for task in tasks:
            video = task.video
            clip = video.clips[0]
            clip_bytes = clip.encoded_data.resolve()
            if clip_bytes is None:
                logger.warning(f"No encoded data for {video.input_video}")
                continue
            src_name = Path(str(video.input_video)).stem
            out_path = out_dir / f"{src_name}_sr.mp4"
            out_path.write_bytes(bytes(clip_bytes))
            logger.info(f"Wrote upscaled video to {out_path} ({clip_bytes.nbytes} bytes)")
        return tasks


def _build_tasks(video_paths: list[Path]) -> list[PipelineTask]:
    """Create pipeline tasks wrapping each video file.

    Each video is wrapped as a SplitPipeTask with a single Video and single Clip,
    so the reusable SuperResolutionStage can process it directly.
    """
    tasks: list[PipelineTask] = []
    for vp in video_paths:
        clip = Clip(
            uuid=uuid4(),
            source_video=str(vp),
            span=(0.0, 0.0),
        )
        video = Video(
            input_video=vp,
            clips=[clip],
        )
        task = SplitPipeTask(
            session_id=str(vp),
            video=video,
        )
        tasks.append(task)
    return tasks


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SeedVR2 Super-Resolution Example Pipeline")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory containing input video files.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to write upscaled videos.")

    sr = parser.add_argument_group("super-resolution")
    sr.add_argument(
        "--sr-variant",
        type=str,
        default="seedvr2_7b",
        choices=["seedvr2_3b", "seedvr2_7b", "seedvr2_7b_sharp"],
        help="SeedVR2 model variant.",
    )
    sr.add_argument("--sr-target-height", type=int, default=720, help="Target output height.")
    sr.add_argument("--sr-target-width", type=int, default=1280, help="Target output width.")
    sr.add_argument("--sr-window-frames", type=int, default=128, help="Frames per inference window.")
    sr.add_argument("--sr-overlap-frames", type=int, default=64, help="Overlap frames between windows.")
    sr.add_argument("--sr-no-blend-overlap", action="store_true", default=False, help="Disable overlap blending.")
    sr.add_argument("--sr-seed", type=int, default=666, help="Random seed for diffusion.")
    sr.add_argument("--sr-cfg-scale", type=float, default=1.0, help="Classifier-free guidance scale.")
    sr.add_argument("--sr-cfg-rescale", type=float, default=0.0, help="CFG rescale factor.")
    sr.add_argument("--sr-sample-steps", type=int, default=1, help="Number of diffusion sampling steps.")
    sr.add_argument("--sr-sp-size", type=int, default=1, help="Sequence parallelism size.")
    sr.add_argument("--sr-out-fps", type=float, default=None, help="Output FPS (None = preserve source).")
    sr.add_argument("--sr-tmp-dir", type=str, default=None, help="Temp directory for window segment files.")
    sr.add_argument("--verbose", action="store_true", default=False, help="Enable verbose logging.")

    return parser.parse_args()


def main() -> None:
    """Run the super-resolution example pipeline."""
    args = _parse_args()

    video_paths = _discover_videos(args.input_dir)
    logger.info(f"Found {len(video_paths)} videos in {args.input_dir}")
    if not video_paths:
        logger.warning("No video files found. Exiting.")
        return

    tasks = _build_tasks(video_paths)

    sr_config = SuperResolutionConfig(
        variant=args.sr_variant,
        target_height=args.sr_target_height,
        target_width=args.sr_target_width,
        window_frames=args.sr_window_frames,
        overlap_frames=args.sr_overlap_frames,
        blend_overlap=not args.sr_no_blend_overlap,
        seed=args.sr_seed,
        cfg_scale=args.sr_cfg_scale,
        cfg_rescale=args.sr_cfg_rescale,
        sample_steps=args.sr_sample_steps,
        sp_size=args.sr_sp_size,
        out_fps=args.sr_out_fps,
        tmp_dir=args.sr_tmp_dir,
        verbose=args.verbose,
    )

    stages: list[CuratorStage | CuratorStageSpec] = [
        _VideoReadStage(),
        *build_super_resolution_stages(sr_config),
        _VideoWriteStage(args.output_dir),
    ]

    run_pipeline(tasks, stages)
    logger.info("Super-resolution pipeline completed")


if __name__ == "__main__":
    main()
