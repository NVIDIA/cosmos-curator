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

r"""Ray Data video splitting pipeline.

Downloads videos, splits them into fixed-stride clips, transcodes each clip
with FFmpeg, and writes the results to local or remote storage.

Usage::

    python -m cosmos_curator.pipelines.ray_data.splitting_pipeline \
        --input-video-path /data/videos \
        --output-clip-path /data/clips
"""

import argparse
import logging

import ray
from ray.data import TaskPoolStrategy

from cosmos_curator.core.interfaces.pipeline_interface import download_models
from cosmos_curator.core.utils.environment import MODEL_WEIGHTS_PREFIX
from cosmos_curator.core.utils.ffmpeg_utils import assert_ffmpeg_supports_h264
from cosmos_curator.core.utils.storage.storage_utils import get_files_relative, get_full_path, get_storage_client
from cosmos_curator.pipelines.ray_data._clip_transcoder import make_transcode_fn
from cosmos_curator.pipelines.ray_data._clip_writer import make_write_fn
from cosmos_curator.pipelines.ray_data._fixed_stride_splitter import make_split_fn
from cosmos_curator.pipelines.ray_data._summary_writer import write_summary
from cosmos_curator.pipelines.ray_data._video_reader import read_video
from cosmos_curator.pipelines.ray_data._vllm_caption import (
    caption_window_rows,
    qwen_model_id,
    qwen_model_source,
    visible_caption_workers,
    write_captioned_metadata_and_summary,
)

logger = logging.getLogger(__name__)

# Per-node cap on concurrent video downloads. Prevents the object-store
# blowout that fractional-CPU tasks would otherwise cause at ramp-up; in
# steady state transcode backpressure keeps in-flight downloads well below
# this ceiling.
_DOWNLOAD_SLOTS_PER_NODE = 16


def _configure_ray_data_progress(*, progress: bool) -> None:
    """Configure Ray Data progress output before creating datasets."""
    ctx = ray.data.DataContext.get_current()
    ctx.enable_progress_bars = progress
    ctx.enable_operator_progress_bars = progress
    ctx.enable_rich_progress_bars = progress
    ctx.print_on_execution_start = progress
    ctx.use_ray_tqdm = False


def _download_slots_for_video_count(*, num_videos: int, num_nodes: int) -> int:
    """Cap read/split task concurrency by both cluster size and input count."""
    if num_videos <= 0:
        return 0
    cluster_cap = _DOWNLOAD_SLOTS_PER_NODE * max(1, num_nodes)
    return min(num_videos, cluster_cap)


def _discover_videos(input_video_path: str, limit: int = 0) -> list[str]:
    """List video files under *input_video_path* and return full paths.

    Works for both local directories and remote storage (S3/Azure).
    """
    client = get_storage_client(input_video_path)
    relative_paths = get_files_relative(input_video_path, client, limit)
    return [str(get_full_path(input_video_path, rp)) for rp in relative_paths]


def _positive_int(value: str) -> int:
    """Parse a positive integer CLI value."""
    try:
        parsed = int(value)
    except ValueError as exc:
        msg = f"{value!r} is not an integer"
        raise argparse.ArgumentTypeError(msg) from exc
    if parsed <= 0:
        msg = f"{value!r} must be positive"
        raise argparse.ArgumentTypeError(msg)
    return parsed


def run(args: argparse.Namespace) -> int:
    """Build and execute the splitting pipeline.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Number of clips written.

    """
    assert_ffmpeg_supports_h264()

    generate_captions = getattr(args, "generate_captions", True)
    caption_model_source: str | None = None
    if generate_captions:
        model_id = qwen_model_id()
        download_models([model_id], args.model_weights_path)
        caption_model_source = qwen_model_source()

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    _configure_ray_data_progress(progress=bool(getattr(args, "progress", False)))

    video_paths = _discover_videos(args.input_video_path, limit=args.limit)
    if not video_paths:
        logger.warning("No videos found in %s", args.input_video_path)
        return 0
    logger.info("Found %d input video(s)", len(video_paths))

    # Seed dataset — one row per video.
    ds: ray.data.Dataset = ray.data.from_items([{"video_path": vp} for vp in video_paths])

    # Fractional CPU (matching Xenna's 0.25 precedent for IO-bound download)
    # with an explicit TaskPoolStrategy size cap scaled to the smaller of
    # cluster size and input rows.
    # Task compute (not actor pool) preserves fusion with split.
    download_slots = _download_slots_for_video_count(
        num_videos=len(video_paths),
        num_nodes=len(ray.nodes()),  # type: ignore[no-untyped-call]
    )

    # Stage 1: Download + extract metadata (1:1).
    ds = ds.map(
        read_video,
        num_cpus=0.25,
        compute=TaskPoolStrategy(size=download_slots),
    )

    # Stage 2: Compute clip spans (1:1, no fan-out). Matching resources +
    # compute strategy keeps this fused with read_video.
    ds = ds.map(
        make_split_fn(
            clip_len_s=args.fixed_stride_split_duration,
            clip_stride_s=args.fixed_stride_split_duration,
            min_clip_length_s=args.fixed_stride_min_clip_length_s,
            limit_clips=args.limit_clips,
        ),
        num_cpus=0.25,
        compute=TaskPoolStrategy(size=download_slots),
    )

    # Stage 3: Transcode + fan-out (1:N — one video in, N clips out).
    ds = ds.flat_map(
        make_transcode_fn(
            encoder=args.transcode_encoder,
            encoder_threads=args.transcode_encoder_threads,
            ffmpeg_batch_size=args.transcode_ffmpeg_batch_size,
            use_input_bit_rate=args.transcode_use_input_video_bit_rate,
        ),
        num_cpus=args.transcode_cpus_per_worker,
    )

    # Stage 4: Write clips to output (1:1). IO-bound upload; fractional CPU
    # matches Xenna's ClipWriterStage at cpus=0.25.
    if generate_captions:
        ds = ds.map(
            make_write_fn(args.output_clip_path, write_metadata=False, keep_clip_bytes=True),
            num_cpus=0.25,
        )

        if caption_model_source is None:
            msg = "caption_model_source must be set when generate_captions is True"
            raise RuntimeError(msg)
        ds = caption_window_rows(
            ds,
            model_source=caption_model_source,
            caption_workers=visible_caption_workers(),
            vllm_max_num_seqs=getattr(args, "vllm_max_num_seqs", 64),
        )

        num_clips = write_captioned_metadata_and_summary(
            ds,
            input_video_path=args.input_video_path,
            output_path=args.output_clip_path,
            num_input_videos=len(video_paths),
        )
        logger.info("Wrote %d clip(s) with captions to %s", num_clips, args.output_clip_path)
        return num_clips

    ds = ds.map(make_write_fn(args.output_clip_path), num_cpus=0.25)

    # Stage 5: Aggregate per-video and write summary.json.
    num_clips = write_summary(
        ds,
        input_video_path=args.input_video_path,
        output_path=args.output_clip_path,
        num_input_videos=len(video_paths),
    )
    logger.info("Wrote %d clip(s) to %s", num_clips, args.output_clip_path)
    return num_clips


def _setup_parser(parser: argparse.ArgumentParser) -> None:
    """Add CLI arguments matching the existing splitting pipeline where applicable."""
    parser.add_argument(
        "--input-video-path",
        type=str,
        required=True,
        help="S3 or local path containing input raw videos.",
    )
    parser.add_argument(
        "--output-clip-path",
        type=str,
        required=True,
        help="S3 or local path to store output clips.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of input videos to process.",
    )
    # --- Splitting ---
    parser.add_argument(
        "--fixed-stride-split-duration",
        type=int,
        default=10,
        help="Duration of clips (in seconds) generated from the fixed stride splitting stage.",
    )
    parser.add_argument(
        "--fixed-stride-min-clip-length-s",
        type=float,
        default=2,
        help="Minimum length of clips (in seconds) for fixed stride splitting stage.",
    )
    parser.add_argument(
        "--limit-clips",
        type=int,
        default=0,
        help="Limit number of clips from each input video to process.",
    )
    parser.add_argument(
        "--no-generate-captions",
        dest="generate_captions",
        action="store_false",
        default=True,
        help="Whether to generate captions for clip windows.",
    )
    parser.add_argument(
        "--model-weights-path",
        type=str,
        default=MODEL_WEIGHTS_PREFIX,
        help=(
            "Local path or S3 prefix for model weights. Used to download model weights to local cache if they are not "
            "already present. If a unix path is provided, it must be accessible from all nodes."
        ),
    )
    parser.add_argument(
        "--vllm-max-num-seqs",
        type=_positive_int,
        default=64,
        help=(
            "Advanced vLLM tuning: maximum active sequences per vLLM engine. Ray Data batch size is derived "
            "from this value and Ray's default per-actor in-flight task count. The default works well on "
            "A100/H100/B200 in benchmark runs; GB200 benchmark runs benefited from higher values."
        ),
    )
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to show Ray Data progress bars. Disabled by default so redirected or tee'd logs stay readable.",
    )
    # --- Transcode ---
    parser.add_argument(
        "--transcode-encoder",
        type=str,
        default="libopenh264",
        choices=["libopenh264"],
        help="Codec for transcoding clips.",
    )
    parser.add_argument(
        "--transcode-encoder-threads",
        type=int,
        default=1,
        help="Number of threads per ffmpeg encoding sub-command for transcoding clips.",
    )
    parser.add_argument(
        "--transcode-ffmpeg-batch-size",
        type=int,
        default=16,
        help="FFmpeg batch size for transcoding clips.",
    )
    parser.add_argument(
        "--transcode-cpus-per-worker",
        type=float,
        default=5.0,
        help="Number of CPUs per transcoding worker.",
    )
    parser.add_argument(
        "--transcode-use-input-video-bit-rate",
        action="store_true",
        default=False,
        help="Whether to use input video's bit rate for encoding clips.",
    )


def main() -> None:
    """Entry point for the Ray Data splitting pipeline."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(
        description="Ray Data video splitting pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _setup_parser(parser)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
