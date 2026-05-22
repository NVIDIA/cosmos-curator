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

Downloads videos, splits them into TransNetV2 or fixed-stride clips,
transcodes each clip with FFmpeg, and writes the results to local or remote
storage.

Usage::

    python -m cosmos_curator.pipelines.ray_data.splitting_pipeline \
        --input-video-path /data/videos \
        --output-clip-path /data/clips
"""

import argparse
import logging
import math
from typing import Any, cast

import attrs
import ray
from ray.data import ActorPoolStrategy, TaskPoolStrategy

from cosmos_curator.core.interfaces.pipeline_interface import download_models
from cosmos_curator.core.utils.environment import MODEL_WEIGHTS_PREFIX
from cosmos_curator.core.utils.ffmpeg_utils import assert_ffmpeg_supports_h264
from cosmos_curator.core.utils.pixi_runtime_envs import ray_data_gpu_runtime_env
from cosmos_curator.core.utils.storage.storage_utils import get_files_relative, get_full_path, get_storage_client
from cosmos_curator.pipelines.ray_data._clip_transcoder import make_transcode_fn
from cosmos_curator.pipelines.ray_data._clip_writer import make_write_fn
from cosmos_curator.pipelines.ray_data._fixed_stride_splitter import make_split_fn
from cosmos_curator.pipelines.ray_data._summary_writer import write_summary
from cosmos_curator.pipelines.ray_data._transnetv2_splitter import (
    TransNetV2Splitter,
    transnetv2_model_ids,
    validate_transnetv2_length_bounds,
)
from cosmos_curator.pipelines.ray_data._video_reader import read_video
from cosmos_curator.pipelines.ray_data._vllm_caption import (
    _max_caption_workers,
    caption_window_rows,
    make_default_vllm_config,
    qwen_model_id,
    qwen_model_source,
    write_captioned_metadata_and_summary,
)
from cosmos_curator.pipelines.video.utils.data_model import VllmConfig

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


def _finite_float(value: str) -> float:
    """Parse a finite floating-point CLI value."""
    try:
        parsed = float(value)
    except ValueError as exc:
        msg = f"{value!r} is not a float"
        raise argparse.ArgumentTypeError(msg) from exc
    if not math.isfinite(parsed):
        msg = f"{value!r} must be finite"
        raise argparse.ArgumentTypeError(msg)
    return parsed


def _probability(value: str) -> float:
    """Parse a probability in the inclusive range [0, 1]."""
    parsed = _finite_float(value)
    if not 0.0 <= parsed <= 1.0:
        msg = f"{value!r} must be between 0 and 1"
        raise argparse.ArgumentTypeError(msg)
    return parsed


def _positive_float(value: str) -> float:
    """Parse a positive floating-point CLI value."""
    parsed = _finite_float(value)
    if parsed <= 0.0:
        msg = f"{value!r} must be positive"
        raise argparse.ArgumentTypeError(msg)
    return parsed


def _as_positive_int(value: object, name: str) -> int:
    """Coerce a direct namespace value to a positive integer."""
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, float) and value.is_integer():
        parsed = int(value)
    elif isinstance(value, str):
        parsed = _positive_int(value)
    else:
        msg = f"{name} must be an integer, got {value!r}"
        raise ValueError(msg)
    if parsed <= 0:
        msg = f"{name} must be positive, got {value!r}"
        raise ValueError(msg)
    return parsed


def _as_positive_float(value: object, name: str) -> float:
    """Coerce a direct namespace value to a positive finite float."""
    if isinstance(value, int | float):
        parsed = float(value)
    elif isinstance(value, str):
        parsed = _positive_float(value)
    else:
        msg = f"{name} must be a float, got {value!r}"
        raise TypeError(msg)
    if not math.isfinite(parsed) or parsed <= 0.0:
        msg = f"{name} must be positive, got {value!r}"
        raise ValueError(msg)
    return parsed


def _required_model_ids(args: argparse.Namespace, *, generate_captions: bool) -> list[str]:
    """Return model IDs required for the selected Ray Data pipeline."""
    model_ids: list[str] = []
    if getattr(args, "splitting_algorithm", "transnetv2") == "transnetv2":
        model_ids.extend(transnetv2_model_ids())
    if generate_captions:
        model_ids.append(qwen_model_id())
    return list(dict.fromkeys(model_ids))


def _validate_transnetv2_cluster_resources(args: argparse.Namespace, *, total_visible_gpus: int) -> None:
    """Fail fast when the selected TransNetV2 stage cannot be scheduled."""
    if getattr(args, "splitting_algorithm", "transnetv2") != "transnetv2":
        return
    decode_cpus = _as_positive_int(
        args.transnetv2_frame_decode_cpus_per_worker,
        "transnetv2_frame_decode_cpus_per_worker",
    )
    gpus_per_worker = _as_positive_float(args.transnetv2_gpus_per_worker, "transnetv2_gpus_per_worker")
    if float(total_visible_gpus) < gpus_per_worker:
        msg = (
            "TransNetV2 splitting requires visible GPUs. "
            "Use --splitting-algorithm fixed-stride or lower --transnetv2-gpus-per-worker."
        )
        raise ValueError(msg)

    ray_nodes = cast("list[dict[str, Any]]", ray.nodes())  # type: ignore[no-untyped-call]
    for node in ray_nodes:
        if not node.get("Alive", False):
            continue
        resources = node.get("Resources", {})
        if not isinstance(resources, dict):
            continue
        node_cpus = float(resources.get("CPU", 0.0))
        node_gpus = float(resources.get("GPU", 0.0))
        if node_cpus >= decode_cpus and node_gpus >= gpus_per_worker:
            return

    msg = (
        "TransNetV2 splitting requires at least one live Ray node with "
        f"{decode_cpus} CPU(s) and {gpus_per_worker:g} GPU(s) available for one worker. "
        "Lower --transnetv2-frame-decode-cpus-per-worker or --transnetv2-gpus-per-worker, "
        "or use --splitting-algorithm fixed-stride."
    )
    raise ValueError(msg)


def _caption_vllm_config(args: argparse.Namespace) -> VllmConfig | None:
    """Return a caption vLLM config only when the CLI overrides model defaults."""
    caption_batch_size = getattr(args, "caption_batch_size", None)
    if caption_batch_size is None:
        return None
    return attrs.evolve(make_default_vllm_config(), batch_size=caption_batch_size)


def _caption_workers_from_downloaded_gpus(total_visible_gpus: int, vllm_config: VllmConfig | None) -> int:
    """Return the caption actor ceiling from the GPU count discovered during model download."""
    worker_config = vllm_config or make_default_vllm_config()
    return _max_caption_workers(total_visible_gpus, worker_config.num_gpus)


def _apply_split_stage(ds: ray.data.Dataset, args: argparse.Namespace, *, download_slots: int) -> ray.data.Dataset:
    """Apply the selected span-generation stage to the video dataset."""
    splitting_algorithm = getattr(args, "splitting_algorithm", "transnetv2")
    if splitting_algorithm == "fixed-stride":
        return ds.map(
            make_split_fn(
                clip_len_s=args.fixed_stride_split_duration,
                clip_stride_s=args.fixed_stride_split_duration,
                min_clip_length_s=args.fixed_stride_min_clip_length_s,
                limit_clips=args.limit_clips,
            ),
            num_cpus=0.25,
            compute=TaskPoolStrategy(size=download_slots),
        )

    if splitting_algorithm == "transnetv2":
        decode_cpus = _as_positive_int(
            args.transnetv2_frame_decode_cpus_per_worker,
            "transnetv2_frame_decode_cpus_per_worker",
        )
        transnetv2_kwargs: dict[str, Any] = {
            "threshold": args.transnetv2_threshold,
            "min_length_s": args.transnetv2_min_length_s,
            "min_length_frames": args.transnetv2_min_length_frames,
            "max_length_s": args.transnetv2_max_length_s,
            "max_length_mode": args.transnetv2_max_length_mode,
            "crop_s": args.transnetv2_crop_s,
            "num_decode_cpus_per_worker": decode_cpus,
            "limit_clips": args.limit_clips,
        }
        transnetv2_fn = cast("Any", TransNetV2Splitter)
        return ds.map(
            transnetv2_fn,
            fn_constructor_kwargs=transnetv2_kwargs,
            num_cpus=decode_cpus,
            num_gpus=args.transnetv2_gpus_per_worker,
            compute=ActorPoolStrategy(min_size=1, max_size=download_slots, initial_size=1),
            runtime_env=ray_data_gpu_runtime_env("unified"),
            scheduling_strategy="DEFAULT",
        )

    msg = f"Unknown splitting algorithm: {splitting_algorithm}"
    raise ValueError(msg)


def run(args: argparse.Namespace) -> int:
    """Build and execute the splitting pipeline.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Number of clips written.

    """
    if getattr(args, "splitting_algorithm", "transnetv2") == "transnetv2":
        validate_transnetv2_length_bounds(args.transnetv2_min_length_s, args.transnetv2_max_length_s)

    assert_ffmpeg_supports_h264()

    generate_captions = getattr(args, "generate_captions", True)
    model_ids = _required_model_ids(args, generate_captions=generate_captions)
    num_gpus_available = 0
    if model_ids:
        num_gpus_available = int(download_models(model_ids, args.model_weights_path))

    caption_model_source: str | None = None
    if generate_captions:
        caption_model_source = qwen_model_source()

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    _configure_ray_data_progress(progress=bool(getattr(args, "progress", False)))

    video_paths = _discover_videos(args.input_video_path, limit=args.limit)
    if not video_paths:
        logger.warning("No videos found in %s", args.input_video_path)
        return 0
    logger.info("Found %d input video(s)", len(video_paths))

    _validate_transnetv2_cluster_resources(args, total_visible_gpus=num_gpus_available)

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

    # Stage 2: Compute clip spans (1:1, no fan-out).
    ds = _apply_split_stage(ds, args, download_slots=download_slots)

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
        vllm_config = _caption_vllm_config(args)
        ds = caption_window_rows(
            ds,
            model_source=caption_model_source,
            caption_workers=_caption_workers_from_downloaded_gpus(num_gpus_available, vllm_config),
            vllm_config=vllm_config,
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
        "--splitting-algorithm",
        type=str,
        default="transnetv2",
        choices=["fixed-stride", "transnetv2"],
        help="Splitting algorithm to use on full videos.",
    )
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
        "--transnetv2-threshold",
        type=_probability,
        default=0.4,
        help=(
            "TransNetV2 probability threshold above which a frame is classified as a shot transition. "
            "Default is 0.4, which prioritizes recall over precision."
        ),
    )
    parser.add_argument(
        "--transnetv2-min-length-s",
        type=float,
        default=2.0,
        help=(
            "Minimum length of clips (in seconds) for TransNetV2 splitting stage. "
            "If specified, will remove any scenes below this length."
        ),
    )
    parser.add_argument(
        "--transnetv2-min-length-frames",
        type=_positive_int,
        default=48,
        help=(
            "Minimum length of clips (in frames) for TransNetV2 splitting stage. "
            "If specified, will remove any scenes below this length."
        ),
    )
    parser.add_argument(
        "--transnetv2-max-length-s",
        type=float,
        default=60.0,
        help=(
            "Maximum length of clips (in seconds) for TransNetV2 splitting stage. "
            "If specified, will deal with the scene by the `max_length_mode` specified."
        ),
    )
    parser.add_argument(
        "--transnetv2-max-length-mode",
        type=str,
        default="stride",
        choices=["truncate", "stride"],
        help=(
            "Maximum length mode for TransNetV2 splitting stage. "
            "If `truncate`, will truncate the scene to `max_length_s`. "
            "If `stride`, will generate a number of max_length_s scenes until the end of the scene. "
            "If the end scene is less than `min_length_s`, it will drop the last scene."
        ),
    )
    parser.add_argument(
        "--transnetv2-crop-s",
        type=float,
        default=0.5,
        help=(
            "Crop size for TransNetV2 splitting stage. If specified, will crop each scene at start and end. "
            "E.g. 0.25 will crop ~250ms from start, and ~250ms from end frame (reducing all clips by ~0.5 seconds). "
            "If cropped scenes result in zero-length scenes, these will be filtered."
        ),
    )
    parser.add_argument(
        "--transnetv2-frame-decode-cpus-per-worker",
        type=_positive_int,
        default=3,
        help="Number of CPU threads per worker for video frame decoding when using ffmpeg_cpu mode.",
    )
    parser.add_argument(
        "--transnetv2-gpus-per-worker",
        type=_positive_float,
        default=0.25,
        help="Number of GPUs per worker for TransNetV2 splitting stage.",
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
        "--caption-batch-size",
        type=_positive_int,
        default=None,
        help="Ray Data caption batch size in clip rows. Defaults to the selected caption model config.",
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
