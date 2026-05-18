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
"""Ray pipelines.

Which:
  - Download videos
  - Split videos into segments
  - Transcode raw videos into clips
  - Generate an embedding for the clip
  - Caption the clip
"""

import argparse
import pathlib
import time
from typing import Any, cast

import attrs
from loguru import logger

from cosmos_curator.core.interfaces.pipeline_interface import run_pipeline
from cosmos_curator.core.interfaces.stage_interface import CuratorStage, CuratorStageSpec
from cosmos_curator.core.utils.config import args_utils
from cosmos_curator.core.utils.ffmpeg_utils import assert_ffmpeg_supports_h264
from cosmos_curator.core.utils.infra.hardware_info import get_gpu_infos
from cosmos_curator.core.utils.infra.profiling import profiling_scope
from cosmos_curator.core.utils.misc.stage_compare import get_stage_name_after, get_stages_to_compare, run_stage_compare
from cosmos_curator.core.utils.misc.stage_replay import (
    StageSaveConfig,
    add_stage_replay_args,
    get_stages_to_replay,
    run_stage_replay,
    validate_stage_replay_args,
)
from cosmos_curator.core.utils.storage.storage_utils import (
    create_path,
    get_full_path,
    is_path_nested,
    verify_path,
)
from cosmos_curator.pipelines.pipeline_args import (
    add_common_args,
)
from cosmos_curator.pipelines.video.captioning.captioning_builders import (
    VLLM_CAPTION_ALGOS,
    CaptionBackendConfig,
    CaptioningConfig,
    EnhanceCaptionConfig,
    GeminiConfig,
    OpenAIConfig,
    T5Config,
    VllmAsyncCaptionConfig,
    build_captioning_stages,
    build_t5_stages,
)
from cosmos_curator.pipelines.video.captioning.per_event_caption_stage import PerEventCaptionStage
from cosmos_curator.pipelines.video.captioning.per_event_cli_args import (
    add_event_caption_args,
    resolve_event_caption_prompt,
)
from cosmos_curator.pipelines.video.captioning.vllm_async_config import (
    add_vllm_async_cli_args,
    build_vllm_async_config,
)
from cosmos_curator.pipelines.video.clipping.clipping_builders import (
    FixedStrideSplitConfig,
    FrameExtractionConfig,
    TranscodeConfig,
    TransNetV2SplitConfig,
    build_fixed_stride_split_stages,
    build_frame_extraction_stages,
    build_transcode_stages,
    build_transnetv2_split_stages,
)
from cosmos_curator.pipelines.video.embedding.embedding_builders import (
    CosmosEmbed1Config,
    EmbeddingBackendConfig,
    EmbeddingConfig,
    InternVideo2Config,
    OpenAIEmbeddingConfig,
    build_embedding_stages,
    get_embedding_model_version,
)
from cosmos_curator.pipelines.video.filtering.aesthetics.aesthetics_builders import (
    AestheticFilterConfig,
    ArtificialTextFilterConfig,
    VideoClassifierConfig,
    VlmFilterConfig,
    build_aesthetic_filter_stages,
    build_artificial_text_filter_stages,
    build_vllm_filter_classifier_stages,
)
from cosmos_curator.pipelines.video.filtering.motion.motion_builders import (
    MotionFilterConfig,
    build_motion_filter_stages,
)
from cosmos_curator.pipelines.video.read_write.metadata_writer_stage import (
    ClipWriterStage,
    consolidate_lance_fragments,
)
from cosmos_curator.pipelines.video.read_write.read_write_builders import (
    IngestConfig,
    OutputConfig,
    build_ingest_stages,
    build_output_stages,
)
from cosmos_curator.pipelines.video.read_write.summary_writers import (
    write_split_summary,
)
from cosmos_curator.pipelines.video.super_resolution.super_resolution_builders import (
    SuperResolutionConfig,
    build_super_resolution_stages,
)
from cosmos_curator.pipelines.video.tracking.cli_args import add_sam3_args
from cosmos_curator.pipelines.video.tracking.sam3_bbox_stage import SAM3QualityConfig
from cosmos_curator.pipelines.video.tracking.tracking_builders import (
    SAM3TrackingConfig,
    build_sam3_tracking_stages,
)
from cosmos_curator.pipelines.video.utils import data_model_compare  # noqa: F401  # registers SplitPipeTaskComparator
from cosmos_curator.pipelines.video.utils.data_model import (
    SplitPipeTask,
    VllmAsyncConfig,
    VllmConfig,
    VllmSamplingConfig,
    WindowConfig,
)
from cosmos_curator.pipelines.video.utils.video_pipe_input import (
    extract_multi_cam_split_tasks,
    extract_single_cam_split_tasks,
    format_session_videos_tree,
)

QWEN2_CAPTION_ALGOS = {"qwen"}
QWEN3_CAPTION_ALGOS = {
    "qwen3_5_27b",
    "qwen3_6_27b",
    "qwen3_6_27b_fp8",
    "qwen3_vl_30b",
    "qwen3_vl_30b_fp8",
    "qwen3_vl_235b",
    "qwen3_vl_235b_fp8",
}
COSMOS_REASON_ALGOS = {"cosmos_r1", "cosmos_r2"}
ALL_CAPTION_ALGOS = VLLM_CAPTION_ALGOS | {"gemini", "openai", "vllm_async"}
MULTICAM_VIDEO_EXTENSIONS: set[str] = {".mp4"}
QWEN3_VL_235B_HIGH_MEMORY_GPU_THRESHOLD_MB = 128_000


def _clamp_num_gpus_for_qwen3_235b(model_variant: str, num_gpus: int) -> int:
    """Warn and clamp num_gpus to the minimum required for large Qwen3-235B variants."""
    _QWEN3_VL_235B_FP8_MIN_GPUS = 4
    if model_variant == "qwen3_vl_235b":
        min_gpus = _get_qwen3_vl_235b_min_gpus()
        if num_gpus < min_gpus:
            logger.warning(f"qwen3_vl_235b requires at least {min_gpus} GPUs, setting num_gpus to {min_gpus}")
            return min_gpus
    elif model_variant == "qwen3_vl_235b_fp8" and num_gpus < _QWEN3_VL_235B_FP8_MIN_GPUS:
        logger.warning(
            f"qwen3_vl_235b_fp8 requires at least {_QWEN3_VL_235B_FP8_MIN_GPUS} GPUs, "
            f"setting num_gpus to {_QWEN3_VL_235B_FP8_MIN_GPUS}"
        )
        return _QWEN3_VL_235B_FP8_MIN_GPUS
    return num_gpus


def _get_qwen3_vl_235b_min_gpus() -> int:
    """Determine the minimum number of GPUs required for qwen3_vl_235b based on available hardware.

    High-memory GPUs (e.g. GB200 with ~192 GB per GPU) can fit the model on fewer GPUs,
    while lower-memory GPUs (e.g. H100 with ~80 GB) need more GPUs for tensor parallelism.

    The threshold is defined by QWEN3_VL_235B_HIGH_MEMORY_GPU_THRESHOLD_MB.

    Returns:
        4 if per-GPU memory exceeds the threshold, 8 otherwise.

    """
    gpu_infos = get_gpu_infos()
    if gpu_infos and gpu_infos[0].memory_total >= QWEN3_VL_235B_HIGH_MEMORY_GPU_THRESHOLD_MB:
        return 4
    return 8


def build_input_data(
    args: argparse.Namespace,
) -> tuple[list[SplitPipeTask], list[str], int, int]:
    """Build input data for the pipeline.

    This function validates input arguments, extracts input data, and returns a list of tasks and relative paths.

    Args:
        args: Command line arguments.

    Returns:
        A tuple containing:
        - A list of SplitPipeTask objects.
        - A list of relative paths to the input videos.
        - The number of processed videos.
        - The number of input videos selected for this run after applying skip/limit semantics.

    """
    # validate input arguments
    verify_path(args.input_video_path)
    verify_path(args.output_clip_path, level=1)
    create_path(args.output_clip_path)
    if is_path_nested(args.input_video_path, args.output_clip_path):
        error_msg = "Do not make input and output paths nested"
        raise ValueError(error_msg)

    if args.multi_cam and args.splitting_algorithm != "fixed-stride":
        error_msg = "Multi-cam only supports fixed-stride splitting; set --splitting-algorithm fixed-stride"
        raise ValueError(error_msg)

    # extract input data
    if args.multi_cam:
        input_tasks = extract_multi_cam_split_tasks(
            sessions_prefix=args.input_video_path,
            primary_camera_keyword=args.primary_camera_keyword,
            video_extensions=MULTICAM_VIDEO_EXTENSIONS,
            input_s3_profile_name=args.input_s3_profile_name,
            limit=args.limit,
            verbose=args.verbose,
        )

        if args.verbose:
            tree_output = format_session_videos_tree(input_tasks, args.input_video_path, limit=3)
            logger.info(tree_output)

        # TODO(jbowles): input_videos_relative is used for summary writing, which needs to be
        # updated to support multicam. See docs/curator/design/multicam.md for the plan for
        # this update.
        input_videos_relative: list[str] = []
        num_processed = 0
        num_input_videos_selected = len(input_tasks)
        logger.info(f"About to process {len(input_tasks)} multi-cam session tasks ...")
    else:
        input_videos, input_videos_relative, num_processed = extract_single_cam_split_tasks(
            input_path=args.input_video_path,
            input_video_list_json_path=args.input_video_list_json_path,
            output_path=args.output_clip_path,
            output_video_path=ClipWriterStage.get_output_path_processed_videos(args.output_clip_path),
            output_clip_chunk_path=ClipWriterStage.get_output_path_processed_clip_chunks(args.output_clip_path),
            input_s3_profile_name=args.input_s3_profile_name,
            input_video_list_s3_profile_name=args.input_video_list_s3_profile_name,
            output_s3_profile_name=args.output_s3_profile_name,
            limit=args.limit,
            verbose=args.verbose,
        )
        input_tasks = [SplitPipeTask(videos=[video], session_id=str(video.input_video)) for video in input_videos]

        if len(input_videos) == 0:
            logger.warning(
                "About to process 0 raw videos - all inputs were already processed. "
                f"Remove the output directory {ClipWriterStage.get_output_path_processed_videos(args.output_clip_path)}"
                " to reprocess.",
            )
        else:
            logger.info(f"About to process {len(input_videos)} raw videos ...")

        if args.verbose:
            logger.debug("\n".join(str(x.input_video) for x in input_videos))
        num_input_videos_selected = len(input_videos)

    return input_tasks, input_videos_relative, num_processed, num_input_videos_selected


def write_summary(
    args: argparse.Namespace,
    input_videos: list[str],
    num_input_videos_selected: int,
    output_tasks: list[SplitPipeTask],
    pipeline_run_time: float,
) -> float:
    """Write a summary of the pipeline run.

    This function writes a summary of the pipeline run, including the total video length and performance metrics.

    Args:
        args: Command line arguments.
        input_videos: List of input video paths.
        num_input_videos_selected: Number of input videos selected for this run after applying skip/limit semantics.
        output_tasks: List of output tasks.
        pipeline_run_time: Total runtime of the pipeline in minutes.

    Returns:
        Total video length in hours.

    """
    write_split_summary(
        args.input_video_path,
        input_videos,
        num_input_videos_selected,
        args.output_clip_path,
        args.output_s3_profile_name,
        output_tasks,
        args.embedding_algorithm,
        args.limit,
        perf_profile=args.perf_profile,
        pipeline_run_time=pipeline_run_time,
        write_all_caption_json=args.write_all_caption_json,
        multi_cam=args.multi_cam,
    )

    if args.perf_profile:
        total_object_size = 0
        for task in output_tasks:
            total_object_size += task.get_major_size()
        logger.info(f"Total object size: {total_object_size:,} bytes")

    total_video_length = 0.0
    for task in output_tasks:
        for video in task.videos:
            if video.clip_chunk_index == 0:
                total_video_length += video.metadata.duration / 3600 if video.metadata.duration else 0

    return total_video_length


def _get_vllm_sampling_defaults() -> dict[str, Any]:
    """Get default values from VllmSamplingConfig.

    Returns:
        Dictionary mapping field names to default values.

    """
    default_config = VllmSamplingConfig()
    return {field.name: getattr(default_config, field.name) for field in attrs.fields(VllmSamplingConfig)}


def _assemble_stages(  # noqa: C901, PLR0912, PLR0915
    args: argparse.Namespace,
) -> list[CuratorStage | CuratorStageSpec]:
    """Assemble the pipeline stage list.

    Constructs config objects from command-line args and calls stage builder
    functions in dependency order. Returns the flat, ordered stage list.

    Args:
        args: Command line arguments.

    Returns:
        The ordered stage list ready for run_pipeline().

    """
    stages: list[CuratorStage | CuratorStageSpec] = []
    # Single CLI source: captioning and writer must observe the same enabled value.
    caption_quality_flags_enabled = args.caption_quality_flags_enabled

    # --- Ingest (always) ---
    stages.extend(
        build_ingest_stages(
            IngestConfig(
                input_path=args.input_video_path,
                num_workers_per_node=args.num_download_workers_per_node,
                input_s3_profile_name=args.input_s3_profile_name,
                verbose=args.verbose,
                perf_profile=args.perf_profile,
            )
        )
    )

    # --- Split (always) ---
    if args.splitting_algorithm == "fixed-stride":
        stages.extend(
            build_fixed_stride_split_stages(
                FixedStrideSplitConfig(
                    clip_len_s=args.fixed_stride_split_duration,
                    clip_stride_s=args.fixed_stride_split_duration,
                    min_clip_length_s=args.fixed_stride_min_clip_length_s,
                    limit_clips=args.limit_clips,
                    verbose=args.verbose,
                    perf_profile=args.perf_profile,
                )
            )
        )
    elif args.splitting_algorithm == "transnetv2":
        # TransNetV2 is a neural-network based shot-detection algorithm
        # that takes strided windows of ~100 frames and detects whether
        # a given frame is a scene transition or not.
        # See https://arxiv.org/abs/2008.04838 for more details.
        stages.extend(
            build_transnetv2_split_stages(
                TransNetV2SplitConfig(
                    threshold=args.transnetv2_threshold,
                    min_length_s=args.transnetv2_min_length_s,
                    min_length_frames=args.transnetv2_min_length_frames,
                    max_length_s=args.transnetv2_max_length_s,
                    max_length_mode=args.transnetv2_max_length_mode,
                    crop_s=args.transnetv2_crop_s,
                    num_gpus_per_worker=args.transnetv2_gpus_per_worker,
                    decoder_mode=args.transnetv2_frame_decoder_mode,
                    num_decode_cpus_per_worker=args.transnetv2_frame_decode_cpus_per_worker,
                    raise_on_pynvc_error=args.transnetv2_frame_decode_raise_on_pynvc_error,
                    limit_clips=args.limit_clips,
                    verbose=args.verbose,
                    perf_profile=args.perf_profile,
                )
            )
        )
    else:
        error_msg = f"{args.splitting_algorithm} algorithm type not implemented."
        raise NotImplementedError(error_msg)

    # --- Transcode (always) ---
    stages.extend(
        build_transcode_stages(
            TranscodeConfig(
                num_cpus_per_worker=args.transcode_cpus_per_worker,
                encoder=args.transcode_encoder,
                encoder_threads=args.transcode_encoder_threads,
                encode_batch_size=args.transcode_ffmpeg_batch_size,
                use_hwaccel=args.transcode_use_hwaccel,
                use_input_bit_rate=args.transcode_use_input_video_bit_rate,
                num_clips_per_chunk=args.clip_re_chunk_size,
                max_output_frames=args.transcode_max_output_frames,
                verbose=args.verbose,
                perf_profile=args.perf_profile,
            )
        )
    )

    # --- Super-resolution (optional) ---
    if args.super_resolution:
        stages.extend(
            build_super_resolution_stages(
                SuperResolutionConfig(
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
                    perf_profile=args.perf_profile,
                )
            )
        )

    # --- Motion filter (optional) ---
    if args.motion_filter != "disable":
        stages.extend(
            build_motion_filter_stages(
                MotionFilterConfig(
                    score_only=args.motion_filter == "score-only",
                    global_mean_threshold=args.motion_global_mean_threshold,
                    per_patch_min_256_threshold=args.motion_per_patch_min_256_threshold,
                    decode_cpus_per_worker=args.motion_decode_cpus_per_worker,
                    decode_target_fps=args.motion_decode_target_fps,
                    decode_target_duration_ratio=args.motion_decode_target_duration_ratio,
                    score_gpus_per_worker=args.motion_score_gpus_per_worker,
                    score_batch_size=args.motion_score_batch_size,
                    verbose=args.verbose,
                    perf_profile=args.perf_profile,
                )
            )
        )

    # --- Frame extraction (shared prerequisite for aesthetics and embedding) ---
    has_aesthetics = args.aesthetic_threshold is not None
    has_embeddings = args.generate_embeddings
    if has_aesthetics or has_embeddings:
        target_fps: list[float | int] = [1, 2] if has_aesthetics and has_embeddings else [1] if has_aesthetics else [2]
        stages.extend(
            build_frame_extraction_stages(
                FrameExtractionConfig(
                    target_fps=target_fps,
                    target_res=args.clip_extraction_target_res,
                    decoder_mode=args.clip_extraction_decoder_mode,
                    cpus_per_worker=args.clip_extraction_cpus_per_worker,
                    perf_profile=args.perf_profile,
                )
            )
        )

    # --- Aesthetic filter (optional) ---
    if has_aesthetics:
        stages.extend(
            build_aesthetic_filter_stages(
                AestheticFilterConfig(
                    score_threshold=args.aesthetic_threshold,
                    reduction=args.aesthetic_reduction,
                    gpus_per_worker=args.aesthetic_gpus_per_worker,
                    verbose=args.verbose,
                    perf_profile=args.perf_profile,
                )
            )
        )

    # --- Artificial text filter (optional) ---
    if args.artificial_text_filter:
        stages.extend(
            build_artificial_text_filter_stages(
                ArtificialTextFilterConfig(
                    use_gpu=not args.artificial_text_detection_use_cpu,
                    gpus_per_worker=args.artificial_text_gpus_per_worker,
                    use_corner_detection=args.artificial_text_use_corner_detection,
                    frame_interval=args.artificial_text_frame_interval,
                    min_duration_frames=args.artificial_text_min_duration_frames,
                    min_duration_frames_corner_ratio=args.artificial_text_corner_ratio,
                    stability_iou_threshold=args.artificial_text_stability_iou_threshold,
                    ignore_corner_region=args.artificial_text_ignore_corner_region,
                    corner_x_margin_norm=args.artificial_text_corner_x_margin,
                    corner_y_margin_norm=args.artificial_text_corner_y_margin,
                    verbose=args.verbose,
                    perf_profile=args.perf_profile,
                )
            )
        )

    # --- VLM semantic filter and/or video classifier (optional) ---
    vlm_filter_num_gpus = _clamp_num_gpus_for_qwen3_235b(args.vlm_filter_model_variant, args.vlm_filter_num_gpus)
    video_classifier_num_gpus = _clamp_num_gpus_for_qwen3_235b(
        args.video_classifier_model_variant, args.video_classifier_num_gpus
    )
    vlm_filter_cfg = (
        VlmFilterConfig(
            score_only=args.vlm_filter == "score-only",
            model_variant=args.vlm_filter_model_variant,
            filter_categories=args.vlm_filter_categories,
            prompt_variant=args.vlm_filter_prompt_variant,
            rejection_threshold=args.vlm_filter_rejection_threshold,
            batch_size=args.vlm_filter_batch_size,
            fp8_enable=args.vlm_filter_fp8_enable,
            max_output_tokens=args.vlm_filter_max_output_tokens,
            num_gpus=vlm_filter_num_gpus,
            use_mmcache=args.qwen_use_vllm_mmcache,
            sampling_fps=args.captioning_sampling_fps,
            window_size=args.captioning_window_size,
            remainder_threshold=args.captioning_remainder_threshold,
            preprocess_dtype=args.qwen_preprocess_dtype,
            model_does_preprocess=args.qwen_model_does_preprocess,
            generate_previews=args.generate_previews,
            verbose=args.verbose,
            perf_profile=args.perf_profile,
            endpoint=args.vlm_filter_endpoint,
            openai_model_name=args.vlm_filter_openai_model_name,
            openai_max_caption_retries=args.vlm_filter_openai_retries,
            openai_retry_delay_seconds=args.vlm_filter_openai_retry_delay_seconds,
            gemini_model_name=args.vlm_filter_gemini_model_name,
            gemini_max_caption_retries=args.vlm_filter_gemini_retries,
            gemini_retry_delay_seconds=args.vlm_filter_gemini_retry_delay_seconds,
        )
        if args.vlm_filter != "disable"
        else None
    )
    video_classifier_cfg = (
        VideoClassifierConfig(
            model_variant=args.video_classifier_model_variant,
            rejection_threshold=args.video_classifier_rejection_threshold,
            batch_size=args.video_classifier_batch_size,
            fp8_enable=args.video_classifier_fp8_enable,
            max_output_tokens=args.video_classifier_max_output_tokens,
            num_gpus=video_classifier_num_gpus,
            use_mmcache=args.qwen_use_vllm_mmcache,
            sampling_fps=args.captioning_sampling_fps,
            window_size=args.captioning_window_size,
            remainder_threshold=args.captioning_remainder_threshold,
            preprocess_dtype=args.qwen_preprocess_dtype,
            model_does_preprocess=args.qwen_model_does_preprocess,
            generate_previews=args.generate_previews,
            verbose=args.verbose,
            perf_profile=args.perf_profile,
            type_allow=",".join(args.video_classifier_allow) if args.video_classifier_allow else None,
            type_block=",".join(args.video_classifier_block) if args.video_classifier_block else None,
            custom_categories=args.video_classifier_use_custom_categories,
            type_allow_file=args.video_classifier_allow_file,
            type_block_file=args.video_classifier_block_file,
            endpoint=args.video_classifier_endpoint,
            openai_model_name=args.video_classifier_openai_model_name,
            openai_max_caption_retries=args.video_classifier_openai_retries,
            openai_retry_delay_seconds=args.video_classifier_openai_retry_delay_seconds,
            gemini_model_name=args.video_classifier_gemini_model_name,
            gemini_max_caption_retries=args.video_classifier_gemini_retries,
            gemini_retry_delay_seconds=args.video_classifier_gemini_retry_delay_seconds,
        )
        if args.video_classifier
        else None
    )
    if vlm_filter_cfg is not None or video_classifier_cfg is not None:
        stages.extend(
            build_vllm_filter_classifier_stages(
                filter_config=vlm_filter_cfg,
                classifier_config=video_classifier_cfg,
            )
        )

    # --- Embedding (optional) ---
    embedding_model_version: str = "unspecified"
    if has_embeddings:
        embedding_backend: EmbeddingBackendConfig
        if args.embedding_algorithm == "openai":
            embedding_backend = OpenAIEmbeddingConfig(
                model_name=args.openai_embedding_model_name,
                max_retries=args.openai_embedding_retries,
                retry_delay_seconds=args.openai_embedding_retry_delay_seconds,
                max_concurrent_requests=args.openai_embedding_max_concurrent_requests,
            )
        elif args.embedding_algorithm.startswith("cosmos-embed1-"):
            embedding_backend = CosmosEmbed1Config(
                variant=args.embedding_algorithm.removeprefix("cosmos-embed1-"),
            )
        else:
            embedding_backend = InternVideo2Config()
        embedding_cfg = EmbeddingConfig(
            backend=embedding_backend,
            gpus_per_worker=args.embedding_gpus_per_worker,
            batch_size=args.embedding_batch_size,
            verbose=args.verbose,
            perf_profile=args.perf_profile,
        )
        embedding_model_version = get_embedding_model_version(embedding_cfg)
        logger.debug(f"Embedding algorithm={args.embedding_algorithm} version={embedding_model_version}")
        stages.extend(build_embedding_stages(embedding_cfg))

    # --- Captioning (optional) ---
    caption_algo = args.captioning_algorithm.lower()
    keep_mp4 = args.generate_previews or args.generate_cosmos_predict_dataset or caption_algo in {"gemini", "openai"}

    if args.generate_captions:
        if caption_algo not in ALL_CAPTION_ALGOS:
            msg = f"Unsupported captioning algorithm: {caption_algo}"
            raise RuntimeError(msg)

        max_tokens: int | None = args.captioning_max_output_tokens
        if max_tokens is not None and max_tokens < 0:
            max_tokens = None

        sampling_config = VllmSamplingConfig(
            temperature=args.vllm_sampling_temperature,
            top_p=args.vllm_sampling_top_p,
            top_k=args.vllm_sampling_top_k,
            repetition_penalty=args.vllm_sampling_repetition_penalty,
            presence_penalty=args.vllm_sampling_presence_penalty,
            frequency_penalty=args.vllm_sampling_frequency_penalty,
            min_p=args.vllm_sampling_min_p,
            min_tokens=args.vllm_sampling_min_tokens,
            max_tokens=max_tokens,
        )

        vllm_config = VllmConfig(
            model_variant=args.captioning_algorithm,
            prompt_variant=args.captioning_prompt_variant,
            prompt_text=args.captioning_prompt_text,
            num_cpus_for_prepare=args.vllm_prepare_num_cpus_per_worker,
            max_retries=args.vllm_max_retries,
            copy_weights_to=pathlib.Path(args.copy_weights_to) if args.copy_weights_to else None,
            sampling_config=sampling_config,
            performance_mode=args.vllm_performance_mode,
        )

        window_config = WindowConfig(
            window_size=args.captioning_window_size,
            remainder_threshold=args.captioning_remainder_threshold,
            sampling_fps=args.captioning_sampling_fps,
            preprocess_dtype=args.qwen_preprocess_dtype,
            use_input_bit_rate=args.transcode_use_input_video_bit_rate,
        )

        if caption_algo in QWEN2_CAPTION_ALGOS | QWEN3_CAPTION_ALGOS:
            vllm_config.batch_size = args.qwen_batch_size
            vllm_config.fp8 = args.qwen_use_fp8_weights  # only used for qwen (i.e. Qwen2.5-VL)
            vllm_config.disable_mmcache = not args.qwen_use_vllm_mmcache
            vllm_config.num_gpus = args.qwen_num_gpus_per_worker
            vllm_config.stage2_caption = args.qwen_stage2_caption
            vllm_config.stage2_prompt_text = args.qwen_stage2_prompt_text
            window_config.preprocess_dtype = args.qwen_preprocess_dtype
            window_config.model_does_preprocess = args.qwen_model_does_preprocess

            vllm_config.num_gpus = _clamp_num_gpus_for_qwen3_235b(caption_algo, vllm_config.num_gpus)

            if caption_algo not in QWEN2_CAPTION_ALGOS and not window_config.model_does_preprocess:
                logger.warning(
                    f"{caption_algo} does not support model_does_preprocess=False, "
                    f"setting model_does_preprocess to True"
                )
                window_config.model_does_preprocess = True

        elif caption_algo in COSMOS_REASON_ALGOS:
            vllm_config.batch_size = args.qwen_batch_size
            vllm_config.fp8 = args.qwen_use_fp8_weights
            vllm_config.disable_mmcache = not args.qwen_use_vllm_mmcache
            vllm_config.num_gpus = args.qwen_num_gpus_per_worker
            vllm_config.stage2_caption = args.qwen_stage2_caption
            vllm_config.stage2_prompt_text = args.qwen_stage2_prompt_text
            window_config.preprocess_dtype = "float16"
            window_config.model_does_preprocess = args.qwen_model_does_preprocess
            if caption_algo == "cosmos_r2":
                vllm_config.preprocess = True
                window_config.model_does_preprocess = True
        elif caption_algo in {"gemini", "openai"}:
            pass
        elif caption_algo == "vllm_async":
            # vllm_async unifies with sync by reusing VllmPrepStage. The CPU
            # prep stage is authoritative for deterministic preprocessing
            # unless the user opts in to vLLM-side preprocessing.
            window_config.model_does_preprocess = bool(args.vllm_async_preprocess)
        elif caption_algo == "nemotron":
            vllm_config.stage2_caption = args.nemotron_stage2_caption

        # Wire up debug frame saving configuration
        if args.debug_save_vllm_frames:
            vllm_config.debug_save_frames = True
            # Use output_clip_path/frames as the base directory for debug frames
            vllm_config.debug_frames_output_dir = pathlib.Path(args.output_clip_path) / "frames"
            logger.info(f"Debug frame saving enabled: output_dir={vllm_config.debug_frames_output_dir}")

        backend: CaptionBackendConfig
        if caption_algo == "gemini":
            backend = GeminiConfig(
                model_name=args.gemini_model_name,
                max_output_tokens=args.captioning_max_output_tokens,
                prompt_variant=args.captioning_prompt_variant,
                prompt_text=args.captioning_prompt_text,
                caption_retries=args.gemini_caption_retries,
                retry_delay_seconds=args.gemini_retry_delay_seconds,
                max_inline_video_bytes=int(args.gemini_max_inline_mb * 1024 * 1024),
                batch_size=args.api_caption_batch_size,
                num_cpus_for_prepare=args.vllm_prepare_num_cpus_per_worker,
            )
        elif caption_algo == "openai":
            backend = OpenAIConfig(
                model_name=args.openai_model_name,
                max_output_tokens=args.captioning_max_output_tokens,
                prompt_variant=args.captioning_prompt_variant,
                prompt_text=args.captioning_prompt_text,
                caption_retries=args.openai_caption_retries,
                retry_delay_seconds=args.openai_retry_delay_seconds,
                batch_size=args.api_caption_batch_size,
                num_cpus_for_prepare=args.vllm_prepare_num_cpus_per_worker,
            )
        elif caption_algo == "vllm_async":
            backend = VllmAsyncCaptionConfig(
                model_name=args.vllm_async_model_name,
                prompt_variant=args.captioning_prompt_variant,
                prompt_text=args.captioning_prompt_text,
                max_concurrent_requests=args.vllm_async_max_concurrent_requests,
                serve_config=build_vllm_async_config(args, sampling_config=sampling_config),
                stage_batch_size=args.vllm_async_stage_batch_size,
                num_workers_per_node=args.vllm_async_num_workers_per_node,
                stage2_caption=args.vllm_async_stage2_caption,
                stage2_prompt_text=args.vllm_async_stage2_prompt_text,
            )
        else:
            backend = vllm_config

        enhance_config: EnhanceCaptionConfig | None = None
        if args.enhance_captions:
            enhance_config = EnhanceCaptionConfig(
                model_variant=args.enhance_captions_lm_variant,
                batch_size=args.enhance_captions_batch_size,
                openai_model=args.enhance_captions_openai_model,
                fp8_enable=args.qwen_lm_use_fp8_weights,
                max_output_tokens=args.enhance_captions_max_output_tokens,
                prompt_variant=args.enhance_captions_prompt_variant,
                prompt_text=args.enhance_captions_prompt_text,
                verbose=args.verbose,
                perf_profile=args.perf_profile,
            )

        stages.extend(
            build_captioning_stages(
                CaptioningConfig(
                    backend=backend,
                    window_config=window_config,
                    keep_mp4=keep_mp4,
                    generate_previews=args.generate_previews,
                    preview_target_fps=args.preview_target_fps,
                    preview_target_height=args.preview_target_height,
                    inflight_batching=args.vllm_use_inflight_batching,
                    enhance_config=enhance_config,
                    caption_quality_flags_enabled=caption_quality_flags_enabled,
                    caption_setup_attempts=args.captioning_setup_attempts,
                    verbose=args.verbose,
                    perf_profile=args.perf_profile,
                )
            )
        )

    # Per-event captioning consumes SAM3 outputs (instances + tracked.mp4),
    # so it requires SAM3 to be explicitly enabled. We don't auto-enable
    # it here — the user should opt into the SAM3 stage knowingly because
    # of its GPU/disk cost.
    if args.event_captioning and not args.sam3:
        msg = "--event-captioning requires --sam3 to also be set"
        raise ValueError(msg)

    # --- SAM3 tracking (optional) ---
    if args.sam3:
        if not args.sam3_prompts:
            msg = "--sam3-prompts must be non-empty when --sam3 is set"
            raise ValueError(msg)
        # Event captioning needs the annotated tracked.mp4 (``#id`` overlay is
        # the VLM's only spatial grounding for object ids), so force-enable
        # annotation when it's on.
        write_annotated_video = args.sam3_write_annotated_video or args.event_captioning
        if args.event_captioning and args.sam3_annotated_video_label_style != "id":
            logger.warning(
                "--sam3-annotated-video-label-style={} is incompatible with the "
                "bundled per-event captioning prompt, which OCRs '#<id>' overlays "
                "for spatial grounding. The VLM may hallucinate object ids; pass "
                "--sam3-annotated-video-label-style id (the default) when "
                "--event-captioning is set.",
                args.sam3_annotated_video_label_style,
            )
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
                    write_annotated_video=write_annotated_video,
                    draw_trails=args.sam3_annotated_video_trails,
                    annotated_video_label_style=args.sam3_annotated_video_label_style,
                    annotated_video_mask_opacity=args.sam3_annotated_video_mask_opacity,
                    verbose=args.verbose,
                )
            )
        )

    # --- Per-event VLM captioning (optional) ---
    if args.event_captioning:
        event_vllm_async_config: VllmAsyncConfig | None = None
        if args.event_caption_backend == "vllm_async":
            # Apply the same multi-GPU floor the per-window vllm_async path
            # exposes. Mutates args in place because build_vllm_async_config
            # reads num_gpus off the namespace; harmless because the same
            # CLI args are not reused after this point.
            event_variant = args.event_caption_vllm_async_model_name
            current_num_gpus = args.event_caption_vllm_async_num_gpus or 1
            args.event_caption_vllm_async_num_gpus = _clamp_num_gpus_for_qwen3_235b(event_variant, current_num_gpus)

            # Build a sampling config from the same global --vllm-sampling-*
            # flags used by the per-window captioner so per-event vllm_async
            # picks up the user's temperature / top_p / top_k overrides.
            event_max_tokens: int | None = args.captioning_max_output_tokens
            if event_max_tokens is not None and event_max_tokens < 0:
                event_max_tokens = None
            event_sampling_config = VllmSamplingConfig(
                temperature=args.vllm_sampling_temperature,
                top_p=args.vllm_sampling_top_p,
                top_k=args.vllm_sampling_top_k,
                repetition_penalty=args.vllm_sampling_repetition_penalty,
                presence_penalty=args.vllm_sampling_presence_penalty,
                frequency_penalty=args.vllm_sampling_frequency_penalty,
                min_p=args.vllm_sampling_min_p,
                min_tokens=args.vllm_sampling_min_tokens,
                max_tokens=event_max_tokens,
            )
            event_vllm_async_config = build_vllm_async_config(
                args, sampling_config=event_sampling_config, prefix="event-caption-"
            )
        stages.append(
            PerEventCaptionStage(
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
                openai_model_name=args.event_caption_openai_model_name,
                openai_max_output_tokens=args.event_caption_openai_max_output_tokens,
                openai_max_retries=args.event_caption_openai_max_retries,
                openai_retry_delay_seconds=args.event_caption_openai_retry_delay_seconds,
                openai_endpoint_key=args.event_caption_openai_endpoint_key,
                vllm_async_config=event_vllm_async_config,
                vllm_async_sampling_fps=args.event_caption_vllm_async_sampling_fps,
                vllm_async_max_output_tokens=args.event_caption_vllm_async_max_output_tokens,
                verbose=args.verbose,
                log_stats=args.perf_profile,
            )
        )

    # --- T5 encoding (optional) ---
    if args.generate_cosmos_predict_dataset:
        stages.extend(
            build_t5_stages(
                T5Config(
                    caption_fields=[args.captioning_algorithm],
                    verbose=args.verbose,
                    perf_profile=args.perf_profile,
                )
            )
        )

    # --- Output (always) ---
    stages.extend(
        build_output_stages(
            OutputConfig(
                output_path=args.output_clip_path,
                input_path=args.input_video_path,
                output_s3_profile_name=args.output_s3_profile_name,
                upload_clips=args.upload_clips,
                upload_clip_info_in_chunks=args.upload_clip_info_in_chunks,
                upload_clip_info_in_lance=args.upload_clip_info_in_lance,
                upload_cds_parquet=args.upload_cds_parquet,
                dry_run=args.dry_run,
                generate_embeddings=has_embeddings,
                embedding_algorithm=args.embedding_algorithm,
                embedding_model_version=embedding_model_version,
                generate_previews=args.generate_previews,
                caption_models=[args.captioning_algorithm],
                enhanced_caption_models=[args.enhance_captions_lm_variant],
                caption_quality_flags_enabled=caption_quality_flags_enabled,
                generate_cosmos_predict_dataset=args.generate_cosmos_predict_dataset,
                num_workers_per_node=args.num_clip_writer_workers_per_node,
                verbose=args.verbose,
                perf_profile=args.perf_profile,
            )
        )
    )

    return stages


def split(args: argparse.Namespace) -> None:
    """Run the split pipeline with profiling and tracing.

    Public entry point that wraps ``_split()`` with ``profiling_scope``
    so that every execution path (CLI, Slurm, NVCF, local launch)
    automatically gets profiling and distributed tracing.

    Args:
        args: Command line arguments.

    """
    with profiling_scope(args):
        _split(args)


def _split(args: argparse.Namespace) -> None:
    """Run the split pipeline.

    This function orchestrates the entire pipeline, from input validation to output generation.
    It validates input arguments, builds input data, and executes the pipeline stages.

    Args:
        args: Command line arguments.

    """
    validate_stage_replay_args(args)
    assert_ffmpeg_supports_h264()

    zero_start = time.time()
    input_tasks, input_videos_relative, _, num_input_videos_selected = build_input_data(args)

    stages = _assemble_stages(args)

    pipeline_start = time.time()

    stage_save_config: StageSaveConfig | None = None
    if args.stage_save:
        task_path = get_full_path(args.output_clip_path, "tasks")
        stage_save_config = StageSaveConfig(
            path=task_path,
            stages=args.stage_save,
            sample_rate=args.stage_save_sample_rate,
            profile_name=args.output_s3_profile_name,
        )

    if len(args.stage_replay) == 0:
        if len(args.stage_compare) == 0:
            # Run the full pipeline
            output_tasks: list[SplitPipeTask] = run_pipeline(
                input_tasks,
                stages,
                args.model_weights_path,
                stage_save_config=stage_save_config,
                args=args,
            )
        else:
            start_stage_name = args.stage_compare[0]
            end_stage_name = (
                args.stage_compare[-1]
                if len(args.stage_compare) > 1
                else get_stage_name_after(stages, start_stage_name)
            )
            stage_to_compare = get_stages_to_compare(stages, start_stage_name, end_stage_name)
            golden_base = args.stage_compare_path if args.stage_compare_path is not None else args.output_clip_path
            compare_result = run_stage_compare(
                stage_to_compare,
                get_full_path(args.output_clip_path, "tasks", start_stage_name),
                get_full_path(golden_base, "tasks", end_stage_name),
                args.stage_compare_atol,
                args.limit,
                args.stage_compare_pass_threshold,
                report_path=get_full_path(args.output_clip_path, "compare", start_stage_name, "report.json"),
                profile_name=args.output_s3_profile_name,
                backend=getattr(args, "stage_compare_backend", "xenna"),
                args=args,
                model_weights_prefix=args.model_weights_path,
            )
            if not compare_result.passed:
                msg = (
                    f"Stage compare pass rate {compare_result.report.pass_rate:.3f} is below "
                    f"threshold {args.stage_compare_pass_threshold:.3f}"
                )
                raise RuntimeError(msg)
            return
    else:
        # Stage replay
        start_stage_name = args.stage_replay[0]
        end_stage_name = args.stage_replay[-1] if len(args.stage_replay) > 1 else start_stage_name
        stage_to_replay = get_stages_to_replay(stages, start_stage_name, end_stage_name)
        replay_path = get_full_path(args.output_clip_path, "tasks", start_stage_name)

        output_tasks = cast(
            "list[SplitPipeTask]",
            run_stage_replay(
                stage_to_replay,
                replay_path,
                args.limit,
                profile_name=args.output_s3_profile_name,
            ),
        )

    if args.upload_clip_info_in_lance:
        consolidate_lance_fragments(args.output_clip_path, args.output_s3_profile_name)

    summary_start = time.time()

    pipeline_run_time = (summary_start - pipeline_start) / 60
    input_build_time = (pipeline_start - zero_start) / 60

    total_video_length = write_summary(
        args,
        input_videos_relative,
        num_input_videos_selected,
        output_tasks,
        pipeline_run_time,
    )

    summary_run_time = (time.time() - summary_start) / 60

    logger.info(
        f"Split-Transcode-Filter-Annotate pipeline: {input_build_time=:.2f} / "
        f"{pipeline_run_time=:.2f} / {summary_run_time=:.2f} mins processing "
        f"time for {total_video_length=:.3f} hours of raw videos",
    )


def _setup_parser(parser: argparse.ArgumentParser) -> None:  # noqa: PLR0915
    """Set up the parser for the split pipeline.

    This function adds arguments to the parser for the split pipeline.

    Args:
        parser: The parser to add arguments to.

    """
    parser.add_argument(
        "--input-video-path",
        type=str,
        required=False,
        default=None,
        help=("S3 or local path which has input raw videos. Not required if --input-presigned-s3-url is provided."),
    )
    parser.add_argument(
        "--input-video-list-json-path",
        type=str,
        default=None,
        help="S3 or local path to a json with a list of specific videos under --input-video-path.",
    )
    parser.add_argument(
        "--input-video-list-s3-profile-name",
        type=str,
        default="default",
        help="S3 profile name to use for input_video_list_json_path.",
    )
    parser.add_argument(
        "--output-clip-path",
        type=str,
        required=False,
        default=None,
        help=(
            "S3 or local path to store output clips. "
            "If omitted and --output-presigned-s3-url is provided, a temporary directory will be used."
        ),
    )
    parser.add_argument(
        "--limit-clips",
        type=int,
        default=0,
        help="limit number of clips from each input video to process.",
    )
    parser.add_argument(
        "--no-generate-embeddings",
        dest="generate_embeddings",
        action="store_false",
        default=True,
        help="Whether to generate embeddings for clips.",
    )
    parser.add_argument(
        "--embedding-algorithm",
        type=str,
        default="internvideo2",
        choices=["cosmos-embed1-224p", "cosmos-embed1-336p", "cosmos-embed1-448p", "internvideo2", "openai"],
        help=(
            "Embedding algorithm to use. The `cosmos-embed1-*` suffix selects the input resolution "
            "(224p, 336p, or 448p): 224p is the fastest with 256-dim output vectors, while 336p and "
            "448p are slower but score higher on retrieval/classification benchmarks and produce "
            "768-dim vectors. Outputs for each variant are written to separate directories "
            "(e.g. `ce1_embd_336p/`), so switching variants against the same output path will not "
            "overwrite existing embeddings."
        ),
    )
    parser.add_argument(
        "--generate-previews",
        dest="generate_previews",
        action="store_true",
        default=False,
        help="Whether to generate previews for clip windows.",
    )
    parser.add_argument(
        "--no-generate-captions",
        dest="generate_captions",
        action="store_false",
        default=True,
        help="Whether to generate captions for clip windows.",
    )
    parser.add_argument(
        "--no-caption-quality-flags",
        dest="caption_quality_flags_enabled",
        action="store_false",
        default=True,
        help="Disable heuristic caption quality flag annotations for supported caption paths.",
    )
    parser.add_argument(
        "--no-upload-clips",
        dest="upload_clips",
        action="store_false",
        default=True,
        help="Whether to upload clips to output path.",
    )
    parser.add_argument(
        "--upload-clip-info-in-lance",
        action="store_true",
        default=False,
        help="Whether to also stage clip metadata/embeddings into Lance and consolidate at the end.",
    )
    parser.add_argument(
        "--upload-clip-info-in-chunks",
        dest="upload_clip_info_in_chunks",
        action="store_true",
        default=False,
        help=(
            "Whether to group clip metadata in chunks as jsonl and "
            "skip writing per-clip embedding pickles, i.e. grouped clip embeddings as parquet only."
        ),
    )
    parser.add_argument(
        "--upload-cds-parquet",
        dest="upload_cds_parquet",
        action="store_true",
        default=False,
        help="Whether to upload parquet files for CDS.",
    )
    parser.add_argument(
        "--generate-cosmos-predict-dataset",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Generate Cosmos-Predict2 post-training dataset.",
    )
    parser.add_argument(
        "--no-write-all-caption-json",
        dest="write_all_caption_json",
        action="store_false",
        default=True,
        help="Whether to write all captions to a single JSON file in the output path.",
    )
    # --- SAM3 tracking + per-event VLM captioning (optional) ---
    add_sam3_args(parser)
    add_event_caption_args(parser)
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
        "--transnetv2-frame-decoder-mode",
        choices=["ffmpeg_cpu", "pynvc"],
        default="ffmpeg_cpu",
        help="Choose between ffmpeg on CPU or PyNvVideoCodec for video decode.",
    )
    parser.add_argument(
        "--transnetv2-frame-decode-cpus-per-worker",
        type=float,
        default=3.0,
        help="Number of CPU threads per worker for video frame decoding when using ffmpeg_cpu mode.",
    )
    parser.add_argument(
        "--transnetv2-frame-decode-raise-on-pynvc-error",
        dest="transnetv2_frame_decode_raise_on_pynvc_error",
        action="store_true",
        default=False,
        help="Disable CPU ffmpeg fallback from PyNvVideoCodec and raise exception (for testing).",
    )
    parser.add_argument(
        "--transnetv2-threshold",
        type=float,
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
        type=int,
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
        "--transnetv2-gpus-per-worker",
        type=float,
        default=0.25,
        help="Number of GPUs per worker for TransNetV2 splitting stage.",
    )
    parser.add_argument(
        "--transcode-cpus-per-worker",
        type=float,
        default=5.0,
        help="Number of CPU threads per worker. The stage uses a batched ffmpeg "
        "commandline with batch_size (--transcode-ffmpeg-batch-size) of ~64 and per-batch thread count of 1.",
    )
    parser.add_argument(
        "--transcode-encoder",
        type=str,
        default="libopenh264",
        choices=["libopenh264", "h264_nvenc"],
        help="Codec for transcoding clips; None to skip transocding.",
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
        help="FFMPEG batchsize for transcoding clips. Each clip/sub-command in "
        "the batch uses --transcode-encoder-threads number of CPU threads",
    )
    parser.add_argument(
        "--transcode-use-hwaccel",
        action="store_true",
        default=False,
        help="Whether to use cuda acceleration for decoding in transcoding stage.",
    )
    parser.add_argument(
        "--transcode-use-input-video-bit-rate",
        action="store_true",
        default=False,
        help="Whether to use input video's bit rate for encoding clips.",
    )
    parser.add_argument(
        "--transcode-max-output-frames",
        type=int,
        default=None,
        help="If set, limit each transcoded clip's frame count by reducing FPS. "
        "For example, 186 for Cosmos Transfer compatibility. Source FPS is never increased.",
    )
    parser.add_argument(
        "--clip-re-chunk-size",
        type=int,
        default=32,
        help="Number of clips per chunk after transcoding stage.",
    )
    # --- Super-resolution args ---
    parser.add_argument(
        "--super-resolution",
        action="store_true",
        default=False,
        help="Enable SeedVR2 video super-resolution on clips after transcoding.",
    )
    parser.add_argument(
        "--sr-variant",
        type=str,
        default="seedvr2_7b",
        choices=["seedvr2_3b", "seedvr2_7b", "seedvr2_7b_sharp"],
        help="SeedVR2 model variant.",
    )
    parser.add_argument("--sr-target-height", type=int, default=720, help="Target output height for SR.")
    parser.add_argument("--sr-target-width", type=int, default=1280, help="Target output width for SR.")
    parser.add_argument("--sr-window-frames", type=int, default=128, help="Frames per SR inference window.")
    parser.add_argument("--sr-overlap-frames", type=int, default=64, help="Overlap frames between SR windows.")
    parser.add_argument(
        "--sr-no-blend-overlap",
        action="store_true",
        default=False,
        help="Disable overlap blending (drop overlap frames from the later window instead).",
    )
    parser.add_argument("--sr-seed", type=int, default=666, help="Random seed for SR diffusion.")
    parser.add_argument("--sr-cfg-scale", type=float, default=1.0, help="Classifier-free guidance scale for SR.")
    parser.add_argument("--sr-cfg-rescale", type=float, default=0.0, help="CFG rescale factor for SR.")
    parser.add_argument("--sr-sample-steps", type=int, default=1, help="Number of diffusion sampling steps for SR.")
    parser.add_argument("--sr-sp-size", type=int, default=1, help="Sequence parallelism size for SR.")
    parser.add_argument(
        "--sr-out-fps", type=float, default=None, help="Output FPS for SR (None = preserve source FPS)."
    )
    parser.add_argument("--sr-tmp-dir", type=str, default=None, help="Temp directory for SR window segment files.")

    parser.add_argument(
        "--motion-filter",
        choices=["disable", "enable", "score-only"],
        default="disable",
        help=(
            "Control motion filtering behavior:\n"
            "  - disable: No filtering or scoring.\n"
            "  - enable: Automatically filter clips based on motion thresholds.\n"
            "      (controlled by --motion-global-mean-threshold and --motion-per-patch-min-256-threshold).\n"
            "  - score-only: Calculate motion scores without filtering clips."
        ),
    )
    parser.add_argument(
        "--motion-global-mean-threshold",
        type=float,
        default=0.00098,
        help=(
            "Threshold for global average motion magnitude. "
            "Clips with global motion below this value may be flagged as low-motion. "
            "Only applies when --motion-filter is set to 'enable' or 'score-only'."
        ),
    )
    parser.add_argument(
        "--motion-per-patch-min-256-threshold",
        type=float,
        default=0.000001,
        help=(
            "Threshold for minimal average motion magnitude in any 256x256-pixel patch. "
            "Clips containing patches below this threshold may be flagged as low-motion. "
            "Only applies when --motion-filter is set to 'enable' or 'score-only'."
        ),
    )
    parser.add_argument(
        "--motion-decode-target-fps",
        type=float,
        default=2.0,
        help="Target frames per second to sample for motion vector decoding.",
    )
    parser.add_argument(
        "--motion-decode-target-duration-ratio",
        type=float,
        default=0.5,
        help="Target ratio of video duration to sample for motion vector decoding (0.5 = 50%%).",
    )
    parser.add_argument(
        "--motion-decode-cpus-per-worker",
        type=float,
        default=2.0,
        help="Number of CPUs per worker allocated to motion vector decoding.",
    )
    parser.add_argument(
        "--motion-score-batch-size",
        type=int,
        default=64,
        help="Batch size for motion score computation.",
    )
    parser.add_argument(
        "--motion-score-gpus-per-worker",
        type=float,
        default=0.5,
        help="Number of GPUs per worker allocated to motion score computation. Set to 0 to use CPU instead of GPU.",
    )
    parser.add_argument(
        "--clip-extraction-target-res",
        type=int,
        default=-1,
        help="Target resolution for clip extraction as (height, width). A value of -1 implies disables resize",
    )
    parser.add_argument(
        "--clip-extraction-cpus-per-worker",
        type=float,
        default=3.0,
        help="Number of CPUs per worker allocated to clip frame extraction.",
    )
    parser.add_argument(
        "--clip-extraction-decoder-mode",
        choices=["extract_frames", "camera_sensor"],
        default="extract_frames",
        help="Decoder mode for clip frame extraction.",
    )
    parser.add_argument(
        "--aesthetic-threshold",
        type=float,
        default=None,
        help="If specified (e.g. 3.5), filter out clips with an aesthetic score below this threshold.",
    )
    parser.add_argument(
        "--aesthetic-reduction",
        choices=[
            "mean",
            "min",
        ],
        default="min",
        help="Method to reduce the frame-level aesthetic scores.",
    )
    parser.add_argument(
        "--aesthetic-gpus-per-worker",
        type=float,
        default=0.25,
        help="Number of GPUs per worker allocated to aesthetic filter.",
    )
    parser.add_argument(
        "--artificial-text-filter",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Filter clips that contain overlay/artificial text (e.g. captions, logos).",
    )
    parser.add_argument(
        "--artificial-text-frame-interval",
        type=int,
        default=3,
        help="Sample every N frames for artificial text detection (with --artificial-text-filter).",
    )
    parser.add_argument(
        "--artificial-text-gpus-per-worker",
        type=float,
        default=0.25,
        help="GPUs per worker for artificial text filter (with --artificial-text-filter).",
    )
    parser.add_argument(
        "--no-artificial-text-corner-detection",
        dest="artificial_text_use_corner_detection",
        action="store_false",
        default=True,
        help="Ignore corner text (e.g. logos); only detect stable overlay text (with --artificial-text-filter).",
    )
    parser.add_argument(
        "--ignore-artificial-text-corner-region",
        dest="artificial_text_ignore_corner_region",
        action="store_true",
        default=False,
        help="Drop detections in corner zones; only center/non-corner text can filter (with --artificial-text-filter).",
    )
    parser.add_argument(
        "--artificial-text-corner-x-margin",
        type=float,
        default=0.1,
        help="Normalized margin from left/right (0-1) for corner zone; default 0.1 (with --artificial-text-filter).",
    )
    parser.add_argument(
        "--artificial-text-corner-y-margin",
        type=float,
        default=0.1,
        help="Normalized margin from top/bottom (0-1) for corner zone; default 0.1 (with --artificial-text-filter).",
    )
    parser.add_argument(
        "--artificial-text-min-duration-frames",
        type=int,
        default=10,
        help="Min frames text must appear to count as overlay (with --artificial-text-filter).",
    )
    parser.add_argument(
        "--artificial-text-corner-ratio",
        type=float,
        default=0.1,
        help="Min fraction of frames with corner text (e.g. logo) to flag (with --artificial-text-filter).",
    )
    parser.add_argument(
        "--artificial-text-stability-iou-threshold",
        type=float,
        default=0.9,
        help="Stability threshold (0-1) for overlay; higher = only very fixed text (with --artificial-text-filter).",
    )
    parser.add_argument(
        "--artificial-text-detection-use-cpu",
        dest="artificial_text_detection_use_cpu",
        action="store_true",
        default=False,
        help=(
            "Run artificial text detection on CPU (unified env) instead of GPU (paddle-ocr env).\n"
            "Use this when the paddle-ocr environment is not installed (e.g. the main/unified image)."
        ),
    )
    parser.add_argument(
        "--vlm-filter",
        dest="vlm_filter",
        choices=["enable", "disable", "score-only"],
        default="disable",
        help=(
            "Whether to enable VLM-based content filtering for video clips.\n"
            "  - enable: Automatically filter clips based on VLM-based content filtering.\n"
            "  - disable: Disable VLM-based content filtering.\n"
            "  - score-only: Calculate VLM-based content filtering results without filtering clips."
        ),
    )
    parser.add_argument(
        "--vlm-filter-prompt-variant",
        dest="vlm_filter_prompt_variant",
        type=str,
        default="default",
        choices=[
            "default",
        ],
        help="Prompt variant for VLM semantic filtering stage.",
    )
    parser.add_argument(
        "--vlm-filter-categories",
        dest="vlm_filter_categories",
        type=str,
        default=None,
        help=(
            "Comma-separated list of categories to filter out (semantic mode). "
            "If not provided, default categories will be used."
        ),
    )
    parser.add_argument(
        "--vlm-filter-rejection-threshold",
        dest="vlm_filter_rejection_threshold",
        type=float,
        default=0.5,
        help="Threshold for VLM filtering stage. If not provided, the default threshold of .5 will be used.",
    )
    parser.add_argument(
        "--vlm-filter-batch-size",
        dest="vlm_filter_batch_size",
        type=int,
        default=16,
        help="Batch size for VLM filtering stage.",
    )
    parser.add_argument(
        "--vlm-filter-model-variant",
        dest="vlm_filter_model_variant",
        type=str,
        default="qwen",
        help="Model variant to use for VLM filtering.",
    )
    parser.add_argument(
        "--vlm-filter-fp8-enable",
        dest="vlm_filter_fp8_enable",
        action="store_true",
        default=False,
        help="Whether to use FP8 weights for VLM filtering model.",
    )
    parser.add_argument(
        "--vlm-filter-max-output-tokens",
        dest="vlm_filter_max_output_tokens",
        type=int,
        default=8192,
        help="Max number of output tokens for VLM filtering model.",
    )
    parser.add_argument(
        "--vlm-filter-num-gpus",
        dest="vlm_filter_num_gpus",
        type=int,
        default=1,
        help="Number of GPUs per worker for VLM filtering model.",
    )
    parser.add_argument(
        "--vlm-filter-endpoint",
        dest="vlm_filter_endpoint",
        choices=["local", "openai", "gemini"],
        default="local",
        help="Inference backend for VLM filtering. 'local' runs vLLM in-process; 'openai' calls an external "
        "OpenAI-compatible endpoint configured under openai.filter in ~/.config/cosmos_curator/config.yaml; "
        "'gemini' calls the Google Gemini API configured under gemini in the config.",
    )
    parser.add_argument(
        "--vlm-filter-openai-model-name",
        dest="vlm_filter_openai_model_name",
        type=str,
        default="auto",
        help="Model name for the OpenAI-compatible filter endpoint. Use 'auto' to query /v1/models.",
    )
    parser.add_argument(
        "--vlm-filter-openai-retries",
        dest="vlm_filter_openai_retries",
        type=int,
        default=3,
        help="Max retries per window for the OpenAI-compatible filter endpoint.",
    )
    parser.add_argument(
        "--vlm-filter-openai-retry-delay-seconds",
        dest="vlm_filter_openai_retry_delay_seconds",
        type=float,
        default=1.0,
        help="Delay in seconds between retries for the OpenAI-compatible filter endpoint.",
    )
    parser.add_argument(
        "--vlm-filter-gemini-model-name",
        dest="vlm_filter_gemini_model_name",
        type=str,
        default="models/gemini-2.5-pro",
        help="Gemini model name for the filter endpoint.",
    )
    parser.add_argument(
        "--vlm-filter-gemini-retries",
        dest="vlm_filter_gemini_retries",
        type=int,
        default=3,
        help="Max retries per window for the Gemini filter endpoint.",
    )
    parser.add_argument(
        "--vlm-filter-gemini-retry-delay-seconds",
        dest="vlm_filter_gemini_retry_delay_seconds",
        type=float,
        default=1.0,
        help="Delay in seconds between retries for the Gemini filter endpoint.",
    )
    parser.add_argument(
        "--video-classifier",
        dest="video_classifier",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable VLM-based video classifier; filter by type allow/block lists. "
            "Set --video-classifier-allow and/or --video-classifier-block. "
            "With --video-classifier-use-custom-categories, allow/block define the full category set."
        ),
    )
    parser.add_argument(
        "--video-classifier-rejection-threshold",
        dest="video_classifier_rejection_threshold",
        type=float,
        default=0.5,
        help="Threshold for VLM filtering stage. If not provided, the default threshold of .5 will be used.",
    )
    parser.add_argument(
        "--video-classifier-use-custom-categories",
        dest="video_classifier_use_custom_categories",
        action="store_true",
        default=False,
        help=(
            "Use custom categories: allow and block lists define the full set of categories. "
            "The model is prompted only for those. Requires at least one of allow/block."
        ),
    )
    parser.add_argument(
        "--video-classifier-allow",
        dest="video_classifier_allow",
        type=str,
        action="append",
        default=None,
        metavar="TYPE",
        help=(
            "Video type(s) to keep (only clips with at least one window matching any pass). "
            "Default: 27 imaginaire taxonomy labels (underscores, no spaces). "
            "With --video-classifier-use-custom-categories, any names; union with block defines categories."
        ),
    )
    parser.add_argument(
        "--video-classifier-block",
        dest="video_classifier_block",
        type=str,
        action="append",
        default=None,
        metavar="TYPE",
        help=(
            "Video type(s) to reject (clips with too many windows matching any are filtered out). "
            "With --video-classifier-use-custom-categories, any names; union with allow defines categories."
        ),
    )
    parser.add_argument(
        "--video-classifier-allow-file",
        dest="video_classifier_allow_file",
        type=str,
        default=None,
        help=(
            "Path to a newline-separated .txt file of categories to allow. "
            "Replaces the default category list; clips matching any allow category are kept."
        ),
    )
    parser.add_argument(
        "--video-classifier-block-file",
        dest="video_classifier_block_file",
        type=str,
        default=None,
        help=(
            "Path to a newline-separated .txt file of categories to block. "
            "Replaces the default category list; clips matching any block category are rejected."
        ),
    )
    parser.add_argument(
        "--video-classifier-batch-size",
        dest="video_classifier_batch_size",
        type=int,
        default=16,
        help="Batch size for video classifier stage.",
    )
    parser.add_argument(
        "--video-classifier-model-variant",
        dest="video_classifier_model_variant",
        type=str,
        default="qwen",
        help="Model variant for video classifier stage.",
    )
    parser.add_argument(
        "--video-classifier-fp8-enable",
        dest="video_classifier_fp8_enable",
        action="store_true",
        default=False,
        help="Whether to use FP8 weights for video classifier model.",
    )
    parser.add_argument(
        "--video-classifier-max-output-tokens",
        dest="video_classifier_max_output_tokens",
        type=int,
        default=8192,
        help="Max number of output tokens for video classifier model.",
    )
    parser.add_argument(
        "--video-classifier-num-gpus",
        dest="video_classifier_num_gpus",
        type=int,
        default=1,
        help="Number of GPUs per worker for video classifier model.",
    )
    parser.add_argument(
        "--video-classifier-endpoint",
        dest="video_classifier_endpoint",
        choices=["local", "openai", "gemini"],
        default="local",
        help="Inference backend for video classifier. 'local' runs vLLM in-process; 'openai' calls an external "
        "OpenAI-compatible endpoint configured under openai.classifier in ~/.config/cosmos_curator/config.yaml; "
        "'gemini' calls the Google Gemini API configured under gemini in the config.",
    )
    parser.add_argument(
        "--video-classifier-openai-model-name",
        dest="video_classifier_openai_model_name",
        type=str,
        default="auto",
        help="Model name for the OpenAI-compatible classifier endpoint. Use 'auto' to query /v1/models.",
    )
    parser.add_argument(
        "--video-classifier-openai-retries",
        dest="video_classifier_openai_retries",
        type=int,
        default=3,
        help="Max retries per window for the OpenAI-compatible classifier endpoint.",
    )
    parser.add_argument(
        "--video-classifier-openai-retry-delay-seconds",
        dest="video_classifier_openai_retry_delay_seconds",
        type=float,
        default=1.0,
        help="Delay in seconds between retries for the OpenAI-compatible classifier endpoint.",
    )
    parser.add_argument(
        "--video-classifier-gemini-model-name",
        dest="video_classifier_gemini_model_name",
        type=str,
        default="models/gemini-2.5-pro",
        help="Gemini model name for the classifier endpoint.",
    )
    parser.add_argument(
        "--video-classifier-gemini-retries",
        dest="video_classifier_gemini_retries",
        type=int,
        default=3,
        help="Max retries per window for the Gemini classifier endpoint.",
    )
    parser.add_argument(
        "--video-classifier-gemini-retry-delay-seconds",
        dest="video_classifier_gemini_retry_delay_seconds",
        type=float,
        default=1.0,
        help="Delay in seconds between retries for the Gemini classifier endpoint.",
    )
    parser.add_argument(
        "--embedding-gpus-per-worker",
        type=float,
        default=0.25,
        help="Number of GPUs per worker for InternVideo2 or Cosmos-Embed1 embedding stage.",
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=8,
        help="Batch size for InternVideo2 embedding stage.",
    )
    parser.add_argument(
        "--captioning-algorithm",
        type=str,
        default="qwen",
        choices=sorted(ALL_CAPTION_ALGOS),
        help="Captioning algorithm to use in annotation pipeline.",
    )
    parser.add_argument(
        "--captioning-window-size",
        type=int,
        default=256,
        help="Window size for captioning algorithm.",
    )
    parser.add_argument(
        "--captioning-remainder-threshold",
        type=int,
        default=128,
        help="Remainder threshold for captioning algorithm.",
    )
    parser.add_argument(
        "--captioning-prompt-variant",
        type=str,
        default="default",
        choices=[
            "default",
            "av",
            "av-surveillance",
        ],
        help="Prompt variant for captioning algorithm.",
    )
    parser.add_argument(
        "--captioning-prompt-text",
        type=str,
        default=None,
        help="Prompt text for captioning algorithm.",
    )
    parser.add_argument(
        "--captioning-sampling-fps",
        type=float,
        default=2.0,
        help="Controls number of frames sampled per second from input clip for captioning model",
    )
    parser.add_argument(
        "--captioning-max-output-tokens",
        type=int,
        default=8192,
        help="Max number of output tokens requested from captioning model",
    )
    parser.add_argument(
        "--captioning-setup-attempts",
        type=int,
        default=1,
        help=(
            "Number of times the vLLM caption stage's setup() may be retried before the actor "
            "pool gives up on a worker. Each retry re-spawns the actor (Ray reschedules), which "
            "can dodge transient placement issues like a leaked CUDA context squatting on the "
            "assigned GPU. Only the vLLM caption backend honors this; other backends ignore it."
        ),
    )
    parser.add_argument(
        "--gemini-model-name",
        type=str,
        default="models/gemini-2.5-pro",
        help="Gemini model name used when --captioning-algorithm is 'gemini'.",
    )
    parser.add_argument(
        "--gemini-caption-retries",
        type=int,
        default=3,
        help="Max number of retries for Gemini caption requests.",
    )
    parser.add_argument(
        "--gemini-retry-delay-seconds",
        type=float,
        default=1.0,
        help="Delay between retries for Gemini caption requests.",
    )
    parser.add_argument(
        "--gemini-max-inline-mb",
        type=float,
        default=20.0,
        help="Maximum inline video size accepted by Gemini when captioning (in megabytes).",
    )
    parser.add_argument(
        "--openai-model-name",
        type=str,
        default="auto",
        help="Model name to use with the OpenAI-compatible caption API ('auto' queries /v1/models).",
    )
    parser.add_argument(
        "--openai-caption-retries",
        type=int,
        default=3,
        help="Max number of retries for OpenAI API caption requests.",
    )
    parser.add_argument(
        "--openai-retry-delay-seconds",
        type=float,
        default=1.0,
        help="Delay between retries for OpenAI API caption requests.",
    )
    # --- vllm_async args (VllmAsyncCaptionStage in-process AsyncLLM) ---
    add_vllm_async_cli_args(parser)
    parser.add_argument(
        "--openai-embedding-model-name",
        type=str,
        default="auto",
        help="Model name to use with the OpenAI-compatible embedding API ('auto' queries /v1/models).",
    )
    parser.add_argument(
        "--openai-embedding-retries",
        type=int,
        default=3,
        help="Max number of retries for OpenAI API embedding requests.",
    )
    parser.add_argument(
        "--openai-embedding-retry-delay-seconds",
        type=float,
        default=1.0,
        help="Delay between retries for OpenAI API embedding requests.",
    )
    parser.add_argument(
        "--openai-embedding-max-concurrent-requests",
        type=int,
        default=8,
        help="Max concurrent requests to the OpenAI-compatible embedding endpoint.",
    )
    parser.add_argument(
        "--qwen-preprocess-dtype",
        type=str,
        default="float16",
        choices=[
            "float32",
            "float16",
            "bfloat16",
            "uint8",
        ],
        help="Precision for tensor preprocess operations in QwenInputPreparationStage.",
    )
    parser.add_argument(
        "--qwen-model-does-preprocess",
        dest="qwen_model_does_preprocess",
        action="store_true",
        default=False,
        help="If set, Qwen will handle preprocessing (resize, rescale, normalize) instead of our code.",
    )
    parser.add_argument(
        "--qwen-stage2-caption",
        dest="qwen_stage2_caption",
        action="store_true",
        default=False,
        help="If set, generated captions are used as input prompts again into QwenVL to refine them",
    )
    parser.add_argument(
        "--qwen-stage2-prompt-text",
        type=str,
        default=None,
        help="Specify the input prompt used to generate stage2 Qwen captions",
    )
    parser.add_argument(
        "--qwen-batch-size",
        type=int,
        default=8,
        help="Batch size for Qwen captioning stage.",
    )
    parser.add_argument(
        "--api-caption-batch-size",
        type=int,
        default=8,
        help="Batch size / async concurrency limit for OpenAI and Gemini captioning stages.",
    )
    parser.add_argument(
        "--qwen-use-vllm-mmcache",
        action="store_true",
        default=False,
        help="vLLM MultiModal Cache Usage, default disabled for better performance and GPU Utilization",
    )
    parser.add_argument(
        "--qwen-use-fp8-weights",
        action="store_true",
        default=False,
        help="Whether to use fp8 weights for Qwen VL model or not.",
    )
    parser.add_argument(
        "--qwen-num-gpus-per-worker",
        type=int,
        default=1,
        help="Number of GPUs per worker for Qwen captioning stage.",
    )
    parser.add_argument(
        "--vllm-prepare-num-cpus-per-worker",
        type=float,
        default=3.0,
        help="Number of CPUs per worker for VllmPrepStage.",
    )
    parser.add_argument(
        "--vllm-performance-mode",
        type=str,
        default="throughput",
        choices=["balanced", "interactivity", "throughput"],
        help=(
            "vLLM performance mode. 'throughput' (default) favors aggregate tokens/sec with "
            "larger CUDA graphs and more aggressive batching. 'interactivity' favors low "
            "per-request latency. 'balanced' is the vLLM default."
        ),
    )
    parser.add_argument(
        "--vllm-use-inflight-batching",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to use inflight batching for vLLM captioning stage.",
    )
    parser.add_argument(
        "--vllm-max-retries",
        type=int,
        default=3,
        help="Number of times to retry vLLM captioning failures",
    )
    parser.add_argument(
        "--copy-weights-to",
        type=str,
        default=None,
        help="Optional directory to copy model weights to before loading. "
        "Useful for copying weights to faster storage, like local NVME on compute nodes, "
        "and can reduce model load time. Common location is /raid/scratch/models.",
    )
    parser.add_argument(
        "--enhance-captions",
        dest="enhance_captions",
        action="store_true",
        default=False,
        help="Whether to further enhance captions with a language model",
    )
    parser.add_argument(
        "--enhance-captions-lm-variant",
        type=str,
        default="qwen_lm",
        choices=["qwen_lm", "gpt_oss_20b", "openai"],
        help="Select language model for enhance captions stage.",
    )
    parser.add_argument(
        "--enhance-captions-openai-model",
        type=str,
        default="auto",
        help="OpenAI model name for caption enhancement ('auto' queries /v1/models).",
    )
    parser.add_argument(
        "--enhance-captions-prompt-variant",
        type=str,
        default="default",
        choices=[
            "default",
            "av",
            "av-surveillance",
        ],
        help="Prompt variant for enhanced captioning algorithm.",
    )
    parser.add_argument(
        "--enhance-captions-prompt-text",
        type=str,
        default=None,
        help="Prompt text for further enhancing captions using EnhanceCaptionStage.",
    )
    parser.add_argument(
        "--enhance-captions-max-output-tokens",
        type=int,
        default=2048,
        help="Max number of output tokens requested from the enhance captions model.",
    )
    parser.add_argument(
        "--enhance-captions-batch-size",
        type=int,
        default=32,
        help="Batch size for enhance captioning stage.",
    )
    parser.add_argument(
        "--qwen-lm-use-fp8-weights",
        action="store_true",
        default=False,
        help="Whether to use fp8 weights for Qwen-LM model or not.",
    )
    parser.add_argument(
        "--preview-target-fps",
        type=int,
        default=1,
        help="Target FPS for preview generation.",
    )
    parser.add_argument(
        "--preview-target-height",
        type=int,
        default=240,
        help="Target height for preview generation.",
    )
    parser.add_argument(
        "--num-download-workers-per-node",
        type=int,
        default=4,
        help="Number of workers to use for downloading videos.",
    )
    parser.add_argument(
        "--num-clip-writer-workers-per-node",
        type=int,
        default=8,
        help="Number of workers to use for writing clips.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="If set only write minimum metadata",
    )
    parser.add_argument(
        "--input-presigned-s3-url",
        type=str,
        default=None,
        help="Presigned S3 URL pointing to a zip archive that contains the input videos.",
    )
    parser.add_argument(
        "--output-presigned-s3-url",
        type=str,
        default=None,
        help="Presigned S3 URL where the zipped output clips will be uploaded.",
    )
    parser.add_argument(
        "--nemotron-stage2-caption",
        action="store_true",
        default=False,
        help="If set, generated captions are used as input prompts again into Nemotron to refine them",
    )
    # vLLM sampling parameters - get defaults from VllmSamplingConfig
    sampling_defaults = _get_vllm_sampling_defaults()
    parser.add_argument(
        "--vllm-sampling-temperature",
        type=float,
        default=sampling_defaults["temperature"],
        help="Temperature for vLLM sampling (higher = more random).",
    )
    parser.add_argument(
        "--vllm-sampling-top-p",
        type=float,
        default=sampling_defaults["top_p"],
        help="Top-p (nucleus sampling) parameter for vLLM.",
    )
    parser.add_argument(
        "--vllm-sampling-top-k",
        type=int,
        default=sampling_defaults["top_k"],
        help="Top-k sampling parameter for vLLM (0 = disabled).",
    )
    parser.add_argument(
        "--vllm-sampling-repetition-penalty",
        type=float,
        default=sampling_defaults["repetition_penalty"],
        help="Repetition penalty for vLLM sampling.",
    )
    parser.add_argument(
        "--vllm-sampling-presence-penalty",
        type=float,
        default=sampling_defaults["presence_penalty"],
        help="Presence penalty for vLLM sampling.",
    )
    parser.add_argument(
        "--vllm-sampling-frequency-penalty",
        type=float,
        default=sampling_defaults["frequency_penalty"],
        help="Frequency penalty for vLLM sampling.",
    )
    parser.add_argument(
        "--vllm-sampling-min-p",
        type=float,
        default=sampling_defaults["min_p"],
        help="Minimum probability threshold for vLLM sampling.",
    )
    parser.add_argument(
        "--vllm-sampling-min-tokens",
        type=int,
        default=sampling_defaults["min_tokens"],
        help=(
            "Minimum tokens to generate before EOS/stop tokens are allowed (0 = disabled). "
            "Prevents empty captions caused by fp8 quantization shifting EOS logits above "
            "content tokens. Harmless for bf16/fp16 models. Default: 16."
        ),
    )
    # Debug arguments for saving vLLM input frames
    parser.add_argument(
        "--debug-save-vllm-frames",
        dest="debug_save_vllm_frames",
        action="store_true",
        default=False,
        help=(
            "Save video frames passed to vLLM as PNGs for debugging. "
            "Frames will be saved to {output-clip-path}/frames/{clip_uuid}/"
        ),
    )
    parser.add_argument(
        "--multi-cam",
        action="store_true",
        default=False,
        help="Use session-based multi-camera input; primary camera at slot 0.",
    )
    parser.add_argument(
        "--primary-camera-keyword",
        type=str,
        default="front",
        help="String to identify the primary camera in session discovery; the matching video is placed at slot 0.",
    )
    # add common args applicable to all pipelines
    add_common_args(parser)
    add_stage_replay_args(parser)


def nvcf_run_split(args: argparse.Namespace) -> None:
    """Run the split pipeline.

    This function orchestrates the entire pipeline, from input validation to output generation.
    It validates input arguments, builds input data, and executes the pipeline stages.

    Args:
        args: Command line arguments.

    """
    args_utils.fill_default_args(args, _setup_parser)
    cli_run_split(args)


def cli_run_split(args: argparse.Namespace) -> None:
    """Run the split pipeline.

    This function orchestrates the entire pipeline, from input validation to output generation.
    It validates input arguments, builds input data, and executes the pipeline stages.

    Args:
        args: Command line arguments.

    """
    split(args)


def add_split_command(
    subparsers: argparse._SubParsersAction,  # type: ignore[type-arg]
) -> argparse.ArgumentParser:
    """Add the split command to the parser.

    This function adds a subparser for the split command to the main parser.
    It sets up the parser with the appropriate arguments and default values.

    Args:
        subparsers: The subparsers action to add the parser to.

    """
    parser = subparsers.add_parser(
        "split",
        help="Split videos into clips.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.set_defaults(func=cli_run_split)
    _setup_parser(parser)
    return parser  # type: ignore[no-any-return]
