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
"""Stage builders for splitting, transcoding, and frame extraction."""

from typing import Literal

import attrs

from cosmos_curator.core.interfaces.stage_interface import CuratorStage, CuratorStageSpec
from cosmos_curator.pipelines.video.clipping.clip_extraction_stages import (
    ClipTranscodingStage,
    FixedStrideExtractorStage,
)
from cosmos_curator.pipelines.video.clipping.clip_frame_extraction_stages import ClipFrameExtractionStage
from cosmos_curator.pipelines.video.clipping.frame_extraction_stages import VideoFrameExtractionStage
from cosmos_curator.pipelines.video.clipping.transnetv2_extraction_stages import TransNetV2ClipExtractionStage
from cosmos_curator.pipelines.video.utils.decoder_utils import FrameExtractionPolicy


@attrs.define(frozen=True)
class TransNetV2SplitConfig:
    """Configuration for TransNetV2-based scene splitting."""

    threshold: float = 0.4
    min_length_s: float = 2.0
    min_length_frames: int = 48
    max_length_s: float = 60.0
    max_length_mode: Literal["truncate", "stride"] = "stride"
    crop_s: float = 0.5
    num_gpus_per_worker: float = 0.25
    decoder_mode: str = "ffmpeg_cpu"
    num_decode_cpus_per_worker: float = 3.0
    raise_on_pynvc_error: bool = False
    limit_clips: int = 0
    verbose: bool = False
    perf_profile: bool = False


@attrs.define(frozen=True)
class FixedStrideSplitConfig:
    """Configuration for fixed-stride clip splitting."""

    clip_len_s: int = 10
    clip_stride_s: int = 10
    min_clip_length_s: float = 2.0
    limit_clips: int = 0
    verbose: bool = False
    perf_profile: bool = False


@attrs.define(frozen=True)
class TranscodeConfig:
    """Configuration for clip transcoding."""

    num_cpus_per_worker: float = 5.0
    encoder: str = "libopenh264"
    encoder_threads: int = 1
    encode_batch_size: int = 16
    use_hwaccel: bool = False
    use_input_bit_rate: bool = False
    num_clips_per_chunk: int = 32
    max_output_frames: int | None = None
    verbose: bool = False
    perf_profile: bool = False


@attrs.define(frozen=True)
class FrameExtractionConfig:
    """Configuration for shared per-clip frame extraction (used by aesthetics and embedding)."""

    target_fps: list[float | int]
    target_res: int = -1
    cpus_per_worker: float = 3.0
    perf_profile: bool = False


def build_transnetv2_split_stages(config: TransNetV2SplitConfig) -> list[CuratorStage | CuratorStageSpec]:
    """Construct and return the frame extraction and scene detection stages."""
    return [
        CuratorStageSpec(
            VideoFrameExtractionStage(
                decoder_mode=config.decoder_mode,
                num_cpus_per_worker=config.num_decode_cpus_per_worker,
                raise_on_pynvc_error_without_cpu_fallback=config.raise_on_pynvc_error,
                verbose=config.verbose,
                log_stats=config.perf_profile,
            ),
        ),
        CuratorStageSpec(
            TransNetV2ClipExtractionStage(
                threshold=config.threshold,
                min_length_s=config.min_length_s,
                min_length_frames=config.min_length_frames,
                max_length_s=config.max_length_s,
                max_length_mode=config.max_length_mode,
                crop_s=config.crop_s,
                num_gpus_per_worker=config.num_gpus_per_worker,
                limit_clips=config.limit_clips,
                verbose=config.verbose,
                log_stats=config.perf_profile,
            ),
            over_provision_factor=2.0,
        ),
    ]


def build_fixed_stride_split_stages(config: FixedStrideSplitConfig) -> list[CuratorStage | CuratorStageSpec]:
    """Construct and return the fixed-stride extractor stage."""
    return [
        CuratorStageSpec(
            FixedStrideExtractorStage(
                clip_len_s=config.clip_len_s,
                clip_stride_s=config.clip_stride_s,
                min_clip_length_s=config.min_clip_length_s,
                limit_clips=config.limit_clips,
                verbose=config.verbose,
                log_stats=config.perf_profile,
            ),
            num_workers_per_node=1,
        ),
    ]


def build_transcode_stages(config: TranscodeConfig) -> list[CuratorStage | CuratorStageSpec]:
    """Construct and return the clip transcoding stage."""
    return [
        ClipTranscodingStage(
            num_cpus_per_worker=config.num_cpus_per_worker,
            encoder=config.encoder,
            encoder_threads=config.encoder_threads,
            encode_batch_size=config.encode_batch_size,
            use_hwaccel=config.use_hwaccel,
            use_input_bit_rate=config.use_input_bit_rate,
            num_clips_per_chunk=config.num_clips_per_chunk,
            max_output_frames=config.max_output_frames,
            verbose=config.verbose,
            log_stats=config.perf_profile,
        ),
    ]


def build_frame_extraction_stages(config: FrameExtractionConfig) -> list[CuratorStage | CuratorStageSpec]:
    """Construct and return the clip frame extraction stage."""
    return [
        ClipFrameExtractionStage(
            extraction_policies=(FrameExtractionPolicy.sequence,),
            target_fps=config.target_fps,
            target_res=(config.target_res, config.target_res),
            num_cpus_per_worker=config.cpus_per_worker,
            log_stats=config.perf_profile,
        ),
    ]
