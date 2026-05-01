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
"""Stage builders for captioning and T5 encoding."""

import attrs

from cosmos_curator.core.interfaces.stage_interface import CuratorStage, CuratorStageSpec
from cosmos_curator.pipelines.video.captioning.captioning_stages import (
    EnhanceCaptionStage,
    T5StageForSplit,
)
from cosmos_curator.pipelines.video.captioning.gemini_caption_stage import ApiPrepStage, GeminiCaptionStage
from cosmos_curator.pipelines.video.captioning.openai_caption_stage import OpenAICaptionStage
from cosmos_curator.pipelines.video.captioning.vllm_async_config import VllmAsyncConfig
from cosmos_curator.pipelines.video.captioning.vllm_caption_stage import VllmCaptionStage, VllmPrepStage
from cosmos_curator.pipelines.video.preview.preview_stages import PreviewStage
from cosmos_curator.pipelines.video.utils.data_model import VllmConfig, WindowConfig

VLLM_CAPTION_ALGOS: frozenset[str] = frozenset(
    {"nemotron", "qwen", "qwen3_5_27b", "qwen3_vl_30b", "qwen3_vl_30b_fp8", "qwen3_vl_235b", "qwen3_vl_235b_fp8"}
    | {"cosmos_r1", "cosmos_r2"}
)


@attrs.define(frozen=True)
class EnhanceCaptionConfig:
    """Configuration for caption enhancement via a language model."""

    model_variant: str = "qwen_lm"
    batch_size: int = 32
    openai_model: str = "auto"
    fp8_enable: bool = False
    max_output_tokens: int = 2048
    prompt_variant: str = "default"
    prompt_text: str | None = None
    verbose: bool = False
    perf_profile: bool = False


@attrs.define(frozen=True)
class GeminiConfig:
    """Configuration specific to the Gemini API captioning path."""

    model_name: str = "models/gemini-2.5-pro"
    max_output_tokens: int = 8192
    prompt_variant: str = "default"
    prompt_text: str | None = None
    caption_retries: int = 3
    retry_delay_seconds: float = 1.0
    max_inline_video_bytes: int = 20 * 1024 * 1024
    batch_size: int = 1
    num_cpus_for_prepare: float = 3.0


@attrs.define(frozen=True)
class OpenAIConfig:
    """Configuration specific to the OpenAI-compatible API captioning path."""

    model_name: str = "auto"
    max_output_tokens: int = 8192
    prompt_variant: str = "default"
    prompt_text: str | None = None
    caption_retries: int = 3
    retry_delay_seconds: float = 1.0
    batch_size: int = 1
    num_cpus_for_prepare: float = 3.0


@attrs.define(frozen=True)
class VllmAsyncCaptionConfig:
    """Configuration for the ``vllm_async`` captioning path."""

    model_name: str = "qwen"
    prompt_variant: str = "default"
    prompt_text: str | None = None
    max_concurrent_requests: int = attrs.field(default=0, validator=attrs.validators.ge(0))
    serve_config: VllmAsyncConfig | None = None
    stage_batch_size: int = 0  # 0 = auto-derive
    num_workers_per_node: int = 0
    stage2_caption: bool = False
    stage2_prompt_text: str | None = None


CaptionBackendConfig = VllmConfig | GeminiConfig | OpenAIConfig | VllmAsyncCaptionConfig
"""Discriminated union of captioning backend configurations.

Exactly one backend config is valid per invocation.  The builder functions
dispatch via ``match`` on the concrete type instead of stringly-typed
``caption_algo`` checks.
"""


@attrs.define(frozen=True)
class CaptioningConfig:
    """Configuration for the captioning phase (prep + caption + optional enhance)."""

    backend: CaptionBackendConfig
    window_config: WindowConfig
    keep_mp4: bool = False
    generate_previews: bool = False
    preview_target_fps: int = 1
    preview_target_height: int = 240
    inflight_batching: bool = True
    enhance_config: EnhanceCaptionConfig | None = None
    verbose: bool = False
    perf_profile: bool = False


@attrs.define(frozen=True)
class T5Config:
    """Configuration for T5 encoding of captions."""

    caption_fields: list[str]
    verbose: bool = False
    perf_profile: bool = False


def _build_captioning_prep_stage(config: CaptioningConfig) -> CuratorStage | CuratorStageSpec:
    """Build the prep stage for the configured caption backend."""
    match config.backend:
        case GeminiConfig() as gcfg:
            return ApiPrepStage(
                window_config=config.window_config,
                model_variant="gemini",
                num_cpus_for_prepare=gcfg.num_cpus_for_prepare,
                verbose=config.verbose,
                log_stats=config.perf_profile,
            )
        case OpenAIConfig() as ocfg:
            return ApiPrepStage(
                window_config=config.window_config,
                model_variant="openai",
                num_cpus_for_prepare=ocfg.num_cpus_for_prepare,
                verbose=config.verbose,
                log_stats=config.perf_profile,
            )
        case VllmConfig() as vcfg:
            vllm_cfg_prepare = attrs.evolve(vcfg, copy_weights_to=None)
            return VllmPrepStage(
                vllm_config=vllm_cfg_prepare,
                window_config=config.window_config,
                keep_mp4=config.keep_mp4,
                verbose=config.verbose,
                log_stats=config.perf_profile,
            )
        case VllmAsyncCaptionConfig() as vsc:
            return _build_vllm_async_prep_stage(config, vsc)
        case _:
            msg = f"Unsupported caption backend type: {type(config.backend).__name__}"  # type: ignore[unreachable]
            raise NotImplementedError(msg)


def _require_vllm_async_serve_config(vsc: VllmAsyncCaptionConfig) -> VllmAsyncConfig:
    """Validate and return the serve_config from a VllmAsyncCaptionConfig."""
    if vsc.serve_config is None:
        msg = "VllmAsyncCaptionConfig.serve_config is required"
        raise ValueError(msg)
    return vsc.serve_config


def _build_vllm_async_prep_stage(config: CaptioningConfig, vsc: VllmAsyncCaptionConfig) -> CuratorStageSpec:
    """Build the vLLM async prep stage (windowing + decode + prompt build)."""
    serve_config = _require_vllm_async_serve_config(vsc)

    from cosmos_curator.pipelines.video.captioning.vllm_async_config import VllmAsyncPrepConfig  # noqa: PLC0415
    from cosmos_curator.pipelines.video.captioning.vllm_async_stage import VllmAsyncPrepStage  # noqa: PLC0415

    prep_config = VllmAsyncPrepConfig(
        model_variant=serve_config.model_variant,
        sampling_config=serve_config.sampling_config,
        prompt_variant=vsc.prompt_variant,
        prompt_text=vsc.prompt_text,
        sample_fps=config.window_config.sampling_fps,
        window_size=config.window_config.window_size,
        remainder_threshold=config.window_config.remainder_threshold,
        keep_mp4=config.generate_previews or config.keep_mp4,
        use_input_bit_rate=config.window_config.use_input_bit_rate,
    )
    stage = VllmAsyncPrepStage(
        prep_config=prep_config,
        verbose=config.verbose,
        log_stats=config.perf_profile,
    )
    return CuratorStageSpec(stage, over_provision_factor=2.0)


def _build_vllm_async_render_stage(config: CaptioningConfig, vsc: VllmAsyncCaptionConfig) -> CuratorStageSpec:
    """Build the vLLM async render stage (TextPrompt -> ProcessorInputs)."""
    serve_config = _require_vllm_async_serve_config(vsc)

    from cosmos_curator.pipelines.video.captioning.vllm_async_stage import VllmAsyncPromptRenderStage  # noqa: PLC0415

    stage = VllmAsyncPromptRenderStage(
        serve_config=serve_config,
        model_name=vsc.model_name,
        verbose=config.verbose,
        log_stats=config.perf_profile,
    )
    return CuratorStageSpec(stage, over_provision_factor=1.5)


def _build_captioning_caption_stage(config: CaptioningConfig) -> CuratorStage | CuratorStageSpec:
    """Build the caption stage for the configured caption backend."""
    match config.backend:
        case VllmConfig() as vcfg:
            return CuratorStageSpec(
                VllmCaptionStage(
                    vllm_config=vcfg,
                    verbose=config.verbose,
                    keep_mp4=config.keep_mp4,
                    log_stats=config.perf_profile,
                    inflight_batching=config.inflight_batching,
                ),
                num_setup_attempts_python=None,
            )
        case GeminiConfig() as gcfg:
            return GeminiCaptionStage(
                model_variant="gemini",
                model_name=gcfg.model_name,
                prompt_variant=gcfg.prompt_variant,
                prompt_text=gcfg.prompt_text,
                max_output_tokens=gcfg.max_output_tokens,
                max_caption_retries=gcfg.caption_retries,
                retry_delay_seconds=gcfg.retry_delay_seconds,
                max_video_size_bytes=gcfg.max_inline_video_bytes,
                batch_size=gcfg.batch_size,
                verbose=config.verbose,
                log_stats=config.perf_profile,
            )
        case OpenAIConfig() as ocfg:
            return OpenAICaptionStage(
                model_name=ocfg.model_name,
                model_variant="openai",
                prompt_variant=ocfg.prompt_variant,
                prompt_text=ocfg.prompt_text,
                max_output_tokens=ocfg.max_output_tokens,
                max_caption_retries=ocfg.caption_retries,
                retry_delay_seconds=ocfg.retry_delay_seconds,
                batch_size=ocfg.batch_size,
                verbose=config.verbose,
                log_stats=config.perf_profile,
            )
        case VllmAsyncCaptionConfig() as vsc:
            return _build_vllm_async_caption_stage(config, vsc)
        case _:
            msg = f"Unsupported caption backend type: {type(config.backend).__name__}"  # type: ignore[unreachable]
            raise NotImplementedError(msg)


def _build_vllm_async_caption_stage(config: CaptioningConfig, vsc: VllmAsyncCaptionConfig) -> CuratorStageSpec:
    """Build the vllm_async caption stage with mode-dependent worker count."""
    serve_config = _require_vllm_async_serve_config(vsc)

    from cosmos_curator.pipelines.video.captioning.vllm_async_stage import VllmAsyncCaptionStage  # noqa: PLC0415

    stage = VllmAsyncCaptionStage(
        serve_config=serve_config,
        model_name=vsc.model_name,
        max_concurrent_requests=vsc.max_concurrent_requests,
        stage_batch_size=vsc.stage_batch_size,
        verbose=config.verbose,
        log_stats=config.perf_profile,
        stage2_caption=vsc.stage2_caption,
        stage2_prompt_text=vsc.stage2_prompt_text,
    )
    if serve_config.data_parallel_size > 1:
        return CuratorStageSpec(stage, num_workers_per_node=1, worker_max_lifetime_m=0)
    if vsc.num_workers_per_node > 0:
        return CuratorStageSpec(stage, num_workers_per_node=vsc.num_workers_per_node, worker_max_lifetime_m=0)
    return CuratorStageSpec(stage, worker_max_lifetime_m=0, over_provision_factor=1.5)


def build_captioning_stages(config: CaptioningConfig) -> list[CuratorStage | CuratorStageSpec]:
    """Construct and return the prep, optional preview, render, caption, and enhance stages."""
    stages: list[CuratorStage | CuratorStageSpec] = [_build_captioning_prep_stage(config)]

    if config.generate_previews:
        stages.append(
            PreviewStage(
                target_fps=config.preview_target_fps,
                target_height=config.preview_target_height,
                verbose=config.verbose,
                log_stats=config.perf_profile,
            )
        )

    if isinstance(config.backend, VllmAsyncCaptionConfig):
        stages.append(_build_vllm_async_render_stage(config, config.backend))

    stages.append(_build_captioning_caption_stage(config))

    if config.enhance_config is not None:
        ecfg = config.enhance_config
        stages.append(
            EnhanceCaptionStage(
                model_variant=ecfg.model_variant,
                batch_size=ecfg.batch_size,
                openai_model=ecfg.openai_model,
                fp8_enable=ecfg.fp8_enable,
                max_output_tokens=ecfg.max_output_tokens,
                prompt_variant=ecfg.prompt_variant,
                prompt_text=ecfg.prompt_text,
                verbose=ecfg.verbose,
                log_stats=ecfg.perf_profile,
            )
        )

    return stages


def build_t5_stages(config: T5Config) -> list[CuratorStage | CuratorStageSpec]:
    """Construct and return the T5 encoding stage."""
    return [
        CuratorStageSpec(
            T5StageForSplit(
                caption_fields=config.caption_fields,
                verbose=config.verbose,
                log_stats=config.perf_profile,
            ),
        ),
    ]
