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
"""Stage builder for clip embedding generation."""

from typing import Literal, cast

import attrs

from cosmos_curator.core.interfaces.stage_interface import CuratorStage, CuratorStageSpec
from cosmos_curator.models.all_models import get_all_models_by_id
from cosmos_curator.pipelines.video.embedding.cosmos_embed1_stages import (
    CosmosEmbed1EmbeddingStage,
    CosmosEmbed1FrameCreationStage,
)
from cosmos_curator.pipelines.video.embedding.internvideo2_stages import (
    InternVideo2EmbeddingStage,
    InternVideo2FrameCreationStage,
)
from cosmos_curator.pipelines.video.embedding.openai_embedding_stage import OpenAIEmbeddingStage

_COSMOS_EMBED1_VARIANTS: frozenset[str] = frozenset({"224p", "336p", "448p"})


@attrs.define(frozen=True)
class InternVideo2Config:
    """Backend config for InternVideo2 embedding (no backend-specific fields)."""


@attrs.define(frozen=True)
class CosmosEmbed1Config:
    """Backend config for CosmosEmbed1 embedding."""

    variant: str = attrs.field(validator=attrs.validators.in_(sorted(_COSMOS_EMBED1_VARIANTS)))


@attrs.define(frozen=True)
class OpenAIEmbeddingConfig:
    """Configuration specific to the OpenAI-compatible API embedding path."""

    model_name: str = "auto"
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    max_concurrent_requests: int = 8


EmbeddingBackendConfig = InternVideo2Config | CosmosEmbed1Config | OpenAIEmbeddingConfig
"""Discriminated union of embedding backend configurations.

Exactly one backend config is valid per invocation.  The builder functions
dispatch via ``match`` on the concrete type instead of stringly-typed
``algorithm`` checks.
"""


@attrs.define(frozen=True)
class EmbeddingConfig:
    """Configuration for clip embedding generation."""

    backend: EmbeddingBackendConfig = attrs.Factory(InternVideo2Config)
    target_fps: float = 2.0
    gpus_per_worker: float = 0.25
    batch_size: int = 8
    verbose: bool = False
    perf_profile: bool = False


def _build_embedding_stage(config: EmbeddingConfig) -> CuratorStage:
    """Construct the embedding stage matching the configured backend."""
    match config.backend:
        case InternVideo2Config():
            return InternVideo2EmbeddingStage(
                num_gpus_per_worker=config.gpus_per_worker,
                batch_size=config.batch_size,
                verbose=config.verbose,
                log_stats=config.perf_profile,
            )
        case CosmosEmbed1Config() as ce1:
            variant = cast("Literal['224p', '336p', '448p']", ce1.variant)
            return CosmosEmbed1EmbeddingStage(
                variant,
                num_gpus_per_worker=config.gpus_per_worker,
                verbose=config.verbose,
                log_stats=config.perf_profile,
            )
        case OpenAIEmbeddingConfig() as oai:
            return OpenAIEmbeddingStage(
                model_name=oai.model_name,
                target_fps=config.target_fps,
                max_retries=oai.max_retries,
                retry_delay_seconds=oai.retry_delay_seconds,
                max_concurrent_requests=oai.max_concurrent_requests,
                verbose=config.verbose,
                log_stats=config.perf_profile,
            )
        case _:
            msg = f"Unsupported embedding backend type: {type(config.backend).__name__}"  # type: ignore[unreachable]
            raise NotImplementedError(msg)


def get_embedding_model_version(config: EmbeddingConfig) -> str:
    """Return the embedding model version string for output metadata."""
    match config.backend:
        case OpenAIEmbeddingConfig() as oai:
            return oai.model_name
        case _:
            model = _build_embedding_stage(config).model
            if model is not None:
                model_id = model.model_id_names[0]
                return str(get_all_models_by_id().get(model_id, {}).get("version", "unspecified"))
            return "unspecified"


def build_embedding_stages(config: EmbeddingConfig) -> list[CuratorStage | CuratorStageSpec]:
    """Construct and return the frame creation and embedding stages."""
    match config.backend:
        case OpenAIEmbeddingConfig():
            # OpenAI embedding reads pre-extracted frames directly; no frame creation stage.
            return [_build_embedding_stage(config)]
        case InternVideo2Config():
            frame_stage: CuratorStage = InternVideo2FrameCreationStage(
                target_fps=config.target_fps,
                verbose=config.verbose,
                log_stats=config.perf_profile,
            )
        case CosmosEmbed1Config() as ce1:
            variant = cast("Literal['224p', '336p', '448p']", ce1.variant)
            frame_stage = CosmosEmbed1FrameCreationStage(
                variant,
                target_fps=config.target_fps,
                verbose=config.verbose,
                log_stats=config.perf_profile,
            )
        case _:
            msg = f"Unsupported embedding backend type: {type(config.backend).__name__}"  # type: ignore[unreachable]
            raise NotImplementedError(msg)
    return [frame_stage, _build_embedding_stage(config)]
