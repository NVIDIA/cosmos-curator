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
"""Stage builder for image embedding generation."""

from typing import Literal, cast

import attrs

from cosmos_curator.core.interfaces.stage_interface import CuratorStage, CuratorStageSpec
from cosmos_curator.pipelines.image.embedding.image_embedding_stages import (
    ImageCLIPEmbeddingStage,
    ImageCosmosEmbed1EmbeddingStage,
    ImageInternVideo2EmbeddingStage,
    ImageOpenAIEmbeddingStage,
)

_COSMOS_EMBED1_VARIANTS: frozenset[str] = frozenset({"224p", "336p", "448p"})


@attrs.define(frozen=True)
class CosmosEmbed1ImageEmbeddingConfig:
    """Backend config for Cosmos-Embed1 image embedding."""

    variant: str = attrs.field(validator=attrs.validators.in_(sorted(_COSMOS_EMBED1_VARIANTS)))


@attrs.define(frozen=True)
class InternVideo2ImageEmbeddingConfig:
    """Backend config for InternVideo2 image embedding (no backend-specific fields)."""


@attrs.define(frozen=True)
class CLIPImageEmbeddingConfig:
    """Backend config for CLIP image embedding (no backend-specific fields)."""


@attrs.define(frozen=True)
class OpenAIImageEmbeddingConfig:
    """Backend config for OpenAI-compatible API image embedding."""

    model_name: str = "auto"
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    max_concurrent_requests: int = 8


ImageEmbeddingBackendConfig = (
    CosmosEmbed1ImageEmbeddingConfig
    | InternVideo2ImageEmbeddingConfig
    | CLIPImageEmbeddingConfig
    | OpenAIImageEmbeddingConfig
)
"""Discriminated union of image embedding backend configurations."""


@attrs.define(frozen=True)
class ImageEmbeddingConfig:
    """Configuration for image embedding generation."""

    backend: ImageEmbeddingBackendConfig = attrs.Factory(InternVideo2ImageEmbeddingConfig)
    gpus_per_worker: float = 1.0
    verbose: bool = False
    perf_profile: bool = False


def build_image_embedding_stages(config: ImageEmbeddingConfig) -> list[CuratorStage | CuratorStageSpec]:
    """Construct and return the embedding stage for images."""
    match config.backend:
        case CosmosEmbed1ImageEmbeddingConfig() as ce1:
            variant = cast("Literal['224p', '336p', '448p']", ce1.variant)
            stage: CuratorStage = ImageCosmosEmbed1EmbeddingStage(
                variant,
                num_gpus_per_worker=config.gpus_per_worker,
                verbose=config.verbose,
                log_stats=config.perf_profile,
            )
        case InternVideo2ImageEmbeddingConfig():
            stage = ImageInternVideo2EmbeddingStage(
                num_gpus_per_worker=config.gpus_per_worker,
                verbose=config.verbose,
                log_stats=config.perf_profile,
            )
        case CLIPImageEmbeddingConfig():
            stage = ImageCLIPEmbeddingStage(
                num_gpus_per_worker=config.gpus_per_worker,
                verbose=config.verbose,
                log_stats=config.perf_profile,
            )
        case OpenAIImageEmbeddingConfig() as oai:
            stage = ImageOpenAIEmbeddingStage(
                model_name=oai.model_name,
                max_retries=oai.max_retries,
                retry_delay_seconds=oai.retry_delay_seconds,
                max_concurrent_requests=oai.max_concurrent_requests,
                verbose=config.verbose,
                log_stats=config.perf_profile,
            )
        case _:
            msg = f"Unsupported image embedding backend type: {type(config.backend).__name__}"  # type: ignore[unreachable]
            raise NotImplementedError(msg)
    return [CuratorStageSpec(stage)]
