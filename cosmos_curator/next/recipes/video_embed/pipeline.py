# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpointed clip inference followed by one atomic Lance enrichment commit."""

import json
from functools import partial

import ray
from loguru import logger

from cosmos_curator.core.utils import environment
from cosmos_curator.core.utils.pixi_runtime_envs import ray_data_gpu_runtime_env
from cosmos_curator.next.core.ray_runtime import (
    configure_ray_data_progress,
    configure_ray_data_stability,
    curator_io_resources,
    ensure_ray_initialized,
)
from cosmos_curator.next.enrichment.contracts import EnrichmentBindings
from cosmos_curator.next.enrichment.execution import ProcessingScope, run_enrichment
from cosmos_curator.next.recipes.video_embed.bindings import resolve_bindings
from cosmos_curator.next.recipes.video_embed.config import Concurrency, ResolvedVideoEmbedConfig
from cosmos_curator.next.recipes.video_embed.contracts import (
    READ_COLUMNS,
    ModelSpec,
    resolve_model_spec,
    validate_model_directory,
)
from cosmos_curator.next.recipes.video_embed.processing import (
    VideoEmbedder,
    download_clip,
)


def _actor_pool(concurrency: Concurrency) -> ray.data.ActorPoolStrategy:
    # Ray's default task prefetching lets serial model actors reach its
    # autoscaling threshold without lowering that threshold for the whole plan.
    return ray.data.ActorPoolStrategy(
        min_size=1,
        max_size=None if concurrency == "auto" else concurrency,
    )


def start_ray(bindings: EnrichmentBindings, config: ResolvedVideoEmbedConfig) -> None:
    """Start Ray with IO slots and the storage environment for media, clips, and staging."""
    ensure_ray_initialized(
        local_resources=curator_io_resources(),
        runtime_env_vars=bindings.storage_env_vars(config.input.media_root),
    )
    configure_ray_data_progress(progress=config.execution.progress)
    configure_ray_data_stability(disable_high_memory_detector=True)


def embed_clips(processing: ProcessingScope, config: ResolvedVideoEmbedConfig, spec: ModelSpec) -> None:
    """Stream downloaded clips through GPU preprocessing and inference to durable results."""
    execution = config.execution
    model_dir = str(validate_model_directory(spec))
    clips = processing.read(READ_COLUMNS)
    # Keep loaded actors while their pool has work in flight between download bursts.
    clips.context.autoscaling_config.actor_pool_util_downscaling_threshold = 0.0
    downloaded = clips.map(
        download_clip,
        fn_kwargs={"storage_profile": execution.storage_profile, "max_attempts": execution.media_attempts},
        num_cpus=0.25,
        # Backpressured tasks retain encoded buffers on their worker heap.
        resources={environment.CURATOR_IO_RESOURCE_NAME: 1.0},
        max_retries=-1,
    )
    inferred = downloaded.map_batches(
        VideoEmbedder,
        fn_constructor_kwargs={"spec": spec, "model_dir": model_dir, "staged_schema": processing.staged_schema},
        batch_format="pyarrow",
        batch_size=execution.inference_batch_size,
        compute=_actor_pool(execution.inference_concurrency),
        num_cpus=1,
        num_gpus=1,
        runtime_env=ray_data_gpu_runtime_env("legacy-transformers"),
        max_concurrency=1,
        max_restarts=-1,
        max_task_retries=-1,
    )
    processing.write(inferred)


def run_config(config: ResolvedVideoEmbedConfig) -> dict[str, object]:
    """Resume compatible GPU work, publish once, and retain failed attempts for recovery."""
    spec = resolve_model_spec(config.model.variant)
    outcome = run_enrichment(
        resolve_bindings(config),
        start_ray=partial(start_ray, config=config),
        process=partial(embed_clips, config=config, spec=spec),
    )
    summary: dict[str, object] = {
        **outcome.summary(),
        "clips_lance_uri": config.input.clips_lance_uri,
        "clips_lance_version": int(outcome.dataset.version),
    }
    logger.info("video-embed: {}", json.dumps(summary, sort_keys=True))
    return summary
