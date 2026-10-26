# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Discovered sources -> checkpointed source plans -> packed spans in one Lance append."""

import json
from functools import partial
from typing import Any

from loguru import logger

from cosmos_curator.next.append.contracts import AppendBindings
from cosmos_curator.next.append.execution import ProcessingScope, run_append
from cosmos_curator.next.core.ray_runtime import (
    configure_ray_data_progress,
    configure_ray_data_stability,
    curator_io_resources,
    ensure_ray_initialized,
)
from cosmos_curator.next.recipes.video_cut import fixed_stride
from cosmos_curator.next.recipes.video_cut.bindings import resolve_bindings
from cosmos_curator.next.recipes.video_cut.config import ResolvedVideoCutConfig, TransNetV2Config
from cosmos_curator.next.recipes.video_cut.discovery import discover_items


def start_ray(bindings: AppendBindings, config: ResolvedVideoCutConfig) -> None:
    """Start Ray with the storage environment and, for TransNetV2's download actors, curator_io slots."""
    download_actors = isinstance(config.split, TransNetV2Config)
    ensure_ray_initialized(
        local_resources=curator_io_resources() if download_actors else None,
        runtime_env_vars=bindings.storage_env_vars(),
    )
    configure_ray_data_progress(progress=config.execution.progress)
    # TransNetV2 bounds payload admission with curator_io; retain logical memory and backpressure.
    configure_ray_data_stability(disable_high_memory_detector=download_actors)


def cut_sources(processing: ProcessingScope, config: ResolvedVideoCutConfig) -> None:
    """Plan each selected source completely with the configured method, filtering checkpoints before media access."""
    if isinstance(config.split, TransNetV2Config):
        # PyTorch loads only for scene cuts.
        from cosmos_curator.next.recipes.video_cut import transnetv2  # noqa: PLC0415

        planned = transnetv2.plan(processing.read(), config)
    else:
        # Reading with the probe task's CPUs lets Ray run each source's read and probe as one task.
        sources = processing.read(num_cpus=config.execution.fixed_stride.probe_cpus)
        planned = fixed_stride.plan(sources, config)
    processing.write(planned)


def run_config(config: ResolvedVideoCutConfig) -> dict[str, Any]:
    """Append newly discovered sources' spans, resuming a saved selection first, and clean verified recovery state."""
    outcome = run_append(
        resolve_bindings(config),
        partial(discover_items, config),
        start_ray=partial(start_ray, config=config),
        process=partial(cut_sources, config=config),
    )
    summary: dict[str, Any] = {
        **outcome.summary(),
        "spans_lance_uri": config.output.spans_lance_uri,
        "spans_lance_version": int(outcome.dataset.version),
    }
    logger.info("video-cut: {}", json.dumps(summary, sort_keys=True))
    return summary
