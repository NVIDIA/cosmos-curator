# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Checkpointed clip captioning followed by one atomic Lance enrichment commit.

Usage::

    python -m cosmos_curator.next.recipes.video_caption.pipeline config.yaml
"""

import argparse
import json
import logging
from collections.abc import Sequence
from functools import partial

from loguru import logger

from cosmos_curator.next.core.ray_runtime import (
    configure_ray_data_eager_actor_autoscaling,
    configure_ray_data_progress,
    configure_ray_data_stability,
    curator_io_resources,
    ensure_ray_initialized,
)
from cosmos_curator.next.enrichment.contracts import EnrichmentBindings
from cosmos_curator.next.enrichment.execution import ProcessingScope, run_enrichment
from cosmos_curator.next.recipes.video_caption.bindings import resolve_bindings
from cosmos_curator.next.recipes.video_caption.config import ResolvedVideoCaptionConfig, resolve_config
from cosmos_curator.next.recipes.video_caption.contracts import (
    DEFAULT_PROMPT,
    READ_COLUMNS,
    CaptionModelSpec,
    build_terminal_result_row,
    caption_contract_digest,
    resolve_model_spec,
    validate_model_directory,
)
from cosmos_curator.next.recipes.video_caption.inference import CaptionRequest, MediaColumns, caption_rows


def start_ray(bindings: EnrichmentBindings, config: ResolvedVideoCaptionConfig) -> None:
    """Start Ray with IO slots and the storage environment for media, clips, and staging."""
    ensure_ray_initialized(
        local_resources=curator_io_resources(),
        # The media root stands in for the per-row clip URIs, which are read
        # from the table and so are not enumerable here.
        runtime_env_vars=bindings.storage_env_vars(config.input.media_root),
    )
    configure_ray_data_progress(progress=config.execution.progress)
    configure_ray_data_stability(disable_high_memory_detector=True)


def caption_clips(processing: ProcessingScope, config: ResolvedVideoCaptionConfig, spec: CaptionModelSpec) -> None:
    """Stream fetched clips through vLLM to durable terminal captions."""
    execution = config.execution
    model_dir = validate_model_directory(spec)
    digest = caption_contract_digest(spec, max_model_len=execution.max_model_len)
    request = CaptionRequest(
        media=MediaColumns(uri="clip_uri", size="clip_size_bytes"),
        identity_columns=("clip_id",),
        prompt_template=DEFAULT_PROMPT,
        build_terminal_row=partial(build_terminal_result_row, spec=spec, digest=digest),
        result_schema=processing.result_schema,
    )
    # Model actors claim GPUs as they appear rather than one replica per decision.
    configure_ray_data_eager_actor_autoscaling()
    clips = processing.read(READ_COLUMNS)
    processing.write(caption_rows(clips, request, execution, model_dir, max_rows=processing.selected_rows))


def run_config(config: ResolvedVideoCaptionConfig) -> dict[str, object]:
    """Resume compatible caption work, publish once, and clean verified recovery state."""
    spec = resolve_model_spec(config.model.variant)
    outcome = run_enrichment(
        resolve_bindings(config),
        start_ray=partial(start_ray, config=config),
        process=partial(caption_clips, config=config, spec=spec),
    )
    summary: dict[str, object] = {
        **outcome.summary(),
        "clips_lance_uri": config.input.clips_lance_uri,
        "clips_lance_version": int(outcome.dataset.version),
    }
    logger.info("video-caption: {}", json.dumps(summary, sort_keys=True))
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    """Run from a strict JSON/YAML config path."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Curator Next video-caption pipeline")
    parser.add_argument("config", help="Path to a JSON/YAML video-caption config.")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="PATH=VALUE",
        help="Small resolved-config override in dotted PATH=VALUE form.",
    )
    args = parser.parse_args(argv)
    run_config(resolve_config(args.config, overrides=args.overrides))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
