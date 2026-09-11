# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Curator Next ``video-caption`` entry point.

Usage::

    python -m cosmos_curator.next.recipes.video_caption.pipeline config.yaml
"""

import argparse
import logging
from collections.abc import Sequence

import lance
from loguru import logger

from cosmos_curator.core.utils.storage.storage_utils import get_lance_storage_options
from cosmos_curator.next.core.ray_runtime import (
    configure_ray_data_eager_actor_autoscaling,
    configure_ray_data_progress,
    configure_ray_data_stability,
    curator_io_resources,
    ensure_ray_initialized,
)
from cosmos_curator.next.recipes.video_caption.config import ResolvedVideoCaptionConfig, resolve_config
from cosmos_curator.next.recipes.video_caption.contracts import (
    caption_contract_digest,
    resolve_model_spec,
    validate_model_directory,
)
from cosmos_curator.next.recipes.video_caption.inference import run_inference_phase
from cosmos_curator.next.recipes.video_caption.lance_state import (
    capture_attempt,
    ensure_caption_fields,
    validate_caption_input,
)
from cosmos_curator.next.recipes.video_caption.publication import publish_staged_results
from cosmos_curator.next.recipes.video_caption.workspace import (
    cleanup_workspace,
    ensure_workspace,
    phase_a_completion_covers,
    resolve_workspace,
    workspace_manifest_exists,
)
from cosmos_curator.next.utils.arrow_utils import s3_compatibility_env_vars


def run_config(config: ResolvedVideoCaptionConfig) -> dict[str, object]:
    """Run one finite caption attempt and return its canonical outcome."""
    uri = config.input.clips_lance_uri
    storage_options = get_lance_storage_options(uri, profile_name=config.execution.storage_profile)
    spec = resolve_model_spec(config.model.variant)
    digest = caption_contract_digest(spec)
    ensure_caption_fields(
        uri,
        storage_options=storage_options,
        spec=spec,
        attempts=config.execution.commit_attempts,
    )
    dataset = lance.dataset(uri, storage_options=storage_options)
    validate_caption_input(dataset, uri=uri)
    attempt = capture_attempt(dataset, spec=spec, digest=digest)
    workspace = resolve_workspace(config, spec, digest)
    workspace_preexisting = workspace_manifest_exists(workspace)
    logger.info(
        "Caption attempt v{} selected {} pending and {} complete fragment(s) for field {}",
        attempt.version,
        len(attempt.pending_fragment_ids),
        len(attempt.complete_fragment_ids),
        spec.caption_field_name,
    )

    if attempt.pending_fragment_ids:
        ensure_workspace(workspace, config, spec, digest)
        phase_a_complete = phase_a_completion_covers(workspace, attempt, digest)
        if phase_a_complete:
            logger.info(
                "Phase A completion marker covers all {} pending fragment(s); skipping model setup and inference",
                len(attempt.pending_fragment_ids),
            )
        else:
            validate_model_directory(spec)
        ensure_ray_initialized(
            local_resources=curator_io_resources(),
            # Every configured location, not just the staging root: each is set
            # independently, so one can reach an endpoint the staging root does
            # not. The media root stands in for the per-row clip URIs, which are
            # read from the table and so are not enumerable here.
            runtime_env_vars=s3_compatibility_env_vars(
                config.input.media_root,
                uri,
                workspace.root_uri,
                storage_profile=config.execution.storage_profile,
            ),
        )
        configure_ray_data_progress(progress=config.execution.progress)
        configure_ray_data_stability(disable_high_memory_detector=True)
        if not phase_a_complete:
            configure_ray_data_eager_actor_autoscaling()
            run_inference_phase(
                dataset,
                attempt,
                config,
                spec,
                digest,
                workspace,
                storage_options=storage_options,
            )
    elif workspace_preexisting:
        # Never remove a pre-existing recovery scope until its manifest is
        # proven to describe this exact result contract.
        ensure_workspace(workspace, config, spec, digest)

    publication = publish_staged_results(
        uri,
        attempt,
        workspace,
        spec,
        digest,
        storage_options=storage_options,
        commit_attempts=config.execution.commit_attempts,
    )
    latest_version = int(lance.dataset(uri, storage_options=storage_options).version)
    cleanup_complete = True
    if attempt.pending_fragment_ids or workspace_preexisting:
        try:
            cleanup_workspace(workspace, storage_profile=config.execution.storage_profile)
        except Exception as exc:  # noqa: BLE001 -- cleanup is explicitly non-canonical best effort
            cleanup_complete = False
            logger.warning(
                "Caption rows are canonical at Lance v{}, but cleanup of {} failed: {!r}",
                latest_version,
                workspace.root_uri,
                exc,
            )

    summary: dict[str, object] = {
        "clips_lance_uri": uri,
        "clips_lance_version": latest_version,
        "attempt_version": attempt.version,
        "selected_fragments": len(attempt.selected_fragment_ids),
        "pending_fragments": len(attempt.pending_fragment_ids),
        "skipped_complete_fragments": len(attempt.complete_fragment_ids),
        "published_fragments": publication.published_fragments,
        "already_committed_fragments": publication.already_committed_fragments,
        "caption_field": spec.caption_field_name,
        "metadata_field": spec.metadata_field_name,
        "caption_contract_digest": digest,
        "cleanup_complete": cleanup_complete,
    }
    logger.info(
        "video-caption finished at Lance v{}: {} fragment(s) published, {} already committed, cleanup_complete={}",
        latest_version,
        publication.published_fragments,
        publication.already_committed_fragments,
        cleanup_complete,
    )
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
