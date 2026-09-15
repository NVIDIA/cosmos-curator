# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Curator Next ``video-caption`` entry point.

Usage::

    python -m cosmos_curator.next.recipes.video_caption.pipeline config.yaml
"""

import argparse
import logging
from collections.abc import Sequence
from functools import partial

import lance
import pyarrow as pa
from loguru import logger

from cosmos_curator.core.utils.storage.storage_utils import get_lance_storage_options
from cosmos_curator.next.core.ray_runtime import (
    configure_ray_data_eager_actor_autoscaling,
    configure_ray_data_progress,
    configure_ray_data_stability,
    curator_io_resources,
    ensure_ray_initialized,
)
from cosmos_curator.next.recipes.video_caption.bindings import (
    CaptionInputBindings,
    CaptionPublicationBindings,
    CaptionResultBindings,
    CaptionRunBindings,
)
from cosmos_curator.next.recipes.video_caption.config import ResolvedVideoCaptionConfig, resolve_config
from cosmos_curator.next.recipes.video_caption.contracts import (
    CHECKPOINT_ADAPTER_VERSION,
    WORKSPACE_SCHEMA_VERSION,
    build_terminal_result_row,
    caption_contract_digest,
    normalized_caption_contract,
    resolve_model_spec,
    validate_model_directory,
    validate_terminal_row,
    validate_terminal_value,
)
from cosmos_curator.next.recipes.video_caption.inference import (
    MAPPED_INPUT_SCHEMA,
    CaptionExecutionConfig,
    decompose_row_addresses,
    result_schema,
    run_inference_phase,
)
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
from cosmos_curator.next.recipes.video_split.lance_sink import validate_clip_table
from cosmos_curator.next.recipes.video_split.records import CLIP_SCHEMA
from cosmos_curator.next.utils.arrow_utils import s3_compatibility_env_vars


def run_config(config: ResolvedVideoCaptionConfig) -> dict[str, object]:
    """Run one finite caption attempt and return its canonical outcome."""
    uri = config.input.clips_lance_uri
    storage_options = get_lance_storage_options(uri, profile_name=config.execution.storage_profile)
    spec = resolve_model_spec(config.model.variant)
    digest = caption_contract_digest(spec)
    result_contract = normalized_caption_contract(spec)
    bindings = CaptionRunBindings(
        input=CaptionInputBindings(
            source_columns=("clip_id", "clip_uri", "clip_size_bytes"),
            source_schema=pa.schema([CLIP_SCHEMA.field(name) for name in ("clip_id", "clip_uri", "clip_size_bytes")]),
            mapped_schema=MAPPED_INPUT_SCHEMA,
            project_batch=decompose_row_addresses,
            row_id_column="clip_id",
            media_uri_column="clip_uri",
            media_size_column="clip_size_bytes",
            media_sha256_column=None,
            staged_binding_columns=("clip_id",),
        ),
        result=CaptionResultBindings(
            caption_field_name=spec.caption_field_name,
            metadata_field_name=spec.metadata_field_name,
            fields=spec.fields,
            staged_schema=result_schema(spec),
            digest=digest,
            build_terminal_row=partial(build_terminal_result_row, spec=spec, digest=digest),
            validate_terminal_value=partial(validate_terminal_value, spec=spec, digest=digest),
            validate_staged_row=partial(validate_terminal_row, spec=spec, digest=digest),
        ),
        publication=CaptionPublicationBindings(
            validate_base_schema=partial(validate_clip_table, uri=uri),
            protected_field_names=tuple(CLIP_SCHEMA.names),
            base_field_owner="video-split",
            transaction_properties={"kind": "video-caption"},
        ),
    )
    inference_config = CaptionExecutionConfig(
        storage_profile=config.execution.storage_profile,
        inference_concurrency=config.execution.inference_concurrency,
        inference_batch_size=config.execution.inference_batch_size,
        max_concurrent_batches=config.execution.max_concurrent_batches,
        tensor_parallel_size=config.execution.tensor_parallel_size,
        pipeline_parallel_size=config.execution.pipeline_parallel_size,
        media_concurrency=config.execution.media_concurrency,
        media_batch_size=config.execution.media_batch_size,
        media_cpus=config.execution.media_cpus,
        media_attempts=config.execution.media_attempts,
        lance_read_batch_size=config.execution.lance_read_batch_size,
        parquet_rows_per_file=config.execution.parquet_rows_per_file,
    )
    ensure_caption_fields(
        uri,
        storage_options=storage_options,
        bindings=bindings,
        attempts=config.execution.commit_attempts,
    )
    dataset = lance.dataset(uri, storage_options=storage_options)
    validate_caption_input(dataset, uri=uri)
    attempt = capture_attempt(dataset, bindings=bindings)
    workspace = resolve_workspace(
        config.output.staging_root_uri,
        spec.caption_field_name,
        digest,
        storage_profile=config.execution.storage_profile,
    )
    manifest: dict[str, object] = {
        "workspace_schema_version": WORKSPACE_SCHEMA_VERSION,
        "media_root": config.input.media_root,
        "clips_lance_uri": config.input.clips_lance_uri,
        "caption_field": spec.caption_field_name,
        "metadata_field": spec.metadata_field_name,
        "field_schemas": result_contract["fields"],
        "model": {
            "id": spec.model_id,
            "revision": spec.revision,
            "runtime_model_dir": str(spec.runtime_model_dir),
        },
        "caption_contract": result_contract,
        "caption_contract_digest": digest,
        "checkpoint_adapter_version": CHECKPOINT_ADAPTER_VERSION,
    }
    workspace_preexisting = workspace_manifest_exists(workspace)
    logger.info(
        "Caption attempt v{} selected {} pending and {} complete fragment(s) for field {}",
        attempt.version,
        len(attempt.pending_fragment_ids),
        len(attempt.complete_fragment_ids),
        spec.caption_field_name,
    )

    if attempt.pending_fragment_ids:
        ensure_workspace(workspace, storage_profile=config.execution.storage_profile, manifest=manifest)
        phase_a_complete = phase_a_completion_covers(workspace, attempt, digest)
        model_dir = None
        if phase_a_complete:
            logger.info(
                "Phase A completion marker covers all {} pending fragment(s); skipping model setup and inference",
                len(attempt.pending_fragment_ids),
            )
        else:
            model_dir = validate_model_directory(spec)
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
        if model_dir is not None:
            configure_ray_data_eager_actor_autoscaling()
            run_inference_phase(
                dataset,
                attempt,
                inference_config,
                bindings,
                model_dir,
                workspace,
                storage_options=storage_options,
            )
    elif workspace_preexisting:
        # Never remove a pre-existing recovery scope until its manifest is
        # proven to describe this exact result contract.
        ensure_workspace(workspace, storage_profile=config.execution.storage_profile, manifest=manifest)

    publication = publish_staged_results(
        uri,
        attempt,
        workspace,
        bindings,
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
