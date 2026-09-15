# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Curator Next ``multimodal-caption`` entry point."""

import argparse
import logging
from collections.abc import Sequence
from functools import partial

import lance
from loguru import logger

from cosmos_curator.core.utils.storage.storage_utils import get_lance_storage_options
from cosmos_curator.models.registry import validate_model_directory
from cosmos_curator.next.core.ray_runtime import (
    configure_ray_data_eager_actor_autoscaling,
    configure_ray_data_progress,
    configure_ray_data_stability,
    curator_io_resources,
    ensure_ray_initialized,
)
from cosmos_curator.next.recipes.multimodal_caption.config import ResolvedMultimodalCaptionConfig, resolve_config
from cosmos_curator.next.recipes.multimodal_caption.contracts import (
    RESULT_VARIANT,
    build_terminal_result_row,
    caption_contract_digest,
    normalized_caption_contract,
    resolve_model,
    result_field_names,
    result_fields,
    staged_result_schema,
    validate_terminal_row,
    validate_terminal_value,
)
from cosmos_curator.next.recipes.multimodal_caption.episode_input import (
    EPISODE_MAPPED_INPUT_SCHEMA,
    EPISODE_SOURCE_COLUMNS,
    EPISODE_SOURCE_SCHEMA,
    EPISODE_STAGED_BINDING_COLUMNS,
    project_episode_camera,
    validate_complete_fragment,
    validate_episode_input,
)
from cosmos_curator.next.recipes.multimodal_split.lance_sink import validate_episode_table
from cosmos_curator.next.recipes.multimodal_split.records import EPISODE_SCHEMA
from cosmos_curator.next.recipes.video_caption.bindings import (
    CaptionInputBindings,
    CaptionPublicationBindings,
    CaptionResultBindings,
    CaptionRunBindings,
)
from cosmos_curator.next.recipes.video_caption.contracts import CHECKPOINT_ADAPTER_VERSION, WORKSPACE_SCHEMA_VERSION
from cosmos_curator.next.recipes.video_caption.inference import CaptionExecutionConfig, run_inference_phase
from cosmos_curator.next.recipes.video_caption.lance_state import capture_attempt, ensure_caption_fields
from cosmos_curator.next.recipes.video_caption.publication import publish_staged_results
from cosmos_curator.next.recipes.video_caption.workspace import (
    cleanup_workspace,
    ensure_workspace,
    phase_a_completion_covers,
    resolve_workspace,
    workspace_manifest_exists,
)
from cosmos_curator.next.utils.arrow_utils import s3_compatibility_env_vars


def run_config(config: ResolvedMultimodalCaptionConfig) -> dict[str, object]:
    """Run one finite episode-caption attempt and return its canonical outcome."""
    uri = config.input.episodes_lance_uri
    storage_profile = config.execution.storage_profile
    storage_options = get_lance_storage_options(uri, profile_name=storage_profile)
    model = resolve_model(config.model.variant)
    camera_key = config.context.camera_key
    digest = caption_contract_digest(model, camera_key)
    caption_field, metadata_field = result_field_names(model.key)
    contract = normalized_caption_contract(model, camera_key)
    bindings = CaptionRunBindings(
        input=CaptionInputBindings(
            source_columns=EPISODE_SOURCE_COLUMNS,
            source_schema=EPISODE_SOURCE_SCHEMA,
            mapped_schema=EPISODE_MAPPED_INPUT_SCHEMA,
            project_batch=partial(project_episode_camera, camera_key=camera_key),
            row_id_column="episode_id",
            media_uri_column="media_location",
            media_size_column="media_size_bytes",
            media_sha256_column="media_sha256",
            staged_binding_columns=EPISODE_STAGED_BINDING_COLUMNS,
        ),
        result=CaptionResultBindings(
            caption_field_name=caption_field,
            metadata_field_name=metadata_field,
            fields=result_fields(model.key),
            staged_schema=staged_result_schema(model.key),
            digest=digest,
            build_terminal_row=partial(build_terminal_result_row, model=model, digest=digest),
            validate_terminal_value=partial(
                validate_terminal_value,
                model=model,
                camera_key=camera_key,
                digest=digest,
            ),
            validate_staged_row=partial(
                validate_terminal_row,
                model=model,
                camera_key=camera_key,
                digest=digest,
            ),
        ),
        publication=CaptionPublicationBindings(
            validate_base_schema=partial(validate_episode_table, uri=uri),
            protected_field_names=tuple(EPISODE_SCHEMA.names),
            base_field_owner="multimodal-split",
            transaction_properties={"kind": "multimodal-caption", "result_variant": RESULT_VARIANT},
            validate_complete_fragment=partial(
                validate_complete_fragment,
                camera_key=camera_key,
                metadata_field_name=metadata_field,
            ),
        ),
    )
    inference_config = CaptionExecutionConfig(
        storage_profile=storage_profile,
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
    representative_media_uri = validate_episode_input(dataset, uri=uri, camera_key=camera_key)
    attempt = capture_attempt(dataset, bindings=bindings)
    workspace = resolve_workspace(
        config.output.staging_root_uri,
        caption_field,
        digest,
        storage_profile=storage_profile,
    )
    manifest: dict[str, object] = {
        "workspace_schema_version": WORKSPACE_SCHEMA_VERSION,
        "episodes_lance_uri": uri,
        "caption_field": caption_field,
        "metadata_field": metadata_field,
        "field_schemas": contract["fields"],
        "result_variant": RESULT_VARIANT,
        "camera_key": camera_key,
        "model": {
            "id": model.model_id,
            "revision": model.revision,
            "runtime_model_dir": str(model.runtime_model_dir),
        },
        "caption_contract": contract,
        "caption_contract_digest": digest,
        "checkpoint_adapter_version": CHECKPOINT_ADAPTER_VERSION,
    }
    workspace_preexisting = workspace_manifest_exists(workspace)
    logger.info(
        "Episode caption attempt v{} selected {} pending and {} complete fragment(s) for field {}",
        attempt.version,
        len(attempt.pending_fragment_ids),
        len(attempt.complete_fragment_ids),
        caption_field,
    )

    if attempt.pending_fragment_ids:
        ensure_workspace(workspace, storage_profile=storage_profile, manifest=manifest)
        phase_a_complete = phase_a_completion_covers(workspace, attempt, digest)
        model_dir = None
        if phase_a_complete:
            logger.info(
                "Phase A completion marker covers all {} pending fragment(s); skipping model setup and inference",
                len(attempt.pending_fragment_ids),
            )
        else:
            model_dir = validate_model_directory(model.runtime_model_dir, model_key=model.key)
        compatibility_locations = [uri, workspace.root_uri]
        if representative_media_uri is not None:
            compatibility_locations.append(representative_media_uri)
        ensure_ray_initialized(
            local_resources=curator_io_resources(),
            runtime_env_vars=s3_compatibility_env_vars(
                *compatibility_locations,
                storage_profile=storage_profile,
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
        ensure_workspace(workspace, storage_profile=storage_profile, manifest=manifest)

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
            cleanup_workspace(workspace, storage_profile=storage_profile)
        except Exception as exc:  # noqa: BLE001 -- cleanup is explicitly non-canonical best effort
            cleanup_complete = False
            logger.warning(
                "Episode caption rows are canonical at Lance v{}, but cleanup of {} failed: {!r}",
                latest_version,
                workspace.root_uri,
                exc,
            )

    summary: dict[str, object] = {
        "episodes_lance_uri": uri,
        "episodes_lance_version": latest_version,
        "attempt_version": attempt.version,
        "selected_fragments": len(attempt.selected_fragment_ids),
        "pending_fragments": len(attempt.pending_fragment_ids),
        "skipped_complete_fragments": len(attempt.complete_fragment_ids),
        "published_fragments": publication.published_fragments,
        "already_committed_fragments": publication.already_committed_fragments,
        "caption_field": caption_field,
        "metadata_field": metadata_field,
        "result_variant": RESULT_VARIANT,
        "camera_key": camera_key,
        "caption_contract_digest": digest,
        "cleanup_complete": cleanup_complete,
    }
    logger.info(
        "multimodal-caption finished at Lance v{}: {} fragment(s) published, {} already committed, cleanup_complete={}",
        latest_version,
        publication.published_fragments,
        publication.already_committed_fragments,
        cleanup_complete,
    )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    """Run from a strict JSON/YAML config path."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Curator Next multimodal-caption pipeline")
    parser.add_argument("config", help="Path to a JSON/YAML multimodal-caption config.")
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
