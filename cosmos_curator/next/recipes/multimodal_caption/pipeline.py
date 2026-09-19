# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Curator Next ``multimodal-caption`` entry point."""

import argparse
import logging
from collections.abc import Mapping, Sequence
from functools import partial
from importlib.resources import files
from typing import Any, cast

import lance
import yaml
from loguru import logger

from cosmos_curator.core.sensors.sensors.ego_trajectory_sensor import DecodedEgoTrajectorySample
from cosmos_curator.core.sensors.utils.protobuf_mapper import ProtobufRowMapper
from cosmos_curator.core.utils.storage.storage_utils import get_lance_storage_options
from cosmos_curator.models.registry import RegisteredModel, validate_model_directory
from cosmos_curator.next.core.ray_runtime import (
    configure_ray_data_eager_actor_autoscaling,
    configure_ray_data_progress,
    configure_ray_data_stability,
    curator_io_resources,
    ensure_ray_initialized,
)
from cosmos_curator.next.recipes.multimodal_caption.config import ResolvedMultimodalCaptionConfig, resolve_config
from cosmos_curator.next.recipes.multimodal_caption.contracts import (
    EGO_DECODER_MAPPING_DIGEST,
    CaptionContextDefinition,
    CaptionInputProfile,
    EgoContextIdentity,
    build_terminal_result_row,
    caption_contract_digest,
    caption_inference_bindings,
    normalized_caption_contract,
    resolve_context_definition,
    resolve_ego_identity,
    resolve_model,
    resolve_prompt_template,
    result_field_names,
    result_fields,
    staged_result_schema,
    validate_terminal_row,
    validate_terminal_value,
)
from cosmos_curator.next.recipes.multimodal_caption.ego_context import EgoTrajectoryContextPolicy
from cosmos_curator.next.recipes.multimodal_caption.ego_context_adapter import append_ego_context_batch
from cosmos_curator.next.recipes.multimodal_caption.episode_input import (
    EGO_EPISODE_MAPPED_INPUT_SCHEMA,
    EGO_EPISODE_SOURCE_COLUMNS,
    EGO_EPISODE_SOURCE_SCHEMA,
    EGO_EPISODE_STAGED_BINDING_COLUMNS,
    EPISODE_MAPPED_INPUT_SCHEMA,
    EPISODE_SOURCE_COLUMNS,
    EPISODE_SOURCE_SCHEMA,
    EPISODE_STAGED_BINDING_COLUMNS,
    project_episode_camera,
    project_episode_camera_and_ego,
    validate_complete_fragment,
    validate_ego_episode_input,
    validate_episode_input,
)
from cosmos_curator.next.recipes.multimodal_split.lance_sink import validate_episode_table
from cosmos_curator.next.recipes.multimodal_split.records import EPISODE_SCHEMA
from cosmos_curator.next.recipes.video_caption.bindings import (
    CaptionInputBindings,
    CaptionPublicationBindings,
    CaptionResultBindings,
    CaptionRunBindings,
    ContextStage,
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
from cosmos_curator.next.utils.identity import canonical_digest

_EGO_MAPPING_RESOURCE = "ego_context_foxglove_pose_in_frame_mapping.yaml"


def _load_ego_mapping() -> dict[str, Any]:
    """Load and verify the one recipe-owned production decoder mapping."""
    resource = files("cosmos_curator.next.recipes.multimodal_caption").joinpath(_EGO_MAPPING_RESOURCE)
    loaded = yaml.safe_load(resource.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        msg = "EgoTrajectory decoder mapping must contain a top-level mapping"
        raise TypeError(msg)
    mapping = cast("dict[str, Any]", loaded)
    fields_mapping = mapping.get("fields")
    if not isinstance(fields_mapping, Mapping):
        msg = "EgoTrajectory decoder mapping must define fields"
        raise TypeError(msg)
    if fields_mapping.get("sensor_timestamp_ns") != {"from": "timestamp", "type": "protobuf_timestamp"}:
        msg = "EgoTrajectory sensor_timestamp_ns must map the embedded protobuf timestamp"
        raise ValueError(msg)
    if fields_mapping.get("align_timestamp_ns") != {
        "from": "$mcap.logtime",
        "type": "timestamp",
        "unit": "ns",
    }:
        msg = "EgoTrajectory align_timestamp_ns must map MCAP log_time as an ns timestamp"
        raise ValueError(msg)
    if fields_mapping.get("pose_valid") != {"default": True}:
        msg = "EgoTrajectory pose_valid must be exactly default: true"
        raise ValueError(msg)
    mapper = ProtobufRowMapper(mapping, target_cls=DecodedEgoTrajectorySample)
    if "align_timestamp_ns" not in mapper.mapped_target_names:
        msg = "EgoTrajectory decoder mapping must explicitly map align_timestamp_ns"
        raise ValueError(msg)
    digest = canonical_digest(mapping)
    if digest != EGO_DECODER_MAPPING_DIGEST:
        msg = f"EgoTrajectory decoder mapping digest mismatch: {digest}"
        raise ValueError(msg)
    return mapping


def _resolve_selected_context(
    config: ResolvedMultimodalCaptionConfig,
) -> tuple[CaptionContextDefinition, EgoContextIdentity | None, str]:
    """Resolve context, configured Ego source, and one exact unrendered prompt."""
    definition = resolve_context_definition(config.context.variant)
    ego_config = getattr(config.context, "ego", None)
    ego = resolve_ego_identity(
        definition,
        sensor_key=None if ego_config is None else ego_config.sensor_key,
        topic=None if ego_config is None else ego_config.topic,
        frame=None if ego_config is None else ego_config.frame,
    )
    prompt = config.inference.prompt
    prompt_template = resolve_prompt_template(
        definition,
        mode=prompt.mode,
        custom_template=getattr(prompt, "template", None),
    )
    return definition, ego, prompt_template


def _resolve_input_runtime(
    definition: CaptionContextDefinition,
    ego: EgoContextIdentity | None,
    *,
    camera_key: str,
    prompt_template: str,
    mapping: dict[str, Any] | None,
) -> tuple[CaptionInputBindings, ContextStage | None, tuple[str, ...]]:
    """Build physical input bindings from the selected context capability."""
    if definition.input_profile is CaptionInputProfile.CAMERA:
        return (
            CaptionInputBindings(
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
            None,
            (),
        )
    build_context = definition.ego_builder
    if ego is None or build_context is None or mapping is None:
        msg = "Ego context resolution did not produce its source, builder, and mapping"
        raise RuntimeError(msg)
    policy = EgoTrajectoryContextPolicy(
        representation_version=ego.representation_version,
        cadence_period_ns=ego.cadence_period_ns,
        max_delta_ns=ego.max_delta_ns,
    )
    return (
        CaptionInputBindings(
            source_columns=EGO_EPISODE_SOURCE_COLUMNS,
            source_schema=EGO_EPISODE_SOURCE_SCHEMA,
            mapped_schema=EGO_EPISODE_MAPPED_INPUT_SCHEMA,
            project_batch=partial(
                project_episode_camera_and_ego,
                camera_key=camera_key,
                ego_sensor_key=ego.sensor_key,
                max_episode_duration_ns=ego.max_episode_duration_ns,
            ),
            row_id_column="episode_id",
            media_uri_column="media_location",
            media_size_column="media_size_bytes",
            media_sha256_column="media_sha256",
            staged_binding_columns=EGO_EPISODE_STAGED_BINDING_COLUMNS,
        ),
        partial(
            append_ego_context_batch,
            schema_name=ego.schema_name,
            protobuf_mapping=mapping,
            topic=ego.topic,
            frame=ego.frame,
            policy=policy,
            prompt_template=prompt_template,
            build_context=build_context,
        ),
        ("context_digest",),
    )


def _preflight_and_register(
    config: ResolvedMultimodalCaptionConfig,
    definition: CaptionContextDefinition,
    ego: EgoContextIdentity | None,
    bindings: CaptionRunBindings,
    storage_options: dict[str, str] | None,
) -> tuple[lance.LanceDataset, str | None, str | None]:
    """Preflight every row before claiming fields, then reopen the registered table."""
    uri = config.input.episodes_lance_uri
    dataset = lance.dataset(uri, storage_options=storage_options)
    if definition.input_profile is CaptionInputProfile.EGO_TRAJECTORY:
        if ego is None:
            msg = "Ego input profile did not resolve an Ego source"
            raise RuntimeError(msg)
        representative_camera_uri, representative_ego_uri = validate_ego_episode_input(
            dataset,
            uri=uri,
            camera_key=config.context.camera_key,
            sensor_key=ego.sensor_key,
            max_episode_duration_ns=ego.max_episode_duration_ns,
        )
    else:
        representative_camera_uri = validate_episode_input(
            dataset,
            uri=uri,
            camera_key=config.context.camera_key,
        )
        representative_ego_uri = None
    ensure_caption_fields(
        uri,
        storage_options=storage_options,
        bindings=bindings,
        attempts=config.execution.commit_attempts,
    )
    return lance.dataset(uri, storage_options=storage_options), representative_camera_uri, representative_ego_uri


def _execution_config(config: ResolvedMultimodalCaptionConfig) -> CaptionExecutionConfig:
    execution = config.execution
    return CaptionExecutionConfig(
        storage_profile=execution.storage_profile,
        inference_concurrency=execution.inference_concurrency,
        inference_batch_size=execution.inference_batch_size,
        max_concurrent_batches=execution.max_concurrent_batches,
        tensor_parallel_size=execution.tensor_parallel_size,
        pipeline_parallel_size=execution.pipeline_parallel_size,
        media_concurrency=execution.media_concurrency,
        media_batch_size=execution.media_batch_size,
        media_cpus=execution.media_cpus,
        media_attempts=execution.media_attempts,
        lance_read_batch_size=execution.lance_read_batch_size,
        parquet_rows_per_file=execution.parquet_rows_per_file,
    )


def _workspace_manifest(  # noqa: PLR0913 -- serialized compatibility identity is explicit
    *,
    uri: str,
    caption_field: str,
    metadata_field: str,
    contract: dict[str, Any],
    definition: CaptionContextDefinition,
    result_key: str,
    camera_key: str,
    model: RegisteredModel,
    digest: str,
) -> dict[str, object]:
    return {
        "workspace_schema_version": WORKSPACE_SCHEMA_VERSION,
        "episodes_lance_uri": uri,
        "caption_field": caption_field,
        "metadata_field": metadata_field,
        "field_schemas": contract["fields"],
        "context_variant": definition.variant,
        "result_key": result_key,
        "camera_key": camera_key,
        "model_key": model.key,
        "model": {
            "id": model.model_id,
            "revision": model.revision,
            "precision": model.precision,
            "runtime_model_dir": str(model.runtime_model_dir),
        },
        "caption_contract": contract,
        "caption_contract_digest": digest,
        "checkpoint_adapter_version": CHECKPOINT_ADAPTER_VERSION,
    }


def _attempt_terminal_counts(
    dataset: lance.LanceDataset,
    fragment_ids: tuple[int, ...],
    metadata_field: str,
) -> dict[str, int]:
    """Count terminal outcomes only for the finite fragments selected by this attempt."""
    counts = {"success": 0, "truncated": 0, "error": 0}
    for fragment_id in fragment_ids:
        fragment = dataset.get_fragment(fragment_id)
        if fragment is None:
            msg = f"Selected caption fragment {fragment_id} is absent after publication"
            raise RuntimeError(msg)
        for batch in fragment.scanner(columns=[metadata_field]).to_batches():
            for row in batch.to_pylist():
                metadata = row[metadata_field]
                if not isinstance(metadata, dict) or metadata.get("status") not in counts:
                    msg = f"Selected caption fragment {fragment_id} has no valid terminal status after publication"
                    raise ValueError(msg)
                counts[str(metadata["status"])] += 1
    return counts


def run_config(config: ResolvedMultimodalCaptionConfig) -> dict[str, object]:  # noqa: PLR0915
    """Run one finite episode-caption attempt and return its canonical outcome."""
    uri = config.input.episodes_lance_uri
    storage_profile = config.execution.storage_profile
    storage_options = get_lance_storage_options(uri, profile_name=storage_profile)
    definition, ego, prompt_template = _resolve_selected_context(config)
    mapping = _load_ego_mapping() if definition.input_profile is CaptionInputProfile.EGO_TRAJECTORY else None
    model = resolve_model(config.model.variant)
    camera_key = config.context.camera_key
    result_key = config.output.result_key
    caption_field, metadata_field = result_field_names(definition.variant, result_key, model.key)
    contract = normalized_caption_contract(model, camera_key, definition, ego, prompt_template, result_key)
    digest = caption_contract_digest(model, camera_key, definition, ego, prompt_template, result_key)
    input_bindings, context_stage, carried_result_columns = _resolve_input_runtime(
        definition,
        ego,
        camera_key=camera_key,
        prompt_template=prompt_template,
        mapping=mapping,
    )
    bindings = CaptionRunBindings(
        input=input_bindings,
        result=CaptionResultBindings(
            caption_field_name=caption_field,
            metadata_field_name=metadata_field,
            fields=result_fields(definition.variant, result_key, model.key, ego=ego, digest=digest),
            staged_schema=staged_result_schema(definition.variant, result_key, model.key, ego=ego, digest=digest),
            digest=digest,
            build_terminal_row=partial(
                build_terminal_result_row,
                model=model,
                definition=definition,
                ego=ego,
                prompt_template=prompt_template,
                result_key=result_key,
                digest=digest,
            ),
            validate_terminal_value=partial(
                validate_terminal_value,
                model=model,
                camera_key=camera_key,
                definition=definition,
                ego=ego,
                prompt_template=prompt_template,
                result_key=result_key,
                digest=digest,
            ),
            validate_staged_row=partial(
                validate_terminal_row,
                model=model,
                camera_key=camera_key,
                definition=definition,
                ego=ego,
                prompt_template=prompt_template,
                result_key=result_key,
                digest=digest,
            ),
            carried_result_columns=carried_result_columns,
        ),
        publication=CaptionPublicationBindings(
            validate_base_schema=partial(validate_episode_table, uri=uri),
            protected_field_names=tuple(EPISODE_SCHEMA.names),
            base_field_owner="multimodal-split",
            transaction_properties={
                "kind": "multimodal-caption",
                "result_key": result_key,
                "context_variant": definition.variant,
                "model_key": model.key,
                "caption_contract_digest": digest,
            },
            validate_complete_fragment=partial(
                validate_complete_fragment,
                camera_key=camera_key,
                metadata_field_name=metadata_field,
                ego_sensor_key=None if ego is None else ego.sensor_key,
            ),
        ),
        prompt_template=prompt_template,
        inference=caption_inference_bindings(model.key),
        context_stage=context_stage,
    )
    dataset, representative_camera_uri, representative_ego_uri = _preflight_and_register(
        config,
        definition,
        ego,
        bindings,
        storage_options,
    )
    attempt = capture_attempt(dataset, bindings=bindings)
    workspace = resolve_workspace(
        config.output.staging_root_uri,
        caption_field,
        digest,
        storage_profile=storage_profile,
    )
    manifest = _workspace_manifest(
        uri=uri,
        caption_field=caption_field,
        metadata_field=metadata_field,
        contract=contract,
        definition=definition,
        result_key=result_key,
        camera_key=camera_key,
        model=model,
        digest=digest,
    )
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
        if representative_camera_uri is not None:
            compatibility_locations.append(representative_camera_uri)
        if representative_ego_uri is not None:
            compatibility_locations.append(representative_ego_uri)
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
                _execution_config(config),
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
    latest_dataset = lance.dataset(uri, storage_options=storage_options)
    latest_version = int(latest_dataset.version)
    terminal_counts = _attempt_terminal_counts(latest_dataset, attempt.selected_fragment_ids, metadata_field)
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
        "terminal_counts": terminal_counts,
        "caption_field": caption_field,
        "metadata_field": metadata_field,
        "result_key": result_key,
        "context_variant": definition.variant,
        "camera_key": camera_key,
        "model_key": model.key,
        "model_id": model.model_id,
        "model_revision": model.revision,
        "model_precision": model.precision,
        "prompt_mode": config.inference.prompt.mode,
        "prompt_template": prompt_template,
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
