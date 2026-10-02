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

"""Curator Next multimodal-embedding episode-to-Lance recipe."""

import argparse
import logging
from collections.abc import Sequence
from functools import partial
from typing import Any

import lance
from loguru import logger

from cosmos_curator.core.utils.storage.storage_utils import get_lance_storage_options
from cosmos_curator.next.core.ray_runtime import (
    configure_ray_data_progress,
    configure_ray_data_stability,
    curator_io_resources,
    ensure_ray_initialized,
)
from cosmos_curator.next.recipes.multimodal_embedding.config import ResolvedMultimodalEmbeddingConfig, resolve_config
from cosmos_curator.next.recipes.multimodal_embedding.contracts import (
    VideoEmbeddingModelSpec,
    build_terminal_result_row,
    embedding_contract_digest,
    normalized_embedding_contract,
    resolve_model,
    result_field_names,
    result_fields,
    staged_result_schema,
    validate_cosmos_model_directory,
    validate_terminal_batch,
    validate_terminal_row,
)
from cosmos_curator.next.recipes.multimodal_embedding.episode_input import (
    EPISODE_MAPPED_INPUT_SCHEMA,
    EPISODE_SOURCE_COLUMNS,
    EPISODE_SOURCE_SCHEMA,
    EPISODE_STAGED_BINDING_COLUMNS,
    project_episode_camera,
    validate_complete_fragment,
    validate_episode_input,
)
from cosmos_curator.next.recipes.multimodal_embedding.inference import run_inference_phase
from cosmos_curator.next.recipes.multimodal_embedding.publication import (
    embedding_result_fingerprint,
    publish_atomic_staged_results,
    reconcile_embedding_replays,
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
from cosmos_curator.next.recipes.video_caption.lance_state import CaptionAttempt, capture_attempt, ensure_caption_fields
from cosmos_curator.next.recipes.video_caption.workspace import (
    cleanup_workspace,
    ensure_workspace,
    phase_a_completion_covers,
    resolve_workspace,
    workspace_manifest_exists,
)
from cosmos_curator.next.utils.arrow_utils import s3_compatibility_env_vars
from cosmos_curator.next.utils.identity import canonical_digest


def _bindings(config: ResolvedMultimodalEmbeddingConfig, model: VideoEmbeddingModelSpec) -> CaptionRunBindings:
    uri = config.input.episodes_lance_uri
    camera_key = config.video.camera_key
    result_key = config.output.result_key
    digest = embedding_contract_digest(model, camera_key=camera_key, result_key=result_key)
    vector_field, metadata_field = result_field_names(result_key, model_key=model.key)
    return CaptionRunBindings(
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
            caption_field_name=vector_field,
            metadata_field_name=metadata_field,
            fields=result_fields(result_key, model_key=model.key, digest=digest),
            staged_schema=staged_result_schema(result_key, model_key=model.key, digest=digest),
            digest=digest,
            build_terminal_row=partial(
                build_terminal_result_row,
                model=model,
                result_key=result_key,
                digest=digest,
            ),
            validate_terminal_value=lambda vector, metadata: validate_terminal_row(
                {vector_field: vector, metadata_field: metadata},
                model=model,
                camera_key=camera_key,
                result_key=result_key,
                digest=digest,
            ),
            validate_staged_row=partial(
                validate_terminal_row,
                model=model,
                camera_key=camera_key,
                result_key=result_key,
                digest=digest,
            ),
            validate_terminal_batch=partial(
                validate_terminal_batch,
                model=model,
                camera_key=camera_key,
                result_key=result_key,
                digest=digest,
            ),
            reconcile_staged_rows=partial(
                reconcile_embedding_replays,
                metadata_field_name=metadata_field,
                validate_staged_row=partial(
                    validate_terminal_row,
                    model=model,
                    camera_key=camera_key,
                    result_key=result_key,
                    digest=digest,
                ),
            ),
            fingerprint_result_values=partial(
                embedding_result_fingerprint, vector_field_name=vector_field, metadata_field_name=metadata_field
            ),
        ),
        publication=CaptionPublicationBindings(
            validate_base_schema=partial(validate_episode_table, uri=uri),
            protected_field_names=tuple(EPISODE_SCHEMA.names),
            base_field_owner="multimodal-split",
            transaction_properties={
                "kind": "multimodal-embedding",
                "result_key": result_key,
                "model_key": model.key,
                "embedding_contract_digest": digest,
            },
            validate_complete_fragment=partial(
                validate_complete_fragment,
                camera_key=camera_key,
                metadata_field_name=metadata_field,
            ),
        ),
        # Required by the shared binding type; embedding execution does not consume caption settings.
        prompt_template="",
    )


def _workspace_manifest(
    config: ResolvedMultimodalEmbeddingConfig,
    model: VideoEmbeddingModelSpec,
    bindings: CaptionRunBindings,
) -> dict[str, object]:
    return {
        "workspace_schema_version": WORKSPACE_SCHEMA_VERSION,
        "episodes_lance_uri": config.input.episodes_lance_uri,
        "embedding_field": bindings.result.caption_field_name,
        "metadata_field": bindings.result.metadata_field_name,
        "embedding_contract": normalized_embedding_contract(
            model, camera_key=config.video.camera_key, result_key=config.output.result_key
        ),
        "embedding_contract_digest": bindings.result.digest,
        "selection": config.selection.model_dump(mode="json"),
        "checkpoint_adapter_version": CHECKPOINT_ADAPTER_VERSION,
        "staging_vector_layout": "nullable_list_v1",
    }


def _terminal_counts(dataset: lance.LanceDataset, attempt: CaptionAttempt, metadata_field: str) -> dict[str, object]:
    counts = {"success": 0, "error": 0, "unsupported_duration": 0}
    errors: dict[str, int] = {}
    for fragment_id in attempt.selected_fragment_ids:
        fragment = dataset.get_fragment(fragment_id)
        if fragment is None:
            msg = f"Selected embedding fragment {fragment_id} disappeared after publication"
            raise RuntimeError(msg)
        for batch in fragment.scanner(columns=[metadata_field]).to_batches():
            for row in batch.to_pylist():
                metadata = row[metadata_field]
                if not isinstance(metadata, dict) or metadata.get("status") not in counts:
                    msg = f"Selected embedding fragment {fragment_id} lacks terminal metadata"
                    raise ValueError(msg)
                status = str(metadata["status"])
                counts[status] += 1
                if status == "error":
                    code = str(metadata["error_code"])
                    errors[code] = errors.get(code, 0) + 1
    return {"terminal_counts": counts, "error_codes": errors}


def run_config(config: ResolvedMultimodalEmbeddingConfig) -> dict[str, Any]:
    """Run one finite embedding attempt and publish canonical episode results."""
    if config.selection.retry_errors:
        msg = "selection.retry_errors is not supported yet. To recompute published errors, use a new output.result_key."
        raise ValueError(msg)
    uri = config.input.episodes_lance_uri
    profile = config.execution.storage_profile
    storage_options = get_lance_storage_options(uri, profile_name=profile)
    model = resolve_model(config.model.variant)
    bindings = _bindings(config, model)
    dataset = lance.dataset(uri, storage_options=storage_options)
    validate_episode_input(dataset, uri=uri, camera_key=config.video.camera_key)
    ensure_caption_fields(
        uri, storage_options=storage_options, bindings=bindings, attempts=config.execution.commit_attempts
    )
    dataset = lance.dataset(uri, storage_options=storage_options)
    # Registration can race with producer appends. Validate the exact immutable
    # snapshot used for selection, including both pending and completed rows.
    representative_media = validate_episode_input(dataset, uri=uri, camera_key=config.video.camera_key)
    attempt = capture_attempt(dataset, bindings=bindings)
    policy_namespace = canonical_digest(config.selection.model_dump(mode="json"))
    workspace = resolve_workspace(
        config.output.staging_root_uri,
        bindings.result.caption_field_name,
        bindings.result.digest,
        storage_profile=profile,
        policy_namespace=policy_namespace,
    )
    manifest = _workspace_manifest(config, model, bindings)
    workspace_preexisting = workspace_manifest_exists(workspace)
    if attempt.pending_fragment_ids:
        ensure_workspace(workspace, storage_profile=profile, manifest=manifest)
        complete = phase_a_completion_covers(workspace, attempt, bindings.result.digest)
        if not complete:
            validate_cosmos_model_directory(model)
        locations = [uri, workspace.root_uri]
        if representative_media is not None:
            locations.append(representative_media)
        ensure_ray_initialized(
            local_resources=curator_io_resources(),
            runtime_env_vars=s3_compatibility_env_vars(*locations, storage_profile=profile),
        )
        configure_ray_data_progress(progress=config.execution.progress)
        configure_ray_data_stability(disable_high_memory_detector=True)
        if not complete:
            run_inference_phase(
                dataset,
                attempt,
                config,
                workspace,
                digest=bindings.result.digest,
                storage_options=storage_options,
            )
    elif workspace_preexisting:
        ensure_workspace(workspace, storage_profile=profile, manifest=manifest)
    publication = publish_atomic_staged_results(
        uri,
        attempt,
        workspace,
        bindings,
        storage_options=storage_options,
        commit_attempts=config.execution.commit_attempts,
    )
    latest = lance.dataset(uri, storage_options=storage_options)
    terminal = _terminal_counts(latest, attempt, bindings.result.metadata_field_name)
    cleanup_complete = True
    if attempt.pending_fragment_ids or workspace_preexisting:
        try:
            cleanup_workspace(workspace, storage_profile=profile)
        except Exception as exc:  # noqa: BLE001 -- published Lance rows remain canonical
            cleanup_complete = False
            logger.warning("Embedding publication committed, but workspace cleanup failed: {!r}", exc)
    return {
        "episodes_lance_uri": uri,
        "episodes_lance_version": int(latest.version),
        "attempt_version": attempt.version,
        "selected_fragments": len(attempt.selected_fragment_ids),
        "pending_fragments": len(attempt.pending_fragment_ids),
        "skipped_complete_fragments": len(attempt.complete_fragment_ids),
        "published_fragments": publication.published_fragments,
        "already_committed_fragments": publication.already_committed_fragments,
        **terminal,
        "embedding_field": bindings.result.caption_field_name,
        "metadata_field": bindings.result.metadata_field_name,
        "result_key": config.output.result_key,
        "camera_key": config.video.camera_key,
        "model_key": model.key,
        "embedding_contract_digest": bindings.result.digest,
        "cleanup_complete": cleanup_complete,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run from one strict JSON or YAML recipe config."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Curator Next multimodal-embedding pipeline")
    parser.add_argument("config", help="Path to a JSON/YAML multimodal-embedding config")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="PATH=VALUE")
    args = parser.parse_args(argv)
    run_config(resolve_config(args.config, overrides=args.overrides))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
