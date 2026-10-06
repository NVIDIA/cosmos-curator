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
from cosmos_curator.next.recipes.multimodal_embedding.config import (
    ResolvedMultimodalEmbeddingConfig,
    resolve_config,
)
from cosmos_curator.next.recipes.multimodal_embedding.contracts import (
    CameraEmbeddingPlan,
    EmbeddingRunPlan,
    normalized_embedding_contract,
    resolve_model,
    validate_cosmos_model_directory,
)
from cosmos_curator.next.recipes.multimodal_embedding.episode_input import (
    validate_episode_cameras,
)
from cosmos_curator.next.recipes.multimodal_embedding.inference import (
    run_inference_phase,
)
from cosmos_curator.next.recipes.multimodal_embedding.publication import (
    camera_bindings,
    publish_atomic_staged_results,
)
from cosmos_curator.next.recipes.video_caption.contracts import CHECKPOINT_ADAPTER_VERSION, WORKSPACE_SCHEMA_VERSION
from cosmos_curator.next.recipes.video_caption.lance_state import (
    CaptionAttempt,
    capture_attempt,
    ensure_caption_fields,
)
from cosmos_curator.next.recipes.video_caption.workspace import (
    cleanup_workspace,
    ensure_workspace,
    phase_a_completion_covers,
    resolve_workspace,
    workspace_manifest_exists,
)
from cosmos_curator.next.utils.arrow_utils import s3_compatibility_env_vars
from cosmos_curator.next.utils.identity import canonical_digest


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


def run_config(config: ResolvedMultimodalEmbeddingConfig) -> dict[str, Any]:  # noqa: C901 -- finite selection, staging and publication
    """Embed selected camera videos with one pipeline and one result commit."""
    if config.selection.retry_errors:
        msg = "selection.retry_errors is not supported yet. To recompute published errors, use a new output.result_key."
        raise ValueError(msg)
    uri, profile = config.input.episodes_lance_uri, config.execution.storage_profile
    storage_options = get_lance_storage_options(uri, profile_name=profile)
    dataset = lance.dataset(uri, storage_options=storage_options)
    keys = tuple(validate_episode_cameras(dataset, uri=uri, selection=config.video.camera_keys))
    model = resolve_model(config.model.variant)
    if not keys:
        # An empty snapshot has no camera namespaces to register or work to stage.
        return {
            "episodes_lance_uri": uri,
            "episodes_lance_version": int(dataset.version),
            "camera_keys": [],
            "selected_fragments": 0,
            "pending_fragments": 0,
            "skipped_complete_fragments": 0,
            "published_fragments": 0,
            "already_committed_fragments": 0,
            "cameras": {},
            "result_key": config.output.result_key,
            "base_result_key": config.output.result_key,
            "tag_camera_key": config.output.tag_camera_key,
            "model_key": model.key,
            "embedding_contract_digest": canonical_digest({"multi_camera_version": 1, "cameras": {}}),
            "cleanup_complete": True,
        }
    result_keys = {key: config.result_key_for_camera(key) for key in keys}
    cameras = {key: CameraEmbeddingPlan.resolve(model, camera_key=key, result_key=result_keys[key]) for key in keys}
    camera_guards = {key: camera_bindings(camera, model, uri=uri) for key, camera in cameras.items()}
    for binding in camera_guards.values():
        ensure_caption_fields(
            uri, storage_options=storage_options, bindings=binding, attempts=config.execution.commit_attempts
        )
    dataset = lance.dataset(uri, storage_options=storage_options)
    media_locations = validate_episode_cameras(dataset, uri=uri, selection=config.video.camera_keys)
    if tuple(media_locations) != keys:
        msg = "Selected camera set changed during field registration; rerun to resolve the current input snapshot"
        raise ValueError(msg)
    camera_attempts = {key: capture_attempt(dataset, bindings=binding) for key, binding in camera_guards.items()}
    pending_fragment_sets = {key: set(attempt.pending_fragment_ids) for key, attempt in camera_attempts.items()}
    fragment_ids = tuple(int(fragment.fragment_id) for fragment in dataset.get_fragments())
    pending_cameras = {
        fragment_id: tuple(key for key, pending_ids in pending_fragment_sets.items() if fragment_id in pending_ids)
        for fragment_id in fragment_ids
    }
    pending_cameras = {fragment_id: selected for fragment_id, selected in pending_cameras.items() if selected}
    plan = EmbeddingRunPlan(uri, config.output.result_key, model, tuple(cameras.values()), pending_cameras)
    # Every camera was validated on this pinned snapshot. Combine its captured
    # state without scanning and validating the same result columns again.
    attempt = CaptionAttempt(
        version=int(dataset.version),
        pending_fragment_ids=tuple(pending_cameras),
        complete_fragment_ids=tuple(fragment_id for fragment_id in fragment_ids if fragment_id not in pending_cameras),
    )
    workspace = resolve_workspace(
        config.output.staging_root_uri,
        plan.workspace_field,
        plan.digest,
        storage_profile=profile,
        policy_namespace=canonical_digest(config.selection.model_dump(mode="json")),
    )
    manifest = {
        "workspace_schema_version": WORKSPACE_SCHEMA_VERSION,
        "episodes_lance_uri": uri,
        "camera_contracts": {
            key: normalized_embedding_contract(model, camera_key=key, result_key=result_key)
            for key, result_key in result_keys.items()
        },
        "selection": config.selection.model_dump(mode="json"),
        "checkpoint_adapter_version": CHECKPOINT_ADAPTER_VERSION,
        "staging_vector_layout": "episode_camera_rows_v1",
    }
    preexisting = workspace_manifest_exists(workspace)
    if attempt.pending_fragment_ids or preexisting:
        ensure_workspace(workspace, storage_profile=profile, manifest=manifest)
    if attempt.pending_fragment_ids:
        complete = phase_a_completion_covers(workspace, attempt, plan.digest)
        if not complete:
            validate_cosmos_model_directory(model)
        ensure_ray_initialized(
            local_resources=curator_io_resources(),
            runtime_env_vars=s3_compatibility_env_vars(
                uri,
                workspace.root_uri,
                *[location for location in media_locations.values() if location is not None],
                storage_profile=profile,
            ),
        )
        configure_ray_data_progress(progress=config.execution.progress)
        configure_ray_data_stability(disable_high_memory_detector=True)
        if not complete:
            run_inference_phase(
                dataset,
                attempt,
                config,
                workspace,
                camera_results={key: (result_key, cameras[key].digest) for key, result_key in result_keys.items()},
                pending_cameras=pending_cameras,
                digest=plan.digest,
                storage_options=storage_options,
            )
    publication = publish_atomic_staged_results(
        uri,
        attempt,
        workspace,
        plan,
        storage_options=storage_options,
        commit_attempts=config.execution.commit_attempts,
    )
    latest = lance.dataset(uri, storage_options=storage_options)
    cleanup_complete = True
    if attempt.pending_fragment_ids or preexisting:
        try:
            cleanup_workspace(workspace, storage_profile=profile)
        except Exception as exc:  # noqa: BLE001 -- published rows remain canonical
            cleanup_complete = False
            logger.warning("Camera results committed, but workspace cleanup failed: {!r}", exc)
    summaries = {
        key: {
            "embedding_field": camera.vector_field,
            "metadata_field": camera.metadata_field,
            "result_key": result_keys[key],
            "effective_result_key": result_keys[key],
            "base_result_key": config.output.result_key,
            "tag_camera_key": config.output.tag_camera_key,
            **_terminal_counts(latest, attempt, camera.metadata_field),
        }
        for key, camera in cameras.items()
    }
    return {
        "episodes_lance_uri": uri,
        "episodes_lance_version": int(latest.version),
        "camera_keys": list(keys),
        "selected_fragments": len(attempt.selected_fragment_ids),
        "pending_fragments": len(attempt.pending_fragment_ids),
        "skipped_complete_fragments": len(attempt.complete_fragment_ids),
        "published_fragments": publication.published_fragments,
        "already_committed_fragments": publication.already_committed_fragments,
        "cameras": summaries,
        "result_key": config.output.result_key,
        "base_result_key": config.output.result_key,
        "tag_camera_key": config.output.tag_camera_key,
        "model_key": model.key,
        "embedding_contract_digest": plan.digest,
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
