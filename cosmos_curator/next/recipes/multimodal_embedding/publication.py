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

"""Publish staged multimodal video embeddings in one guarded Lance transaction.

Fragment preparation and input guards reuse the existing publication engine.
This recipe owns its single-update boundary, transaction receipts, and vector
fingerprints. Atomic visibility and concurrent-writer safety depend on Lance's
commit handler and the backend's atomic create-if-absent operations. Native AWS
S3 supplies conditional writes; local filesystems must supply equivalent atomic
creation or rename operations. Custom S3 endpoints and shared mounts require
deployment-specific qualification. A storage profile selects connection settings,
not a verified commit capability. This recipe supplies no external commit lock.
See docs/curator/design/multimodal-av-video-embedding.md for backend requirements
and the distinction between backend guarantees and deployment qualification.
"""

import time
from collections.abc import Iterator
from dataclasses import replace
from functools import partial
from typing import Any, Never, cast

import lance
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as pads
import ray
from lance.fragment import FragmentMetadata, LanceFragment
from ray.data.expressions import col

from cosmos_curator.next.recipes.multimodal_embedding.contracts import (
    CameraEmbeddingPlan,
    EmbeddingRunPlan,
    VideoEmbeddingModelSpec,
    build_terminal_result_row,
    validate_terminal_batch,
    validate_terminal_row,
)
from cosmos_curator.next.recipes.multimodal_embedding.episode_input import (
    CAMERA_WORK_ID,
    EPISODE_MAPPED_INPUT_SCHEMA,
    EPISODE_PUBLICATION_SCHEMA,
    EPISODE_SOURCE_COLUMNS,
    EPISODE_SOURCE_SCHEMA,
    EPISODE_STAGED_BINDING_COLUMNS,
    camera_work_id,
    project_episode_camera,
    project_episode_publication,
    validate_complete_fragment,
)
from cosmos_curator.next.recipes.multimodal_embedding.inference import camera_staged_schema
from cosmos_curator.next.recipes.multimodal_split.lance_sink import validate_episode_table
from cosmos_curator.next.recipes.multimodal_split.records import EPISODE_SCHEMA
from cosmos_curator.next.recipes.video_caption import publication as fragment_publication
from cosmos_curator.next.recipes.video_caption.bindings import (
    CaptionInputBindings,
    CaptionPublicationBindings,
    CaptionResultBindings,
    CaptionRunBindings,
)
from cosmos_curator.next.recipes.video_caption.inference import staged_result_files
from cosmos_curator.next.recipes.video_caption.lance_state import (
    CaptionAttempt,
    FragmentCaptionState,
    classify_fragment,
    fragment_fingerprint,
    fragment_metadata_json,
    leaf_field_ids,
)
from cosmos_curator.next.recipes.video_caption.publication import (
    AttemptFragmentGuard,
    PreparedFragmentPublication,
    PreparedFragmentUpdate,
    PublicationError,
    PublicationSummary,
)
from cosmos_curator.next.recipes.video_caption.workspace import CaptionWorkspace
from cosmos_curator.next.utils.identity import canonical_digest

_ATOMIC_PUBLICATION_KEY = "cosmos_curator.atomic_publication"
_RECEIPT_READ_ATTEMPTS = 3
_RECEIPT_BACKOFF_S = 0.25


def camera_bindings(camera: CameraEmbeddingPlan, model: VideoEmbeddingModelSpec, *, uri: str) -> CaptionRunBindings:
    """Adapt one explicit camera plan to the unchanged shared Lance guard helpers."""
    camera_key, result_key, digest = camera.camera_key, camera.result_key, camera.digest
    vector_field, metadata_field = camera.vector_field, camera.metadata_field
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
            fields=camera.fields,
            staged_schema=camera.staged_schema,
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


def _multi_fragment_has_pending(fragment: LanceFragment, *, cameras: dict[str, CaptionRunBindings]) -> bool:
    # Validate every camera, including completed cameras in a partially pending fragment.
    states = [classify_fragment(fragment, bindings=bindings) for bindings in cameras.values()]
    return FragmentCaptionState.PENDING in states


def _validate_multi_batch(
    table: pa.Table, *, cameras: tuple[CameraEmbeddingPlan, ...], model: VideoEmbeddingModelSpec
) -> None:
    for camera in cameras:
        camera_key, result_key = camera.camera_key, camera.result_key
        validate_terminal_batch(
            table,
            model=model,
            camera_key=camera_key,
            result_key=result_key,
            digest=camera.digest,
        )


def _validate_multi_row(
    row: dict[str, Any], *, cameras: tuple[CameraEmbeddingPlan, ...], model: VideoEmbeddingModelSpec
) -> None:
    for camera in cameras:
        camera_key, result_key = camera.camera_key, camera.result_key
        fields = camera.fields.names
        values = {name: row[name] for name in fields}
        values.update(
            {
                name: row[name]
                for name in ("episode_id", "materialization_digest", "start_ns", "exclusive_end_ns")
                if name in row
            }
        )
        validate_terminal_row(
            values,
            model=model,
            camera_key=camera_key,
            result_key=result_key,
            digest=camera.digest,
        )


def _prepare_camera_rows(
    staged: pa.Table,
    fragment: LanceFragment,
    *,
    cameras: tuple[CameraEmbeddingPlan, ...],
    model: VideoEmbeddingModelSpec,
    pending_cameras: dict[int, tuple[str, ...]],
) -> pa.Table:
    """Validate/reconcile pair checkpoints, carry completed cameras, and assemble exact episode rows."""
    camera_plans = {camera.camera_key: camera for camera in cameras}
    fields = [name for camera in cameras for name in camera.fields.names]
    source = fragment.scanner(columns=[*EPISODE_SOURCE_COLUMNS, *fields], with_row_address=True).to_table()
    existing = source.to_pylist()
    projections = {key: project_episode_camera(source, camera_key=key).to_pylist() for key in camera_plans}
    output = project_episode_publication(source).to_pylist()
    expected = {
        (index, key) for index in range(len(output)) for key in pending_cameras.get(int(fragment.fragment_id), ())
    }
    selected: dict[tuple[int, str], dict[str, Any]] = {}
    ranks: dict[tuple[int, str], tuple[int, str]] = {}
    for row in staged.to_pylist():
        pair = (int(row["row_offset"]), str(row["camera_key"]))
        index, camera_key = pair
        if (
            camera_key not in camera_plans
            or not 0 <= index < len(output)
            or int(row["fragment_id"]) != int(fragment.fragment_id)
        ):
            msg = "Camera staging contains an unexpected episode/camera pair"
            raise ValueError(msg)
        current = projections[camera_key][index]
        if any(row[name] != current[name] for name in EPISODE_MAPPED_INPUT_SCHEMA.names) or row[
            CAMERA_WORK_ID
        ] != camera_work_id(current["episode_id"], camera_key):
            msg = "Camera staging does not match its pinned episode/media binding"
            raise ValueError(msg)
        camera = camera_plans[camera_key]
        result_key = camera.result_key
        vector_name, metadata_name = camera.vector_field, camera.metadata_field
        values = {vector_name: row["embedding"], metadata_name: row["embedding_metadata"]}
        validate_terminal_row(
            {**current, **values},
            model=model,
            camera_key=camera_key,
            result_key=result_key,
            digest=camera.digest,
        )
        if pair not in expected:
            # A fresh attempt can capture cameras published while this workspace was interrupted.
            # These cameras were already complete at capture; active competing writers are rejected.
            validate_terminal_row(
                {**current, **{name: existing[index][name] for name in (vector_name, metadata_name)}},
                model=model,
                camera_key=camera_key,
                result_key=result_key,
                digest=camera.digest,
            )
            continue
        rank = (0 if row["embedding_metadata"]["status"] == "success" else 1, canonical_digest(row))
        if pair not in ranks or rank < ranks[pair]:
            ranks[pair], selected[pair] = rank, values
    if set(selected) != expected:
        msg = "Camera staging does not cover every selected episode/camera pair"
        raise ValueError(msg)
    for index, row in enumerate(output):
        for camera_key, camera in camera_plans.items():
            pair = (index, camera_key)
            names = camera.fields.names
            row.update(selected[pair] if pair in selected else {name: existing[index][name] for name in names})
    schema_fields = [*EPISODE_PUBLICATION_SCHEMA]
    for camera in cameras:
        schema_fields.extend(list(camera.staged_schema)[len(EPISODE_MAPPED_INPUT_SCHEMA) :])
    return pa.Table.from_pylist(output, schema=pa.schema(schema_fields))


def _multi_result_fingerprint(table: pa.Table, *, cameras: tuple[CameraEmbeddingPlan, ...]) -> str:
    """Bind the receipt to every camera's logical values, including null-vector metadata."""
    return canonical_digest(
        {
            camera.camera_key: embedding_result_fingerprint(
                table.select(["episode_id", camera.vector_field, camera.metadata_field]),
                vector_field_name=camera.vector_field,
                metadata_field_name=camera.metadata_field,
            )
            for camera in cameras
        }
    )


def _validate_completed_cameras(fragment: LanceFragment, *, cameras: tuple[CameraEmbeddingPlan, ...]) -> None:
    for camera in cameras:
        validate_complete_fragment(fragment, camera_key=camera.camera_key, metadata_field_name=camera.metadata_field)


def _aggregate_terminal_row(*_args: object, **_kwargs: object) -> Never:
    """Aggregate guards never build or validate a terminal value for just one camera."""
    msg = "An aggregate embedding plan requires per-camera terminal rows or a complete publication batch"
    raise TypeError(msg)


def publication_bindings(plan: EmbeddingRunPlan) -> CaptionRunBindings:
    """Construct a narrow shared-guard adapter explicitly from the aggregate plan.

    Field-name anchors are retained only for workspace/log/receipt compatibility.
    Episode projection, terminal validation, completion checks, and ownership are
    aggregate; no first-camera callbacks or caption inference settings are copied.
    """
    cameras = {
        camera.camera_key: camera_bindings(camera, plan.model, uri=plan.episodes_lance_uri) for camera in plan.cameras
    }
    return CaptionRunBindings(
        input=CaptionInputBindings(
            source_columns=EPISODE_SOURCE_COLUMNS,
            source_schema=EPISODE_SOURCE_SCHEMA,
            mapped_schema=EPISODE_PUBLICATION_SCHEMA,
            project_batch=project_episode_publication,
            row_id_column="episode_id",
            # Publication does not fetch media; per-camera guards own media identity.
            media_uri_column="",
            media_size_column="",
            media_sha256_column=None,
            staged_binding_columns=tuple(EPISODE_PUBLICATION_SCHEMA.names[2:]),
        ),
        result=CaptionResultBindings(
            caption_field_name=plan.workspace_field,
            metadata_field_name=f"{plan.workspace_field}_metadata",
            fields=plan.fields,
            staged_schema=plan.staged_schema,
            digest=plan.digest,
            build_terminal_row=_aggregate_terminal_row,
            validate_terminal_value=_aggregate_terminal_row,
            validate_staged_row=partial(_validate_multi_row, cameras=plan.cameras, model=plan.model),
            validate_terminal_batch=partial(_validate_multi_batch, cameras=plan.cameras, model=plan.model),
            fingerprint_result_values=partial(_multi_result_fingerprint, cameras=plan.cameras),
            fragment_has_pending=partial(_multi_fragment_has_pending, cameras=cameras),
            staged_storage_schema=camera_staged_schema(plan.model),
            prepare_staged_rows=partial(
                _prepare_camera_rows, cameras=plan.cameras, model=plan.model, pending_cameras=plan.pending_cameras
            ),
        ),
        publication=CaptionPublicationBindings(
            validate_base_schema=partial(validate_episode_table, uri=plan.episodes_lance_uri),
            protected_field_names=tuple(EPISODE_SCHEMA.names),
            base_field_owner="multimodal-split",
            transaction_properties={
                "kind": "multimodal-embedding",
                "result_key": plan.base_result_key,
                "model_key": plan.model.key,
                "embedding_contract_digest": plan.digest,
            },
            validate_complete_fragment=partial(_validate_completed_cameras, cameras=plan.cameras),
        ),
        prompt_template="",
    )


def _parse_camera_publication(payload: str, fragment_id: int) -> PreparedFragmentPublication:
    return fragment_publication._parse_prepared_publication(payload, expected_fragment_id=fragment_id)  # noqa: SLF001


def _iter_camera_publications(dataset: ray.data.Dataset) -> Iterator[PreparedFragmentPublication]:
    try:
        for batch in dataset.iter_batches(prefetch_batches=0, batch_size=None, batch_format="pyarrow"):
            table = batch if isinstance(batch, pa.Table) else pa.Table.from_batches([batch])
            for row in table.to_pylist():
                yield _parse_camera_publication(str(row["payload"]), int(row["fragment_id"]))
    except Exception as exc:
        publication_error = fragment_publication._find_publication_error(exc)  # noqa: SLF001
        if publication_error is not None:
            raise publication_error from exc
        raise


def _prepare_camera_fragment(  # noqa: PLR0913 -- worker preparation pins source and current snapshots
    staged: pa.Table,
    *,
    uri: str,
    attempt_version: int,
    current_version: int,
    plan: EmbeddingRunPlan,
    storage_options: dict[str, str] | None,
) -> pa.Table:
    """Prepare captured pending camera columns; reject overlapping result writes."""
    bindings = publication_bindings(plan)

    fragment_ids = pc.unique(staged["fragment_id"]).to_pylist()
    if len(fragment_ids) != 1 or fragment_ids[0] is None:
        msg = "Camera publication requires exactly one fragment ID"
        raise PublicationError(msg)
    fragment_id = int(fragment_ids[0])
    original = lance.dataset(uri, version=attempt_version, storage_options=storage_options)
    current = lance.dataset(uri, version=current_version, storage_options=storage_options)
    fragment_publication.assert_nullable_schema_extension(original.schema, current.schema)
    # Reuse internal engine helpers without changing caption's publication API.
    guard = fragment_publication._capture_fragment_guards(original, (fragment_id,), bindings)[fragment_id]  # noqa: SLF001
    fragment = fragment_publication.require_fragment(current, fragment_id)
    fragment_publication.assert_pending_identity(current, fragment, guard, bindings)
    pending_fields = [
        field
        for camera in plan.cameras
        if camera.camera_key in plan.pending_cameras.get(fragment_id, ())
        for field in camera.fields
    ]
    if not pending_fields:
        msg = f"No captured pending camera fields for fragment {fragment_id}"
        raise PublicationError(msg)
    prepare_rows = bindings.result.prepare_staged_rows
    assert prepare_rows is not None
    table = fragment_publication._validate_staged_group(prepare_rows(staged, fragment), fragment, guard, bindings)  # noqa: SLF001
    fingerprint = bindings.result.fingerprint_result_values
    assert fingerprint is not None
    pending_bindings = replace(bindings, result=replace(bindings.result, fields=pa.schema(pending_fields)))
    descriptor = fragment_publication.prepare_fragment_update(current, fragment, table, pending_bindings)
    result_schema = pa.schema(
        [bindings.input.mapped_schema.field(bindings.input.row_id_column), *bindings.result.fields]
    )
    prepared = PreparedFragmentPublication(
        guard, fingerprint(table.select(result_schema.names).cast(result_schema)), descriptor
    )
    payload = fragment_publication._serialize_prepared_publication(prepared)  # noqa: SLF001
    return pa.Table.from_pylist(
        [{"fragment_id": fragment_id, "payload": payload}],
        schema=fragment_publication.PREPARED_PUBLICATION_SCHEMA,
    )


def _prepare_camera_publications(  # noqa: PLR0913 -- Ray preparation uses immutable versioned snapshots
    uri: str,
    attempt: CaptionAttempt,
    workspace: CaptionWorkspace,
    plan: EmbeddingRunPlan,
    *,
    current_version: int,
    storage_options: dict[str, str] | None,
) -> ray.data.Dataset:
    """Prepare publication from the explicit embedding run plan."""
    bindings = publication_bindings(plan)

    files = staged_result_files(workspace)
    if not files:
        msg = "No staged camera results exist for the pending fragments"
        raise PublicationError(msg)
    fragment_publication._validate_parquet_schemas_distributed(files, workspace, bindings)  # noqa: SLF001
    staged = ray.data.read_parquet(
        list(files),
        filesystem=workspace.filesystem,
        schema=bindings.result.staged_storage_schema,
        override_num_blocks=max(len(files), len(attempt.pending_fragment_ids)),
    )
    return (
        staged.filter(expr=col("fragment_id").is_in(list(attempt.pending_fragment_ids)))
        .groupby("fragment_id", num_partitions=len(attempt.pending_fragment_ids))
        .map_groups(
            cast("Any", _prepare_camera_fragment),
            batch_format="pyarrow",
            zero_copy_batch=True,
            fn_kwargs={
                "uri": uri,
                "attempt_version": attempt.version,
                "current_version": current_version,
                "plan": plan,
                "storage_options": storage_options,
            },
            num_cpus=1,
            memory=8 * 1024**3,
        )
    )


def _reprepare_camera_fragment(  # noqa: PLR0913 -- retry task preserves the worker preparation contract
    fragment_id: int,
    *,
    workspace: CaptionWorkspace,
    uri: str,
    attempt_version: int,
    current_version: int,
    plan: EmbeddingRunPlan,
    storage_options: dict[str, str] | None,
) -> PreparedFragmentPublication:
    """Prepare publication from the explicit embedding run plan."""

    def prepare() -> pa.Table:
        files = staged_result_files(workspace)
        if not files:
            msg = f"No staged camera results exist while restaging fragment {fragment_id}"
            raise PublicationError(msg)
        staged = pads.dataset(files, format="parquet", filesystem=workspace.filesystem)
        return _prepare_camera_fragment(
            staged.to_table(filter=pc.field("fragment_id") == fragment_id),
            uri=uri,
            attempt_version=attempt_version,
            current_version=current_version,
            plan=plan,
            storage_options=storage_options,
        )

    task = ray.remote(num_cpus=1, memory=8 * 1024**3)(prepare)
    try:
        payload = ray.get(task.remote())
    except Exception as exc:
        publication_error = fragment_publication._find_publication_error(exc)  # noqa: SLF001
        if publication_error is not None:
            raise publication_error from exc
        raise
    return _parse_camera_publication(str(payload["payload"][0].as_py()), fragment_id)


def embedding_result_fingerprint(table: pa.Table, *, vector_field_name: str, metadata_field_name: str) -> str:
    """Hash logical embedding values independently of hidden bytes behind Arrow nulls."""
    normalized = table
    vector_name = vector_field_name
    vector = normalized[vector_name].combine_chunks()
    if pa.types.is_fixed_size_list(vector.type) and vector.null_count:
        # Casting a null Parquet list to a fixed-size list leaves arbitrary
        # child bytes behind its null parent. Lance instead writes null child
        # values. Canonicalize both before hashing their Arrow IPC streams.
        child_nulls = np.repeat(vector.is_null().to_numpy(zero_copy_only=False), vector.type.list_size)
        child_values = np.array(vector.values.to_numpy(zero_copy_only=False), dtype=np.float32, copy=True)
        child_values[child_nulls] = 0.0
        canonical_children = pa.array(child_values, mask=child_nulls, type=vector.type.value_type)
        canonical_vector = pa.FixedSizeListArray.from_arrays(
            canonical_children, type=vector.type, mask=vector.is_null()
        )
        normalized = normalized.set_column(
            normalized.schema.get_field_index(vector_name), table.schema.field(vector_name), canonical_vector
        )
        # Lance may repeat dictionary-encoded error strings behind nulls in
        # successful rows. Hash logical metadata values, not those hidden bytes.
        metadata_name = metadata_field_name
        metadata_field = table.schema.field(metadata_name)
        canonical_metadata = pa.array(normalized[metadata_name].to_pylist(), type=metadata_field.type)
        normalized = normalized.set_column(
            normalized.schema.get_field_index(metadata_name), metadata_field, canonical_metadata
        )
    return fragment_publication.arrow_table_fingerprint(normalized)


def publish_atomic_staged_results(  # noqa: PLR0913 -- explicit publication inputs
    uri: str,
    attempt: CaptionAttempt,
    workspace: CaptionWorkspace,
    plan: EmbeddingRunPlan,
    *,
    storage_options: dict[str, str] | None,
    commit_attempts: int,
) -> PublicationSummary:
    """Publish all pending results in one Lance update on an atomic-commit-capable backend.

    Backend requirements and qualification limits are documented at module level.
    """
    bindings = publication_bindings(plan)

    if commit_attempts < 1:
        msg = f"Embedding commit attempts must be positive, got {commit_attempts}"
        raise ValueError(msg)
    if not attempt.pending_fragment_ids:
        fragment_publication.verify_selected_fragments(uri, attempt, bindings, storage_options=storage_options)
        return PublicationSummary(published_fragments=0, already_committed_fragments=0)
    with fragment_publication.raise_original_map_exceptions():
        attempt_dataset = lance.dataset(uri, version=attempt.version, storage_options=storage_options)
        guards = fragment_publication._capture_fragment_guards(  # noqa: SLF001 -- reuse internal fragment guards
            attempt_dataset, attempt.complete_fragment_ids, bindings
        )
        prepared_publications = _prepare_camera_publications(
            uri,
            attempt,
            workspace,
            plan,
            # Prepare against the captured snapshot so an own receipt can be checked
            # before conflict guards, even when a previous response was lost.
            current_version=attempt.version,
            storage_options=storage_options,
        )
        return _publish_atomic_fragments(
            uri,
            attempt_dataset,
            attempt,
            workspace,
            plan,
            prepared_publications,
            guards=guards,
            storage_options=storage_options,
            commit_attempts=commit_attempts,
        )


def _has_atomic_receipt(dataset: lance.LanceDataset, *, after_version: int, receipt: str) -> bool:
    """Search this attempt's transactions to reconcile a lost commit response."""
    for version in range(int(dataset.version), after_version, -1):
        transaction = dataset.read_transaction(version)
        if (
            transaction is not None
            and (transaction.transaction_properties or {}).get(_ATOMIC_PUBLICATION_KEY) == receipt
        ):
            return True
    return False


def _commit_atomic_update(
    uri: str,
    transaction: lance.Transaction,
    *,
    storage_options: dict[str, str] | None,
) -> None:
    """Commit one complete selection without automatically rebasing stale descriptors."""
    lance.LanceDataset.commit(uri, transaction, storage_options=storage_options, max_retries=0)


def _assert_atomic_inputs(
    original: lance.LanceDataset,
    current: lance.LanceDataset,
    guards: dict[int, AttemptFragmentGuard],
    bindings: CaptionRunBindings,
    completed_fingerprints: dict[int, str],
) -> None:
    fragment_publication.assert_nullable_schema_extension(original.schema, current.schema)
    protected = (
        *bindings.publication.protected_field_names,
        *bindings.result.fields.names,
    )
    if leaf_field_ids(original, protected) != leaf_field_ids(current, protected):
        msg = "Input or result field identities changed during atomic publication"
        raise PublicationError(msg)
    for fragment_id, guard in guards.items():
        fragment = fragment_publication.require_fragment(current, fragment_id)
        fragment_publication.assert_base_identity(current, fragment, guard, bindings)
        # Deliberately re-read and hash every previously complete selected result:
        # another writer may have changed it since the attempt snapshot. This
        # costs O(completed vector data) on each commit/reconciliation check,
        # even when only a small append is being published. Its wall-clock cost
        # at production table sizes has not been benchmarked.
        if fragment_id in completed_fingerprints and (
            classify_fragment(fragment, bindings=bindings) is not FragmentCaptionState.COMPLETE
            or fragment_publication.result_values_fingerprint(fragment, bindings) != completed_fingerprints[fragment_id]
        ):
            msg = f"Completed result fragment {fragment_id} changed during atomic publication"
            raise PublicationError(msg)


def _verify_atomic_results(
    current: lance.LanceDataset,
    fingerprints: dict[int, str],
    bindings: CaptionRunBindings,
) -> None:
    """Verify staged results on the exact snapshot which carries their receipt."""
    for fragment_id, fingerprint in fingerprints.items():
        fragment = fragment_publication.require_fragment(current, fragment_id)
        if classify_fragment(fragment, bindings=bindings) is not FragmentCaptionState.COMPLETE:
            msg = f"Fragment {fragment_id} has a publication receipt but is still pending"
            raise PublicationError(msg)
        if fragment_publication.result_values_fingerprint(fragment, bindings) != fingerprint:
            msg = f"Fragment {fragment_id} is terminal but differs from its authoritative staged result"
            raise PublicationError(msg)


def _reconcile_atomic_commit(  # noqa: PLR0913 -- receipt verification binds the complete attempted selection
    uri: str,
    original: lance.LanceDataset,
    attempt: CaptionAttempt,
    bindings: CaptionRunBindings,
    *,
    storage_options: dict[str, str] | None,
    guards: dict[int, AttemptFragmentGuard],
    completed_fingerprints: dict[int, str],
    fingerprints: dict[int, str],
    receipt: str,
) -> bool:
    """Allow brief read visibility delays after an ambiguous commit response.

    This bounded wait does not qualify backends lacking atomic commit primitives.
    A receipt and its canonical results must agree on the same immutable snapshot.
    """
    for read_attempt in range(_RECEIPT_READ_ATTEMPTS):
        try:
            current = lance.dataset(uri, storage_options=storage_options)
        except (OSError, RuntimeError):
            if read_attempt == _RECEIPT_READ_ATTEMPTS - 1:
                raise
        else:
            _assert_atomic_inputs(original, current, guards, bindings, completed_fingerprints)
            if _has_atomic_receipt(current, after_version=attempt.version, receipt=receipt):
                _verify_atomic_results(current, fingerprints, bindings)
                return True
        if read_attempt < _RECEIPT_READ_ATTEMPTS - 1:
            time.sleep(_RECEIPT_BACKOFF_S * 2**read_attempt)
    return False


def _publish_atomic_fragments(  # noqa: C901, PLR0912, PLR0913, PLR0915 -- guarded atomic update
    uri: str,
    original: lance.LanceDataset,
    attempt: CaptionAttempt,
    workspace: CaptionWorkspace,
    plan: EmbeddingRunPlan,
    prepared_dataset: ray.data.Dataset,
    *,
    guards: dict[int, AttemptFragmentGuard],
    storage_options: dict[str, str] | None,
    commit_attempts: int,
) -> PublicationSummary:
    """Collect descriptors and commit all pending fragments in one Lance Update.

    Atomicity relies on the backend requirements documented at module level.
    """
    bindings = publication_bindings(plan)
    expected = set(attempt.pending_fragment_ids)
    prepared: dict[int, PreparedFragmentPublication] = {}
    candidates = _iter_camera_publications(prepared_dataset)
    for candidate in candidates:
        fragment_id = candidate.guard.fragment_id
        if fragment_id not in expected or fragment_id in prepared:
            msg = f"Unexpected or duplicate atomic publication fragment {fragment_id}"
            raise PublicationError(msg)
        prepared[fragment_id] = candidate
        guards[fragment_id] = candidate.guard
    if set(prepared) != expected:
        msg = "Prepared atomic publication does not cover every pending fragment"
        raise PublicationError(msg)
    fingerprints = {key: value.staged_values_fingerprint for key, value in prepared.items()}
    completed_fingerprints = {
        key: fragment_publication.result_values_fingerprint(
            fragment_publication.require_fragment(original, key), bindings
        )
        for key in attempt.complete_fragment_ids
    }
    receipt = canonical_digest(
        {
            "workspace": workspace.root_uri,
            "contract_digest": bindings.result.digest,
            "selected": fingerprints,
            "complete": completed_fingerprints,
        }
    )
    expected_modified = {
        fragment_id: leaf_field_ids(
            original,
            tuple(
                name
                for camera in plan.cameras
                if camera.camera_key in plan.pending_cameras.get(fragment_id, ())
                for name in camera.fields.names
            ),
        )
        for fragment_id in expected
    }
    last_error: Exception | None = None
    for _commit_attempt in range(commit_attempts):
        current = lance.dataset(uri, storage_options=storage_options)
        _assert_atomic_inputs(original, current, guards, bindings, completed_fingerprints)
        if _has_atomic_receipt(current, after_version=attempt.version, receipt=receipt):
            _verify_atomic_results(current, fingerprints, bindings)
            return PublicationSummary(published_fragments=0, already_committed_fragments=len(expected))
        descriptors: list[PreparedFragmentUpdate] = []
        stale = False
        for fragment_id in sorted(expected):
            fragment = fragment_publication.require_fragment(current, fragment_id)
            # Only this attempt's receipt permits a pending-to-complete transition.
            # Unrelated camera/caption writes do not change these protected bindings.
            try:
                fragment_publication.assert_pending_identity(current, fragment, guards[fragment_id], bindings)
            except PublicationError as exc:
                msg = (
                    f"Selected embedding fields changed during publication in fragment {fragment_id}. "
                    "Use one active writer per result-field set; rerun after the other writer finishes."
                )
                raise PublicationError(msg) from exc
            candidate = prepared[fragment_id]
            descriptor = candidate.descriptor
            if descriptor is None:
                msg = f"Missing update descriptor for pending fragment {fragment_id}"
                raise PublicationError(msg)
            if descriptor.source_fingerprint != fragment_fingerprint(fragment.metadata):
                candidate = _reprepare_camera_fragment(
                    fragment_id,
                    uri=uri,
                    attempt_version=attempt.version,
                    current_version=int(current.version),
                    workspace=workspace,
                    plan=plan,
                    storage_options=storage_options,
                )
                if (
                    candidate.guard != guards[fragment_id]
                    or candidate.staged_values_fingerprint != fingerprints[fragment_id]
                ):
                    msg = f"Restaged result fragment {fragment_id} differs from the authoritative staged selection"
                    raise PublicationError(msg)
                prepared[fragment_id] = candidate
                descriptor = candidate.descriptor
                if descriptor is None:
                    msg = f"Missing restaged update descriptor for pending fragment {fragment_id}"
                    raise PublicationError(msg)
            modified = set(descriptor.modified_field_ids)
            if not modified or modified != expected_modified[fragment_id]:
                msg = "Camera descriptor must modify exactly the captured pending camera fields"
                raise PublicationError(msg)
            fragment_publication.validate_descriptor(descriptor, modified)
            if descriptor.source_metadata_json != fragment_metadata_json(fragment.metadata):
                stale = True
                break
            descriptors.append(descriptor)
        if stale:
            continue
        transaction = lance.Transaction(
            read_version=int(current.version),
            operation=lance.LanceOperation.Update(
                updated_fragments=[FragmentMetadata.from_json(value.updated_metadata_json) for value in descriptors],
                fields_modified=sorted({field for value in descriptors for field in value.modified_field_ids}),
            ),
            transaction_properties={
                **bindings.publication.transaction_properties,
                "field": bindings.result.caption_field_name,
                "contract_digest": bindings.result.digest,
                "source_version": str(current.version),
                _ATOMIC_PUBLICATION_KEY: receipt,
            },
        )
        try:
            _commit_atomic_update(uri, transaction, storage_options=storage_options)
        except (OSError, RuntimeError) as exc:
            last_error = exc
            if not _reconcile_atomic_commit(
                uri,
                original,
                attempt,
                bindings,
                storage_options=storage_options,
                guards=guards,
                completed_fingerprints=completed_fingerprints,
                fingerprints=fingerprints,
                receipt=receipt,
            ):
                continue
            return PublicationSummary(published_fragments=0, already_committed_fragments=len(expected))
        fragment_publication.verify_selected_fragments(
            uri,
            attempt,
            bindings,
            storage_options=storage_options,
            guards=guards,
            staged_value_fingerprints=fingerprints,
        )
        return PublicationSummary(published_fragments=len(descriptors), already_committed_fragments=0)
    msg = (
        f"Could not reconcile atomic result publication after {commit_attempts} attempt(s). "
        "Staged results are retained; rerun with the same configuration to reconcile publication."
    )
    raise PublicationError(msg) from last_error
