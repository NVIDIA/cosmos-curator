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

import lance
import numpy as np
import pyarrow as pa
import ray
from lance.fragment import FragmentMetadata
from loguru import logger

from cosmos_curator.next.recipes.video_caption import publication as fragment_publication
from cosmos_curator.next.recipes.video_caption.bindings import CaptionRunBindings, TerminalRowValidator
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


def reconcile_embedding_replays(  # noqa: PLR0913 -- replay identity and validation inputs stay explicit
    table: pa.Table,
    current_bindings: pa.Table,
    fragment_id: int,
    row_count: int,
    *,
    metadata_field_name: str,
    validate_staged_row: TerminalRowValidator,
) -> pa.Table:
    """Choose one canonical replay per row after checking every staged binding."""
    offsets = [int(value) for value in table["row_offset"].to_pylist()]
    if any(offset < 0 or offset >= row_count for offset in offsets):
        msg = f"Staged fragment {fragment_id} contains an out-of-range replay row"
        raise PublicationError(msg)
    staged_bindings = table.select(current_bindings.column_names).cast(current_bindings.schema)
    expected_bindings = current_bindings.take(pa.array(offsets, type=pa.int64()))
    if not staged_bindings.equals(expected_bindings):
        msg = f"Staged fragment {fragment_id} has replay rows with incompatible input bindings"
        raise PublicationError(msg)
    chosen: dict[int, tuple[tuple[int, str], int]] = {}
    for index, row in enumerate(table.to_pylist()):
        try:
            validate_staged_row(row)
        except (TypeError, ValueError) as exc:
            msg = f"Staged fragment {fragment_id} contains a non-canonical replay row: {exc}"
            raise PublicationError(msg) from exc
        offset = offsets[index]
        metadata = row[metadata_field_name]
        # A replay can observe a later media error after an earlier successful
        # stage. Preserve the verified vector when either outcome is valid.
        priority = 0 if metadata["status"] == "success" else 1
        order = (priority, canonical_digest(row))
        if offset not in chosen or order < chosen[offset][0]:
            chosen[offset] = (order, index)
    if len(chosen) != row_count:
        msg = f"Staged fragment {fragment_id} is missing result rows after replay reconciliation"
        raise PublicationError(msg)
    logger.warning(
        "Reconciled {} replayed staged row(s) for fragment {}",
        table.num_rows - row_count,
        fragment_id,
    )
    return table.take(pa.array([chosen[offset][1] for offset in range(row_count)], type=pa.int64()))


def publish_atomic_staged_results(  # noqa: PLR0913 -- the publication boundary keeps recovery inputs explicit
    uri: str,
    attempt: CaptionAttempt,
    workspace: CaptionWorkspace,
    bindings: CaptionRunBindings,
    *,
    storage_options: dict[str, str] | None,
    commit_attempts: int,
) -> PublicationSummary:
    """Publish all pending results in one Lance update on an atomic-commit-capable backend.

    Backend requirements and qualification limits are documented at module level.
    """
    if commit_attempts < 1:
        msg = f"Embedding commit attempts must be positive, got {commit_attempts}"
        raise ValueError(msg)
    if not attempt.pending_fragment_ids:
        fragment_publication.verify_selected_fragments(uri, attempt, bindings, storage_options=storage_options)
        return PublicationSummary(published_fragments=0, already_committed_fragments=0)

    with fragment_publication.raise_original_map_exceptions():
        attempt_dataset, guards, prepared_publications = fragment_publication.prepare_staged_publications(
            uri, attempt, workspace, bindings, storage_options=storage_options
        )
        return _publish_atomic_fragments(
            uri,
            attempt_dataset,
            attempt,
            workspace,
            bindings,
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
        bindings.result.caption_field_name,
        bindings.result.metadata_field_name,
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


def _publish_atomic_fragments(  # noqa: C901, PLR0912, PLR0913, PLR0915 -- guarded transaction and reconciliation
    uri: str,
    original: lance.LanceDataset,
    attempt: CaptionAttempt,
    workspace: CaptionWorkspace,
    bindings: CaptionRunBindings,
    prepared_dataset: ray.data.Dataset,
    *,
    guards: dict[int, AttemptFragmentGuard],
    storage_options: dict[str, str] | None,
    commit_attempts: int,
) -> PublicationSummary:
    """Collect descriptors and commit all pending fragments in one Lance Update.

    Atomicity relies on the backend requirements documented at module level.
    """
    expected = set(attempt.pending_fragment_ids)
    prepared: dict[int, PreparedFragmentPublication] = {}
    for candidate in fragment_publication.iter_prepared_publications(prepared_dataset):
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
            "field": bindings.result.caption_field_name,
            "selected": [
                {
                    "id": key,
                    "rows": guard.row_count,
                    "row_ids": guard.ordered_row_ids_fingerprint,
                    "input_binding": guard.base_binding_fingerprint,
                    "staged_values": fingerprints.get(key, completed_fingerprints.get(key)),
                }
                for key, guard in sorted(guards.items())
            ],
        }
    )
    expected_modified = leaf_field_ids(
        original, (bindings.result.caption_field_name, bindings.result.metadata_field_name)
    )
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
            if classify_fragment(fragment, bindings=bindings) is not FragmentCaptionState.PENDING:
                msg = f"Competing writer changed result fragment {fragment_id} without this publication receipt"
                raise PublicationError(msg)
            candidate = prepared[fragment_id]
            fragment_publication.assert_pending_identity(current, fragment, candidate.guard, bindings)
            descriptor = candidate.descriptor
            if descriptor is None or descriptor.source_fingerprint != fragment_fingerprint(fragment.metadata):
                candidate = fragment_publication.reprepare_fragment_with_ray(
                    fragment_id,
                    uri=uri,
                    attempt=attempt,
                    workspace=workspace,
                    bindings=bindings,
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
                msg = f"Competing writer completed result fragment {fragment_id} while preparing atomic publication"
                raise PublicationError(msg)
            fragment_publication.validate_descriptor(descriptor, expected_modified)
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
                fields_modified=sorted(expected_modified),
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
        return PublicationSummary(published_fragments=len(expected), already_committed_fragments=0)
    msg = (
        f"Could not reconcile atomic result publication after {commit_attempts} attempt(s). "
        "Staged results are retained; rerun with the same configuration to reconcile publication."
    )
    raise PublicationError(msg) from last_error
