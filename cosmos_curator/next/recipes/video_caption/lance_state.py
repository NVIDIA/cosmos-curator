# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Caption field registration, finite-attempt snapshots, and fragment identity."""

import json
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import lance
import pyarrow as pa
from lance.fragment import FragmentMetadata, LanceFragment

from cosmos_curator.next.recipes.video_caption.bindings import CaptionRunBindings
from cosmos_curator.next.recipes.video_transcode.contracts import validate_clip_table
from cosmos_curator.next.utils.identity import canonical_digest


class FragmentCaptionState(StrEnum):
    """The only two compatible states; every other shape raises."""

    PENDING = "pending"
    COMPLETE = "complete"


@dataclass(frozen=True)
class CaptionAttempt:
    """The finite snapshot and fragment membership selected by one start."""

    version: int
    pending_fragment_ids: tuple[int, ...]
    complete_fragment_ids: tuple[int, ...]

    @property
    def selected_fragment_ids(self) -> tuple[int, ...]:
        """Return every fragment fixed at the start of the attempt."""
        return self.pending_fragment_ids + self.complete_fragment_ids


def ensure_caption_fields(
    uri: str,
    *,
    storage_options: dict[str, str] | None,
    bindings: CaptionRunBindings,
    attempts: int,
) -> lance.LanceDataset:
    """Validate or atomically add both owned fields, retrying disjoint schema races."""
    if attempts < 1:
        msg = f"Schema registration attempts must be positive, got {attempts}"
        raise ValueError(msg)
    last_error: Exception | None = None
    for _ in range(attempts):
        dataset = lance.dataset(uri, storage_options=storage_options)
        bindings.publication.validate_base_schema(dataset)
        presence = _field_set_presence(dataset, bindings)
        if presence == "present":
            return dataset
        if presence == "partial":
            msg = f"Caption field set {bindings.result.caption_field_name!r} is partially present in {uri}"
            raise ValueError(msg)
        try:
            dataset.add_columns(bindings.result.fields)
        except (OSError, RuntimeError) as exc:
            # A disjoint schema commit may have won. Reopen on the next loop;
            # deterministic storage failures still surface after the bounded retries.
            last_error = exc
            continue
        registered = lance.dataset(uri, storage_options=storage_options)
        _assert_present_fields_match(registered, bindings)
        return registered
    latest = lance.dataset(uri, storage_options=storage_options)
    if _field_set_presence(latest, bindings) == "present":
        return latest
    msg = f"Could not atomically register caption field set {bindings.result.caption_field_name!r} on {uri}"
    raise RuntimeError(msg) from last_error


def validate_caption_input(dataset: lance.LanceDataset, *, uri: str) -> None:
    """Validate video-caption's clip-owned input contract and global ``clip_id`` uniqueness."""
    validate_clip_table(dataset, uri=uri)
    query = (
        "SELECT COUNT(*) AS rows, COUNT(DISTINCT clip_id) AS clip_ids, "
        "SUM(CASE WHEN clip_id IS NULL THEN 1 ELSE 0 END) AS null_clip_ids, "
        "SUM(CASE WHEN clip_uri IS NULL THEN 1 ELSE 0 END) AS null_clip_uris, "
        "SUM(CASE WHEN clip_size_bytes IS NULL THEN 1 ELSE 0 END) AS null_clip_sizes FROM dataset"
    )
    batches = dataset.sql(query).build().to_batch_records()
    rows = [row for batch in batches for row in batch.to_pylist()]
    if len(rows) != 1:
        msg = f"Could not validate canonical clip identities in {uri}"
        raise RuntimeError(msg)
    stats = rows[0]
    row_count = int(stats["rows"])
    distinct = int(stats["clip_ids"])
    null_counts = [int(stats[name] or 0) for name in ("null_clip_ids", "null_clip_uris", "null_clip_sizes")]
    if distinct != row_count:
        msg = f"Canonical clip_id values in {uri} are not globally unique ({distinct} distinct for {row_count} rows)"
        raise ValueError(msg)
    if any(null_counts):
        msg = f"Canonical clip_id, clip_uri, and clip_size_bytes values in {uri} must all be non-null"
        raise ValueError(msg)


def capture_attempt(
    dataset: lance.LanceDataset,
    *,
    bindings: CaptionRunBindings,
) -> CaptionAttempt:
    """Classify every fragment of one pinned snapshot."""
    pending: list[int] = []
    complete: list[int] = []
    for fragment in dataset.get_fragments():
        state = classify_fragment(fragment, bindings=bindings)
        target = pending if state is FragmentCaptionState.PENDING else complete
        target.append(int(fragment.fragment_id))
    return CaptionAttempt(
        version=int(dataset.version),
        pending_fragment_ids=tuple(pending),
        complete_fragment_ids=tuple(complete),
    )


def classify_fragment(  # noqa: C901, PLR0912 -- shared per-row and batch validation paths preserve compatibility
    fragment: LanceFragment,
    *,
    bindings: CaptionRunBindings,
) -> FragmentCaptionState:
    """Return pending/complete or reject mixed, stale, and corrupt values."""
    result = bindings.result
    if result.fragment_has_pending is not None:
        return FragmentCaptionState.PENDING if result.fragment_has_pending(fragment) else FragmentCaptionState.COMPLETE
    saw_pending = False
    saw_complete = False
    result_columns = [result.caption_field_name, result.metadata_field_name]
    for raw_batch in fragment.scanner(columns=result_columns).to_batches():
        batch = pa.Table.from_batches([raw_batch])
        if result.validate_terminal_batch is None:
            for row in batch.to_pylist():
                caption = row[result.caption_field_name]
                metadata = row[result.metadata_field_name]
                if caption is None and metadata is None:
                    saw_pending = True
                else:
                    saw_complete = True
                    try:
                        result.validate_terminal_value(caption, metadata)
                    except (TypeError, ValueError) as exc:
                        msg = f"Fragment {fragment.fragment_id} has incompatible or corrupt caption state: {exc}"
                        raise ValueError(msg) from exc
                if saw_pending and saw_complete:
                    msg = f"Fragment {fragment.fragment_id} mixes pending and terminal caption rows"
                    raise ValueError(msg)
        else:
            metadata = batch[result.metadata_field_name]
            result_values = batch[result.caption_field_name]
            pending = metadata.is_null().to_numpy(zero_copy_only=False) & result_values.is_null().to_numpy(
                zero_copy_only=False
            )
            saw_pending |= bool(pending.any())
            saw_complete |= bool((~pending).any())
            if saw_pending and saw_complete:
                msg = f"Fragment {fragment.fragment_id} mixes pending and terminal caption rows"
                raise ValueError(msg)
            if saw_complete:
                try:
                    result.validate_terminal_batch(batch)
                except (TypeError, ValueError) as exc:
                    msg = f"Fragment {fragment.fragment_id} has incompatible or corrupt caption state: {exc}"
                    raise ValueError(msg) from exc
    # Lance does not normally retain empty fragments. Treat one as complete: it
    # contains no pending work and update_columns must never be invoked for it.
    if saw_pending:
        return FragmentCaptionState.PENDING
    validate_complete_fragment = bindings.publication.validate_complete_fragment
    if validate_complete_fragment is not None:
        try:
            validate_complete_fragment(fragment)
        except (TypeError, ValueError) as exc:
            msg = f"Fragment {fragment.fragment_id} has incompatible or corrupt caption state: {exc}"
            raise ValueError(msg) from exc
    return FragmentCaptionState.COMPLETE


def fragment_metadata_json(metadata: FragmentMetadata) -> str:
    """Serialize physical fragment metadata canonically."""
    return json.dumps(metadata.to_json(), allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def fragment_fingerprint(metadata: FragmentMetadata) -> str:
    """Fingerprint the complete serialized physical fragment identity."""
    return canonical_digest(metadata.to_json())


def protected_binding_fingerprint(metadata: FragmentMetadata, protected_field_ids: set[int]) -> str:
    """Fingerprint row layout and files binding split/caption-owned leaves."""
    raw: Any = metadata.to_json()
    if not isinstance(raw, dict):
        msg = "FragmentMetadata.to_json() returned a non-object value"
        raise TypeError(msg)
    protected_files = []
    for file_entry in raw.get("files", []):
        if not isinstance(file_entry, dict):
            continue
        fields = file_entry.get("fields", [])
        columns = file_entry.get("column_indices", [])
        if len(fields) != len(columns):
            msg = "Fragment data file has mismatched field and column index counts"
            raise ValueError(msg)
        selected = [
            (field, column) for field, column in zip(fields, columns, strict=True) if int(field) in protected_field_ids
        ]
        if selected:
            protected_files.append(
                {
                    **file_entry,
                    "fields": [field for field, _ in selected],
                    "column_indices": [column for _, column in selected],
                }
            )
    identity = {
        "id": raw.get("id"),
        "physical_rows": raw.get("physical_rows"),
        "deletion_file": raw.get("deletion_file"),
        "overlays": raw.get("overlays"),
        "row_id_meta": raw.get("row_id_meta"),
        "files": protected_files,
    }
    return canonical_digest(identity)


def leaf_field_ids(dataset: lance.LanceDataset, field_names: tuple[str, ...]) -> set[int]:
    """Return Lance IDs for primitive leaves beneath the named top-level fields."""
    ids: set[int] = set()
    fields_by_name = {field.name(): field for field in dataset.lance_schema.fields()}
    for name in field_names:
        field = fields_by_name[name]
        ids.update(_leaf_ids(field))
    return ids


def _leaf_ids(field: Any) -> set[int]:  # noqa: ANN401 -- pylance's LanceField is not exposed in its type stubs
    children = field.children()
    if not children:
        return {int(field.id())}
    ids: set[int] = set()
    for child in children:
        ids.update(_leaf_ids(child))
    return ids


def _field_set_presence(dataset: lance.LanceDataset, bindings: CaptionRunBindings) -> str:
    result = bindings.result
    names = set(dataset.schema.names)
    found = sum(name in names for name in result.fields.names)
    if found == 0:
        return "absent"
    if found == len(result.fields):
        _assert_present_fields_match(dataset, bindings)
        return "present"
    return "partial"


def _assert_present_fields_match(dataset: lance.LanceDataset, bindings: CaptionRunBindings) -> None:
    result = bindings.result
    mismatched = [
        expected.name
        for expected in result.fields
        if expected.name not in dataset.schema.names
        or not dataset.schema.field(expected.name).equals(expected, check_metadata=True)
    ]
    if mismatched:
        msg = f"Existing caption field set {result.caption_field_name!r} is incompatible: {', '.join(mismatched)}"
        raise ValueError(msg)
