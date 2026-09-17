# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate span coverage, stage original fragments, and publish one transaction."""

import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import lance
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.fs as pafs
from lance.fragment import FragmentMetadata, LanceFragment

from cosmos_curator.core.utils.misc.retry_utils import do_with_retries
from cosmos_curator.core.utils.storage.storage_utils import get_lance_storage_options
from cosmos_curator.next.recipes.video_cut.contracts import SPAN_SCHEMA
from cosmos_curator.next.recipes.video_transcode.config import ResolvedVideoTranscodeConfig
from cosmos_curator.next.recipes.video_transcode.contracts import (
    CLIP_FIELDS,
    PUBLICATION_KEY,
    PUBLICATION_VERSION,
    RESULT_SCHEMA,
    completed_publication,
    make_clip_id,
)
from cosmos_curator.next.utils.arrow_utils import arrow_filesystem, filesystem_path
from cosmos_curator.next.utils.identity import canonical_digest

PREPARED_SCHEMA = pa.schema(
    [
        ("fragment_id", pa.int64()),
        ("fragment", pa.large_string()),
        ("source_fragment", pa.large_string()),
        ("fields_modified", pa.list_(pa.int32())),
    ]
)


@dataclass(frozen=True)
class Workspace:
    """One compatible input snapshot, result schema, and media destination."""

    root: str

    @property
    def results(self) -> str:
        """Return the durable clip result directory."""
        return f"{self.root}/results"

    @property
    def checkpoints(self) -> str:
        """Return Ray's committed span index directory."""
        return f"{self.root}/checkpoints"


def resolve_workspace(config: ResolvedVideoTranscodeConfig, dataset: lance.LanceDataset) -> Workspace:
    """Keep checkpoint reuse separate from execution tuning."""
    scope = {
        "workspace_version": 2,
        "input_uri": config.input.spans_lance_uri,
        "input_version": int(dataset.version),
        "contract_id": config.transcode.contract_id,
        "media_root": config.input.media_root,
        "result_schema": RESULT_SCHEMA.serialize().to_pybytes().hex(),
    }
    return Workspace(f"{config.output.staging_root_uri}/{canonical_digest(scope)}")


def open_input(config: ResolvedVideoTranscodeConfig) -> lance.LanceDataset:
    """Register nullable clip fields, then pin an append-only span snapshot."""
    uri = config.input.spans_lance_uri
    options = get_lance_storage_options(uri, profile_name=config.execution.storage_profile)

    def register() -> lance.LanceDataset:
        dataset = lance.dataset(uri, storage_options=options)
        _validate_span_schema(dataset)
        found = set(CLIP_FIELDS.names) & set(dataset.schema.names)
        if not found:
            dataset.add_columns(CLIP_FIELDS)
            dataset = lance.dataset(uri, storage_options=options)
        _validate_clip_schema(dataset)
        return dataset

    dataset = do_with_retries(
        register, exceptions_to_retry=(OSError, RuntimeError), max_attempts=config.execution.commit_attempts
    )
    record = completed_publication(dataset)
    if record is not None and (
        record.get("contract_id") != config.transcode.contract_id or record.get("media_root") != config.input.media_root
    ):
        msg = "Table already contains a different transcode contract or media destination"
        raise ValueError(msg)
    if any(fragment.metadata.num_deletions for fragment in dataset.get_fragments()):
        msg = "video-transcode requires an unmodified video-cut row set"
        raise ValueError(msg)
    stats = next(
        iter(
            dataset.sql("SELECT COUNT(*) AS n, COUNT(DISTINCT span_id) AS ids FROM dataset").build().to_batch_records()
        )
    ).to_pylist()[0]
    if stats["n"] != stats["ids"]:
        msg = "Input span_id values must be non-null and globally unique"
        raise ValueError(msg)
    return dataset


def _validate_span_schema(dataset: lance.LanceDataset) -> None:
    for field in SPAN_SCHEMA:
        if field.name not in dataset.schema.names or not dataset.schema.field(field.name).equals(
            field, check_metadata=True
        ):
            msg = f"Lance input does not have the video-cut span schema: {field.name}"
            raise ValueError(msg)


def _validate_clip_schema(dataset: lance.LanceDataset) -> None:
    for field in CLIP_FIELDS:
        if field.name not in dataset.schema.names or not dataset.schema.field(field.name).equals(
            field, check_metadata=True
        ):
            msg = f"Input has missing or incompatible transcode fields: {field.name}"
            raise ValueError(msg)


def _validate_result(row: dict[str, Any], config: ResolvedVideoTranscodeConfig) -> None:
    clip_id = make_clip_id(row["span_id"], config.transcode.contract_id)
    if (
        row["transcode_contract_id"] != config.transcode.contract_id
        or row["clip_id"] != clip_id
        or row["clip_uri"] != f"{config.input.media_root}/clips/{clip_id}.mp4"
        or any(
            row[name] is None or row[name] <= 0
            for name in ("clip_size_bytes", "clip_duration_ns", "clip_width", "clip_height", "clip_frame_rate")
        )
        or not math.isfinite(row["clip_frame_rate"])
        or row["clip_video_codec"] != "h264"
        or (row["clip_frame_count"] is not None and row["clip_frame_count"] < 0)
    ):
        msg = f"Invalid completed transcode result for {row['span_id']} (contract, destination, or clip metadata)"
        raise ValueError(msg)


def fragment_complete(fragment: LanceFragment, config: ResolvedVideoTranscodeConfig) -> bool:
    """Require an entirely pending or entirely valid, completed fragment."""
    pending = complete = False
    for batch in fragment.scanner(columns=["span_id", *CLIP_FIELDS.names], batch_size=4096).to_batches():
        for row in batch.to_pylist():
            if all(row[name] is None for name in CLIP_FIELDS.names):
                pending = True
            else:
                _validate_result(row, config)
                complete = True
            if pending and complete:
                msg = f"Fragment {fragment.fragment_id} mixes pending and completed transcode rows"
                raise ValueError(msg)
    return not pending


def pending_fragment_ids(dataset: lance.LanceDataset, config: ResolvedVideoTranscodeConfig) -> tuple[int, ...]:
    """Select complete physical fragments, never individual missing rows."""
    if not set(CLIP_FIELDS.names) & set(dataset.schema.names):
        return tuple(int(fragment.fragment_id) for fragment in dataset.get_fragments())
    _validate_clip_schema(dataset)
    return tuple(
        int(fragment.fragment_id) for fragment in dataset.get_fragments() if not fragment_complete(fragment, config)
    )


def verify_completed(
    dataset: lance.LanceDataset, config: ResolvedVideoTranscodeConfig, fragment_ids: Sequence[int] | None = None
) -> None:
    """Verify only the finite input selection; concurrent appends can remain pending."""
    _validate_span_schema(dataset)
    _validate_clip_schema(dataset)
    selected = fragment_ids if fragment_ids is not None else [f.fragment_id for f in dataset.get_fragments()]
    for fragment_id in selected:
        fragment = dataset.get_fragment(fragment_id)
        if fragment is None or not fragment_complete(fragment, config):
            msg = f"Selected transcode fragment {fragment_id} is absent or pending"
            raise ValueError(msg)


def prepare_fragment(batch: pa.Table, *, config: ResolvedVideoTranscodeConfig, version: int) -> pa.Table:
    """Validate complete, ordered span coverage before adding any column files."""
    if not batch.schema.equals(RESULT_SCHEMA, check_metadata=True):
        msg = "Invalid transcode result schema"
        raise ValueError(msg)
    ids = pc.unique(batch["fragment_id"]).to_pylist()
    if len(ids) != 1:
        msg = "Publication requires one complete fragment per group"
        raise ValueError(msg)
    dataset = lance.dataset(
        config.input.spans_lance_uri,
        version=version,
        storage_options=get_lance_storage_options(
            config.input.spans_lance_uri, profile_name=config.execution.storage_profile
        ),
    )
    _validate_clip_schema(dataset)
    fragment = dataset.get_fragment(ids[0])
    if fragment is None:
        msg = f"Unexpected staged fragment {ids[0]}"
        raise ValueError(msg)
    ordered = batch.take(pc.sort_indices(batch, sort_keys=[("row_offset", "ascending")]))
    expected = fragment.to_table(columns=["span_id"])
    if (
        ordered.num_rows != expected.num_rows
        or ordered["row_offset"].to_pylist() != list(range(expected.num_rows))
        or not ordered["span_id"].equals(expected["span_id"])
    ):
        msg = f"Missing, duplicate, or misaligned span results for fragment {ids[0]}"
        raise ValueError(msg)
    for row in ordered.to_pylist():
        _validate_result(row, config)
    values = ordered.select(["span_id", *CLIP_FIELDS.names])
    metadata, fields_modified = fragment.update_columns(values, left_on="span_id", right_on="span_id")
    return pa.Table.from_pylist(
        [
            {
                "fragment_id": ids[0],
                "fragment": json.dumps(metadata.to_json()),
                "source_fragment": json.dumps(fragment.metadata.to_json()),
                "fields_modified": fields_modified,
            }
        ],
        schema=PREPARED_SCHEMA,
    )


def _field_ids(dataset: lance.LanceDataset, names: Sequence[str]) -> dict[str, int]:
    return {field.name(): int(field.id()) for field in dataset.lance_schema.fields() if field.name() in names}


def _binding(fragment: LanceFragment, field_ids: set[int]) -> str:
    """Identify physical rows and only the files/column positions a reader depends on."""
    raw = fragment.metadata.to_json()
    files = []
    for data_file in raw["files"]:
        fields = data_file["fields"]
        positions = data_file.get("column_indices", list(range(len(fields))))
        selected = [(field, position) for field, position in zip(fields, positions, strict=True) if field in field_ids]
        if selected:
            files.append({**data_file, "fields": selected, "column_indices": None})
    return canonical_digest(
        {
            "id": raw["id"],
            "physical_rows": raw.get("physical_rows"),
            "deletion_file": raw.get("deletion_file"),
            "row_id_meta": raw.get("row_id_meta"),
            "overlays": raw.get("overlays"),
            "files": files,
        }
    )


def _verify_inputs(original: lance.LanceDataset, current: lance.LanceDataset) -> None:
    _validate_span_schema(current)
    _validate_clip_schema(current)
    protected = [*SPAN_SCHEMA.names, *CLIP_FIELDS.names]
    if _field_ids(original, protected) != _field_ids(current, protected):
        msg = "Input or transcode field identities changed during publication"
        raise ValueError(msg)
    base_ids = set(_field_ids(original, SPAN_SCHEMA.names).values())
    for before in original.get_fragments():
        after = current.get_fragment(before.fragment_id)
        if after is None or _binding(before, base_ids) != _binding(after, base_ids):
            msg = f"Video-cut rows or physical binding changed for fragment {before.fragment_id}"
            raise ValueError(msg)


def _refresh_candidates(
    candidates: list[dict[str, Any]],
    original: lance.LanceDataset,
    current: lance.LanceDataset,
    reprepare: Callable[[int], list[dict[str, Any]]] | None,
) -> list[dict[str, Any]]:
    """Rebuild stale column descriptors while protecting span and transcode-owned data."""
    expected = {int(candidate["fragment_id"]) for candidate in candidates}
    protected_ids = set(_field_ids(original, [*SPAN_SCHEMA.names, *CLIP_FIELDS.names]).values())
    refresh = False
    for candidate in candidates:
        fragment_id = int(candidate["fragment_id"])
        before = original.get_fragment(fragment_id)
        after = current.get_fragment(fragment_id)
        assert before is not None
        assert after is not None
        if _binding(before, protected_ids) != _binding(after, protected_ids):
            msg = f"Transcode-owned physical binding changed for pending fragment {fragment_id}"
            raise ValueError(msg)
        refresh |= json.loads(candidate["source_fragment"]) != after.metadata.to_json()
    if refresh:
        if reprepare is None:
            msg = "Concurrent enrichment requires repreparing the transcode column files"
            raise RuntimeError(msg)
        candidates = reprepare(int(current.version))
    if {int(item["fragment_id"]) for item in candidates} != expected or len(candidates) != len(expected):
        msg = "Reprepared transcode fragments do not exactly cover the selected input"
        raise ValueError(msg)
    fields_modified = set(_field_ids(original, CLIP_FIELDS.names).values())
    for candidate in candidates:
        fragment = current.get_fragment(int(candidate["fragment_id"]))
        assert fragment is not None
        if json.loads(candidate["source_fragment"]) != fragment.metadata.to_json():
            msg = "Prepared transcode descriptor does not preserve the current fragment files"
            raise ValueError(msg)
        if set(candidate["fields_modified"]) != fields_modified:
            msg = "Prepared transcode update modifies unexpected fields"
            raise ValueError(msg)
    return candidates


def commit_fragments(
    candidates: list[dict[str, Any]],
    *,
    config: ResolvedVideoTranscodeConfig,
    dataset: lance.LanceDataset,
    reprepare: Callable[[int], list[dict[str, Any]]] | None = None,
) -> lance.LanceDataset:
    """Atomically fill selected fragments while retaining concurrent appends/enrichment.

    Repreparation only reads checkpointed metadata and rewrites column files; it
    never repeats media work. One logical writer owns the transcode columns.
    """
    expected = set(pending_fragment_ids(dataset, config))
    actual = [int(candidate["fragment_id"]) for candidate in candidates]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        msg = "Prepared transcode fragments do not exactly cover the selected input"
        raise ValueError(msg)
    record = {
        "publication_version": PUBLICATION_VERSION,
        "contract_id": config.transcode.contract_id,
        "media_root": config.input.media_root,
        "source_version": int(dataset.version),
        "spans": dataset.count_rows(),
        "fragment_ids": sorted(expected),
    }
    options = get_lance_storage_options(config.input.spans_lance_uri, profile_name=config.execution.storage_profile)
    fields_modified = set(_field_ids(dataset, CLIP_FIELDS.names).values())
    selected = [int(fragment.fragment_id) for fragment in dataset.get_fragments()]

    def commit() -> lance.LanceDataset:
        nonlocal candidates
        current = lance.dataset(config.input.spans_lance_uri, storage_options=options)
        _verify_inputs(dataset, current)
        completed = {
            int(fragment.fragment_id)
            for fragment in current.get_fragments()
            if fragment.fragment_id in expected and fragment_complete(fragment, config)
        }
        if completed == expected and (expected or completed_publication(current) is not None):
            verify_completed(current, config, selected)
            return current
        if completed:
            msg = "Another transcode writer partially published the selected fragments"
            raise ValueError(msg)
        candidates = _refresh_candidates(candidates, dataset, current, reprepare)
        transaction = lance.Transaction(
            # Advancing this version is safe only after checking every source
            # descriptor against current above, or repreparing against current.
            read_version=int(current.version),
            operation=lance.LanceOperation.Update(
                updated_fragments=[FragmentMetadata.from_json(candidate["fragment"]) for candidate in candidates],
                fields_modified=sorted(fields_modified),
            ),
            transaction_properties={
                "kind": "video-transcode",
                "contract_id": config.transcode.contract_id,
                PUBLICATION_KEY.decode(): json.dumps(record, sort_keys=True),
            },
        )
        try:
            published = lance.LanceDataset.commit(
                config.input.spans_lance_uri, transaction, storage_options=options, max_retries=0
            )
        except Exception:
            current = lance.dataset(config.input.spans_lance_uri, storage_options=options)
            _verify_inputs(dataset, current)
            if completed_publication(current) == record:
                verify_completed(current, config, selected)
                return current
            raise
        verify_completed(published, config, selected)
        return published

    return do_with_retries(
        commit, exceptions_to_retry=(OSError, RuntimeError), max_attempts=config.execution.commit_attempts
    )


def cleanup_workspace(workspace: Workspace, config: ResolvedVideoTranscodeConfig) -> None:
    """Delete only the configuration-derived workspace used by this invocation."""
    fs = arrow_filesystem(workspace.root, storage_profile=config.execution.storage_profile)
    path = filesystem_path(workspace.root)
    if fs.get_file_info(path).type != pafs.FileType.NotFound:
        fs.delete_dir(path)
