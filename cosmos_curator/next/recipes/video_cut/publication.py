# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stage packed Lance fragments and atomically append each run's new sources.

One logical cutter owns source selection and temporary source Parquet. Publication
can rebuild new fragment files after a failure without repeating media work.
Input identities and outcome counts travel in the Append transaction, including
sources with no spans. Retain this transaction history for cross-run recovery.
"""

import json
import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import lance
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.fs as pafs
from lance.fragment import write_fragments
from loguru import logger

from cosmos_curator.core.utils.storage.storage_utils import get_lance_storage_options
from cosmos_curator.next.recipes.video_cut.config import ResolvedVideoCutConfig
from cosmos_curator.next.recipes.video_cut.contracts import (
    SOURCE_SCHEMA,
    SPAN_FIELDS,
    SPAN_SCHEMA,
    make_source_id,
    span_schema,
)
from cosmos_curator.next.recipes.video_cut.storage import arrow_filesystem, filesystem_path
from cosmos_curator.next.utils import lance_fragment_recovery
from cosmos_curator.next.utils.identity import canonical_digest
from cosmos_curator.next.utils.lance_utils import LANCE_DATA_STORAGE_VERSION

_PUBLICATION_KEY = b"video_cut.publication"
_PUBLICATION_VERSION = 3
_LEGACY_PUBLICATION_VERSION = 2
_FRAGMENT_SCHEMA = pa.schema([pa.field("fragment", pa.string(), nullable=False)])


@dataclass(frozen=True)
class Workspace:
    """Recovery results and Ray checkpoints retained until the final commit."""

    root: str

    @property
    def results(self) -> str:
        """Return the immutable source-result directory."""
        return f"{self.root}/results"

    @property
    def checkpoints(self) -> str:
        """Return the native Ray checkpoint directory."""
        return f"{self.root}/checkpoints"


def resolve_workspace(config: ResolvedVideoCutConfig, input_snapshot_id: str) -> Workspace:
    """Isolate compatible recovery state by path, without a separate manifest."""
    scope = {
        "workspace_version": 2,
        "source_schema": SOURCE_SCHEMA.serialize().to_pybytes().hex(),
        "cutting_contract": config.split.cutting_contract,
        "input_snapshot_id": input_snapshot_id,
        "spans_lance_uri": config.output.spans_lance_uri,
    }
    return Workspace(f"{config.output.staging_root_uri}/{canonical_digest(scope)}")


def cleanup_workspace(workspace: Workspace, config: ResolvedVideoCutConfig) -> None:
    """Remove the workspace used by this invocation after publication, tolerating absence."""
    fs = arrow_filesystem(workspace.root, config.execution.storage_profile)
    path = filesystem_path(workspace.root)
    if fs.get_file_info(path).type != pafs.FileType.NotFound:
        fs.delete_dir(path)


def open_span_table(config: ResolvedVideoCutConfig, input_snapshot_id: str) -> lance.LanceDataset:
    """Create an empty canonical table or validate its schema and ownership."""
    schema = span_schema(config.split, input_snapshot_id)
    dataset = lance_fragment_recovery.open_or_create_table(
        uri=config.output.spans_lance_uri,
        storage_profile=config.execution.storage_profile,
        schema=schema,
        data_storage_version=LANCE_DATA_STORAGE_VERSION,
        kind="video-cut",
    )
    actual = dataset.schema.metadata or {}
    if any(actual.get(key) != value for key, value in (schema.metadata or {}).items()):
        msg = "Lance table belongs to a different video-cut contract; use a new output location"
        raise ValueError(msg)
    return dataset


def source_result_files(workspace: Workspace, config: ResolvedVideoCutConfig) -> list[str]:
    """List the source results used by outcome counting and span expansion."""
    fs = arrow_filesystem(workspace.results, config.execution.storage_profile)
    files = fs.get_file_info(
        pafs.FileSelector(filesystem_path(workspace.results), recursive=True, allow_not_found=True)
    )
    return [
        file.path
        for file in sorted(files, key=lambda info: info.path)
        if file.type == pafs.FileType.File and file.path.endswith(".parquet")
    ]


def publication_records(dataset: lance.LanceDataset) -> Iterator[dict[str, Any]]:
    """Read committed run receipts, newest first, including the old snapshot receipt."""
    for version in range(int(dataset.version), 0, -1):
        transaction = dataset.read_transaction(version)
        if transaction is None:
            continue
        encoded = (transaction.transaction_properties or {}).get(_PUBLICATION_KEY.decode())
        if encoded is not None:
            record = json.loads(encoded)
            if not isinstance(record, dict) or record.get("publication_version") != _PUBLICATION_VERSION:
                msg = "Unsupported video-cut publication record"
                raise ValueError(msg)
            yield record
    encoded = (dataset.schema.metadata or {}).get(_PUBLICATION_KEY)
    if encoded is None:
        return
    record = json.loads(encoded)
    if not isinstance(record, dict) or record.get("publication_version") != _LEGACY_PUBLICATION_VERSION:
        msg = "Unsupported video-cut publication record"
        raise ValueError(msg)
    snapshot = (dataset.schema.metadata or {}).get(b"video_cut.input_snapshot", b"").decode()
    yield record | {"input_snapshot": snapshot}


def completed_publication(dataset: lance.LanceDataset, input_snapshot_id: str | None = None) -> dict[str, Any] | None:
    """Recognize a committed input selection without depending on the current table version."""
    return next(
        (
            record
            for record in publication_records(dataset)
            if input_snapshot_id is None or record.get("input_snapshot") == input_snapshot_id
        ),
        None,
    )


def pending_source_uris(dataset: lance.LanceDataset, source_uris: tuple[str, ...]) -> tuple[str, ...]:
    """Skip committed sources, including empty plans/errors that have no canonical span row."""
    published = {source_id for record in publication_records(dataset) for source_id in record.get("source_ids", [])}
    # Also recognize spans from the previous snapshot implementation, whose
    # completion record did not retain the identities of zero-span sources.
    for batch in dataset.to_batches(columns=["source_id"]):
        published.update(pc.unique(batch.column("source_id")).to_pylist())
    return tuple(uri for uri in source_uris if make_source_id(uri) not in published)


def select_source_results(batch: pa.Table, *, source_ids: tuple[str, ...]) -> pa.Table:
    """Filter checkpointed plans without round-tripping Arrow types/nullability through Python rows."""
    if not batch.schema.equals(SOURCE_SCHEMA):
        msg = "Invalid source-result schema"
        raise ValueError(msg)
    return batch.filter(pc.is_in(batch["source_id"], value_set=pa.array(source_ids, type=pa.string())))


def expand_sources(batch: pa.Table, *, contract_id: str, spans_per_fragment: int) -> Iterator[pa.Table]:
    """Check provenance and flatten recipe-owned plans in bounded Arrow batches."""
    if not batch.schema.equals(SOURCE_SCHEMA):
        msg = "Invalid source-result schema"
        raise ValueError(msg)
    for name, expected in (("record_schema_version", 1), ("cut_contract_id", contract_id), ("stream_index", 0)):
        if pc.all(pc.equal(batch[name], expected)).as_py() is False:
            msg = "Invalid cutting provenance in result Parquet"
            raise ValueError(msg)
    sources = batch.filter(pc.equal(batch["status"], "success"))
    spans = pc.list_flatten(sources["spans"])
    parents = pc.list_parent_indices(sources["spans"])
    span_names = {field.name for field in SPAN_FIELDS}
    # A typed empty batch also covers sources containing only empty plans or errors.
    for offset in range(0, max(len(spans), 1), spans_per_fragment):
        items = spans.slice(offset, spans_per_fragment)
        indices = parents.slice(offset, spans_per_fragment)
        yield pa.Table.from_arrays(
            [
                pc.struct_field(items, field.name)
                if field.name in span_names
                else pc.take(sources[field.name], indices)
                for field in SPAN_SCHEMA
            ],
            schema=SPAN_SCHEMA,
        )


def stage_fragment(
    batch: pa.Table, *, schema: pa.Schema, spans_lance_uri: str, spans_per_fragment: int, storage_profile: str
) -> pa.Table:
    """Write one packed fragment and return only its metadata to the driver."""
    if not batch.num_rows:
        return pa.Table.from_pylist([], schema=_FRAGMENT_SCHEMA)
    if batch.num_rows > spans_per_fragment:
        msg = "Publication batch exceeds the configured fragment size"
        raise ValueError(msg)
    fragments = write_fragments(
        batch.cast(schema),
        spans_lance_uri,
        schema=schema,
        mode="overwrite",
        max_rows_per_file=batch.num_rows,
        data_storage_version=LANCE_DATA_STORAGE_VERSION,
        storage_options=get_lance_storage_options(spans_lance_uri, profile_name=storage_profile),
    )
    if len(fragments) != 1 or fragments[0].num_rows != batch.num_rows:
        msg = "One packed batch must stage exactly one complete Lance fragment"
        raise RuntimeError(msg)
    return pa.Table.from_pylist([{"fragment": json.dumps(fragments[0].to_json())}], schema=_FRAGMENT_SCHEMA)


def commit_fragments(
    candidates: list[str],
    *,
    config: ResolvedVideoCutConfig,
    input_snapshot_id: str,
    publication: dict[str, Any],
    read_version: int,
) -> lance.LanceDataset:
    """Append new fragments and their source receipt in one atomic transaction.

    One logical cutter owns source selection. Other pipelines may extend the
    schema or populate their own columns while these fragments are staged.
    """
    source_ids = publication.get("source_ids", [])
    if len(source_ids) != publication["source_results"] or len(set(source_ids)) != len(source_ids):
        msg = "Cut publication must record every new source exactly once, including zero-span outcomes"
        raise ValueError(msg)
    fragments = [lance.FragmentMetadata.from_json(candidate) for candidate in candidates]
    record = publication | {
        "publication_version": _PUBLICATION_VERSION,
        "input_snapshot": input_snapshot_id,
        "spans": sum(fragment.num_rows for fragment in fragments),
    }
    transaction = lance.Transaction(
        read_version=read_version,
        operation=lance.LanceOperation.Append(fragments),
        transaction_properties={
            "kind": "video-cut",
            "operation": "append-spans",
            "input_snapshot": input_snapshot_id,
            _PUBLICATION_KEY.decode(): json.dumps(record, sort_keys=True),
        },
    )
    for attempt in range(1, config.execution.commit_attempts + 1):
        current = open_span_table(config, input_snapshot_id)
        if completed_publication(current, input_snapshot_id) is not None:
            return current
        try:
            committed = lance.LanceDataset.commit(
                config.output.spans_lance_uri,
                transaction,
                storage_options=get_lance_storage_options(
                    config.output.spans_lance_uri, profile_name=config.execution.storage_profile
                ),
                enable_v2_manifest_paths=True,
            )
        except Exception:
            # The receipt shares the append: a lost response cannot duplicate spans.
            current = open_span_table(config, input_snapshot_id)
            if completed_publication(current, input_snapshot_id) is not None:
                return current
            if attempt == config.execution.commit_attempts:
                raise
            logger.warning("video-cut publication attempt {} failed; retrying", attempt)
            time.sleep(min(2**attempt, 16))
        else:
            logger.info(
                "Published {} spans in {} fragment(s) at Lance version {}",
                record["spans"],
                len(fragments),
                committed.version,
            )
            return committed
    msg = "Publication attempts exhausted"
    raise AssertionError(msg)
