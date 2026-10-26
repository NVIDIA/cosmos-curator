# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise shared append publication with real Arrow/Lance data and no Ray execution."""

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import patch

import attrs
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.fs as pafs
import pyarrow.parquet as pq
import pytest
from lance.fragment import FragmentMetadata

from cosmos_curator.next.append import commit, contracts, execution
from cosmos_curator.next.append.attempts import AppendAttempt, reconcile_and_select_attempt
from cosmos_curator.next.append.config import AppendExecutionConfig
from cosmos_curator.next.core.ray_checkpoints import parquet_stable_schema

ITEM_FIELDS = pa.schema(
    [pa.field("item_id", pa.string(), nullable=False), pa.field("uri", pa.string(), nullable=False)]
)
RESULT_FIELDS = pa.schema(
    [
        pa.field("item_id", pa.string(), nullable=False),
        pa.field("values", pa.list_(pa.int64()), nullable=False),
        pa.field("error", pa.string()),
    ]
)
ROW_FIELDS = pa.schema(
    [
        pa.field("item_id", pa.string(), nullable=False),
        pa.field("row_id", pa.string(), nullable=False),
        pa.field("value", pa.int64()),
        pa.field("error", pa.string()),
    ]
)


def validate_results(batch: pa.RecordBatch) -> None:
    """Accept successes and terminal errors without values; reject anything else as non-terminal."""
    for row in batch.to_pylist():
        if row["error"] is not None and row["values"]:
            msg = f"Invalid test result for {row['item_id']}"
            raise ValueError(msg)


def expand_results(results: pa.Table) -> Iterator[pa.Table]:
    """Append one row per value of each successful item, one item per table."""
    for row in results.to_pylist():
        yield pa.Table.from_pylist(
            [
                {"item_id": row["item_id"], "row_id": f"{row['item_id']}:{index}", "value": value}
                for index, value in enumerate(row["values"])
            ]
            or [
                {"item_id": row["item_id"], "row_id": f"{row['item_id']}:outcome", "value": None, "error": row["error"]}
            ],
            schema=ROW_FIELDS,
        )


def make_spec(*, owner: str = "item-recipe", result_contract: Mapping[str, Any] | None = None) -> contracts.AppendSpec:
    """Exercise lists that stage in Parquet-stable form and items with empty outcomes."""
    return contracts.AppendSpec(
        owner=owner,
        id_column="item_id",
        input_fields=ITEM_FIELDS,
        result_fields=RESULT_FIELDS,
        output_fields=ROW_FIELDS,
        result_contract=dict(result_contract or {"version": 1}),
        validate_batch=validate_results,
        expand=expand_results,
        error_column="error",
    )


def make_bindings(
    uri: str,
    spec: contracts.AppendSpec,
    *,
    staging_root: str,
    result_contract: Mapping[str, Any] | None = None,
    **execution_settings: Any,  # noqa: ANN401 -- AppendExecutionConfig fields
) -> contracts.AppendBindings:
    """Bind local test data, optionally replacing the spec's result contract."""
    settings = {"commit_attempts": 1, "rows_per_fragment": 2, **execution_settings}
    return contracts.AppendBindings(
        uri=uri,
        spec=spec if result_contract is None else attrs.evolve(spec, result_contract=dict(result_contract)),
        staging_root=staging_root,
        execution=AppendExecutionConfig(**settings),
    )


def make_items(item_ids: Sequence[str]) -> pa.Table:
    """Discover one item per ID, in the order given."""
    return pa.Table.from_pylist(
        [{"item_id": item_id, "uri": f"{item_id}.bin"} for item_id in item_ids], schema=ITEM_FIELDS
    )


def make_results(spec: contracts.AppendSpec, values: Mapping[str, Sequence[int] | None]) -> pa.Table:
    """Build one canonical result per item; ``None`` records a terminal error."""
    return pa.Table.from_pylist(
        [
            {
                "item_id": item_id,
                "values": item_values or [],
                "error": f"unreadable {item_id}" if item_values is None else None,
            }
            for item_id, item_values in values.items()
        ],
        schema=spec.result_fields,
    )


def select(bindings: contracts.AppendBindings, items: pa.Table, *, max_items: int | None = None) -> AppendAttempt:
    """Open the table and resume or select an attempt through the public steps, optionally limiting a new selection."""
    if max_items is not None:
        bindings = attrs.evolve(bindings, execution=bindings.execution.model_copy(update={"max_items": max_items}))
    return reconcile_and_select_attempt(bindings, lambda: items)


def stage_results(attempt: AppendAttempt, results: pa.Table, *, complete: bool = True) -> None:
    """Encode canonical results for Parquet and write the selection into its workspace without Ray.

    Derive storage types from the supplied schema so fault tests can still stage
    malformed results for the publisher to reject.
    """
    assert attempt.workspace is not None
    id_column = attempt.bindings.spec.id_column
    selected = results.filter(pc.is_in(results[id_column], value_set=attempt.selection[id_column].combine_chunks()))
    directory = Path(attempt.workspace.results)
    directory.mkdir(parents=True, exist_ok=True)
    staged = selected.cast(parquet_stable_schema(selected.schema))
    pq.write_table(staged, directory / f"staged-{len(list(directory.iterdir()))}.parquet")
    if complete:
        attempt.workspace.record_completion(storage_profile=attempt.bindings.execution.storage_profile)


def stage_locally(
    files: Sequence[str],
    *,
    filesystem: pafs.FileSystem,
    bindings: contracts.AppendBindings,
) -> list[FragmentMetadata]:
    """Replace Ray's read, packing, and collection while retaining the production worker callables."""
    tables = [
        rows
        for path in files
        for rows in execution._appended_rows(
            pq.read_table(path, filesystem=filesystem).replace_schema_metadata(None), spec=bindings.spec
        )
    ]
    rows_per_fragment = bindings.execution.rows_per_fragment
    rows = pa.concat_tables(tables)
    descriptors = [
        descriptor
        for offset in range(0, rows.num_rows, rows_per_fragment)
        for descriptor in commit.stage_fragment(rows.slice(offset, rows_per_fragment), bindings=bindings)[
            "fragment"
        ].to_pylist()
    ]
    return [FragmentMetadata.from_json(descriptor) for descriptor in descriptors]


def publish_attempt(attempt: AppendAttempt, results: pa.Table | None = None) -> execution.AppendOutcome:
    """Publish an attempt through the recipes' adapter, optionally staging results first, without Ray."""
    if results is not None:
        stage_results(attempt, results)
    with patch.object(execution, "stage_append_fragments", stage_locally):
        return execution.publish_append(attempt)


@attrs.define
class Runs:
    """The recipe arguments and outcome of each replaced ``run_append`` call, and an optional discovery limit."""

    calls: list[tuple[dict[str, Any], execution.AppendOutcome]] = attrs.Factory(list)
    discovered: pa.Table | None = None

    @contextmanager
    def discovering(self, items: pa.Table) -> Iterator[None]:
        """Within the block, the recipe's discovery finds only these of its items."""
        self.discovered = items
        try:
            yield
        finally:
            self.discovered = None


def replace_ray(
    pipeline: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    *,
    results: pa.Table | None,
    start_ray: Callable[[contracts.AppendBindings], None] | None = None,
    process: Callable[[execution.ProcessingScope], None] | None = None,
) -> Runs:
    """Keep a recipe's run sequence while replacing its Ray setup, processing plan, and fragment staging.

    ``pipeline`` is the recipe module whose ``run_config`` calls its imported
    ``run_append``. With ``results``, stage them into whichever selection the run
    will process, as a completed plan would. Record each call's remaining recipe
    arguments and outcome, so checks use the callbacks the recipe actually passes.
    ``Runs.discovering`` limits the discovered items without changing the recipe's config.
    """
    runs = Runs()

    def run_append(
        bindings: contracts.AppendBindings,
        discover: Callable[[], pa.Table],
        **kwargs: Any,  # noqa: ANN401
    ) -> execution.AppendOutcome:
        def discover_items() -> pa.Table:
            items = discover()
            if runs.discovered is not None:
                id_column = bindings.spec.id_column
                items = items.filter(pc.is_in(items[id_column], value_set=runs.discovered[id_column].combine_chunks()))
            return items

        def stage(processing: execution.ProcessingScope) -> None:
            if results is None:
                msg = "Processing must not run"
                raise AssertionError(msg)
            stage_results(processing._attempt, results)

        replaced = {
            "start_ray": start_ray or (lambda _bindings: None),
            "process": process or stage,
        }
        outcome = execution.run_append(bindings, discover_items, **{**kwargs, **replaced})
        runs.calls.append((kwargs, outcome))
        return outcome

    monkeypatch.setattr(pipeline, "run_append", run_append)
    monkeypatch.setattr(execution, "stage_append_fragments", stage_locally)
    return runs
