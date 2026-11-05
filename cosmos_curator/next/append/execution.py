# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run an append attempt's Ray Data processing and publish its rows in one atomic Lance append.

``run_append`` is the shared run sequence. It creates or validates the target
table, resumes an attempt or discovers and selects new work, starts the recipe's
Ray runtime when items are selected, runs the
recipe's plan inside ``checkpointed_processing`` unless processing already
completed, and publishes. The processing scope owns both ends of that
checkpointed plan: it reads the frozen selection and stages validated item
results inside one native Ray checkpoint. ``publish_append`` expands those
results into packed fragments with Ray Data, appends successful, empty, and
error outcomes together, then removes the verified workspace.
"""

from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from functools import partial, update_wrapper
from typing import Any, cast

import attrs
import lance
import pyarrow as pa
import pyarrow.fs as pafs
import ray
from lance.fragment import FragmentMetadata
from ray.data.block import BlockMetadata
from ray.data.context import DataContext
from ray.data.datasource import Datasource, ReadTask

from cosmos_curator.next.append import commit, contracts
from cosmos_curator.next.append.attempts import AppendAttempt, reconcile_and_select_attempt
from cosmos_curator.next.core.ray_checkpoints import STAGED_FILE_EXTENSION, read_staged_results, read_staged_table
from cosmos_curator.next.core.recovery_workspace import CheckpointWriter, RecoveryWorkspace, checkpoint_writer
from cosmos_curator.next.utils.arrow_utils import arrow_filesystem, filesystem_path

# Staged files the driver reads at once to prove coverage. Each costs an open and
# one read, so object-storage latency, not bandwidth, bounds the check.
_COVERAGE_READ_CONCURRENCY = 32


def _read_items(rows: list[dict[str, Any]], schema: pa.Schema) -> Iterator[pa.Table]:
    yield pa.Table.from_pylist(rows, schema=schema)


class SelectionDatasource(Datasource):
    """Expose an attempt's selected items at Ray's native checkpoint-filter boundary.

    Read tasks carry their items' values, so workers read no recovery state.
    """

    def __init__(self, items: pa.Table) -> None:
        """Capture the frozen selection without reading any media."""
        super().__init__()  # type: ignore[no-untyped-call]
        self._items = items

    def estimate_inmemory_data_size(self) -> int:
        """Estimate the small item records."""
        return int(self._items.nbytes)

    def get_read_tasks(
        self, parallelism: int, per_task_row_limit: int | None = None, data_context: DataContext | None = None
    ) -> list[ReadTask]:
        """Split the selection into at most ``parallelism`` balanced tasks; the scope requests one per item."""
        del data_context
        if parallelism < 1:
            msg = "Read parallelism must be positive"
            raise ValueError(msg)
        total = self._items.num_rows
        count = min(parallelism, total)
        tasks = []
        for index in range(count):
            start, stop = index * total // count, (index + 1) * total // count
            rows = self._items.slice(start, stop - start).to_pylist()
            tasks.append(
                ReadTask(
                    # Ray names a read task over 1 MiB by its read_fn.__name__, which a bare partial lacks.
                    update_wrapper(partial(_read_items, rows, self._items.schema), _read_items),
                    BlockMetadata(num_rows=len(rows), size_bytes=None, input_files=None, exec_stats=None),
                    schema=self._items.schema,
                    per_task_row_limit=per_task_row_limit,
                )
            )
        return tasks


@attrs.frozen
class ProcessingScope:
    """Both ends of one checkpointed plan from an attempt's selected items to staged results.

    Recipes use ``read``, ``write``, and ``result_schema``; ``checkpointed_processing``
    owns the attempt and the checkpoint writer behind them.
    """

    _attempt: AppendAttempt
    _writer: CheckpointWriter

    @property
    def result_schema(self) -> pa.Schema:
        """Return the canonical recipe result schema; the writer owns its storage representation."""
        return self._writer.result_schema

    def read(self, *, num_cpus: float | None = None) -> ray.data.Dataset:
        """Read the selected work items, with checkpoint filtering before expensive work.

        Each item is its own read block, so per-item stages parallelize across
        workers that join later; Ray's default would derive the block count from
        the CPUs present when the plan starts. Pass the first stage's
        ``num_cpus`` when that task stage sets no other remote args: Ray then
        fuses each read into it instead of running a read task per block.
        """
        selection = self._attempt.selection
        return cast(
            "ray.data.Dataset",
            ray.data.read_datasource(
                SelectionDatasource(selection), override_num_blocks=selection.num_rows, num_cpus=num_cpus
            ),
        )

    def write(self, results: ray.data.Dataset) -> None:
        """Validate and stage canonical item results, checkpoint them, then record processing completion.

        Rows must match ``result_schema``, one per item; a checkpointed result can
        always be published.
        """
        self._writer.write(results)


@contextmanager
def checkpointed_processing(attempt: AppendAttempt) -> Iterator[ProcessingScope]:
    """Construct and execute one plan from ``scope.read`` to ``scope.write`` inside a Ray checkpoint.

    The checkpoint keys on the spec's item ID and is retained until the attempt's
    publication is verified.
    """
    workspace = attempt.workspace
    if workspace is None or not attempt.selection.num_rows:
        msg = "Processing requires an attempt with selected items and a recovery workspace"
        raise ValueError(msg)
    bindings = attempt.bindings
    spec = bindings.spec
    with checkpoint_writer(
        workspace,
        id_column=spec.id_column,
        result_schema=spec.result_fields,
        validate=partial(contracts.canonical_results, spec=spec),
        execution=bindings.execution,
    ) as writer:
        yield ProcessingScope(attempt, writer)


def staged_result_files(workspace: RecoveryWorkspace, *, filesystem: pafs.FileSystem) -> list[str]:
    """List the workspace's staged result files as filesystem paths, in a stable order."""
    entries = filesystem.get_file_info(
        pafs.FileSelector(filesystem_path(workspace.results), recursive=True, allow_not_found=True)
    )
    return sorted(
        entry.path
        for entry in entries
        if entry.type == pafs.FileType.File and entry.path.endswith(f".{STAGED_FILE_EXTENSION}")
    )


@attrs.frozen
class AppendOutcome:
    """The table after an attempt, what it appended, and whether its recovery workspace was removed.

    ``errors`` counts this selection's terminal errors. Their diagnostics remain
    in the canonical rows after cleanup and history pruning.
    """

    attempt: AppendAttempt
    dataset: lance.LanceDataset = attrs.field(eq=False)
    cleanup_complete: bool
    appended_rows: int = 0
    errors: int = 0

    def summary(self) -> dict[str, object]:
        """Report the shared run-summary vocabulary; recipes add their own fields."""
        return {
            **self.attempt.counts(),
            "appended_rows": self.appended_rows,
            "errors": self.errors,
            "cleanup_complete": self.cleanup_complete,
            "contract_digest": self.attempt.bindings.spec.contract_digest,
        }


def summary_message(kind: str, summary: Mapping[str, object]) -> str:
    """Render the one-line CLI result shared by append recipes."""
    remaining = summary["remaining_items"]
    pending = (
        "discovery deferred; rerun to check for pending items" if remaining is None else f"{remaining} item(s) pending"
    )
    return (
        f"{kind}: published {summary['selected_items']} item(s) as {summary['appended_rows']} row(s) "
        f"with {summary['errors']} terminal error(s); {pending}"
    )


def _staged_columns(path: str, *, filesystem: pafs.FileSystem, columns: Sequence[str]) -> pa.Table:
    with filesystem.open_input_file(path) as file:
        return read_staged_table(file).select(columns)


def _require_coverage(
    files: Sequence[str], *, filesystem: pafs.FileSystem, spec: contracts.AppendSpec, selected: tuple[str, ...]
) -> int:
    """Check exact staged item coverage file by file and count terminal errors without retaining messages.

    Files are read whole and concurrently; only their ID and error columns are kept.
    """
    columns = [spec.id_column, *([spec.error_column] if spec.error_column is not None else [])]
    read = partial(_staged_columns, filesystem=filesystem, columns=columns)
    remaining = set(selected)
    errors = 0
    with ThreadPoolExecutor(max_workers=_COVERAGE_READ_CONCURRENCY, thread_name_prefix="staged-coverage") as pool:
        for staged in pool.map(read, files):
            ids = staged[spec.id_column].to_pylist()
            if None in ids or len(ids) != len(set(ids)) or not set(ids) <= remaining:
                msg = f"Staged {spec.id_column} results do not exactly cover the selected items"
                raise ValueError(msg)
            remaining.difference_update(ids)
            if spec.error_column is not None:
                errors += staged.num_rows - staged[spec.error_column].null_count
    if remaining:
        msg = f"Staged {spec.id_column} results do not exactly cover the selected items"
        raise ValueError(msg)
    return errors


def _appended_rows(staged: pa.Table, *, spec: contracts.AppendSpec) -> Iterator[pa.Table]:
    """Validate a block of staged item results on a worker and expand it into appended rows."""
    yield from contracts.expanded_rows(contracts.canonical_results(staged, spec), spec)


def stage_append_fragments(
    files: Sequence[str],
    *,
    filesystem: pafs.FileSystem,
    bindings: contracts.AppendBindings,
) -> list[FragmentMetadata]:
    """Expand staged item results, pack rows across items into fragments, and collect only their descriptors.

    The bindings supply validation, expansion, the target, and fragment sizing.
    Rows from different items share fragments; only the last fragment may hold
    fewer than ``rows_per_fragment`` rows.
    """
    if not ray.is_initialized():
        # Ray Data would otherwise start a default local runtime without the
        # recipe's resources or storage environment.
        msg = "Start Ray with the recipe's runtime before publishing selected items"
        raise RuntimeError(msg)
    staged = read_staged_results(list(files), filesystem=filesystem)
    rows = staged.map_batches(partial(_appended_rows, spec=bindings.spec), batch_format="pyarrow", batch_size=None)
    packed = rows.repartition(target_num_rows_per_block=bindings.execution.rows_per_fragment, strict=True)
    descriptors = packed.map_batches(
        partial(commit.stage_fragment, bindings=bindings), batch_format="pyarrow", batch_size=None
    )
    fragments: list[FragmentMetadata] = []
    for batch in descriptors.iter_batches(batch_format="pyarrow", batch_size=None, prefetch_batches=0):
        fragments.extend(
            FragmentMetadata.from_json(fragment) for fragment in cast("pa.Table", batch)["fragment"].to_pylist()
        )
    return fragments


def publish_append(attempt: AppendAttempt) -> AppendOutcome:
    """Append every selected item's outcome atomically, then remove its workspace.

    Publication records the owner as ``kind`` and the result contract digest as
    provenance. Rows carry completion and diagnostics, including for empty/error
    outcomes. An attempt without selected items has nothing to publish. Cleanup
    is best effort and runs only after canonical verification.
    """
    bindings = attempt.bindings
    spec = bindings.spec
    if not attempt.selection.num_rows:
        return AppendOutcome(attempt, attempt.dataset, cleanup_complete=True)
    workspace = attempt.workspace
    if workspace is None:
        msg = "Publication requires the attempt's recovery workspace"
        raise ValueError(msg)
    execution = bindings.execution
    filesystem = arrow_filesystem(workspace.root, storage_profile=execution.storage_profile)
    files = staged_result_files(workspace, filesystem=filesystem)
    errors = _require_coverage(files, filesystem=filesystem, spec=spec, selected=attempt.selected)
    fragments = stage_append_fragments(
        files,
        filesystem=filesystem,
        bindings=bindings,
    )
    published = commit.commit_append(fragments, selected=attempt.selected, dataset=attempt.dataset, bindings=bindings)
    cleaned = workspace.cleanup(storage_profile=execution.storage_profile)
    appended_rows = sum(fragment.num_rows for fragment in fragments)
    return AppendOutcome(attempt, published, cleaned, appended_rows, errors)


def run_append(
    bindings: contracts.AppendBindings,
    discover: Callable[[], pa.Table],
    *,
    start_ray: Callable[[contracts.AppendBindings], None],
    process: Callable[[ProcessingScope], None],
) -> AppendOutcome:
    """Open the target, resume saved work or select pending items, process what is missing, and publish once.

    ``discover`` returns work items in ``input_fields`` only when no unfinished
    selection exists. Resumption needs no source listing and reports unknown
    input and remaining counts.
    ``start_ray`` initializes the recipe's Ray runtime whenever items are
    selected, even when a valid processing marker skips ``process``: publication
    expands and stages fragments with Ray Data. It receives the bindings, whose
    ``storage_env_vars`` cover the table and staging root. ``process`` builds and
    executes the recipe's plan from ``scope.read`` to ``scope.write``. An attempt
    without selected items starts no Ray. ``execution.max_items`` bounds only a
    new selection.
    """
    attempt = reconcile_and_select_attempt(bindings, discover)
    if attempt.selection.num_rows:
        start_ray(bindings)
        if not attempt.processing_complete():
            with checkpointed_processing(attempt) as processing:
                process(processing)
    return publish_append(attempt)
