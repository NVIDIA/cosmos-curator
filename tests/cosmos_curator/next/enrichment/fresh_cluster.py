# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fresh-cluster recovery scenario that every adopting recipe runs on real Ray.

The conformance suite checks shared recovery without Ray; this scenario checks that a
recipe's actual Ray Data plan is checkpointable. An ``@pytest.mark.integration`` test
calls ``run_in_fresh_driver`` with a module-level function of its test module. That
function installs stand-ins for the recipe's expensive stages, each recording the IDs it
processes, and calls ``assert_fresh_cluster_resume`` with the recipe's ``RecipeUnderTest``.
The recipe's ``retune`` must stay schedulable on the scenario's small local cluster.
"""

import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import attrs
import lance
import pyarrow as pa
import pyarrow.parquet as pq
import ray
from coverage import Coverage
from ray.data.context import DataContext

from cosmos_curator.core.utils import environment
from cosmos_curator.next.enrichment import inputs
from cosmos_curator.next.enrichment.attempts import reconcile_and_select_attempt
from cosmos_curator.next.enrichment.execution import ProcessingScope, checkpointed_processing
from tests.cosmos_curator.next.enrichment.conformance import RecipeUnderTest

# Own both Ray clusters in a child process so restarting them cannot replace the
# suite's shared cluster. Save child coverage alongside pytest-cov's data when enabled.
_DRIVER = """
import importlib
import sys

import coverage

module, function, data_file, *args = sys.argv[1:]
cov = coverage.Coverage(data_file=data_file, data_suffix=True) if data_file else None
if cov is not None:
    cov.start()
try:
    getattr(importlib.import_module(module), function)(*args)
finally:
    if cov is not None:
        cov.stop()
        cov.save()
"""


def run_in_fresh_driver(target: str, *args: str, timeout: float = 240) -> None:
    """Run ``module:function`` with string arguments in a new driver process that owns its Ray clusters."""
    module, function = target.split(":")
    active_coverage = Coverage.current()
    data_file = active_coverage.get_option("run:data_file") if active_coverage is not None else ""
    result = subprocess.run(  # noqa: S603 -- fixed driver with test-owned arguments
        [sys.executable, "-c", _DRIVER, module, function, str(data_file or ""), *args],
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def start_local_cluster(num_cpus: int) -> None:
    """Start a small local cluster with one IO slot, so every IO-bound stage proves it holds the slot."""
    ray.init(
        address="local",
        num_cpus=num_cpus,
        num_gpus=0,
        resources={environment.CURATOR_IO_RESOURCE_NAME: 1},
        include_dashboard=False,
        log_to_driver=False,
        object_store_memory=128 * 1024**2,
    )
    DataContext.get_current().enable_progress_bars = False


def _selected_ids(dataset: lance.LanceDataset, id_column: str, selected: Sequence[int]) -> list[str]:
    return sorted(
        value
        for fragment_id in selected
        for value in dataset.get_fragment(fragment_id).to_table(columns=[id_column])[id_column].to_pylist()
    )


def assert_fresh_cluster_resume(
    recipe: RecipeUnderTest,
    config: Any,  # noqa: ANN401 -- each recipe's resolved config
    *,
    process: Callable[[ProcessingScope, Any], None],
    work_logs: Callable[[], Mapping[str, Sequence[str]]],
    num_cpus: int,
    before_resume: Callable[[], None] = lambda: None,
) -> dict[str, Any]:
    """Interrupt a recipe's Ray plan, resume it on a fresh cluster, and publish through ``run_config``.

    1. Processing only the first selected fragment leaves its results durable and every
       selected fragment pending in Lance.
    2. After a cluster restart, a disjoint column write, and ``recipe.retune``, recovery
       keeps the saved selection and workspace. ``before_resume`` may then change recipe
       state, such as removing media.
    3. A fresh cluster finishes the selection, and each expensive step ran exactly once per ID.
    4. With the processing marker removed, the recipe's own ``run_config`` reprocesses
       nothing, commits once, keeps the disjoint column, and removes the workspace.

    ``process`` runs the recipe's plan for a scope and config; ``work_logs`` returns the IDs
    each expensive step has processed so far. Returns the run summary for recipe checks.
    """
    bindings = recipe.resolve_bindings(config)
    id_column = bindings.spec.id_column
    attempt = reconcile_and_select_attempt(inputs.open_enrichment_input(bindings))
    workspace = attempt.workspace
    assert workspace is not None
    assert len(attempt.selected) > 1, "The scenario interrupts after the first of several selected fragments"
    dataset = attempt.input.dataset
    selected_ids = _selected_ids(dataset, id_column, attempt.selected)
    try:
        start_local_cluster(num_cpus)
        with checkpointed_processing(attrs.evolve(attempt, selected=attempt.selected[:1])) as processing:
            process(processing, config)
        assert DataContext.get_current().checkpoint_config is None
        staged = pq.read_table(workspace.results, columns=[id_column])[id_column].to_pylist()
        assert sorted(staged) == _selected_ids(dataset, id_column, attempt.selected[:1])
        assert inputs.pending_fragment_ids(lance.dataset(dataset.uri), bindings) == attempt.selected
        # Simulate a crash after some results are durable, before the processing marker.
        Path(workspace.completion).unlink()
        ray.shutdown()

        lance.dataset(dataset.uri).add_columns(pa.schema([pa.field("disjoint_enrichment", pa.string())]))
        tuned = recipe.retune(config)
        resumed = reconcile_and_select_attempt(inputs.open_enrichment_input(recipe.resolve_bindings(tuned)))
        assert (resumed.selected, resumed.workspace) == (attempt.selected, workspace)
        before_resume()
        start_local_cluster(num_cpus)
        with checkpointed_processing(resumed) as processing:
            process(processing, tuned)
        assert DataContext.get_current().checkpoint_config is None
        staged = pq.read_table(workspace.results, columns=[id_column])[id_column].to_pylist()
        assert sorted(staged) == selected_ids
        work = {step: sorted(ids) for step, ids in work_logs().items()}
        assert work, "Stand-ins must record the IDs each expensive step processes"
        assert all(ids == selected_ids for ids in work.values()), work
        assert resumed.processing_complete()

        Path(workspace.completion).unlink()
        summary = recipe.pipeline.run_config(tuned)
        assert {step: sorted(ids) for step, ids in work_logs().items()} == work
        published = lance.dataset(dataset.uri)
        assert published.version == resumed.input.dataset.version + 1
        assert (summary["selected_fragments"], summary["selected_rows"]) == (len(attempt.selected), len(selected_ids))
        assert (summary["remaining_rows"], summary["cleanup_complete"]) == (0, True)
        assert not Path(workspace.root).exists()
        assert "disjoint_enrichment" in published.schema.names
        assert inputs.pending_fragment_ids(published, bindings) == ()
        return summary
    finally:
        ray.shutdown()
