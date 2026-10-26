# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fresh-cluster recovery scenario that every adopting append recipe runs on real Ray.

The conformance suite checks shared recovery without Ray; this scenario checks that a
recipe's actual Ray Data plan is checkpointable. An ``@pytest.mark.integration`` test
calls ``run_in_fresh_driver`` from ``tests/cosmos_curator/next/fresh_driver.py`` with a
module-level function of its test module. That function installs stand-ins for the
recipe's expensive stages, each recording the item IDs it processes, and calls
``assert_fresh_cluster_resume`` with the recipe's ``RecipeUnderTest``. The recipe's
``retune`` must stay schedulable on the scenario's small local cluster.
"""

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import attrs
import lance
import pyarrow as pa
import pyarrow.parquet as pq
import ray
from ray.data.context import DataContext

from cosmos_curator.next.append import inputs
from cosmos_curator.next.append.attempts import reconcile_and_select_attempt
from cosmos_curator.next.append.execution import ProcessingScope, checkpointed_processing
from tests.cosmos_curator.next.append.conformance import RecipeUnderTest
from tests.cosmos_curator.next.fresh_driver import start_local_cluster


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

    1. Processing only the first selected item leaves its result durable and nothing
       published in Lance.
    2. After a cluster restart, a disjoint column write, and ``recipe.retune``, recovery
       keeps the saved selection and workspace. ``before_resume`` may then change recipe
       state, such as removing media.
    3. A fresh cluster finishes the selection, and each expensive step ran exactly once per item.
    4. With the processing marker removed, the recipe's own ``run_config`` reprocesses
       nothing, appends once, keeps the disjoint column, and removes the workspace.

    ``process`` runs the recipe's plan for a scope and config; ``work_logs`` returns the
    item IDs each expensive step has processed so far. Returns the run summary for recipe checks.
    """
    bindings = recipe.resolve_bindings(config)
    id_column = bindings.spec.id_column
    attempt = reconcile_and_select_attempt(bindings, lambda: recipe.discover(config))
    workspace = attempt.workspace
    assert workspace is not None
    assert len(attempt.selected) > 1, "The scenario interrupts after the first of several selected items"
    try:
        start_local_cluster(num_cpus)
        with checkpointed_processing(attrs.evolve(attempt, selection=attempt.selection.slice(0, 1))) as processing:
            process(processing, config)
        assert DataContext.get_current().checkpoint_config is None
        staged = pq.read_table(workspace.results, columns=[id_column])[id_column].to_pylist()
        assert staged == list(attempt.selected[:1])
        assert inputs.published_item_ids(lance.dataset(bindings.uri), bindings, attempt.selected).isdisjoint(
            attempt.selected
        )
        # Simulate a crash after some results are durable, before the processing marker.
        Path(workspace.completion).unlink()
        ray.shutdown()

        lance.dataset(bindings.uri).add_columns(pa.schema([pa.field("disjoint_enrichment", pa.string())]))
        tuned = recipe.retune(config)
        resumed = reconcile_and_select_attempt(recipe.resolve_bindings(tuned), lambda: recipe.discover(tuned))
        assert (resumed.selected, resumed.workspace) == (attempt.selected, workspace)
        before_resume()
        start_local_cluster(num_cpus)
        with checkpointed_processing(resumed) as processing:
            process(processing, tuned)
        assert DataContext.get_current().checkpoint_config is None
        staged = pq.read_table(workspace.results, columns=[id_column])[id_column].to_pylist()
        assert sorted(staged) == list(attempt.selected)
        work = {step: sorted(ids) for step, ids in work_logs().items()}
        assert work, "Stand-ins must record the item IDs each expensive step processes"
        assert all(ids == list(attempt.selected) for ids in work.values()), work
        assert resumed.processing_complete()

        Path(workspace.completion).unlink()
        summary = recipe.pipeline.run_config(tuned)
        assert {step: sorted(ids) for step, ids in work_logs().items()} == work
        published = lance.dataset(bindings.uri)
        assert published.version == resumed.dataset.version + 1
        assert (summary["selected_items"], summary["remaining_items"]) == (len(attempt.selected), None)
        assert summary["cleanup_complete"] is True
        assert not Path(workspace.root).exists()
        assert "disjoint_enrichment" in published.schema.names
        assert inputs.published_item_ids(published, bindings, attempt.selected) >= set(attempt.selected)
        return summary
    finally:
        ray.shutdown()
