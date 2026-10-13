# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise conformance with non-video summaries and three fragments whose IDs start above zero."""

from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

import attrs
import lance
import pyarrow as pa
import pytest

from cosmos_curator.next.enrichment import contracts, execution, inputs
from tests.cosmos_curator.next.enrichment.conformance import EnrichmentConformance, RecipeUnderTest
from tests.cosmos_curator.next.enrichment.helpers import make_bindings


@attrs.frozen
class ItemConfig:
    """A synthetic recipe's resolved config: its kind and enrichment bindings."""

    kind: str
    bindings: contracts.EnrichmentBindings


@pytest.fixture
def config(tmp_path: Path, spec: contracts.EnrichmentSpec) -> ItemConfig:
    """Bind the shared item spec under the kind its owned fields name."""
    bindings = make_bindings(
        str(tmp_path / "items.lance"),
        spec,
        staging_root=str(tmp_path / "staging"),
        result_contract={"version": 1},
        commit_attempts=2,
    )
    return ItemConfig(kind="item-recipe", bindings=bindings)


@pytest.fixture
def enrichment_input(config: ItemConfig) -> inputs.EnrichmentInput:
    """Build an immutable input with fragments 1, 2, and 3 before opening any recovery state."""
    dataset = lance.write_dataset(
        pa.Table.from_pylist(
            [{"item_id": f"item-{index}", "uri": f"item-{index}.bin"} for index in range(8)],
            schema=config.bindings.spec.input_fields,
        ),
        config.bindings.uri,
        max_rows_per_file=2,
        data_storage_version="2.2",
    )
    dataset.delete("item_id IN ('item-0', 'item-1')")
    enrichment_input = inputs.open_enrichment_input(config.bindings)
    assert tuple(fragment.fragment_id for fragment in enrichment_input.dataset.get_fragments()) == (1, 2, 3)
    return enrichment_input


@pytest.fixture
def dataset(enrichment_input: inputs.EnrichmentInput) -> lance.LanceDataset:
    """Expose the same pinned table to all conformance checks."""
    return enrichment_input.dataset


@pytest.fixture
def results(enrichment_input: inputs.EnrichmentInput) -> pa.Table:
    """Supply valid scalar results at the actual physical addresses."""
    return pa.Table.from_pylist(
        [
            {
                "fragment_id": fragment.fragment_id,
                "row_offset": offset,
                "item_id": item_id,
                "result": offset,
                "metadata": {"status": "success"},
            }
            for fragment in enrichment_input.dataset.get_fragments()
            for offset, item_id in enumerate(fragment.to_table(columns=["item_id"])["item_id"].to_pylist())
        ],
        schema=enrichment_input.bindings.spec.staged_schema,
    )


def _retune(config: ItemConfig) -> ItemConfig:
    execution_config = config.bindings.execution.model_copy(update={"commit_attempts": 3})
    return attrs.evolve(config, bindings=attrs.evolve(config.bindings, execution=execution_config))


def _recontract(config: ItemConfig) -> ItemConfig:
    spec = attrs.evolve(config.bindings.spec, result_contract={"version": 2})
    return attrs.evolve(config, bindings=attrs.evolve(config.bindings, spec=spec))


@pytest.fixture
def recipe() -> RecipeUnderTest:
    """Run the shared lifecycle with item provenance and no video-specific summary fields."""
    pipeline = ModuleType("item_recipe")
    pipeline.run_enrichment = execution.run_enrichment

    def run_config(config: ItemConfig) -> dict[str, object]:
        outcome = pipeline.run_enrichment(
            config.bindings,
            start_ray=Mock(side_effect=AssertionError("Conformance must replace Ray setup")),
            process=Mock(side_effect=AssertionError("Conformance must replace processing")),
            transaction_properties=lambda _attempt: {"item_recipe.note": "synthetic"},
        )
        return {**outcome.summary(), "items_lance_version": int(outcome.dataset.version)}

    pipeline.run_config = run_config
    return RecipeUnderTest(
        pipeline=pipeline,
        resolve_bindings=lambda config: config.bindings,
        retune=_retune,
        recontract=_recontract,
    )


class TestItemConformance(EnrichmentConformance):
    """The suite supports recipe-specific output names and independently chosen fragment layouts."""
