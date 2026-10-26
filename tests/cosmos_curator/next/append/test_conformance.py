# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise conformance with non-video summaries, terminal errors, and items discovered out of order."""

from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

import attrs
import pyarrow as pa
import pytest

from cosmos_curator.next.append import contracts, execution
from tests.cosmos_curator.next.append.conformance import AppendConformance, RecipeUnderTest
from tests.cosmos_curator.next.append.helpers import make_bindings, make_items, make_results


@attrs.frozen
class ItemConfig:
    """A synthetic recipe's resolved config: its kind, append bindings, and discovered items."""

    kind: str
    bindings: contracts.AppendBindings
    items: pa.Table = attrs.field(eq=False)


@pytest.fixture
def config(tmp_path: Path, spec: contracts.AppendSpec) -> ItemConfig:
    """Discover four items out of ID order under the kind the spec's owner names."""
    bindings = make_bindings(str(tmp_path / "items.lance"), spec, staging_root=str(tmp_path / "staging"))
    return ItemConfig(kind="item-recipe", bindings=bindings, items=make_items(["d", "a", "c", "b"]))


@pytest.fixture
def results(spec: contracts.AppendSpec) -> pa.Table:
    """Two items append rows, one appends none, and one is a terminal error."""
    return make_results(spec, {"a": [1, 2], "b": [], "c": None, "d": [3, 4, 5]})


def _retune(config: ItemConfig) -> ItemConfig:
    execution_config = config.bindings.execution.model_copy(update={"parquet_rows_per_file": 3, "rows_per_fragment": 3})
    return attrs.evolve(config, bindings=attrs.evolve(config.bindings, execution=execution_config))


def _recontract(config: ItemConfig) -> ItemConfig:
    spec = attrs.evolve(config.bindings.spec, result_contract={"version": 2})
    return attrs.evolve(config, bindings=attrs.evolve(config.bindings, spec=spec))


@pytest.fixture
def recipe() -> RecipeUnderTest:
    """Run the shared lifecycle with no video-specific summary fields."""
    pipeline = ModuleType("item_recipe")
    pipeline.run_append = execution.run_append

    def run_config(config: ItemConfig) -> dict[str, object]:
        outcome = pipeline.run_append(
            config.bindings,
            lambda: config.items,
            start_ray=Mock(side_effect=AssertionError("Conformance must replace Ray setup")),
            process=Mock(side_effect=AssertionError("Conformance must replace processing")),
        )
        return {**outcome.summary(), "items_lance_version": int(outcome.dataset.version)}

    pipeline.run_config = run_config
    return RecipeUnderTest(
        pipeline=pipeline,
        resolve_bindings=lambda config: config.bindings,
        discover=lambda config: config.items,
        retune=_retune,
        recontract=_recontract,
    )


class TestItemConformance(AppendConformance):
    """The suite supports terminal errors and arbitrary discovery order."""
