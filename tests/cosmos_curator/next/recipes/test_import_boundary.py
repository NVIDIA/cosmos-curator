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

"""Keep the execution engine out of every module of a recipe except its driver.

Planning, splitting, reading bytes and naming artifacts should run under a
profiler, under pytest, and one day under something other than Ray Data. They do
today in both recipes checked here: ``ray`` is reached from ``pipeline`` and from
nowhere else. Nothing defended that, and it would not go by a stage importing
``ray``; it would go by a stage importing a helper that does, which is why this
walks the import graph rather than reading a file's own imports.

Two recipes, not all of them. The check in ``import_boundary`` is general and
another recipe adopts it by adding a line here, but the property is not universal
and asserting it on code its owners have not opted into makes the rule arrive as
someone else's blocker.
"""

from pathlib import Path

import pytest

from tests.cosmos_curator.next.recipes.import_boundary import REPO_ROOT, engine_import_chain

_RECIPES_ROOT = REPO_ROOT / "cosmos_curator" / "next" / "recipes"
# Recipe to the modules that own the execution engine, and so the only ones that
# may name it. Recipe-relative paths, not bare names: a recipe may well grow a
# nested pipeline.py, and excluding it by name would exempt a module nobody
# declared. ``pipeline_kind`` is never among them: it is the CLI surface, its
# job is to invoke the driver, and it defers that import so no command pays for
# Ray unless it runs a pipeline.
_DRIVERS = {
    "data_integrity": frozenset({"pipeline.py"}),
    "multimodal_split": frozenset({"pipeline.py"}),
}


@pytest.mark.parametrize("recipe", sorted(_DRIVERS), ids=str)
def test_only_the_driver_reaches_the_execution_engine(recipe: str) -> None:
    """A module that depends on Ray, however indirectly, cannot be profiled or tested without a cluster."""
    drivers = _DRIVERS[recipe]
    violations = [
        " -> ".join(chain)
        for module in sorted((_RECIPES_ROOT / recipe).rglob("*.py"))
        if (
            module.relative_to(_RECIPES_ROOT / recipe).as_posix() not in drivers
            and (chain := engine_import_chain(module)) is not None
        )
    ]

    assert violations == [], (
        f"only {', '.join(sorted(drivers))} may reach the execution engine from {recipe}; hand the module a "
        "callable or a sink the driver installs instead:\n" + "\n".join(f"  {v}" for v in violations)
    )


def _walk(tmp_path: Path, source: str) -> list[str] | None:
    module = tmp_path / "probe.py"
    module.write_text(source, encoding="utf-8")
    return engine_import_chain(module)


def test_a_block_that_runs_counts_however_it_is_guarded(tmp_path: Path) -> None:
    """``TYPE_CHECKING`` is a shape, not a suffix.

    ``if not TYPE_CHECKING:`` reads as ending in the same word while meaning the
    opposite, and the ``else`` of a real guard is the branch that runs. Matching
    on the text hid both, which is a false negative in the direction that
    matters: an import the interpreter executes, unseen.
    """
    assert _walk(tmp_path, "from typing import TYPE_CHECKING\nif not TYPE_CHECKING:\n    import ray\n") is not None
    assert (
        _walk(tmp_path, "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    import x\nelse:\n    import ray\n")
        is not None
    )


def test_an_import_only_a_type_checker_reads_still_costs_nothing(tmp_path: Path) -> None:
    """The reason the guard is honoured at all: four recipe modules use it."""
    assert _walk(tmp_path, "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    import ray\n") is None
    assert _walk(tmp_path, "import typing\nif typing.TYPE_CHECKING:\n    import ray\n") is None
