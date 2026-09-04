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

"""Find the import path from a module to the execution engine.

Planning a session, splitting a camera, naming an artifact and reading bytes are
all things a recipe can do without Ray. Where that holds it is worth defending
rather than remembering: it is what lets a stage be profiled from a plain
script, tested without a cluster, and driven by something other than Ray Data
later. The way it is lost is rarely a stage importing ``ray``; it is a stage
importing a helper that does, which names the engine nowhere -- so this follows
the graph rather than reading a file's own imports.

Modelled on ``tests/cosmos_curator/core/sensors/test_import_boundary.py``, which
enforces the same kind of rule for the sensor library.
"""

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).parents[4]

_ENGINE_PREFIXES = ("ray", "cosmos_xenna")


def _package_of(path: Path) -> tuple[str, ...]:
    """Return the dotted package a file belongs to, as parts.

    A module's package is its directory, so the file's own name goes -- and for
    an ``__init__`` that name *is* the directory's stand-in, so the same drop
    gives the right answer for both.
    """
    try:
        parts = path.relative_to(REPO_ROOT).with_suffix("").parts
    except ValueError:
        # Outside the tree, so it belongs to no first-party package and a
        # relative import from it resolves to nothing this walk can follow.
        return ()
    return parts[:-1]


def _absolute(module: str | None, level: int, package: tuple[str, ...]) -> str:
    """Resolve a possibly relative ``from`` target to a dotted name, or "" if it escapes the tree."""
    if not level:
        return module or ""
    base = package[: len(package) - (level - 1)]
    if not base:
        return ""
    return ".".join((*base, module)) if module else ".".join(base)


def _guards_type_checking(test: ast.expr) -> bool:
    """Say whether an ``if`` guards a block that only a type checker will read.

    The shape of the test, not its text. ``if not TYPE_CHECKING:`` reads as
    ending in ``TYPE_CHECKING`` while meaning the opposite, and a block that runs
    is a block whose imports count.
    """
    match test:
        case ast.Name(id=name):
            return name == "TYPE_CHECKING"
        case ast.Attribute(attr=attribute):
            return attribute == "TYPE_CHECKING"
        case _:
            return False


def _imported_modules(path: Path) -> list[str]:
    """Return the modules *path* imports at import time, dotted and unresolved.

    Module scope only, and both halves of ``from a.b import c``:

    An import inside a function is a dependency of the code path that calls it,
    not of the module, and deferring one is how this codebase already keeps Ray
    out of places that must not need it -- the storage layer defers
    ``nvcf_utils`` precisely so ``S3Prefix`` stays usable from a client-only
    environment with no Ray installed (commit 568985746). A ``TYPE_CHECKING``
    block costs nothing at runtime for the same reason. Counting either would
    condemn the mechanism that solves the problem.

    ``from a.b import c`` records ``a.b.c`` as well as ``a.b``, because ``c`` may
    itself be a module and that is where the dependency lives; recording only
    the package walks into an ``__init__`` that imports nothing and stops a hop
    short of the helper that does the importing.

    A relative import is resolved against the file's own package before either
    of those apply. ``from . import helpers`` gives no module name at all and
    ``from ...utils import x`` gives one that resolves to nothing on its own, so
    left unresolved both leave the walk with a name it cannot follow -- and a
    helper reached only that way would pass unseen, which is the whole failure
    this is meant to catch.
    """
    package = _package_of(path)
    names: list[str] = []

    def visit(nodes: list[ast.AST]) -> None:
        for child in nodes:
            match child:
                case ast.FunctionDef() | ast.AsyncFunctionDef():
                    continue
                case ast.If() if _guards_type_checking(child.test):
                    # The body costs nothing at runtime; the ``else`` is the
                    # runtime path and still counts.
                    visit(list(child.orelse))
                    continue
                case ast.Import(names=aliases):
                    names.extend(alias.name for alias in aliases)
                case ast.ImportFrom(module=module, names=aliases, level=level):
                    absolute = _absolute(module, level, package)
                    if absolute:
                        names.append(absolute)
                        names.extend(f"{absolute}.{alias.name}" for alias in aliases)
            visit(list(ast.iter_child_nodes(child)))

    visit(list(ast.iter_child_nodes(ast.parse(path.read_text(encoding="utf-8")))))
    return names


def _names_an_engine(module: str) -> bool:
    return any(module == prefix or module.startswith(f"{prefix}.") for prefix in _ENGINE_PREFIXES)


def _module_file(module: str) -> Path | None:
    """Return the file backing a ``cosmos_curator`` module, or None for anything else."""
    if not module.startswith("cosmos_curator."):
        return None
    stem = REPO_ROOT / Path(*module.split("."))
    for candidate in (stem.with_suffix(".py"), stem / "__init__.py"):
        if candidate.is_file():
            return candidate
    return None


def engine_import_chain(start: Path) -> list[str] | None:
    """Return the import path from *start* to ``ray`` or ``cosmos_xenna``, or None if it never reaches one.

    The walk follows first-party modules only: third-party is where it stops,
    since the rule is about this codebase's own layering, and the chain is
    returned rather than a bool so a failure names the hop that has to change.
    """
    # Repo-relative where possible, since that is what a failure message reads
    # best as; a file from outside the tree keeps whatever name it came with.
    origin = str(start.relative_to(REPO_ROOT)) if start.is_relative_to(REPO_ROOT) else str(start)
    queue: list[tuple[Path, list[str]]] = [(start, [origin])]
    seen = {start}
    while queue:
        path, chain = queue.pop(0)
        for module in _imported_modules(path):
            if _names_an_engine(module):
                return [*chain, module]
            following = _module_file(module)
            if following is not None and following not in seen:
                seen.add(following)
                queue.append((following, [*chain, module]))
    return None
