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

"""Guard against subprocesses that silently inherit a controlling terminal.

Ray (>=2.57) puts each worker in its own process group. If the process tree has a
controlling terminal, those groups are *background* groups relative to it, and the kernel
stops any background process that reads stdin by sending it SIGTTIN. A stopped ffmpeg never
returns, so the owning stage sits at 0% CPU with zero completions and no error -- the
failure mode this guard exists to prevent.

The local launcher disconnects stdin before starting Ray, while retaining terminal output
for progress bars (see ``test_launch_local.test_launch_command``). Library code also runs
under interactive ``ssh -t`` and ``srun --pty`` paths in ``slurm_shell``, so every spawn must
decide stdin explicitly rather than inherit it.

This is a structural guard, not a behavioral test: regex sweeps of this codebase have twice
missed a spawn helper (``check_call``), and a new unguarded call site arrived from ``main``
days after the original fix. Parsing the AST catches call sites that string matching does
not.
"""

import ast
from collections.abc import Iterator
from pathlib import Path

_REPO_ROOT = Path(__file__).parents[1]
_PACKAGE_ROOT = _REPO_ROOT / "cosmos_curator"

# ``subprocess`` helpers that spawn a child process.
_SPAWN_ATTRS = frozenset({"run", "call", "check_call", "check_output", "Popen"})

# Passing either keyword replaces the inherited descriptor: ``input`` implies a pipe.
_STDIN_KWARGS = frozenset({"stdin", "input"})

# Launcher and driver entrypoints run in the foreground on a developer's terminal, where
# inheriting stdin is intentional (``docker run``, ``ssh``, ``srun``, ``docker buildx``).
# They are not worker code, so SIGTTIN does not apply to them.
_FOREGROUND_PATHS = (
    "cosmos_curator/client/",
    "cosmos_curator/scripts/",
    "cosmos_curator/core/cf/nvcf_main.py",
)


def _library_sources() -> Iterator[tuple[Path, str]]:
    """Yield (path, repo-relative posix path) for library modules subject to the guard."""
    for path in sorted(_PACKAGE_ROOT.rglob("*.py")):
        relative = path.relative_to(_REPO_ROOT).as_posix()
        if relative.startswith(_FOREGROUND_PATHS):
            continue
        yield path, relative


def _spawn_helper_name(node: ast.Call) -> str | None:
    """Return the helper name when the call is ``subprocess.<spawn helper>(...)``, else None."""
    func = node.func
    if not isinstance(func, ast.Attribute) or func.attr not in _SPAWN_ATTRS:
        return None
    if isinstance(func.value, ast.Name) and func.value.id == "subprocess":
        return func.attr
    return None


def _decides_stdin(node: ast.Call) -> bool:
    """Return True when the call sets stdin, or forwards kwargs we cannot inspect statically."""
    return any(keyword.arg is None or keyword.arg in _STDIN_KWARGS for keyword in node.keywords)


def _scan_spawns() -> tuple[int, list[str]]:
    """Return the total number of library spawn sites and the ones that leave stdin inherited."""
    total = 0
    offenders = []
    for path, relative in _library_sources():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            helper = _spawn_helper_name(node)
            if helper is None:
                continue
            total += 1
            if not _decides_stdin(node):
                offenders.append(f"{relative}:{node.lineno} subprocess.{helper}")
    return total, offenders


def _direct_subprocess_imports() -> list[str]:
    """Return ``path:line`` for imports that would route around the AST check."""
    offenders = []
    for path, relative in _library_sources():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "subprocess":
                bound = sorted(alias.name for alias in node.names if alias.name in _SPAWN_ATTRS)
                if bound:
                    offenders.append(f"{relative}:{node.lineno} from subprocess import {', '.join(bound)}")
            elif isinstance(node, ast.Import):
                offenders.extend(
                    f"{relative}:{node.lineno} import subprocess as {alias.asname}"
                    for alias in node.names
                    if alias.name == "subprocess" and alias.asname
                )
    return offenders


def test_guard_actually_scans_the_package() -> None:
    """Fail loudly rather than passing vacuously when the package tree is not where we expect.

    The checks below locate sources relative to this file. Run outside a full checkout -- a
    copied-out test file, a trimmed sdist -- the globs would match nothing and every check
    would report success without having read a single line of code.
    """
    assert _PACKAGE_ROOT.is_dir(), f"expected the package at {_PACKAGE_ROOT}; run these tests from a full checkout"
    total, _ = _scan_spawns()
    assert total > 0, (
        f"scanned {sum(1 for _ in _library_sources())} modules under {_PACKAGE_ROOT} but found no "
        "subprocess spawn sites, so the stdin guard below cannot be proving anything"
    )


def test_library_subprocess_calls_decide_stdin() -> None:
    """Every spawn outside launcher code must pass stdin (usually ``subprocess.DEVNULL``) or input."""
    _, offenders = _scan_spawns()
    assert not offenders, (
        "These subprocess calls inherit stdin and will be stopped with SIGTTIN if the process tree "
        "has a controlling terminal, hanging the stage with no error:\n  "
        + "\n  ".join(offenders)
        + "\n\nPass stdin=subprocess.DEVNULL (or input=...) at each site. If the call is a foreground "
        f"launcher entrypoint that should inherit the terminal, add its path to _FOREGROUND_PATHS in "
        f"{Path(__file__).name}."
    )


def test_subprocess_spawn_helpers_are_not_imported_directly() -> None:
    """Bare or aliased spawn imports would bypass the check above, so keep using ``subprocess.<helper>``."""
    offenders = _direct_subprocess_imports()
    assert not offenders, (
        "These imports hide subprocess spawns from the stdin guard:\n  "
        + "\n  ".join(offenders)
        + "\n\nImport the module and call subprocess.<helper>(...) instead."
    )
