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

"""Enforce the sensors library self-containment rule.

No module under cosmos_curator.core.sensors may import from outside that package.
See the design rule documented in cosmos_curator/core/sensors/__init__.py.
"""

import ast
from pathlib import Path

_SENSORS_ROOT = Path(__file__).parents[4] / "cosmos_curator" / "core" / "sensors"
_SENSORS_PREFIX = "cosmos_curator.core.sensors"
_ALLOWED_EXTERNAL_PREFIXES = (
    # stdlib and third-party are fine; only cosmos_curator.* imports are restricted
)


def _external_cosmos_imports(path: Path) -> list[str]:
    """Return any cosmos_curator imports in *path* that are outside the sensors package."""
    rel = path.relative_to(_SENSORS_ROOT.parent.parent.parent)
    tree = ast.parse(path.read_text())
    violations = []
    for node in ast.walk(tree):
        match node:
            case ast.Import(names=names):
                violations.extend(
                    f"{rel}: import {alias.name}"
                    for alias in names
                    if alias.name.startswith("cosmos_curator.") and not alias.name.startswith(_SENSORS_PREFIX)
                )
            case ast.ImportFrom(module=module) if (
                module and module.startswith("cosmos_curator.") and not module.startswith(_SENSORS_PREFIX)
            ):
                violations.append(f"{rel}: from {module} import ...")
    return violations


def test_sensors_library_has_no_external_cosmos_imports() -> None:
    """Every .py file under cosmos_curator/core/sensors/ must only import from within that package."""
    violations: list[str] = []
    for py_file in sorted(_SENSORS_ROOT.rglob("*.py")):
        violations.extend(_external_cosmos_imports(py_file))

    assert violations == [], "sensors library imports from outside cosmos_curator.core.sensors:\n" + "\n".join(
        f"  {v}" for v in violations
    )
