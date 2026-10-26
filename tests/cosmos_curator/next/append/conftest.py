# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Three discovered items: one appends two rows, one appends none, and one is a terminal error."""

from pathlib import Path

import pyarrow as pa
import pytest

from cosmos_curator.next.append import contracts
from tests.cosmos_curator.next.append.helpers import make_bindings, make_items, make_results, make_spec


@pytest.fixture
def spec() -> contracts.AppendSpec:
    """Exercise list results, empty outcomes, and terminal errors with the same publisher."""
    return make_spec()


@pytest.fixture
def bindings(tmp_path: Path, spec: contracts.AppendSpec) -> contracts.AppendBindings:
    """Bind a local table and staging root; two rows per fragment make packing observable."""
    return make_bindings(str(tmp_path / "rows.lance"), spec, staging_root=str(tmp_path / "staging"))


@pytest.fixture
def items() -> pa.Table:
    """Discover the items out of ID order, as discovery may."""
    return make_items(["c", "a", "b"])


@pytest.fixture
def results(spec: contracts.AppendSpec) -> pa.Table:
    """Build one canonical terminal result per item: two values, none, and an error."""
    return make_results(spec, {"a": [1, 2], "b": [], "c": None})
