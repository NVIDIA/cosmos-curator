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

"""Utilities for working with PyArrow tables."""

import pyarrow as pa  # type: ignore[import-untyped]


def with_column(table: pa.Table, name: str, col: pa.Array | pa.ChunkedArray) -> pa.Table:
    """Return *table* with *name* set to *col*, replacing it if it already exists."""
    if name in table.column_names:
        table = table.drop(name)
    return table.append_column(name, col)
