# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A two-fragment item table with scalar and nested outputs, shared by the publisher tests."""

from pathlib import Path

import lance
import pyarrow as pa
import pytest

from cosmos_curator.next.enrichment import contracts, inputs
from tests.cosmos_curator.next.enrichment.helpers import make_bindings, validate_items


@pytest.fixture
def spec() -> contracts.EnrichmentSpec:
    """Exercise scalar outputs and nested metadata with the same publisher."""
    items = pa.schema([pa.field("item_id", pa.string(), nullable=False), ("uri", pa.string())])
    fields = contracts.owned_fields(
        pa.schema([("result", pa.int64()), ("metadata", pa.struct([("status", pa.string())]))]),
        owner="item-recipe",
        field_set="result",
    )
    return contracts.EnrichmentSpec(
        id_column="item_id",
        input_fields=items,
        output_fields=fields,
        result_contract={},
        validate_batch=validate_items,
    )


@pytest.fixture
def dataset(tmp_path: Path, spec: contracts.EnrichmentSpec) -> lance.LanceDataset:
    """Keep the input across two fragments so coverage is observable."""
    uri = str(tmp_path / "input.lance")
    lance.write_dataset(
        pa.Table.from_pylist(
            [{"item_id": "a", "uri": "a.mp4"}, {"item_id": "b", "uri": "b.mp4"}], schema=spec.input_fields
        ),
        uri,
        max_rows_per_file=1,
        data_storage_version="2.2",
    )
    return inputs.register_fields(make_bindings(uri, spec))
