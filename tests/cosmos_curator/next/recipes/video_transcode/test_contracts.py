# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Transcode provenance lookup remains meaningful after history pruning."""

import json
from datetime import timedelta
from pathlib import Path

import lance
import pyarrow as pa
import pytest

from cosmos_curator.next.recipes.video_transcode.contracts import PUBLICATION_KEY, completed_publication


@pytest.mark.parametrize("keep_receipt", [False, True])
def test_publication_lookup_after_pruning(tmp_path: Path, *, keep_receipt: bool) -> None:
    """A retained receipt survives gaps; a pruned receipt remains unavailable."""
    record = {"publication_version": 1, "contract_id": "test", "spans": 1}
    uri = str(tmp_path / "history.lance")
    dataset = lance.write_dataset(
        pa.table({"item_id": [1]}), uri, transaction_properties={PUBLICATION_KEY: json.dumps(record)}
    )
    if keep_receipt:
        dataset.tags.create("keep-publication", dataset.version)
    dataset = lance.write_dataset(pa.table({"item_id": [2]}), uri, mode="append")
    dataset = lance.write_dataset(pa.table({"item_id": [3]}), uri, mode="append")
    dataset.cleanup_old_versions(older_than=timedelta(seconds=0), error_if_tagged_old_versions=False)
    assert completed_publication(lance.dataset(uri)) == (record if keep_receipt else None)
