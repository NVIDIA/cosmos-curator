# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Receipt discovery respects retained history and the caller's pinned version."""

from datetime import timedelta
from pathlib import Path
from unittest.mock import Mock

import lance
import pyarrow as pa
import pytest

from cosmos_curator.next.enrichment.transactions import find_transaction


@pytest.fixture
def dataset(tmp_path: Path) -> lance.LanceDataset:
    """Leave two differently identified receipts among unrelated commits."""
    uri = str(tmp_path / "history.lance")
    for version, properties in enumerate(({"receipt": "first"}, {}, {"receipt": "second"}, {}), start=1):
        dataset = lance.write_dataset(
            pa.table({"item_id": [version]}),
            uri,
            mode="create" if version == 1 else "append",
            transaction_properties=properties,
        )
    return dataset


def test_receipt_lookup_respects_identity_and_snapshot_bounds(dataset: lance.LanceDataset) -> None:
    """An old read handle cannot observe a future receipt from the same table."""
    latest = find_transaction(dataset, property_key="receipt")
    assert latest is not None
    assert latest.transaction_properties["receipt"] == "second"
    first = find_transaction(dataset, property_key="receipt", property_value="first")
    assert first is not None
    assert first.transaction_properties["receipt"] == "first"
    assert find_transaction(dataset, property_key="receipt", after_version=3) is None
    assert find_transaction(dataset, property_key="receipt", property_value="first", after_version=1) is None
    assert find_transaction(dataset, property_key="absent") is None
    pinned = lance.dataset(dataset.uri, version=2)
    transaction = find_transaction(pinned, property_key="receipt")
    assert transaction is not None
    assert transaction.transaction_properties["receipt"] == "first"
    assert find_transaction(pinned, property_key="receipt", property_value="second") is None


def test_pruned_gaps_do_not_hide_retained_receipts(
    dataset: lance.LanceDataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cleanup may leave tagged receipts below holes in the retained version history."""
    dataset.tags.create("keep-first", 1)
    dataset.cleanup_old_versions(older_than=timedelta(seconds=0), error_if_tagged_old_versions=False)
    current = lance.dataset(dataset.uri)
    assert {ref["version"] for ref in current.version_refs()} == {1, 4}
    read = Mock(wraps=current.read_transaction)
    monkeypatch.setattr(current, "read_transaction", read)
    transaction = find_transaction(current, property_key="receipt")
    assert transaction is not None
    assert transaction.transaction_properties["receipt"] == "first"
    assert [call.args[0] for call in read.call_args_list] == [4, 1]
    assert find_transaction(current, property_key="receipt", property_value="second") is None


@pytest.mark.parametrize("operation", ["version_refs", "read_transaction"])
def test_storage_failures_are_not_absent_receipts(
    dataset: lance.LanceDataset, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    """A broken retained-history read must not authorize cleanup or a new publication."""
    monkeypatch.setattr(dataset, operation, Mock(side_effect=OSError("storage unavailable")))
    with pytest.raises(OSError, match="storage unavailable"):
        find_transaction(dataset, property_key="receipt")


def test_unchanged_snapshot_needs_no_history_listing(
    dataset: lance.LanceDataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Normal first publication has no newer transaction to reconcile."""
    listing = Mock(side_effect=AssertionError("No history lookup is needed"))
    monkeypatch.setattr(dataset, "version_refs", listing)
    assert find_transaction(dataset, property_key="receipt", after_version=dataset.version) is None
    listing.assert_not_called()
