# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Find publication evidence in the retained history of a pinned Lance snapshot."""

from collections.abc import Iterator

import lance


def iter_transactions(
    dataset: lance.LanceDataset,
    *,
    property_key: str,
    property_value: str | None = None,
    after_version: int = 0,
) -> Iterator[lance.Transaction]:
    """Yield matching transactions newest first, at or before the pinned version.

    Enumerate retained version references so historical pruning does not leave
    holes in an integer walk. A missing match is not publication proof. Errors
    reading listed versions still propagate, including storage failures or a
    concurrent prune; callers must not interpret those errors as an absent receipt.
    """
    pinned_version = int(dataset.version)
    if after_version >= pinned_version:
        return
    versions = sorted(
        (int(ref["version"]) for ref in dataset.version_refs() if after_version < ref["version"] <= pinned_version),
        reverse=True,
    )
    for version in versions:
        transaction = dataset.read_transaction(version)
        if transaction is None:
            continue
        properties = transaction.transaction_properties or {}
        if property_key in properties and (property_value is None or properties[property_key] == property_value):
            yield transaction


def find_transaction(
    dataset: lance.LanceDataset,
    *,
    property_key: str,
    property_value: str | None = None,
    after_version: int = 0,
) -> lance.Transaction | None:
    """Find the newest retained receipt matching the identity and snapshot bounds."""
    return next(
        iter_transactions(
            dataset, property_key=property_key, property_value=property_value, after_version=after_version
        ),
        None,
    )
