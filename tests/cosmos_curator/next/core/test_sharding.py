# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the shared work-shard assignment primitive.

The partition properties are asserted over a work-unit universe the test owns,
never over a real fragment listing: what makes the assignment correct is that
every id has exactly one owner whatever the ids happen to be.
"""

import pytest
from pydantic import ValidationError

from cosmos_curator.next.core.sharding import ShardConfig

# Deliberately sparse and non-monotonic. Lance fragment ids are dense on a
# producer-written table, but a compaction or a partial rewrite leaves holes,
# and ownership must not depend on density or on the order ids are walked in.
_SPARSE_UNIVERSE = (7, 0, 41, 3, 1_000_003, 12, 8, 999, 4, 40)


def test_a_single_shard_owns_the_whole_universe() -> None:
    """A count of one claims every work unit.

    Equivalent to an unsharded run in what it OWNS, but not interchangeable with
    one: an absent scope and an explicit single shard digest to different
    recovery namespaces (see ``test_workspace``).
    """
    shard = ShardConfig(count=1, index=0)

    assert [unit for unit in _SPARSE_UNIVERSE if shard.owns(unit)] == list(_SPARSE_UNIVERSE)


def test_the_assigned_residue_is_the_one_the_operator_is_told() -> None:
    """Pin WHICH partition, not merely that the assignment is one.

    Every other test here is satisfied by any total assignment into
    ``[0, count)``, so a shifted residue such as ``(unit + 1) % count`` passes
    them all. The run log states the relation to operators as a fact - "the
    fragments whose manifest id mod count equals index" - so the relation itself
    needs pinning, in the module that owns it rather than in one adopter's test.
    """
    assert [unit for unit in range(9) if ShardConfig(count=3, index=1).owns(unit)] == [1, 4, 7]


def test_shards_of_different_counts_overlap() -> None:
    """A shared count is a PRECONDITION of disjointness, not a property of it.

    Nothing detects this at runtime: each scope resolves to its own recovery
    namespace, so two jobs claiming the same work unit stage and commit
    independently. Pinned so the precondition cannot be quietly dropped from the
    documentation that states it.
    """
    coarse = ShardConfig(count=2, index=0)
    fine = ShardConfig(count=4, index=0)

    assert coarse.owns(4)
    assert fine.owns(4)


@pytest.mark.parametrize("count", [2, 3, 8, 16])
def test_every_work_unit_has_exactly_one_owner(count: int) -> None:
    """The shards of one count partition the universe: no gap, no overlap."""
    owners = [
        [unit for unit in _SPARSE_UNIVERSE if ShardConfig(count=count, index=index).owns(unit)]
        for index in range(count)
    ]

    claimed = [unit for owned in owners for unit in owned]
    assert sorted(claimed) == sorted(_SPARSE_UNIVERSE)
    assert len(claimed) == len(set(claimed))


def test_ownership_ignores_the_order_work_units_are_offered_in() -> None:
    """Ownership is a function of the id alone, so a reordered walk owns the same set."""
    shard = ShardConfig(count=4, index=1)

    forward = {unit for unit in _SPARSE_UNIVERSE if shard.owns(unit)}
    reversed_walk = {unit for unit in reversed(_SPARSE_UNIVERSE) if shard.owns(unit)}

    assert forward == reversed_walk
    assert forward, "the fixture must give this shard work, or the equality is vacuous"


def test_a_count_above_the_universe_size_leaves_a_shard_empty() -> None:
    """An index owning nothing is a clean no-op, not an error."""
    universe = (0, 1, 2)
    shard = ShardConfig(count=8, index=5)

    assert [unit for unit in universe if shard.owns(unit)] == []


def test_an_index_at_the_count_is_rejected() -> None:
    """The relation between the two bounded fields is what the validator adds.

    ``index=4`` is a legal value for its own ``ge=0`` bound and ``count=4`` for
    its ``ge=1`` bound, so only the cross-field rule can refuse this pair - and
    it must, because such a shard owns no work unit at any universe size.
    """
    with pytest.raises(ValidationError, match="shard index must be less than count"):
        ShardConfig(count=4, index=4)


def test_a_non_positive_count_is_rejected() -> None:
    """A count of zero has no modulus; index=0 keeps its own bound satisfied."""
    with pytest.raises(ValidationError, match="greater than or equal to 1"):
        ShardConfig(count=0, index=0)


def test_a_negative_index_is_rejected() -> None:
    """A negative index never equals a non-negative remainder; count stays valid."""
    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        ShardConfig(count=4, index=-1)


def test_both_fields_are_required_so_a_half_configured_scope_cannot_resolve() -> None:
    """The pair is atomic: a lone count must fail rather than default to shard 0."""
    with pytest.raises(ValidationError, match="index"):
        ShardConfig.model_validate({"count": 8})


def test_the_label_reports_the_zero_based_scope() -> None:
    """The operator-facing scope string, pinned because it reaches logs and summaries."""
    assert ShardConfig(count=8, index=3).label == "3/8"


def test_an_unknown_field_is_rejected() -> None:
    """Strict config settings hold, so a misspelled scope key cannot be ignored."""
    with pytest.raises(ValidationError, match="extra_forbidden"):
        ShardConfig.model_validate({"count": 8, "index": 3, "indices": 4})
