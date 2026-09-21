# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Assign durable work units to one of several independently launched jobs.

This module answers exactly one question - WHICH WORK UNITS BELONG TO THIS SHARD
- so that several jobs dividing one logical run derive their scope from a single
ownership function rather than from one implementation each. A divergent
ownership function is a correctness defect (gaps and overlaps between jobs),
which is why the assignment is shared even while there is one consumer; a
duplicated declaration would be harmless and is not what this module exists for.

Publication safety, commit semantics, and recovery identity stay recipe-local,
and adopting this type does NOT make a recipe safe to shard. Two questions a
recipe must answer for itself first:

- Does its checkpoint id stay unique AND interpretable across every mutation a
  multi-job campaign permits? A physical row address does not.
- Are its work units genuinely independent? A recipe whose result depends on a
  corpus-wide reduction produces different output when its rows are divided,
  silently.

See ``docs/curator/design/curator-next-sharding.md`` for the mechanism, the
terminology it fixes (work shard versus Ray Data block versus Lance fragment),
how it composes with Ray Data's job-level checkpointing, and the checklist a
recipe works through before adopting it. A worked adoption, with its correctness
argument and operator workflow, belongs to the adopting recipe's own design
document.
"""

from typing import Self

from pydantic import BaseModel, Field, model_validator

from cosmos_curator.core.utils.config.pydantic_config import STRICT_CONFIG_MODEL_CONFIG


class ShardConfig(BaseModel):
    """One job's execution scope over a stable work-unit universe.

    Both fields are required so the pair is atomic: a half-configured scope would
    silently run shard 0 while the remaining work was claimed by no job at all.
    The enclosing config makes the whole object optional instead, where absent
    means unsharded.

    Execution scope, never result semantics: which job produced a work unit must
    not reach what that unit produces, so this value belongs in a run's logs,
    summary, and recovery namespace and in no result digest.

    Attributes:
        count: Total shards the work universe is divided into.
        index: Zero-based index of the shard this job owns.

    """

    model_config = STRICT_CONFIG_MODEL_CONFIG

    count: int = Field(ge=1, description="Total shards the work universe is divided into.")
    index: int = Field(ge=0, description="Zero-based index of the shard this job owns.")

    @model_validator(mode="after")
    def _require_index_within_count(self) -> Self:
        """Reject an index outside the partition the count declares."""
        # Each field is independently bounded by its own ``Field`` constraint;
        # the RELATION between them is not, and an out-of-range index owns no
        # work unit at all - a job that starts, finds nothing, and exits zero.
        if self.index >= self.count:
            msg = f"shard index must be less than count, got index={self.index} count={self.count}"
            raise ValueError(msg)
        return self

    def owns(self, work_unit_id: int) -> bool:
        """Return whether this shard owns the given stable work-unit id.

        Modulo over the id, so ownership is a pure function of the UNIVERSE and
        needs no manifest, no shared state, and no agreement between jobs beyond
        the count. Two jobs planning at different moments therefore derive the
        same partition: across the ``count`` shards of ONE count, every id has
        exactly one owner, the shards are pairwise disjoint, and their union is
        the whole universe.

        A shared ``count`` is a precondition of that disjointness, not a
        property of it. Shards of DIFFERENT counts overlap - ``count=2,index=0``
        and ``count=4,index=0`` both claim every id divisible by four - and
        nothing detects it, because each resolves to its own recovery namespace
        and so stages and commits independently. Hold ``count`` fixed for a
        campaign.
        """
        return work_unit_id % self.count == self.index

    @property
    def label(self) -> str:
        """Return the human-readable scope, zero-based, as ``"3/8"``."""
        return f"{self.index}/{self.count}"

    def identity(self) -> dict[str, int]:
        """Return the scope as it enters a consumer's recovery-identity digest.

        Owned here rather than spelled out at the call site so that a field
        added to this model reaches every digest keyed on it. The failure a
        consumer-side field list invites is silent and unrecoverable: two scopes
        digesting equal share one staging directory, where whichever publishes
        first deletes the other's staged work.
        """
        return self.model_dump(mode="json")
