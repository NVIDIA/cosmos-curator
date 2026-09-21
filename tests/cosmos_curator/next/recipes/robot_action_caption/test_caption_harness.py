# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The harness's own reporting contracts for fake-engine and digest observations.

The ledger is how every recovery test answers WHICH CLIPS REACHED THE ENGINE, so
a ledger that merged two clips into one entry would report less work than really
happened - and an assertion that a clip was not re-sent on a retry would pass
because the harness, not the recipe, decided the answer. The filename contract
therefore needs a test of its own, which no other module can carry: the recipe
never reads the ledger, and ``test_recovery_integration.py`` runs whole real
attempts whose clip ids come from a table it does not choose them for.

The engine stage under test is the real one, because the name it writes is half
of the contract the ledger reads. Only the ``Dataset`` it maps over stands in,
which is what keeps this off a Ray cluster; the recipe's preprocess and
postprocess hooks cannot affect a filename, so identity hooks stand in for them.

The digest readers are assertion inputs too. They fail closed on a null
``caption_contract_digest`` rather than manufacturing the plausible string
``"None"``, which could hide an unfinished row from a recovery assertion.
"""

import pathlib
import types
from collections.abc import Callable
from typing import Any

import lance
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cosmos_curator.next.recipes.robot_action_caption.contracts import (
    CAPTION_COLUMN,
    CAPTION_DIGEST_COLUMN,
    CAPTION_ERROR_COLUMN,
    CAPTION_MODEL_COLUMN,
    CAPTION_STATUS_COLUMN,
)
from cosmos_curator.next.recipes.robot_action_caption.datasource import (
    CHECKPOINT_ID_COLUMN,
    FRAGMENT_ID_COLUMN,
)
from cosmos_curator.next.recipes.robot_action_caption.request import STAGED_SCHEMA
from cosmos_curator.next.recipes.robot_action_caption.table import CLIP_ID_COLUMN
from tests.cosmos_curator.next.conftest import row_address
from tests.cosmos_curator.next.recipes.robot_action_caption import caption_harness

type DigestTableFactory = Callable[[list[str | None]], str]


class _SingleBatch:
    """Stands in for the ``Dataset`` the fake transform maps over, applying the stage once.

    The transform's only use of its argument is ``map_batches``, so handing the
    batch straight to the stage runs the real engine - and its real ledger
    write - without starting Ray.
    """

    def __init__(self, batch: pa.Table) -> None:
        self._batch = batch

    def map_batches(self, engine: Any, **_kwargs: Any) -> pa.Table:  # noqa: ANN401 - the stage callable
        return engine(self._batch)


def _record_invocations(ledger: caption_harness.InferenceLedger, clip_ids: tuple[str, ...]) -> None:
    """Drive the fake engine stage once over one pending row per clip id.

    Only the three identity columns are seeded. The caption and generation
    columns the stage's output schema declares are left for it to fill or null,
    because what is under test is the name the invocation is recorded under.
    """
    batch = pa.table(
        {
            CLIP_ID_COLUMN: pa.array(list(clip_ids), pa.string()),
            FRAGMENT_ID_COLUMN: pa.array([0] * len(clip_ids), pa.int64()),
            CHECKPOINT_ID_COLUMN: pa.array([row_address(0, offset) for offset in range(len(clip_ids))], pa.uint64()),
        }
    )
    transform = caption_harness.fake_transform_builder(ledger=ledger)(
        types.SimpleNamespace(batch_size=len(clip_ids)),
        preprocess=dict,
        postprocess=dict,
    )
    transform(_SingleBatch(batch))


@pytest.fixture
def staged_digest_table(tmp_path: pathlib.Path) -> DigestTableFactory:
    """Return a writer for staged rows carrying caller-selected digests."""

    def write(digests: list[str | None]) -> str:
        result_dir = tmp_path / "inference" / "namespace" / "results" / "fragment_id=0"
        result_dir.mkdir(parents=True)
        rows = [
            {
                CLIP_ID_COLUMN: f"c{index}",
                FRAGMENT_ID_COLUMN: 0,
                CHECKPOINT_ID_COLUMN: row_address(0, index),
                CAPTION_COLUMN: f"caption {index}",
                CAPTION_STATUS_COLUMN: "success",
                CAPTION_ERROR_COLUMN: None,
                CAPTION_DIGEST_COLUMN: digest,
                CAPTION_MODEL_COLUMN: "test/model",
            }
            for index, digest in enumerate(digests)
        ]
        pq.write_table(pa.Table.from_pylist(rows, schema=STAGED_SCHEMA), result_dir / "part.parquet")
        return str(tmp_path)

    return write


@pytest.fixture
def published_digest_table(tmp_path: pathlib.Path) -> DigestTableFactory:
    """Return a writer for caption tables carrying caller-selected digests."""

    def write(digests: list[str | None]) -> str:
        uri = str(tmp_path / "clips.lance")
        lance.write_dataset(pa.table({CAPTION_DIGEST_COLUMN: pa.array(digests, pa.string())}), uri)
        return uri

    return write


def test_the_ledger_keeps_dotted_clip_ids_that_share_a_prefix_apart(tmp_path: pathlib.Path) -> None:
    """Each clip the engine saw is reported once, whatever its id contains.

    The two dotted ids share everything before their dot, so a reader that split
    the recorded name at the FIRST dot rather than the last would report the one
    entry ``clip`` for both. ``c0`` is the shape every other test uses and is
    here as the control: without it the assertion could be satisfied by a reader
    that had stopped parsing names at all.
    """
    ledger = caption_harness.InferenceLedger(directory=tmp_path / "engine-calls")
    ledger.directory.mkdir()

    _record_invocations(ledger, ("clip.01", "clip.02", "c0"))

    assert ledger.clip_ids == ["c0", "clip.01", "clip.02"]


def test_staged_rows_rejects_a_null_digest_before_constructing_the_row(
    staged_digest_table: DigestTableFactory,
) -> None:
    """Reject a missing staged identity before it can become a ``StagedRow``."""
    staging_root = staged_digest_table([None])

    with pytest.raises(ValueError, match="staged row for clip_id 'c0' has null caption_contract_digest"):
        caption_harness.staged_rows(staging_root)


def test_published_digests_rejects_a_null_digest(published_digest_table: DigestTableFactory) -> None:
    """Refuse an unfinished caption table instead of returning the string ``"None"``."""
    uri = published_digest_table([None])

    with pytest.raises(
        ValueError,
        match="published_digests requires every row to carry caption_contract_digest",
    ):
        caption_harness.published_digests(uri)


def test_staged_rows_preserves_a_valid_digest(staged_digest_table: DigestTableFactory) -> None:
    """Return a valid staged digest byte-for-byte unchanged."""
    digest = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
    staging_root = staged_digest_table([digest])

    assert [row.digest for row in caption_harness.staged_rows(staging_root)] == [digest]


def test_published_digests_preserves_a_valid_digest(published_digest_table: DigestTableFactory) -> None:
    """Return a valid published digest byte-for-byte unchanged."""
    digest = "fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210"
    uri = published_digest_table([digest])

    assert caption_harness.published_digests(uri) == {digest}
