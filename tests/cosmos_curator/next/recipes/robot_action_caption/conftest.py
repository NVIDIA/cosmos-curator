# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixtures owning the temporary state a local caption attempt needs.

The builders live in ``caption_harness``; this module only binds them to a test's
``tmp_path`` and undoes the one monkeypatch each. Read the harness module's
docstring for what is faked and what is real.
"""

import itertools
import pathlib
from collections.abc import Sequence

import pytest

from cosmos_curator.core.utils import environment
from cosmos_curator.next.recipes.robot_action_caption import inference
from cosmos_curator.next.recipes.robot_action_caption.config import ResolvedRobotActionCaptionConfig
from tests.cosmos_curator.next.recipes.robot_action_caption import caption_harness


@pytest.fixture
def staged_weights(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> pathlib.Path:
    """Point the environment cache root at a minimal registry-validator fixture."""
    root = caption_harness.stage_model_weights(tmp_path / "model-weights")
    monkeypatch.setattr(environment, "CONTAINER_PATHS_MODEL_WEIGHT_CACHE_DIR", root)
    return root


@pytest.fixture
def caption_clips(tmp_path: pathlib.Path) -> caption_harness.ClipsTable:
    """Return a builder for a local clip table under this test's directory."""

    def build(
        *,
        rows: int = 8,
        rows_per_file: int = 4,
        selection: Sequence[str | None] | None = None,
        omit_selection: bool = False,
    ) -> str:
        return caption_harness.build_clips_table(
            tmp_path,
            rows=rows,
            rows_per_file=rows_per_file,
            selection=selection,
            omit_selection=omit_selection,
        )

    return build


@pytest.fixture
def caption_config(tmp_path: pathlib.Path) -> caption_harness.CaptionConfig:
    """Return a builder for a resolved config staging under this test's directory."""

    def build(
        uri: str,
        *,
        max_clip_bytes: int | None = None,
        media_attempts: int | None = None,
        geometry_rows: int | None = None,
        progress: bool | None = None,
        selected_only: bool | None = None,
        shard: tuple[int, int] | None = None,
    ) -> ResolvedRobotActionCaptionConfig:
        return caption_harness.build_caption_config(
            uri,
            staging_root=tmp_path / "staging",
            max_clip_bytes=max_clip_bytes,
            media_attempts=media_attempts,
            geometry_rows=geometry_rows,
            progress=progress,
            selected_only=selected_only,
            shard=shard,
        )

    return build


@pytest.fixture
def fake_engine(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> caption_harness.FakeEngine:
    """Return an installer for the fake engine stage, one ledger per installation.

    A test installs it more than once when it drives two attempts, so each
    installation gets its own ledger and the second attempt's invocations are
    never confused with the first's.
    """
    ledgers = itertools.count()

    def install(
        *,
        fails_on: tuple[str, ...] = (),
        caption_tag: str = "attempt",
        barrier_root: str | pathlib.Path | None = None,
    ) -> caption_harness.InferenceLedger:
        ledger = caption_harness.InferenceLedger(directory=tmp_path / f"engine-calls-{next(ledgers)}")
        ledger.directory.mkdir(parents=True)
        monkeypatch.setattr(
            inference,
            "build_vllm_transform",
            caption_harness.fake_transform_builder(
                ledger=ledger,
                fails_on=fails_on,
                caption_tag=caption_tag,
                barrier_root=pathlib.Path(barrier_root) if barrier_root is not None else None,
            ),
        )
        return ledger

    return install
