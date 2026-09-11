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

"""Tests for deciding which spans of a session every sensor can serve."""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from cosmos_curator.next.recipes.multimodal_split import plan as plan_module
from cosmos_curator.next.recipes.multimodal_split.config import ResolvedMultimodalSplitConfig
from cosmos_curator.next.recipes.multimodal_split.identities import make_source_id
from cosmos_curator.next.recipes.multimodal_split.plan import (
    ALREADY_CANONICAL,
    PLANNING_RESULT_SCHEMA,
    first_uncovered,
    remaining_episode_ids_batch,
    span_planner,
)

_MS = 1_000_000
_EPOCH = 1_700_000_000 * 1_000_000_000


@pytest.mark.parametrize(
    ("offset_ns", "covered"),
    [
        (49 * _MS, True),
        # Exactly at the tolerance. The comparison is ``worst > max``, so equal
        # is covered -- and a flip to ``>=`` would pass every other test here,
        # which all sit well clear of the boundary.
        (50 * _MS, True),
        (50 * _MS + 1, False),
        (51 * _MS, False),
    ],
)
def test_the_tolerance_is_inclusive_at_its_boundary(offset_ns: int, *, covered: bool) -> None:
    """A frame exactly ``max_delta_s`` from a reference still serves that moment.

    Which way the boundary falls decides whether a rig at the edge of its
    tolerance yields clips or is refused, and nothing else in the suite pins it:
    every other case is a comfortable margin either side.
    """
    max_delta_ns = 50 * _MS
    references = np.array([_EPOCH, _EPOCH + 100 * _MS], dtype=np.int64)
    timeline = references + offset_ns

    missing = first_uncovered(references, {"front": timeline}, max_delta_ns)

    assert (missing is None) is covered
    if missing is not None:
        assert missing == ("front", offset_ns)


def _existing_row(entry: dict[str, object], source_media_location: str) -> dict[str, object]:
    return {
        "record_schema_version": entry["record_schema_version"],
        "media_contract_version": entry["media_contract_version"],
        "episode_id": entry["episode_id"],
        "source_id": entry["source_id"],
        "sampling_fps": entry["sampling_fps"],
        "timestamp_source": entry["timestamp_source"],
        "split_contract_digest": entry["split_contract_digest"],
        "camera_media": [{"camera_key": "front", "source_media_location": source_media_location}],
    }


def test_exact_canonical_episodes_skip_before_manifest_writes(
    tmp_path: Path,
    make_session: Callable[..., Path],
    make_config: Callable[..., ResolvedMultimodalSplitConfig],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The inert reconciliation map removes exact episodes before any artifact writer."""
    root = tmp_path / "recordings"
    session = make_session(root, "session-000")
    item = {"source_session_id": "session-000", "session_uri": str(session)}
    config = make_config(root, tmp_path / "first")
    first = span_planner(item, config=config)
    camera_plan = first[0]
    source_id = make_source_id(str(session))
    existing = {
        str(entry["episode_id"]): _existing_row(entry, str(camera_plan["sensor_uri"])) for entry in camera_plan["spans"]
    }
    monkeypatch.setattr(
        plan_module,
        "write_clip_manifest",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("manifest rewritten")),
    )

    reconciled = span_planner(
        item,
        config=make_config(root, tmp_path / "second"),
        reconciliation={source_id: existing},
    )

    assert len(reconciled) == len(existing)
    assert {row["result_kind"] for row in reconciled} == {ALREADY_CANONICAL}
    assert len({row["episode_id"] for row in reconciled}) == len(existing)


def test_remaining_episode_projection_deduplicates_multicamera_spans(
    tmp_path: Path,
    make_session: Callable[..., Path],
    make_config: Callable[..., ResolvedMultimodalSplitConfig],
) -> None:
    """One planned episode is counted once even though every camera carries its span."""
    root = tmp_path / "recordings"
    session = make_session(root, "session-000")
    camera_plan = span_planner(
        {"source_session_id": "session-000", "session_uri": str(session)},
        config=make_config(root, tmp_path / "media"),
    )[0]
    second_camera = {**camera_plan, "sensor_key": "left"}
    plans = pa.Table.from_pylist([camera_plan, second_camera], schema=PLANNING_RESULT_SCHEMA)

    episode_ids = remaining_episode_ids_batch(plans)["split_clip_id"].to_pylist()

    assert episode_ids == sorted(span["clip_id"] for span in camera_plan["spans"])


def test_changed_camera_source_conflicts_before_manifest_writes(
    tmp_path: Path,
    make_session: Callable[..., Path],
    make_config: Callable[..., ResolvedMultimodalSplitConfig],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A logical retry cannot replace bytes sourced from a different camera object."""
    root = tmp_path / "recordings"
    session = make_session(root, "session-000")
    item = {"source_session_id": "session-000", "session_uri": str(session)}
    config = make_config(root, tmp_path / "first")
    camera_plan = span_planner(item, config=config)[0]
    entry = camera_plan["spans"][0]
    source_id = make_source_id(str(session))
    existing = _existing_row(entry, "/different/front.mp4")
    monkeypatch.setattr(
        plan_module,
        "write_clip_manifest",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("manifest rewritten")),
    )

    with pytest.raises(ValueError, match="conflicts"):
        span_planner(item, config=config, reconciliation={source_id: {str(entry["episode_id"]): existing}})
