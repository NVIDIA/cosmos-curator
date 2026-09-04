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

"""Tests for ``multimodal-split`` clip identity."""

from collections.abc import Callable
from pathlib import Path

import pytest

from cosmos_curator.next.media.spans import Span
from cosmos_curator.next.recipes.multimodal_split.config import ResolvedMultimodalSplitConfig
from cosmos_curator.next.recipes.multimodal_split.identities import make_clip_id, make_source_id

from .conftest import split_all

_S = 1_000_000_000
_SOURCE_FPS = 30
# A camera's timeline starts wherever the recording did, not at zero.
_EPOCH = 1_700_000_000 * _S


def test_clip_ids_are_stable_across_runs(
    tmp_path: Path,
    make_session: Callable[..., Path],
    make_config: Callable[..., ResolvedMultimodalSplitConfig],
) -> None:
    """A rerun must address the same clip, or retries would duplicate rather than replace."""
    root = tmp_path / "recordings"
    make_session(root, "session-000")
    item = {"source_session_id": "session-000", "session_uri": str(root / "session-000")}

    first = split_all(item, config=make_config(root, tmp_path / "a"))
    second = split_all(item, config=make_config(root, tmp_path / "b"))

    assert [clip["clip_id"] for clip in first] == [clip["clip_id"] for clip in second]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("memory_gb", 4.0),
        ("read_buffer_bytes", 1 << 22),
        ("progress", True),
        ("planner_num_cpus", 0.5),
    ],
)
def test_execution_settings_leave_the_clip_id_alone(
    tmp_path: Path,
    make_config: Callable[..., ResolvedMultimodalSplitConfig],
    field: str,
    value: object,
) -> None:
    """A setting that changes no written byte must not change where the bytes land.

    The complement of the tests that pin what *does* re-key a clip: those would
    all still pass if the id hashed the whole config, which would scatter
    identical media across keys every time an operator retuned a run.
    """
    config = make_config(tmp_path / "recordings", tmp_path / "media")
    retuned = make_config(tmp_path / "recordings", tmp_path / "media", **{field: value})
    span = Span(start_ns=_EPOCH, end_ns=_EPOCH + _S)

    source_id = make_source_id("s3://bucket/recordings/session-000")

    assert make_clip_id(source_id, span, config) == make_clip_id(source_id, span, retuned)


def test_the_same_session_name_under_two_prefixes_is_two_sources(
    tmp_path: Path,
    make_config: Callable[..., ResolvedMultimodalSplitConfig],
) -> None:
    """A session ID is a directory name, so two datasets can each hold a ``session-000``.

    Named alone they key the same clips into the same place, and one run's media
    lands on the other's. Keying on where the session is keeps them apart, which
    is the contract ``robot_action_split`` already works under.
    """
    config = make_config(tmp_path / "recordings", tmp_path / "media")
    span = Span(start_ns=_EPOCH, end_ns=_EPOCH + _S)

    one = make_source_id("s3://bucket/recordings-a/session-000")
    other = make_source_id("s3://bucket/recordings-b/session-000")

    assert one != other
    assert make_clip_id(one, span, config) != make_clip_id(other, span, config)


def test_a_trailing_slash_does_not_make_a_second_source(tmp_path: Path) -> None:
    """Discovery joins IDs onto a prefix, and a prefix an operator typed may or may not end in one."""
    del tmp_path
    assert make_source_id("s3://bucket/recordings/session-000") == make_source_id("s3://bucket/recordings/session-000/")
