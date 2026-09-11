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

"""Tests for ``multimodal-split`` clip and episode identities."""

from collections.abc import Callable
from pathlib import Path

from cosmos_curator.next.media.spans import Span
from cosmos_curator.next.recipes.multimodal_split.config import ResolvedMultimodalSplitConfig
from cosmos_curator.next.recipes.multimodal_split.identities import (
    make_clip_id,
    make_episode_id,
    make_materialization_digest,
    make_source_id,
    make_split_contract_digest,
)
from cosmos_curator.next.recipes.multimodal_split.plan import span_planner

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


def test_no_setting_changes_the_clip_id(
    tmp_path: Path,
    make_session: Callable[..., Path],
    make_config: Callable[..., ResolvedMultimodalSplitConfig],
) -> None:
    """A clip id names a moment, so nothing about how the moment was rendered reaches it.

    Exercised through ``span_planner``, which is the only caller that holds a
    config at all. Calling ``make_clip_id`` directly under two configs would
    prove nothing now that it takes none -- the assertion has to run through the
    code that could still pass one.

    Settings used to be folded in so a rerun under new settings could not
    overwrite the old media. That made the identity double as overwrite
    protection, which is ``output.overwrite``'s job, and meant changing a bitrate
    re-keyed a whole dataset.
    """
    root = tmp_path / "recordings"
    make_session(root, "session-000")
    item = {"source_session_id": "session-000", "session_uri": str(root / "session-000")}

    def clip_ids(**overrides: object) -> list[str]:
        config = make_config(root, tmp_path / "media", **overrides)
        return [str(entry["clip_id"]) for row in span_planner(item, config=config) for entry in row["spans"]]

    baseline = clip_ids()
    assert baseline, "the fixture session has to plan at least one clip for this to assert anything"
    for field, value in (
        ("video_bitrate", "8M"),
        ("gop_size", 30),
        ("max_delta_s", 0.02),
        ("memory_gb", 8.0),
        ("read_buffer_bytes", 1 << 20),
    ):
        assert clip_ids(**{field: value}) == baseline, f"{field} reached the clip id"


def test_a_different_span_is_a_different_clip() -> None:
    """The moment is the identity, so two spans of one session cannot collide."""
    source_id = make_source_id("s3://bucket/recordings/session-000")
    first = Span(start_ns=_EPOCH, end_ns=_EPOCH + _S)
    second = Span(start_ns=_EPOCH + _S, end_ns=_EPOCH + 2 * _S)

    assert make_clip_id(source_id, first) != make_clip_id(source_id, second)


def test_the_same_session_name_under_two_prefixes_is_two_sources() -> None:
    """A session ID is a directory name, so two datasets can each hold a ``session-000``.

    Named alone they key the same clips into the same place, and one run's media
    lands on the other's. Keying on where the session is keeps them apart, which
    is the contract ``robot_action_split`` already works under.
    """
    span = Span(start_ns=_EPOCH, end_ns=_EPOCH + _S)

    one = make_source_id("s3://bucket/recordings-a/session-000")
    other = make_source_id("s3://bucket/recordings-b/session-000")

    assert one != other
    assert make_clip_id(one, span) != make_clip_id(other, span)


def test_a_trailing_slash_does_not_make_a_second_source(tmp_path: Path) -> None:
    """Discovery joins IDs onto a prefix, and a prefix an operator typed may or may not end in one."""
    del tmp_path
    assert make_source_id("s3://bucket/recordings/session-000") == make_source_id("s3://bucket/recordings/session-000/")


def test_episode_and_materialization_digest_fixtures(
    tmp_path: Path,
    make_config: Callable[..., ResolvedMultimodalSplitConfig],
) -> None:
    """Kind tags, versions, and normalized payloads remain pinned by stable hashes."""
    config = make_config(
        tmp_path / "input",
        tmp_path / "output",
        cameras={"left": "left.mp4", "front": "front.mp4"},
        video_bitrate="4M",
        gop_size=30,
    )
    span = Span(start_ns=_S, end_ns=2 * _S)
    source_id = make_source_id("/input/session")
    episode_id = make_episode_id(source_id, span)
    split_clip_id = make_clip_id(source_id, span)
    split_digest = make_split_contract_digest(config)
    camera_media = [
        {
            "camera_key": "front",
            "media_sha256": "1" * 64,
            "timeline_sha256": "2" * 64,
            "observations_sha256": "3" * 64,
        },
        {
            "camera_key": "left",
            "media_sha256": "4" * 64,
            "timeline_sha256": "5" * 64,
            "observations_sha256": "6" * 64,
        },
    ]

    assert source_id == "e0f6ba3e1fd54cf7adc3e4491d1caca4e5337e9f86cbc2c5f0bf0175abd081d5"
    assert episode_id == "7bc8dd600c888cd42eb331f95b841438d9cc2db59e5468609956e61543d3c490"
    assert split_clip_id == "06c13fc958dbf9468c99710c63777749752f320583620710c8c68e1299a2d855"
    # The output and input roots are excluded, but the fixture config retains
    # its actual temporary path only in those excluded fields.
    assert split_digest == "6fbb071ddbeffc8e94f52a8d70553737584a3fcb12e56f4f779f4fa81c8aef74"
    assert (
        make_materialization_digest(
            episode_id=episode_id,
            split_clip_id=split_clip_id,
            split_contract_digest=split_digest,
            sampling_fps=30,
            row_count=30,
            camera_media=camera_media,
        )
        == "a7aa78bcaebdb2bcc0d594c5882b3d3ad3065e6b31fe6dcf3f28c9443ec92ad3"
    )


def test_split_contract_includes_rendering_but_excludes_execution(
    tmp_path: Path,
    make_config: Callable[..., ResolvedMultimodalSplitConfig],
) -> None:
    """One defining and one execution-only change pin the digest projection."""
    baseline = make_config(tmp_path / "input", tmp_path / "output")
    defining = make_config(tmp_path / "input", tmp_path / "output", video_bitrate="8M")
    execution = make_config(tmp_path / "input", tmp_path / "elsewhere", progress=True, memory_gb=8.0)

    assert make_split_contract_digest(defining) != make_split_contract_digest(baseline)
    assert make_split_contract_digest(execution) == make_split_contract_digest(baseline)
