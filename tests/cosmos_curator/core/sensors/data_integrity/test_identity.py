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

"""Unit tests for stream / session identity derivation."""

import pathlib

import pytest

from cosmos_curator.core.sensors.data_integrity import identity


def test_stream_id_is_stable_across_calls() -> None:
    """The id is a persisted key, so it must not depend on interpreter run or call order."""
    first = identity.stream_id("s3://bucket/clips/a/front.mp4")
    second = identity.stream_id("s3://bucket/clips/a/front.mp4")
    assert first == second
    assert len(first) == 32


@pytest.mark.parametrize(
    ("left", "right"),
    [
        # Scheme case: URI schemes are case-insensitive, so these address one object.
        ("S3://bucket/clips/a.mp4", "s3://bucket/clips/a.mp4"),
        # A trailing slash is how a prefix gets written by hand, not part of the name.
        ("s3://bucket/clips/", "s3://bucket/clips"),
    ],
)
def test_equivalent_spellings_share_one_id(left: str, right: str) -> None:
    """Two spellings of the same object must not become two rows in the store."""
    assert identity.stream_id(left) == identity.stream_id(right)


@pytest.mark.parametrize(
    "spelling",
    [
        "s3://bucket//clips/a.mp4",
        "s3://bucket/clips/./a.mp4",
        "s3://bucket/clips/b/../a.mp4",
    ],
)
def test_cloud_object_keys_are_opaque(spelling: str) -> None:
    """An object key is a string, not a path: folding it would merge distinct objects."""
    assert identity.stream_id(spelling) != identity.stream_id("s3://bucket/clips/a.mp4")


def test_local_dot_segments_are_folded() -> None:
    """A local path *is* a path, so the two spellings really are one file."""
    assert identity.stream_id("/data/clips/./a.mp4") == identity.stream_id("/data/clips/b/../a.mp4")


def test_key_case_is_never_folded() -> None:
    """S3 keys and POSIX paths are case-sensitive; folding them would merge distinct objects."""
    assert identity.stream_id("s3://bucket/A.mp4") != identity.stream_id("s3://bucket/a.mp4")


def test_relative_local_path_resolves_against_the_current_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """A relative path and its absolute form are the same stream, discovered two ways."""
    monkeypatch.chdir(tmp_path)
    # Built from the working directory rather than tmp_path so the comparison holds on
    # platforms where the temp root is reached through a symlink; normalisation
    # deliberately leaves symlinks alone, so an id depends on the path as written.
    absolute = pathlib.Path.cwd() / "clips" / "a.mp4"
    assert identity.stream_id("clips/a.mp4") == identity.stream_id(str(absolute))


def test_selector_distinguishes_streams_inside_one_file() -> None:
    """One MCAP holds many topics, so the file alone cannot be the key."""
    front = identity.stream_id(
        "s3://bucket/drive.mcap",
        selector_type=identity.SELECTOR_MCAP_TOPIC,
        selector_value="/camera/front",
    )
    rear = identity.stream_id(
        "s3://bucket/drive.mcap",
        selector_type=identity.SELECTOR_MCAP_TOPIC,
        selector_value="/camera/rear",
    )
    assert front != rear


def test_selector_type_is_part_of_the_key() -> None:
    """The same selector value under two addressing schemes is not the same stream."""
    as_index = identity.stream_id("s3://b/x", selector_type=identity.SELECTOR_VIDEO_STREAM, selector_value="0")
    as_topic = identity.stream_id("s3://b/x", selector_type=identity.SELECTOR_MCAP_TOPIC, selector_value="0")
    assert as_index != as_topic


def test_id_does_not_depend_on_how_the_stream_was_discovered() -> None:
    """Checking a video directly and checking its session must produce one identity.

    This is why the id hashes the absolute source rather than a session root plus a
    relative key: otherwise the two would write two rows for the same bytes.
    """
    direct = identity.stream_id("s3://bucket/clips/uuid/front.mp4")
    within_session = identity.stream_id("s3://bucket/clips/uuid/front.mp4")
    assert direct == within_session
    # ...and the session context that differs between the two runs is descriptive only.
    assert identity.relative_key("s3://bucket/clips/uuid", "s3://bucket/clips/uuid/front.mp4") == "front.mp4"
    assert identity.relative_key(None, "s3://bucket/clips/uuid/front.mp4") is None


def test_locator_namespace_reports_the_backend() -> None:
    """Recorded so a future session table can group by backend without re-parsing URIs."""
    assert identity.locator_namespace("s3://bucket/a.mp4") == identity.NAMESPACE_S3
    assert identity.locator_namespace("az://container/a.mp4") == identity.NAMESPACE_AZURE
    assert identity.locator_namespace("/data/a.mp4") == identity.NAMESPACE_LOCAL


def test_session_id_is_namespaced_by_backend() -> None:
    """The same prefix under two backends is two sessions, not one."""
    assert identity.session_id("s3://bucket/clips/a") != identity.session_id("az://bucket/clips/a")
    assert identity.session_id(None) is None


def test_session_id_ignores_a_trailing_slash() -> None:
    """A prefix with and without its trailing slash is the same session."""
    assert identity.session_id("s3://bucket/clips/a/") == identity.session_id("s3://bucket/clips/a")


def test_relative_key_handles_nesting() -> None:
    """Sessions nest one level deeper than the flat case (a recorder subdirectory)."""
    key = identity.relative_key("s3://bucket/clips/uuid", "s3://bucket/clips/uuid/recorder_0/front.mp4")
    assert key == "recorder_0/front.mp4"


def test_relative_key_is_none_when_the_source_escapes_the_session() -> None:
    """An escaping '../' result would describe a relationship that does not hold."""
    assert identity.relative_key("s3://bucket/clips/uuid", "s3://bucket/other/front.mp4") is None
