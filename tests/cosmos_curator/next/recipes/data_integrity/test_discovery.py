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

"""Unit tests for session stream discovery."""

import pathlib

import pytest

from cosmos_curator.next.recipes.data_integrity import discovery
from cosmos_curator.next.recipes.data_integrity.discovery import discover_streams


def _touch(path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def test_local_dir_finds_videos_recursively_and_ignores_side_files(tmp_path: pathlib.Path) -> None:
    """Discovery recurses recorder-XX subdirs for videos and skips rig JSON / markers."""
    _touch(tmp_path / "recorder-00" / "a.mp4")
    _touch(tmp_path / "recorder-01" / "b.mkv")
    _touch(tmp_path / "rig_meta.json")
    _touch(tmp_path / "_UPLOADED_TS")
    _touch(tmp_path / "notes.txt")

    found = discover_streams(str(tmp_path))

    assert found == sorted([str(tmp_path / "recorder-00" / "a.mp4"), str(tmp_path / "recorder-01" / "b.mkv")])


def test_local_dir_limit_caps_and_takes_sorted_prefix(tmp_path: pathlib.Path) -> None:
    """limit>0 returns the first N in sorted order."""
    for name in ("c.mp4", "a.mp4", "b.mp4"):
        _touch(tmp_path / name)

    found = discover_streams(str(tmp_path), limit=2)

    assert found == [str(tmp_path / "a.mp4"), str(tmp_path / "b.mp4")]


def test_local_stream_globs_keep_only_the_named_subprefixes(tmp_path: pathlib.Path) -> None:
    """A session holding copies of its cameras measures only the ones the globs name."""
    _touch(tmp_path / "recorder-00" / "front.mp4")
    _touch(tmp_path / "recorder-10" / "front.mp4")
    _touch(tmp_path / "__original" / "front.mp4")
    _touch(tmp_path / "triage" / "recorder-00" / "front.mp4")
    _touch(tmp_path / "derived" / "front.mp4")

    found = discover_streams(str(tmp_path), stream_globs=("recorder-*/*",))

    assert found == sorted([str(tmp_path / "recorder-00" / "front.mp4"), str(tmp_path / "recorder-10" / "front.mp4")])


def test_local_stream_globs_are_anchored_at_the_session(tmp_path: pathlib.Path) -> None:
    """The pattern matches the whole session-relative path, so a nested copy is excluded."""
    _touch(tmp_path / "triage" / "recorder-00" / "front.mp4")

    assert discover_streams(str(tmp_path), stream_globs=("recorder-*/*",)) == []


def test_local_stream_globs_accept_several_patterns(tmp_path: pathlib.Path) -> None:
    """Streams matching any one of the globs are kept, so mixed layouts stay expressible."""
    _touch(tmp_path / "recorder-00" / "front.mp4")
    _touch(tmp_path / "rear.mp4")
    _touch(tmp_path / "derived" / "front.mp4")

    found = discover_streams(str(tmp_path), stream_globs=("recorder-*/*", "*.mp4"))

    assert found == sorted([str(tmp_path / "recorder-00" / "front.mp4"), str(tmp_path / "rear.mp4")])


def test_local_stream_globs_can_span_subdirectories(tmp_path: pathlib.Path) -> None:
    """'**' spans depth, for a recorder that nests its cameras further down."""
    _touch(tmp_path / "recorder-00" / "cam" / "front.mp4")
    _touch(tmp_path / "derived" / "front.mp4")

    found = discover_streams(str(tmp_path), stream_globs=("recorder-*/**",))

    assert found == [str(tmp_path / "recorder-00" / "cam" / "front.mp4")]


def test_local_stream_globs_cap_after_filtering(tmp_path: pathlib.Path) -> None:
    """Limit counts kept streams, so a cap is not spent on excluded copies."""
    _touch(tmp_path / "__original" / "a.mp4")
    _touch(tmp_path / "__original" / "b.mp4")
    _touch(tmp_path / "recorder-00" / "c.mp4")

    found = discover_streams(str(tmp_path), limit=2, stream_globs=("recorder-*/*",))

    assert found == [str(tmp_path / "recorder-00" / "c.mp4")]


def test_relative_session_path_with_globs_still_matches(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A session named '.' walks to paths without that prefix; matching is on the relative path."""
    _touch(tmp_path / "recorder-00" / "front.mp4")
    monkeypatch.chdir(tmp_path)

    assert discover_streams(".", stream_globs=("recorder-*/*",)) == [str(pathlib.Path("recorder-00/front.mp4"))]


def test_single_file_ignores_stream_globs(tmp_path: pathlib.Path) -> None:
    """An explicitly named file is already a selection of one; the globs do not re-filter it."""
    video = tmp_path / "solo.mp4"
    _touch(video)
    assert discover_streams(str(video), stream_globs=("recorder-*/*",)) == [str(video)]


def test_single_file_returns_one_element(tmp_path: pathlib.Path) -> None:
    """A lone file path is returned as a single-element list."""
    video = tmp_path / "solo.mp4"
    _touch(video)
    assert discover_streams(str(video)) == [str(video)]


def test_single_non_video_file_is_filtered_out(tmp_path: pathlib.Path) -> None:
    """An explicit non-video file (e.g. rig_meta.json) is filtered, not returned as a bogus stream."""
    side = tmp_path / "rig_meta.json"
    _touch(side)
    assert discover_streams(str(side)) == []


def test_negative_limit_raises(tmp_path: pathlib.Path) -> None:
    """A negative limit is rejected at the discovery boundary, consistently across backends."""
    with pytest.raises(ValueError, match="limit"):
        discover_streams(str(tmp_path), limit=-1)


def test_missing_local_path_raises(tmp_path: pathlib.Path) -> None:
    """A nonexistent local path raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        discover_streams(str(tmp_path / "nope"))


@pytest.mark.parametrize("scheme", ["s3", "az"])
def test_cloud_prefix_delegates_to_listing_with_video_suffixes(monkeypatch: pytest.MonkeyPatch, scheme: str) -> None:
    """Both cloud schemes reach list_storage_objects with the video suffix filter and limit, sorted."""
    captured: dict[str, object] = {}
    prefix = f"{scheme}://b/clips/u/"

    def _fake_list(listed_prefix: str, **kwargs: object) -> list[str]:
        captured["prefix"] = listed_prefix
        captured.update(kwargs)
        return [f"{prefix}recorder-01/b.mp4", f"{prefix}recorder-00/a.mp4"]

    monkeypatch.setattr(discovery, "list_storage_objects", _fake_list)

    found = discover_streams(prefix, limit=5, s3_profile_name="maglev", endpoint_url="https://endpoint.io")

    assert found == [f"{prefix}recorder-00/a.mp4", f"{prefix}recorder-01/b.mp4"]
    assert captured["prefix"] == prefix
    assert captured["limit"] == 5
    assert captured["suffixes"] == discovery.VIDEO_SUFFIXES
    assert captured["endpoint_url"] == "https://endpoint.io"
    assert captured["s3_profile_name"] == "maglev"


@pytest.mark.parametrize("scheme", ["s3", "az"])
def test_cloud_stream_globs_filter_the_listing(monkeypatch: pytest.MonkeyPatch, scheme: str) -> None:
    """Cloud discovery keeps only the named subprefixes, and lists uncapped so the cap survives it."""
    captured: dict[str, object] = {}
    prefix = f"{scheme}://b/clips/u/"

    def _fake_list(listed_prefix: str, **kwargs: object) -> list[str]:
        captured["prefix"] = listed_prefix
        captured.update(kwargs)
        return [
            f"{prefix}__original/front.mp4",
            f"{prefix}recorder-00/front.mp4",
            f"{prefix}recorder-10/front.mp4",
            f"{prefix}triage/recorder-00/front.mp4",
        ]

    monkeypatch.setattr(discovery, "list_storage_objects", _fake_list)

    found = discover_streams(prefix, limit=1, stream_globs=("recorder-*/*",))

    # The cap is spent on a kept stream, not on the copies that sort ahead of it.
    assert found == [f"{prefix}recorder-00/front.mp4"]
    assert captured["limit"] == 0


def test_cloud_prefix_without_trailing_slash_matches(monkeypatch: pytest.MonkeyPatch) -> None:
    """A session prefix is written both ways by hand; both make the same relative paths."""
    monkeypatch.setattr(
        discovery,
        "list_storage_objects",
        lambda *_a, **_k: ["s3://b/clips/u/recorder-00/front.mp4", "s3://b/clips/u/derived/front.mp4"],
    )

    found = discover_streams("s3://b/clips/u", stream_globs=("recorder-*/*",))

    assert found == ["s3://b/clips/u/recorder-00/front.mp4"]


def test_stream_outside_the_session_is_dropped_with_a_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    """A listing that escapes the session cannot be judged against the globs, so it is not kept."""
    monkeypatch.setattr(
        discovery,
        "list_storage_objects",
        lambda *_a, **_k: ["s3://b/elsewhere/recorder-00/front.mp4"],
    )

    assert discover_streams("s3://b/clips/u/", stream_globs=("recorder-*/*",)) == []
