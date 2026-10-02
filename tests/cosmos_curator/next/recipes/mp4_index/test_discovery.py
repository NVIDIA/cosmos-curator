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
"""Tests for shared MP4 source discovery.

The local-path tests exercise the real listing logic directly -- glob/exclude
matching is what's under test there, not which storage backend served the
listing. The S3 fast path is tested separately with a fake ``S3Client``.
"""

import pathlib

import pytest

from cosmos_curator.core.utils.storage.s3_client import S3Client, S3Prefix
from cosmos_curator.core.utils.storage.storage_client import StorageClient
from cosmos_curator.core.utils.storage.storage_utils import get_files_relative
from cosmos_curator.next.recipes.mp4_index import discovery
from cosmos_curator.next.recipes.mp4_index.config import Mp4IndexInputConfig
from cosmos_curator.next.recipes.mp4_index.contracts import CONTAINER_KIND_NONFRAGMENTED


def _touch(root: pathlib.Path, relative: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def test_discovers_a_unit_shaped_layout(tmp_path: pathlib.Path) -> None:
    """<session_id>/unit-NN/*.mp4 at arbitrary depth -- one glob, no session-level special-casing."""
    _touch(tmp_path, "sess-a/unit-00/camera.mp4")
    _touch(tmp_path, "sess-a/unit-01/camera.mp4")
    _touch(tmp_path, "sess-a/unit-00/camera.mp4.timestamps")  # not a match
    _touch(tmp_path, "sess-b/unit-00/camera.mp4")

    sources = discovery.discover_videos(path_prefix=str(tmp_path), globs=["*/unit-*/*.mp4"])

    assert [s.media_location for s in sources] == sorted(
        [
            f"{tmp_path}/sess-a/unit-00/camera.mp4",
            f"{tmp_path}/sess-a/unit-01/camera.mp4",
            f"{tmp_path}/sess-b/unit-00/camera.mp4",
        ]
    )


def test_exclude_globs_drops_a_globs_hit(tmp_path: pathlib.Path) -> None:
    """A broad include plus a narrow exclude is a direct statement, not a clever single pattern."""
    _touch(tmp_path, "sess-a/unit-00/camera.mp4")
    _touch(tmp_path, "sess-a/__original/camera.mp4")

    sources = discovery.discover_videos(
        path_prefix=str(tmp_path), globs=["*.mp4"], exclude_globs=["*/__original/*.mp4"]
    )

    assert [s.media_location for s in sources] == [f"{tmp_path}/sess-a/unit-00/camera.mp4"]


def test_multiple_globs_match_any(tmp_path: pathlib.Path) -> None:
    """A video matches if any pattern in the list matches -- real layouts commonly need more than one shape."""
    _touch(tmp_path, "sess-a/unit-00/camera.mp4")
    _touch(tmp_path, "sess-a/flat_camera.mp4")

    sources = discovery.discover_videos(path_prefix=str(tmp_path), globs=["*/unit-*/*.mp4", "*/flat_camera.mp4"])

    assert sorted(s.media_location for s in sources) == sorted(
        [f"{tmp_path}/sess-a/unit-00/camera.mp4", f"{tmp_path}/sess-a/flat_camera.mp4"]
    )


def test_media_location_is_the_full_path(tmp_path: pathlib.Path) -> None:
    """A caller reads the file straight from media_location -- it must be a real, fetchable path."""
    _touch(tmp_path, "sess-a/unit-00/camera.mp4")

    sources = discovery.discover_videos(path_prefix=str(tmp_path), globs=["*/unit-*/*.mp4"])

    assert len(sources) == 1
    assert pathlib.Path(sources[0].media_location).exists()


def test_limit_caps_after_matching_not_before(tmp_path: pathlib.Path) -> None:
    """A limit that capped the listing could exclude every real match behind unrelated noise."""
    _touch(tmp_path, "sess-a/unit-00/zzz_not_a_camera.txt")
    _touch(tmp_path, "sess-a/unit-00/camera_a.mp4")
    _touch(tmp_path, "sess-b/unit-00/camera_b.mp4")

    sources = discovery.discover_videos(path_prefix=str(tmp_path), globs=["*/unit-*/*.mp4"], limit=1)

    assert len(sources) == 1
    assert sources[0].media_location == f"{tmp_path}/sess-a/unit-00/camera_a.mp4"


def test_no_matches_returns_empty_not_an_error(tmp_path: pathlib.Path) -> None:
    """An empty result is a legitimate outcome for a glob that matches nothing -- not a failure."""
    _touch(tmp_path, "sess-a/unit-00/radar.mcap")

    sources = discovery.discover_videos(path_prefix=str(tmp_path), globs=["*/unit-*/*.mp4"])

    assert sources == []


class _FakeS3Client(S3Client):
    """A duck-typed stand-in for ``S3Client`` -- only the two methods this module calls."""

    def __init__(self, keys: list[str]) -> None:  # no real boto3 session wanted, so super().__init__() is skipped
        self._keys = keys

    def list_child_prefixes(self, uri: S3Prefix) -> list[str]:
        prefix = str(uri).rstrip("/") + "/"
        matching = [key for key in self._keys if key.startswith(prefix)]
        if not matching:
            msg = f"Prefix does not exist or contains no objects: {prefix}"
            raise FileNotFoundError(msg)
        return sorted({key[len(prefix) :].split("/", 1)[0] for key in matching if "/" in key[len(prefix) :]})

    def list_recursive_directory(self, uri: S3Prefix, limit: int = 0) -> list[str]:  # type: ignore[override]
        del limit  # unused -- this fake never truncates
        prefix = str(uri)
        return [key for key in self._keys if key.startswith(prefix) and not key.endswith("/")]


def test_discover_videos_s3_walks_sessions_via_the_two_phase_listing(monkeypatch: pytest.MonkeyPatch) -> None:
    """The two-phase (top-level, then per-session) walk finds the same videos a flat listing would.

    Goes through the public discover_videos, not the private
    _list_s3_relative_paths_concurrently directly -- the two-phase walk is an
    internal optimization, not part of the contract a caller (or a harmless
    refactor renaming/inlining it) should be coupled to.
    """
    client = _FakeS3Client(
        [
            "s3://bucket/samples/sess-a/unit-00/camera.mp4",
            "s3://bucket/samples/sess-a/unit-01/camera.mp4",
            "s3://bucket/samples/sess-b/unit-00/camera.mp4",
        ]
    )
    monkeypatch.setattr(discovery, "get_storage_client", lambda *_a, **_k: client)

    sources = discovery.discover_videos(path_prefix="s3://bucket/samples", globs=["*.mp4"])

    assert sorted(s.media_location for s in sources) == sorted(
        [
            "s3://bucket/samples/sess-a/unit-00/camera.mp4",
            "s3://bucket/samples/sess-a/unit-01/camera.mp4",
            "s3://bucket/samples/sess-b/unit-00/camera.mp4",
        ]
    )


def test_discover_videos_s3_falls_back_for_a_flat_layout(monkeypatch: pytest.MonkeyPatch) -> None:
    """No subdirectories at all (every key sits directly under the prefix) still finds everything."""
    client = _FakeS3Client(["s3://bucket/samples/camera_a.mp4", "s3://bucket/samples/camera_b.mp4"])
    monkeypatch.setattr(discovery, "get_storage_client", lambda *_a, **_k: client)

    sources = discovery.discover_videos(path_prefix="s3://bucket/samples", globs=["*.mp4"])

    assert sorted(s.media_location for s in sources) == sorted(
        ["s3://bucket/samples/camera_a.mp4", "s3://bucket/samples/camera_b.mp4"]
    )


def test_discover_videos_s3_empty_prefix_returns_empty_not_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A prefix with nothing under it is a legitimate empty result, not an error."""
    client = _FakeS3Client([])
    monkeypatch.setattr(discovery, "get_storage_client", lambda *_a, **_k: client)

    sources = discovery.discover_videos(path_prefix="s3://bucket/samples", globs=["*.mp4"])

    assert sources == []


def test_discover_videos_s3_takes_the_fast_path_not_the_generic_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """An S3Client (even this fake, real subclass) takes the concurrent per-session fast path.

    Proven, not merely matched by coincidence: the fast path calls
    get_files_relative once *per session* (a session-scoped path); the
    generic fallback calls it exactly once, with the whole top-level prefix.
    Recording every call's path distinguishes the two -- if the isinstance
    dispatch silently routed to the generic fallback instead, this would see
    one call at "s3://bucket/samples" rather than one per session.
    """
    client = _FakeS3Client(
        [
            "s3://bucket/samples/sess-a/unit-00/camera.mp4",
            "s3://bucket/samples/sess-b/unit-00/camera.mp4",
        ]
    )
    monkeypatch.setattr(discovery, "get_storage_client", lambda *_a, **_k: client)

    calls: list[str] = []

    def _recording_get_files_relative(path: str, *, client: StorageClient | None = None) -> list[str]:
        calls.append(path)
        return get_files_relative(path, client=client)

    monkeypatch.setattr(discovery, "get_files_relative", _recording_get_files_relative)

    sources = discovery.discover_videos(path_prefix="s3://bucket/samples", globs=["*.mp4"])

    assert sorted(calls) == ["s3://bucket/samples/sess-a", "s3://bucket/samples/sess-b"]
    assert sorted(s.media_location for s in sources) == sorted(
        ["s3://bucket/samples/sess-a/unit-00/camera.mp4", "s3://bucket/samples/sess-b/unit-00/camera.mp4"]
    )


def test_discover_videos_uses_the_generic_fallback_for_a_non_s3_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """Any other StorageClient goes through get_files_relative, no S3-specific code.

    Goes through the public discover_videos, not the private
    _list_remote_relative_paths directly -- same reasoning as this file's own
    S3-path tests: the isinstance dispatch is an internal optimization, not
    part of the contract a caller should be coupled to.
    """

    class _FakeGenericClient:
        """Not an S3Client -- exercises the isinstance dispatch's other branch."""

        @staticmethod
        def list_recursive_directory(uri: object, limit: int = 0) -> list[str]:  # noqa: ARG004
            return ["sess-a/unit-00/camera.mp4"]  # relative to path_prefix, like get_files_relative's own contract

    monkeypatch.setattr(discovery, "get_storage_client", lambda *_a, **_k: _FakeGenericClient())
    monkeypatch.setattr(discovery, "get_files_relative", lambda path, client: client.list_recursive_directory(path))

    sources = discovery.discover_videos(path_prefix="az://container/samples", globs=["*.mp4"])

    assert [s.media_location for s in sources] == ["az://container/samples/sess-a/unit-00/camera.mp4"]


def test_resolve_input_selection_uris_mode_sorts_and_dedupes() -> None:
    """Uris mode does no listing, but deduplicates and sorts for a deterministic run order."""
    input_config = Mp4IndexInputConfig(
        uris=("s3://bucket/b.mp4", "s3://bucket/a.mp4", "s3://bucket/a.mp4"),
        container_kind=CONTAINER_KIND_NONFRAGMENTED,
    )
    sources = discovery.resolve_input_selection(input_config)

    assert [s.media_location for s in sources] == ["s3://bucket/a.mp4", "s3://bucket/b.mp4"]


def test_resolve_input_selection_path_prefix_mode_dispatches_to_discover_videos(tmp_path: pathlib.Path) -> None:
    """path_prefix mode goes through discover_videos, globs and all."""
    _touch(tmp_path, "sess-a/unit-00/camera.mp4")

    input_config = Mp4IndexInputConfig(
        path_prefix=str(tmp_path), globs=("*/unit-*/*.mp4",), container_kind=CONTAINER_KIND_NONFRAGMENTED
    )
    sources = discovery.resolve_input_selection(input_config)

    assert [s.media_location for s in sources] == [f"{tmp_path}/sess-a/unit-00/camera.mp4"]
