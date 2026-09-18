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

"""Tests for reading a ``multimodal-split`` session's artifacts."""

import io
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import av
import numpy as np
import pytest

from cosmos_curator.core.utils.storage.s3_client import S3Client, S3Prefix
from cosmos_curator.next.recipes.multimodal_split import storage as storage_module
from cosmos_curator.next.recipes.multimodal_split.uris import slashed

_S = 1_000_000_000
_SOURCE_FPS = 30
# A camera's timeline starts wherever the recording did, not at zero.
_EPOCH = 1_700_000_000 * _S


def test_a_remote_camera_is_opened_with_the_configured_storage_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reads must use the credentials the writes already use, or a run half-works.

    The recipe never inspects the scheme: ``get_storage_client`` returns ``None``
    for a local path and a client for a remote one, and that is the only thing
    that decides whether transport params are supplied.
    """
    opened: dict[str, Any] = {}

    class _FakeS3Client(S3Client):
        def __init__(self) -> None:
            """Skip the real constructor: this pins wiring, not credential loading."""
            self.s3 = "boto3-client-sentinel"

    def fake_get_storage_client(target_path: str, *, profile_name: str) -> object | None:
        opened["target_path"] = target_path
        opened["profile_name"] = profile_name
        return _FakeS3Client()

    def fake_open(uri: str, mode: str, *, transport_params: dict[str, Any]) -> io.BytesIO:
        opened["uri"] = uri
        opened["mode"] = mode
        opened["transport_params"] = transport_params
        return io.BytesIO(b"")

    monkeypatch.setattr(storage_module, "get_storage_client", fake_get_storage_client)
    monkeypatch.setattr(storage_module.smart_open, "open", fake_open)

    with storage_module.open_sensor_source("s3://bucket/session-000/camera/front.mp4", storage_profile="hyperion"):
        pass

    assert opened["profile_name"] == "hyperion"
    assert opened["uri"] == "s3://bucket/session-000/camera/front.mp4"
    assert opened["mode"] == "rb"
    assert opened["transport_params"] == {"client": "boto3-client-sentinel"}


class _FakeS3Client(S3Client):
    """A client that carries a sentinel instead of credentials.

    These tests pin which transport params reach smart_open, not how a session
    is authenticated.
    """

    def __init__(self) -> None:
        self.s3 = "boto3-client-sentinel"


def _remote_client(target_path: str, *, profile_name: str) -> object:
    """Stand in for a URI the storage layer recognizes as remote."""
    del target_path, profile_name
    return _FakeS3Client()


def _local_path(target_path: str, *, profile_name: str) -> object | None:
    """Stand in for a URI the storage layer recognizes as local."""
    del target_path, profile_name
    return None


def test_a_remote_session_is_listed_relative_to_itself(monkeypatch: pytest.MonkeyPatch) -> None:
    """One listing serves the whole rig, so it has to come back keyed the way a pattern is written.

    Patterns are session-relative -- ``camera/front.mp4`` -- while a listing
    answers in absolute keys. Returning those unchanged would match nothing and
    fail the rig as a missing sensor rather than as a listing bug.
    """
    listed: dict[str, object] = {}

    class _ListingClient(_FakeS3Client):
        def list_recursive_directory(self, uri: object, limit: int = 0) -> list[object]:
            listed["prefix"] = uri.path  # type: ignore[attr-defined]
            listed["limit"] = limit
            return [S3Prefix(f"{uri.path}{name}") for name in ("camera/front.mp4", "camera/front.mp4.timestamps.npy")]  # type: ignore[attr-defined]

    monkeypatch.setattr(storage_module, "get_storage_client", lambda *_a, **_k: _ListingClient())

    paths = storage_module.list_session_paths("s3://bucket/sessions/session-000", storage_profile="hyperion")

    assert paths == ["camera/front.mp4", "camera/front.mp4.timestamps.npy"]
    # Slashed, or listing session-000 would also answer for session-0001.
    assert listed["prefix"] == "s3://bucket/sessions/session-000/"


def test_a_remote_session_with_no_client_says_so_instead_of_walking_a_local_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An s3 URI with no usable client is a storage failure, not an empty session.

    ``get_storage_client`` answers ``None`` for a local path and for a remote one
    it could not build a client for, so the scheme is what tells those apart.
    Without the check the listing walks a filesystem path that was never local,
    finds nothing, and reports a rig missing every sensor -- naming the
    recording rather than the credentials.
    """
    monkeypatch.setattr(storage_module, "get_storage_client", lambda *_a, **_k: None)

    with pytest.raises(NotImplementedError, match="hyperion"):
        storage_module.list_session_paths("s3://bucket/sessions/session-000", storage_profile="hyperion")


def test_a_read_buffer_reaches_s3_and_never_a_local_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """``buffer_size`` is an S3 transport param, and the builtin open rejects it.

    A local path goes through the builtin ``open``, which buffers by block and
    takes no such argument, so passing it there is a TypeError at the first read
    rather than a slower one.
    """
    seen: list[dict[str, Any]] = []

    def fake_open(uri: str, mode: str, *, transport_params: dict[str, Any]) -> io.BytesIO:
        del uri, mode
        seen.append(transport_params)
        return io.BytesIO(b"")

    monkeypatch.setattr(storage_module.smart_open, "open", fake_open)

    monkeypatch.setattr(storage_module, "get_storage_client", _remote_client)
    with storage_module.open_sensor_source("s3://bucket/front.mp4", read_buffer_bytes=1 << 22):
        pass

    monkeypatch.setattr(storage_module, "get_storage_client", _local_path)
    with storage_module.open_sensor_source("/local/front.mp4", read_buffer_bytes=1 << 22):
        pass

    assert seen[0]["buffer_size"] == 1 << 22
    assert seen[1] == {}


def test_an_unset_read_buffer_leaves_smart_open_its_own_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset means unset: 128 KiB is smart_open's choice, not one this recipe restates."""
    seen: list[dict[str, Any]] = []

    def fake_open(uri: str, mode: str, *, transport_params: dict[str, Any]) -> io.BytesIO:
        del uri, mode
        seen.append(transport_params)
        return io.BytesIO(b"")

    monkeypatch.setattr(storage_module, "get_storage_client", _remote_client)
    monkeypatch.setattr(storage_module.smart_open, "open", fake_open)

    with storage_module.open_sensor_source("s3://bucket/front.mp4"):
        pass

    assert "buffer_size" not in seen[0]


@pytest.mark.parametrize(
    ("path", "pattern", "expected"),
    [
        ("recorder-00/camera_front.mp4", "*/camera_front.mp4", True),
        # The divergence this exists to stop: fnmatch's ``*`` spans separators,
        # so this matched on S3 and not on a local copy of the same session.
        ("recorder-00/sub/camera_front.mp4", "*/camera_front.mp4", False),
        ("camera_front.mp4", "*/camera_front.mp4", False),
        ("recorder-00/camera_rear.mp4", "*/camera_front.mp4", False),
        ("recorder-00/sub/camera_front.mp4", "*/*/camera_front.mp4", True),
        ("camera/front.mp4", "camera/front.mp4", True),
    ],
)
def test_a_wildcard_spans_one_path_segment_on_either_backend(path: str, pattern: str, *, expected: bool) -> None:
    """``*`` is a whole segment, and S3 and a local path must agree on that.

    The local branch globs, where ``*`` never crosses a separator. The S3 branch
    matches strings, where by default it does -- so the same config would find a
    nested camera remotely and report it missing locally.
    """
    assert storage_module._matches_by_segment(path, pattern) is expected

    with TemporaryDirectory() as directory:
        root = Path(directory)
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).touch()
        globbed = [str(found.relative_to(root)) for found in root.glob(pattern) if found.is_file()]

    assert (path in globbed) is expected


@pytest.mark.parametrize(
    ("location", "expected"),
    [
        ("s3://bucket/sessions/foo", "s3://bucket/sessions/foo/"),
        ("s3://bucket/sessions/foo/", "s3://bucket/sessions/foo/"),
        ("/data/sessions/foo", "/data/sessions/foo/"),
    ],
)
def test_a_prefix_is_slashed_so_a_listing_cannot_reach_a_sibling(location: str, expected: str) -> None:
    """Object storage matches a prefix as a raw string, so the slash is what bounds it.

    Without it, listing ``sessions/foo`` also finds ``sessions/foo-bar/...`` --
    a prefix holding nothing answers with a sibling's contents. Both the session
    listing and the preflight read probe depend on this, which is why there is
    one of it.
    """
    assert slashed(location) == expected
    assert not "sessions/foo-bar/cam.mp4".startswith(slashed("sessions/foo"))


def _write_synthetic_video(path: Path, *, frames: int = 10, fps: int = 30) -> None:
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libopenh264", rate=fps)
        stream.width, stream.height, stream.pix_fmt = 16, 16, "yuv420p"
        for index in range(frames):
            image = np.full((16, 16, 3), (index * 20) % 256, dtype=np.uint8)
            for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)


def test_load_container_pts_ns_indexes_the_video_when_no_sidecar_exists() -> None:
    """The fallback for a camera with no sensor timeline: read PTS from the container itself."""
    with TemporaryDirectory() as directory:
        path = Path(directory) / "front.mp4"
        _write_synthetic_video(path, frames=10, fps=30)

        pts_ns = storage_module.load_container_pts_ns(str(path))

    assert len(pts_ns) == 10
    assert pts_ns.dtype == np.int64
    assert np.all(np.diff(pts_ns) > 0)


def test_load_container_pts_ns_rejects_a_video_with_no_displayable_frames(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty container has no PTS to fall back to, and must not read as a valid one.

    ``make_index_and_metadata`` is faked directly: a real container with zero
    muxed frames is one PyAV never actually writes to disk, so there is no
    encoder path that produces this index on purpose.
    """

    class _FakeVideoIndex:
        display_pts_ns = np.array([], dtype=np.int64)

    def fake_make_index_and_metadata(*_args: object, **_kwargs: object) -> tuple[object, object]:
        return _FakeVideoIndex(), object()

    monkeypatch.setattr(storage_module, "make_index_and_metadata", fake_make_index_and_metadata)

    with TemporaryDirectory() as directory:
        path = Path(directory) / "front.mp4"
        _write_synthetic_video(path, frames=1, fps=30)

        with pytest.raises(ValueError, match="no displayable frames"):
            storage_module.load_container_pts_ns(str(path))
