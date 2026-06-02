# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Tests for cosmos_curator.core.utils.storage.presigned_s3_zip."""

import argparse
import contextlib
import io
import math
import os
import socketserver
import threading
import zipfile
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import ClassVar

import pytest

from cosmos_curator.core.utils.storage.presigned_s3_zip import (
    _create_zip_archive,
    _download_and_extract_zip_single_node,
    _validate_archive_size,
    _validate_upload_completion,
    _write_split_metadata,
    zip_and_upload_directory,
    zip_and_upload_directory_multipart,
)


class _ThreadedHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


@contextlib.contextmanager
def _serve(handler_cls: type[BaseHTTPRequestHandler]) -> Iterator[HTTPServer]:
    """Spin up a simple threaded HTTP server for the duration of the context."""
    server: HTTPServer = _ThreadedHTTPServer(("127.0.0.1", 0), handler_cls)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        thread.join()


def test_create_zip_archive_success(tmp_path: Path) -> None:
    """Create an archive from nested files and validate contents."""
    src_dir = tmp_path / "data"
    src_dir.mkdir()
    (src_dir / "top.txt").write_text("root file", encoding="utf-8")
    nested = src_dir / "nested"
    nested.mkdir()
    (nested / "inner.bin").write_bytes(b"\x00\x01\x02")

    archive_path = _create_zip_archive(str(src_dir))

    assert archive_path.exists()
    with zipfile.ZipFile(archive_path) as zf:
        names = set(zf.namelist())
        assert {"top.txt", "nested/", "nested/inner.bin"} == names
        assert zf.read("top.txt") == b"root file"
        assert zf.read("nested/inner.bin") == b"\x00\x01\x02"

    archive_path.unlink(missing_ok=True)


def test_create_zip_archive_invalid_directory(tmp_path: Path) -> None:
    """Raise when attempting to zip a directory that does not exist."""
    missing_dir = tmp_path / "missing"
    with pytest.raises(ValueError, match="does not exist"):
        _create_zip_archive(str(missing_dir))


def test_download_and_extract_zip_single_node_returns_inner_directory(tmp_path: Path) -> None:
    """Return the single top-level directory when extracting."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("single_dir/file.txt", "payload")
    zip_bytes = buf.getvalue()

    class DownloadHandler(BaseHTTPRequestHandler):
        payload: ClassVar[bytes] = zip_bytes

        def do_GET(self) -> None:
            self.send_response(200)
            self.send_header("Content-Length", str(len(self.payload)))
            self.end_headers()
            self.wfile.write(self.payload)

        def log_message(self, _format: str, *_args: object) -> None:
            """Silence handler logging."""

    with _serve(DownloadHandler) as server:
        url = f"http://127.0.0.1:{server.server_address[1]}/archive.zip"
        extracted = _download_and_extract_zip_single_node(url, tmp_dir=str(tmp_path))

    extracted_path = Path(extracted)
    assert extracted_path.name == "single_dir"
    assert (extracted_path / "file.txt").read_text(encoding="utf-8") == "payload"


def test_download_and_extract_zip_single_node_multiple_top_level(tmp_path: Path) -> None:
    """Return extraction dir when multiple top-level entries exist."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("root_a.txt", "a")
        zf.writestr("root_b.txt", "b")
    zip_bytes = buf.getvalue()

    class DownloadHandler(BaseHTTPRequestHandler):
        payload: ClassVar[bytes] = zip_bytes

        def do_GET(self) -> None:
            self.send_response(200)
            self.send_header("Content-Length", str(len(self.payload)))
            self.end_headers()
            self.wfile.write(self.payload)

        def log_message(self, _format: str, *_args: object) -> None:
            """Silence handler logging."""

    with _serve(DownloadHandler) as server:
        url = f"http://127.0.0.1:{server.server_address[1]}/archive.zip"
        extracted = _download_and_extract_zip_single_node(url, tmp_dir=str(tmp_path))

    extracted_path = Path(extracted)
    assert extracted_path.name == "extracted"
    assert sorted(p.name for p in extracted_path.iterdir()) == ["root_a.txt", "root_b.txt"]


def test_zip_and_upload_directory_round_trip(tmp_path: Path) -> None:
    """Zip a directory, upload it, and validate round-trip."""
    src_dir = tmp_path / "upload"
    src_dir.mkdir()
    file_a = src_dir / "alpha.txt"
    file_b = src_dir / "beta" / "nested.bin"
    file_b.parent.mkdir()
    file_a.write_text("alpha", encoding="utf-8")
    expected_bytes = os.urandom(64)
    file_b.write_bytes(expected_bytes)

    class UploadHandler(BaseHTTPRequestHandler):
        received: ClassVar[bytes | None] = None

        def do_PUT(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length)
            type(self).received = body
            self.send_response(200)
            self.end_headers()

        def log_message(self, _format: str, *_args: object) -> None:
            """Silence handler logging."""

    with _serve(UploadHandler) as server:
        url = f"http://127.0.0.1:{server.server_address[1]}/upload"
        zip_and_upload_directory(str(src_dir), url)

    assert UploadHandler.received is not None
    with zipfile.ZipFile(io.BytesIO(UploadHandler.received)) as zf:
        assert set(zf.namelist()) == {"beta/", "beta/nested.bin", "alpha.txt"}
        assert zf.read("alpha.txt") == b"alpha"
        assert zf.read("beta/nested.bin") == expected_bytes


def test_write_split_metadata_skips_all_captions_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Presigned split uploads should not rebuild aggregate captions unless opted in."""
    called = False

    def fake_write_all_window_captions(**_kwargs: object) -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(
        "cosmos_curator.core.utils.storage.presigned_s3_zip._write_all_window_captions",
        fake_write_all_window_captions,
    )

    _write_split_metadata(
        argparse.Namespace(input_video_path="/input", write_all_caption_json=False),
        "/output",
    )

    assert called is False


def test_write_split_metadata_rebuilds_all_captions_when_opted_in(monkeypatch: pytest.MonkeyPatch) -> None:
    """The positive all-captions option should reach the presigned metadata rewrite path."""
    called = False

    monkeypatch.setattr(
        "cosmos_curator.core.utils.storage.presigned_s3_zip.get_storage_client",
        lambda *args, **_kwargs: f"client:{args[0]}",
    )
    monkeypatch.setattr(
        "cosmos_curator.core.utils.storage.presigned_s3_zip.get_files_relative",
        lambda *_args, **_kwargs: ["video.mp4"],
    )

    def fake_write_all_window_captions(**_kwargs: object) -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(
        "cosmos_curator.core.utils.storage.presigned_s3_zip._write_all_window_captions",
        fake_write_all_window_captions,
    )

    _write_split_metadata(
        argparse.Namespace(input_video_path="/input", write_all_caption_json=True),
        "/output",
    )

    assert called is True


def test_zip_and_upload_directory_multipart(tmp_path: Path) -> None:
    """Split large uploads across presigned part URLs and reassemble."""
    src_dir = tmp_path / "multipart"
    src_dir.mkdir()
    for idx in range(3):
        (src_dir / f"file_{idx}.bin").write_bytes(os.urandom(2048))

    archive_for_size = _create_zip_archive(str(src_dir))
    archive_bytes = archive_for_size.read_bytes()
    archive_for_size.unlink(missing_ok=True)

    received_parts: list[bytes] = []

    class MultipartHandler(BaseHTTPRequestHandler):
        def do_PUT(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length)
            received_parts.append(body)
            self.send_response(200)
            self.end_headers()

        def log_message(self, _format: str, *_args: object) -> None:
            """Silence handler logging."""

    part_count = 3
    chunk_size = max(1, math.ceil(len(archive_bytes) / part_count))
    assert len(archive_bytes) > chunk_size

    with _serve(MultipartHandler) as server:
        base_url = f"http://127.0.0.1:{server.server_address[1]}"
        part_urls = [f"{base_url}/part/{idx}" for idx in range(part_count)]
        multipart_config = {"uploadId": "upload", "key": "output.zip", "parts": part_urls}
        zip_and_upload_directory_multipart(str(src_dir), multipart_config, chunk_size_bytes=chunk_size)

    assert len(received_parts) >= 2  # ensure multipart behaviour occurred
    combined = b"".join(received_parts)
    with zipfile.ZipFile(io.BytesIO(combined)) as zf:
        assert set(zf.namelist()) == {
            "file_0.bin",
            "file_1.bin",
            "file_2.bin",
        }
        for idx in range(3):
            assert zf.read(f"file_{idx}.bin") == (src_dir / f"file_{idx}.bin").read_bytes()


def test_validate_archive_size_raises_when_exceeding() -> None:
    """Detect when the archive size exceeds available parts."""
    with pytest.raises(ValueError, match="exceeds maximum expected size"):
        _validate_archive_size(archive_size=301, part_urls=["url1", "url2"], chunk_size_bytes=150)


def test_validate_upload_completion_detects_mismatch() -> None:
    """Detect unfinished uploads when bytes uploaded do not match."""
    with pytest.raises(ValueError, match="Upload size mismatch"):
        _validate_upload_completion(bytes_uploaded=99, archive_size=100)
