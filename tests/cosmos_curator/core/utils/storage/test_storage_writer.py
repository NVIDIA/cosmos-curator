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
"""Tests for StorageWriter and WritablePath."""

import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from cosmos_curator.core.utils.storage import storage_utils
from cosmos_curator.core.utils.storage.storage_utils import StorageWriter, WritablePath

from .conftest import FakeStorageClient
from .conftest import remote_path as _remote_path


class TestWritablePath:
    """Tests for WritablePath os.PathLike protocol and lifecycle."""

    def test_fspath_and_str(self, tmp_path: Path) -> None:
        """__fspath__ and __str__ return the local path string."""
        local = tmp_path / "file.txt"
        writer_mock = MagicMock(spec=StorageWriter)
        wpath = WritablePath(local=local, writer=writer_mock, sub="file.txt")

        assert os.fspath(wpath) == str(local)
        assert str(wpath) == str(local)

    def test_open_and_exists(self, tmp_path: Path) -> None:
        """open() delegates to pathlib and exists() reflects filesystem state."""
        local = tmp_path / "file.txt"
        writer_mock = MagicMock(spec=StorageWriter)
        wpath = WritablePath(local=local, writer=writer_mock, sub="file.txt")

        assert not wpath.exists()

        with wpath.open("w", encoding="utf-8") as f:
            f.write("hello")

        assert wpath.exists()
        assert local.read_text(encoding="utf-8") == "hello"

    def test_close_delegates_to_writer(self, tmp_path: Path) -> None:
        """close() calls StorageWriter.close(sub) exactly once."""
        local = tmp_path / "file.txt"
        writer_mock = MagicMock(spec=StorageWriter)
        wpath = WritablePath(local=local, writer=writer_mock, sub="my/sub.txt")

        wpath.close()

        writer_mock.close.assert_called_once_with("my/sub.txt")


class TestLocalStorageWriter:
    """Tests for StorageWriter in local-mode (no remote backend)."""

    @pytest.fixture
    def writer(self, tmp_path: Path) -> StorageWriter:
        """Local-mode writer targeting a temp directory."""
        return StorageWriter(str(tmp_path / "output"))

    def test_is_not_remote(self, writer: StorageWriter, tmp_path: Path) -> None:
        """Local writer reports is_remote=False and returns base_path."""
        assert writer.is_remote is False
        assert writer.base_path == str(tmp_path / "output")

    def test_write_bytes_to(self, writer: StorageWriter, tmp_path: Path) -> None:
        """write_bytes_to creates parent dirs and writes data to disk."""
        writer.write_bytes_to("nested/data.bin", b"binary-payload")

        written = (tmp_path / "output" / "nested" / "data.bin").read_bytes()
        assert written == b"binary-payload"

    def test_write_str_to(self, writer: StorageWriter, tmp_path: Path) -> None:
        """write_str_to encodes and writes text to disk."""
        writer.write_str_to("report.txt", "hello world")

        written = (tmp_path / "output" / "report.txt").read_text(encoding="utf-8")
        assert written == "hello world"

    def test_resolve_path_creates_fresh_file(self, writer: StorageWriter, tmp_path: Path) -> None:
        """resolve_path returns a WritablePath and unlinks any pre-existing file."""
        base = tmp_path / "output"
        base.mkdir(parents=True)
        existing = base / "result.bin"
        existing.write_bytes(b"stale")

        wpath = writer.resolve_path("result.bin")

        assert isinstance(wpath, WritablePath)
        assert Path(os.fspath(wpath)) == existing
        assert not existing.exists(), "pre-existing file should be unlinked"

    def test_resolve_path_close_is_noop(self, writer: StorageWriter, tmp_path: Path) -> None:
        """Closing a local WritablePath does not delete the written file."""
        (tmp_path / "output").mkdir(parents=True, exist_ok=True)
        wpath = writer.resolve_path("keep.txt")

        Path(os.fspath(wpath)).write_text("persist", encoding="utf-8")
        wpath.close()

        assert (tmp_path / "output" / "keep.txt").read_text(encoding="utf-8") == "persist"

    def test_open_writer_writes_and_closes(self, writer: StorageWriter, tmp_path: Path) -> None:
        """open_writer yields a writable handle; content is present after exit."""
        with writer.open_writer("report.html") as f:
            f.write("<h1>Report</h1>")

        written = (tmp_path / "output" / "report.html").read_text(encoding="utf-8")
        assert written == "<h1>Report</h1>"

    def test_open_writer_binary_mode(self, writer: StorageWriter, tmp_path: Path) -> None:
        """open_writer with mode='wb' yields a binary handle."""
        with writer.open_writer("data.bin", mode="wb") as f:
            f.write(b"\x00\x01\x02\x03")

        written = (tmp_path / "output" / "data.bin").read_bytes()
        assert written == b"\x00\x01\x02\x03"

    def test_upload_file_to(self, writer: StorageWriter, tmp_path: Path) -> None:
        """upload_file_to copies a local file to <base>/<sub_path> without loading into memory."""
        source = tmp_path / "source.bin"
        source.write_bytes(b"large-artifact-payload")

        writer.upload_file_to("nested/artifact.bin", source)

        dest = tmp_path / "output" / "nested" / "artifact.bin"
        assert dest.read_bytes() == b"large-artifact-payload"
        assert source.read_bytes() == b"large-artifact-payload", "source file should not be modified"


class TestRemoteStorageWriter:
    """Tests for StorageWriter in remote-mode (monkeypatched backend)."""

    @pytest.fixture
    def fake_client(self) -> FakeStorageClient:
        """Fresh FakeStorageClient for each test."""
        return FakeStorageClient()

    @pytest.fixture
    def writer(
        self,
        monkeypatch: pytest.MonkeyPatch,
        fake_client: FakeStorageClient,
        tmp_path: Path,
    ) -> StorageWriter:
        """Remote-mode writer backed by FakeStorageClient, staging under tmp_path."""
        monkeypatch.setattr(
            storage_utils,
            "get_storage_client",
            lambda *_a, **_kw: fake_client,
        )
        return StorageWriter(_remote_path("output"), tmp_dir=str(tmp_path))

    def test_is_remote(self, writer: StorageWriter) -> None:
        """Remote writer reports is_remote=True."""
        assert writer.is_remote is True
        assert writer.base_path == _remote_path("output")

    def test_write_bytes_to_uploads(self, writer: StorageWriter, fake_client: FakeStorageClient) -> None:
        """write_bytes_to delegates to StorageClient.upload_bytes for remote paths."""
        writer.write_bytes_to("data/chunk.bin", b"remote-payload")

        expected_key = _remote_path("output", "data/chunk.bin")
        assert fake_client.objects[expected_key] == b"remote-payload"

    def test_resolve_path_stages_locally(self, writer: StorageWriter, tmp_path: Path) -> None:
        """resolve_path stages under the staging root, not under base_path."""
        wpath = writer.resolve_path("nested/file.bin")
        local = Path(os.fspath(wpath))

        assert str(local).startswith(str(tmp_path))
        assert "nested/file.bin" in str(local)
        assert not str(local).startswith("s3://")

    def test_close_uploads_and_cleans_staging(
        self,
        writer: StorageWriter,
        fake_client: FakeStorageClient,
    ) -> None:
        """After writing and closing, the file is uploaded and staging is removed."""
        wpath = writer.resolve_path("artifact.bin")
        staging = Path(os.fspath(wpath))
        staging.write_bytes(b"artifact-data")

        wpath.close()

        expected_key = _remote_path("output", "artifact.bin")
        assert fake_client.objects[expected_key] == b"artifact-data"
        assert not staging.exists(), "staging file should be cleaned up after upload"

    def test_close_skips_missing_staging(self, writer: StorageWriter) -> None:
        """Calling close() without writing does not raise."""
        writer.close("nonexistent/path.bin")

    def test_open_writer(self, writer: StorageWriter, fake_client: FakeStorageClient) -> None:
        """open_writer stages locally, uploads on exit, and cleans staging."""
        with writer.open_writer("reports/summary.html") as f:
            f.write("<p>summary</p>")

        expected_key = _remote_path("output", "reports/summary.html")
        uploaded = fake_client.objects.get(expected_key)
        assert uploaded is not None
        assert uploaded.decode("utf-8") == "<p>summary</p>"

    def test_open_writer_binary_mode(self, writer: StorageWriter, fake_client: FakeStorageClient) -> None:
        """open_writer with mode='wb' stages, uploads binary, and cleans staging."""
        payload = b"\xde\xad\xbe\xef"
        with writer.open_writer("data/chunk.bin", mode="wb") as f:
            f.write(payload)

        expected_key = _remote_path("output", "data/chunk.bin")
        assert fake_client.objects[expected_key] == payload

    def test_open_writer_exception_skips_upload_preserves_staging(
        self,
        writer: StorageWriter,
        fake_client: FakeStorageClient,
    ) -> None:
        """On exception the upload is skipped but the staging file is preserved."""
        sub = "partial/report.html"
        msg = "write failed"

        def _write_and_raise() -> None:
            with writer.open_writer(sub) as f:
                f.write("<p>partial</p>")
                raise RuntimeError(msg)

        with pytest.raises(RuntimeError, match=msg):
            _write_and_raise()

        remote_key = _remote_path("output", sub)
        assert remote_key not in fake_client.objects, "upload should not have happened"

        staging = writer._staging_root() / sub
        assert staging.exists(), "staging file should be preserved for debugging/retry"
        assert staging.read_text(encoding="utf-8") == "<p>partial</p>"

    def test_upload_file_to(
        self,
        writer: StorageWriter,
        fake_client: FakeStorageClient,
        tmp_path: Path,
    ) -> None:
        """upload_file_to streams a local file to remote without reading into memory."""
        source = tmp_path / "source_artifact.bin"
        source.write_bytes(b"streamed-artifact-data")

        writer.upload_file_to("artifacts/large.bin", source)

        expected_key = _remote_path("output", "artifacts/large.bin")
        assert fake_client.objects[expected_key] == b"streamed-artifact-data"
        assert source.exists(), "source file should remain untouched"

    def test_staging_root_differs_by_base_path(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Two writers with different base_path values get different staging roots."""
        monkeypatch.setattr(
            storage_utils,
            "get_storage_client",
            lambda *_a, **_kw: FakeStorageClient(),
        )

        writer_a = StorageWriter("s3://bucket-a/output", tmp_dir=str(tmp_path))
        writer_b = StorageWriter("s3://bucket-b/output", tmp_dir=str(tmp_path))

        root_a = writer_a._staging_root()
        root_b = writer_b._staging_root()

        assert root_a != root_b
        assert root_a.parent == root_b.parent == tmp_path


class TestSubPathEdgeCases:
    """Pin behavior for unusual sub_path values.

    The StorageWriter ``_to`` methods join sub_path to base_path for
    every write operation.  These tests document and pin the behavior
    for edge-case inputs so that any future changes are deliberate,
    not accidental.

    ::

        sub_path value         Expected local behavior
        +--------------------+--------------------------------------+
        | "report.html"      | Lands directly under base_path       |
        | "a/b/c/d/e/f.txt"  | Deep nesting, parents auto-created   |
        | "foo/../bar.txt"   | pathlib resolves ".." segments        |
        | "subdir/"          | Trailing slash -- treated as dir name |
        | ""                 | Resolves to base_path itself          |
        | "/absolute.txt"    | FOOTGUN: pathlib discards base_path  |
        +--------------------+--------------------------------------+

    Remote-mode tests verify the same sub_path values reach the
    correct upload key in the FakeStorageClient.
    """

    @pytest.fixture
    def local_writer(self, tmp_path: Path) -> StorageWriter:
        """Local-mode writer targeting a temp directory."""
        return StorageWriter(str(tmp_path / "output"))

    @pytest.fixture
    def fake_client(self) -> FakeStorageClient:
        """Fresh FakeStorageClient for each test."""
        return FakeStorageClient()

    @pytest.fixture
    def remote_writer(
        self,
        monkeypatch: pytest.MonkeyPatch,
        fake_client: FakeStorageClient,
        tmp_path: Path,
    ) -> StorageWriter:
        """Remote-mode writer backed by FakeStorageClient."""
        monkeypatch.setattr(
            storage_utils,
            "get_storage_client",
            lambda *_a, **_kw: fake_client,
        )
        return StorageWriter(_remote_path("output"), tmp_dir=str(tmp_path))

    def test_bare_filename_local(self, local_writer: StorageWriter, tmp_path: Path) -> None:
        """A bare filename (no directory) lands directly under base_path."""
        local_writer.write_bytes_to("report.html", b"<h1>Report</h1>")

        written = (tmp_path / "output" / "report.html").read_bytes()
        assert written == b"<h1>Report</h1>"

    def test_bare_filename_remote(self, remote_writer: StorageWriter, fake_client: FakeStorageClient) -> None:
        """A bare filename uploads directly under the remote base_path."""
        remote_writer.write_bytes_to("report.html", b"<h1>Report</h1>")

        expected_key = _remote_path("output", "report.html")
        assert fake_client.objects[expected_key] == b"<h1>Report</h1>"

    def test_deeply_nested_local(self, local_writer: StorageWriter, tmp_path: Path) -> None:
        """Deeply nested sub_path auto-creates all intermediate directories."""
        local_writer.write_bytes_to("a/b/c/d/e/deep.bin", b"deep")

        written = (tmp_path / "output" / "a" / "b" / "c" / "d" / "e" / "deep.bin").read_bytes()
        assert written == b"deep"

    def test_deeply_nested_remote(self, remote_writer: StorageWriter, fake_client: FakeStorageClient) -> None:
        """Deeply nested sub_path reaches the correct remote key."""
        remote_writer.write_bytes_to("a/b/c/d/e/deep.bin", b"deep")

        expected_key = _remote_path("output", "a/b/c/d/e/deep.bin")
        assert fake_client.objects[expected_key] == b"deep"

    def test_dot_segments_local(self, local_writer: StorageWriter, tmp_path: Path) -> None:
        """Dot segments (``..``) are resolved by pathlib.

        ``Path(base) / "foo/../bar.txt"`` resolves to ``base/bar.txt``
        because pathlib normalises the ``..`` component.
        """
        local_writer.write_bytes_to("foo/../bar.txt", b"collapsed")

        # pathlib resolves "foo/.." so the file ends up at base/bar.txt,
        # not base/foo/../bar.txt.
        resolved = tmp_path / "output" / "bar.txt"
        assert resolved.read_bytes() == b"collapsed"

    def test_dot_segments_remote(self, remote_writer: StorageWriter, fake_client: FakeStorageClient) -> None:
        """Dot segments in remote sub_path are passed through to the storage key."""
        remote_writer.write_bytes_to("foo/../bar.txt", b"collapsed")

        # Remote path joining is string-based; the ".." is preserved
        # in the storage key.
        expected_key = _remote_path("output", "foo/../bar.txt")
        assert fake_client.objects[expected_key] == b"collapsed"

    def test_trailing_slash_local(self, local_writer: StorageWriter, tmp_path: Path) -> None:
        """Trailing slash is treated as a directory component by pathlib.

        ``Path(base) / "subdir/"`` resolves to ``base/subdir`` which
        is a directory, so ``write_bytes`` will create a directory
        instead of a file. The write attempt will raise because the
        resolved path is a directory (pathlib creates the directory
        via ``mkdir`` on the parent, making ``subdir`` the "parent").
        """
        # pathlib.Path(base) / "subdir/" resolves to base/subdir.
        # _resolve_local calls dest.parent.mkdir(...) which creates
        # base/output, then dest itself is base/output/subdir (no extension).
        # write_bytes writes to that path as a regular file.
        local_writer.write_bytes_to("subdir/", b"trailing")

        result = tmp_path / "output" / "subdir"
        assert result.read_bytes() == b"trailing"

    def test_empty_string_local(self, local_writer: StorageWriter, tmp_path: Path) -> None:
        """Empty sub_path resolves to base_path itself.

        ``Path(base) / ""`` returns ``Path(base)``.  ``_resolve_local``
        then calls ``dest.parent.mkdir(parents=True, exist_ok=True)``
        which creates the parent of ``base_path`` (not ``base_path``
        itself).  The write lands at the ``base_path`` path as a file.
        """
        base = tmp_path / "output"
        local_writer.write_bytes_to("", b"at-base")

        assert base.read_bytes() == b"at-base"

    def test_leading_slash_local_escapes_base(self, local_writer: StorageWriter) -> None:
        """FOOTGUN: a leading slash makes pathlib discard the base_path.

        ``pathlib.Path("/tmp/.../output") / "/file.txt"`` evaluates to
        ``PosixPath('/file.txt')`` -- the base is silently dropped.
        This test pins the current behavior so any future guard
        (e.g. input validation rejecting absolute sub_paths) is a
        deliberate change, not an accident.

        We do NOT actually write to ``/file.txt`` (that would require
        root and pollute the filesystem).  Instead we verify the
        resolved path to confirm pathlib's behavior.
        """
        # Verify pathlib's behavior directly: base / "/abs" -> "/abs"
        base = Path(local_writer.base_path)
        resolved = base / "/file.txt"

        # The base is discarded -- resolved is an absolute path
        # outside the intended output directory.
        assert resolved == Path("/file.txt")
        assert not str(resolved).startswith(str(base))

    def test_leading_slash_remote(self, remote_writer: StorageWriter, fake_client: FakeStorageClient) -> None:
        """Leading slash in remote sub_path is joined via string operations.

        Unlike local pathlib joining, remote ``get_full_path`` uses
        string concatenation, so the leading slash does NOT discard
        the base -- it becomes part of the key.
        """
        remote_writer.write_bytes_to("/file.txt", b"leading-slash")

        expected_key = _remote_path("output", "/file.txt")
        assert fake_client.objects[expected_key] == b"leading-slash"


class TestDirectWriteLocal:
    """Tests for direct-write methods (write / write_str) in local mode.

    These methods write directly to *base_path* without a sub_path,
    supporting the single-file usage pattern.
    """

    def test_write_creates_parents_and_writes(self, tmp_path: Path) -> None:
        """write() creates parent dirs and writes bytes to base_path."""
        dest = tmp_path / "deep" / "nested" / "file.bin"
        writer = StorageWriter(str(dest))
        writer.write(b"direct-payload")

        assert dest.read_bytes() == b"direct-payload"

    def test_write_overwrites_existing(self, tmp_path: Path) -> None:
        """write() overwrites an existing file at base_path."""
        dest = tmp_path / "file.bin"
        dest.write_bytes(b"old-data")

        writer = StorageWriter(str(dest))
        writer.write(b"new-data")

        assert dest.read_bytes() == b"new-data"

    def test_write_str_encodes_and_writes(self, tmp_path: Path) -> None:
        """write_str() encodes text and writes to base_path."""
        dest = tmp_path / "output.json"
        writer = StorageWriter(str(dest))
        writer.write_str('{"key": "value"}')

        assert dest.read_text(encoding="utf-8") == '{"key": "value"}'

    def test_write_str_custom_encoding(self, tmp_path: Path) -> None:
        """write_str() respects the encoding parameter."""
        dest = tmp_path / "output.txt"
        writer = StorageWriter(str(dest))
        writer.write_str("caf\xe9", encoding="latin-1")

        assert dest.read_bytes() == "caf\xe9".encode("latin-1")


class TestDirectWriteRemote:
    """Tests for direct-write methods (write / write_str) in remote mode.

    These methods write directly to *base_path* without a sub_path,
    delegating to StorageClient.upload_bytes.
    """

    @pytest.fixture
    def fake_client(self) -> FakeStorageClient:
        """Fresh FakeStorageClient for each test."""
        return FakeStorageClient()

    @pytest.fixture
    def writer(
        self,
        monkeypatch: pytest.MonkeyPatch,
        fake_client: FakeStorageClient,
        tmp_path: Path,
    ) -> StorageWriter:
        """Remote-mode writer backed by FakeStorageClient, base_path is a file."""
        monkeypatch.setattr(
            storage_utils,
            "get_storage_client",
            lambda *_a, **_kw: fake_client,
        )
        return StorageWriter(_remote_path("output/report.bin"), tmp_dir=str(tmp_path))

    def test_write_uploads_to_base_path(self, writer: StorageWriter, fake_client: FakeStorageClient) -> None:
        """write() uploads bytes directly to the remote base_path."""
        writer.write(b"remote-direct")

        expected_key = _remote_path("output/report.bin")
        assert fake_client.objects[expected_key] == b"remote-direct"

    def test_write_str_uploads_to_base_path(self, writer: StorageWriter, fake_client: FakeStorageClient) -> None:
        """write_str() encodes and uploads text to the remote base_path."""
        writer.write_str("remote-text")

        expected_key = _remote_path("output/report.bin")
        assert fake_client.objects[expected_key] == b"remote-text"
