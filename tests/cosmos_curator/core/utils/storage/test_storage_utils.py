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
"""Tests for cosmos_curator.core.utils.storage.storage_utils."""

import urllib.parse
from pathlib import Path

import pytest

from cosmos_curator.core.utils.storage import azure_client, s3_client, storage_utils
from cosmos_curator.core.utils.storage.storage_client import StoragePrefix

from .conftest import FakeStorageClient
from .conftest import remote_path as _remote_path


def test_is_remote_path_detects_known_schemes(tmp_path: Path) -> None:
    """Detect remote schemes and reject invalid inputs."""
    local_example = str(tmp_path / "file")
    assert storage_utils.is_remote_path(_remote_path("data"))
    assert storage_utils.is_remote_path("az://container/blob")
    assert not storage_utils.is_remote_path(local_example)
    assert not storage_utils.is_remote_path(None)


def test_get_storage_client_dispatches_to_implementations(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Request appropriate backend client based on path scheme."""
    s3_stub = object()
    azure_stub = object()
    s3_args: tuple[str, str, bool, bool] | None = None
    azure_args: tuple[str, str, bool, bool] | None = None

    def fake_s3_create(target_path: str, profile_name: str, *, can_overwrite: bool, can_delete: bool) -> object:
        nonlocal s3_args
        s3_args = (target_path, profile_name, can_overwrite, can_delete)
        return s3_stub

    def fake_azure_create(
        target_path: str,
        profile_name: str,
        *,
        can_overwrite: bool,
        can_delete: bool,
    ) -> object:
        nonlocal azure_args
        azure_args = (target_path, profile_name, can_overwrite, can_delete)
        return azure_stub

    monkeypatch.setattr(s3_client, "create_s3_client", fake_s3_create)
    monkeypatch.setattr(azure_client, "create_azure_client", fake_azure_create)

    assert (
        storage_utils.get_storage_client(
            _remote_path("path"),
            profile_name="profile",
            can_overwrite=True,
            can_delete=True,
        )
        is s3_stub
    )
    assert s3_args == (_remote_path("path"), "profile", True, True)

    assert (
        storage_utils.get_storage_client(
            "az://container/blob",
            profile_name="azure-profile",
        )
        is azure_stub
    )
    assert azure_args == ("az://container/blob", "azure-profile", False, False)
    assert storage_utils.get_storage_client(str(tmp_path / "local")) is None


def test_get_lance_storage_options_from_profiles(monkeypatch: pytest.MonkeyPatch) -> None:
    """Build Lance storage_options from configured S3/Azure profiles."""

    class DummyS3Config:
        aws_access_key_id = "id"
        aws_secret_access_key = "secret"  # noqa: S105
        aws_session_token = "token"  # noqa: S105
        endpoint_url = "http://localhost:9000"
        region = "us-west-2"

    class DummyAzureConfig:
        connection_string = "UseDevelopmentStorage=true"
        account_url = "https://acct.blob.core.windows.net"
        account_name = "acct"
        account_key = "key"

    def fake_s3_config(
        profile_name: str = "default",
        *,
        can_overwrite: bool = False,
        can_delete: bool = False,
    ) -> DummyS3Config:
        assert profile_name == "profile"
        assert can_overwrite is True
        assert can_delete is False
        return DummyS3Config()

    def fake_azure_config(
        *,
        profile_name: str = "default",
        can_overwrite: bool = False,
        can_delete: bool = False,
    ) -> DummyAzureConfig:
        assert profile_name == "az-profile"
        assert can_overwrite is True
        assert can_delete is False
        return DummyAzureConfig()

    monkeypatch.setattr(s3_client, "get_s3_client_config", fake_s3_config)
    monkeypatch.setattr(azure_client, "get_azure_client_config", fake_azure_config)

    s3_options = storage_utils.get_lance_storage_options(_remote_path("dataset"), profile_name="profile")
    assert s3_options == {
        "aws_access_key_id": "id",
        "aws_secret_access_key": "secret",
        "aws_session_token": "token",
        "aws_region": "us-west-2",
        "aws_endpoint": "http://localhost:9000",
    }

    azure_options = storage_utils.get_lance_storage_options("az://container/data", profile_name="az-profile")
    assert azure_options == {
        "account_name": "acct",
        "account_key": "key",
        "account_url": "https://acct.blob.core.windows.net",
        "connection_string": "UseDevelopmentStorage=true",
    }

    assert storage_utils.get_lance_storage_options("/tmp/local") is None  # noqa: S108


def test_path_to_prefix_validates_remote_paths() -> None:
    """Accept only valid remote storage paths."""
    prefix = storage_utils.path_to_prefix(_remote_path("root"))
    assert isinstance(prefix, StoragePrefix)
    assert str(prefix) == _remote_path("root")
    with pytest.raises(ValueError, match="not a valid remote storage path"):
        storage_utils.path_to_prefix("/not/remote")


def test_read_helpers_consume_local_paths(tmp_path: Path) -> None:
    """Exercise local byte/text/json readers."""
    data_file = tmp_path / "data.bin"
    data_file.write_bytes(b"payload")
    assert storage_utils.read_bytes(data_file) == b"payload"
    assert storage_utils.read_text(data_file) == "payload"

    json_file = tmp_path / "data.json"
    json_file.write_text('{"value": 1}', encoding="utf-8")
    assert storage_utils.read_json_file(json_file) == {"value": 1}


def test_read_bytes_accepts_file_uri(tmp_path: Path) -> None:
    """read_bytes reads a ``file://`` URI, the form producers emit via Path.as_uri().

    Artifact producers record durable URIs with ``Path.as_uri()`` (e.g.
    ``robot_action_split``'s ``action_data_uri``). A consumer reading that string
    back must resolve the ``file://`` scheme to the underlying local path rather
    than treating it as the literal path ``file:/...``.
    """
    data_file = tmp_path / "artifact.bin"
    data_file.write_bytes(b"payload")
    file_uri = data_file.as_uri()

    assert file_uri.startswith("file://")
    assert storage_utils.read_bytes(file_uri) == b"payload"


def test_read_bytes_file_uri_decodes_percent_escapes(tmp_path: Path) -> None:
    """A ``file://`` URI with percent-encoded characters resolves to the real path."""
    data_file = tmp_path / "has space.bin"
    data_file.write_bytes(b"spaced")
    file_uri = data_file.as_uri()

    assert "%20" in file_uri
    assert storage_utils.read_bytes(file_uri) == b"spaced"


def test_read_bytes_file_uri_localhost_authority_is_case_insensitive(tmp_path: Path) -> None:
    """A ``file://`` URI with an uppercase ``localhost`` authority still resolves locally.

    RFC 3986 makes the URI authority case-insensitive, so ``file://LOCALHOST/x``
    must resolve to the same local path as ``file:///x`` rather than falling
    through to a mangled literal path.
    """
    data_file = tmp_path / "artifact.bin"
    data_file.write_bytes(b"payload")
    uri = data_file.as_uri().replace("file://", "file://LOCALHOST", 1)

    assert uri.startswith("file://LOCALHOST/")
    assert storage_utils.read_bytes(uri) == b"payload"


def test_read_bytes_remote_path_uses_storage_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure remote reads use the provided storage client."""
    remote_path = _remote_path("root", "sample.bin")
    fake_client = FakeStorageClient({remote_path: b"remote-bytes"})
    monkeypatch.setattr(storage_utils, "get_storage_client", lambda *_args, **_kwargs: fake_client)

    assert storage_utils.read_bytes(remote_path) == b"remote-bytes"


def test_path_exists_handles_remote_and_local(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Confirm path_exists checks both local files and remote objects."""
    remote_path = _remote_path("root", "exists.bin")
    fake_client = FakeStorageClient({remote_path: b"data"})
    monkeypatch.setattr(storage_utils, "get_storage_client", lambda *_args, **_kwargs: fake_client)

    assert storage_utils.path_exists(remote_path) is True
    missing_remote = _remote_path("root", "missing.bin")
    assert storage_utils.path_exists(missing_remote) is False

    local_file = tmp_path / "local.txt"
    local_file.write_text("ok", encoding="utf-8")
    assert storage_utils.path_exists(local_file) is True


def test_path_exists_accepts_file_uri_for_a_present_file(tmp_path: Path) -> None:
    """A present file is found through its ``file://`` URI as well as its bare path.

    ``path_exists`` gates readers such as ``read_bytes``, which resolves the
    ``file://`` scheme. Disagreeing on the same string would let an existence
    check report an object that the reader can read as missing.
    """
    data_file = tmp_path / "artifact.npz"
    data_file.write_bytes(b"payload")

    assert storage_utils.path_exists(str(data_file)) is True
    assert storage_utils.path_exists(data_file.as_uri()) is True


def test_path_exists_reports_a_missing_file_uri_as_absent(tmp_path: Path) -> None:
    """An absent file is absent through its ``file://`` URI as well as its bare path."""
    missing = tmp_path / "never-written.npz"

    assert storage_utils.path_exists(str(missing)) is False
    assert storage_utils.path_exists(missing.as_uri()) is False


def test_path_exists_answers_for_a_bare_path_urlparse_rejects() -> None:
    """A bare path that no URI parser accepts is still answered, not raised on.

    ``urlparse`` reads a bracketed authority as a malformed IPv6 URL, but a
    double-slash path containing brackets is a legal filesystem path, and
    ``path_exists`` promises a bool for any string it is given.
    """
    assert storage_utils.path_exists("//host[0]/share/missing.bin") is False


def test_verify_path_respects_level(tmp_path: Path) -> None:
    """Verify level parameter walks the parent chain."""
    parent = tmp_path / "parent"
    child_dir = parent / "child"
    child_dir.mkdir(parents=True)
    file_path = child_dir / "file.txt"

    with pytest.raises(FileNotFoundError):
        storage_utils.verify_path(str(file_path))

    storage_utils.verify_path(str(file_path), level=1)


def test_create_path_builds_missing_directories(tmp_path: Path) -> None:
    """Create directories when they are absent."""
    target = tmp_path / "nested" / "dir"
    storage_utils.create_path(str(target))
    assert target.exists()


def test_is_path_nested_detects_relationships(tmp_path: Path) -> None:
    """Detect subset relationships between POSIX paths."""
    base = str(tmp_path / "base")
    child = str(tmp_path / "base" / "child")
    other = str(tmp_path / "other")
    assert storage_utils.is_path_nested(base, child)
    assert storage_utils.is_path_nested(child, base)
    assert not storage_utils.is_path_nested(base, other)


@pytest.mark.parametrize(
    ("clip_location", "input_root", "expected", "error_match"),
    [
        pytest.param("{root}/clips/abc.mp4", "{root}", "{root}/clips/abc.mp4", None, id="local-nested"),
        pytest.param("clips/abc.mp4", "{root}", None, "outside", id="local-relative-not-cwd-relative-is-rejected"),
        pytest.param("{root}/../outside/secret.mp4", "{root}", None, "outside", id="local-traversal"),
        pytest.param("{tmp}/var-secrets/secrets.json", "{root}", None, "outside", id="local-absolute-sibling"),
        pytest.param(_remote_path("input", "clips", "a.mp4"), "{root}", None, "scheme", id="remote-clip-local-root"),
        pytest.param("{root}/clips/a.mp4", _remote_path("input"), None, "scheme", id="local-clip-remote-root"),
        pytest.param(
            _remote_path("input", "clips", "abc.mp4"),
            _remote_path("input"),
            _remote_path("input", "clips", "abc.mp4"),
            None,
            id="remote-nested",
        ),
        pytest.param(
            _remote_path("input-evil", "clips", "a.mp4"),
            _remote_path("input"),
            None,
            "outside",
            id="remote-sibling-prefix-string-startswith-is-not-containment",
        ),
        pytest.param(
            "s3://other-bucket/input/clips/a.mp4",
            _remote_path("input"),
            None,
            "outside",
            id="remote-different-bucket",
        ),
        pytest.param(
            "az://test-container/input/clips/a.mp4",
            _remote_path("input"),
            None,
            "scheme",
            id="remote-azure-clip-s3-root",
        ),
        pytest.param(
            _remote_path("input", "clips", "a.mp4"),
            "az://test-container/input",
            None,
            "scheme",
            id="remote-s3-clip-azure-root",
        ),
    ],
)
def test_resolve_clip_location(
    tmp_path: Path,
    clip_location: str,
    input_root: str,
    expected: str | None,
    error_match: str | None,
) -> None:
    """clip_location must resolve inside input_root: same scheme, no escape, no sibling-prefix confusion."""
    root = tmp_path / "input"
    fmt = {"root": root, "tmp": tmp_path}
    clip_location = clip_location.format(**fmt)
    input_root = input_root.format(**fmt)

    if error_match is not None:
        with pytest.raises(ValueError, match=error_match):
            storage_utils.resolve_clip_location(clip_location, input_root)
        return

    result = storage_utils.resolve_clip_location(clip_location, input_root)
    assert str(result) == expected.format(**fmt)


@pytest.mark.parametrize(
    ("clip_format", "root_format"),
    [
        pytest.param("uri", "uri", id="uri-clip-uri-root"),
        pytest.param("uri", "path", id="uri-clip-path-root"),
        pytest.param("path", "uri", id="path-clip-uri-root"),
    ],
)
def test_resolve_clip_location_accepts_local_file_uris(
    tmp_path: Path,
    clip_format: str,
    root_format: str,
) -> None:
    """Resolve file URIs consistently with bare local paths."""
    root = tmp_path / "input root"
    clip = root / "clips" / "clip name.mp4"
    clip_location = clip.as_uri() if clip_format == "uri" else str(clip)
    input_root = root.as_uri() if root_format == "uri" else str(root)

    result = storage_utils.resolve_clip_location(clip_location, input_root)

    assert result == clip.resolve()


def test_resolve_clip_location_relative_path_is_always_cwd_relative(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Resolve relative clip locations against the process working directory."""
    root = tmp_path / "input"
    clip = root / "clips" / "abc.mp4"
    clip.parent.mkdir(parents=True)
    clip.write_bytes(b"data")
    monkeypatch.chdir(root)  # cwd == root: "clips/abc.mp4" resolves to the real file

    result = storage_utils.resolve_clip_location("clips/abc.mp4", str(root))

    assert result == clip.resolve()


def test_resolve_clip_location_relative_path_outside_cwd_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A relative clip_location that doesn't land under input_root via cwd is rejected, not joined."""
    root = tmp_path / "input"
    root.mkdir()
    monkeypatch.chdir(tmp_path)  # cwd != root; "clips/abc.mp4" would resolve outside root

    with pytest.raises(ValueError, match="outside"):
        storage_utils.resolve_clip_location("clips/abc.mp4", str(root))


def test_get_full_path_appends_components(tmp_path: Path) -> None:
    """Build full paths for both local and remote prefixes."""
    local_base = tmp_path / "data"
    result = storage_utils.get_full_path(local_base, "sub", "file.txt")
    assert isinstance(result, Path)
    assert result == local_base / "sub" / "file.txt"

    remote_result = storage_utils.get_full_path(_remote_path("root"), "nested", "file.txt")
    assert isinstance(remote_result, StoragePrefix)
    assert str(remote_result) == _remote_path("root", "nested", "file.txt")

    remote_prefix = storage_utils.path_to_prefix(_remote_path("root"))
    chained = storage_utils.get_full_path(remote_prefix, "child")
    assert isinstance(chained, StoragePrefix)
    assert str(chained) == _remote_path("root", "child")


def test_get_files_relative_from_local_tree(tmp_path: Path) -> None:
    """Return sorted relative files for local directories."""
    base = tmp_path / "dataset"
    (base / "dir").mkdir(parents=True)
    (base / "dir" / "one.txt").write_text("1", encoding="utf-8")
    (base / "two.txt").write_text("2", encoding="utf-8")

    files = storage_utils.get_files_relative(str(base))
    assert files == ["dir/one.txt", "two.txt"]


def test_get_files_relative_from_local_tree_with_limit(tmp_path: Path) -> None:
    """Apply the provided limit for local directory listings."""
    base = tmp_path / "dataset"
    (base / "dir").mkdir(parents=True)
    (base / "dir" / "one.txt").write_text("1", encoding="utf-8")
    (base / "two.txt").write_text("2", encoding="utf-8")

    files = storage_utils.get_files_relative(str(base), limit=1)
    assert files == ["dir/one.txt"]


def test_get_files_relative_from_remote_prefix() -> None:
    """Filter remote objects based on the provided limit."""
    remote_root = _remote_path("root")
    fake_client = FakeStorageClient(
        {
            f"{remote_root}/alpha.txt": b"",
            f"{remote_root}/nested/beta.txt": b"",
            f"{remote_root}/nested/gamma.txt": b"",
        },
    )
    files = storage_utils.get_files_relative(remote_root, client=fake_client, limit=2)
    assert files == ["alpha.txt", "nested/beta.txt"]
    assert fake_client.last_list_limit == 2


def test_get_files_relative_from_remote_prefix_defensively_truncates_limit() -> None:
    """Apply a final limit even if a backend over-returns listing results."""
    remote_root = _remote_path("root")

    class OverReturningClient(FakeStorageClient):
        def list_recursive_directory(self, uri: StoragePrefix, limit: int = 0) -> list[StoragePrefix]:
            self.last_list_limit = limit
            prefix = str(uri).rstrip("/") + "/"
            return [storage_utils.path_to_prefix(path) for path in sorted(self.objects) if path.startswith(prefix)]

    fake_client = OverReturningClient(
        {
            f"{remote_root}/alpha.txt": b"",
            f"{remote_root}/nested/beta.txt": b"",
            f"{remote_root}/nested/gamma.txt": b"",
        },
    )
    files = storage_utils.get_files_relative(remote_root, client=fake_client, limit=2)
    assert files == ["alpha.txt", "nested/beta.txt"]
    assert fake_client.last_list_limit == 2


def test_get_directories_relative_extracts_top_level(tmp_path: Path) -> None:
    """Summarize top-level directories from local data."""
    base = tmp_path / "dirs"
    (base / "a").mkdir(parents=True)
    (base / "a" / "x.txt").write_text("x", encoding="utf-8")
    (base / "b").mkdir()
    (base / "b" / "y.txt").write_text("y", encoding="utf-8")

    dirs = storage_utils.get_directories_relative(str(base))
    assert dirs == ["a", "b"]


def test_get_next_file_returns_first_available(tmp_path: Path) -> None:
    """Return the next available sequential file name."""
    output = tmp_path / "output"
    output.mkdir()
    (output / "clip_0.json").write_text("0", encoding="utf-8")
    (output / "clip_1.json").write_text("1", encoding="utf-8")

    next_file = storage_utils.get_next_file("clip", "json", str(output))
    assert isinstance(next_file, Path)
    assert next_file.name == "clip_2.json"


def test_backup_file_creates_incremental_backups(tmp_path: Path) -> None:
    """Create sequential .bak files for local paths."""
    target = tmp_path / "file.txt"
    target.write_text("first", encoding="utf-8")
    storage_utils.backup_file(target)
    bak1 = tmp_path / "file.txt.bak1"
    assert bak1.read_text(encoding="utf-8") == "first"
    assert not target.exists()

    target.write_text("second", encoding="utf-8")
    storage_utils.backup_file(target)
    bak2 = tmp_path / "file.txt.bak2"
    assert bak2.read_text(encoding="utf-8") == "second"


def test_extract_parquet_files_filters_and_limits(tmp_path: Path) -> None:
    """Filter parquet files and honor limits."""
    base = tmp_path / "parquet"
    (base / "nested").mkdir(parents=True)
    (base / "a.parquet").write_text("a", encoding="utf-8")
    (base / "nested" / "b.parquet").write_text("b", encoding="utf-8")
    (base / "ignore.txt").write_text("c", encoding="utf-8")

    results = storage_utils.extract_parquet_files(str(base), profile_name="default", limit=1)
    assert len(results) == 1
    assert isinstance(results[0], Path)
    assert results[0].name == "a.parquet"


def _oserror_from_s3_code(code: str) -> OSError:
    """Build the OSError-wraps-ClientError chain that smart_open produces for an S3 error."""
    import botocore.exceptions  # noqa: PLC0415 -- only needed in this helper

    boto_exc = botocore.exceptions.ClientError(
        error_response={"Error": {"Code": code, "Message": "test"}},
        operation_name="GetObject",
    )
    wrapped = OSError(f"unable to access object (code={code})")
    wrapped.__cause__ = boto_exc
    return wrapped


def test_is_missing_object_error_detects_local_filenotfound() -> None:
    """FileNotFoundError (raised by Python's open() for local paths) reads as missing."""
    assert storage_utils.is_missing_object_error(FileNotFoundError("nope"))


def test_is_missing_object_error_detects_s3_nosuchkey_chained_in_oserror() -> None:
    """smart_open wraps boto3's NoSuchKey ClientError in OSError; the chained cause flags it as missing."""
    assert storage_utils.is_missing_object_error(_oserror_from_s3_code("NoSuchKey"))


def test_is_missing_object_error_rejects_other_client_errors() -> None:
    """A non-NoSuchKey ClientError (e.g. AccessDenied) is treated as 'present but unreadable'."""
    assert not storage_utils.is_missing_object_error(_oserror_from_s3_code("AccessDenied"))


def test_is_missing_object_error_rejects_unrelated_exceptions() -> None:
    """A plain ValueError (e.g. decode failure) is not 'missing' -- it's 'unreadable'."""
    assert not storage_utils.is_missing_object_error(ValueError("bad bytes"))


def test_backend_key_groups_urls_by_scheme_and_bucket() -> None:
    """Two URLs share a key exactly when they share a scheme and a bucket."""
    assert storage_utils.backend_key("s3://bucket-a/x.bin") == storage_utils.backend_key("s3://bucket-a/y.bin")
    assert storage_utils.backend_key("s3://bucket-a/x.bin") != storage_utils.backend_key("s3://bucket-b/x.bin")


def test_backend_key_collapses_local_paths_to_one_backend() -> None:
    """A bare path and a ``file://`` URL name the same backend, so one cache entry serves both."""
    assert storage_utils.backend_key("/data/action/a.bin") == storage_utils.backend_key("file:///data/action/a.bin")


def test_backend_key_returns_a_key_for_an_authority_the_parser_rejects() -> None:
    """A malformed authority yields a key rather than raising, so one bad row cannot fail a batch.

    The URLs this keys are untrusted table data read row by row, and every caller
    is a read path whose contract is that an unusable URL costs its own row. An
    unmatched bracket is what makes this authority unparseable: a tab or a newline
    would be DELETED by the parser rather than rejected by it, so such a value
    would exercise the ordinary path instead of the fallback.
    """
    malformed = "s3://[bad/clip.mp4"
    with pytest.raises(ValueError, match="IPv6"):
        urllib.parse.urlparse(malformed)

    assert storage_utils.backend_key(malformed)


def test_backend_key_keeps_backends_apart_when_the_authority_is_unparseable() -> None:
    """The fallback key merges two URLs only when their raw authority text is identical.

    One shared constant for every rejected URL would route reads for one backend
    through another backend's cached client, which is the whole failure this key
    exists to prevent; objects under a single malformed authority still share one
    entry rather than resolving per object.
    """
    assert storage_utils.backend_key("s3://[bad/a.mp4") != storage_utils.backend_key("az://[bad/a.mp4")
    assert storage_utils.backend_key("s3://[bad/a.mp4") == storage_utils.backend_key("s3://[bad/b.mp4")
