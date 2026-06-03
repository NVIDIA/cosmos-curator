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
