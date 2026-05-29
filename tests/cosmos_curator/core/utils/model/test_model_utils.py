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
"""Unit tests for cosmos_curator.core.utils.model.model_utils."""

import importlib
import pathlib
import shutil
import subprocess
import threading
from types import SimpleNamespace

import huggingface_hub
import pytest

from cosmos_curator.core.utils import environment
from cosmos_curator.core.utils.model import model_utils
from cosmos_curator.core.utils.storage import storage_client, storage_utils


def _identity(path: str) -> str:
    """Return the provided path string (helper for monkeypatch)."""
    return path


def _return_none(_: str) -> None:
    """Return None regardless of the argument (helper for monkeypatch)."""


@pytest.fixture
def mock_rclone_copy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock `rclone copy` so copy tests do not require rclone installed."""

    def fake_run(
        cmd: list[str],
        *,
        capture_output: bool,
        text: bool,
        check: bool,
    ) -> SimpleNamespace:
        assert capture_output is True
        assert text is True
        assert check is True
        assert cmd[0:3] == ["rclone", "copy", "--progress"]

        ignore_existing = "--ignore-existing" in cmd
        size_only = "--size-only" in cmd
        source = pathlib.Path(cmd[-2])
        destination = pathlib.Path(cmd[-1])

        for item in source.rglob("*"):
            if not item.is_file():
                continue
            relative = item.relative_to(source)
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)

            if target.exists():
                if ignore_existing:
                    continue
                if size_only and target.stat().st_size == item.stat().st_size:
                    continue
            shutil.copy2(item, target)

        return SimpleNamespace(stdout="", stderr="", returncode=0)

    monkeypatch.setattr(model_utils.subprocess, "run", fake_run)


def test_hack_copydir_to_cloud_storage_uploads_all_files(tmp_path: pathlib.Path) -> None:
    """Ensure the helper enqueues every file and waits for completion."""
    base_path = pathlib.Path(tmp_path)
    source = base_path / "weights"
    (source / "nested").mkdir(parents=True, exist_ok=True)
    (source / "nested2").mkdir(parents=True, exist_ok=True)
    (source / "root.txt").write_text("root")
    (source / "nested" / "params.bin").write_bytes(b"params")
    (source / "nested2" / "vocab.json").write_text("{}")

    class DummyUploader:
        def __init__(self) -> None:
            self.tasks: list[tuple[pathlib.Path, str]] = []
            self.block_called = False

        def add_task_file(self, path: pathlib.Path, dest: str) -> None:
            self.tasks.append((path, dest))

        def block_until_done(self) -> None:
            self.block_called = True

    uploader = DummyUploader()

    class DummyClient:
        def make_background_uploader(self) -> DummyUploader:
            return uploader

    destination = "s3://bucket/models/"
    model_utils._hack_copydir_to_cloud_storage(DummyClient(), source, destination)  # type: ignore[arg-type]

    expected = {
        (source / "root.txt", destination + "root.txt"),
        (source / "nested" / "params.bin", destination + "nested/params.bin"),
        (source / "nested2" / "vocab.json", destination + "nested2/vocab.json"),
    }
    assert set(uploader.tasks) == expected
    assert uploader.block_called


def test_upload_model_weights_requires_valid_prefix(tmp_path: pathlib.Path) -> None:
    """Ensure uploading without a real prefix is rejected."""

    class DummyClient:
        def make_background_uploader(self) -> None:
            error_msg = "Should not be constructed when validation fails"
            raise AssertionError(error_msg)

    with pytest.raises(ValueError, match="must be set to a valid S3 prefix"):
        model_utils._upload_model_weights_to_cloud_storage(DummyClient(), "model", tmp_path)  # type: ignore[arg-type]


def test_upload_model_weights_pushes_to_expected_prefix(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """Ensure uploads target the generated prefix and delegate to the copy helper."""
    captured: dict[str, object] = {}

    def fake_copy(client: object, source: pathlib.Path, destination: str) -> None:
        captured["client"] = client
        captured["source"] = source
        captured["destination"] = destination

    monkeypatch.setattr(model_utils, "_hack_copydir_to_cloud_storage", fake_copy)

    class DummyClient:
        """Simple stub client."""

    prefix = "s3://alt/"
    result = model_utils._upload_model_weights_to_cloud_storage(DummyClient(), "gpt", tmp_path, prefix)  # type: ignore[arg-type]
    assert result == f"{prefix}gpt/"
    assert captured["source"] == tmp_path
    assert captured["destination"] == f"{prefix}gpt/"


def test_download_model_weights_from_cloud_storage_syncs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """Ensure remote weights sync to the cache dir and use the storage client."""
    cache_dir = tmp_path / "cache"
    monkeypatch.setattr(environment, "CONTAINER_PATHS_MODEL_WEIGHT_CACHE_DIR", cache_dir)
    monkeypatch.setattr(storage_utils, "path_to_prefix", _identity)

    calls: list[tuple[str, pathlib.Path, bool, int]] = []

    class DummyClient:
        def sync_remote_to_local(
            self,
            storage_prefix: str,
            destination: pathlib.Path,
            *,
            delete: bool,
            chunk_size_bytes: int,
        ) -> None:
            calls.append((storage_prefix, destination, delete, chunk_size_bytes))

    dummy_client = DummyClient()
    prefix = "s3://bucket/models/"
    weights = "gpt2"

    def fake_get_client(path: str) -> DummyClient:
        assert path == f"{prefix}{weights}/"
        return dummy_client

    monkeypatch.setattr(storage_utils, "get_storage_client", fake_get_client)

    destination = model_utils._download_model_weights_from_cloud_storage_to_workspace(weights, prefix)
    assert destination == cache_dir / weights
    assert destination.exists()
    assert calls == [
        (
            f"{prefix}{weights}/",
            destination,
            True,
            storage_client.DOWNLOAD_CHUNK_SIZE_BYTES,
        )
    ]


def test_download_model_weights_from_cloud_storage_errors_without_client(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """Ensure a missing storage client triggers cleanup and a ValueError."""
    cache_dir = tmp_path / "cache_missing_client"
    monkeypatch.setattr(environment, "CONTAINER_PATHS_MODEL_WEIGHT_CACHE_DIR", cache_dir)
    monkeypatch.setattr(storage_utils, "path_to_prefix", _identity)
    monkeypatch.setattr(storage_utils, "get_storage_client", _return_none)

    with pytest.raises(ValueError, match="Failed to create storage client"):
        model_utils._download_model_weights_from_cloud_storage_to_workspace("model-x", "s3://bucket/")
    assert not (cache_dir / "model-x").exists()


def test_download_model_weights_from_cloud_storage_skips_existing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """Ensure no remote calls are made when weights already exist locally."""
    cache_dir = tmp_path / "cache_existing"
    monkeypatch.setattr(environment, "CONTAINER_PATHS_MODEL_WEIGHT_CACHE_DIR", cache_dir)
    destination = cache_dir / "already"
    destination.mkdir(parents=True, exist_ok=True)

    def fail_path_to_prefix(_: str) -> None:
        error_msg = "path_to_prefix should not be invoked when destination exists"
        raise AssertionError(error_msg)

    def fail_get_client(_: str) -> None:
        error_msg = "get_storage_client should not be invoked when destination exists"
        raise AssertionError(error_msg)

    monkeypatch.setattr(storage_utils, "path_to_prefix", fail_path_to_prefix)
    monkeypatch.setattr(storage_utils, "get_storage_client", fail_get_client)

    returned = model_utils._download_model_weights_from_cloud_storage_to_workspace("already", "s3://bucket/")
    assert returned == destination


def test_download_model_weights_from_cloud_storage_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure the public helper fans out to the private downloader for all names."""
    calls: list[tuple[str, str]] = []
    lock = threading.Lock()

    def fake_download(name: str, prefix: str) -> None:
        with lock:
            calls.append((name, prefix))

    monkeypatch.setattr(model_utils, "_download_model_weights_from_cloud_storage_to_workspace", fake_download)
    model_utils.download_model_weights_from_cloud_storage_to_workspace(["a", "b"], "remote-prefix")
    assert sorted(calls) == [("a", "remote-prefix"), ("b", "remote-prefix")]


def test_download_model_weights_from_local_to_workspace_copies_tree(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """Ensure local weight directories are copied into the cache."""
    cache_dir = tmp_path / "cache_local"
    monkeypatch.setattr(environment, "CONTAINER_PATHS_MODEL_WEIGHT_CACHE_DIR", cache_dir)

    local_root = tmp_path / "local_root"
    model_dir = local_root / "model-a"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "weights.bin").write_bytes(b"abc")
    (model_dir / "config.json").write_text("{}")

    model_utils._download_model_weights_from_local_to_workspace("model-a", str(local_root))
    destination = cache_dir / "model-a"
    assert (destination / "weights.bin").read_bytes() == b"abc"
    assert (destination / "config.json").read_text() == "{}"


def test_download_model_weights_from_local_to_workspace_batch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """Ensure the public helper fans out to the local downloader for all names."""
    calls: list[tuple[str, str]] = []
    lock = threading.Lock()

    def fake_download(name: str, root: str) -> None:
        with lock:
            calls.append((name, root))

    monkeypatch.setattr(model_utils, "_download_model_weights_from_local_to_workspace", fake_download)
    source_root = tmp_path / "source"
    model_utils.download_model_weights_from_local_to_workspace(["x", "y"], str(source_root))
    assert sorted(calls) == [("x", str(source_root)), ("y", str(source_root))]


def test_download_model_weights_from_huggingface_uses_api_token(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """Ensure Hugging Face downloads honor the config token and destination."""
    token_value = "hf-test-token"  # noqa: S105
    config = SimpleNamespace(huggingface=SimpleNamespace(api_key=token_value))
    monkeypatch.setattr(model_utils, "load_config", lambda: config)

    captured: dict[str, object] = {}

    def fake_snapshot_download(
        *,
        repo_id: str,
        revision: str | None,
        local_dir: pathlib.Path,
        token: str | None,
        allow_patterns: list[str] | None,
    ) -> None:
        captured.update(
            {
                "repo_id": repo_id,
                "revision": revision,
                "local_dir": local_dir,
                "token": token,
                "allow_patterns": allow_patterns,
            }
        )
        local_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)

    destination = tmp_path / "hf_dest"
    model_utils._download_model_weights_from_huggingface_to_workspace(
        "repo",
        "main",
        ["*.bin"],
        destination,
    )
    assert captured["repo_id"] == "repo"
    assert captured["revision"] == "main"
    assert captured["local_dir"] == destination
    assert captured["token"] == token_value
    assert captured["allow_patterns"] == ["*.bin"]


def test_download_model_weights_from_huggingface_invokes_reduce_for_t5(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """Ensure only the T5 model triggers the reduction pass."""
    monkeypatch.setattr(environment, "CONTAINER_PATHS_MODEL_WEIGHT_CACHE_DIR", tmp_path)

    def fake_download(
        _model_id: str,
        _revision: str | None,
        _allow_patterns: list[str] | None,
        destination: pathlib.Path,
    ) -> None:
        destination.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(model_utils, "_download_model_weights_from_huggingface_to_workspace", fake_download)
    reduce_calls: list[pathlib.Path] = []
    monkeypatch.setattr(model_utils, "_reduce_t5_model_weights", reduce_calls.append)

    model_utils.download_model_weights_from_huggingface_to_workspace("google-t5/t5-11b", "rev1")
    assert reduce_calls == [tmp_path / "google-t5/t5-11b"]

    reduce_calls.clear()
    model_utils.download_model_weights_from_huggingface_to_workspace("some-other-model", None)
    assert reduce_calls == []


def test_reduce_t5_model_weights_filters_encoder(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    """Ensure the reducer keeps encoder/shared weights and short-circuits on subsequent calls."""
    destination = tmp_path / "t5"
    destination.mkdir(parents=True, exist_ok=True)
    src = destination / "pytorch_model.bin"
    src.write_bytes(b"original")

    class FakeTorch:
        def __init__(self) -> None:
            self.saved: tuple[object, object] | None = None
            self.loaded = 0

        def load(
            self,
            path: pathlib.Path,
            *,
            map_location: object,
            weights_only: bool,
        ) -> dict[str, str]:
            self.loaded += 1
            assert path == src
            assert map_location == "cpu"
            assert weights_only is True
            return {
                "encoder.layer1": "enc",
                "decoder.layer1": "dec",
                "shared.weight": "shared",
            }

        def save(self, obj: dict[str, str], path: pathlib.Path) -> None:
            self.saved = (obj.copy(), path)
            path.write_text("saved")

    fake_torch = FakeTorch()
    real_import_module = importlib.import_module

    def fake_import_module(name: str) -> object:
        if name == "torch":
            return fake_torch
        return real_import_module(name)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    model_utils._reduce_t5_model_weights(destination)
    assert fake_torch.saved == (
        {"encoder.layer1": "enc", "shared.weight": "shared"},
        destination / "pytorch_model.bin.reduced",
    )
    assert fake_torch.loaded == 1

    # Reduced file already exists; ensure we short-circuit without re-loading.
    fake_torch.saved = ("unchanged", fake_torch.saved[1])
    model_utils._reduce_t5_model_weights(destination)
    assert fake_torch.loaded == 1
    assert fake_torch.saved[0] == "unchanged"


@pytest.mark.usefixtures("mock_rclone_copy")
def test_copy_model_weights_copies_new_files(tmp_path: pathlib.Path) -> None:
    """Test that copy_model_weights copies files that don't exist in destination."""
    source = tmp_path / "source"
    dest = tmp_path / "dest"
    source.mkdir()

    # Create source files including nested directories
    (source / "model.bin").write_bytes(b"model_data")
    (source / "config.json").write_text('{"key": "value"}')
    subdir = source / "subdir"
    subdir.mkdir()
    (subdir / "weights.safetensors").write_bytes(b"safetensor_data")
    subdir2 = subdir / "subdir2"
    subdir2.mkdir()
    (subdir2 / "weights2.safetensors").write_bytes(b"safetensor_data2")

    # Copy to destination
    model_utils.copy_model_weights(source, dest)

    # Verify all files were copied
    assert (dest / "model.bin").read_bytes() == b"model_data"
    assert (dest / "config.json").read_text() == '{"key": "value"}'
    assert (dest / "subdir" / "weights.safetensors").read_bytes() == b"safetensor_data"
    assert (dest / "subdir" / "subdir2" / "weights2.safetensors").read_bytes() == b"safetensor_data2"


@pytest.mark.parametrize(
    ("source_content", "dest_content", "size_check", "expected_content"),
    [
        (b"model_data", b"model_data", True, b"model_data"),
        (b"new_model_data", b"old_data", True, b"new_model_data"),
        (b"model_data", b"old_data", False, b"old_data"),
    ],
    ids=["same_size_check_true", "diff_size_check_true", "existing_size_check_false"],
)
@pytest.mark.usefixtures("mock_rclone_copy")
def test_copy_model_weights_with_existing_files(
    tmp_path: pathlib.Path,
    source_content: bytes,
    dest_content: bytes,
    *,
    size_check: bool,
    expected_content: bytes,
) -> None:
    """Test copy_model_weights behavior with existing files and different size_check settings."""
    source = tmp_path / "source"
    dest = tmp_path / "dest"
    source.mkdir()
    dest.mkdir()

    # Create files
    source_file = source / "model.bin"
    source_file.write_bytes(source_content)
    dest_file = dest / "model.bin"
    dest_file.write_bytes(dest_content)

    # Copy
    model_utils.copy_model_weights(source, dest, size_check=size_check)

    # Verify the content is as expected (whether skipped or copied)
    assert dest_file.read_bytes() == expected_content


@pytest.mark.parametrize(
    ("setup_func", "error_type", "error_match"),
    [
        (lambda _: None, FileNotFoundError, "Source directory does not exist"),
        (lambda p: p.write_text("content"), ValueError, "Source path is not a directory"),
    ],
    ids=["source_missing", "source_is_file"],
)
def test_copy_model_weights_error_cases(
    tmp_path: pathlib.Path,
    setup_func: object,
    error_type: type[Exception],
    error_match: str,
) -> None:
    """Test copy_model_weights raises appropriate errors for invalid inputs."""
    source = tmp_path / "source"
    setup_func(source)  # type: ignore[operator]
    dest = tmp_path / "dest"

    with pytest.raises(error_type, match=error_match):
        model_utils.copy_model_weights(source, dest)


@pytest.mark.usefixtures("mock_rclone_copy")
def test_copy_model_weights_creates_dest_if_missing(tmp_path: pathlib.Path) -> None:
    """Test that copy_model_weights creates destination directory if it doesn't exist."""
    source = tmp_path / "source"
    dest = tmp_path / "deep" / "nested" / "dest"
    source.mkdir()

    (source / "model.bin").write_bytes(b"data")

    # Destination doesn't exist yet
    assert not dest.exists()

    # Copy should create it
    model_utils.copy_model_weights(source, dest)

    assert dest.exists()
    assert (dest / "model.bin").read_bytes() == b"data"


@pytest.mark.usefixtures("mock_rclone_copy")
def test_copy_model_weights_copies_new_files_when_size_check_false(tmp_path: pathlib.Path) -> None:
    """Test that copy_model_weights still copies new files when size_check=False."""
    source = tmp_path / "source"
    dest = tmp_path / "dest"
    source.mkdir()
    dest.mkdir()

    # Create source files
    (source / "model.bin").write_bytes(b"model_data")
    (source / "config.json").write_text('{"key": "value"}')

    # Create only one file in destination
    (dest / "model.bin").write_bytes(b"existing")

    # Copy with size_check=False
    model_utils.copy_model_weights(source, dest, size_check=False)

    # Existing file should be unchanged
    assert (dest / "model.bin").read_bytes() == b"existing"
    # New file should be copied
    assert (dest / "config.json").read_text() == '{"key": "value"}'


def test_copy_model_weights_raises_when_rclone_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    """Test user-facing error when rclone executable is unavailable."""
    source = tmp_path / "source"
    dest = tmp_path / "dest"
    source.mkdir()

    def fake_run(cmd: list[str], *, capture_output: bool, text: bool, check: bool) -> None:
        del cmd, capture_output, text, check
        raise FileNotFoundError

    monkeypatch.setattr(model_utils.subprocess, "run", fake_run)
    with pytest.raises(OSError, match="rclone command not found"):
        model_utils.copy_model_weights(source, dest)


def test_copy_model_weights_raises_when_rclone_fails(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    """Test user-facing error when rclone returns a failure code."""
    source = tmp_path / "source"
    dest = tmp_path / "dest"
    source.mkdir()

    def fake_run(cmd: list[str], *, capture_output: bool, text: bool, check: bool) -> None:
        del capture_output, text, check
        raise subprocess.CalledProcessError(returncode=3, cmd=cmd, stderr="copy failed")

    monkeypatch.setattr(model_utils.subprocess, "run", fake_run)
    with pytest.raises(OSError, match="rclone failed with exit code 3"):
        model_utils.copy_model_weights(source, dest)
