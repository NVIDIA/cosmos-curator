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


"""Tests for proving a run can reach its storage before it starts."""

from collections.abc import Callable
from pathlib import Path

import pytest

from cosmos_curator.core.utils.storage.s3_client import S3Client
from cosmos_curator.core.utils.storage.storage_client import StoragePrefix
from cosmos_curator.next.recipes.multimodal_split import preflight as preflight_module
from cosmos_curator.next.recipes.multimodal_split.preflight import check_readable, check_writable


class _FakeClient(S3Client):
    """A remote client that records what it was asked to do."""

    def __init__(self, calls: list[tuple[str, str]], *, fail_on: str | None = None) -> None:
        self.calls = calls
        self.fail_on = fail_on

    def _maybe_fail(self, name: str) -> None:
        if self.fail_on == name:
            msg = "Not found S3 creds from /dev/shm/s3_creds_file"
            raise ValueError(msg)

    def list_recursive_directory(self, uri: StoragePrefix, limit: int = 0) -> list[StoragePrefix]:
        self.calls.append(("list", str(uri.path)))
        self._maybe_fail("list")
        assert limit == 1, "a read probe lists one object, not the whole prefix"
        return []

    def upload_bytes(self, dest: StoragePrefix, data: bytes) -> None:
        del data
        self.calls.append(("upload", str(dest.path)))
        self._maybe_fail("upload")

    def delete_object(self, dest: StoragePrefix) -> None:
        self.calls.append(("delete", str(dest.path)))
        self._maybe_fail("delete")


def _client_returning(calls: list[tuple[str, str]], *, fail_on: str | None = None) -> Callable[..., _FakeClient]:
    """Stand in for the dispatch that answers None locally and a client remotely."""

    def resolve(target_path: str, **kwargs: object) -> _FakeClient:
        del target_path, kwargs
        return _FakeClient(calls, fail_on=fail_on)

    return resolve


def test_a_local_input_that_is_not_a_directory_is_refused(tmp_path: Path) -> None:
    """The check exists to fail here rather than once per session inside Ray."""
    with pytest.raises(RuntimeError, match="Cannot read the input prefix"):
        check_readable(str(tmp_path / "absent"))


@pytest.mark.parametrize("authority", ["", "localhost"])
def test_a_file_uri_input_is_resolved_before_the_directory_check(tmp_path: Path, authority: str) -> None:
    """Verify local file URIs resolve to filesystem paths."""
    input_path = tmp_path / "recordings"
    input_path.mkdir()
    location = input_path.as_uri() if not authority else f"file://{authority}{input_path}"

    check_readable(location)


def test_a_file_uri_with_a_remote_authority_is_refused(tmp_path: Path) -> None:
    """Verify remote file authorities are refused for input."""
    location = f"file://other-host{tmp_path}/recordings"

    with pytest.raises(RuntimeError, match="Unsupported local file URI"):
        check_readable(location)


def test_a_failed_file_uri_input_names_the_configured_location(tmp_path: Path) -> None:
    """Verify input failures retain the configured URI in diagnostics."""
    location = (tmp_path / "absent").as_uri()

    with pytest.raises(RuntimeError, match="Cannot read the input prefix") as error:
        check_readable(location)
    assert location in str(error.value)


def test_a_local_output_is_proven_writable_and_left_clean(tmp_path: Path) -> None:
    """Writing is the only way to learn whether writing is permitted."""
    check_writable(str(tmp_path / "out"))

    assert [entry.name for entry in (tmp_path / "out").iterdir()] == []


@pytest.mark.parametrize("authority", ["", "localhost"])
def test_a_file_uri_output_probe_uses_the_resolved_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    authority: str,
) -> None:
    """Verify output probes use resolved URI paths."""
    monkeypatch.chdir(tmp_path)
    output_path = tmp_path / "out"
    location = output_path.as_uri() if not authority else f"file://{authority}{output_path}"

    check_writable(location)

    assert output_path.is_dir()
    assert list(output_path.iterdir()) == []
    assert not (tmp_path / "file:").exists()


def test_a_file_uri_with_a_remote_authority_is_refused_for_output(tmp_path: Path) -> None:
    """Verify remote file authorities are refused for output."""
    location = f"file://other-host{tmp_path}/media"

    with pytest.raises(RuntimeError, match="Unsupported local file URI"):
        check_writable(location)


def test_a_remote_read_probe_lists_one_object(monkeypatch: pytest.MonkeyPatch) -> None:
    """One page, not the bucket.

    Object storage has no prefix to stat, and listing the whole thing to prove
    credentials would cost more than the run it precedes.
    """
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(preflight_module, "get_storage_client", _client_returning(calls))

    check_readable("s3://bucket/recordings")

    # With the trailing slash: a prefix is matched as a raw string, so listing
    # "recordings" would also find "recordings-old/...", and an empty prefix
    # would pass on a sibling's contents.
    assert calls == [("list", "s3://bucket/recordings/")]


def test_a_remote_write_probe_is_removed_again(monkeypatch: pytest.MonkeyPatch) -> None:
    """The probe proves permission; leaving it behind would be litter."""
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(preflight_module, "get_storage_client", _client_returning(calls))

    check_writable("s3://bucket/media")

    assert [name for name, _ in calls] == ["upload", "delete"]
    written, removed = calls[0][1], calls[1][1]
    assert written == removed
    assert ".multimodal-split-preflight" in written


def test_missing_credentials_are_reported_before_the_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """The failure this exists for: one clear error, not one per session."""
    monkeypatch.setattr(preflight_module, "get_storage_client", _client_returning([], fail_on="list"))

    with pytest.raises(RuntimeError, match="Cannot read the input prefix"):
        check_readable("s3://bucket/recordings")


def test_a_probe_that_cannot_be_removed_does_not_fail_the_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """The write already proved what this checks; refusing over a leftover byte is the wrong trade."""
    monkeypatch.setattr(preflight_module, "get_storage_client", _client_returning([], fail_on="delete"))

    check_writable("s3://bucket/media")


def test_a_credential_failure_names_the_prefix_that_wanted_it(monkeypatch: pytest.MonkeyPatch) -> None:
    """Resolving a client is where a missing credential file is found.

    That error names the file it looked in and nothing else, so raised as it
    comes it says a credential is missing without saying which of a run's two
    locations needed it.
    """

    def refuse(target_path: str, **kwargs: object) -> _FakeClient:
        del target_path, kwargs
        msg = "Not found S3 creds from /dev/shm/s3_creds_file"
        raise ValueError(msg)

    monkeypatch.setattr(preflight_module, "get_storage_client", refuse)

    with pytest.raises(RuntimeError, match=r"Cannot read the input prefix s3://bucket/recordings"):
        check_readable("s3://bucket/recordings")
    with pytest.raises(RuntimeError, match=r"Cannot write to the output path s3://bucket/media"):
        check_writable("s3://bucket/media")


def test_a_local_directory_that_exists_is_accepted(tmp_path: Path) -> None:
    """The success path, which the failure cases alone cannot pin.

    A check that always raised, or never did, would otherwise pass every test
    here -- the file is full of assertions that it rejects things.
    """
    (tmp_path / "recordings").mkdir()

    check_readable(str(tmp_path / "recordings"))
