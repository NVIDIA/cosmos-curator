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

"""Storage-helper contracts for ``next.utils.storage``.

Covers ``write_media`` (including the ``file://`` locations ``artifact_uri``
hands out), ``remove_prefix`` (whole staging tree / prefix), ``remove_object``
(a single durable artifact), ``read_media_if_present``, and ``write_if_absent``.
The removal and read helpers share the degenerate paths every cleanup call
relies on: an absent target is a no-op, and a non-absence failure propagates
rather than being reported as success.

The URI rules those entry points apply live in ``next.utils.local_uri`` and are
covered in ``test_local_uri``; what remains here is that ``write_if_absent`` and
``artifact_uri`` really do apply them, which no test of the parser alone shows.

Remote branches use fake storage clients, so no network or credentials are
needed. The capped, retrying media read layered on these helpers is covered
separately in ``test_media_reader``.
"""

import pathlib
from unittest.mock import Mock, call

import pytest
from botocore.exceptions import ClientError

from cosmos_curator.core.utils.storage.azure_client import AzurePrefix
from cosmos_curator.core.utils.storage.s3_client import S3Client
from cosmos_curator.core.utils.storage.storage_client import StoragePrefix
from cosmos_curator.next.utils import storage


class _FakeDeleteClient:
    """Minimal delete-capable storage client recording what it was asked to delete.

    Stands in for an ``S3Client`` / ``AzureClient`` on the remote branch: it
    yields a fixed listing for ``list_recursive_directory`` and remembers every
    ``delete_object`` target so a test can assert the recursive/single delete
    contract without a real backend. The listing is fixed but the requested
    prefix is recorded, because ``deleted`` alone cannot catch a caller that
    swept the wrong subtree - the canned reply comes back whatever prefix it was
    handed, so only ``listed`` distinguishes the two.

    Attributes:
        listed: Prefix of every ``list_recursive_directory`` call, in order.
        deleted: Target of every ``delete_object`` call, in order.

    """

    def __init__(self, listing: list[StoragePrefix], *, present: bool = True) -> None:
        self._listing = listing
        self._present = present
        self.listed: list[StoragePrefix] = []
        self.deleted: list[StoragePrefix] = []

    def list_recursive_directory(self, uri: StoragePrefix, _limit: int = 0) -> list[StoragePrefix]:
        self.listed.append(uri)
        return self._listing

    def object_exists(self, _dest: StoragePrefix) -> bool:
        return self._present

    def delete_object(self, dest: StoragePrefix) -> None:
        self.deleted.append(dest)


class _FakeReadClient:
    """Minimal read client: reports presence and yields fixed bytes for the object.

    Stands in for an ``S3Client`` on the remote read branch. ``object_exists``
    drives the not-found-to-None mapping (a real client returns False on a 404),
    and ``download_object_as_bytes`` returns the stored payload only when present.
    """

    def __init__(self, *, present: bool, data: bytes = b"") -> None:
        self._present = present
        self._data = data

    def object_exists(self, _dest: StoragePrefix) -> bool:
        return self._present

    def download_object_as_bytes(self, _uri: StoragePrefix, _chunk_size_bytes: int = 0) -> bytes:
        return self._data


def test_a_file_uri_destination_is_written_where_it_names(tmp_path: pathlib.Path) -> None:
    """A file:// location must land where it names, not under the scheme.

    ``Path`` reads the scheme as a directory, so the object goes into a relative
    ``file:/`` tree under whatever the worker's cwd was -- silently, so the run
    reports success and the bytes are somewhere else.
    """
    storage.write_media(f"{tmp_path.as_uri()}/nested/out.bin", b"payload")

    assert (tmp_path / "nested" / "out.bin").read_bytes() == b"payload"


def test_the_uri_this_package_hands_out_is_one_it_can_write_to(tmp_path: pathlib.Path) -> None:
    """``artifact_uri`` normalizes a local path to ``file://``; writing must round-trip it."""
    recorded = storage.artifact_uri(str(tmp_path / "clip.mp4"))
    # Checked before writing, not after: a plain path would write fine, so an
    # artifact_uri that stopped returning a URI would leave this test passing
    # for a shape it does not cover.
    assert recorded.startswith("file://")

    storage.write_media(recorded, b"payload")

    assert (tmp_path / "clip.mp4").read_bytes() == b"payload"


def test_remove_prefix_deletes_a_local_directory_tree(tmp_path: pathlib.Path) -> None:
    """A local prefix and everything under it is removed from disk."""
    root = tmp_path / "staging"
    (root / "sub").mkdir(parents=True)
    (root / "sub" / "shard.parquet").write_bytes(b"data")

    storage.remove_prefix(str(root))

    assert not root.exists()


def test_remove_prefix_absent_local_target_is_a_noop(tmp_path: pathlib.Path) -> None:
    """Removing a prefix that never existed is a no-op, not an error (re-run path)."""
    storage.remove_prefix(str(tmp_path / "never_written"))


def test_remove_prefix_propagates_non_absence_local_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """A permissions-style failure surfaces; only table-absence is swallowed."""
    root = tmp_path / "staging"
    root.mkdir()

    def raise_permission_error(*_args: object, **_kwargs: object) -> None:
        message = "Operation not permitted"
        raise PermissionError(message)

    monkeypatch.setattr(storage.shutil, "rmtree", raise_permission_error)
    with pytest.raises(PermissionError, match="Operation not permitted"):
        storage.remove_prefix(str(root))


@pytest.mark.parametrize("partial_failure", [False, True])
def test_remove_prefix_s3_batches_deletes_and_reports_failures(
    monkeypatch: pytest.MonkeyPatch, *, partial_failure: bool
) -> None:
    """Use bulk requests for every page and propagate failures inside a successful response."""
    client = object.__new__(S3Client)
    client.can_delete = True
    client.s3 = backend = Mock()
    keys = [{"Key": f"run/{index}"} for index in range(1_073)]
    pages = [keys[:1_000], keys[1_000:]]
    backend.get_paginator.return_value.paginate.return_value = [{"Contents": page} for page in pages]
    backend.delete_objects.side_effect = [
        {},
        {"Errors": [{"Key": "run/1000", "Message": "AccessDenied"}]} if partial_failure else {},
    ]
    get_client = Mock(return_value=client)
    monkeypatch.setattr(storage, "get_storage_client", get_client)

    if partial_failure:
        with pytest.raises(OSError, match="AccessDenied"):
            storage.remove_prefix("s3://bucket/run", storage_profile="test-profile")
    else:
        storage.remove_prefix("s3://bucket/run", storage_profile="test-profile")

    get_client.assert_called_once_with("s3://bucket/run", profile_name="test-profile", can_delete=True)
    backend.get_paginator.assert_called_once_with("list_objects_v2")
    backend.get_paginator.return_value.paginate.assert_called_once_with(Bucket="bucket", Prefix="run/")
    assert backend.delete_objects.call_args_list == [
        call(Bucket="bucket", Delete={"Objects": page, "Quiet": True}) for page in pages
    ]
    backend.delete_object.assert_not_called()


def test_remove_prefix_remote_ignores_per_object_missing_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """A listed object already gone at delete time is skipped without aborting the sweep.

    The miss sits in the middle of the listing on purpose. A helper that stopped
    at the first absent object would still empty a listing whose last entry is
    the missing one, so only an entry AFTER the miss can distinguish skipping it
    from aborting on it.
    """
    listing: list[StoragePrefix] = [
        AzurePrefix("az://container/run/a"),
        AzurePrefix("az://container/run/b"),
        AzurePrefix("az://container/run/c"),
    ]

    class _ClientMissingOnSecond(_FakeDeleteClient):
        def delete_object(self, dest: StoragePrefix) -> None:
            if dest.path.endswith("run/b"):
                raise FileNotFoundError(dest.path)
            super().delete_object(dest)

    client = _ClientMissingOnSecond(listing)
    monkeypatch.setattr(storage, "get_storage_client", lambda *_a, **_k: client)

    storage.remove_prefix("az://container/run")

    assert client.deleted == [listing[0], listing[2]]


def test_remove_prefix_remote_deletes_every_listed_object_az(monkeypatch: pytest.MonkeyPatch) -> None:
    """The same remote contract holds for az:// prefixes."""
    listing: list[StoragePrefix] = [AzurePrefix("az://container/run/a"), AzurePrefix("az://container/run/b")]
    client = _FakeDeleteClient(listing)
    monkeypatch.setattr(storage, "get_storage_client", lambda *_a, **_k: client)

    storage.remove_prefix("az://container/run")

    assert client.listed == [AzurePrefix("az://container/run")]
    assert client.deleted == listing


def test_remove_prefix_remote_without_client_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """A remote URI with no delete-capable client is a hard error, not a silent skip."""
    monkeypatch.setattr(storage, "get_storage_client", lambda *_a, **_k: None)

    with pytest.raises(ValueError, match="no storage client"):
        storage.remove_prefix("s3://bucket/run")


def test_remove_object_deletes_a_local_file(tmp_path: pathlib.Path) -> None:
    """A single local object is removed while its parent directory is untouched."""
    target = tmp_path / "report.json"
    target.write_text("{}")

    storage.remove_object(str(target))

    assert not target.exists()
    assert tmp_path.exists()


def test_remove_object_absent_local_target_is_a_noop(tmp_path: pathlib.Path) -> None:
    """Removing an object that never existed is a no-op, not an error."""
    storage.remove_object(str(tmp_path / "missing.json"))


def test_remove_object_propagates_non_absence_local_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """A permissions-style failure on a single-object delete surfaces, not swallowed."""
    target = tmp_path / "report.json"
    target.write_text("{}")

    def raise_permission_error(_self: pathlib.Path, *_args: object, **_kwargs: object) -> None:
        message = "Operation not permitted"
        raise PermissionError(message)

    monkeypatch.setattr(storage.Path, "unlink", raise_permission_error)
    with pytest.raises(PermissionError, match="Operation not permitted"):
        storage.remove_object(str(target))


def test_remove_object_refuses_a_local_directory(tmp_path: pathlib.Path) -> None:
    """remove_object deletes single objects, never a tree: a directory propagates.

    This is the invariant that distinguishes remove_object from remove_prefix.
    The real stdlib error must surface rather than be swallowed as an absent
    no-op; its exact type is platform-dependent (``IsADirectoryError`` on Linux,
    ``PermissionError`` on macOS), so this pins the shared ``OSError`` contract -
    critically, it is not the ``FileNotFoundError`` the helper tolerates.
    """
    directory = tmp_path / "staging"
    directory.mkdir()

    with pytest.raises(OSError):  # noqa: PT011 - type is platform-dependent; see docstring
        storage.remove_object(str(directory))

    assert directory.exists()


def test_remove_object_remote_absent_target_is_a_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    """A remote 404 (object_exists False) deletes nothing and raises no error."""
    client = _FakeDeleteClient(listing=[], present=False)
    monkeypatch.setattr(storage, "get_storage_client", lambda *_a, **_k: client)

    storage.remove_object("s3://bucket/run/report.json")

    assert client.deleted == []


def test_remove_object_remote_deletes_the_single_object(monkeypatch: pytest.MonkeyPatch) -> None:
    """The remote branch deletes exactly the one object named, never a listing."""
    client = _FakeDeleteClient(listing=[])
    monkeypatch.setattr(storage, "get_storage_client", lambda *_a, **_k: client)

    storage.remove_object("s3://bucket/run/report.json")

    assert client.listed == []
    assert len(client.deleted) == 1
    assert client.deleted[0].path.endswith("report.json")


def test_remove_object_remote_deletes_the_single_object_az(monkeypatch: pytest.MonkeyPatch) -> None:
    """The same single-object delete contract holds for az:// URIs."""
    client = _FakeDeleteClient(listing=[])
    monkeypatch.setattr(storage, "get_storage_client", lambda *_a, **_k: client)

    storage.remove_object("az://container/run/report.json")

    assert client.listed == []
    assert len(client.deleted) == 1
    assert client.deleted[0].path.endswith("report.json")


def test_remove_object_remote_without_client_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """A remote object URI with no delete-capable client is a hard error."""
    monkeypatch.setattr(storage, "get_storage_client", lambda *_a, **_k: None)

    with pytest.raises(ValueError, match="no storage client"):
        storage.remove_object("s3://bucket/run/report.json")


def test_read_media_if_present_returns_local_bytes(tmp_path: pathlib.Path) -> None:
    """A present local object yields its exact bytes."""
    target = tmp_path / "report.json"
    target.write_bytes(b'{"schema": 1}')

    assert storage.read_media_if_present(str(target)) == b'{"schema": 1}'


def test_read_media_if_present_absent_local_target_returns_none(tmp_path: pathlib.Path) -> None:
    """An absent local object collapses to None, not a FileNotFoundError."""
    assert storage.read_media_if_present(str(tmp_path / "missing.json")) is None


def test_read_media_if_present_propagates_non_absence_local_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """A permissions-style read failure surfaces; only absence maps to None."""
    target = tmp_path / "report.json"
    target.write_bytes(b"{}")

    def raise_permission_error(_self: pathlib.Path) -> bytes:
        message = "Operation not permitted"
        raise PermissionError(message)

    monkeypatch.setattr(storage.Path, "read_bytes", raise_permission_error)
    with pytest.raises(PermissionError, match="Operation not permitted"):
        storage.read_media_if_present(str(target))


def test_read_media_if_present_remote_absent_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """A remote 404 (object_exists False) reads as None without downloading."""
    monkeypatch.setattr(storage, "get_storage_client", lambda *_a, **_k: _FakeReadClient(present=False))

    assert storage.read_media_if_present("s3://bucket/run/report.json") is None


def test_read_media_if_present_remote_returns_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    """A present remote object yields its downloaded bytes."""
    monkeypatch.setattr(storage, "get_storage_client", lambda *_a, **_k: _FakeReadClient(present=True, data=b"payload"))

    assert storage.read_media_if_present("s3://bucket/run/report.json") == b"payload"


def test_read_media_if_present_remote_returns_bytes_az(monkeypatch: pytest.MonkeyPatch) -> None:
    """The same read contract holds for az:// URIs."""
    monkeypatch.setattr(storage, "get_storage_client", lambda *_a, **_k: _FakeReadClient(present=True, data=b"payload"))

    assert storage.read_media_if_present("az://container/run/report.json") == b"payload"


def test_read_media_if_present_remote_without_client_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """A remote URI with no readable storage client is a hard error, not a silent None."""
    monkeypatch.setattr(storage, "get_storage_client", lambda *_a, **_k: None)

    with pytest.raises(ValueError, match="no storage client"):
        storage.read_media_if_present("s3://bucket/run/report.json")


class _FakeS3Api:
    """Minimal ``client.s3`` stand-in for the conditional-create path.

    Records the kwargs of every ``put_object`` so a test can assert the
    conditional header that makes the create atomic, and refuses the write with
    the real 412 shape when the object is already there.

    Attributes:
        calls: Keyword arguments of every ``put_object`` call, in order.

    """

    def __init__(self, *, existing: bool) -> None:
        self._existing = existing
        self.calls: list[dict[str, object]] = []

    def put_object(self, **kwargs: object) -> None:
        self.calls.append(kwargs)
        if self._existing:
            raise ClientError(
                {"Error": {"Code": "PreconditionFailed"}, "ResponseMetadata": {"HTTPStatusCode": 412}},
                "PutObject",
            )


class _FakeS3Client:
    """Storage client exposing only the raw ``s3`` handle the create path uses."""

    def __init__(self, *, existing: bool = False) -> None:
        self.s3 = _FakeS3Api(existing=existing)


def test_write_if_absent_creates_an_absent_local_object(tmp_path: pathlib.Path) -> None:
    """An absent object is written and the call reports that it created it."""
    target = tmp_path / "nested" / "workspace.json"

    assert storage.write_if_absent(str(target), b'{"schema": 1}') is True
    assert target.read_bytes() == b'{"schema": 1}'


def test_write_if_absent_leaves_an_existing_local_object_untouched(tmp_path: pathlib.Path) -> None:
    """A second call creates nothing and does not overwrite the first call's bytes."""
    target = tmp_path / "workspace.json"
    storage.write_if_absent(str(target), b"first")

    assert storage.write_if_absent(str(target), b"second") is False
    assert target.read_bytes() == b"first"


def test_write_if_absent_reports_losing_the_local_create_race(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """A concurrent creator between the presence check and the link makes this call the loser.

    The existence check cannot see a writer that lands microseconds later, so the
    link is the real arbiter: only its ``FileExistsError`` distinguishes "I
    created it" from "somebody else did", which is what a namespace claim needs.
    """

    def raise_file_exists(*_args: object, **_kwargs: object) -> None:
        raise FileExistsError

    monkeypatch.setattr(storage.os, "link", raise_file_exists)

    assert storage.write_if_absent(str(tmp_path / "workspace.json"), b"mine") is False


def test_write_if_absent_accepts_a_file_uri(tmp_path: pathlib.Path) -> None:
    """A ``file://`` URI resolves to the same local object a bare path would."""
    target = tmp_path / "workspace.json"

    assert storage.write_if_absent(target.as_uri(), b"payload") is True
    assert target.read_bytes() == b"payload"


def test_write_if_absent_refuses_a_local_target_that_is_a_directory(tmp_path: pathlib.Path) -> None:
    """A non-file entry at the target name is refused, not reported as an existing object.

    ``False`` means "another caller created this object", and the caller acts on
    it by reading those bytes back. A directory can never become the object, so
    the honest answer names the occupied path instead of sending the caller into
    a read that fails with an errno the name does not explain.
    """
    target = tmp_path / "workspace.json"
    target.mkdir()

    with pytest.raises(ValueError, match="exists and is not a regular file"):
        storage.write_if_absent(str(target), b"payload")


def test_write_if_absent_refuses_a_local_target_that_is_a_dangling_symlink(tmp_path: pathlib.Path) -> None:
    """A link to nothing is refused too, because the create underneath it cannot land.

    ``exists()`` follows the link and reports absence, so this is the one
    occupied name a presence check alone misses; ``os.link`` would then fail and
    the caller would be told somebody else won.
    """
    target = tmp_path / "workspace.json"
    target.symlink_to(tmp_path / "missing")

    with pytest.raises(ValueError, match="exists and is not a regular file"):
        storage.write_if_absent(str(target), b"payload")


def test_write_if_absent_accepts_a_symlink_to_an_existing_file_as_an_existing_object(
    tmp_path: pathlib.Path,
) -> None:
    """A link resolving to a regular file is an existing object, so the create reports False.

    The positive control for the refusals above: the guard rejects names that
    cannot hold the object, not every name that is not itself a plain file.
    """
    real = tmp_path / "real.json"
    real.write_bytes(b"first")
    target = tmp_path / "workspace.json"
    target.symlink_to(real)

    assert storage.write_if_absent(str(target), b"second") is False
    assert real.read_bytes() == b"first"


def test_write_if_absent_rejects_a_scheme_with_no_atomic_create() -> None:
    """A scheme with no create-if-absent primitive fails loudly rather than racing."""
    with pytest.raises(ValueError, match="atomic create-if-absent is unavailable for scheme 'az'"):
        storage.write_if_absent("az://container/run/workspace.json", b"payload")


def test_write_if_absent_creates_an_absent_s3_object_conditionally(monkeypatch: pytest.MonkeyPatch) -> None:
    """The remote create is issued with the conditional header that makes it atomic.

    Without ``IfNoneMatch`` the put would succeed unconditionally and silently
    overwrite another run's manifest, so the header is the mechanism and not an
    implementation detail.
    """
    client = _FakeS3Client()
    monkeypatch.setattr(storage, "S3Client", _FakeS3Client)
    monkeypatch.setattr(storage, "get_storage_client", lambda *_a, **_k: client)

    assert storage.write_if_absent("s3://bucket/run/workspace.json", b"payload") is True
    assert client.s3.calls[0]["IfNoneMatch"] == "*"


def test_write_if_absent_reports_an_existing_s3_object(monkeypatch: pytest.MonkeyPatch) -> None:
    """A refused precondition is the remote form of "it was already there"."""
    client = _FakeS3Client(existing=True)
    monkeypatch.setattr(storage, "S3Client", _FakeS3Client)
    monkeypatch.setattr(storage, "get_storage_client", lambda *_a, **_k: client)

    assert storage.write_if_absent("s3://bucket/run/workspace.json", b"payload") is False


def test_write_if_absent_rejects_a_one_slash_file_uri() -> None:
    """Atomic local creation applies the same canonical file-URI contract as reads."""
    with pytest.raises(ValueError, match="Unsupported local file URI"):
        storage.write_if_absent("file:/abs/path.mp4", b"payload")


def test_write_if_absent_rejects_a_file_uri_with_an_unencoded_delimiter() -> None:
    """The create path applies the same delimiter contract as the read path.

    The write direction is where truncation does the damage: a claim on
    ``file:///runs#2`` that silently lands on ``/runs`` overwrites nothing but
    claims the wrong namespace, and the caller is told it won.
    """
    with pytest.raises(ValueError, match="Unsupported local file URI"):
        storage.write_if_absent("file:///data/runs#2/workspace.json", b"payload")


def test_artifact_uri_rejects_a_file_uri_with_an_unencoded_delimiter() -> None:
    """A truncated location must not be normalized into a durable manifest record.

    ``artifact_uri`` output is written into rows and receipts, so accepting the
    truncation here would persist a URI naming an object the run never touched.
    """
    with pytest.raises(ValueError, match="Unsupported local file URI"):
        storage.artifact_uri("file:///data/runs#2/clip.mp4")


class _ConflictingS3Api:
    """``client.s3`` stand-in answering a fixed number of 409 conflicts first.

    409 ``ConditionalRequestConflict`` is what S3 returns when another
    conditional write is in flight against the same key. Unlike 412 it says
    nothing about whether the object exists, so it is the code the create path
    must retry rather than read as an answer.

    Attributes:
        calls: Keyword arguments of every ``put_object`` call, in order.

    """

    def __init__(self, *, conflicts: int) -> None:
        self._remaining = conflicts
        self.calls: list[dict[str, object]] = []

    def put_object(self, **kwargs: object) -> None:
        self.calls.append(kwargs)
        if self._remaining > 0:
            self._remaining -= 1
            raise ClientError(
                {"Error": {"Code": "ConditionalRequestConflict"}, "ResponseMetadata": {"HTTPStatusCode": 409}},
                "PutObject",
            )


class _ConflictingS3Client:
    """Storage client exposing only the raw ``s3`` handle the create path uses."""

    def __init__(self, *, conflicts: int) -> None:
        self.s3 = _ConflictingS3Api(conflicts=conflicts)


def test_write_if_absent_retries_a_concurrent_conditional_write_conflict(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 409 is a lost race against another writer, not proof the object is there.

    Read as "already exists" it would report a create that never happened and
    hand the caller an object to read that may never be written. Asserting the
    call count as well as the verdict, because returning ``True`` on the first
    attempt would satisfy the verdict alone.
    """
    client = _ConflictingS3Client(conflicts=2)
    monkeypatch.setattr(storage, "S3Client", _ConflictingS3Client)
    monkeypatch.setattr(storage, "get_storage_client", lambda *_a, **_k: client)

    assert storage.write_if_absent("s3://bucket/run/workspace.json", b"payload") is True
    assert len(client.s3.calls) == 3


def test_write_if_absent_raises_when_conflicts_outlast_the_bounded_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    """Spending the retry budget raises rather than falling through to a verdict.

    Whether the object exists is still unknown at exhaustion, so both booleans
    are wrong: ``True`` claims a create that did not happen and ``False`` sends
    the caller to read an object that may not be there.
    """
    monkeypatch.setattr(storage, "_S3_CONFLICT_MAX_ATTEMPTS", 2)
    client = _ConflictingS3Client(conflicts=99)
    monkeypatch.setattr(storage, "S3Client", _ConflictingS3Client)
    monkeypatch.setattr(storage, "get_storage_client", lambda *_a, **_k: client)

    with pytest.raises(ClientError, match="ConditionalRequestConflict"):
        storage.write_if_absent("s3://bucket/run/workspace.json", b"payload")
    assert len(client.s3.calls) == 2
