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

"""``MediaObjectReader`` contracts: the byte cap, the retry budget, the fault class.

Covers the three decisions a caller depends on:

* the size ceiling is enforced before the transfer, and again on the delivered
  bytes when the store reports no size;
* a transport fault spends the whole attempt budget, while an object fault and a
  credential fault each stop after one attempt;
* which of the three ``MediaReadError`` subclasses a given backend failure
  becomes, since that choice decides drop-the-row versus fail-the-run;
* a reader survives a pickle round-trip after it has read, and the copy still
  reads, because Ray pickles one to reach a worker.

Every remote branch is exercised through a fake storage client, so no test needs
network or credentials.
"""

import pathlib
import pickle
import threading
from collections.abc import Callable
from typing import Never

import pytest
import tenacity
from azure.core.exceptions import ClientAuthenticationError, ResourceNotFoundError
from botocore.exceptions import ClientError, NoCredentialsError

from cosmos_curator.core.utils.storage.storage_client import StoragePrefix, StorageStat
from cosmos_curator.next.utils import media_reader


class _FakeMediaClient:
    """Storage client for the reader: one metadata answer, one transfer answer.

    Both answers may be an exception instance, which the fake raises instead of
    returning, so a test can drive each fault class without a backend. Call
    counts are recorded because the retry contract is about how many attempts a
    fault gets, not about the delay between them.

    Attributes:
        stat_calls: Number of ``stat`` calls.
        download_calls: Number of ``download_object_as_bytes`` calls.

    """

    def __init__(self, *, stat_result: StorageStat | Exception, download_result: bytes | Exception) -> None:
        self._stat_result = stat_result
        self._download_result = download_result
        # Holds a lock so the fake is as unpicklable as the clients it stands in
        # for, which carry connection-pool locks and SSL contexts. Without it a
        # fake pickles cleanly, and a test asserting that a reader survives the
        # round-trip would pass whether or not the reader excludes its cache.
        self._connection_lock = threading.Lock()
        self.stat_calls = 0
        self.download_calls = 0

    def stat(self, _dest: StoragePrefix) -> StorageStat:
        self.stat_calls += 1
        if isinstance(self._stat_result, Exception):
            raise self._stat_result
        return self._stat_result

    def download_object_as_bytes(self, _uri: StoragePrefix, _chunk_size_bytes: int = 0) -> bytes:
        self.download_calls += 1
        if isinstance(self._download_result, Exception):
            raise self._download_result
        return self._download_result


type InstallStatErrorClient = Callable[[Exception], _FakeMediaClient]
type WrapProviderError = Callable[[Exception], Exception]


def _direct_error(error: Exception) -> Exception:
    """Return a provider error without a wrapper."""
    return error


def _one_level_chained_error(error: Exception) -> Exception:
    """Return a missing-object wrapper whose direct cause is ``error``."""
    wrapper = FileNotFoundError("storage wrapper reported a missing object")
    wrapper.__cause__ = error
    return wrapper


def _s3_client_error(code: str) -> ClientError:
    """Build an S3 ``ClientError`` carrying one provider code."""
    return ClientError(
        {"Error": {"Code": code}, "ResponseMetadata": {"HTTPStatusCode": 404}},
        "HeadObject",
    )


def _azure_resource_not_found_error(code: str | None) -> ResourceNotFoundError:
    """Build an Azure resource error with an optional provider code."""
    error = ResourceNotFoundError(message=f"Azure resource missing: {code}")
    if code is not None:
        error.error_code = code
    return error


@pytest.fixture
def instant_retry_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Collapse tenacity's backoff to nothing for the retry-budget tests.

    The reader's contract is the number of attempts a transport fault gets; the
    schedule between them is deliberately not a contract, so waiting through it
    would only make the suite slow.
    """
    monkeypatch.setattr(tenacity.nap.time, "sleep", lambda _seconds: None)


@pytest.fixture
def install_stat_error_client(monkeypatch: pytest.MonkeyPatch) -> InstallStatErrorClient:
    """Install a fake remote client whose metadata lookup raises the supplied error."""

    def install(error: Exception) -> _FakeMediaClient:
        client = _FakeMediaClient(stat_result=error, download_result=b"unused")
        monkeypatch.setattr(media_reader, "get_storage_client", lambda *_a, **_k: client)
        return client

    return install


def test_media_object_reader_returns_a_local_object_verbatim(tmp_path: pathlib.Path) -> None:
    """A readable object within the cap yields its exact bytes."""
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"\x00\x01\x02")

    assert media_reader.MediaObjectReader().read(str(clip), max_bytes=16) == b"\x00\x01\x02"


def test_media_object_reader_refuses_an_oversized_object_before_transferring_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """The cap is enforced off the metadata, so the oversized bytes are never pulled in.

    Reading first and checking after would let one mis-populated URI naming a
    whole source recording exhaust the worker's heap, which costs every other row
    the worker was holding rather than just this one.
    """
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"0123456789")

    def fail_if_read(_self: pathlib.Path) -> bytes:
        message = "the object was transferred despite exceeding the cap"
        raise RuntimeError(message)

    monkeypatch.setattr(pathlib.Path, "read_bytes", fail_if_read)

    with pytest.raises(media_reader.MediaObjectError, match="above the 4-byte limit"):
        media_reader.MediaObjectReader().read(str(clip), max_bytes=4)


@pytest.mark.parametrize("cap", [0, -1])
def test_media_object_reader_refuses_a_cap_no_object_could_satisfy(cap: int, tmp_path: pathlib.Path) -> None:
    """A non-positive cap is the caller's fault, never the object's.

    The classification is the whole point, not the message: as a
    ``MediaObjectError`` this reaches the caller as a terminal per-row verdict,
    and a re-run skips rows already carrying one - so one bad argument would
    mark every clip permanently unreadable while the run exited cleanly.
    """
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"\x00\x01\x02")

    with pytest.raises(ValueError, match="max_bytes must be at least 1"):
        media_reader.MediaObjectReader().read(str(clip), max_bytes=cap)


def test_media_object_reader_refuses_an_unsatisfiable_cap_before_touching_the_store(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """The cap is judged before any metadata lookup, so an impossible limit costs no round trip."""
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"\x00\x01\x02")
    resolutions: list[str] = []

    def record_resolution(uri: str) -> pathlib.Path:
        resolutions.append(uri)
        return pathlib.Path(uri)

    monkeypatch.setattr(media_reader, "local_media_path", record_resolution)

    with pytest.raises(ValueError, match="max_bytes must be at least 1"):
        media_reader.MediaObjectReader().read(str(clip), max_bytes=0)

    assert resolutions == []


def test_media_object_reader_reports_a_missing_object_as_an_object_fault(tmp_path: pathlib.Path) -> None:
    """An absent clip is deterministic for that clip alone, never a run failure."""
    with pytest.raises(media_reader.MediaObjectError, match="is missing"):
        media_reader.MediaObjectReader().read(str(tmp_path / "absent.mp4"), max_bytes=16)


def test_local_metadata_permission_error_is_systemic(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    """A local metadata permission failure is an unusable read path, not one object."""
    clip = tmp_path / "clip.mp4"
    original_stat = pathlib.Path.stat

    def deny_stat(path: pathlib.Path, *, follow_symlinks: bool = True) -> object:
        if path == clip:
            message = "metadata access denied"
            raise PermissionError(message)
        return original_stat(path, follow_symlinks=follow_symlinks)

    with monkeypatch.context() as scoped_patch:
        scoped_patch.setattr(pathlib.Path, "stat", deny_stat)
        with pytest.raises(media_reader.MediaSystemicError, match="not authorized"):
            media_reader.MediaObjectReader().read(str(clip), max_bytes=16)


def test_local_download_permission_error_is_systemic(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    """A local byte-read permission failure is an unusable read path, not one object."""
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"abc")

    def deny_read(_path: pathlib.Path) -> Never:
        message = "content access denied"
        raise PermissionError(message)

    monkeypatch.setattr(pathlib.Path, "read_bytes", deny_read)

    with pytest.raises(media_reader.MediaSystemicError, match="not authorized"):
        media_reader.MediaObjectReader().read(clip.as_uri(), max_bytes=16)


def test_media_object_reader_reports_a_zero_byte_object_as_an_object_fault(tmp_path: pathlib.Path) -> None:
    """A present but empty clip is unusable, and no retry can make it usable."""
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"")

    with pytest.raises(media_reader.MediaObjectError, match="is empty"):
        media_reader.MediaObjectReader().read(str(clip), max_bytes=16)


def test_media_object_reader_reports_an_unsupported_scheme_as_an_unreadable_object() -> None:
    """A scheme no backend claims costs the one row that carries it, never the run.

    Refusing it as systemic would trade a published error row for every other
    pending row in the table, and nothing about the URI makes the other rows
    unreadable.

    Which object-fault branch answers is pinned because it is not the obvious
    one: nothing claims ``gs://``, so the local-path boundary refuses the scheme
    with a ``ValueError`` instead of coercing it to a bogus relative path. That
    lands in the URI-shape branch, which names an unreadable location, rather
    than the missing-object branch - a terminal, row-local ``MediaObjectError``
    either way.
    """
    with pytest.raises(media_reader.MediaObjectError, match="does not name a readable object"):
        media_reader.MediaObjectReader().read("gs://bucket/clips/a.mp4", max_bytes=16)


def test_media_object_reader_reports_a_one_slash_file_uri_as_an_unreadable_object() -> None:
    """A malformed local URI remains a row-local object fault at the reader boundary."""
    with pytest.raises(media_reader.MediaObjectError, match="does not name a readable object"):
        media_reader.MediaObjectReader().read("file:/abs/path.mp4", max_bytes=16)


def test_media_object_reader_reports_a_malformed_uri_as_an_object_fault() -> None:
    """A URI the syscall refuses outright is this object's fault, not the transfer's.

    The embedded null is the case that separates the location verdict from the
    transport one: a plain unknown scheme already stats as absent, whereas this
    reaches the syscall and is rejected there, so only the location branch can
    keep it out of the retryable class.
    """
    with pytest.raises(media_reader.MediaObjectError, match="does not name a readable object"):
        media_reader.MediaObjectReader().read("gs://bucket/clip\x00.mp4", max_bytes=16)


@pytest.mark.usefixtures("instant_retry_sleep")
def test_media_object_reader_does_not_retry_a_malformed_uri(monkeypatch: pytest.MonkeyPatch) -> None:
    """A URI that cannot name an object consumes one attempt, not the whole budget."""
    resolutions: list[str] = []
    resolve = media_reader.local_media_path

    def counting_resolve(uri: str) -> pathlib.Path:
        resolutions.append(uri)
        return resolve(uri)

    monkeypatch.setattr(media_reader, "local_media_path", counting_resolve)

    with pytest.raises(media_reader.MediaObjectError, match="does not name a readable object"):
        media_reader.MediaObjectReader(max_attempts=3).read("gs://bucket/clip\x00.mp4", max_bytes=16)

    assert len(resolutions) == 1


def test_media_object_reader_reports_an_unresolvable_profile_as_a_systemic_fault(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A profile that builds no client is a configuration fault, not a per-clip one."""
    monkeypatch.setattr(media_reader, "get_storage_client", lambda *_a, **_k: None)

    with pytest.raises(media_reader.MediaSystemicError, match="no storage client available"):
        media_reader.MediaObjectReader().read("s3://bucket/clips/a.mp4", max_bytes=16)


@pytest.mark.usefixtures("instant_retry_sleep")
def test_media_object_reader_spends_its_whole_budget_on_a_transport_fault(monkeypatch: pytest.MonkeyPatch) -> None:
    """A transient transfer failure is retried up to the budget, then surfaces."""
    client = _FakeMediaClient(
        stat_result=StorageStat(size_bytes=3),
        download_result=ConnectionResetError("connection reset by peer"),
    )
    monkeypatch.setattr(media_reader, "get_storage_client", lambda *_a, **_k: client)

    with pytest.raises(media_reader.MediaTransportError, match="media read failed transiently"):
        media_reader.MediaObjectReader(max_attempts=3).read("s3://bucket/clips/a.mp4", max_bytes=16)

    assert client.download_calls == 3


@pytest.mark.usefixtures("instant_retry_sleep")
def test_media_object_reader_does_not_retry_a_credential_fault(monkeypatch: pytest.MonkeyPatch) -> None:
    """Rejected credentials fail on the first attempt: a retry sends an identically doomed request."""
    client = _FakeMediaClient(
        stat_result=ClientError(
            {"Error": {"Code": "AccessDenied"}, "ResponseMetadata": {"HTTPStatusCode": 403}},
            "HeadObject",
        ),
        download_result=b"unused",
    )
    monkeypatch.setattr(media_reader, "get_storage_client", lambda *_a, **_k: client)

    with pytest.raises(media_reader.MediaSystemicError, match="rejected with AccessDenied"):
        media_reader.MediaObjectReader(max_attempts=3).read("s3://bucket/clips/a.mp4", max_bytes=16)

    assert client.stat_calls == 1


def test_credential_error_precedes_a_chained_missing_object_code(
    install_stat_error_client: InstallStatErrorClient,
) -> None:
    """Keep a credential failure systemic when its cause looks object-local."""
    error = ClientAuthenticationError(message="credential rejected")
    error.__cause__ = _s3_client_error("NoSuchKey")
    client = install_stat_error_client(error)

    with pytest.raises(media_reader.MediaSystemicError, match="not authorized"):
        media_reader.MediaObjectReader().read("s3://bucket/clips/a.mp4", max_bytes=16)

    assert client.stat_calls == 1


@pytest.mark.parametrize(
    ("credential_error", "uri"),
    [
        (PermissionError("permission rejected"), "s3://bucket/clips/a.mp4"),
        (PermissionError("permission rejected"), "az://container/clips/a.mp4"),
        (NoCredentialsError(), "s3://bucket/clips/a.mp4"),
        (ClientAuthenticationError(message="credential rejected"), "az://container/clips/a.mp4"),
    ],
    ids=["s3-permission", "azure-permission", "s3-credentials", "azure-authentication"],
)
def test_chained_credential_error_precedes_a_missing_object_wrapper(
    credential_error: Exception,
    uri: str,
    install_stat_error_client: InstallStatErrorClient,
) -> None:
    """Keep a wrapped credential failure systemic instead of persisting one row verdict."""
    client = install_stat_error_client(_one_level_chained_error(credential_error))

    with pytest.raises(media_reader.MediaSystemicError, match="not authorized"):
        media_reader.MediaObjectReader().read(uri, max_bytes=16)

    assert client.stat_calls == 1


@pytest.mark.parametrize("code", ["404", "NoSuchKey", "NotFound"])
@pytest.mark.parametrize(
    "wrap_error",
    [_direct_error, _one_level_chained_error],
    ids=["direct", "one-level-chained"],
)
def test_s3_missing_object_code_is_object_local(
    code: str,
    wrap_error: WrapProviderError,
    install_stat_error_client: InstallStatErrorClient,
) -> None:
    """Refuse only the row carrying an S3 missing-key code."""
    client = install_stat_error_client(wrap_error(_s3_client_error(code)))

    with pytest.raises(media_reader.MediaObjectError, match=rf"is missing.*S3 {code}"):
        media_reader.MediaObjectReader().read("s3://bucket/clips/a.mp4", max_bytes=16)

    assert client.stat_calls == 1


@pytest.mark.parametrize(
    "wrap_error",
    [_direct_error, _one_level_chained_error],
    ids=["direct", "one-level-chained"],
)
def test_s3_missing_bucket_code_is_systemic(
    wrap_error: WrapProviderError,
    install_stat_error_client: InstallStatErrorClient,
) -> None:
    """Fail the read path for a missing S3 bucket, including through a wrapper."""
    client = install_stat_error_client(wrap_error(_s3_client_error("NoSuchBucket")))

    with pytest.raises(media_reader.MediaSystemicError, match="NoSuchBucket"):
        media_reader.MediaObjectReader().read("s3://bucket/clips/a.mp4", max_bytes=16)

    assert client.stat_calls == 1


@pytest.mark.parametrize(
    "wrap_error",
    [_direct_error, _one_level_chained_error],
    ids=["direct", "one-level-chained"],
)
def test_azure_missing_blob_code_is_object_local(
    wrap_error: WrapProviderError,
    install_stat_error_client: InstallStatErrorClient,
) -> None:
    """Refuse only the row carrying a missing Azure blob URI."""
    client = install_stat_error_client(wrap_error(_azure_resource_not_found_error("BlobNotFound")))

    with pytest.raises(media_reader.MediaObjectError, match=r"is missing.*Azure BlobNotFound"):
        media_reader.MediaObjectReader().read("az://container/clips/a.mp4", max_bytes=16)

    assert client.stat_calls == 1


@pytest.mark.parametrize(
    "wrap_error",
    [_direct_error, _one_level_chained_error],
    ids=["direct", "one-level-chained"],
)
def test_azure_missing_container_code_is_systemic(
    wrap_error: WrapProviderError,
    install_stat_error_client: InstallStatErrorClient,
) -> None:
    """Fail the read path for a missing Azure container, including through a wrapper."""
    client = install_stat_error_client(wrap_error(_azure_resource_not_found_error("ContainerNotFound")))

    with pytest.raises(media_reader.MediaSystemicError, match="ContainerNotFound"):
        media_reader.MediaObjectReader().read("az://container/clips/a.mp4", max_bytes=16)

    assert client.stat_calls == 1


@pytest.mark.parametrize(
    "wrap_error",
    [_direct_error, _one_level_chained_error],
    ids=["direct", "one-level-chained"],
)
def test_azure_resource_error_without_code_keeps_object_local_fallback(
    wrap_error: WrapProviderError,
    install_stat_error_client: InstallStatErrorClient,
) -> None:
    """Keep the established row-local fallback for an Azure resource error without a code."""
    client = install_stat_error_client(wrap_error(_azure_resource_not_found_error(None)))

    with pytest.raises(media_reader.MediaObjectError, match=r"is missing.*without an error code"):
        media_reader.MediaObjectReader().read("az://container/clips/a.mp4", max_bytes=16)

    assert client.stat_calls == 1


@pytest.mark.usefixtures("instant_retry_sleep")
@pytest.mark.parametrize(
    "wrap_error",
    [_direct_error, _one_level_chained_error],
    ids=["direct", "one-level-chained"],
)
def test_unknown_s3_provider_code_is_retried_as_transport(
    wrap_error: WrapProviderError,
    install_stat_error_client: InstallStatErrorClient,
) -> None:
    """Retry an unknown S3 provider code instead of persisting a missing-object verdict."""
    client = install_stat_error_client(wrap_error(_s3_client_error("UnexpectedProviderCode")))

    with pytest.raises(media_reader.MediaTransportError, match="failed transiently"):
        media_reader.MediaObjectReader(max_attempts=3).read("s3://bucket/clips/a.mp4", max_bytes=16)

    assert client.stat_calls == 3


@pytest.mark.usefixtures("instant_retry_sleep")
@pytest.mark.parametrize(
    "wrap_error",
    [_direct_error, _one_level_chained_error],
    ids=["direct", "one-level-chained"],
)
def test_unknown_azure_provider_code_is_retried_as_transport(
    wrap_error: WrapProviderError,
    install_stat_error_client: InstallStatErrorClient,
) -> None:
    """Retry an unknown Azure provider code instead of persisting a missing-object verdict."""
    client = install_stat_error_client(wrap_error(_azure_resource_not_found_error("UnexpectedProviderCode")))

    with pytest.raises(media_reader.MediaTransportError, match="failed transiently"):
        media_reader.MediaObjectReader(max_attempts=3).read("az://container/clips/a.mp4", max_bytes=16)

    assert client.stat_calls == 3


@pytest.mark.usefixtures("instant_retry_sleep")
def test_media_object_reader_does_not_retry_an_object_fault(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing object consumes one attempt, not the whole budget."""
    client = _FakeMediaClient(
        stat_result=StorageStat(size_bytes=3),
        download_result=FileNotFoundError("s3://bucket/clips/a.mp4"),
    )
    monkeypatch.setattr(media_reader, "get_storage_client", lambda *_a, **_k: client)

    with pytest.raises(media_reader.MediaObjectError, match="is missing"):
        media_reader.MediaObjectReader(max_attempts=3).read("s3://bucket/clips/a.mp4", max_bytes=16)

    assert client.download_calls == 1


def test_media_object_reader_caps_delivered_bytes_when_the_store_reports_no_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A backend that omits the size still cannot smuggle an oversized object through.

    The metadata check is the cheap path, not the guarantee: the size is optional
    on the stat result, and the object can also be replaced between the lookup and
    the transfer.
    """
    client = _FakeMediaClient(stat_result=StorageStat(), download_result=b"0123456789")
    monkeypatch.setattr(media_reader, "get_storage_client", lambda *_a, **_k: client)

    with pytest.raises(media_reader.MediaObjectError, match="above the 4-byte limit"):
        media_reader.MediaObjectReader().read("s3://bucket/clips/a.mp4", max_bytes=4)


def test_media_object_reader_builds_one_client_per_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """The client is resolved once and reused, so a batch pays for one handshake."""
    client = _FakeMediaClient(stat_result=StorageStat(size_bytes=3), download_result=b"abc")
    resolutions: list[str] = []

    def resolve(uri: str, **_kwargs: object) -> _FakeMediaClient:
        resolutions.append(uri)
        return client

    monkeypatch.setattr(media_reader, "get_storage_client", resolve)
    reader = media_reader.MediaObjectReader()

    reader.read("s3://bucket/clips/a.mp4", max_bytes=16)
    reader.read("s3://bucket/clips/b.mp4", max_bytes=16)

    assert len(resolutions) == 1


def test_media_object_reader_survives_pickling_after_it_has_read(monkeypatch: pytest.MonkeyPatch) -> None:
    """A reader that already holds a resolved client still pickles.

    Ray pickles a reader to reach a worker, and nothing orders that hop before
    the reader's first read. Carrying the cache would make the hop fail on
    whichever client the reader happened to resolve first.
    """
    client = _FakeMediaClient(stat_result=StorageStat(size_bytes=3), download_result=b"abc")
    monkeypatch.setattr(media_reader, "get_storage_client", lambda *_a, **_k: client)
    reader = media_reader.MediaObjectReader(storage_profile="prod", max_attempts=3)
    reader.read("s3://bucket/clips/a.mp4", max_bytes=16)

    # The cache is excluded from equality, so this compares the configuration
    # the copy must arrive with.
    assert pickle.loads(pickle.dumps(reader)) == reader  # noqa: S301


def test_media_object_reader_rebuilds_its_client_after_unpickling(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unpickled reader resolves a client again and reads.

    Excluding the cache is only safe if it regenerates, so the copy is required
    to read rather than merely to unpickle.
    """
    client = _FakeMediaClient(stat_result=StorageStat(size_bytes=3), download_result=b"abc")
    resolutions: list[str] = []

    def resolve(uri: str, **_kwargs: object) -> _FakeMediaClient:
        resolutions.append(uri)
        return client

    monkeypatch.setattr(media_reader, "get_storage_client", resolve)
    reader = media_reader.MediaObjectReader()
    reader.read("s3://bucket/clips/a.mp4", max_bytes=16)

    copy = pickle.loads(pickle.dumps(reader))  # noqa: S301

    assert copy.read("s3://bucket/clips/b.mp4", max_bytes=16) == b"abc"
    assert len(resolutions) == 2
