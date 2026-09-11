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
"""Tests for S3 client listing and download semantics."""

import io
import subprocess
import sys
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from botocore.exceptions import ClientError

from cosmos_curator.core.utils.storage import s3_client
from cosmos_curator.core.utils.storage.azure_client import AzurePrefix
from cosmos_curator.core.utils.storage.s3_client import S3Client, S3ClientConfig, S3Prefix
from cosmos_curator.core.utils.storage.storage_client import StorageStat, join_under


class _FakePaginator:
    """Paginator stand-in that records how many pages were actually consumed.

    Pages are yielded lazily so a test can prove an early-exiting listing stopped
    fetching, which a list of pages could not show.
    """

    def __init__(self, pages: list[dict[str, Any]]) -> None:
        self._pages = pages
        self.last_paginate_kwargs: dict[str, object] | None = None
        self.pages_yielded = 0

    def paginate(self, **kwargs: object) -> Iterator[dict[str, Any]]:
        self.last_paginate_kwargs = kwargs
        return self._iter_pages()

    def _iter_pages(self) -> Iterator[dict[str, Any]]:
        for page in self._pages:
            self.pages_yielded += 1
            yield page


class _FakeS3:
    def __init__(self, pages: list[dict[str, Any]]) -> None:
        self.paginator = _FakePaginator(pages)
        self.uploads: list[tuple[str, str, str]] = []

    def get_paginator(self, name: str) -> _FakePaginator:
        assert name == "list_objects_v2"
        return self.paginator

    def upload_file(self, local_path: str, bucket: str, prefix: str, **_kwargs: object) -> None:
        self.uploads.append((local_path, bucket, prefix))


def _client_error(code: str) -> ClientError:
    """Build the ClientError botocore raises for a HeadObject the store refused."""
    return ClientError({"Error": {"Code": code, "Message": "refused"}}, "HeadObject")


class _FakeHeadS3:
    """HeadObject stand-in that serves one canned answer and counts its calls.

    The call count is what proves ``stat`` issues a single request and is not wrapped
    in a retry loop.
    """

    def __init__(self, response: dict[str, Any] | None = None, error: ClientError | None = None) -> None:
        self._response = response
        self._error = error
        self.head_object_calls: list[dict[str, object]] = []

    def head_object(self, **kwargs: object) -> dict[str, Any]:
        self.head_object_calls.append(kwargs)
        if self._error is not None:
            raise self._error
        assert self._response is not None
        return self._response


def _head_client(
    response: dict[str, Any] | None = None,
    error: ClientError | None = None,
) -> tuple[S3Client, _FakeHeadS3]:
    """Build a client whose only wired-up S3 call is HeadObject."""
    fake = _FakeHeadS3(response, error)
    client = object.__new__(S3Client)
    client.s3 = fake
    return client, fake


class _TraceLogger:
    def __init__(self) -> None:
        self.messages: list[str] = []
        self.warnings: list[str] = []

    def trace(self, message: str) -> None:
        self.messages.append(message)

    def warning(self, message: str) -> None:
        self.warnings.append(message)


class _FakeStreamingBody:
    """Botocore StreamingBody stand-in that records its read and close calls.

    ``raise_on_read`` makes ``read`` fail the way a connection dropped mid-stream
    would, so a test can assert the stream is still closed on that path.
    """

    def __init__(self, payload: bytes, *, raise_on_read: bool = False) -> None:
        self._payload = payload
        self._raise_on_read = raise_on_read
        self.read_count = 0
        self.closed = False

    def read(self) -> bytes:
        self.read_count += 1
        if self._raise_on_read:
            msg = "connection reset mid-stream"
            raise OSError(msg)
        return self._payload

    def close(self) -> None:
        self.closed = True


class _FakeDownloadS3:
    """Single-object S3 stand-in that records which download shape was used.

    Passing ``report_content_length=False`` omits ``ContentLength`` from the
    ``get_object`` response, leaving the object unmeasured. Passing
    ``raise_on_read=True`` makes the served body fail when it is read.
    """

    def __init__(self, payload: bytes, *, report_content_length: bool = True, raise_on_read: bool = False) -> None:
        self._payload = payload
        self._report_content_length = report_content_length
        self._raise_on_read = raise_on_read
        self.bodies: list[_FakeStreamingBody] = []
        self.get_object_calls: list[dict[str, object]] = []
        self.download_fileobj_calls: list[tuple[str, str]] = []

    def get_object(self, **kwargs: object) -> dict[str, Any]:
        self.get_object_calls.append(kwargs)
        body = _FakeStreamingBody(self._payload, raise_on_read=self._raise_on_read)
        self.bodies.append(body)
        response: dict[str, Any] = {"Body": body}
        if self._report_content_length:
            response["ContentLength"] = len(self._payload)
        return response

    def download_fileobj(self, bucket: str, key: str, fileobj: io.BytesIO, **_kwargs: object) -> None:
        self.download_fileobj_calls.append((bucket, key))
        fileobj.write(self._payload)


def test_list_recursive_respects_limit_within_large_page() -> None:
    """Trim results to exact limit when a single page contains more entries than requested."""
    pages = [
        {
            "Contents": [
                {"Key": "root/a.mp4"},
                {"Key": "root/b.mp4"},
                {"Key": "root/c.mp4"},
            ]
        }
    ]
    client = object.__new__(S3Client)
    client.s3 = _FakeS3(pages)

    results = client.list_recursive(S3Prefix("s3://bucket/root"), limit=2)
    assert len(results) == 2
    assert [item["Key"] for item in results] == ["root/a.mp4", "root/b.mp4"]


def test_list_recursive_respects_limit_across_pages() -> None:
    """Trim to exact limit when overflow occurs after reading a subsequent page."""
    pages = [
        {
            "Contents": [
                {"Key": "root/a.mp4"},
            ]
        },
        {
            "Contents": [
                {"Key": "root/b.mp4"},
                {"Key": "root/c.mp4"},
            ]
        },
    ]
    client = object.__new__(S3Client)
    client.s3 = _FakeS3(pages)

    results = client.list_recursive(S3Prefix("s3://bucket/root"), limit=2)
    assert len(results) == 2
    assert [item["Key"] for item in results] == ["root/a.mp4", "root/b.mp4"]


def test_list_recursive_without_limit_returns_all_pages() -> None:
    """Return all objects when no limit is specified."""
    pages = [
        {"Contents": [{"Key": "root/a.mp4"}]},
        {"Contents": [{"Key": "root/b.mp4"}]},
    ]
    client = object.__new__(S3Client)
    client.s3 = _FakeS3(pages)

    results = client.list_recursive(S3Prefix("s3://bucket/root"), limit=0)
    assert len(results) == 2
    assert [item["Key"] for item in results] == ["root/a.mp4", "root/b.mp4"]


def test_stat_returns_the_metadata_carried_by_one_head_object() -> None:
    """The HEAD ``object_exists`` used to discard now answers the size question too."""
    last_modified = datetime(2026, 8, 1, 12, 0, tzinfo=UTC)
    client, fake = _head_client({"ContentLength": 1234, "LastModified": last_modified, "ETag": '"abc123"'})

    stat = client.stat(S3Prefix("s3://bucket/root/a.mp4"))

    assert stat == StorageStat(size_bytes=1234, last_modified=last_modified, etag="abc123")
    assert fake.head_object_calls == [{"Bucket": "bucket", "Key": "root/a.mp4"}]


def test_stat_raises_file_not_found_for_a_missing_object() -> None:
    """Follow ``os.stat``: absence is an error, not an empty answer."""
    client, _ = _head_client(error=_client_error("404"))

    with pytest.raises(FileNotFoundError, match=r"s3://bucket/root/a\.mp4"):
        client.stat(S3Prefix("s3://bucket/root/a.mp4"))


def test_a_missing_object_costs_exactly_one_request() -> None:
    """``stat`` must not be wrapped in ``do_with_retries``.

    The retry wrappers in this package back off for up to 256 seconds. A progress
    display that stats many objects would stall for minutes on the first one that is
    legitimately absent, so absence has to answer on the first request.
    """
    client, fake = _head_client(error=_client_error("404"))

    assert client.object_exists(S3Prefix("s3://bucket/root/a.mp4")) is False
    assert len(fake.head_object_calls) == 1


def test_object_exists_is_true_for_a_present_object() -> None:
    """A successful stat is reported as existence."""
    client, _ = _head_client({"ContentLength": 7})

    assert client.object_exists(S3Prefix("s3://bucket/root/a.mp4")) is True


def test_a_forbidden_head_still_propagates_out_of_object_exists() -> None:
    """A 403 must never be reported as "does not exist".

    Without ``s3:ListBucket`` AWS answers 403 for an object that is merely absent, so
    a wrapper that caught broadly would turn every unreadable bucket into silently
    missing data instead of a credentials error.
    """
    client, _ = _head_client(error=_client_error("403"))

    with pytest.raises(ClientError, match="403"):
        client.object_exists(S3Prefix("s3://bucket/root/a.mp4"))


def test_a_store_specific_missing_key_code_still_propagates() -> None:
    """Only the literal ``404`` is translated, which callers rely on.

    A HEAD carries no body for botocore to read a richer code from, so a store
    replying ``NoSuchKey`` is doing something AWS does not; downstream code
    distinguishes the two and would lose that if this mapped every 404-ish code.
    """
    client, _ = _head_client(error=_client_error("NoSuchKey"))

    with pytest.raises(ClientError, match="NoSuchKey"):
        client.object_exists(S3Prefix("s3://bucket/root/a.mp4"))


def test_suffix_filtered_listing_counts_matches_not_listed_objects() -> None:
    """The limit means N videos, which is what ``list_recursive`` cannot promise.

    ``list_recursive(..., limit=2)`` over these keys returns ``a.mp4`` and ``a.json``
    -- one video for a caller that asked for two.
    """
    pages = [{"Contents": [{"Key": "root/a.mp4"}, {"Key": "root/a.json"}, {"Key": "root/b.mp4"}]}]
    client = object.__new__(S3Client)
    client.s3 = _FakeS3(pages)

    results = client.list_recursive_with_suffixes(S3Prefix("s3://bucket/root"), (".mp4",), limit=2)

    assert [str(item) for item in results] == ["s3://bucket/root/a.mp4", "s3://bucket/root/b.mp4"]


def test_suffix_filtered_listing_stops_paging_once_the_limit_is_met() -> None:
    """The cap lives inside the pagination loop, so later pages are never fetched."""
    pages = [{"Contents": [{"Key": "root/a.mp4"}]}, {"Contents": [{"Key": "root/b.mp4"}]}]
    fake = _FakeS3(pages)
    client = object.__new__(S3Client)
    client.s3 = fake

    results = client.list_recursive_with_suffixes(S3Prefix("s3://bucket/root"), (".mp4",), limit=1)

    assert [str(item) for item in results] == ["s3://bucket/root/a.mp4"]
    assert fake.paginator.pages_yielded == 1


def test_a_listed_key_containing_a_glob_character_does_not_abort_the_listing() -> None:
    """An object named ``a*foo.mp4`` is real, and one of them must not lose the rest.

    Every result here is wrapped in an ``S3Prefix``, so a constructor that refused
    glob characters would raise from inside the pagination loop and drop the whole
    enumeration rather than the single key. The mistyped-glob rejection that used to
    live in the constructor now applies only to configured locations.
    """
    pages = [{"Contents": [{"Key": "root/a*foo.mp4"}, {"Key": "root/b?bar.mp4"}, {"Key": "root/c.mp4"}]}]
    client = object.__new__(S3Client)
    client.s3 = _FakeS3(pages)

    results = client.list_recursive_with_suffixes(S3Prefix("s3://bucket/root"), (".mp4",))

    assert [str(item) for item in results] == [
        "s3://bucket/root/a*foo.mp4",
        "s3://bucket/root/b?bar.mp4",
        "s3://bucket/root/c.mp4",
    ]


def test_suffix_matching_ignores_case_on_both_sides() -> None:
    """An uppercase extension in the store and in the filter both still match."""
    pages = [{"Contents": [{"Key": "root/a.MP4"}, {"Key": "root/b.mkv"}, {"Key": "root/c.txt"}]}]
    client = object.__new__(S3Client)
    client.s3 = _FakeS3(pages)

    results = client.list_recursive_with_suffixes(S3Prefix("s3://bucket/root"), (".mp4", ".MKV"))

    assert [str(item) for item in results] == ["s3://bucket/root/a.MP4", "s3://bucket/root/b.mkv"]


def test_a_zero_limit_lists_every_match_across_every_page() -> None:
    """Zero means unlimited here, as it already does for ``list_recursive``."""
    pages = [{"Contents": [{"Key": "root/a.mp4"}]}, {"Contents": [{"Key": "root/b.mp4"}]}]
    fake = _FakeS3(pages)
    client = object.__new__(S3Client)
    client.s3 = fake

    results = client.list_recursive_with_suffixes(S3Prefix("s3://bucket/root"), (".mp4",), limit=0)

    assert [str(item) for item in results] == ["s3://bucket/root/a.mp4", "s3://bucket/root/b.mp4"]
    assert fake.paginator.pages_yielded == len(pages)


def test_an_empty_suffix_filter_is_rejected_rather_than_matching_nothing() -> None:
    """``str.endswith(())`` is False, so an empty filter would quietly return nothing."""
    client = object.__new__(S3Client)
    client.s3 = _FakeS3([{"Contents": [{"Key": "root/a.mp4"}]}])

    with pytest.raises(ValueError, match="suffixes must not be empty"):
        client.list_recursive_with_suffixes(S3Prefix("s3://bucket/root"), ())


def test_an_empty_suffix_among_several_is_rejected_rather_than_matching_everything() -> None:
    """``str.endswith("")`` is True for every key, so one stray element voids the filter.

    A trailing separator in a configured or command-line suffix list produces exactly
    this, and the result is not an error but a listing of the wrong objects -- capped
    at ``limit``, so it looks plausible.
    """
    client = object.__new__(S3Client)
    client.s3 = _FakeS3([{"Contents": [{"Key": "root/a.mp4"}, {"Key": "root/a.json"}]}])

    with pytest.raises(ValueError, match="suffixes must not contain an empty string"):
        client.list_recursive_with_suffixes(S3Prefix("s3://bucket/root"), (".mp4", ""))


def test_a_prefix_for_another_backend_is_refused_rather_than_asserted() -> None:
    """The wrong prefix type must fail the same way with and without ``python -O``.

    ``get_storage_client`` hands back whichever client the path implied, so a caller
    holding one of those and a prefix built from a different URI is a real mix-up. The
    check was an ``assert``, which optimization strips, leaving the mismatch to surface
    further in as a missing-attribute error against a half-built request.
    """
    client, _ = _head_client()

    with pytest.raises(TypeError, match="S3Client requires an S3Prefix, got AzurePrefix"):
        client.stat(AzurePrefix("az://container/blob"))


def test_upload_file_emits_one_trace_instead_of_per_object_info(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Bulk uploads stay quiet at normal log levels while retaining opt-in detail."""
    fake_s3 = _FakeS3([])
    trace_logger = _TraceLogger()
    client = object.__new__(S3Client)
    client.s3 = fake_s3
    client.can_overwrite = True
    monkeypatch.setattr(s3_client, "logger", trace_logger)
    local_path = str(tmp_path / "clip.mp4")

    client.upload_file(local_path, S3Prefix("s3://bucket/clips/clip.mp4"))

    assert fake_s3.uploads == [(local_path, "bucket", "clips/clip.mp4")]
    assert trace_logger.messages == [f"Uploaded {local_path} to s3://bucket/clips/clip.mp4"]


def test_client_uses_configured_region(monkeypatch: pytest.MonkeyPatch) -> None:
    """A profile region must reach boto3 and win over the ambient environment."""
    monkeypatch.setenv("AWS_REGION", "eu-central-1")

    client = S3Client(
        S3ClientConfig(
            aws_access_key_id="test-key-id",
            aws_secret_access_key="test-secret",  # noqa: S106
            region="us-west-2",
        )
    )

    assert client.session.region_name == "us-west-2"
    assert client.s3.meta.region_name == "us-west-2"


def test_client_without_configured_region_defers_to_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """No configured region leaves boto3's own resolution chain untouched."""
    monkeypatch.delenv("AWS_REGION", raising=False)
    monkeypatch.setenv("AWS_DEFAULT_REGION", "eu-central-1")

    client = S3Client(
        S3ClientConfig(
            aws_access_key_id="test-key-id",
            aws_secret_access_key="test-secret",  # noqa: S106
        )
    )

    assert client.s3.meta.region_name == "eu-central-1"


def _write_aws_credentials(directory: Path) -> Path:
    """Write a one-profile AWS shared credentials file and return its path."""
    credentials = directory / "credentials"
    credentials.write_text(
        "[curator-test]\naws_access_key_id = profile-key\naws_secret_access_key = profile-secret\nregion = us-west-1\n"
    )
    return credentials


@pytest.fixture
def aws_profile_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point boto3 at a throwaway credentials file, with rival env credentials set.

    The environment credentials are deliberate: botocore drops its env provider once a
    profile is named explicitly, so their presence is what proves the profile was
    honoured rather than merely not contradicted. The region variables are cleared for
    the opposite reason -- they outrank a profile's own ``region``, so an ambient one
    on the CI runner would mask it.
    """
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(_write_aws_credentials(tmp_path)))
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "env-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "env-secret")
    monkeypatch.delenv("AWS_PROFILE", raising=False)
    monkeypatch.delenv("AWS_REGION", raising=False)
    monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)


@pytest.mark.usefixtures("aws_profile_env")
def test_a_configured_profile_name_builds_the_client_from_that_profile() -> None:
    """A profile name reaches boto3, and the endpoint override reaches the client.

    The endpoint assertion is the regression: the no-explicit-credentials branch used
    to drop ``endpoint_url`` entirely, which silently sent every request for an
    S3-compatible store to AWS instead.
    """
    client = S3Client(S3ClientConfig(profile_name="curator-test", endpoint_url="https://s3.example.invalid"))

    assert client.session.profile_name == "curator-test"
    assert client.session.get_credentials().access_key == "profile-key"
    assert client.s3.meta.endpoint_url == "https://s3.example.invalid"
    assert client.s3.meta.region_name == "us-west-1"


@pytest.mark.usefixtures("aws_profile_env")
def test_profile_credentials_stay_refreshable_instead_of_being_frozen() -> None:
    """The client must sign with the session's live credentials, not a snapshot.

    ``method`` is the discriminator: resolving the chain here and handing the result
    back as explicit keys would report ``explicit`` instead. Rotating the session's
    own credential object and re-signing then shows the signer reads that object per
    request rather than a construction-time copy, which is what lets an SSO or
    instance credential renew mid-run instead of expiring the job.

    A presigned URL is the cheapest signature to inspect: it carries the access key
    id in its ``X-Amz-Credential`` parameter and is computed without any network I/O.
    """
    client = S3Client(S3ClientConfig(profile_name="curator-test"))
    credentials = client.session.get_credentials()

    assert credentials.method == "shared-credentials-file"

    params = {"Bucket": "some-bucket", "Key": "some-key"}
    signed_before_rotation = client.s3.generate_presigned_url("get_object", Params=params)
    assert "profile-key" in signed_before_rotation

    credentials.access_key = "rotated-key"
    signed_after_rotation = client.s3.generate_presigned_url("get_object", Params=params)

    assert "rotated-key" in signed_after_rotation
    assert "profile-key" not in signed_after_rotation


def test_explicit_credentials_win_over_a_profile_name(monkeypatch: pytest.MonkeyPatch) -> None:
    """The explicit-credentials branch is untouched, region and endpoint included.

    ``profile_name`` names a profile that does not exist, so a branch that consulted
    it would raise ``ProfileNotFound`` rather than pass.
    """
    monkeypatch.setenv("AWS_REGION", "eu-central-1")

    client = S3Client(
        S3ClientConfig(
            aws_access_key_id="test-key-id",
            aws_secret_access_key="test-secret",  # noqa: S106
            region="us-west-2",
            profile_name="no-such-profile",
            endpoint_url="https://s3.example.invalid",
        )
    )

    assert client.session.get_credentials().access_key == "test-key-id"
    assert client.s3.meta.region_name == "us-west-2"
    assert client.s3.meta.endpoint_url == "https://s3.example.invalid"


def test_a_config_with_neither_keys_nor_profile_uses_the_default_chain_quietly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No credentials and no profile is a supported path, not a "should not happen".

    It is how a caller asks for boto3's own resolution, so it must pick up the
    environment and must not log a warning while doing it.
    """
    monkeypatch.delenv("AWS_PROFILE", raising=False)
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "env-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "env-secret")
    recorder = _TraceLogger()
    monkeypatch.setattr(s3_client, "logger", recorder)

    client = S3Client(S3ClientConfig())

    assert client.session.get_credentials().access_key == "env-key"
    assert recorder.warnings == []


_ENDPOINT_PRECEDENCE = [
    pytest.param(
        ("https://explicit.invalid", "https://s3-env.invalid", "https://env.invalid", "https://profile.invalid"),
        "https://explicit.invalid",
        id="an-explicit-argument-outranks-everything",
    ),
    pytest.param(
        (None, "https://s3-env.invalid", "https://env.invalid", "https://profile.invalid"),
        "https://s3-env.invalid",
        id="the-s3-specific-variable-outranks-the-generic-one",
    ),
    pytest.param(
        (None, None, "https://env.invalid", "https://profile.invalid"),
        "https://env.invalid",
        id="the-generic-variable-outranks-a-profile-file",
    ),
    pytest.param(
        (None, None, None, "https://profile.invalid"),
        "https://profile.invalid",
        id="a-profile-file-endpoint-wins-over-nothing",
    ),
    pytest.param((None, None, None, None), None, id="nothing-configured-defers-to-boto3"),
]


@pytest.mark.parametrize(("candidates", "expected"), _ENDPOINT_PRECEDENCE)
def test_endpoint_resolution_follows_the_documented_precedence(
    monkeypatch: pytest.MonkeyPatch,
    candidates: tuple[str | None, str | None, str | None, str | None],
    expected: str | None,
) -> None:
    """Reproduce the order the data-integrity CLIs already depend on."""
    explicit, env_s3, env_generic, profile_endpoint_url = candidates
    for name, value in (("AWS_ENDPOINT_URL_S3", env_s3), ("AWS_ENDPOINT_URL", env_generic)):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)

    resolved = s3_client.resolve_s3_endpoint_url(explicit, profile_endpoint_url=profile_endpoint_url)

    assert resolved == expected


def test_endpoint_resolution_is_not_wired_into_the_profile_file_factory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The Curator profile file's endpoint is returned verbatim, environment or not.

    ``COSMOS_S3_PROFILE_PATH`` is populated in every deployment environment and
    ``get_s3_client_config`` returns this helper's result unchanged, so letting an
    ``AWS_ENDPOINT_URL`` exported for some unrelated tool outrank it would redirect
    existing pipelines. Callers that do want the full chain call
    ``resolve_s3_endpoint_url`` themselves.
    """
    profile_path = tmp_path / "s3_creds"
    profile_path.write_text(
        "[default]\n"
        "aws_access_key_id = file-key\n"
        "aws_secret_access_key = file-secret\n"
        "endpoint_url = https://profile.invalid\n"
    )
    monkeypatch.setenv("AWS_ENDPOINT_URL_S3", "https://s3-env.invalid")

    config = s3_client._make_s3_client_config(profile_path)

    assert config.endpoint_url == "https://profile.invalid"
    assert config.profile_name is None


DownloadClientFactory = Callable[..., tuple[S3Client, _FakeDownloadS3]]


@pytest.fixture
def make_download_client() -> DownloadClientFactory:
    """Return a closure building a client wired to a fresh fake S3 for one object."""

    def _factory(
        payload: bytes, *, report_content_length: bool = True, raise_on_read: bool = False
    ) -> tuple[S3Client, _FakeDownloadS3]:
        fake = _FakeDownloadS3(payload, report_content_length=report_content_length, raise_on_read=raise_on_read)
        client = object.__new__(S3Client)
        client.s3 = fake
        return client, fake

    return _factory


def test_small_object_is_served_by_a_single_get_object(make_download_client: DownloadClientFactory) -> None:
    """Serve an object below the multipart threshold from one get_object response."""
    payload = b"below-threshold-payload"
    client, fake = make_download_client(payload)

    data = client.download_object_as_bytes(S3Prefix("s3://bucket/root/a.mp4"), chunk_size_bytes=len(payload) + 1)

    assert data == payload
    assert fake.get_object_calls == [{"Bucket": "bucket", "Key": "root/a.mp4"}]
    assert fake.download_fileobj_calls == []


def test_large_object_is_served_by_the_managed_transfer(make_download_client: DownloadClientFactory) -> None:
    """Serve an object above the multipart threshold through the managed transfer."""
    payload = b"above-threshold-payload"
    client, fake = make_download_client(payload)

    data = client.download_object_as_bytes(S3Prefix("s3://bucket/root/a.mp4"), chunk_size_bytes=len(payload) - 1)

    assert data == payload
    assert fake.download_fileobj_calls == [("bucket", "root/a.mp4")]


def test_object_sized_exactly_at_the_threshold_is_served_by_the_managed_transfer(
    make_download_client: DownloadClientFactory,
) -> None:
    """Treat the threshold as exclusive: an object of exactly that size takes the managed transfer."""
    payload = b"exactly-at-threshold"
    client, fake = make_download_client(payload)

    data = client.download_object_as_bytes(S3Prefix("s3://bucket/root/a.mp4"), chunk_size_bytes=len(payload))

    assert data == payload
    assert fake.download_fileobj_calls == [("bucket", "root/a.mp4")]


def test_unread_body_is_closed_when_the_managed_transfer_is_used(
    make_download_client: DownloadClientFactory,
) -> None:
    """Close the get_object probe's stream instead of leaking it when the transfer takes over."""
    payload = b"above-threshold-payload"
    client, fake = make_download_client(payload)

    client.download_object_as_bytes(S3Prefix("s3://bucket/root/a.mp4"), chunk_size_bytes=len(payload) - 1)

    assert len(fake.bodies) == 1
    assert fake.bodies[0].closed
    assert fake.bodies[0].read_count == 0


def test_body_is_closed_when_the_small_path_read_fails(
    make_download_client: DownloadClientFactory,
) -> None:
    """Close the stream even when reading it raises, so a failed small read leaks no connection.

    A caller that retries re-enters this method on every attempt, so a stream left
    open on the failing path would leak once per attempt rather than once per run.
    """
    payload = b"below-threshold-payload"
    client, fake = make_download_client(payload, raise_on_read=True)

    with pytest.raises(OSError, match="connection reset mid-stream"):
        client.download_object_as_bytes(S3Prefix("s3://bucket/root/a.mp4"), chunk_size_bytes=len(payload) + 1)

    assert fake.bodies[0].closed


def test_response_without_content_length_falls_back_to_the_managed_transfer(
    make_download_client: DownloadClientFactory,
) -> None:
    """Hand an unmeasured object to the managed transfer rather than reading its body blind."""
    payload = b"unmeasured-payload"
    client, fake = make_download_client(payload, report_content_length=False)

    data = client.download_object_as_bytes(S3Prefix("s3://bucket/root/a.mp4"), chunk_size_bytes=len(payload) + 1)

    assert data == payload
    assert fake.download_fileobj_calls == [("bucket", "root/a.mp4")]
    assert fake.bodies[0].read_count == 0


def _sync_client(objects: list[dict[str, Any]]) -> tuple[S3Client, list[tuple[str, str, str]]]:
    """Build a client whose sync path lists ``objects`` and records ``download_file``."""
    downloads: list[tuple[str, str, str]] = []

    class _FakeSyncS3:
        def download_file(self, bucket: str, key: str, filename: str, **_kwargs: object) -> None:
            downloads.append((bucket, key, filename))
            path = Path(filename)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"ok")

    client = object.__new__(S3Client)
    client.s3 = _FakeSyncS3()
    client.list_recursive = lambda *_a, **_k: objects  # type: ignore[method-assign]
    return client, downloads


def test_sync_remote_to_local_writes_nested_keys_under_dest(tmp_path: Path) -> None:
    """A nested weights key stays inside the cache directory."""
    dest = tmp_path / "cache"
    client, downloads = _sync_client(
        [
            {
                "Key": "weights/nested/model.bin",
                "Size": 2,
                "LastModified": datetime(2026, 1, 1, tzinfo=UTC),
            }
        ]
    )

    client.sync_remote_to_local(S3Prefix("s3://bucket/weights"), dest)

    expected = dest / "nested" / "model.bin"
    assert expected.read_bytes() == b"ok"
    assert downloads == [("bucket", "weights/nested/model.bin", str(expected.resolve()))]


def test_sync_remote_to_local_rejects_a_key_that_escapes_dest(tmp_path: Path) -> None:
    """Object keys with ``..`` must not write beside the destination."""
    dest = tmp_path / "cache"
    client, downloads = _sync_client(
        [
            {
                "Key": "weights/../escape.bin",
                "Size": 2,
                "LastModified": datetime(2026, 1, 1, tzinfo=UTC),
            }
        ]
    )

    with pytest.raises(ValueError, match="outside"):
        client.sync_remote_to_local(S3Prefix("s3://bucket/weights"), dest)

    assert downloads == []
    assert not (tmp_path / "escape.bin").exists()


def test_sync_remote_to_local_rejects_escaping_keys_before_any_download(tmp_path: Path) -> None:
    """A later traversal key must not leave earlier objects on disk."""
    dest = tmp_path / "cache"
    client, downloads = _sync_client(
        [
            {
                "Key": "weights/good.bin",
                "Size": 2,
                "LastModified": datetime(2026, 1, 1, tzinfo=UTC),
            },
            {
                "Key": "weights/../escape.bin",
                "Size": 2,
                "LastModified": datetime(2026, 1, 1, tzinfo=UTC),
            },
        ]
    )

    with pytest.raises(ValueError, match="outside"):
        client.sync_remote_to_local(S3Prefix("s3://bucket/weights"), dest)

    assert downloads == []
    assert not (dest / "good.bin").exists()
    assert not (tmp_path / "escape.bin").exists()


def test_sync_remote_to_local_delete_walks_the_resolved_dest(tmp_path: Path) -> None:
    """Stale files are removed from the real directory even when dest is a symlink."""
    real = tmp_path / "real"
    real.mkdir()
    dest = tmp_path / "cache"
    dest.symlink_to(real)
    (real / "stale.bin").write_bytes(b"old")
    client, downloads = _sync_client([])

    client.sync_remote_to_local(S3Prefix("s3://bucket/weights"), dest, delete=True)

    assert downloads == []
    assert not (real / "stale.bin").exists()


def test_join_under_rejects_a_dest_symlink_that_escapes_root(tmp_path: Path) -> None:
    """A planted leaf symlink that points outside dest is the zip analog we can have."""
    dest = tmp_path / "cache"
    dest.mkdir()
    (dest / "model.bin").symlink_to(tmp_path / "outside.bin")
    with pytest.raises(ValueError, match="outside"):
        join_under(dest.resolve(), "model.bin")


# Blocks ``ray`` at the import system level, then does what the client CLI does: import the
# module and validate an S3 location with ``S3Prefix``. Run as a subprocess so the guard is
# meaningful in environments that *do* have ray (``dev``, ``default``) and so a partially
# imported module cannot leak into the rest of the test session.
_IMPORT_WITHOUT_RAY = """
import sys


class _RayBlocker:
    def find_spec(self, name, path=None, target=None):
        if name == "ray" or name.startswith("ray."):
            msg = "ray is unavailable in the client-only 'tools' environment"
            raise ImportError(msg)
        return None


sys.meta_path.insert(0, _RayBlocker())

from cosmos_curator.core.utils.storage.s3_client import S3Prefix

assert S3Prefix("s3://some-bucket/some/key").bucket == "some-bucket"
assert "ray" not in sys.modules, "importing s3_client pulled in ray"
"""


def test_module_imports_without_ray(repo_root: Path) -> None:
    """Importing this module must not require ray.

    ``S3Prefix`` is pure string validation, and the client CLI uses it to validate ``s3://``
    locations in ``cosmos-curator pipeline validate``. The client-only ``tools`` pixi
    environment has no ray, so a module-scope ``nvcf_utils`` import (which imports ray) makes
    validating any S3 config fail with ``ModuleNotFoundError: No module named 'ray'``.
    """
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", _IMPORT_WITHOUT_RAY],
        capture_output=True,
        text=True,
        check=False,
        cwd=repo_root,
    )

    assert result.returncode == 0, f"s3_client is not importable without ray:\n{result.stderr}"
