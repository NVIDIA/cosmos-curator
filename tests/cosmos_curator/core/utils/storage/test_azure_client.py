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
"""Tests for Azure client metadata lookup and suffix-filtered listing."""

from collections.abc import Iterator
from datetime import UTC, datetime

import pytest
from azure.core.exceptions import HttpResponseError, ResourceNotFoundError

from cosmos_curator.core.utils.storage.azure_client import AzureClient, AzurePrefix
from cosmos_curator.core.utils.storage.s3_client import S3Prefix
from cosmos_curator.core.utils.storage.storage_client import StorageStat

_PREFIX = "az://test-container/root"


class _FakeBlobProperties:
    def __init__(self, size: int | None, last_modified: datetime | None, etag: str | None) -> None:
        self.size = size
        self.last_modified = last_modified
        self.etag = etag


class _FakeBlobClient:
    """Blob stand-in serving one canned properties answer and counting its calls."""

    def __init__(self, props: _FakeBlobProperties | None = None, error: Exception | None = None) -> None:
        self._props = props
        self._error = error
        self.property_calls = 0

    def get_blob_properties(self) -> _FakeBlobProperties:
        self.property_calls += 1
        if self._error is not None:
            raise self._error
        assert self._props is not None
        return self._props


class _FakeBlob:
    def __init__(self, name: str, size: int = 1) -> None:
        self.name = name
        self.size = size


class _FakeContainerClient:
    """Container stand-in whose listing records how many entries were consumed.

    Azure's ``list_blobs`` is a lazily paged iterator, so the consumed count is what
    shows an early-exiting listing stopped fetching rather than filtering afterwards.
    """

    def __init__(self, blobs: list[_FakeBlob]) -> None:
        self._blobs = blobs
        self.blobs_yielded = 0
        self.name_starts_with: str | None = None

    def list_blobs(self, name_starts_with: str = "") -> Iterator[_FakeBlob]:
        self.name_starts_with = name_starts_with
        return self._iter_blobs()

    def _iter_blobs(self) -> Iterator[_FakeBlob]:
        for blob in self._blobs:
            self.blobs_yielded += 1
            yield blob


class _FakeServiceClient:
    def __init__(
        self,
        blob_client: _FakeBlobClient | None = None,
        container_client: _FakeContainerClient | None = None,
    ) -> None:
        self._blob_client = blob_client
        self._container_client = container_client
        self.requested_blob: tuple[str, str] | None = None

    def get_blob_client(self, container: str, blob: str) -> _FakeBlobClient:
        self.requested_blob = (container, blob)
        assert self._blob_client is not None
        return self._blob_client

    def get_container_client(self, container: str) -> _FakeContainerClient:
        assert self._container_client is not None
        assert container == "test-container"
        return self._container_client


def _blob_lookup_client(
    props: _FakeBlobProperties | None = None,
    error: Exception | None = None,
) -> tuple[AzureClient, _FakeServiceClient, _FakeBlobClient]:
    """Build a client whose only wired-up Azure call is ``get_blob_properties``."""
    blob_client = _FakeBlobClient(props, error)
    service = _FakeServiceClient(blob_client=blob_client)
    client = object.__new__(AzureClient)
    client.service_client = service
    return client, service, blob_client


def _listing_client(blobs: list[_FakeBlob]) -> tuple[AzureClient, _FakeContainerClient]:
    """Build a client whose only wired-up Azure call is ``list_blobs``."""
    container = _FakeContainerClient(blobs)
    client = object.__new__(AzureClient)
    client.service_client = _FakeServiceClient(container_client=container)
    return client, container


def test_stat_returns_the_metadata_carried_by_one_properties_lookup() -> None:
    """The lookup ``object_exists`` used to discard now answers the size question too."""
    last_modified = datetime(2026, 8, 1, 12, 0, tzinfo=UTC)
    client, service, _ = _blob_lookup_client(_FakeBlobProperties(1234, last_modified, '"abc123"'))

    stat = client.stat(AzurePrefix(f"{_PREFIX}/a.mp4"))

    assert stat == StorageStat(size_bytes=1234, last_modified=last_modified, etag="abc123")
    assert service.requested_blob == ("test-container", "root/a.mp4")


def test_stat_raises_file_not_found_for_a_missing_blob() -> None:
    """Follow ``os.stat``: absence is an error, not an empty answer."""
    client, _, _ = _blob_lookup_client(error=ResourceNotFoundError("gone"))

    with pytest.raises(FileNotFoundError, match=r"az://test-container/root/a\.mp4"):
        client.stat(AzurePrefix(f"{_PREFIX}/a.mp4"))


def test_a_missing_blob_costs_exactly_one_request() -> None:
    """``stat`` must not be wrapped in ``do_with_retries``.

    Absence is a normal answer here, so it has to come back on the first request
    rather than after the 256-second backoff the transfer paths use.
    """
    client, _, blob_client = _blob_lookup_client(error=ResourceNotFoundError("gone"))

    assert client.object_exists(AzurePrefix(f"{_PREFIX}/a.mp4")) is False
    assert blob_client.property_calls == 1


def test_object_exists_is_true_for_a_present_blob() -> None:
    """A successful stat is reported as existence."""
    client, _, _ = _blob_lookup_client(_FakeBlobProperties(7, None, None))

    assert client.object_exists(AzurePrefix(f"{_PREFIX}/a.mp4")) is True


def test_a_non_not_found_error_still_propagates_out_of_object_exists() -> None:
    """Only ResourceNotFoundError means absence.

    An authorization failure reported as "does not exist" would turn a credentials
    problem into silently missing data.
    """
    client, _, _ = _blob_lookup_client(error=HttpResponseError("forbidden"))

    with pytest.raises(HttpResponseError, match="forbidden"):
        client.object_exists(AzurePrefix(f"{_PREFIX}/a.mp4"))


def test_suffix_filtered_listing_counts_matches_not_listed_blobs() -> None:
    """The limit means N videos, which is what ``list_recursive`` cannot promise."""
    client, container = _listing_client([_FakeBlob("root/a.mp4"), _FakeBlob("root/a.json"), _FakeBlob("root/b.mp4")])

    results = client.list_recursive_with_suffixes(AzurePrefix(_PREFIX), (".mp4",), limit=2)

    assert [str(item) for item in results] == [
        "az://test-container/root/a.mp4",
        "az://test-container/root/b.mp4",
    ]
    assert container.name_starts_with == "root"


def test_suffix_filtered_listing_stops_consuming_once_the_limit_is_met() -> None:
    """Abandoning the paged iterator early is what keeps a small limit cheap."""
    client, container = _listing_client([_FakeBlob("root/a.mp4"), _FakeBlob("root/b.mp4")])

    results = client.list_recursive_with_suffixes(AzurePrefix(_PREFIX), (".mp4",), limit=1)

    assert [str(item) for item in results] == ["az://test-container/root/a.mp4"]
    assert container.blobs_yielded == 1


def test_suffix_matching_ignores_case_on_both_sides() -> None:
    """An uppercase extension in the store and in the filter both still match."""
    client, _ = _listing_client([_FakeBlob("root/a.MP4"), _FakeBlob("root/b.mkv"), _FakeBlob("root/c.txt")])

    results = client.list_recursive_with_suffixes(AzurePrefix(_PREFIX), (".mp4", ".MKV"))

    assert [str(item) for item in results] == [
        "az://test-container/root/a.MP4",
        "az://test-container/root/b.mkv",
    ]


def test_a_zero_byte_match_is_reported_rather_than_treated_as_a_directory() -> None:
    """``list_recursive`` drops zero-byte entries; a suffix match must not be dropped.

    A zero-byte object whose name ends in a video suffix is a truncated upload, which
    is exactly what a data-integrity caller needs to see.
    """
    client, _ = _listing_client([_FakeBlob("root/truncated.mp4", size=0)])

    results = client.list_recursive_with_suffixes(AzurePrefix(_PREFIX), (".mp4",))

    assert [str(item) for item in results] == ["az://test-container/root/truncated.mp4"]


def test_an_empty_suffix_filter_is_rejected_rather_than_matching_nothing() -> None:
    """``str.endswith(())`` is False, so an empty filter would quietly return nothing."""
    client, _ = _listing_client([_FakeBlob("root/a.mp4")])

    with pytest.raises(ValueError, match="suffixes must not be empty"):
        client.list_recursive_with_suffixes(AzurePrefix(_PREFIX), ())


def test_an_empty_suffix_among_several_is_rejected_rather_than_matching_everything() -> None:
    """Mirrors the S3 client's guard, since both share the base-class helper."""
    client, _ = _listing_client([_FakeBlob("root/a.mp4"), _FakeBlob("root/a.json")])

    with pytest.raises(ValueError, match="suffixes must not contain an empty string"):
        client.list_recursive_with_suffixes(AzurePrefix(_PREFIX), (".mp4", ""))


def test_a_prefix_for_another_backend_is_refused_rather_than_asserted() -> None:
    """Mirrors the S3 client's guard, for the same reason.

    The check was an ``assert``, which ``python -O`` strips, so a caller's mix-up would
    surface later as a missing-attribute error rather than naming what went wrong.
    """
    client, _, _ = _blob_lookup_client(_FakeBlobProperties(7, None, None))

    with pytest.raises(TypeError, match="AzureClient requires an AzurePrefix, got S3Prefix"):
        client.stat(S3Prefix("s3://bucket/key"))


@pytest.mark.parametrize(
    "path",
    [
        "az://test-container/root/2026-08-01--12-00/a.mp4",
        "az://test-container/root/cam--01/a.mp4",
        "az://test-container/a--b",
    ],
)
def test_consecutive_hyphens_in_a_blob_name_are_accepted(path: str) -> None:
    """Azure forbids consecutive hyphens in a container name, not in a blob name.

    The rule belongs to the segment before the first delimiter. Applied to the whole
    path it rejects timestamped and camera-numbered layouts that the backend stores
    happily, and it does so as a "malformed URI" the operator cannot act on.
    """
    assert AzurePrefix(path).container == "test-container"


@pytest.mark.parametrize(
    "path",
    [
        "az://my--container/a.mp4",
        "az://a--b/root/a.mp4",
    ],
)
def test_consecutive_hyphens_in_the_container_are_still_rejected(path: str) -> None:
    """The container half of the rule is real: Azure will not serve such a name."""
    with pytest.raises(ValueError, match="Invalid Azure path format"):
        AzurePrefix(path)


def test_listing_does_not_abort_on_a_blob_whose_name_has_consecutive_hyphens() -> None:
    """One such blob used to fail the whole enumeration, not just its own entry.

    ``list_recursive_with_suffixes`` builds an ``AzurePrefix`` per match, so a single
    offending name raised out of the loop and abandoned every result behind it.
    """
    client, _ = _listing_client([_FakeBlob("root/cam--01/a.mp4"), _FakeBlob("root/b.mp4")])

    results = client.list_recursive_with_suffixes(AzurePrefix(_PREFIX), (".mp4",))

    assert [str(item) for item in results] == [
        "az://test-container/root/cam--01/a.mp4",
        "az://test-container/root/b.mp4",
    ]
