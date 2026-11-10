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

"""Tests for S3 candidate session discovery against a recording fake client."""

from typing import Any

import pytest
from botocore.exceptions import ClientError

from cosmos_curator.core.utils.storage.s3_client import S3Client
from cosmos_curator.next.recipes.multimodal_split import discovery
from cosmos_curator.next.recipes.multimodal_split.config import MultimodalSplitInputConfig
from cosmos_curator.next.recipes.multimodal_split.discovery import discover_candidate_sessions

# Keys under s3://example-bucket/recordings/. Sessions are the immediate child
# prefixes; the loose manifest object and the deeper artifacts must be ignored.
_BUCKET_KEYS = (
    "recordings/session-b/camera/front.mp4",
    "recordings/session-b/imu.mcap",
    "recordings/session-a/camera/front.mp4",
    "recordings/session-a/camera/left.mp4",
    "recordings/session-c/imu.mcap",
    "recordings/manifest.json",
    "recordings-other/session-z/imu.mcap",
)


class _FakePaginator:
    """Reproduces the CommonPrefixes/Contents split of a delimited list_objects_v2."""

    def __init__(self, keys: tuple[str, ...], calls: list[dict[str, Any]]) -> None:
        self._keys = keys
        self._calls = calls

    def paginate(self, **kwargs: Any) -> list[dict[str, Any]]:  # noqa: ANN401
        self._calls.append(kwargs)
        prefix = kwargs["Prefix"]
        delimiter = kwargs.get("Delimiter")
        matching = [key for key in self._keys if key.startswith(prefix)]
        if not delimiter:
            return [{"Contents": [{"Key": key} for key in matching]}]

        contents: list[dict[str, str]] = []
        common: list[dict[str, str]] = []
        for key in matching:
            remainder = key[len(prefix) :]
            head, separator, _ = remainder.partition(delimiter)
            if separator:
                entry = {"Prefix": f"{prefix}{head}{delimiter}"}
                if entry not in common:
                    common.append(entry)
            else:
                contents.append({"Key": key})
        # Split across pages to prove pagination is consumed rather than assumed.
        return [
            {"CommonPrefixes": common[:1], "Contents": contents},
            {"CommonPrefixes": common[1:]},
        ]


class _FakeS3Client(S3Client):
    """An S3Client whose boto3 layer is replaced by the fake paginator."""

    def __init__(self, keys: tuple[str, ...] = _BUCKET_KEYS) -> None:
        self.paginate_calls: list[dict[str, Any]] = []
        self.downloaded: list[str] = []
        self.existence_checks: list[str] = []
        self._paginator = _FakePaginator(keys, self.paginate_calls)
        self.payloads: dict[str, bytes] = {}

    @property
    def s3(self) -> Any:  # noqa: ANN401
        return self

    def get_paginator(self, operation_name: str) -> _FakePaginator:
        assert operation_name == "list_objects_v2"
        return self._paginator

    def object_exists(self, dest: Any) -> bool:  # noqa: ANN401
        self.existence_checks.append(str(dest))
        return str(dest) in self.payloads

    def download_object_as_bytes(self, uri: Any, chunk_size_bytes: int = 0) -> bytes:  # noqa: ANN401, ARG002
        self.downloaded.append(str(uri))
        return self.payloads[str(uri)]


@pytest.fixture
def fake_s3(monkeypatch: pytest.MonkeyPatch) -> _FakeS3Client:
    """Install a fake S3 client for every storage lookup discovery performs."""
    client = _FakeS3Client()
    monkeypatch.setattr(discovery, "get_storage_client", lambda *_args, **_kwargs: client)
    return client


@pytest.mark.usefixtures("fake_s3")
def test_s3_prefix_lists_only_immediate_child_prefixes() -> None:
    """Loose objects beside the sessions and sibling prefixes are not candidates."""
    config = MultimodalSplitInputConfig(
        mp4_index_uri="/index.lance", input_path_prefix="s3://example-bucket/recordings"
    )

    table = discover_candidate_sessions(config)

    assert table.to_pylist() == [
        {"source_session_id": "session-a", "session_uri": "s3://example-bucket/recordings/session-a"},
        {"source_session_id": "session-b", "session_uri": "s3://example-bucket/recordings/session-b"},
        {"source_session_id": "session-c", "session_uri": "s3://example-bucket/recordings/session-c"},
    ]


def test_s3_prefix_discovery_delegates_the_child_scope_to_the_store(fake_s3: _FakeS3Client) -> None:
    """A delimited listing keeps discovery O(sessions) instead of O(all objects)."""
    config = MultimodalSplitInputConfig(
        mp4_index_uri="/index.lance", input_path_prefix="s3://example-bucket/recordings"
    )

    discover_candidate_sessions(config)

    assert fake_s3.paginate_calls == [
        {"Bucket": "example-bucket", "Prefix": "recordings/", "Delimiter": "/"},
    ]


def test_s3_prefix_discovery_never_reads_or_probes_a_session_artifact(fake_s3: _FakeS3Client) -> None:
    """Discovery neither downloads nor HEADs any camera, sensor, or calibration object.

    Asserting only on downloads would miss existence probing, which is the cheaper
    and likelier way for artifact resolution to leak into discovery.
    """
    config = MultimodalSplitInputConfig(
        mp4_index_uri="/index.lance", input_path_prefix="s3://example-bucket/recordings"
    )

    discover_candidate_sessions(config)

    assert fake_s3.downloaded == []
    assert fake_s3.existence_checks == []


def test_s3_bucket_root_prefix_lists_from_the_bucket(fake_s3: _FakeS3Client) -> None:
    """A bucket-root prefix produces neither a leading nor a doubled separator."""
    config = MultimodalSplitInputConfig(mp4_index_uri="/index.lance", input_path_prefix="s3://example-bucket")

    table = discover_candidate_sessions(config)

    assert fake_s3.paginate_calls == [{"Bucket": "example-bucket", "Prefix": "", "Delimiter": "/"}]
    assert table.column("session_uri").to_pylist() == [
        "s3://example-bucket/recordings",
        "s3://example-bucket/recordings-other",
    ]


def test_s3_session_id_list_is_read_and_joined_to_the_prefix(fake_s3: _FakeS3Client) -> None:
    """A remote list file drives selection without listing the prefix at all."""
    fake_s3.payloads["s3://example-bucket/sessions.txt"] = b"  session-c \n\nsession-a\nsession-c\n"
    config = MultimodalSplitInputConfig(
        mp4_index_uri="/index.lance",
        input_path_prefix="s3://example-bucket/recordings",
        session_id_list_path="s3://example-bucket/sessions.txt",
    )

    table = discover_candidate_sessions(config)

    assert table.column("source_session_id").to_pylist() == ["session-a", "session-c"]
    assert table.column("session_uri").to_pylist() == [
        "s3://example-bucket/recordings/session-a",
        "s3://example-bucket/recordings/session-c",
    ]
    assert fake_s3.paginate_calls == []
    assert fake_s3.existence_checks == ["s3://example-bucket/sessions.txt"]


@pytest.mark.usefixtures("fake_s3")
def test_s3_limit_is_applied_after_sorting() -> None:
    """The limit takes a stable prefix of the canonical order."""
    config = MultimodalSplitInputConfig(
        mp4_index_uri="/index.lance",
        input_path_prefix="s3://example-bucket/recordings",
        limit=2,
    )

    table = discover_candidate_sessions(config)

    assert table.column("source_session_id").to_pylist() == ["session-a", "session-b"]


def test_an_unavailable_s3_client_fails_loudly(monkeypatch: pytest.MonkeyPatch) -> None:
    """Discovery never silently returns an empty selection when S3 is unreachable."""
    monkeypatch.setattr(discovery, "get_storage_client", lambda *_args, **_kwargs: None)
    config = MultimodalSplitInputConfig(
        mp4_index_uri="/index.lance", input_path_prefix="s3://example-bucket/recordings"
    )

    with pytest.raises(TypeError, match="S3 client"):
        discover_candidate_sessions(config)


def test_s3_session_id_list_tolerates_a_utf8_byte_order_mark(fake_s3: _FakeS3Client) -> None:
    """A BOM must not silently become part of the first session ID."""
    fake_s3.payloads["s3://example-bucket/sessions.txt"] = "\ufeffsession-a\nsession-b\n".encode()
    config = MultimodalSplitInputConfig(
        mp4_index_uri="/index.lance",
        input_path_prefix="s3://example-bucket/recordings",
        session_id_list_path="s3://example-bucket/sessions.txt",
    )

    table = discover_candidate_sessions(config)

    assert table.column("source_session_id").to_pylist() == ["session-a", "session-b"]


def test_missing_s3_session_id_list_fails_fast_like_a_missing_local_one(fake_s3: _FakeS3Client) -> None:
    """A typo in a remote list path must not burn the read-retry budget first."""
    config = MultimodalSplitInputConfig(
        mp4_index_uri="/index.lance",
        input_path_prefix="s3://example-bucket/recordings",
        session_id_list_path="s3://example-bucket/missing.txt",
    )

    with pytest.raises(FileNotFoundError, match="Session ID list does not exist"):
        discover_candidate_sessions(config)

    assert fake_s3.downloaded == []


def test_missing_s3_prefix_fails_instead_of_curating_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    """S3 answers a listing of a nonexistent prefix with 200 and no keys.

    Without this guard a transposed character in the prefix is indistinguishable
    from an empty result, and the run exits successfully having produced nothing.
    """
    client = _FakeS3Client()
    monkeypatch.setattr(discovery, "get_storage_client", lambda *_args, **_kwargs: client)
    config = MultimodalSplitInputConfig(
        mp4_index_uri="/index.lance", input_path_prefix="s3://example-bucket/recordigns"
    )

    with pytest.raises(FileNotFoundError, match="does not exist or contains no objects"):
        discover_candidate_sessions(config)


def test_a_prefix_holding_only_loose_objects_is_an_empty_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    """A prefix that exists but has no child prefixes is empty, not missing."""
    client = _FakeS3Client(keys=("recordings/manifest.json",))
    monkeypatch.setattr(discovery, "get_storage_client", lambda *_args, **_kwargs: client)
    config = MultimodalSplitInputConfig(
        mp4_index_uri="/index.lance", input_path_prefix="s3://example-bucket/recordings"
    )

    table = discover_candidate_sessions(config)

    assert table.num_rows == 0


def test_double_slash_keys_do_not_produce_an_empty_session_id(monkeypatch: pytest.MonkeyPatch) -> None:
    """A key with a doubled separator yields a blank child that must be dropped."""
    client = _FakeS3Client(keys=("recordings//stray.mp4", "recordings/session-a/imu.mcap"))
    monkeypatch.setattr(discovery, "get_storage_client", lambda *_args, **_kwargs: client)
    config = MultimodalSplitInputConfig(
        mp4_index_uri="/index.lance", input_path_prefix="s3://example-bucket/recordings"
    )

    table = discover_candidate_sessions(config)

    assert table.column("source_session_id").to_pylist() == ["session-a"]


def _client_error(code: str, status: int) -> ClientError:
    return ClientError(
        {"Error": {"Code": code, "Message": "denied"}, "ResponseMetadata": {"HTTPStatusCode": status}},
        "HeadObject",
    )


class _RefusingS3Client(_FakeS3Client):
    """An S3 client whose HEAD on the session ID list is refused by the store."""

    def __init__(self, error: ClientError) -> None:
        super().__init__()
        self._error = error

    def object_exists(self, dest: Any) -> bool:  # noqa: ANN401
        self.existence_checks.append(str(dest))
        raise self._error


def _list_config() -> MultimodalSplitInputConfig:
    return MultimodalSplitInputConfig(
        mp4_index_uri="/index.lance",
        input_path_prefix="s3://example-bucket/recordings",
        session_id_list_path="s3://example-bucket/sessions.txt",
    )


@pytest.mark.parametrize(
    ("code", "status"),
    [("404", 404), ("NotFound", 404), ("NoSuchKey", 200)],
)
def test_a_store_reporting_a_missing_list_by_key_code_is_a_missing_list(
    monkeypatch: pytest.MonkeyPatch,
    code: str,
    status: int,
) -> None:
    """S3Client.object_exists maps only the literal "404", so key-level codes reach us raw.

    An S3-compatible store answering NoSuchKey to a HEAD must still produce the
    intended message rather than a bare ClientError.
    """
    client = _RefusingS3Client(_client_error(code, status))
    monkeypatch.setattr(discovery, "get_storage_client", lambda *_args, **_kwargs: client)

    with pytest.raises(FileNotFoundError, match="Session ID list does not exist"):
        discover_candidate_sessions(_list_config())


@pytest.mark.parametrize(("code", "status"), [("403", 403), ("AccessDenied", 403)])
def test_a_forbidden_list_reports_permissions_and_the_listbucket_ambiguity(
    monkeypatch: pytest.MonkeyPatch,
    code: str,
    status: int,
) -> None:
    """Without s3:ListBucket, AWS answers 403 for an object that merely does not exist."""
    client = _RefusingS3Client(_client_error(code, status))
    monkeypatch.setattr(discovery, "get_storage_client", lambda *_args, **_kwargs: client)

    with pytest.raises(PermissionError, match="s3:ListBucket") as caught:
        discover_candidate_sessions(_list_config())

    assert "s3:GetObject" in str(caught.value)


def test_an_unexpected_store_error_is_not_disguised_as_a_missing_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only missing and forbidden are translated; everything else propagates unchanged."""
    error = _client_error("InternalError", 500)
    client = _RefusingS3Client(error)
    monkeypatch.setattr(discovery, "get_storage_client", lambda *_args, **_kwargs: client)

    with pytest.raises(ClientError) as caught:
        discover_candidate_sessions(_list_config())

    assert caught.value is error
