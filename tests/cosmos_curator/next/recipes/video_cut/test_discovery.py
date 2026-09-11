# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recursive selection through paginated S3 listings, before any output writes."""

from collections.abc import Iterator
from pathlib import Path

import pytest
from botocore.exceptions import ClientError
from botocore.stub import Stubber

from cosmos_curator.core.utils.storage.s3_client import S3Client, S3ClientConfig
from cosmos_curator.next.recipes.video_cut import discovery, pipeline
from cosmos_curator.next.recipes.video_cut.config import InputConfig, resolve_config_data


@pytest.fixture
def storage(monkeypatch: pytest.MonkeyPatch) -> Iterator[Stubber]:
    """Exercise the shared S3 paginator against deterministic SDK responses."""
    client = S3Client(
        S3ClientConfig(aws_access_key_id="testing", aws_secret_access_key="testing", region="us-east-1")  # noqa: S106
    )
    monkeypatch.setattr(discovery, "s3_client", lambda _: client)
    with Stubber(client.s3) as stubber:
        yield stubber
        stubber.assert_no_pending_responses()


def test_explicit_sources_do_not_access_storage(monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit selection needs neither a listing nor extra object metadata requests."""
    monkeypatch.setattr(discovery, "s3_client", lambda _: pytest.fail("explicit sources must not access S3"))
    config = InputConfig(uris=("s3://test-bucket/b.mp4", "s3://test-bucket/a.mp4", "s3://test-bucket/b.mp4"))
    assert discovery.resolve_input_uris(config, storage_profile="training") == (
        "s3://test-bucket/a.mp4",
        "s3://test-bucket/b.mp4",
    )


def test_root_selects_all_mp4_pages_and_preserves_keys(storage: Stubber, monkeypatch: pytest.MonkeyPatch) -> None:
    """Directory boundaries, mixed-case suffixes, nested keys, and duplicates are handled."""
    storage.add_response(
        "list_objects_v2",
        {
            "IsTruncated": True,
            "NextContinuationToken": "next",
            "Contents": [{"Key": "raw/nested/z.MP4"}, {"Key": "raw/readme.txt"}, {"Key": "raw/folder.mp4/"}],
        },
        {"Bucket": "test-bucket", "Prefix": "raw/"},
    )
    storage.add_response(
        "list_objects_v2",
        {"IsTruncated": False, "Contents": [{"Key": "raw/a%2Fb//c.mp4"}, {"Key": "raw/nested/z.MP4"}]},
        {"Bucket": "test-bucket", "Prefix": "raw/", "ContinuationToken": "next"},
    )
    client = discovery.s3_client("training")
    profiles = []

    def selected_client(profile: str) -> S3Client:
        profiles.append(profile)
        return client

    monkeypatch.setattr(discovery, "s3_client", selected_client)
    assert discovery.resolve_input_uris(InputConfig(root_uri="s3://test-bucket/raw/"), storage_profile="training") == (
        "s3://test-bucket/raw/a%2Fb//c.mp4",
        "s3://test-bucket/raw/nested/z.MP4",
    )
    assert profiles == ["training"]


def test_bucket_root_selects_top_level_and_nested_mp4s(storage: Stubber) -> None:
    """A bucket root uses an empty listing prefix, including objects at the bucket top."""
    storage.add_response(
        "list_objects_v2",
        {"IsTruncated": False, "Contents": [{"Key": "top.mp4"}, {"Key": "nested/a.mp4"}]},
        {"Bucket": "test-bucket", "Prefix": ""},
    )
    assert discovery.resolve_input_uris(InputConfig(root_uri="s3://test-bucket"), storage_profile="default") == (
        "s3://test-bucket/nested/a.mp4",
        "s3://test-bucket/top.mp4",
    )


def test_root_rejects_sibling_objects(storage: Stubber) -> None:
    """An unexpected backend response cannot expand selection beyond the chosen root."""
    storage.add_response(
        "list_objects_v2",
        {"IsTruncated": False, "Contents": [{"Key": "raw-old/a.mp4"}]},
        {"Bucket": "test-bucket", "Prefix": "raw/"},
    )
    with pytest.raises(ValueError, match="outside"):
        discovery.resolve_input_uris(InputConfig(root_uri="s3://test-bucket/raw"), storage_profile="default")


@pytest.mark.parametrize("unavailable", [False, True])
def test_failed_discovery_never_initializes_ray_or_outputs(
    storage: Stubber, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, unavailable: bool
) -> None:
    """Empty selections and listing failures fail before creating durable state."""
    expected = {"Bucket": "test-bucket", "Prefix": "raw/"}
    if unavailable:
        storage.add_client_error("list_objects_v2", service_error_code="AccessDenied", expected_params=expected)
    else:
        storage.add_response(
            "list_objects_v2", {"IsTruncated": False, "Contents": [{"Key": "raw/readme.txt"}]}, expected
        )
    root = tmp_path / "output"
    config = resolve_config_data(
        {
            "schema_version": 1,
            "kind": "video-cut",
            "input": {"root_uri": "s3://test-bucket/raw"},
            "output": {"media_root": str(root)},
        }
    )
    monkeypatch.setattr(pipeline, "ensure_ray_initialized", lambda **_: pytest.fail("unexpected Ray initialization"))
    with pytest.raises(ClientError if unavailable else ValueError, match="AccessDenied" if unavailable else "No MP4"):
        pipeline.run_config(config)
    assert not root.exists()
