# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recursive selection through paginated S3 listings, deferred when recovering saved work."""

from collections.abc import Iterator
from pathlib import Path

import lance
import pyarrow as pa
import pytest
from botocore.exceptions import ClientError
from botocore.stub import Stubber

from cosmos_curator.core.utils.storage.s3_client import S3Client, S3ClientConfig
from cosmos_curator.next.recipes.video_cut import discovery, pipeline
from cosmos_curator.next.recipes.video_cut.bindings import resolve_bindings
from cosmos_curator.next.recipes.video_cut.config import InputConfig, ResolvedVideoCutConfig, resolve_config_data
from cosmos_curator.next.recipes.video_cut.contracts import INPUT_SCHEMA, make_source_id
from tests.cosmos_curator.next.append.helpers import replace_ray, select, stage_results


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


def test_explicit_sources_do_not_access_storage(
    config: ResolvedVideoCutConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An explicit selection needs neither a listing nor extra object metadata requests."""
    monkeypatch.setattr(discovery, "s3_client", lambda _: pytest.fail("explicit sources must not access S3"))
    config = config.model_copy(
        update={
            "input": InputConfig(uris=("s3://test-bucket/b.mp4", "s3://test-bucket/a.mp4", "s3://test-bucket/b.mp4"))
        }
    )
    items = discovery.discover_items(config)
    assert items.schema.equals(INPUT_SCHEMA)
    assert items.to_pylist() == [
        {"source_id": make_source_id(uri), "source_uri": uri}
        for uri in ("s3://test-bucket/a.mp4", "s3://test-bucket/b.mp4")
    ]


def test_root_selects_all_mp4_pages_and_preserves_keys(
    storage: Stubber, config: ResolvedVideoCutConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
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
    config = config.model_copy(
        update={
            "input": InputConfig(root_uri="s3://test-bucket/raw/"),
            "execution": config.execution.model_copy(update={"storage_profile": "training"}),
        }
    )
    assert discovery.discover_items(config)["source_uri"].to_pylist() == [
        "s3://test-bucket/raw/a%2Fb//c.mp4",
        "s3://test-bucket/raw/nested/z.MP4",
    ]
    assert profiles == ["training"]


def test_bucket_root_selects_top_level_and_nested_mp4s(storage: Stubber, config: ResolvedVideoCutConfig) -> None:
    """A bucket root uses an empty listing prefix, including objects at the bucket top."""
    storage.add_response(
        "list_objects_v2",
        {"IsTruncated": False, "Contents": [{"Key": "top.mp4"}, {"Key": "nested/a.mp4"}]},
        {"Bucket": "test-bucket", "Prefix": ""},
    )
    config = config.model_copy(update={"input": InputConfig(root_uri="s3://test-bucket")})
    assert discovery.discover_items(config)["source_uri"].to_pylist() == [
        "s3://test-bucket/nested/a.mp4",
        "s3://test-bucket/top.mp4",
    ]


def test_root_rejects_sibling_objects(storage: Stubber, config: ResolvedVideoCutConfig) -> None:
    """An unexpected backend response cannot expand selection beyond the chosen root."""
    storage.add_response(
        "list_objects_v2",
        {"IsTruncated": False, "Contents": [{"Key": "raw-old/a.mp4"}]},
        {"Bucket": "test-bucket", "Prefix": "raw/"},
    )
    config = config.model_copy(update={"input": InputConfig(root_uri="s3://test-bucket/raw")})
    with pytest.raises(ValueError, match="outside"):
        discovery.discover_items(config)


def test_failed_discovery_never_initializes_ray_or_selects_work(
    storage: Stubber, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Listing failures propagate before selecting work or starting media processing."""
    expected = {"Bucket": "test-bucket", "Prefix": "raw/"}
    storage.add_client_error("list_objects_v2", service_error_code="AccessDenied", expected_params=expected)
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
    with pytest.raises(ClientError, match="AccessDenied"):
        pipeline.run_config(config)
    assert lance.dataset(config.output.spans_lance_uri).count_rows() == 0
    assert not Path(config.output.staging_root_uri).exists()


@pytest.mark.parametrize("saved", [False, True])
def test_empty_discovery_only_runs_without_a_saved_selection(
    storage: Stubber,
    config: ResolvedVideoCutConfig,
    results: pa.Table,
    monkeypatch: pytest.MonkeyPatch,
    *,
    saved: bool,
) -> None:
    """A saved selection publishes without listing; otherwise empty discovery is a no-op."""
    assert config.input.uris is not None
    if saved:
        attempt = select(resolve_bindings(config), discovery.discover_items(config))
        stage_results(attempt, results)
    settings = config.model_dump(mode="json")
    settings["input"] = {"root_uri": "s3://test-bucket/raw"}
    config = resolve_config_data(settings)
    if not saved:
        storage.add_response(
            "list_objects_v2",
            {"IsTruncated": False, "Contents": [{"Key": "raw/readme.txt"}]},
            {"Bucket": "test-bucket", "Prefix": "raw/"},
        )
    runs = replace_ray(pipeline, monkeypatch, results=None)
    summary = pipeline.run_config(config)
    assert (summary["input_items"], summary["remaining_items"]) == ((None, None) if saved else (0, 0))
    assert summary["cleanup_complete"] is True
    assert (summary["selected_items"], summary["appended_rows"]) == ((4, 8) if saved else (0, 0))
    if saved:
        assert attempt.workspace is not None
        assert not Path(attempt.workspace.root).exists()
    else:
        assert runs.calls[0][1].attempt.workspace is None
