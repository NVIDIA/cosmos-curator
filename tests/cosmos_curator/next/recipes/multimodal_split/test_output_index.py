# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Atomic publication of one immutable camera-index receipt."""

import hashlib
import io
import shutil
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import lance
import numpy as np
import pytest

from cosmos_curator.core.utils.storage.s3_client import S3Client
from cosmos_curator.next.recipes.mp4_index.lance_sink import open_or_create_manifest_table
from cosmos_curator.next.recipes.multimodal_split import output_index, storage


def test_retry_checksum_streams_s3_through_a_bounded_read_buffer(monkeypatch: pytest.MonkeyPatch) -> None:
    """The real smart_open reader hashes a remote payload without an unbounded body read."""
    payload = b"video payload" * (256 * 1024)

    class BoundedBody(io.BytesIO):
        def read(self, size: int = -1) -> bytes:
            assert 0 < size <= 1024 * 1024
            return super().read(size)

    client = Mock(spec=S3Client)
    client.s3 = Mock()
    client.s3.get_object.return_value = {
        "Body": BoundedBody(payload),
        "ContentLength": len(payload),
        "ResponseMetadata": {"HTTPStatusCode": 200, "RetryAttempts": 0},
    }
    monkeypatch.setattr(storage, "get_storage_client", lambda *_a, **_k: client)
    assert output_index._payload_sha256("s3://bucket/clip.mp4", "default") == hashlib.sha256(payload).hexdigest()
    client.s3.get_object.assert_called_once()


@pytest.fixture
def output_candidate(tmp_path: Path, make_session: Callable[..., Path]) -> tuple[Path, dict[str, Any]]:
    """Prepare an encoded MP4 and the identity/timing fields for its output index."""
    session = make_session(tmp_path / "input", "session", frames=30)
    local = session / "camera/front.mp4"
    timestamps = np.load(session / "camera/front.mp4.timestamps.npy")
    destination = str(tmp_path / "output/front.mp4")
    return local, {
        "destination": destination,
        "uri": output_index.output_index_uri(destination),
        "session_id": "session",
        "camera_key": "front",
        "alignment_timestamps_ns": timestamps,
        "source_timestamps_ns": timestamps,
        "split_contract_digest": "contract",
        "storage_profile": "default",
    }


@pytest.mark.parametrize("conflicting", [False, True])
def test_overlapping_output_index_publications_keep_one_receipt(
    tmp_path: Path,
    output_candidate: tuple[Path, dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
    *,
    conflicting: bool,
) -> None:
    """Matching attempts converge; different receipts cannot append or replace the winner."""
    local, fields = output_candidate
    before_commit = threading.Barrier(2)
    commit = lance.LanceDataset.commit

    def overlapping_commit(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401 -- synchronize native commit attempts
        before_commit.wait(timeout=10)
        return commit(*args, **kwargs)

    monkeypatch.setattr(lance.LanceDataset, "commit", staticmethod(overlapping_commit))

    def publish(index: int) -> dict[str, Any]:
        encoded = tmp_path / f"attempt-{index}/front.mp4"
        encoded.parent.mkdir()
        shutil.copyfile(local, encoded)
        overrides = {"split_contract_digest": "different-contract"} if conflicting and index else {}
        return output_index.publish_output_index(encoded, **(fields | overrides))

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(publish, index) for index in range(2)]
        results = []
        errors = []
        for future in futures:
            try:
                results.append(future.result(timeout=30))
            except ValueError as error:
                errors.append(str(error))
    dataset = lance.dataset(fields["uri"])
    assert dataset.count_rows() == 1
    assert dataset.version == 1
    assert len(results) == (1 if conflicting else 2), errors
    if conflicting:
        assert len(errors) == 1
        assert "conflicts" in errors[0]
    else:
        assert errors == []
        assert results[0] == results[1]
    assert (
        output_index.find_output_index(fields["destination"], uri=fields["uri"], storage_profile="default") is not None
    )


def test_empty_index_from_the_old_writer_fails_without_overwriting(
    output_candidate: tuple[Path, dict[str, Any]],
) -> None:
    """A legacy empty manifest is rejected explicitly rather than overwritten during a race."""
    local, fields = output_candidate
    open_or_create_manifest_table(uri=fields["uri"], storage_profile="default", schema=output_index.OUTPUT_INDEX_SCHEMA)
    with pytest.raises(ValueError, match=r"empty.*fresh output"):
        output_index.publish_output_index(local, **fields)
    dataset = lance.dataset(fields["uri"])
    assert dataset.count_rows() == 0
    assert dataset.version == 1


def test_lost_index_commit_response_reuses_the_complete_receipt(
    output_candidate: tuple[Path, dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A successful create followed by an I/O error is confirmed from the committed row."""
    local, fields = output_candidate
    commit = lance.LanceDataset.commit

    def lost_response(*args: object, **kwargs: object) -> None:
        commit(*args, **kwargs)
        msg = "lost commit response"
        raise OSError(msg)

    monkeypatch.setattr(lance.LanceDataset, "commit", staticmethod(lost_response))
    descriptor = output_index.publish_output_index(local, **fields)
    dataset = lance.dataset(fields["uri"])
    assert dataset.count_rows() == 1
    assert dataset.version == 1
    assert descriptor == output_index.index_descriptor(dataset.to_table(), uri=fields["uri"])
