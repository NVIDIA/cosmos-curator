# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Phase A read, fan-out, and checkpointed Parquet staging around the shared caption plan."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import lance
import pyarrow as pa
import pyarrow.fs as pafs
import pyarrow.parquet as pq
import pytest
import ray
import ray.cloudpickle as ray_cloudpickle
from lance.fragment import LanceFragment

from cosmos_curator.next.legacy.caption_fragment_recovery import inference
from cosmos_curator.next.legacy.caption_fragment_recovery.bindings import CaptionRunBindings
from cosmos_curator.next.legacy.caption_fragment_recovery.inference import (
    CaptionExecutionConfig,
    run_inference_phase,
)
from cosmos_curator.next.legacy.caption_fragment_recovery.lance_state import (
    CaptionAttempt,
    capture_attempt,
    ensure_caption_fields,
)
from cosmos_curator.next.legacy.caption_fragment_recovery.workspace import (
    CaptionWorkspace,
    phase_a_completion_covers,
    staged_result_files,
)
from cosmos_curator.next.recipes.video_caption import inference as caption_inference
from cosmos_curator.next.recipes.video_caption.contracts import CaptionModelSpec
from cosmos_curator.next.recipes.video_transcode.contracts import CLIP_SCHEMA
from tests.cosmos_curator.next.conftest import CURATOR_IO_RESOURCE, CURATOR_IO_SLOTS
from tests.cosmos_curator.next.legacy.caption_fragment_recovery.helpers import (
    append_clip_fragment,
    write_clip_fragment,
)


class _FakeCaptionProcessor:
    """Small CPU stand-in that preserves Ray LLM's postprocessed row shape."""

    def __init__(self, expected_ids: set[str]) -> None:
        self._expected_ids = expected_ids

    def __call__(self, dataset: ray.data.Dataset) -> ray.data.Dataset:
        return dataset.map(_fake_inference, fn_kwargs={"expected_ids": self._expected_ids})


def _fake_inference(row: dict[str, Any], *, expected_ids: set[str]) -> dict[str, object]:
    clip_id = str(row["clip_id"])
    if clip_id not in expected_ids or row.get("download_error_type") is not None:
        msg = f"Checkpoint filter leaked unexpected clip {clip_id!r} into media/inference"
        raise AssertionError(msg)
    return {
        "fragment_id": row["fragment_id"],
        "row_offset": row["row_offset"],
        "clip_id": clip_id,
        "caption_text": f"caption for {clip_id}",
        "prompt_token_count": 4,
        "generated_token_count": 8,
        "__inference_error__": "",
    }


def _workspace(tmp_path: Path) -> CaptionWorkspace:
    root = tmp_path / "workspace"
    return CaptionWorkspace(
        root_uri=str(root),
        manifest_uri=str(root / "workspace.json"),
        results_uri=str(root / "results"),
        checkpoints_uri=str(root / "checkpoints"),
        filesystem=pafs.LocalFileSystem(),
    )


def _execution(**updates: object) -> CaptionExecutionConfig:
    settings: dict[str, Any] = {
        "storage_profile": "default",
        "inference_concurrency": "auto",
        "inference_batch_size": 32,
        "max_concurrent_batches": 8,
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "media_concurrency": "auto",
        "media_batch_size": 4,
        "media_cpus": 0.25,
        "media_attempts": 3,
        "lance_read_batch_size": 256,
        "parquet_rows_per_file": 4_096,
    }
    return CaptionExecutionConfig(**(settings | updates))


def test_caption_lance_read_task_reopens_with_storage_options_without_pickling_fragments(
    clip_dataset_factory: Callable[..., tuple[str, lance.LanceDataset]],
    clip_row_factory: Callable[..., dict[str, object]],
    monkeypatch: pytest.MonkeyPatch,
    caption_bindings: CaptionRunBindings,
) -> None:
    """Ray tasks carry credentials and fragment IDs, never storage-blind fragment handles."""
    uri, dataset = clip_dataset_factory(rows=[clip_row_factory(0)])
    attempt = CaptionAttempt(version=int(dataset.version), pending_fragment_ids=(0,), complete_fragment_ids=())
    storage_options = {
        "aws_access_key_id": "id",
        "aws_secret_access_key": "secret",
        "aws_session_token": "token",
        "aws_endpoint": "https://s3.example.test",
    }
    datasource = inference.caption_lance_datasource(
        dataset,
        attempt,
        storage_options=storage_options,
        batch_size=17,
        source_columns=caption_bindings.input.source_columns,
        source_schema=caption_bindings.input.source_schema,
    )
    read_task = datasource.get_read_tasks(parallelism=1)[0]

    def reject_fragment_pickle(_fragment: LanceFragment) -> object:
        msg = "a live LanceFragment crossed the Ray serialization boundary"
        raise AssertionError(msg)

    monkeypatch.setattr(LanceFragment, "__reduce__", reject_fragment_pickle)
    calls: list[tuple[str, int, dict[str, str] | None]] = []
    fragment_token = object()
    table = pa.table(
        {
            "clip_id": ["clip-0"],
            "clip_uri": ["s3://bucket/clip-0.mp4"],
            "clip_size_bytes": pa.array([1], type=pa.int64()),
            "_rowaddr": pa.array([0], type=pa.uint64()),
        }
    )

    class FakeDataset:
        def get_fragment(self, fragment_id: int) -> object | None:
            assert fragment_id == 0
            return fragment_token

        def scanner(self, **kwargs: object) -> object:
            assert kwargs == {
                "columns": ["clip_id", "clip_uri", "clip_size_bytes"],
                "fragments": [fragment_token],
                "with_row_address": True,
                "batch_size": 17,
            }

            class FakeScanner:
                def to_reader(self) -> pa.RecordBatchReader:
                    return pa.RecordBatchReader.from_batches(table.schema, table.to_batches())

            return FakeScanner()

    def reopen_dataset(
        reopened_uri: str,
        *,
        version: int,
        storage_options: dict[str, str] | None,
    ) -> FakeDataset:
        calls.append((reopened_uri, version, storage_options))
        return FakeDataset()

    monkeypatch.setattr(inference.lance, "dataset", reopen_dataset)

    restored_task = ray_cloudpickle.loads(ray_cloudpickle.dumps(read_task))
    output = list(restored_task())

    assert calls == [(uri, attempt.version, storage_options)]
    assert len(output) == 1
    assert output[0].equals(table)


@pytest.mark.integration
def test_checkpointed_phase_a_reuses_results_against_a_newer_input_superset(
    tmp_path: Path,
    clip_dataset_factory: Callable[..., tuple[str, lance.LanceDataset]],
    clip_row_factory: Callable[..., dict[str, object]],
    caption_spec: CaptionModelSpec,
    caption_digest: str,
    caption_bindings: CaptionRunBindings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Committed clip IDs skip media and inference after a post-attempt split append."""
    old_paths = [tmp_path / "clip-0.mp4", tmp_path / "clip-1.mp4"]
    rows = []
    for index, path in enumerate(old_paths):
        payload = f"video-{index}".encode()
        path.write_bytes(payload)
        row = clip_row_factory(index)
        row["clip_uri"] = str(path)
        row["clip_size_bytes"] = len(payload)
        rows.append(row)
    uri, _ = clip_dataset_factory(rows=rows, rows_per_fragment=1)
    dataset = ensure_caption_fields(uri, storage_options=None, bindings=caption_bindings, attempts=3)
    attempt = capture_attempt(dataset, bindings=caption_bindings)
    config = _execution(media_batch_size=1, parquet_rows_per_file=2)
    workspace = _workspace(tmp_path)

    ray.shutdown()
    # The shared cluster's own slot count, not a literal: ``curator_io_cluster``
    # restarts any cluster whose count differs, so a copy here would silently
    # trigger a mid-session restart the moment the fixture's constant moved.
    ray.init(num_cpus=4, resources={CURATOR_IO_RESOURCE: CURATOR_IO_SLOTS}, include_dashboard=False)
    try:
        monkeypatch.setattr(
            caption_inference,
            "_build_llm_processor",
            lambda *_args, **_kwargs: _FakeCaptionProcessor({"clip-0", "clip-1"}),
        )
        run_inference_phase(
            dataset,
            attempt,
            config,
            caption_bindings,
            caption_spec.runtime_model_dir,
            workspace,
            storage_options=None,
        )
        assert phase_a_completion_covers(workspace, attempt, caption_digest) is True
        assert len(staged_result_files(workspace)) == 1
        assert len(list(Path(workspace.checkpoints_uri).glob("*.parquet"))) == 1

        for path in old_paths:
            path.unlink()
        new_payload = b"new-video"
        new_path = tmp_path / "clip-99.mp4"
        new_path.write_bytes(new_payload)
        appended_row = clip_row_factory(99)
        appended_row["clip_uri"] = str(new_path)
        appended_row["clip_size_bytes"] = len(new_payload)
        candidate = write_clip_fragment(
            pa.Table.from_pylist([appended_row], schema=CLIP_SCHEMA),
            uri=uri,
            storage_profile="default",
        )
        assert candidate is not None
        append_clip_fragment(candidate, uri=uri, storage_profile="default", attempts=3)
        latest = lance.dataset(uri)
        superset_attempt = capture_attempt(latest, bindings=caption_bindings)
        assert phase_a_completion_covers(workspace, superset_attempt, caption_digest) is False
        monkeypatch.setattr(
            caption_inference,
            "_build_llm_processor",
            lambda *_args, **_kwargs: _FakeCaptionProcessor({"clip-99"}),
        )

        run_inference_phase(
            latest,
            superset_attempt,
            config,
            caption_bindings,
            caption_spec.runtime_model_dir,
            workspace,
            storage_options=None,
        )
        assert phase_a_completion_covers(workspace, superset_attempt, caption_digest) is True
    finally:
        ray.data.DataContext.get_current().checkpoint_config = None
        ray.shutdown()

    result_tables = [pq.read_table(path, filesystem=workspace.filesystem) for path in staged_result_files(workspace)]
    staged = pa.concat_tables(result_tables)
    assert sorted(staged["clip_id"].to_pylist()) == ["clip-0", "clip-1", "clip-99"]
