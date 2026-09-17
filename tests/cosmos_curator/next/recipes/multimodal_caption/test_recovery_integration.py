# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Representative local Phase A recovery and fragment-publication E2E."""

from collections.abc import Callable
from hashlib import sha256
from pathlib import Path
from typing import Any

import lance
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import ray

from cosmos_curator.next.recipes.multimodal_caption import pipeline
from cosmos_curator.next.recipes.multimodal_caption.config import resolve_config_data
from cosmos_curator.next.recipes.multimodal_caption.contracts import (
    caption_contract_digest,
    resolve_model,
    result_field_names,
)
from cosmos_curator.next.recipes.multimodal_split.records import EPISODE_SCHEMA
from cosmos_curator.next.recipes.video_caption import inference
from cosmos_curator.next.recipes.video_caption.inference import staged_result_files
from cosmos_curator.next.recipes.video_caption.workspace import CaptionWorkspace


class _FakeEpisodeProcessor:
    """CPU stand-in that records the exact selected camera entering inference."""

    def __call__(self, dataset: ray.data.Dataset) -> ray.data.Dataset:
        return dataset.map(_fake_inference)


def _fake_inference(row: dict[str, Any]) -> dict[str, object]:
    return {
        "fragment_id": row["fragment_id"],
        "row_offset": row["row_offset"],
        "episode_id": row["episode_id"],
        "materialization_digest": row["materialization_digest"],
        "camera_key": row["camera_key"],
        "media_location": row["media_location"],
        "media_size_bytes": row["media_size_bytes"],
        "media_sha256": row["media_sha256"],
        "caption_text": f"caption for {row['episode_id']}",
        "prompt_token_count": 4,
        "generated_token_count": 8,
        "__inference_error__": "",
    }


@pytest.mark.integration
def test_phase_a_restart_publishes_one_selected_camera_result_per_episode(  # noqa: PLR0915
    tmp_path: Path,
    camera_factory: Callable[..., dict[str, object]],
    episode_factory: Callable[..., dict[str, object]],
    curator_io_cluster: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Durable Phase A survives interruption and publishes only front-wide evidence."""
    del curator_io_cluster
    rows: list[dict[str, object]] = []
    expected: list[tuple[str, str, str]] = []
    for index in range(2):
        left_payload = f"left-{index}".encode()
        wide_payload = f"wide-{index}".encode()
        left_path = tmp_path / f"left-{index}.mp4"
        wide_path = tmp_path / f"wide-{index}.mp4"
        left_path.write_bytes(left_payload)
        wide_path.write_bytes(wide_payload)
        wide_sha = sha256(wide_payload).hexdigest()
        cameras = [
            camera_factory(
                "front_left",
                "1",
                location=str(left_path),
                size=len(left_payload),
                media_sha256=sha256(left_payload).hexdigest(),
            ),
            camera_factory(
                "front_wide",
                "2",
                location=str(wide_path),
                size=len(wide_payload),
                media_sha256=wide_sha,
            ),
        ]
        row = episode_factory(index, camera_media=cameras)
        rows.append(row)
        expected.append((str(row["episode_id"]), str(wide_path), wide_sha))

    episodes_uri = str(tmp_path / "episodes.lance")
    lance.write_dataset(
        pa.Table.from_pylist(rows, schema=EPISODE_SCHEMA),
        episodes_uri,
        data_storage_version="2.2",
    )
    config_data: dict[str, Any] = {
        "schema_version": 1,
        "kind": "multimodal-caption",
        "input": {"episodes_lance_uri": episodes_uri},
        "context": {"variant": "video_only", "camera_key": "front_wide"},
        "execution": {"media_batch_size": 1, "inference_batch_size": 2, "parquet_rows_per_file": 2},
    }
    config = resolve_config_data(config_data)
    observed: list[tuple[str, str, str]] = []
    monkeypatch.setattr(pipeline, "validate_model_directory", lambda path, **_kwargs: path)
    monkeypatch.setattr(
        inference,
        "_build_llm_processor",
        lambda *_args, **_kwargs: _FakeEpisodeProcessor(),
    )
    original_publish = pipeline.publish_staged_results
    interrupted_workspace: CaptionWorkspace | None = None

    def interrupt_after_phase_a(
        _uri: str,
        _attempt: object,
        workspace: CaptionWorkspace,
        _bindings: object,
        **_kwargs: object,
    ) -> object:
        nonlocal interrupted_workspace
        interrupted_workspace = workspace
        files = staged_result_files(workspace)
        assert files
        staged = pa.concat_tables([pq.read_table(path, filesystem=workspace.filesystem) for path in files])
        observed.extend(
            (str(row["episode_id"]), str(row["media_location"]), str(row["media_sha256"])) for row in staged.to_pylist()
        )
        msg = "simulated interruption after durable Phase A"
        raise InterruptedError(msg)

    monkeypatch.setattr(pipeline, "publish_staged_results", interrupt_after_phase_a)
    with pytest.raises(InterruptedError, match="after durable Phase A"):
        pipeline.run_config(config)

    assert observed == expected
    assert interrupted_workspace is not None
    assert Path(interrupted_workspace.phase_a_completion_uri).is_file()

    monkeypatch.setattr(pipeline, "publish_staged_results", original_publish)
    monkeypatch.setattr(
        pipeline,
        "validate_model_directory",
        lambda *_args, **_kwargs: pytest.fail("recovery must not validate model files"),
    )
    monkeypatch.setattr(
        pipeline,
        "run_inference_phase",
        lambda *_args, **_kwargs: pytest.fail("recovery must not repeat Phase A"),
    )
    summary = pipeline.run_config(config)

    model = resolve_model("qwen3_8_27b_fp8")
    caption_field, metadata_field = result_field_names(model.key)
    published = lance.dataset(episodes_uri).to_table(columns=["episode_id", caption_field, metadata_field]).to_pylist()
    assert summary["published_fragments"] == 1
    assert len(published) == 2
    assert [row[caption_field] for row in published] == ["caption for episode-0", "caption for episode-1"]
    assert [row[metadata_field]["camera_key"] for row in published] == ["front_wide", "front_wide"]
    assert [row[metadata_field]["media_sha256"] for row in published] == [item[2] for item in expected]
    assert [row[metadata_field]["materialization_digest"] for row in published] == [
        row["materialization_digest"] for row in rows
    ]
    expected_contract_digest = caption_contract_digest(model, "front_wide")
    assert [row[metadata_field]["contract_digest"] for row in published] == [expected_contract_digest] * len(rows)
    assert not Path(interrupted_workspace.root_uri).exists()

    published_version = lance.dataset(episodes_uri).version
    monkeypatch.setattr(
        pipeline,
        "ensure_workspace",
        lambda *_args, **_kwargs: pytest.fail("same-camera rerun must not create a workspace"),
    )
    monkeypatch.setattr(
        pipeline,
        "ensure_ray_initialized",
        lambda *_args, **_kwargs: pytest.fail("same-camera rerun must not initialize Ray"),
    )
    same_camera_summary = pipeline.run_config(config)
    unchanged = lance.dataset(episodes_uri).to_table(columns=["episode_id", caption_field, metadata_field]).to_pylist()
    assert same_camera_summary["published_fragments"] == 0
    assert same_camera_summary["already_committed_fragments"] == 0
    assert lance.dataset(episodes_uri).version == published_version
    assert unchanged == published

    front_left_config = resolve_config_data(
        config_data | {"context": {"variant": "video_only", "camera_key": "front_left"}}
    )
    with pytest.raises(ValueError, match="incompatible"):
        pipeline.run_config(front_left_config)
