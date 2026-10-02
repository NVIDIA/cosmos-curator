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
    resolve_context_definition,
    resolve_model,
    resolve_prompt_template,
    result_field_names,
)
from cosmos_curator.next.recipes.multimodal_split.records import EPISODE_SCHEMA
from cosmos_curator.next.recipes.video_caption import inference
from cosmos_curator.next.recipes.video_caption.bindings import CaptionRunBindings
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
def test_phase_a_restart_and_distinct_camera_results_coexist(  # noqa: PLR0915
    tmp_path: Path,
    camera_factory: Callable[..., dict[str, object]],
    episode_factory: Callable[..., dict[str, object]],
    curator_io_cluster: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Durable Phase A, distinct camera fields, and exact reruns preserve both results."""
    del curator_io_cluster
    rows: list[dict[str, object]] = []
    expected: list[tuple[str, str, str]] = []
    expected_left: list[tuple[str, str]] = []
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
                "left_fisheye",
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
        expected_left.append((str(row["episode_id"]), sha256(left_payload).hexdigest()))

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
        "inference": {"prompt": {"mode": "default"}},
        "output": {"tag_camera_key": True},
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
    original_inference_phase = pipeline.run_inference_phase
    original_ensure_workspace = pipeline.ensure_workspace
    original_ensure_ray_initialized = pipeline.ensure_ray_initialized
    interrupted_workspace: CaptionWorkspace | None = None

    def interrupt_after_phase_a(
        _uri: str,
        _attempt: object,
        workspace: CaptionWorkspace,
        _bindings: CaptionRunBindings,
        **_kwargs: object,
    ) -> object:
        nonlocal interrupted_workspace
        properties = _bindings.publication.transaction_properties
        assert properties["result_key"] == "default_front_wide"
        assert "base_result_key" not in properties
        assert "tag_camera_key" not in properties
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
    definition = resolve_context_definition("video_only")
    prompt = resolve_prompt_template(definition, mode="default", custom_template=None)
    caption_field, metadata_field = result_field_names(definition.variant, "default_front_wide", model.key)
    published = lance.dataset(episodes_uri).to_table(columns=["episode_id", caption_field, metadata_field]).to_pylist()
    assert summary["published_fragments"] == 1
    assert (summary["base_result_key"], summary["tag_camera_key"], summary["result_key"]) == (
        "default",
        True,
        "default_front_wide",
    )
    assert [row[metadata_field]["result_key"] for row in published] == ["default_front_wide"] * len(rows)
    assert all(
        "base_result_key" not in row[metadata_field] and "tag_camera_key" not in row[metadata_field]
        for row in published
    )
    assert summary["terminal_counts"] == {"success": 2, "truncated": 0, "error": 0}
    assert len(published) == 2
    assert [row[caption_field] for row in published] == ["caption for episode-0", "caption for episode-1"]
    assert [row[metadata_field]["camera_key"] for row in published] == ["front_wide", "front_wide"]
    assert [row[metadata_field]["media_sha256"] for row in published] == [item[2] for item in expected]
    assert [row[metadata_field]["materialization_digest"] for row in published] == [
        row["materialization_digest"] for row in rows
    ]
    expected_contract_digest = caption_contract_digest(
        model, "front_wide", definition, None, prompt, "default_front_wide"
    )
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
    manual = resolve_config_data(config_data | {"output": {"result_key": "default_front_wide"}})
    assert pipeline.run_config(manual)["published_fragments"] == 0
    assert lance.dataset(episodes_uri).version == published_version

    monkeypatch.setattr(pipeline, "ensure_workspace", original_ensure_workspace)
    monkeypatch.setattr(pipeline, "ensure_ray_initialized", original_ensure_ray_initialized)
    monkeypatch.setattr(pipeline, "run_inference_phase", original_inference_phase)
    monkeypatch.setattr(pipeline, "validate_model_directory", lambda path, **_kwargs: path)
    front_left_data = config_data | {
        "context": {"variant": "video_only", "camera_key": "left_fisheye"},
    }
    front_left_config = resolve_config_data(front_left_data)
    left_summary = pipeline.run_config(front_left_config)
    left_caption_field, left_metadata_field = result_field_names(definition.variant, "default_left_fisheye", model.key)
    together = (
        lance.dataset(episodes_uri)
        .to_table(columns=["episode_id", caption_field, metadata_field, left_caption_field, left_metadata_field])
        .to_pylist()
    )
    assert left_summary["published_fragments"] == 1
    assert left_summary["terminal_counts"] == {"success": 2, "truncated": 0, "error": 0}
    assert [row[left_caption_field] for row in together] == ["caption for episode-0", "caption for episode-1"]
    assert [(row["episode_id"], row[left_metadata_field]["media_sha256"]) for row in together] == expected_left
    assert [row[left_metadata_field]["camera_key"] for row in together] == ["left_fisheye"] * len(rows)
    assert left_summary["result_key"] == "default_left_fisheye"
    assert [row[left_metadata_field]["contract_digest"] for row in together] == [
        caption_contract_digest(model, "left_fisheye", definition, None, prompt, "default_left_fisheye")
    ] * len(rows)
    assert [row[left_metadata_field]["materialization_digest"] for row in together] == [
        row["materialization_digest"] for row in rows
    ]
    assert [row[caption_field] for row in together] == [row[caption_field] for row in published]
    assert [row[metadata_field] for row in together] == [row[metadata_field] for row in published]

    left_version = lance.dataset(episodes_uri).version
    monkeypatch.setattr(
        pipeline,
        "ensure_workspace",
        lambda *_args, **_kwargs: pytest.fail("second-camera rerun must not create a workspace"),
    )
    monkeypatch.setattr(
        pipeline,
        "ensure_ray_initialized",
        lambda *_args, **_kwargs: pytest.fail("second-camera rerun must not initialize Ray"),
    )
    left_rerun_summary = pipeline.run_config(front_left_config)
    assert left_rerun_summary["published_fragments"] == 0
    assert lance.dataset(episodes_uri).version == left_version
    assert (
        lance.dataset(episodes_uri)
        .to_table(columns=["episode_id", caption_field, metadata_field, left_caption_field, left_metadata_field])
        .to_pylist()
        == together
    )

    incompatible = resolve_config_data(
        config_data
        | {
            "context": {"variant": "video_only", "camera_key": "left_fisheye"},
            "output": {"result_key": "default_front_wide"},
        }
    )
    with pytest.raises(ValueError, match="set tag_camera_key: true"):
        pipeline.run_config(incompatible)
    assert lance.dataset(episodes_uri).version == left_version

    # Omitted untagged output still binds default on this same multicamera table.
    monkeypatch.setattr(pipeline, "ensure_workspace", original_ensure_workspace)
    monkeypatch.setattr(pipeline, "ensure_ray_initialized", original_ensure_ray_initialized)
    untagged_data = {key: value for key, value in config_data.items() if key != "output"}
    untagged = resolve_config_data(untagged_data)
    assert pipeline.run_config(untagged)["published_fragments"] == 1
    version = lance.dataset(episodes_uri).version
    monkeypatch.setattr(pipeline, "run_inference_phase", lambda *_a, **_k: pytest.fail("rerun must skip inference"))
    assert pipeline.run_config(untagged)["published_fragments"] == 0
    with pytest.raises(ValueError, match="set tag_camera_key: true"):
        pipeline.run_config(
            resolve_config_data(
                untagged_data
                | {
                    "context": {"variant": "video_only", "camera_key": "left_fisheye"},
                }
            )
        )
    assert lance.dataset(episodes_uri).version == version
    assert lance.dataset(episodes_uri).to_table(columns=list(together[0])).to_pylist() == together
