# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the actual Ray checkpoint/publication plan with a CPU stand-in for vLLM."""

import json
from pathlib import Path
from typing import Any

import lance
import pyarrow as pa
import pytest
import ray

from cosmos_curator.core.utils import environment
from cosmos_curator.next.recipes.video_caption import inference, pipeline
from cosmos_curator.next.recipes.video_caption.config import ResolvedVideoCaptionConfig
from cosmos_curator.next.recipes.video_caption.contracts import resolve_model_spec
from cosmos_curator.next.recipes.video_caption.inference import MediaColumns, fetch_media_batch
from tests.cosmos_curator.next.enrichment.fresh_cluster import assert_fresh_cluster_resume
from tests.cosmos_curator.next.fresh_driver import run_in_fresh_driver
from tests.cosmos_curator.next.recipes.video_caption.conftest import RECIPE, write_clips


def _log(row: dict[str, Any], name: str) -> None:
    with (Path(row["clip_uri"]).parent.parent / name).open("a") as log:
        log.write(json.dumps(row["clip_id"]) + "\n")


def recording_fetch(batch: pa.Table, *, media: MediaColumns, storage_profile: str, attempts: int) -> pa.Table:
    """Record actual media reads to prove checkpoint filtering happens before download."""
    assert ray.get_runtime_context().get_assigned_resources()[environment.CURATOR_IO_RESOURCE_NAME] == 1
    for row in batch.to_pylist():
        _log(row, "downloads.jsonl")
    return fetch_media_batch(batch, media=media, storage_profile=storage_profile, attempts=attempts)


def _fake_generation(row: dict[str, Any]) -> dict[str, Any]:
    """Return Ray LLM's output shape: a bypassed download error, or postprocessed caption values."""
    _log(row, "inference.jsonl")
    identity = {name: row[name] for name in ("fragment_id", "row_offset", "clip_id")}
    if row["download_error_type"] is not None:
        return identity | {
            "download_error_type": row["download_error_type"],
            "__inference_error__": f"{row['download_error_type']}: {row['download_error_message']}",
        }
    assert row["video_data_url"].startswith("data:video/mp4;base64,")
    return identity | {
        "download_error_type": None,
        "__inference_error__": "",
        "caption_text": f"caption for {row['clip_id']}",
        "prompt_token_count": 4,
        "generated_token_count": 8,
    }


class CpuCaptioner:
    """Stand in for the vLLM processor while keeping its row contract."""

    def __call__(self, dataset: ray.data.Dataset) -> ray.data.Dataset:
        """Caption every fetched clip that has media."""
        return dataset.map(_fake_generation)


def assert_resume(config_json: str) -> None:
    """Resume on a fresh cluster after durable results, including a clip whose media disappeared."""
    config = ResolvedVideoCaptionConfig.model_validate_json(config_json)
    spec = resolve_model_spec(config.model.variant)
    write_clips(config)
    root = Path(config.input.media_root)

    def remove_unfinished_clip() -> None:
        # One unfinished clip disappears: it still needs a terminal row and checkpoint.
        clips = lance.dataset(config.input.clips_lance_uri)
        Path(clips.get_fragment(1).to_table(columns=["clip_uri"]).column(0)[-1].as_py()).unlink()

    def work_logs() -> dict[str, list[str]]:
        return {
            name: [json.loads(line) for line in (root / name).read_text().splitlines()]
            for name in ("downloads.jsonl", "inference.jsonl")
        }

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(inference, "fetch_media_batch", recording_fetch)
        monkeypatch.setattr(inference, "_build_llm_processor", lambda *_args, **_kwargs: CpuCaptioner())
        monkeypatch.setattr(pipeline, "validate_model_directory", lambda _spec: root)
        summary = assert_fresh_cluster_resume(
            RECIPE,
            config,
            process=lambda processing, resolved: pipeline.caption_clips(processing, resolved, spec),
            work_logs=work_logs,
            num_cpus=4,
            before_resume=remove_unfinished_clip,
        )
    published = lance.dataset(config.input.clips_lance_uri)
    assert summary["clips_lance_version"] == published.version
    captions = published.to_table(columns=[spec.caption_field_name, spec.metadata_field_name])
    statuses = [metadata["status"] for metadata in captions.column(1).to_pylist()]
    assert sorted(statuses) == ["error", "success", "success", "success"]
    assert captions.column(0).null_count == 1


@pytest.mark.integration
def test_checkpoint_resume_on_fresh_cluster(config: ResolvedVideoCaptionConfig) -> None:
    """Exercise IO-limited media fetches, the inference stand-in, and checkpoint recovery."""
    run_in_fresh_driver(
        "tests.cosmos_curator.next.recipes.video_caption.test_ray_pipeline:assert_resume", config.model_dump_json()
    )
