# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real local Ray execution with fresh processes and native chunk-checkpoint recovery."""

from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import pyarrow as pa
import pytest

from cosmos_curator.next.recipes.robot_action_split import pipeline, processing
from cosmos_curator.next.recipes.robot_action_split.config import ResolvedRobotActionSplitConfig
from tests.cosmos_curator.next.append.fresh_cluster import assert_fresh_cluster_resume
from tests.cosmos_curator.next.fresh_driver import run_in_fresh_driver
from tests.cosmos_curator.next.recipes.robot_action_split.conftest import RECIPE

_MODULE = "tests.cosmos_curator.next.recipes.robot_action_split.test_ray_recovery"


def _cut(
    _source: str, cuts: list[dict[str, Any]], *, on_cut: Callable[[dict[str, Any]], None], **_kwargs: object
) -> list[dict[str, Any]]:
    """Stand in for ``cut_plan`` over placeholder media: write each cut's bytes and report it exact."""
    records = []
    for cut in cuts:
        Path(cut["output"]).write_bytes(b"clip bytes")
        record = {"output": cut["output"], "success": True, "expected_frames": cut["endFrame"] - cut["startFrame"] + 1}
        on_cut(record)
        records.append(record)
    return records


def _cut_and_log(batch: pa.Table, *, config: ResolvedRobotActionSplitConfig, log: Path) -> pa.Table:
    """Record each chunk a worker processes, then run the real chunk processing with only the cut replaced."""
    with log.open("a") as stream:
        stream.writelines(f"{chunk_id}\n" for chunk_id in batch["chunk_id"].to_pylist())
    processing.cut_plan = _cut
    return processing.cut_chunks(batch, config=config)


def assert_chunk_resume(config_json: str, log_dir: str) -> None:
    """Run the shared fresh-cluster scenario with real chunk processing over placeholder media."""
    config = ResolvedRobotActionSplitConfig.model_validate_json(config_json)
    log = Path(log_dir, "cut.txt")
    pipeline.cut_chunks = partial(_cut_and_log, log=log)  # type: ignore[assignment]
    summary = assert_fresh_cluster_resume(
        RECIPE,
        config,
        process=pipeline.cut_clips,
        work_logs=lambda: {"cut": log.read_text().splitlines() if log.exists() else []},
        num_cpus=4,
    )
    # Three views of two clips each; the wrist view's chunk MP4 is missing.
    assert (summary["appended_rows"], summary["errors"]) == (6, 1)


@pytest.mark.integration
def test_fresh_cluster_resumes_chunk_checkpoints(config: ResolvedRobotActionSplitConfig, tmp_path: Path) -> None:
    """The real plan filters checkpointed chunks on a fresh cluster and publishes through ``run_config``."""
    wrist_chunk = next(Path(config.input.uris[0]).glob("dataset/videos/*wrist/chunk-000/file-000.mp4"))
    wrist_chunk.unlink()
    run_in_fresh_driver(f"{_MODULE}:assert_chunk_resume", config.model_dump_json(), str(tmp_path))
