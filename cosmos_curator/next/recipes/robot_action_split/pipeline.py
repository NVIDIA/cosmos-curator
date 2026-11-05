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

"""Discovered chunks -> checkpointed chunk outcomes -> packed clip rows in one Lance append.

Shared append publication owns selection, checkpoints, and recovery (see
``docs/curator/design/curator-next-robot-action-split.md``, "Cross-Run Recovery").
Run through the generic pipeline CLI with a ``robot-action-split`` config.
"""

import json
from collections import Counter
from functools import partial
from typing import Any, cast

from loguru import logger

from cosmos_curator.next.append.contracts import AppendBindings
from cosmos_curator.next.append.execution import AppendOutcome, ProcessingScope, run_append
from cosmos_curator.next.core.ray_runtime import (
    configure_ray_data_progress,
    configure_ray_data_stability,
    ensure_ray_initialized,
)
from cosmos_curator.next.recipes.robot_action_split.bindings import resolve_bindings
from cosmos_curator.next.recipes.robot_action_split.config import ResolvedRobotActionSplitConfig
from cosmos_curator.next.recipes.robot_action_split.discovery import discover_items
from cosmos_curator.next.recipes.robot_action_split.processing import cut_chunks
from cosmos_curator.next.utils.arrow_utils import s3_compatibility_env_vars


def start_ray(bindings: AppendBindings, config: ResolvedRobotActionSplitConfig) -> None:
    """Start Ray with the storage environment for the outputs and the source datasets."""
    # Each location needs the environment its own profile's endpoint asks for; the union cannot conflict.
    sources = s3_compatibility_env_vars(*config.input.uris, storage_profile=config.input.storage_profile)
    ensure_ray_initialized(runtime_env_vars=bindings.storage_env_vars(config.output.media_root) | sources)
    configure_ray_data_progress(progress=config.execution.progress)
    configure_ray_data_stability(no_progress_timeout_s=config.execution.no_progress_timeout_s)


def cut_clips(processing: ProcessingScope, config: ResolvedRobotActionSplitConfig) -> None:
    """Cut each selected chunk completely in one task, filtering checkpointed chunks before any download."""
    # Reading with the cut task's CPUs lets Ray run each chunk's read and cut as one task.
    chunks = processing.read(num_cpus=config.execution.cut_cpus)
    outcomes = chunks.map_batches(
        cast("Any", cut_chunks),
        batch_size=None,
        batch_format="pyarrow",
        fn_kwargs={"config": config},
        num_cpus=config.execution.cut_cpus,
    )
    processing.write(outcomes)


def failed_clip_stages(outcome: AppendOutcome) -> dict[str, int]:
    """Count this run's published failed clips by ``error_stage``, reading only the fragments it appended.

    The shared summary's ``errors`` counts chunks that failed as a whole; a clip
    can also fail inside a finished chunk, and both are permanent for this table.
    """
    before = {fragment.fragment_id for fragment in outcome.attempt.dataset.get_fragments()}
    appended = [fragment for fragment in outcome.dataset.get_fragments() if fragment.fragment_id not in before]
    if not appended:
        return {}
    stages: Counter[str] = Counter()
    scanner = outcome.dataset.scanner(columns=["error_stage"], filter="error_stage IS NOT NULL", fragments=appended)
    for batch in scanner.to_batches():
        stages.update(batch.column("error_stage").to_pylist())
    return dict(sorted(stages.items()))


def run_config(config: ResolvedRobotActionSplitConfig) -> dict[str, Any]:
    """Append newly discovered chunks' clips, resuming a saved selection first, and clean verified recovery state."""
    logger.info(f"robot-action-split: source_dataset={config.input.source_dataset} lance_uri={config.output.lance_uri}")
    outcome = run_append(
        resolve_bindings(config),
        partial(discover_items, config),
        start_ray=partial(start_ray, config=config),
        process=partial(cut_clips, config=config),
    )
    failed_clips = failed_clip_stages(outcome)
    summary: dict[str, Any] = {
        **outcome.summary(),
        "failed_clips": sum(failed_clips.values()),
        "failed_clips_by_stage": failed_clips,
        "clips_lance_uri": config.output.lance_uri,
        "clips_lance_version": int(outcome.dataset.version),
    }
    logger.info("robot-action-split: {}", json.dumps(summary, sort_keys=True))
    return summary
