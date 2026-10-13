# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Download admission and plan-scoped execution settings without starting Ray."""

from unittest.mock import Mock

import pytest
from ray.data import ActorPoolStrategy
from ray.data._internal.execution.execution_callback import ExecutionCallback
from ray.data.context import DataContext

from cosmos_curator.next.core.ray_autoscaling import ReleasePendingDownloadActors
from cosmos_curator.next.enrichment import inputs
from cosmos_curator.next.enrichment.attempts import reconcile_and_select_attempt
from cosmos_curator.next.enrichment.execution import checkpointed_processing
from cosmos_curator.next.recipes.video_transcode import pipeline, processing
from cosmos_curator.next.recipes.video_transcode.config import ResolvedVideoTranscodeConfig


def test_download_admission_is_actor_bound_and_plan_scoped(
    config: ResolvedVideoTranscodeConfig,
    enrichment_input: inputs.EnrichmentInput,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ready actors own the IO budget; neither startup capacity nor ambient settings fix the pool size."""
    context = DataContext.get_current()
    monkeypatch.setattr(context, "custom_execution_callback_classes", [ExecutionCallback])
    monkeypatch.setattr(context.autoscaling_config, "actor_pool_util_upscaling_threshold", 1.75)
    plan = Mock()
    for method in ("groupby", "map_groups", "sort", "map", "flat_map", "map_batches", "repartition"):
        getattr(plan, method).return_value = plan

    def read(*_args: object, **_kwargs: object) -> Mock:
        plan.context = context.copy()
        return plan

    monkeypatch.setattr(pipeline.ray.data, "read_datasource", read)
    snapshot = Mock(side_effect=AssertionError("Download admission must not snapshot cluster resources"))
    monkeypatch.setattr(pipeline.ray, "cluster_resources", snapshot)
    with checkpointed_processing(reconcile_and_select_attempt(enrichment_input)) as scope:
        pipeline.transcode_spans(scope, config)

    plan.map.assert_called_once()
    args, kwargs = plan.map.call_args
    assert args == (processing.SourceDownloader,)
    assert kwargs["fn_constructor_kwargs"] == {"config": config}
    assert kwargs["num_cpus"] == 0.25
    assert kwargs["resources"] == {"curator_io": 1.0}
    assert kwargs["max_concurrency"] == 1
    assert kwargs["max_restarts"] == kwargs["max_task_retries"] == -1
    strategy = kwargs["compute"]
    assert isinstance(strategy, ActorPoolStrategy)
    assert strategy.min_size == strategy.initial_size == 1
    assert strategy.max_size == float("inf")
    assert strategy.max_tasks_in_flight_per_actor == 1
    assert plan.context.autoscaling_config.actor_pool_util_upscaling_threshold == 1.0
    assert plan.context.custom_execution_callback_classes == [ExecutionCallback, ReleasePendingDownloadActors]
    assert context.custom_execution_callback_classes == [ExecutionCallback]
    assert context.autoscaling_config.actor_pool_util_upscaling_threshold == 1.75
    snapshot.assert_not_called()
    assert plan.flat_map.call_args.kwargs["num_cpus"] == config.execution.transcode_cpus
    assert plan.flat_map.call_args.kwargs["memory"] == int(config.execution.transcode_memory_gib * 1024**3)
    assert "resources" not in plan.flat_map.call_args.kwargs
    assert "resources" not in plan.map_batches.call_args_list[0].kwargs
