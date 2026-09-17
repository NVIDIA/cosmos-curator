# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Download admission and plan-scoped execution settings without starting Ray."""

from unittest.mock import Mock

import lance
import pytest
from ray.data import ActorPoolStrategy
from ray.data.context import DataContext

from cosmos_curator.next.recipes.video_transcode import pipeline, processing
from cosmos_curator.next.recipes.video_transcode.config import ResolvedVideoTranscodeConfig
from cosmos_curator.next.recipes.video_transcode.publication import resolve_workspace


@pytest.mark.parametrize("fail_write", [False, True])
def test_download_admission_is_actor_bound_and_plan_scoped(
    config: ResolvedVideoTranscodeConfig,
    dataset: lance.LanceDataset,
    monkeypatch: pytest.MonkeyPatch,
    *,
    fail_write: bool,
) -> None:
    """Ready actors own the IO budget; neither startup capacity nor ambient settings fix the pool size."""
    context = DataContext.get_current()
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
    workspace = resolve_workspace(config, dataset)
    if fail_write:
        plan.write_parquet.side_effect = RuntimeError("injected write failure")
        with pytest.raises(RuntimeError, match="injected write failure"):
            pipeline.transcode_spans(dataset, config, workspace)
    else:
        pipeline.transcode_spans(dataset, config, workspace)

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
    assert context.autoscaling_config.actor_pool_util_upscaling_threshold == 1.75
    assert context.checkpoint_config is None
    snapshot.assert_not_called()
    assert plan.flat_map.call_args.kwargs["num_cpus"] == config.execution.transcode_cpus
    assert plan.flat_map.call_args.kwargs["memory"] == int(config.execution.transcode_memory_gib * 1024**3)
    assert "resources" not in plan.flat_map.call_args.kwargs
    assert "resources" not in plan.map_batches.call_args.kwargs
