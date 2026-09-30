# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Method dispatch, source checkpoint scope, and independent download/GPU admission."""

from unittest.mock import Mock

import pytest
from ray.data import ActorPoolStrategy
from ray.data._internal.execution.execution_callback import ExecutionCallback
from ray.data.context import DataContext

from cosmos_curator.next.core.ray_autoscaling import ReleasePendingDownloadActors
from cosmos_curator.next.recipes.video_cut import pipeline, transnetv2
from cosmos_curator.next.recipes.video_cut.config import ResolvedVideoCutConfig, resolve_config_data
from cosmos_curator.next.recipes.video_cut.publication import resolve_workspace


@pytest.mark.parametrize("fail_write", [False, True])
def test_transnetv2_actor_resources_and_checkpoint_scope(
    config: ResolvedVideoCutConfig, snapshot_id: str, monkeypatch: pytest.MonkeyPatch, *, fail_write: bool
) -> None:
    """CPU decode and GPU inference scale separately, without freezing startup resource capacity."""
    raw = config.model_dump(mode="json")
    raw["split"] = {"method": "transnetv2"}
    config = resolve_config_data(raw)
    context = DataContext.get_current()
    monkeypatch.setattr(context, "custom_execution_callback_classes", [ExecutionCallback])
    monkeypatch.setattr(context.autoscaling_config, "actor_pool_util_upscaling_threshold", 1.75)
    plan = Mock()
    for method in ("map", "map_batches", "repartition"):
        getattr(plan, method).return_value = plan

    def read(*_args: object, **_kwargs: object) -> Mock:
        assert context.checkpoint_config.id_column == "source_id"
        plan.context = context.copy()
        return plan

    monkeypatch.setattr(pipeline.ray.data, "read_datasource", read)
    snapshot = Mock(side_effect=AssertionError("Actor pools must not freeze startup resource capacity"))
    monkeypatch.setattr(pipeline.ray, "cluster_resources", snapshot)
    workspace = resolve_workspace(config, snapshot_id)
    if fail_write:
        plan.write_parquet.side_effect = RuntimeError("injected write failure")
        with pytest.raises(RuntimeError, match="injected write failure"):
            pipeline.cut_sources(config, workspace, config.input.uris)
    else:
        pipeline.cut_sources(config, workspace, config.input.uris)
    download_args, download = plan.map.call_args
    assert download_args == (transnetv2.SourceDownloader,)
    assert download["resources"] == {"curator_io": 1.0}
    assert download["num_cpus"] == 0.25
    assert plan.map_batches.call_count == 2
    decode_args, decode = plan.map_batches.call_args_list[0]
    assert decode_args == (transnetv2.decode_sources,)
    assert (decode["num_cpus"], decode["num_gpus"], decode["memory"]) == (3, 0, 2 * 1024**3)
    assert decode["batch_size"] == 1
    assert decode["fn_kwargs"] == {"execution": config.execution}
    assert "resources" not in decode
    assert "compute" not in decode  # Stateless CPU tasks release their reservations after each task.
    inference_args, inference = plan.map_batches.call_args_list[1]
    assert inference_args == (transnetv2.TransNetV2Planner,)
    assert (inference["num_cpus"], inference["num_gpus"]) == (1, 0.25)
    assert inference["batch_size"] == 1
    assert inference["memory"] == 2 * 1024**3
    assert inference["runtime_env"]["env_vars"]["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"] == "0"
    assert "resources" not in inference
    for stage in (download, inference):
        assert isinstance(stage["compute"], ActorPoolStrategy)
        assert stage["compute"].max_size == float("inf")
        assert stage["compute"].max_tasks_in_flight_per_actor == 1
    assert plan.context.autoscaling_config.actor_pool_util_upscaling_threshold == 1.0
    assert plan.context.custom_execution_callback_classes == [ExecutionCallback, ReleasePendingDownloadActors]
    assert context.custom_execution_callback_classes == [ExecutionCallback]
    assert context.autoscaling_config.actor_pool_util_upscaling_threshold == 1.75
    assert context.checkpoint_config is None
    snapshot.assert_not_called()


def test_nvdec_shares_one_gpu_actor_with_inference(config: ResolvedVideoCutConfig) -> None:
    """GPU decode avoids a CPU decode stage and reserves memory for the combined work."""
    raw = config.model_dump(mode="json")
    raw["split"] = {"method": "transnetv2", "decoder": "nvdec"}
    config = resolve_config_data(
        raw,
        overrides=["execution.transnetv2.memory_gib=3", "execution.transnetv2.inference_memory_gib=2"],
    )
    sources = Mock()
    sources.context = DataContext.get_current().copy()
    sources.map.return_value = sources
    pipeline._transnetv2_plan(sources, config)
    sources.map_batches.assert_called_once()
    args, settings = sources.map_batches.call_args
    assert args == (transnetv2.TransNetV2Planner,)
    assert settings["fn_constructor_kwargs"]["split"].decoder == "nvdec"
    assert (settings["num_cpus"], settings["num_gpus"], settings["memory"]) == (1, 0.25, 5 * 1024**3)
    assert settings["compute"].max_size == float("inf")
