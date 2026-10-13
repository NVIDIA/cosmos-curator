# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Embedding plan wiring inside the shared checkpoint scope, without starting Ray or loading a model."""

from unittest.mock import Mock

import pytest
from ray.data import ActorPoolStrategy
from ray.data.context import DataContext

from cosmos_curator.next.enrichment.attempts import reconcile_and_select_attempt
from cosmos_curator.next.enrichment.execution import checkpointed_processing
from cosmos_curator.next.enrichment.inputs import EnrichmentInput
from cosmos_curator.next.recipes.video_embed import pipeline, processing
from cosmos_curator.next.recipes.video_embed.config import ResolvedVideoEmbedConfig
from cosmos_curator.next.recipes.video_embed.contracts import ModelSpec


def test_plan_reads_clip_ids_and_reserves_io_before_inference(
    config: ResolvedVideoEmbedConfig,
    enrichment_input: EnrichmentInput,
    spec: ModelSpec,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The plan reads clip IDs, URIs, and sizes; downloads hold IO slots; GPU actors stay loaded between bursts."""
    context = DataContext.get_current()
    monkeypatch.setattr(context, "checkpoint_config", None)
    plan = Mock()
    datasources = []
    for method in ("map", "map_batches", "repartition"):
        getattr(plan, method).return_value = plan

    def read(datasource: object, **_kwargs: object) -> Mock:
        datasources.append(datasource)
        plan.context = context.copy()
        return plan

    monkeypatch.setattr(pipeline.ray.data, "read_datasource", read)
    monkeypatch.setattr(pipeline, "validate_model_directory", Mock(return_value="/unused-model"))
    with checkpointed_processing(reconcile_and_select_attempt(enrichment_input)) as scope:
        pipeline.embed_clips(scope, config, spec)

    (datasource,) = datasources
    assert datasource._schema.names == ["clip_id", "clip_uri", "clip_size_bytes", "fragment_id", "row_offset"]
    args, kwargs = plan.map.call_args
    assert args == (processing.download_clip,)
    assert kwargs["resources"] == {"curator_io": 1.0}
    inference_args, inference = plan.map_batches.call_args_list[0]
    assert inference_args == (processing.VideoEmbedder,)
    assert isinstance(inference["compute"], ActorPoolStrategy)
    assert inference["num_gpus"] == 1
    assert plan.context.autoscaling_config.actor_pool_util_downscaling_threshold == 0.0
