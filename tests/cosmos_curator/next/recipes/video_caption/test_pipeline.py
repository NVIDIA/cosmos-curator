# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Caption plan wiring, clip readiness, and publication, without starting Ray or loading a model."""

from functools import partial
from pathlib import Path
from typing import Any
from unittest.mock import Mock, call

import lance
import pyarrow as pa
import pyarrow.compute as pc
import pytest
from ray.data.context import DataContext

from cosmos_curator.next.core import recovery_bindings
from cosmos_curator.next.enrichment import execution, inputs
from cosmos_curator.next.enrichment.attempts import reconcile_and_select_attempt
from cosmos_curator.next.enrichment.execution import checkpointed_processing
from cosmos_curator.next.recipes.video_caption import inference, pipeline
from cosmos_curator.next.recipes.video_caption.bindings import resolve_bindings
from cosmos_curator.next.recipes.video_caption.config import ResolvedVideoCaptionConfig
from cosmos_curator.next.recipes.video_caption.contracts import CaptionModelSpec
from cosmos_curator.next.recipes.video_transcode.contracts import CLIP_SCHEMA
from tests.cosmos_curator.next.enrichment.helpers import prepare_results
from tests.cosmos_curator.next.recipes.video_caption.conftest import clip_rows, write_clips


def test_plan_reserves_io_for_fetches_and_bounds_replicas_by_the_selection(
    config: ResolvedVideoCaptionConfig,
    enrichment_input: inputs.EnrichmentInput,
    caption_spec: CaptionModelSpec,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The plan reads clip IDs, URIs, and sizes; fixed-size fetches hold IO slots; the vLLM pool follows the work."""
    monkeypatch.setattr(DataContext.get_current(), "checkpoint_config", None)
    plan = Mock()
    for method in ("map", "map_batches", "repartition"):
        getattr(plan, method).return_value = plan
    datasources = []

    def read(datasource: object, **_kwargs: object) -> Mock:
        datasources.append(datasource)
        return plan

    built: list[dict[str, Any]] = []

    def build_processor(*args: object, **kwargs: object) -> Mock:
        built.append({"args": args, **kwargs})
        return Mock(return_value=plan)

    eager = Mock()
    monkeypatch.setattr(execution.ray.data, "read_datasource", read)
    monkeypatch.setattr(pipeline, "validate_model_directory", Mock(return_value=Path("/unused-model")))
    monkeypatch.setattr(pipeline, "configure_ray_data_eager_actor_autoscaling", eager)
    monkeypatch.setattr(inference.ray, "available_resources", lambda: {"GPU": 2.0})
    monkeypatch.setattr(inference, "_build_llm_processor", build_processor)
    with checkpointed_processing(reconcile_and_select_attempt(enrichment_input)) as scope:
        pipeline.caption_clips(scope, config, caption_spec)

    eager.assert_called_once_with()
    (datasource,) = datasources
    assert datasource._schema.names == ["clip_id", "clip_uri", "clip_size_bytes", "fragment_id", "row_offset"]
    fixed_blocks, first_fetch = plan.mock_calls[:2]
    assert fixed_blocks == call.repartition(
        target_num_rows_per_block=config.execution.lance_read_batch_size, strict=True
    )
    assert first_fetch.args == (inference.fetch_media_batch,)
    (fetch_args, fetch), (canonical_args, canonical), _stage = plan.map_batches.call_args_list
    assert fetch_args == (inference.fetch_media_batch,)
    assert fetch["resources"] == {"curator_io": 1.0}
    assert fetch["fn_kwargs"]["media"] == inference.MediaColumns(uri="clip_uri", size="clip_size_bytes")
    assert canonical_args == (inference.canonicalize_llm_batch,)
    request = canonical["fn_kwargs"]["request"]
    assert request.identity_columns == ("clip_id",)
    assert request.result_schema.equals(enrichment_input.bindings.spec.result_schema, check_metadata=True)
    (processor,) = built
    assert processor["args"][1:] == (Path("/unused-model"), ("clip_id",))
    # Four selected clips bound the pool; two live GPUs seed it.
    assert (processor["initial_inference_workers"], processor["inference_worker_ceiling"]) == (2, 4)


def test_ray_workers_reach_media_clips_and_staging(
    config: ResolvedVideoCaptionConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each location is configured on its own, so workers get settings for every one they open."""
    locations: list[tuple[str, ...]] = []

    def classify(*urls: str, storage_profile: str) -> dict[str, str]:
        assert storage_profile == config.execution.storage_profile
        locations.append(urls)
        return {"AWS_RESPONSE_CHECKSUM_VALIDATION": "when_required"}

    initialized: list[dict[str, object]] = []
    monkeypatch.setattr(recovery_bindings, "s3_compatibility_env_vars", classify)
    monkeypatch.setattr(pipeline, "ensure_ray_initialized", lambda **kwargs: initialized.append(kwargs))
    monkeypatch.setattr(pipeline, "curator_io_resources", lambda: {"curator_io": 16.0})
    monkeypatch.setattr(pipeline, "configure_ray_data_progress", Mock())
    monkeypatch.setattr(pipeline, "configure_ray_data_stability", Mock())

    pipeline.start_ray(resolve_bindings(config), config)

    assert locations == [
        (config.input.clips_lance_uri, config.output.staging_root_uri, config.input.media_root),
    ]
    assert initialized == [
        {
            "local_resources": {"curator_io": 16.0},
            "runtime_env_vars": {"AWS_RESPONSE_CHECKSUM_VALIDATION": "when_required"},
        }
    ]


def test_duplicate_clip_ids_are_rejected_before_any_work(config: ResolvedVideoCaptionConfig) -> None:
    """Checkpoints key on clip IDs, so a duplicate would silently skip a clip."""
    rows = clip_rows(config, count=2)
    rows[1]["clip_id"] = rows[0]["clip_id"]
    lance.write_dataset(
        pa.Table.from_pylist(rows, schema=CLIP_SCHEMA), config.input.clips_lance_uri, data_storage_version="2.2"
    )

    with pytest.raises(ValueError, match="globally unique"):
        pipeline.run_config(config)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("clip_size_bytes", 1, None),
        ("clip_id", None, "non-null clip_id"),
        ("clip_uri", None, "non-null URIs"),
        ("clip_size_bytes", None, "positive byte sizes"),
        ("clip_size_bytes", 0, "positive byte sizes"),
        ("clip_size_bytes", -1, "positive byte sizes"),
    ],
)
def test_appended_spans_need_ready_clips(
    config: ResolvedVideoCaptionConfig, field: str, value: int | None, message: str | None
) -> None:
    """New spans must have usable clips; readiness depends on current values, not publication history."""
    dataset = write_clips(config)
    row = clip_rows(config, count=1)[0] | {"span_id": "appended-span", "clip_id": "appended-clip", field: value}
    lance.write_dataset(
        pa.Table.from_pylist([row], schema=dataset.schema), dataset.uri, mode="append", data_storage_version="2.2"
    )
    bindings = resolve_bindings(config)
    if message is None:
        inputs.open_enrichment_input(bindings)
    else:
        with pytest.raises(ValueError, match=message):
            inputs.open_enrichment_input(bindings)


def test_run_publishes_captions_and_reports_the_clip_table(
    config: ResolvedVideoCaptionConfig,
    results: pa.Table,
    caption_spec: CaptionModelSpec,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Terminal captions, truncations, and media errors become visible together in the clip table."""
    monkeypatch.setattr(pipeline, "start_ray", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(pipeline, "caption_clips", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(execution, "prepare_enrichment_fragments", partial(prepare_results, results))

    summary = pipeline.run_config(config)

    published = lance.dataset(config.input.clips_lance_uri)
    assert (summary["clips_lance_uri"], summary["clips_lance_version"]) == (
        config.input.clips_lance_uri,
        published.version,
    )
    assert (summary["selected_rows"], summary["field_set"]) == (4, caption_spec.caption_field_name)
    statuses = [
        metadata["status"]
        for metadata in published.to_table(columns=[caption_spec.metadata_field_name]).column(0).to_pylist()
    ]
    assert sorted(statuses) == ["error", "success", "success", "truncated"]


def test_captions_from_per_fragment_publication_remain_complete(
    config: ResolvedVideoCaptionConfig,
    enrichment_input: inputs.EnrichmentInput,
    results: pa.Table,
    caption_spec: CaptionModelSpec,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fields and digests are unchanged, so tables captioned one fragment at a time need no new work."""
    dataset = enrichment_input.dataset
    for fragment in dataset.get_fragments():
        values = results.filter(pc.equal(results["fragment_id"], fragment.fragment_id))
        update = values.select(["clip_id", *caption_spec.fields.names]).cast(
            pa.schema([CLIP_SCHEMA.field("clip_id"), *caption_spec.fields])
        )
        current = lance.dataset(dataset.uri)
        metadata, modified = current.get_fragment(fragment.fragment_id).update_columns(
            update, left_on="clip_id", right_on="clip_id"
        )
        lance.LanceDataset.commit(
            dataset.uri,
            lance.LanceOperation.Update(updated_fragments=[metadata], fields_modified=modified),
            read_version=current.version,
        )
    monkeypatch.setattr(pipeline, "start_ray", Mock(side_effect=AssertionError("Ray must not start")))

    summary = pipeline.run_config(config)

    assert (summary["selected_fragments"], summary["remaining_fragments"]) == (0, 0)
    assert summary["clips_lance_version"] == lance.dataset(dataset.uri).version
