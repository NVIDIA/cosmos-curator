# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Video-cut runs: the span-table summary, method prerequisites inside processing, and earlier table formats.

Shared publication and recovery behavior is covered by ``test_conformance.py``, and each cutting
method's stages by its own test module.
"""

import json
from functools import partial
from unittest.mock import Mock

import lance
import pyarrow as pa
import pytest

from cosmos_curator.next.recipes.video_cut import pipeline, transnetv2
from cosmos_curator.next.recipes.video_cut.bindings import resolve_bindings
from cosmos_curator.next.recipes.video_cut.config import ResolvedVideoCutConfig, resolve_config_data
from cosmos_curator.next.recipes.video_cut.contracts import SPAN_SCHEMA, make_source_id, source_result
from cosmos_curator.next.recipes.video_cut.discovery import discover_items
from tests.cosmos_curator.next.append.helpers import replace_ray, select, stage_results
from tests.cosmos_curator.next.recipes.video_cut.conftest import probed_source


def test_summary_adds_the_span_table(
    config: ResolvedVideoCutConfig, results: pa.Table, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The shared summary gains the span table's location and version without counting its rows."""
    replace_ray(pipeline, monkeypatch, results=results)
    monkeypatch.setattr(
        lance.LanceDataset, "count_rows", Mock(side_effect=AssertionError("Summary must not scan rows"))
    )
    summary = pipeline.run_config(config)
    dataset = lance.dataset(config.output.spans_lance_uri)
    assert summary == {
        "input_items": 4,
        "selected_items": 4,
        "remaining_items": 0,
        "appended_rows": 8,
        "errors": 1,
        "cleanup_complete": True,
        "contract_digest": resolve_bindings(config).spec.contract_digest,
        "spans_lance_uri": config.output.spans_lance_uri,
        "spans_lance_version": int(dataset.version),
    }
    (reported,) = dataset.to_table(
        filter="error_stage IS NOT NULL", columns=["source_id", "source_uri", "error_stage", "error_message"]
    ).to_pylist()
    assert reported == {
        "source_id": make_source_id("s3://test-bucket/error.mp4"),
        "source_uri": "s3://test-bucket/error.mp4",
        "error_stage": "source-read",
        "error_message": "synthetic missing source",
    }


def test_completed_transnetv2_plans_publish_without_weights(
    config: ResolvedVideoCutConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fresh driver publishes completed plans without inference prerequisites."""
    settings = config.model_dump(mode="json")
    settings["split"] = {"method": "transnetv2"}
    config = resolve_config_data(settings)
    bindings = resolve_bindings(config)
    assert config.input.uris is not None
    attempt = select(bindings, discover_items(config))
    results = pa.Table.from_pylist(
        [source_result(probed_source(uri, 100), []) for uri in config.input.uris], schema=bindings.spec.result_fields
    )
    stage_results(attempt, results)
    weights = Mock(side_effect=FileNotFoundError("weights are not mounted on this driver"))
    monkeypatch.setattr(transnetv2, "validate_weights", weights)
    monkeypatch.setattr(pipeline, "ensure_ray_initialized", Mock())
    replace_ray(pipeline, monkeypatch, results=None, start_ray=partial(pipeline.start_ray, config=config))

    summary = pipeline.run_config(config)

    assert summary["selected_items"] == len(config.input.uris)
    assert summary["cleanup_complete"] is True
    weights.assert_not_called()


def test_tables_from_earlier_formats_are_rejected_without_mutation(
    config: ResolvedVideoCutConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tables without the shared contract binding need a new output location; no migration exists."""
    legacy_schema = SPAN_SCHEMA.with_metadata({b"video_cut.contract": json.dumps(config.split.cutting_contract)})
    legacy = lance.write_dataset(
        legacy_schema.empty_table(), config.output.spans_lance_uri, schema=legacy_schema, data_storage_version="2.2"
    )
    replace_ray(pipeline, monkeypatch, results=None, start_ray=Mock(side_effect=AssertionError("Ray started")))
    with pytest.raises(
        ValueError, match="different producer, item ID column, or result contract; use a new output location"
    ):
        pipeline.run_config(config)
    assert lance.dataset(config.output.spans_lance_uri).version == legacy.version
