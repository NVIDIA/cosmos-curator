# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Transcode provenance, downstream receipts, and clip validation.

Shared recovery behavior is covered by ``test_conformance.py``.
"""

from functools import partial
from pathlib import Path
from unittest.mock import Mock

import lance
import pyarrow as pa
import pytest

from cosmos_curator.core.utils.misc import retry_utils
from cosmos_curator.next.enrichment import execution, inputs
from cosmos_curator.next.enrichment.attempts import reconcile_and_select_attempt
from cosmos_curator.next.recipes.video_caption.contracts import caption_contract_digest, resolve_model_spec
from cosmos_curator.next.recipes.video_caption.lance_state import ensure_caption_fields, validate_caption_input
from cosmos_curator.next.recipes.video_cut.contracts import SPAN_SCHEMA
from cosmos_curator.next.recipes.video_cut.publication import completed_publication as completed_cut_publication
from cosmos_curator.next.recipes.video_transcode import pipeline
from cosmos_curator.next.recipes.video_transcode.bindings import resolve_bindings
from cosmos_curator.next.recipes.video_transcode.config import ResolvedVideoTranscodeConfig
from cosmos_curator.next.recipes.video_transcode.contracts import (
    completed_publication,
    make_clip_id,
    validate_clip_table,
)
from cosmos_curator.next.recipes.video_transcode.pipeline import publication_properties
from tests.cosmos_curator.next.enrichment.helpers import prepare_results, publish_results
from tests.cosmos_curator.next.recipes.video_caption.helpers import caption_run_bindings


def _append_span(dataset: lance.LanceDataset) -> lance.LanceDataset:
    row = dataset.to_table(columns=SPAN_SCHEMA.names).slice(0, 1).to_pylist()[0]
    row["span_id"] = "new-span"
    return lance.write_dataset(
        pa.Table.from_pylist([row], schema=SPAN_SCHEMA), dataset.uri, mode="append", data_storage_version="2.2"
    )


def test_publication_feeds_caption(
    enrichment_input: inputs.EnrichmentInput, dataset: lance.LanceDataset, results: pa.Table
) -> None:
    """The transcode receipt describes the published spans, and caption accepts the output."""
    published = publish_results(results, enrichment_input, properties=publication_properties)
    record = completed_publication(published)
    assert record is not None
    assert (record["spans"], record["fragment_ids"], record["source_version"]) == (4, [0, 1], dataset.version)
    validate_caption_input(published, uri=dataset.uri)
    spec = resolve_model_spec("qwen3_8_27b_fp8")
    captioned = ensure_caption_fields(
        dataset.uri,
        storage_options=None,
        bindings=caption_run_bindings(spec, caption_contract_digest(spec)),
        attempts=1,
    )
    assert inputs.open_enrichment_input(enrichment_input.bindings).dataset.version == captioned.version


def test_next_run_fills_only_appended_fragments(
    config: ResolvedVideoTranscodeConfig, enrichment_input: inputs.EnrichmentInput, results: pa.Table
) -> None:
    """Incremental publication retains completed files and counts every published span."""
    first = publish_results(results, enrichment_input, properties=publication_properties)
    files = [fragment.metadata.to_json() for fragment in first.get_fragments()]
    _append_span(first)
    clip_id = make_clip_id("new-span", config.transcode.contract_id)
    row = results.slice(0, 1).to_pylist()[0] | {
        "span_id": "new-span",
        "clip_id": clip_id,
        "clip_uri": f"{config.input.media_root}/clips/{clip_id}.mp4",
        "fragment_id": 2,
        "row_offset": 0,
    }
    current = inputs.open_enrichment_input(enrichment_input.bindings)
    published = publish_results(
        pa.Table.from_pylist([row], schema=results.schema), current, properties=publication_properties
    )
    assert inputs.pending_fragment_ids(published, enrichment_input.bindings) == ()
    assert [fragment.metadata.to_json() for fragment in published.get_fragments()[:2]] == files
    assert completed_publication(published)["spans"] == 5


@pytest.mark.parametrize("corruption", ["wrong-contract", "wrong-uri"])
def test_results_must_match_the_encoding_and_destination(
    enrichment_input: inputs.EnrichmentInput, dataset: lance.LanceDataset, results: pa.Table, corruption: str
) -> None:
    """Clip metadata from another contract or media root cannot be published."""
    rows = results.to_pylist()
    rows[0]["transcode_contract_id" if corruption == "wrong-contract" else "clip_uri"] = "wrong"
    with pytest.raises(ValueError, match="Invalid completed transcode result"):
        publish_results(pa.Table.from_pylist(rows, schema=results.schema), enrichment_input)
    assert lance.dataset(dataset.uri).version == dataset.version


def test_input_accepts_committed_fragments_without_whole_table_completion(
    config: ResolvedVideoTranscodeConfig, dataset: lance.LanceDataset
) -> None:
    """Committed fragments are independently ready even while cutting continues."""
    assert completed_cut_publication(dataset) is None
    bindings = resolve_bindings(config)
    current = inputs.open_enrichment_input(bindings).dataset
    assert inputs.pending_fragment_ids(current, bindings) == (0, 1)


def test_resumed_selection_leaves_appends_out_of_the_downstream_receipt(
    config: ResolvedVideoTranscodeConfig,
    enrichment_input: inputs.EnrichmentInput,
    results: pa.Table,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Finishing a frozen selection cannot advertise newly appended spans as usable clips."""
    attempt = reconcile_and_select_attempt(enrichment_input)
    assert attempt.workspace is not None
    attempt.workspace.record_completion(storage_profile=config.execution.storage_profile)
    _append_span(enrichment_input.dataset)
    monkeypatch.setattr(pipeline, "start_ray", lambda _bindings, **_kwargs: None)
    monkeypatch.setattr(pipeline, "transcode_spans", Mock(side_effect=AssertionError("Durable media must be reused")))
    monkeypatch.setattr(execution, "prepare_enrichment_fragments", partial(prepare_results, results))
    summary = pipeline.run_config(config)
    current = lance.dataset(enrichment_input.dataset.uri)
    assert summary["remaining_rows"] == 1
    assert completed_publication(current)["spans"] == 4
    with pytest.raises(ValueError, match="changed after video-transcode publication"):
        validate_clip_table(current, uri=current.uri)


def test_empty_input_publishes_without_ray(
    config: ResolvedVideoTranscodeConfig, dataset: lance.LanceDataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An empty input still gets a durable transcode completion record, once."""
    lance.write_dataset(
        pa.Table.from_pylist([], schema=dataset.schema),
        dataset.uri,
        mode="overwrite",
        data_storage_version="2.2",
    )
    monkeypatch.setattr(pipeline, "start_ray", lambda *_args, **_kwargs: pytest.fail("Ray should not start"))
    result = pipeline.run_config(config)
    assert result["input_rows"] == 0
    assert result["selected_fragments"] == result["remaining_fragments"] == 0
    assert completed_publication(lance.dataset(dataset.uri))["spans"] == 0
    assert pipeline.run_config(config)["clips_lance_version"] == result["clips_lance_version"]


def test_lost_empty_receipt_response_is_not_recorded_twice(
    config: ResolvedVideoTranscodeConfig,
    dataset: lance.LanceDataset,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The zero-span receipt needs no recovery workspace: a retry finds the committed receipt."""
    empty = lance.write_dataset(
        pa.Table.from_pylist([], schema=dataset.schema),
        dataset.uri,
        mode="overwrite",
        data_storage_version="2.2",
    )
    lance_commit = lance.LanceDataset.commit
    commits = []

    def lose_response(*args: object, **kwargs: object) -> None:
        commits.append(lance_commit(*args, **kwargs))
        message = "lost response"
        raise OSError(message)

    monkeypatch.setattr(lance.LanceDataset, "commit", lose_response)
    monkeypatch.setattr(retry_utils.time, "sleep", lambda _seconds: None)
    retry_logger = Mock()
    monkeypatch.setattr(retry_utils, "logger", retry_logger)
    monkeypatch.setattr(pipeline, "start_ray", lambda *_args, **_kwargs: pytest.fail("Ray should not start"))

    summary = pipeline.run_config(config)

    published = lance.dataset(dataset.uri)
    assert len(commits) == 1
    assert summary["clips_lance_version"] == published.version == empty.version + 1
    assert summary["cleanup_complete"] is True
    assert (completed_publication(published)["spans"], completed_publication(published)["fragment_ids"]) == (0, [])
    assert not Path(config.output.staging_root_uri).exists()
    (warning,) = retry_logger.warning.call_args_list
    assert "lost response" in warning.args[0]
