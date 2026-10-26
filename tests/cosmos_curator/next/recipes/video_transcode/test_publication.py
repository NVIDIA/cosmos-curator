# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Transcode publication and downstream clip validation.

Shared recovery behavior is covered by ``test_conformance.py``.
"""

from datetime import timedelta
from functools import partial
from pathlib import Path
from unittest.mock import Mock

import lance
import pyarrow as pa
import pytest

from cosmos_curator.next.enrichment import execution, inputs
from cosmos_curator.next.enrichment.attempts import reconcile_and_select_attempt
from cosmos_curator.next.recipes.video_caption.contracts import caption_contract_digest, resolve_model_spec
from cosmos_curator.next.recipes.video_caption.lance_state import ensure_caption_fields, validate_caption_input
from cosmos_curator.next.recipes.video_cut.contracts import SPAN_SCHEMA
from cosmos_curator.next.recipes.video_transcode import pipeline
from cosmos_curator.next.recipes.video_transcode.bindings import resolve_bindings
from cosmos_curator.next.recipes.video_transcode.config import ResolvedVideoTranscodeConfig
from cosmos_curator.next.recipes.video_transcode.contracts import make_clip_id
from tests.cosmos_curator.next.enrichment.helpers import prepare_results, publish_attempt, publish_results
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
    """Caption accepts published clip values without a recipe-specific receipt."""
    published = publish_results(results, enrichment_input)
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
    """Incremental publication retains completed files and makes every span ready for captioning."""
    first = publish_results(results, enrichment_input)
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
    published = publish_results(pa.Table.from_pylist([row], schema=results.schema), current)
    assert inputs.pending_fragment_ids(published, enrichment_input.bindings) == ()
    assert [fragment.metadata.to_json() for fragment in published.get_fragments()[:2]] == files
    validate_caption_input(published, uri=published.uri)


def test_caption_readiness_survives_pruned_transcode_history(
    enrichment_input: inputs.EnrichmentInput, results: pa.Table
) -> None:
    """Current clip values remain usable after their upstream publication history is pruned."""
    published = publish_results(results, enrichment_input)
    published.add_columns(pa.schema([pa.field("unrelated", pa.string())]))
    latest = lance.dataset(published.uri)
    latest.cleanup_old_versions(older_than=timedelta(seconds=0))
    assert [version["version"] for version in latest.versions()] == [latest.version]
    validate_caption_input(latest, uri=latest.uri)


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
    bindings = resolve_bindings(config)
    current = inputs.open_enrichment_input(bindings).dataset
    assert current.version == dataset.version
    assert inputs.pending_fragment_ids(current, bindings) == (0, 1)


@pytest.mark.parametrize("change", ["encoding", "media_root"])
def test_existing_clips_reject_a_different_contract_without_a_recipe_receipt(
    config: ResolvedVideoTranscodeConfig,
    enrichment_input: inputs.EnrichmentInput,
    results: pa.Table,
    monkeypatch: pytest.MonkeyPatch,
    change: str,
) -> None:
    """Shared terminal validation protects completed clips before processing new work."""
    published = publish_results(results, enrichment_input)
    _append_span(published)
    if change == "encoding":
        config = config.model_copy(update={"transcode": config.transcode.model_copy(update={"video_bitrate": "8M"})})
    else:
        config = config.model_copy(
            update={"input": config.input.model_copy(update={"media_root": f"{config.input.media_root}/different"})}
        )
    monkeypatch.setattr(pipeline, "start_ray", Mock(side_effect=AssertionError("Must reject before starting Ray")))
    version = lance.dataset(published.uri).version
    with pytest.raises(ValueError, match="Invalid completed transcode result"):
        pipeline.run_config(config)
    assert lance.dataset(published.uri).version == version


def test_resumed_selection_leaves_appended_spans_unready_for_caption(
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
    with pytest.raises(ValueError, match="non-null and globally unique clip_id"):
        validate_caption_input(current, uri=current.uri)


def test_empty_input_needs_no_ray_or_completion_commit(
    config: ResolvedVideoTranscodeConfig, dataset: lance.LanceDataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An empty table with registered clip fields is immediately ready for captioning."""
    empty = lance.write_dataset(
        pa.Table.from_pylist([], schema=dataset.schema),
        dataset.uri,
        mode="overwrite",
        data_storage_version="2.2",
    )
    monkeypatch.setattr(pipeline, "start_ray", lambda *_args, **_kwargs: pytest.fail("Ray should not start"))
    result = pipeline.run_config(config)
    assert result["input_rows"] == 0
    assert result["selected_fragments"] == result["remaining_fragments"] == 0
    assert result["clips_lance_version"] == empty.version
    assert pipeline.run_config(config)["clips_lance_version"] == result["clips_lance_version"]
    validate_caption_input(empty, uri=empty.uri)
    assert not Path(config.output.staging_root_uri).exists()


def test_outcome_rows_are_skipped_across_fragment_batches(
    config: ResolvedVideoTranscodeConfig,
    dataset: lance.LanceDataset,
    results: pa.Table,
) -> None:
    """Mixed and outcome-only fragments stay intact while one fragment of spans is enriched per run."""
    spans = dataset.to_table(columns=SPAN_SCHEMA.names).to_pylist()
    empty = spans[0] | {"source_id": "empty", "span_id": None, "start_ns": None, "end_ns": None}
    error = empty | {
        "source_id": "error",
        "error_stage": "source-read",
        "error_message": "missing",
    }
    source_rows = [empty, spans[0], error, *spans[1:], empty | {"source_id": "another-empty"}]
    rewritten = lance.write_dataset(
        pa.Table.from_pylist(source_rows, schema=SPAN_SCHEMA),
        dataset.uri,
        mode="overwrite",
        max_rows_per_file=3,
        data_storage_version="2.2",
    )
    config = config.model_copy(update={"execution": config.execution.model_copy(update={"max_fragments": 1})})
    bindings = resolve_bindings(config)
    first, second, _ = [fragment.fragment_id for fragment in rewritten.get_fragments()]
    addresses = [(first, 1), (second, 0), (second, 1), (second, 2)]
    staged = pa.Table.from_pylist(
        [
            row | {"fragment_id": fragment_id, "row_offset": offset}
            for row, (fragment_id, offset) in zip(results.to_pylist(), addresses, strict=True)
        ],
        schema=results.schema,
    )
    for fragment_id in (first, second):
        attempt = reconcile_and_select_attempt(inputs.open_enrichment_input(bindings))
        assert attempt.selected == (fragment_id,)
        source = execution.selection_datasource(attempt, columns=[])
        read = pa.concat_tables([batch for task in source.get_read_tasks(3) for batch in task()])
        assert read["row_offset"].to_pylist() == [offset for fid, offset in addresses if fid == fragment_id]
        if fragment_id == second:
            with pytest.raises(ValueError, match="Missing, duplicate, or misaligned"):
                publish_attempt(staged.slice(0, 3), attempt)
        outcome = publish_attempt(staged, attempt)
        assert outcome.summary()["remaining_fragments"] == (1 if fragment_id == first else 0)
    latest = lance.dataset(dataset.uri)
    assert latest.to_table(columns=SPAN_SCHEMA.names).equals(pa.Table.from_pylist(source_rows, schema=SPAN_SCHEMA))
    assert latest.to_table(filter="span_id IS NULL", columns=["clip_id"])["clip_id"].null_count == 3
    assert reconcile_and_select_attempt(inputs.open_enrichment_input(bindings)).selected == ()
    validate_caption_input(latest, uri=latest.uri)


@pytest.mark.parametrize("status", ["empty", "error"])
@pytest.mark.parametrize("with_clips", [False, True])
def test_appending_outcomes_needs_no_transcode_work_and_keeps_caption_ready(
    config: ResolvedVideoTranscodeConfig,
    enrichment_input: inputs.EnrichmentInput,
    results: pa.Table,
    status: str,
    monkeypatch: pytest.MonkeyPatch,
    *,
    with_clips: bool,
) -> None:
    """Later empty/error outcomes remain ready, with or without earlier completed clips."""
    dataset = enrichment_input.dataset
    source = dataset.to_table(columns=SPAN_SCHEMA.names).to_pylist()[0]
    source |= {"span_id": None, "start_ns": None, "end_ns": None}
    if status == "error":
        source |= {"error_stage": "source-read", "error_message": "missing"}
    if with_clips:
        publish_results(results, enrichment_input)
    else:
        lance.write_dataset(
            pa.Table.from_pylist([source], schema=dataset.schema),
            dataset.uri,
            mode="overwrite",
            data_storage_version="2.2",
        )
    monkeypatch.setattr(pipeline, "start_ray", Mock(side_effect=AssertionError("Ray must not start")))
    assert pipeline.run_config(config)["selected_fragments"] == 0
    before = lance.dataset(dataset.uri)
    validate_caption_input(before, uri=before.uri)
    source |= {"source_id": "later-source", "source_uri": "s3://test-bucket/later.mp4"}
    appended = lance.write_dataset(
        pa.Table.from_pylist([source], schema=before.schema),
        dataset.uri,
        mode="append",
        data_storage_version="2.2",
    )
    summary = pipeline.run_config(config)
    assert summary["selected_fragments"] == 0
    assert summary["clips_lance_version"] == appended.version
    latest = lance.dataset(dataset.uri)
    validate_caption_input(latest, uri=latest.uri)
    assert latest.count_rows() == before.count_rows() + 1
