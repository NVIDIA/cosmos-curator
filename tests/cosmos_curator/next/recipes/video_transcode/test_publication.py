# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Atomic column publication, incomplete results, and recovery boundaries."""

import json
import sys
from typing import Any

import lance
import pyarrow as pa
import pyarrow.compute as pc
import pytest

from cosmos_curator.next.recipes.video_caption.contracts import caption_contract_digest, resolve_model_spec
from cosmos_curator.next.recipes.video_caption.lance_state import ensure_caption_fields, validate_caption_input
from cosmos_curator.next.recipes.video_cut.contracts import SPAN_SCHEMA
from cosmos_curator.next.recipes.video_transcode import pipeline, publication
from cosmos_curator.next.recipes.video_transcode.config import ResolvedVideoTranscodeConfig
from cosmos_curator.next.recipes.video_transcode.contracts import (
    CLIP_FIELDS,
    RESULT_SCHEMA,
    completed_publication,
    make_clip_id,
)
from tests.cosmos_curator.next.recipes.video_caption.helpers import caption_run_bindings


def _prepare(
    results: pa.Table, dataset: lance.LanceDataset, config: ResolvedVideoTranscodeConfig
) -> list[dict[str, Any]]:
    candidates = []
    pending = publication.pending_fragment_ids(dataset, config)
    for fragment in dataset.get_fragments():
        if fragment.fragment_id not in pending:
            continue
        rows = results.filter(pc.equal(results["fragment_id"], fragment.fragment_id))
        candidates.extend(publication.prepare_fragment(rows, config=config, version=dataset.version).to_pylist())
    return candidates


def _append_span(dataset: lance.LanceDataset) -> lance.LanceDataset:
    row = dataset.to_table(columns=SPAN_SCHEMA.names).slice(0, 1).to_pylist()[0]
    row["span_id"] = "new-span"
    return lance.write_dataset(
        pa.Table.from_pylist([row], schema=SPAN_SCHEMA), dataset.uri, mode="append", data_storage_version="2.2"
    )


@pytest.mark.parametrize("during_commit", [False, True])
def test_concurrent_append_preserves_new_pending_fragment(
    config: ResolvedVideoTranscodeConfig,
    dataset: lance.LanceDataset,
    results: pa.Table,
    monkeypatch: pytest.MonkeyPatch,
    *,
    during_commit: bool,
) -> None:
    """An append before or during the transaction leaves old results complete and new spans pending."""
    candidates = _prepare(results, dataset, config)
    original_files = {f.fragment_id: f.metadata.to_json()["files"] for f in dataset.get_fragments()}
    commit = lance.LanceDataset.commit
    appended = False

    def race(*args: object, **kwargs: object) -> lance.LanceDataset:
        nonlocal appended
        if not appended:
            appended = True
            _append_span(dataset)
        return commit(*args, **kwargs)

    if during_commit:
        monkeypatch.setattr(lance.LanceDataset, "commit", race)
    else:
        _append_span(dataset)
    published = publication.commit_fragments(candidates, config=config, dataset=dataset)
    assert published.count_rows() == 5
    assert publication.pending_fragment_ids(published, config) == (2,)
    assert published.get_fragment(2).to_table(columns=["clip_uri"])["clip_uri"].null_count == 1
    for fragment in published.get_fragments()[:2]:
        assert fragment.metadata.to_json()["files"][:1] == original_files[fragment.fragment_id]
        assert len(fragment.metadata.to_json()["files"]) == 2
        assert fragment.metadata.num_deletions == 0
    # A fresh attempt selects only the new full fragment, not the completed spans.
    tasks = pipeline.SpanDatasource(publication.open_input(config), config).get_read_tasks(2)
    assert [row["span_id"] for task in tasks for batch in task() for row in batch.to_pylist()] == ["new-span"]


@pytest.mark.parametrize("during_commit", [False, True])
def test_concurrent_same_fragment_enrichment_restages_without_media_work(
    config: ResolvedVideoTranscodeConfig,
    dataset: lance.LanceDataset,
    results: pa.Table,
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
    *,
    during_commit: bool,
) -> None:
    """Refreshing an Update from checkpointed results retains another writer's column file."""
    dataset.add_columns(pa.schema([pa.field("other", pa.string())]))
    candidates = _prepare(results, dataset, config)
    commit = lance.LanceDataset.commit
    updates = 0

    def enrich() -> None:
        current = lance.dataset(dataset.uri)
        rows = current.get_fragment(0).to_table(columns=["span_id"])
        rows = rows.append_column("other", pa.array(["keep"] * len(rows)))
        metadata, fields = current.get_fragment(0).update_columns(rows, left_on="span_id", right_on="span_id")
        commit(
            dataset.uri,
            lance.LanceOperation.Update(updated_fragments=[metadata], fields_modified=fields),
            read_version=current.version,
        )

    def race(*args: object, **kwargs: object) -> lance.LanceDataset:
        nonlocal updates
        if updates == 0:
            updates += 1
            enrich()
        return commit(*args, **kwargs)

    if during_commit:
        monkeypatch.setattr(lance.LanceDataset, "commit", race)
    else:
        enrich()
    reprepared = []

    def reprepare(version: int) -> list[dict[str, Any]]:
        reprepared.append(version)
        return _prepare(results, lance.dataset(dataset.uri, version=version), config)

    # Other tests can remove Loguru's default sinks. Bind this test's captured
    # stderr explicitly so the retry assertion does not depend on test order.
    sink = pipeline.logger.add(sys.stderr, level="WARNING")
    try:
        published = publication.commit_fragments(candidates, config=config, dataset=dataset, reprepare=reprepare)
    finally:
        pipeline.logger.remove(sink)
    assert len(reprepared) == 1
    assert published.get_fragment(0).to_table(columns=["other"])["other"].to_pylist() == ["keep", "keep"]
    assert publication.pending_fragment_ids(published, config) == ()
    captured = capfd.readouterr()
    if during_commit:
        assert "Retrying" in captured.err


def test_modified_input_rows_reject_prepared_transcodes(
    config: ResolvedVideoTranscodeConfig,
    dataset: lance.LanceDataset,
    results: pa.Table,
) -> None:
    """Stable fragment IDs alone cannot authorize updates after source values changed."""
    candidates = _prepare(results, dataset, config)
    fragment = dataset.get_fragment(0)
    rows = fragment.to_table(columns=["span_id"]).append_column("source_uri", pa.array(["changed.mp4"] * 2))
    metadata, fields = fragment.update_columns(rows, left_on="span_id", right_on="span_id")
    lance.LanceDataset.commit(
        dataset.uri,
        lance.LanceOperation.Update(updated_fragments=[metadata], fields_modified=fields),
        read_version=dataset.version,
    )
    with pytest.raises(ValueError, match="Video-cut rows or physical binding changed"):
        publication.commit_fragments(candidates, config=config, dataset=dataset)
    assert lance.dataset(dataset.uri).to_table(columns=["clip_uri"])["clip_uri"].null_count == 4


def test_mixed_fragment_is_rejected(
    config: ResolvedVideoTranscodeConfig,
    dataset: lance.LanceDataset,
    results: pa.Table,
) -> None:
    """Only full-fragment publication is a supported recovery state."""
    values = results.slice(0, 1).select(["span_id", *CLIP_FIELDS.names])
    metadata, fields = dataset.get_fragment(0).update_columns(values, left_on="span_id", right_on="span_id")
    current = lance.LanceDataset.commit(
        dataset.uri,
        lance.LanceOperation.Update(updated_fragments=[metadata], fields_modified=fields),
        read_version=dataset.version,
    )
    with pytest.raises(ValueError, match="mixes pending and completed"):
        publication.pending_fragment_ids(current, config)


def test_nullable_registration_writes_no_placeholder_data(
    config: ResolvedVideoTranscodeConfig,
    dataset: lance.LanceDataset,
) -> None:
    """Schema initialization preserves base files and adds no physical all-null file."""
    assert set(CLIP_FIELDS.names) <= set(dataset.schema.names)
    assert all(len(fragment.metadata.to_json()["files"]) == 1 for fragment in dataset.get_fragments())
    assert publication.open_input(config).version == dataset.version


def test_next_transcode_run_fills_only_appended_fragments(
    config: ResolvedVideoTranscodeConfig,
    dataset: lance.LanceDataset,
    results: pa.Table,
) -> None:
    """Incremental publication retains the exact files of already completed fragments."""
    first = publication.commit_fragments(_prepare(results, dataset, config), config=config, dataset=dataset)
    files = [fragment.metadata.to_json() for fragment in first.get_fragments()]
    current = _append_span(first)
    clip_id = make_clip_id("new-span", config.transcode.contract_id)
    row = results.slice(0, 1).to_pylist()[0] | {
        "span_id": "new-span",
        "clip_id": clip_id,
        "clip_uri": f"{config.input.media_root}/clips/{clip_id}.mp4",
        "fragment_id": 2,
        "row_offset": 0,
    }
    candidates = _prepare(pa.Table.from_pylist([row], schema=RESULT_SCHEMA), current, config)
    assert [candidate["fragment_id"] for candidate in candidates] == [2]
    published = publication.commit_fragments(candidates, config=config, dataset=current)
    assert publication.pending_fragment_ids(published, config) == ()
    assert [fragment.metadata.to_json() for fragment in published.get_fragments()[:2]] == files
    assert completed_publication(published)["spans"] == 5


def test_single_commit_preserves_fragments_and_feeds_caption(
    config: ResolvedVideoTranscodeConfig, dataset: lance.LanceDataset, results: pa.Table
) -> None:
    """Column files stay invisible until one commit; caption accepts the new output."""
    before = dataset.to_table(columns=SPAN_SCHEMA.names)
    candidates = _prepare(results, dataset, config)
    assert lance.dataset(dataset.uri).to_table(columns=["clip_uri"])["clip_uri"].null_count == 4
    published = publication.commit_fragments(candidates, config=config, dataset=dataset)
    assert published.version == dataset.version + 1
    assert published.to_table(columns=SPAN_SCHEMA.names).equals(before, check_metadata=False)
    assert [f.fragment_id for f in published.get_fragments()] == [f.fragment_id for f in dataset.get_fragments()]
    assert completed_publication(published)["spans"] == 4
    validate_caption_input(published, uri=dataset.uri)
    spec = resolve_model_spec("qwen3_8_27b_fp8")
    captioned = ensure_caption_fields(
        dataset.uri,
        storage_options=None,
        bindings=caption_run_bindings(spec, caption_contract_digest(spec)),
        attempts=1,
    )
    assert publication.open_input(config).version == captioned.version


@pytest.mark.parametrize(
    "corruption", ["missing", "duplicate", "wrong-id", "wrong-offset", "wrong-contract", "wrong-uri"]
)
def test_bad_results_never_publish(
    config: ResolvedVideoTranscodeConfig, dataset: lance.LanceDataset, results: pa.Table, corruption: str
) -> None:
    """Counts alone are insufficient: validate every row's address and identity."""
    rows = results.slice(0, 2).to_pylist()
    if corruption == "missing":
        rows.pop()
    elif corruption == "duplicate":
        rows[1] = dict(rows[0])
    else:
        field = {
            "wrong-id": "span_id",
            "wrong-offset": "row_offset",
            "wrong-contract": "transcode_contract_id",
            "wrong-uri": "clip_uri",
        }[corruption]
        rows[0][field] = 123 if field == "row_offset" else "wrong"
    with pytest.raises(ValueError, match=r"span results|transcode result"):
        publication.prepare_fragment(
            pa.Table.from_pylist(rows, schema=RESULT_SCHEMA), config=config, version=dataset.version
        )
    assert lance.dataset(dataset.uri).version == dataset.version


def test_missing_fragment_and_concurrent_schema_extension(
    config: ResolvedVideoTranscodeConfig, dataset: lance.LanceDataset, results: pa.Table
) -> None:
    """Require all selected fragments but tolerate another producer's nullable fields."""
    candidates = _prepare(results, dataset, config)
    with pytest.raises(ValueError, match="exactly cover"):
        publication.commit_fragments(candidates[:1], config=config, dataset=dataset)
    original_version = dataset.version
    dataset.add_columns(pa.schema([pa.field("other", pa.string())]))
    result = publication.commit_fragments(
        candidates, config=config, dataset=lance.dataset(dataset.uri, version=original_version)
    )
    assert "other" in result.schema.names
    assert result.to_table(columns=["clip_uri"])["clip_uri"].null_count == 0


@pytest.mark.parametrize("append_after_commit", [False, True])
def test_lost_commit_response_is_reconciled(
    config: ResolvedVideoTranscodeConfig,
    dataset: lance.LanceDataset,
    results: pa.Table,
    monkeypatch: pytest.MonkeyPatch,
    *,
    append_after_commit: bool,
) -> None:
    """The completion record shares the transaction, preventing a duplicate commit."""
    candidates = _prepare(results, dataset, config)
    commit = lance.LanceDataset.commit
    calls = []

    def lose_response(*args: object, **kwargs: object) -> None:
        calls.append(1)
        commit(*args, **kwargs)
        if append_after_commit:
            _append_span(dataset)
        message = "lost response"
        raise OSError(message)

    monkeypatch.setattr(lance.LanceDataset, "commit", lose_response)
    published = publication.commit_fragments(candidates, config=config, dataset=dataset)
    assert published.version == dataset.version + 1 + int(append_after_commit)
    assert len(calls) == 1


def test_completed_run_skips_ray_and_staging(
    config: ResolvedVideoTranscodeConfig,
    dataset: lance.LanceDataset,
    results: pa.Table,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rerun after commit needs only canonical Lance state."""
    publication.commit_fragments(_prepare(results, dataset, config), config=config, dataset=dataset)
    monkeypatch.setattr(pipeline, "ensure_ray_initialized", lambda **_: pytest.fail("Ray should not start"))
    monkeypatch.setattr(pipeline, "cleanup_workspace", lambda *_: pytest.fail("staging should not be touched"))
    assert pipeline.run_config(config)["cleanup_complete"] is None


def test_input_accepts_committed_fragments_without_whole_table_completion(
    config: ResolvedVideoTranscodeConfig, dataset: lance.LanceDataset
) -> None:
    """Committed fragments are independently ready even while cutting continues."""
    dataset.update_schema_metadata({"video_cut.publication": None})
    current = publication.open_input(config)
    assert publication.pending_fragment_ids(current, config) == (0, 1)


def test_workspace_excludes_execution_settings(
    config: ResolvedVideoTranscodeConfig, dataset: lance.LanceDataset
) -> None:
    """Changing checkpoint size and CPU tuning reuses durable span results."""
    changed = config.model_copy(
        update={"execution": config.execution.model_copy(update={"spans_per_checkpoint": 1000, "transcode_cpus": 8.0})}
    )
    assert publication.resolve_workspace(config, dataset) == publication.resolve_workspace(changed, dataset)
    changed = config.model_copy(update={"transcode": config.transcode.model_copy(update={"video_bitrate": "8M"})})
    assert publication.resolve_workspace(config, dataset) != publication.resolve_workspace(changed, dataset)


def test_empty_completed_cut_publishes_without_ray(
    config: ResolvedVideoTranscodeConfig, dataset: lance.LanceDataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A completed zero-span dataset still gets a durable transcode completion record."""
    metadata = dict(dataset.schema.metadata or {})
    metadata[b"video_cut.publication"] = json.dumps({"publication_version": 2, "spans": 0}).encode()
    lance.write_dataset(
        pa.Table.from_pylist([], schema=dataset.schema.with_metadata(metadata)),
        dataset.uri,
        mode="overwrite",
        data_storage_version="2.2",
    )
    monkeypatch.setattr(pipeline, "ensure_ray_initialized", lambda **_: pytest.fail("Ray should not start"))
    result = pipeline.run_config(config)
    assert result["spans"] == 0
    assert completed_publication(lance.dataset(dataset.uri))["spans"] == 0


def test_cleanup_failure_keeps_completed_output(
    config: ResolvedVideoTranscodeConfig,
    dataset: lance.LanceDataset,
    results: pa.Table,
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    """Cleanup is best effort after publication, never a reason to transcode again."""
    candidates = _prepare(results, dataset, config)
    monkeypatch.setattr(pipeline, "ensure_ray_initialized", lambda **_: None)
    monkeypatch.setattr(pipeline, "configure_ray_data_progress", lambda **_: None)
    monkeypatch.setattr(pipeline, "configure_ray_data_stability", lambda **_kwargs: None)
    monkeypatch.setattr(pipeline, "transcode_spans", lambda *_: None)
    monkeypatch.setattr(
        pipeline, "publish_results", lambda ds, cfg, _: publication.commit_fragments(candidates, config=cfg, dataset=ds)
    )

    def fail_cleanup(*_: object) -> None:
        message = "cleanup unavailable"
        raise OSError(message)

    monkeypatch.setattr(pipeline, "cleanup_workspace", fail_cleanup)
    # Loguru's original sink can retain pytest's session-level stderr object.
    # Bind an additional sink to this test's capture, including under pytest -s.
    sink = pipeline.logger.add(sys.stderr, level="WARNING")
    try:
        assert pipeline.run_config(config)["cleanup_complete"] is False
        assert pipeline.run_config(config)["cleanup_complete"] is None
    finally:
        pipeline.logger.remove(sink)
    assert "workspace cleanup failed" in capfd.readouterr().err
