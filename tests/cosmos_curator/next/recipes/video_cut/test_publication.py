# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Atomic append publication, ambiguous responses, and incremental source selection."""

import json
from pathlib import Path
from typing import Any

import lance
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cosmos_curator.next.media.ffmpeg import VideoMetadata
from cosmos_curator.next.recipes.video_cut import pipeline, publication
from cosmos_curator.next.recipes.video_cut.config import ResolvedVideoCutConfig, resolve_config_data
from cosmos_curator.next.recipes.video_cut.contracts import (
    SOURCE_SCHEMA,
    SPAN_FIELDS,
    SPAN_SCHEMA,
    make_source_id,
    span_schema,
)
from cosmos_curator.next.recipes.video_cut.processing import source_result
from cosmos_curator.next.utils.identity import canonical_digest


def _stage(config: ResolvedVideoCutConfig, *, duration: int = 5_000_000_000) -> tuple[pa.Table, dict[str, Any]]:
    snapshot_id = canonical_digest(config.input.uris)
    workspace = publication.resolve_workspace(config, snapshot_id)
    publication.open_span_table(config, snapshot_id)
    row = source_result(config.input.uris[0], VideoMetadata(duration, 64, 48, 30.0, None, "h264"), 50, config.split)
    empty = source_result(config.input.uris[1], VideoMetadata(100, 64, 48, 30.0, None, "h264"), 50, config.split)
    outcomes = pa.Table.from_pylist([row, empty], schema=SOURCE_SCHEMA)
    Path(workspace.results).mkdir(parents=True)
    pq.write_table(outcomes, Path(workspace.results) / "source.parquet")
    record = {
        "source_results": 2,
        "errors": 0,
        "errors_uri": None,
        "source_ids": [make_source_id(uri) for uri in config.input.uris],
    }
    return outcomes, record


def _candidates(outcomes: pa.Table, config: ResolvedVideoCutConfig) -> list[str]:
    return [
        row["fragment"]
        for batch in publication.expand_sources(
            outcomes, contract_id=config.split.contract_id, spans_per_fragment=config.execution.spans_per_fragment
        )
        for row in publication.stage_fragment(
            batch,
            schema=span_schema(config.split, canonical_digest(config.input.uris)),
            spans_lance_uri=config.output.spans_lance_uri,
            spans_per_fragment=config.execution.spans_per_fragment,
            storage_profile=config.execution.storage_profile,
        ).to_pylist()
    ]


@pytest.mark.parametrize(
    ("override", "reuses"),
    [
        ("execution.sources_per_checkpoint=10", True),
        ("execution.spans_per_fragment=99", True),
        ("split.stride_s=1.0", False),
        ("input.uris=[s3://test-bucket/another.mp4]", False),
        ("output.spans_lance_uri=s3://test-bucket/another-output", False),
    ],
)
def test_workspace_path_isolates_recovery_without_a_manifest(
    config: ResolvedVideoCutConfig, snapshot_id: str, override: str, *, reuses: bool
) -> None:
    """Result changes isolate recovery while execution tuning reuses the same workspace."""
    workspace = publication.resolve_workspace(config, snapshot_id)
    changed = resolve_config_data(config.model_dump(mode="json"), overrides=[override])
    selected = publication.resolve_workspace(changed, canonical_digest(changed.input.uris))
    assert (selected == workspace) is reuses
    assert not Path(workspace.root).exists()


@pytest.mark.parametrize(
    ("schema", "reuses"),
    [
        pytest.param(pa.schema(list(SOURCE_SCHEMA)), True, id="equivalent-schema"),
        pytest.param(SOURCE_SCHEMA.append(pa.field("probe_detail", pa.string())), False, id="added-field"),
        pytest.param(
            SOURCE_SCHEMA.set(SOURCE_SCHEMA.get_field_index("source_width"), pa.field("source_width", pa.int64())),
            False,
            id="changed-type",
        ),
        pytest.param(
            SOURCE_SCHEMA.set(SOURCE_SCHEMA.get_field_index("source_uri"), pa.field("source_uri", pa.string())),
            False,
            id="changed-nullability",
        ),
        pytest.param(
            SOURCE_SCHEMA.set(
                SOURCE_SCHEMA.get_field_index("spans"),
                pa.field(
                    "spans", pa.list_(pa.struct([*SPAN_FIELDS, pa.field("confidence", pa.float32())])), nullable=False
                ),
            ),
            False,
            id="changed-nested-field",
        ),
    ],
)
def test_workspace_schema_changes_isolate_recovery(
    config: ResolvedVideoCutConfig,
    snapshot_id: str,
    monkeypatch: pytest.MonkeyPatch,
    schema: pa.Schema,
    *,
    reuses: bool,
) -> None:
    """Only an equivalent source schema can reuse existing source plans and checkpoints."""
    _stage(config)
    original = publication.resolve_workspace(config, snapshot_id)
    existing = publication.source_result_files(original, config)
    assert len(existing) == 1

    monkeypatch.setattr(publication, "SOURCE_SCHEMA", schema)
    selected = publication.resolve_workspace(config, snapshot_id)

    assert (selected.checkpoints == original.checkpoints) is reuses
    assert publication.source_result_files(selected, config) == (existing if reuses else [])
    assert Path(existing[0]).is_file()


def test_all_fragments_publish_atomically_and_replay_is_noop(config: ResolvedVideoCutConfig, snapshot_id: str) -> None:
    """All new sources append in exactly one commit; the receipt makes replay a no-op."""
    outcomes, record = _stage(config)
    before = publication.open_span_table(config, snapshot_id)
    candidates = _candidates(outcomes, config)
    assert len(candidates) == 2
    assert publication.open_span_table(config, snapshot_id).count_rows() == 0
    dataset = publication.commit_fragments(
        candidates, config=config, input_snapshot_id=snapshot_id, publication=record, read_version=int(before.version)
    )
    assert dataset.count_rows() == 3
    assert len(dataset.get_fragments()) == 2
    assert dataset.version == before.version + 1
    assert len(set(dataset.to_table(columns=["span_id"])["span_id"].to_pylist())) == 3
    assert publication.completed_publication(dataset)["spans"] == 3
    # Even a stale attempt and different fragment order recognizes the atomic completion record.
    again = publication.commit_fragments(
        list(reversed(candidates)),
        config=config,
        input_snapshot_id=snapshot_id,
        publication=record,
        read_version=int(before.version),
    )
    assert again.version == dataset.version


def test_ambiguous_commit_is_resolved(
    config: ResolvedVideoCutConfig, monkeypatch: pytest.MonkeyPatch, snapshot_id: str
) -> None:
    """A lost commit response does not cause another publication or duplicate spans."""
    outcomes, record = _stage(config)
    candidates = _candidates(outcomes, config)
    commit = lance.LanceDataset.commit
    calls = []

    def lost_response(*args: object, **kwargs: object) -> lance.LanceDataset:
        calls.append(1)
        commit(*args, **kwargs)
        message = "response lost"
        raise OSError(message)

    monkeypatch.setattr(lance.LanceDataset, "commit", lost_response)
    dataset = publication.commit_fragments(
        candidates, config=config, input_snapshot_id=snapshot_id, publication=record, read_version=1
    )
    assert dataset.count_rows() == 3
    assert dataset.version == 2
    assert calls == [1]


def test_zero_spans_keep_valid_empty_table(config: ResolvedVideoCutConfig, snapshot_id: str) -> None:
    """An all-zero result commits its source accounting without creating data fragments."""
    outcomes, record = _stage(config, duration=100)
    assert _candidates(outcomes, config) == []
    dataset = publication.commit_fragments(
        [], config=config, input_snapshot_id=snapshot_id, publication=record, read_version=1
    )
    assert dataset.count_rows() == 0
    assert dataset.schema.equals(SPAN_SCHEMA, check_metadata=False)
    assert publication.completed_publication(dataset)["source_results"] == 2


@pytest.mark.parametrize("sources", [["s3://test-bucket/long.mp4"], ["s3://test-bucket/new.mp4"]])
def test_changed_source_snapshot_selects_only_new_sources(
    config: ResolvedVideoCutConfig,
    monkeypatch: pytest.MonkeyPatch,
    sources: list[str],
    snapshot_id: str,
) -> None:
    """A different selection can reuse the table without probing completed sources."""
    outcomes, record = _stage(config)
    publication.commit_fragments(
        _candidates(outcomes, config),
        config=config,
        input_snapshot_id=snapshot_id,
        publication=record,
        read_version=1,
    )
    raw = config.model_dump(mode="json")
    raw["input"] = {"uris": sources}
    changed = resolve_config_data(raw)
    monkeypatch.setattr(pipeline, "ensure_ray_initialized", lambda **_: pytest.fail("unexpected Ray initialization"))
    current = publication.open_span_table(changed, canonical_digest(changed.input.uris))
    assert publication.pending_source_uris(current, changed.input.uris) == (
        changed.input.uris if sources[0].endswith("new.mp4") else ()
    )
    if not sources[0].endswith("new.mp4"):
        assert pipeline.run_config(changed)["source_results"] == 0
    assert Path(publication.resolve_workspace(config, snapshot_id).root).exists()


def test_publication_preserves_concurrent_changes_and_additional_columns(
    config: ResolvedVideoCutConfig, snapshot_id: str
) -> None:
    """An append preserves nullable schema extensions committed while staging."""
    outcomes, record = _stage(config)
    candidates = _candidates(outcomes, config)
    concurrent = publication.open_span_table(config, snapshot_id)
    concurrent.add_columns(pa.schema([pa.field("annotation", pa.string())]))
    result = publication.commit_fragments(
        candidates, config=config, input_snapshot_id=snapshot_id, publication=record, read_version=1
    )
    assert result.to_table(columns=["annotation"])["annotation"].null_count == 3


def test_new_run_appends_without_changing_existing_enriched_fragments(
    config: ResolvedVideoCutConfig,
    snapshot_id: str,
) -> None:
    """An independently selected source set adds fragments and preserves all earlier files."""
    outcomes, record = _stage(config)
    first = publication.commit_fragments(
        _candidates(outcomes, config),
        config=config,
        input_snapshot_id=snapshot_id,
        publication=record,
        read_version=1,
    )
    first.add_columns(pa.schema([pa.field("annotation", pa.string())]))
    raw = config.model_dump(mode="json")
    raw["input"] = {"uris": ["s3://test-bucket/new-long.mp4", "s3://test-bucket/new-short.mp4"]}
    changed = resolve_config_data(raw)
    new_snapshot = canonical_digest(changed.input.uris)
    new_outcomes, new_record = _stage(changed)
    candidates = _candidates(new_outcomes, changed)
    # Populate an existing fragment after the cutter has staged its new ones.
    read_version = int(first.version)
    values = first.get_fragment(0).to_table(columns=["span_id"])
    values = values.append_column("annotation", pa.array(["retained"] * values.num_rows))
    fragment, fields = first.get_fragment(0).update_columns(values, left_on="span_id", right_on="span_id")
    enriched = lance.LanceDataset.commit(
        first.uri,
        lance.LanceOperation.Update(updated_fragments=[fragment], fields_modified=fields),
        read_version=first.version,
    )
    existing_files = [fragment.metadata.to_json() for fragment in enriched.get_fragments()]
    published = publication.commit_fragments(
        candidates,
        config=changed,
        input_snapshot_id=new_snapshot,
        publication=new_record,
        read_version=read_version,
    )
    assert published.count_rows() == 6
    assert [fragment.metadata.to_json() for fragment in published.get_fragments()[:2]] == existing_files
    assert published.get_fragment(0).to_table(columns=["annotation"])["annotation"].to_pylist() == ["retained"] * 2
    assert published.get_fragment(2).to_table(columns=["annotation"])["annotation"].null_count == 2
    assert publication.pending_source_uris(published, (*config.input.uris, *changed.input.uris)) == ()
    # Replaying either receipt after unrelated versions does not append again.
    again = publication.commit_fragments(
        _candidates(outcomes, config),
        config=config,
        input_snapshot_id=snapshot_id,
        publication=record,
        read_version=1,
    )
    assert again.version == published.version


def test_growing_selection_processes_only_new_sources(
    config: ResolvedVideoCutConfig,
    snapshot_id: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The run path excludes both successful and zero-span sources before Ray starts."""
    outcomes, record = _stage(config)
    current = publication.commit_fragments(
        _candidates(outcomes, config),
        config=config,
        input_snapshot_id=snapshot_id,
        publication=record,
        read_version=1,
    )
    raw = config.model_dump(mode="json")
    new_uri = "s3://test-bucket/new.mp4"
    raw["input"]["uris"].append(new_uri)
    changed = resolve_config_data(raw)
    observed = []

    def run_pending(
        cfg: ResolvedVideoCutConfig, pending: tuple[str, ...], selection: str
    ) -> tuple[lance.LanceDataset, dict[str, Any], bool]:
        assert cfg == changed
        observed.append(pending)
        assert selection == canonical_digest(changed.input.uris)
        return current, {"source_results": 1, "errors": 0, "errors_uri": None}, True

    monkeypatch.setattr(pipeline, "_run_pending", run_pending)
    summary = pipeline.run_config(changed)
    assert observed == [(new_uri,)]
    assert summary["selected_sources"] == 3
    assert summary["source_results"] == 1


def test_legacy_completed_snapshot_can_be_reopened_and_extended(
    config: ResolvedVideoCutConfig,
    snapshot_id: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Existing v2 metadata remains readable while the input selection may grow."""
    outcomes, _ = _stage(config)
    spans = pa.concat_tables(
        list(publication.expand_sources(outcomes, contract_id=config.split.contract_id, spans_per_fragment=2))
    )
    metadata = dict(span_schema(config.split).metadata or {})
    metadata[b"video_cut.input_snapshot"] = snapshot_id.encode()
    metadata[b"video_cut.publication"] = json.dumps(
        {"publication_version": 2, "spans": 3, "source_results": 2, "errors": 0, "errors_uri": None}
    ).encode()
    lance.write_dataset(
        spans.cast(span_schema(config.split).with_metadata(metadata)),
        config.output.spans_lance_uri,
        mode="overwrite",
        data_storage_version="2.2",
    )
    monkeypatch.setattr(
        pipeline, "ensure_ray_initialized", lambda **_: pytest.fail("Legacy completion should skip Ray")
    )
    assert pipeline.run_config(config)["source_results"] == 2
    current = publication.open_span_table(config, "different-selection")
    assert publication.pending_source_uris(current, (config.input.uris[0], "s3://test-bucket/new.mp4")) == (
        "s3://test-bucket/new.mp4",
    )


@pytest.mark.parametrize("legacy_record", [False, True])
def test_completed_run_reopens_without_touching_staging(
    config: ResolvedVideoCutConfig,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    snapshot_id: str,
    *,
    legacy_record: bool,
) -> None:
    """Completed reopens ignore legacy cleanup paths, including under a different staging profile."""
    outcomes, record = _stage(config)
    workspace = publication.resolve_workspace(config, snapshot_id)
    moved = resolve_config_data(
        config.model_dump(mode="json"),
        overrides=[f"output.staging_root_uri={tmp_path / 'other-staging'}", "execution.storage_profile=other"],
    )
    sibling = publication.resolve_workspace(moved, snapshot_id)
    Path(sibling.results).mkdir(parents=True)
    sibling_result = Path(sibling.results, "retained.parquet")
    sibling_result.write_bytes(b"unrelated recovery state")
    if legacy_record:
        record["workspace_root"] = sibling.root
    dataset = publication.commit_fragments(
        _candidates(outcomes, config), config=config, input_snapshot_id=snapshot_id, publication=record, read_version=1
    )
    monkeypatch.setattr(pipeline, "ensure_ray_initialized", lambda **_: pytest.fail("unexpected Ray initialization"))
    monkeypatch.setattr(pipeline, "cleanup_workspace", lambda *_: pytest.fail("unexpected workspace cleanup"))
    first = pipeline.run_config(config)
    assert first["spans"] == 3
    assert first["cleanup_complete"] is None
    assert pipeline.run_config(moved) == first
    assert Path(workspace.results, "source.parquet").is_file()
    assert sibling_result.read_bytes() == b"unrelated recovery state"
    assert publication.open_span_table(config, snapshot_id).version == dataset.version


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_resumed_workspace_cleanup_is_best_effort(
    config: ResolvedVideoCutConfig,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
    loguru_records: list[dict[str, Any]],
    snapshot_id: str,
    *,
    cleanup_fails: bool,
) -> None:
    """Only the workspace used for publication is cleaned, and cleanup failures leave valid output."""
    outcomes, record = _stage(config)
    candidates = _candidates(outcomes, config)
    workspace = publication.resolve_workspace(config, snapshot_id)
    other = resolve_config_data(
        config.model_dump(mode="json"), overrides=[f"output.spans_lance_uri={tmp_path / 'other.lance'}"]
    )
    sibling = publication.resolve_workspace(other, snapshot_id)
    Path(sibling.results).mkdir(parents=True)
    sibling_result = Path(sibling.results, "retained.parquet")
    sibling_result.write_bytes(b"other output recovery state")

    def interrupted_cleanup(workspace: publication.Workspace, _: ResolvedVideoCutConfig) -> None:
        Path(workspace.results, "source.parquet").unlink()
        msg = "cleanup interrupted"
        raise OSError(msg)

    monkeypatch.setattr(pipeline, "ensure_ray_initialized", lambda: None)
    monkeypatch.setattr(pipeline, "configure_ray_data_progress", lambda **_: None)
    monkeypatch.setattr(pipeline, "configure_ray_data_stability", lambda: None)
    monkeypatch.setattr(pipeline, "cut_sources", lambda *_: None)
    monkeypatch.setattr(
        pipeline,
        "publish_spans",
        lambda *_args, **_kwargs: publication.commit_fragments(
            candidates, config=config, input_snapshot_id=snapshot_id, publication=record, read_version=1
        ),
    )
    if cleanup_fails:
        monkeypatch.setattr(pipeline, "cleanup_workspace", interrupted_cleanup)
    summary = pipeline.run_config(config)
    assert summary["spans"] == 3
    assert summary["cleanup_complete"] is (not cleanup_fails)
    assert Path(workspace.root).exists() is cleanup_fails
    assert sibling_result.read_bytes() == b"other output recovery state"

    monkeypatch.setattr(pipeline, "ensure_ray_initialized", lambda **_: pytest.fail("unexpected Ray initialization"))
    monkeypatch.setattr(pipeline, "cleanup_workspace", lambda *_: pytest.fail("unexpected workspace cleanup"))
    assert pipeline.run_config(config) == summary | {"cleanup_complete": None}
    assert Path(workspace.root).exists() is cleanup_fails
    assert publication.open_span_table(config, snapshot_id).version == summary["spans_lance_version"]
    # Consume output from any existing console sink for pytest -s.
    capfd.readouterr()
    warnings = [record["message"] for record in loguru_records if record["level"].name == "WARNING"]
    if cleanup_fails:
        (warning,) = warnings
        assert "cleanup interrupted" in warning
    else:
        assert not warnings


def test_table_rejects_incompatible_contract_but_allows_new_staging(
    config: ResolvedVideoCutConfig, tmp_path: Path, snapshot_id: str
) -> None:
    """The canonical result depends on the cutting contract, independently of recovery storage."""
    publication.open_span_table(config, snapshot_id)
    changed = resolve_config_data(config.model_dump(mode="json"), overrides=["split.stride_s=1.0"])
    with pytest.raises(ValueError, match="different video-cut"):
        publication.open_span_table(changed, snapshot_id)
    moved = resolve_config_data(
        config.model_dump(mode="json"), overrides=[f"output.staging_root_uri={tmp_path / 'other'}"]
    )
    assert publication.open_span_table(moved, snapshot_id).version == 1


def test_wrong_cutting_provenance_is_rejected(config: ResolvedVideoCutConfig) -> None:
    """Private source results must still belong to the selected cutting contract."""
    row = source_result(
        config.input.uris[0], VideoMetadata(5_000_000_000, 64, 48, 30.0, None, "h264"), 50, config.split
    )
    row["cut_contract_id"] = "wrong-contract"
    with pytest.raises(ValueError, match="Invalid cutting provenance"):
        list(
            publication.expand_sources(
                pa.Table.from_pylist([row], schema=SOURCE_SCHEMA),
                contract_id=config.split.contract_id,
                spans_per_fragment=config.execution.spans_per_fragment,
            )
        )


def test_arrow_expansion_preserves_parent_fields_across_chunks(config: ResolvedVideoCutConfig) -> None:
    """Empty/error sources and Arrow chunk boundaries do not shift a span to another source."""
    rows = [
        source_result(
            f"s3://test-bucket/{index}.mp4",
            VideoMetadata(duration, 64 + index, 48, 30.0, None, "h264"),
            50,
            config.split,
        )
        for index, duration in enumerate([100, 5_000_000_000, 100, 3_000_000_000])
    ]
    rows[2].update(status="error", error_stage="source-probe", error_message="bad media")
    outcomes = pa.concat_tables([pa.Table.from_pylist([row], schema=SOURCE_SCHEMA) for row in rows])
    expanded = list(
        publication.expand_sources(
            outcomes, contract_id=config.split.contract_id, spans_per_fragment=config.execution.spans_per_fragment
        )
    )
    assert [batch.num_rows for batch in expanded] == [2, 2, 1]
    expected = [
        {field.name: span[field.name] if field.name in span else row[field.name] for field in SPAN_SCHEMA}
        for row in rows
        for span in row["spans"]
    ]
    assert pa.concat_tables(expanded).equals(pa.Table.from_pylist(expected, schema=SPAN_SCHEMA))
