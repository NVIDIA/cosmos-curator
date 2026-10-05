# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Atomic enrichment, concurrent captions, and durable recovery boundaries."""

import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import lance
import pyarrow as pa
import pyarrow.compute as pc
import pytest

from cosmos_curator.next.recipes.video_caption.contracts import (
    caption_contract_digest,
    terminal_metadata,
)
from cosmos_curator.next.recipes.video_caption.contracts import (
    resolve_model_spec as caption_spec,
)
from cosmos_curator.next.recipes.video_embed import pipeline, publication, state
from cosmos_curator.next.recipes.video_embed.config import ResolvedVideoEmbedConfig
from cosmos_curator.next.recipes.video_embed.contracts import ModelSpec
from cosmos_curator.next.recipes.video_transcode.contracts import CLIP_SCHEMA


def prepare_all(
    results: pa.Table, config: ResolvedVideoEmbedConfig, spec: ModelSpec, version: int
) -> list[dict[str, Any]]:
    """Exercise real worker staging directly, without starting Ray in unit tests."""
    candidates = []
    for fragment_id in pc.unique(results["fragment_id"]).to_pylist():
        group = results.filter(pc.equal(results["fragment_id"], fragment_id))
        candidates.extend(publication.prepare_fragment(group, config=config, spec=spec, version=version).to_pylist())
    return candidates


def publish(
    results: pa.Table, config: ResolvedVideoEmbedConfig, spec: ModelSpec, dataset: lance.LanceDataset
) -> lance.LanceDataset:
    """Publish a finite result with the same reprepare boundary as the Ray driver."""
    return publication.commit_fragments(
        prepare_all(results, config, spec, int(dataset.version)),
        config=config,
        spec=spec,
        dataset=dataset,
        workspace=state.resolve_workspace(config, dataset, spec),
        pending=state.pending_fragment_ids(dataset, spec),
        reprepare=lambda version: prepare_all(results, config, spec, version),
    )


def test_one_commit_preserves_rows_and_base_files(
    results: pa.Table,
    config: ResolvedVideoEmbedConfig,
    spec: ModelSpec,
    dataset: lance.LanceDataset,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """All fragments become terminal in one version, and a rerun does no media/model work."""
    published = publish(results, config, spec, dataset)
    assert published.version == dataset.version + 1
    assert state.pending_fragment_ids(published, spec) == ()
    assert published.to_table(columns=[spec.vector_field])[spec.vector_field].null_count == 1
    for before, after in zip(dataset.get_fragments(), published.get_fragments(), strict=True):
        assert before.fragment_id == after.fragment_id
        assert before.metadata.to_json()["files"] == after.metadata.to_json()["files"][:1]
        assert after.metadata.num_deletions == 0
    transaction = published.read_transaction(published.version)
    assert transaction.transaction_properties["kind"] == "video-embed"
    assert len(transaction.operation.updated_fragments) == 2

    def unexpected(**_kwargs: object) -> None:
        pytest.fail("Completed embedding rerun must not start Ray or load a model")

    monkeypatch.setattr(pipeline, "ensure_ray_initialized", unexpected)
    assert pipeline.run_config(config)["clips_lance_version"] == published.version


def test_run_verifies_each_snapshot_once_before_cleanup(
    results: pa.Table,
    config: ResolvedVideoEmbedConfig,
    spec: ModelSpec,
    dataset: lance.LanceDataset,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Publication verifies once; a completed rerun still verifies before removing recovery state."""
    workspace = state.resolve_workspace(config, dataset, spec)
    verify = Mock(wraps=publication.verify_completed)
    monkeypatch.setattr(publication, "verify_completed", verify)
    monkeypatch.setattr(pipeline, "verify_completed", verify)
    for name in (
        "ensure_ray_initialized",
        "configure_ray_data_progress",
        "configure_ray_data_stability",
        "embed_clips",
    ):
        monkeypatch.setattr(pipeline, name, Mock())
    monkeypatch.setattr(
        pipeline,
        "publish_results",
        lambda snapshot, *_args, **_kwargs: publish(results, config, spec, snapshot),
    )
    remove_prefix = pipeline.remove_prefix

    def cleanup(uri: str, *, storage_profile: str) -> None:
        verify.assert_called_once()
        remove_prefix(uri, storage_profile=storage_profile)

    monkeypatch.setattr(pipeline, "remove_prefix", cleanup)
    summary = pipeline.run_config(config)
    assert summary["clips_lance_version"] == dataset.version + 1
    assert summary["pending_fragments"] == 2
    assert summary["cleanup_complete"] is True
    assert not Path(workspace.root).exists()

    verify.reset_mock()
    state.ensure_workspace(workspace, config)
    summary = pipeline.run_config(config)
    assert summary["pending_fragments"] == 0
    assert summary["cleanup_complete"] is True
    verify.assert_called_once()
    assert not Path(workspace.root).exists()


def _write_caption(uri: str, commit: Callable[..., lance.LanceDataset]) -> None:
    current = lance.dataset(uri)
    spec = caption_spec("qwen3_8_27b_fp8")
    current.add_columns(spec.fields)
    fragment = current.get_fragment(0)
    rows = fragment.to_table(columns=["clip_id"]).to_pylist()
    metadata = terminal_metadata(
        spec,
        caption_contract_digest(spec),
        status="success",
        prompt_token_count=10,
        generated_token_count=5,
        error_type=None,
        error_message=None,
    )
    values = pa.Table.from_pylist(
        [row | {spec.caption_field_name: "keep this caption", spec.metadata_field_name: metadata} for row in rows],
        schema=pa.schema([("clip_id", pa.string()), *spec.fields]),
    )
    prepared, fields = fragment.update_columns(values, left_on="clip_id", right_on="clip_id")
    commit(
        uri,
        lance.LanceOperation.Update(updated_fragments=[prepared], fields_modified=fields),
        read_version=current.version,
    )


@pytest.mark.parametrize("during_commit", [False, True])
def test_caption_race_reprepares_only_column_files(
    results: pa.Table,
    config: ResolvedVideoEmbedConfig,
    spec: ModelSpec,
    dataset: lance.LanceDataset,
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
    *,
    during_commit: bool,
) -> None:
    """Preserve same-fragment captions both before and inside the embedding commit race."""
    workspace = state.resolve_workspace(config, dataset, spec)
    pending = state.pending_fragment_ids(dataset, spec)
    candidates = prepare_all(results, config, spec, int(dataset.version))
    commit = lance.LanceDataset.commit
    raced = False
    versions = []

    def race(*args: Any, **kwargs: Any) -> lance.LanceDataset:  # noqa: ANN401 -- preserve Lance's overloads
        nonlocal raced
        if not raced:
            raced = True
            _write_caption(dataset.uri, commit)
        return commit(*args, **kwargs)

    def reprepare(version: int) -> list[dict[str, Any]]:
        versions.append(version)
        return prepare_all(results, config, spec, version)

    if during_commit:
        monkeypatch.setattr(lance.LanceDataset, "commit", race)
    else:
        _write_caption(dataset.uri, commit)
    published = publication.commit_fragments(
        candidates,
        config=config,
        spec=spec,
        dataset=dataset,
        workspace=workspace,
        pending=pending,
        reprepare=reprepare,
    )
    assert len(versions) == 1
    assert state.resolve_workspace(config, published, spec) == workspace
    assert published.to_table(columns=[caption_spec("qwen3_8_27b_fp8").caption_field_name]).column(0).to_pylist() == [
        "keep this caption",
        "keep this caption",
        None,
        None,
    ]
    assert published.to_table(columns=spec.fields.names).equals(results.select(spec.fields.names).cast(spec.fields))
    captured = capfd.readouterr()
    if during_commit:
        assert "retry" in (captured.out + captured.err).lower()


def test_ambiguous_commit_uses_durable_receipt(
    results: pa.Table,
    config: ResolvedVideoEmbedConfig,
    spec: ModelSpec,
    dataset: lance.LanceDataset,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A lost success response does not cause a duplicate commit or repeated inference."""
    commit = lance.LanceDataset.commit
    calls = 0

    def lose_response(*args: Any, **kwargs: Any) -> lance.LanceDataset:  # noqa: ANN401
        nonlocal calls
        calls += 1
        commit(*args, **kwargs)
        msg = "lost commit response"
        raise OSError(msg)

    monkeypatch.setattr(lance.LanceDataset, "commit", lose_response)
    published = publish(results, config, spec, dataset)
    assert calls == 1
    assert published.version == dataset.version + 1


@pytest.mark.parametrize("fault", ["missing", "duplicate", "wrong_clip", "wrong_offset", "wrong_dimension", "nan"])
def test_bad_staged_rows_leave_all_fragments_pending(
    results: pa.Table,
    config: ResolvedVideoEmbedConfig,
    spec: ModelSpec,
    dataset: lance.LanceDataset,
    fault: str,
) -> None:
    """Coverage and value validation happen before the one canonical commit."""
    rows = results.to_pylist()
    if fault == "missing":
        rows.pop(0)
    elif fault == "duplicate":
        rows.append(rows[0])
    elif fault == "wrong_clip":
        rows[0]["clip_id"] = "another-clip"
    elif fault == "wrong_offset":
        rows[0]["row_offset"] = 100
    elif fault == "wrong_dimension":
        rows[0][spec.vector_field] = [1.0]
    else:
        rows[0][spec.vector_field][0] = float("nan")
    with pytest.raises(ValueError, match=r"Missing|finite unit"):
        publish(pa.Table.from_pylist(rows, schema=spec.result_schema), config, spec, dataset)
    assert lance.dataset(dataset.uri).version == dataset.version
    assert state.pending_fragment_ids(lance.dataset(dataset.uri), spec) == (0, 1)


def test_missing_whole_fragment_is_rejected(
    results: pa.Table,
    config: ResolvedVideoEmbedConfig,
    spec: ModelSpec,
    dataset: lance.LanceDataset,
) -> None:
    """A successful first group's staging cannot hide absent work in another fragment."""
    with pytest.raises(ValueError, match="exactly cover"):
        publish(results.slice(0, 2), config, spec, dataset)
    assert lance.dataset(dataset.uri).version == dataset.version


def test_changed_input_is_rejected(
    results: pa.Table,
    config: ResolvedVideoEmbedConfig,
    spec: ModelSpec,
    dataset: lance.LanceDataset,
) -> None:
    """Same IDs cannot rebind staged embeddings to different media."""
    current = lance.dataset(dataset.uri)
    fragment = current.get_fragment(0)
    rows = fragment.to_table(columns=["clip_id"])
    rows = rows.append_column("clip_uri", pa.array(["different.mp4"] * len(rows), type=pa.large_string()))
    prepared, fields = fragment.update_columns(rows, left_on="clip_id", right_on="clip_id")
    lance.LanceDataset.commit(
        dataset.uri,
        lance.LanceOperation.Update(updated_fragments=[prepared], fields_modified=fields),
        read_version=current.version,
    )
    with pytest.raises(ValueError, match="physical bindings changed"):
        publish(results, config, spec, dataset)


def test_workspace_tuning_and_manifest(
    config: ResolvedVideoEmbedConfig,
    spec: ModelSpec,
    dataset: lance.LanceDataset,
) -> None:
    """Batching changes reuse checkpoints; incompatible manifests fail closed."""
    workspace = state.resolve_workspace(config, dataset, spec)
    state.ensure_workspace(workspace, config)
    tuned = config.model_copy(update={"execution": config.execution.model_copy(update={"inference_batch_size": 8})})
    assert state.resolve_workspace(tuned, dataset, spec) == workspace
    state.ensure_workspace(workspace, tuned)
    path = Path(workspace.root, "workspace.json")
    path.write_text(json.dumps({"wrong": "scope"}), encoding="utf-8")
    with pytest.raises(ValueError, match="workspace does not match"):
        state.ensure_workspace(workspace, config)


@pytest.mark.parametrize("fault", ["duplicate_id", "null_uri", "zero_size"])
def test_invalid_input_refuses_media_work(
    config: ResolvedVideoEmbedConfig,
    spec: ModelSpec,
    dataset: lance.LanceDataset,
    fault: str,
) -> None:
    """The checkpoint key and media descriptors must be trustworthy before inference."""
    rows = dataset.to_table(columns=CLIP_SCHEMA.names).to_pylist()
    if fault == "duplicate_id":
        rows[0]["clip_id"] = rows[1]["clip_id"]
    elif fault == "null_uri":
        rows[0]["clip_uri"] = None
    else:
        rows[0]["clip_size_bytes"] = 0
    lance.write_dataset(
        pa.Table.from_pylist(rows, schema=CLIP_SCHEMA.with_metadata(dataset.schema.metadata)),
        dataset.uri,
        mode="overwrite",
        data_storage_version="2.2",
    )
    with pytest.raises(ValueError, match="globally unique"):
        state.open_input(config, spec)


def test_changed_model_contract_does_not_overwrite_existing_vectors(
    results: pa.Table,
    config: ResolvedVideoEmbedConfig,
    spec: ModelSpec,
    dataset: lance.LanceDataset,
) -> None:
    """A different model revision requires a new field contract, not a silent replacement."""
    published = publish(results, config, spec, dataset)
    different = replace(spec, revision="a" * 40)
    with pytest.raises(ValueError, match="Incompatible or missing embedding provenance"):
        state.pending_fragment_ids(published, different)


def test_completed_rerun_preserves_an_incompatible_workspace(
    results: pa.Table,
    config: ResolvedVideoEmbedConfig,
    spec: ModelSpec,
    dataset: lance.LanceDataset,
) -> None:
    """Canonical success does not authorize deleting recovery state with a mismatched manifest."""
    workspace = state.resolve_workspace(config, dataset, spec)
    state.ensure_workspace(workspace, config)
    path = Path(workspace.root, "workspace.json")
    path.write_text('{"incompatible": true}', encoding="utf-8")
    publish(results, config, spec, dataset)
    with pytest.raises(ValueError, match="workspace does not match"):
        pipeline.run_config(config)
    assert path.exists()
