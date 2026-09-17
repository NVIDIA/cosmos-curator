# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for fragment-atomic caption publication and reconciliation."""

from collections.abc import Callable, Iterator
from pathlib import Path

import lance
import pyarrow as pa
import pyarrow.fs as pafs
import pyarrow.parquet as pq
import pytest

from cosmos_curator.next.recipes.video_caption import publication
from cosmos_curator.next.recipes.video_caption.bindings import CaptionRunBindings
from cosmos_curator.next.recipes.video_caption.contracts import CaptionModelSpec, terminal_metadata
from cosmos_curator.next.recipes.video_caption.inference import result_schema
from cosmos_curator.next.recipes.video_caption.lance_state import CaptionAttempt, capture_attempt, ensure_caption_fields
from cosmos_curator.next.recipes.video_caption.publication import (
    PreparedFragmentPublication,
    PreparedFragmentUpdate,
    PublicationError,
    publish_staged_results,
)
from cosmos_curator.next.recipes.video_caption.workspace import CaptionWorkspace
from cosmos_curator.next.recipes.video_transcode.contracts import CLIP_SCHEMA
from tests.cosmos_curator.next.recipes.video_caption.helpers import append_clip_fragment, write_clip_fragment

_DISTRIBUTED_PUBLICATION_DATASET = publication._prepared_publication_dataset


def _workspace(tmp_path: Path) -> CaptionWorkspace:
    root = tmp_path / "caption-workspace"
    return CaptionWorkspace(
        root_uri=str(root),
        manifest_uri=str(root / "workspace.json"),
        results_uri=str(root / "results"),
        checkpoints_uri=str(root / "checkpoints"),
        filesystem=pafs.LocalFileSystem(),
    )


def _setup_attempt(  # integration fixture inputs stay explicit
    tmp_path: Path,
    factory: Callable[..., tuple[str, lance.LanceDataset]],
    spec: CaptionModelSpec,
    digest: str,
    bindings: CaptionRunBindings,
    *,
    count: int = 2,
    rows_per_fragment: int = 2,
) -> tuple[str, CaptionAttempt, CaptionWorkspace, pa.Table]:
    uri, _ = factory(count=count, rows_per_fragment=rows_per_fragment)
    dataset = ensure_caption_fields(uri, storage_options=None, bindings=bindings, attempts=3)
    attempt = capture_attempt(dataset, bindings=bindings)
    rows = []
    for fragment_id in attempt.pending_fragment_ids:
        fragment = dataset.get_fragment(fragment_id)
        assert fragment is not None
        clip_ids = fragment.to_table(columns=["clip_id"])["clip_id"].to_pylist()
        for row_offset, clip_id in enumerate(clip_ids):
            rows.append(
                {
                    "fragment_id": fragment_id,
                    "row_offset": row_offset,
                    "clip_id": clip_id,
                    spec.caption_field_name: f"caption for {clip_id}",
                    spec.metadata_field_name: terminal_metadata(
                        spec,
                        digest,
                        status="success",
                        prompt_token_count=10,
                        generated_token_count=20,
                        error_type=None,
                        error_message=None,
                    ),
                }
            )
    staged = pa.Table.from_pylist(rows, schema=result_schema(spec))
    workspace = _workspace(tmp_path)
    Path(workspace.results_uri).mkdir(parents=True)
    pq.write_table(staged, Path(workspace.results_uri) / "part-0.parquet")
    return uri, attempt, workspace, staged


@pytest.fixture(autouse=True)
def _run_publication_workers_in_process(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise publication contracts without starting a Ray cluster in unit tests."""
    if request.node.get_closest_marker("env") is not None:
        return

    def prepare_one(  # mirrors the explicit worker boundary
        fragment_id: int,
        *,
        uri: str,
        attempt_version: int,
        workspace: CaptionWorkspace,
        bindings: CaptionRunBindings,
        storage_options: dict[str, str] | None,
    ) -> PreparedFragmentPublication:
        payload = publication._prepare_fragment_from_workspace(
            fragment_id,
            uri=uri,
            attempt_version=attempt_version,
            workspace=workspace,
            bindings=bindings,
            storage_options=storage_options,
        )
        return publication._parse_prepared_publication(payload, expected_fragment_id=fragment_id)

    def validate_schemas(
        files: tuple[str, ...],
        workspace: CaptionWorkspace,
        bindings: CaptionRunBindings,
    ) -> None:
        paths = pa.Table.from_pylist([{"path": path} for path in files], schema=publication._RESULT_PATH_SCHEMA)
        publication._validate_parquet_schema_batch(
            paths,
            filesystem=workspace.filesystem,
            expected=bindings.result.staged_schema,
        )

    def prepare_all(  # mirrors the explicit Ray plan boundary
        files: tuple[str, ...],
        *,
        uri: str,
        attempt: CaptionAttempt,
        workspace: CaptionWorkspace,
        bindings: CaptionRunBindings,
        storage_options: dict[str, str] | None,
    ) -> tuple[PreparedFragmentPublication, ...]:
        del files
        return tuple(
            prepare_one(
                fragment_id,
                uri=uri,
                attempt_version=attempt.version,
                workspace=workspace,
                bindings=bindings,
                storage_options=storage_options,
            )
            for fragment_id in attempt.pending_fragment_ids
        )

    def iter_prepared(
        prepared: tuple[PreparedFragmentPublication, ...],
    ) -> Iterator[PreparedFragmentPublication]:
        yield from prepared

    def reprepare(  # mirrors the explicit retry task boundary
        fragment_id: int,
        *,
        uri: str,
        attempt: CaptionAttempt,
        workspace: CaptionWorkspace,
        bindings: CaptionRunBindings,
        storage_options: dict[str, str] | None,
    ) -> PreparedFragmentPublication:
        return prepare_one(
            fragment_id,
            uri=uri,
            attempt_version=attempt.version,
            workspace=workspace,
            bindings=bindings,
            storage_options=storage_options,
        )

    monkeypatch.setattr(publication, "_validate_parquet_schemas_distributed", validate_schemas)
    monkeypatch.setattr(publication, "_prepared_publication_dataset", prepare_all)
    monkeypatch.setattr(publication, "_iter_prepared_publications", iter_prepared)
    monkeypatch.setattr(publication, "_reprepare_fragment_with_ray", reprepare)


def _commit_disjoint_values(uri: str, *, field_name: str = "other_enrichment") -> None:
    dataset = lance.dataset(uri)
    if field_name not in dataset.schema.names:
        dataset.add_columns(pa.schema([pa.field(field_name, pa.string())]))
        dataset = lance.dataset(uri)
    fragment = dataset.get_fragment(0)
    assert fragment is not None
    clip_ids = fragment.to_table(columns=["clip_id"])["clip_id"].to_pylist()
    schema = pa.schema([pa.field("clip_id", pa.string()), pa.field(field_name, pa.string())])
    table = pa.Table.from_pylist(
        [{"clip_id": clip_id, field_name: f"other-{clip_id}"} for clip_id in clip_ids],
        schema=schema,
    )
    metadata, field_ids = fragment.update_columns(
        pa.RecordBatchReader.from_batches(schema, table.to_batches()),
        left_on="clip_id",
        right_on="clip_id",
    )
    transaction = lance.Transaction(
        read_version=dataset.version,
        operation=lance.LanceOperation.Update(updated_fragments=[metadata], fields_modified=field_ids),
    )
    lance.LanceDataset.commit(uri, transaction, max_retries=0)


def test_distributed_publication_plan_hash_groups_pending_fragments(
    tmp_path: Path,
    caption_spec: CaptionModelSpec,
    caption_bindings: CaptionRunBindings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The unit suite verifies Ray plan construction without executing the plan."""
    calls: dict[str, object] = {}

    class _DatasetPlan:
        def filter(self, *, expr: object) -> "_DatasetPlan":
            calls["filter"] = expr
            return self

        def groupby(self, key: str, *, num_partitions: int) -> "_DatasetPlan":
            calls["groupby"] = (key, num_partitions)
            return self

        def map_groups(self, function: object, **kwargs: object) -> "_DatasetPlan":
            calls["map_groups"] = (function, kwargs)
            return self

    plan = _DatasetPlan()

    def read_parquet(paths: list[str], **kwargs: object) -> "_DatasetPlan":
        calls["read_parquet"] = (paths, kwargs)
        return plan

    monkeypatch.setattr(publication.ray.data, "read_parquet", read_parquet)
    workspace = _workspace(tmp_path)
    attempt = CaptionAttempt(version=9, pending_fragment_ids=(2, 4), complete_fragment_ids=())

    result = _DISTRIBUTED_PUBLICATION_DATASET(
        ("part-0.parquet", "part-1.parquet"),
        uri=str(tmp_path / "clips.lance"),
        attempt=attempt,
        workspace=workspace,
        bindings=caption_bindings,
        storage_options=None,
    )

    assert result is plan
    paths, read_kwargs = calls["read_parquet"]
    assert paths == ["part-0.parquet", "part-1.parquet"]
    assert read_kwargs["filesystem"] is workspace.filesystem
    assert read_kwargs["schema"].equals(result_schema(caption_spec), check_metadata=True)
    assert read_kwargs["override_num_blocks"] == 2
    assert calls["filter"] is not None
    assert calls["groupby"] == ("fragment_id", 2)
    function, map_kwargs = calls["map_groups"]
    assert function is publication.prepare_fragment_publication
    assert map_kwargs["batch_format"] == "pyarrow"
    assert map_kwargs["zero_copy_batch"] is True
    assert map_kwargs["num_cpus"] == 1
    assert map_kwargs["memory"] == 8 * 1024**3


@pytest.mark.integration
def test_distributed_publication_executes_on_ray(
    tmp_path: Path,
    clip_dataset_factory: Callable[..., tuple[str, lance.LanceDataset]],
    caption_spec: CaptionModelSpec,
    caption_digest: str,
    caption_bindings: CaptionRunBindings,
) -> None:
    """Exercise the real Ray shuffle and worker boundary on a provisioned integration cluster."""
    uri, attempt, workspace, _ = _setup_attempt(
        tmp_path,
        clip_dataset_factory,
        caption_spec,
        caption_digest,
        caption_bindings,
        count=4,
        rows_per_fragment=2,
    )

    summary = publish_staged_results(
        uri,
        attempt,
        workspace,
        caption_bindings,
        storage_options=None,
        commit_attempts=3,
    )

    assert summary == publication.PublicationSummary(published_fragments=2, already_committed_fragments=0)


def test_publication_commits_each_fragment_atomically_and_recovers_exact_results(
    tmp_path: Path,
    clip_dataset_factory: Callable[..., tuple[str, lance.LanceDataset]],
    caption_spec: CaptionModelSpec,
    caption_digest: str,
    caption_bindings: CaptionRunBindings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every pending fragment commits independently and exact reruns skip it."""
    messages: list[str] = []

    class _RecordingLogger:
        @staticmethod
        def info(message: str, *args: object) -> None:
            messages.append(message.format(*args))

    monkeypatch.setattr(publication, "logger", _RecordingLogger())
    uri, attempt, workspace, staged = _setup_attempt(
        tmp_path,
        clip_dataset_factory,
        caption_spec,
        caption_digest,
        caption_bindings,
        count=4,
        rows_per_fragment=2,
    )

    first = publish_staged_results(
        uri,
        attempt,
        workspace,
        caption_bindings,
        storage_options=None,
        commit_attempts=3,
    )
    second = publish_staged_results(
        uri,
        attempt,
        workspace,
        caption_bindings,
        storage_options=None,
        commit_attempts=3,
    )

    assert first == publication.PublicationSummary(published_fragments=2, already_committed_fragments=0)
    assert second == publication.PublicationSummary(published_fragments=0, already_committed_fragments=2)
    assert any("Starting caption Lance publication" in message for message in messages)
    assert any("Caption fragment 0 published" in message and "commit Lance v" in message for message in messages)
    assert any("Caption fragment 0 is already canonical" in message for message in messages)
    assert any("Verifying caption publication" in message and "before cleanup" in message for message in messages)
    assert any(
        "Verified caption publication" in message and "2 selected fragment(s), 4 row(s) canonical" in message
        for message in messages
    )
    current = lance.dataset(uri).to_table(
        columns=["clip_id", caption_spec.caption_field_name, caption_spec.metadata_field_name]
    )
    assert current.to_pylist() == staged.select(current.schema.names).to_pylist()


@pytest.mark.parametrize("corruption", ["duplicate-offset", "missing-row", "wrong-digest"])
def test_incomplete_or_corrupt_staged_groups_never_publish(
    tmp_path: Path,
    clip_dataset_factory: Callable[..., tuple[str, lance.LanceDataset]],
    caption_spec: CaptionModelSpec,
    caption_digest: str,
    corruption: str,
    caption_bindings: CaptionRunBindings,
) -> None:
    """Missing, duplicate, or incompatible recovery rows cannot change Lance."""
    uri, attempt, workspace, staged = _setup_attempt(
        tmp_path, clip_dataset_factory, caption_spec, caption_digest, caption_bindings
    )
    rows = staged.to_pylist()
    if corruption == "duplicate-offset":
        rows[1]["row_offset"] = 0
    elif corruption == "missing-row":
        rows.pop()
    else:
        rows[0][caption_spec.metadata_field_name]["contract_digest"] = "wrong"
    pq.write_table(
        pa.Table.from_pylist(rows, schema=result_schema(caption_spec)),
        Path(workspace.results_uri) / "part-0.parquet",
    )

    with pytest.raises(PublicationError):
        publish_staged_results(
            uri,
            attempt,
            workspace,
            caption_bindings,
            storage_options=None,
            commit_attempts=2,
        )
    assert lance.dataset(uri).to_table(columns=[caption_spec.metadata_field_name])[0].null_count == 2


def test_schema_validation_rejects_incompatible_result_files(
    tmp_path: Path,
    clip_dataset_factory: Callable[..., tuple[str, lance.LanceDataset]],
    caption_spec: CaptionModelSpec,
    caption_digest: str,
    caption_bindings: CaptionRunBindings,
) -> None:
    """Every result footer is validated before preparing Lance descriptors."""
    uri, attempt, workspace, staged = _setup_attempt(
        tmp_path, clip_dataset_factory, caption_spec, caption_digest, caption_bindings
    )
    fields = [
        field.with_metadata(None) if field.name == caption_spec.caption_field_name else field for field in staged.schema
    ]
    incompatible = pa.Table.from_arrays(staged.columns, schema=pa.schema(fields))
    pq.write_table(incompatible, Path(workspace.results_uri) / "part-0.parquet")

    with pytest.raises(PublicationError, match="incompatible schema"):
        publish_staged_results(
            uri,
            attempt,
            workspace,
            caption_bindings,
            storage_options=None,
            commit_attempts=2,
        )


def test_disjoint_same_fragment_enrichment_is_preserved(
    tmp_path: Path,
    clip_dataset_factory: Callable[..., tuple[str, lance.LanceDataset]],
    caption_spec: CaptionModelSpec,
    caption_digest: str,
    caption_bindings: CaptionRunBindings,
) -> None:
    """A compatible enrichment file added before Phase B survives captioning."""
    uri, attempt, workspace, _ = _setup_attempt(
        tmp_path, clip_dataset_factory, caption_spec, caption_digest, caption_bindings
    )
    _commit_disjoint_values(uri)

    summary = publish_staged_results(
        uri,
        attempt,
        workspace,
        caption_bindings,
        storage_options=None,
        commit_attempts=3,
    )

    rows = lance.dataset(uri).to_table(columns=["other_enrichment", caption_spec.caption_field_name]).to_pylist()
    assert summary.published_fragments == 1
    assert [row["other_enrichment"] for row in rows] == ["other-clip-0", "other-clip-1"]
    assert all(row[caption_spec.caption_field_name] is not None for row in rows)


def test_stale_descriptor_is_discarded_and_restaged_after_a_disjoint_commit_race(
    tmp_path: Path,
    clip_dataset_factory: Callable[..., tuple[str, lance.LanceDataset]],
    caption_spec: CaptionModelSpec,
    caption_digest: str,
    monkeypatch: pytest.MonkeyPatch,
    caption_bindings: CaptionRunBindings,
) -> None:
    """A lost same-fragment race restages against latest without inference."""
    uri, attempt, workspace, _ = _setup_attempt(
        tmp_path, clip_dataset_factory, caption_spec, caption_digest, caption_bindings
    )
    original_commit = publication.commit_prepared_update
    calls = 0

    def racing_commit(
        target_uri: str,
        descriptor: PreparedFragmentUpdate,
        target_bindings: CaptionRunBindings,
        *,
        storage_options: dict[str, str] | None,
    ) -> int:
        nonlocal calls
        calls += 1
        if calls == 1:
            _commit_disjoint_values(uri)
        return original_commit(
            target_uri,
            descriptor,
            target_bindings,
            storage_options=storage_options,
        )

    monkeypatch.setattr(publication, "commit_prepared_update", racing_commit)

    summary = publish_staged_results(
        uri,
        attempt,
        workspace,
        caption_bindings,
        storage_options=None,
        commit_attempts=4,
    )

    assert calls == 2
    assert summary.published_fragments == 1
    rows = lance.dataset(uri).to_table(columns=["other_enrichment", caption_spec.caption_field_name]).to_pylist()
    assert all(row["other_enrichment"] is not None and row[caption_spec.caption_field_name] is not None for row in rows)


@pytest.mark.parametrize("commit_attempts", [1, 3])
def test_ambiguous_commit_response_is_resolved_from_canonical_state(  # integration inputs
    tmp_path: Path,
    clip_dataset_factory: Callable[..., tuple[str, lance.LanceDataset]],
    caption_spec: CaptionModelSpec,
    caption_digest: str,
    monkeypatch: pytest.MonkeyPatch,
    commit_attempts: int,
    caption_bindings: CaptionRunBindings,
) -> None:
    """A response lost after commit is resolved even on the last allowed attempt."""
    uri, attempt, workspace, _ = _setup_attempt(
        tmp_path, clip_dataset_factory, caption_spec, caption_digest, caption_bindings
    )
    original_commit = publication.commit_prepared_update
    called = False

    def ambiguous_commit(
        target_uri: str,
        descriptor: PreparedFragmentUpdate,
        target_bindings: CaptionRunBindings,
        *,
        storage_options: dict[str, str] | None,
    ) -> int:
        nonlocal called
        result = original_commit(
            target_uri,
            descriptor,
            target_bindings,
            storage_options=storage_options,
        )
        if not called:
            called = True
            msg = "response lost after durable commit"
            raise OSError(msg)
        return result

    monkeypatch.setattr(publication, "commit_prepared_update", ambiguous_commit)

    summary = publish_staged_results(
        uri,
        attempt,
        workspace,
        caption_bindings,
        storage_options=None,
        commit_attempts=commit_attempts,
    )

    assert called is True
    assert summary == publication.PublicationSummary(published_fragments=0, already_committed_fragments=1)


def test_split_owned_physical_change_is_rejected(
    tmp_path: Path,
    clip_dataset_factory: Callable[..., tuple[str, lance.LanceDataset]],
    caption_spec: CaptionModelSpec,
    caption_digest: str,
    caption_bindings: CaptionRunBindings,
) -> None:
    """A rewrite of video-transcode-owned physical data is never rebound."""
    uri, attempt, workspace, _ = _setup_attempt(
        tmp_path, clip_dataset_factory, caption_spec, caption_digest, caption_bindings
    )
    dataset = lance.dataset(uri)
    fragment = dataset.get_fragment(0)
    assert fragment is not None
    schema = pa.schema([pa.field("clip_id", pa.string()), pa.field("clip_uri", pa.large_string())])
    table = pa.Table.from_pylist(
        [
            {"clip_id": "clip-0", "clip_uri": "s3://rewritten/clip-0.mp4"},
            {"clip_id": "clip-1", "clip_uri": "s3://rewritten/clip-1.mp4"},
        ],
        schema=schema,
    )
    metadata, field_ids = fragment.update_columns(
        pa.RecordBatchReader.from_batches(schema, table.to_batches()),
        left_on="clip_id",
        right_on="clip_id",
    )
    transaction = lance.Transaction(
        read_version=dataset.version,
        operation=lance.LanceOperation.Update(updated_fragments=[metadata], fields_modified=field_ids),
    )
    lance.LanceDataset.commit(uri, transaction, max_retries=0)

    with pytest.raises(PublicationError, match="video-transcode-owned physical binding"):
        publish_staged_results(
            uri,
            attempt,
            workspace,
            caption_bindings,
            storage_options=None,
            commit_attempts=2,
        )


def test_split_append_after_attempt_waits_for_the_next_start(
    tmp_path: Path,
    clip_dataset_factory: Callable[..., tuple[str, lance.LanceDataset]],
    caption_spec: CaptionModelSpec,
    caption_digest: str,
    caption_bindings: CaptionRunBindings,
) -> None:
    """Fragments appended after V_attempt remain pending for the next start."""
    uri, attempt, workspace, _ = _setup_attempt(
        tmp_path, clip_dataset_factory, caption_spec, caption_digest, caption_bindings
    )
    appended = lance.dataset(uri, version=attempt.version).to_table(columns=CLIP_SCHEMA.names).slice(0, 1)
    replacements = {
        "source_id": pa.array(["source-99"], type=pa.string()),
        "source_uri": pa.array(["s3://input/source-99.mp4"], type=pa.large_string()),
        "clip_id": pa.array(["clip-99"], type=pa.string()),
        "clip_uri": pa.array(["s3://output/clips/clip-99.mp4"], type=pa.large_string()),
    }
    for name, values in replacements.items():
        appended = appended.set_column(appended.schema.get_field_index(name), name, values)
    appended = appended.cast(CLIP_SCHEMA)
    candidate = write_clip_fragment(appended, uri=uri, storage_profile="default")
    assert candidate is not None
    append_clip_fragment(candidate, uri=uri, storage_profile="default", attempts=3)

    publish_staged_results(
        uri,
        attempt,
        workspace,
        caption_bindings,
        storage_options=None,
        commit_attempts=3,
    )

    latest = lance.dataset(uri)
    appended_fragment = latest.get_fragment(1)
    assert appended_fragment is not None
    appended_values = appended_fragment.to_table(
        columns=[caption_spec.caption_field_name, caption_spec.metadata_field_name]
    )
    row = appended_values.to_pylist()[0]
    assert row == {caption_spec.caption_field_name: None, caption_spec.metadata_field_name: None}
