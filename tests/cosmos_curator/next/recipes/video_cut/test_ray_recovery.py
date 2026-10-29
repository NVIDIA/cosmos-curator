# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real local Ray execution with fresh processes and native checkpoint recovery."""

import json
import traceback
from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path
from typing import NoReturn
from unittest.mock import patch

import lance
import pyarrow as pa
import pytest
import ray
from botocore.exceptions import ClientError
from lance.fragment import FragmentMetadata
from ray.data.checkpoint import CheckpointConfig
from ray.data.checkpoint.checkpoint_writer import CheckpointWriter
from ray.data.context import DataContext

from cosmos_curator.core.utils.storage.s3_client import S3Prefix
from cosmos_curator.next.append import commit
from cosmos_curator.next.append.attempts import reconcile_and_select_attempt
from cosmos_curator.next.core.ray_checkpoints import STAGED_FILE_EXTENSION, read_staged_table, write_staged_table
from cosmos_curator.next.media.ffmpeg import ProbeError, VideoMetadata
from cosmos_curator.next.recipes.video_cut import discovery, fixed_stride, pipeline
from cosmos_curator.next.recipes.video_cut.bindings import resolve_bindings
from cosmos_curator.next.recipes.video_cut.config import (
    ExecutionConfig,
    FixedStrideConfig,
    ResolvedVideoCutConfig,
    resolve_config,
    resolve_config_data,
)
from cosmos_curator.next.recipes.video_cut.contracts import make_source_id
from cosmos_curator.next.recipes.video_cut.discovery import discover_items
from cosmos_curator.next.recipes.video_cut.fixed_stride import plan_sources
from tests.cosmos_curator.next.append.fresh_cluster import assert_fresh_cluster_resume
from tests.cosmos_curator.next.fresh_driver import run_in_fresh_driver, start_local_cluster
from tests.cosmos_curator.next.recipes.video_cut.conftest import RECIPE, plan_outcome

_MODULE = "tests.cosmos_curator.next.recipes.video_cut.test_ray_recovery"


def _probe(uri: str, *, execution: object, allowed: set[str], log: Path, failures: int) -> tuple[VideoMetadata, int]:
    """Stand in for remote FFprobe: record each source, fail its first ``failures`` probes, then settle it."""
    del execution
    assert uri in allowed, f"Checkpointed source reached probing: {uri}"
    with log.open("a") as stream:
        stream.write(uri + "\n")
    if failures:
        counter = log.with_name(make_source_id(uri) + ".attempts")
        attempts = int(counter.read_text()) if counter.exists() else 0
        counter.write_text(str(attempts + 1))
        if attempts < failures:
            msg = "injected FFprobe failure"
            raise ProbeError(msg)
    if uri.endswith("error.mp4"):
        raise ClientError({"Error": {"Code": "NoSuchKey", "Message": "synthetic missing source"}}, "HeadObject")
    duration = 100 if uri.endswith("short.mp4") else 5_000_000_000
    if "/large-" in uri:
        duration = 120_000_000_000_000
    return VideoMetadata(duration, 64, 48, 30.0, None, "h264"), 50


def _plan_with_probe(
    batch: pa.Table,
    *,
    split: FixedStrideConfig,
    execution: ExecutionConfig,
    probe: Callable[..., tuple[VideoMetadata, int]],
) -> pa.Table:
    """Replace only media access inside the worker; run the real source planning and failure policy."""
    with patch.object(fixed_stride, "probe_source", probe):
        return plan_sources(batch, split=split, execution=execution)


class _ListedSources:
    """Stand in for recursive S3 discovery with the URIs a test lists beside its config."""

    def __init__(self, listing: Path) -> None:
        self._listing = listing

    def list_recursive_with_suffixes(self, root: S3Prefix, suffixes: tuple[str, ...]) -> list[S3Prefix]:
        assert (root.path, suffixes) == ("s3://test-bucket/", (".mp4",))
        return [S3Prefix(uri) for uri in json.loads(self._listing.read_text())]


def _interrupted_commit(
    fragments: Sequence[FragmentMetadata], *, config: ResolvedVideoCutConfig, root: Path, **_kwargs: object
) -> NoReturn:
    """Record the staged fragment sizes, prove nothing is visible yet, and lose the driver before committing."""
    (root / "staged-fragments.json").write_text(json.dumps(sorted(fragment.num_rows for fragment in fragments)))
    assert lance.dataset(config.output.spans_lance_uri).count_rows() == 0
    msg = "injected driver loss before commit"
    raise RuntimeError(msg)


def _unexpected(*_args: object, **_kwargs: object) -> NoReturn:
    msg = "A completed publication reached Ray setup or fragment staging"
    raise AssertionError(msg)


def _assert_fails(run: Callable[[], object], message: str) -> None:
    """Require ``run`` to raise ``message`` somewhere in its exception chain, which Ray may wrap."""
    try:
        run()
    except Exception as exc:
        if message not in "".join(traceback.format_exception(exc)):
            raise
        return
    msg = f"The attempt did not fail with {message!r}"
    raise AssertionError(msg)


def run_attempt(config_path: str, allowed: str, faults: str) -> None:
    """Run ``run_config`` in this fresh driver and Ray head, then require a rerun to find nothing to do.

    Only media probing is replaced; datasource filtering, task planning,
    file-sink checkpoints, and Lance writes/commits are real. The cluster has no
    curator_io slots. ``faults`` can interrupt the commit and fail each
    source's first probes; its ``fails_with`` names the error the run must then
    raise, and the rerun is skipped. Probed URIs are logged to ``probes.txt``
    and the summary is written beside the config.
    """
    path = Path(config_path)
    config = resolve_config(path)
    allowed_uris = set(json.loads(allowed))
    injected = json.loads(faults)
    if config.input.root_uri is not None:
        discovery.s3_client = lambda _profile: _ListedSources(path.parent / "listed-sources.json")
    fixed_stride.plan_sources = partial(
        _plan_with_probe,
        probe=partial(
            _probe, allowed=allowed_uris, log=path.parent / "probes.txt", failures=injected["probe_failures"]
        ),
    )
    if injected["interrupt"]:
        commit.commit_append = partial(_interrupted_commit, config=config, root=path.parent)
    start_local_cluster(4, io_slots=0)
    try:
        if injected["fails_with"] is not None:
            _assert_fails(partial(pipeline.run_config, config), injected["fails_with"])
            return
        summary = pipeline.run_config(config)
        assert DataContext.get_current().checkpoint_config is None
        # A rerun finds nothing to do: no Ray setup, fragment staging, or commit.
        allowed_uris.clear()
        pipeline.ensure_ray_initialized = _unexpected
        commit.stage_fragment = _unexpected
        again = pipeline.run_config(config)
        assert (again["selected_items"], again["appended_rows"]) == (0, 0)
        assert again["spans_lance_version"] == summary["spans_lance_version"]
        Path(f"{config_path}.summary").write_text(json.dumps(summary))
    finally:
        ray.shutdown()


def _attempt(
    config: ResolvedVideoCutConfig,
    root: Path,
    name: str,
    allowed: Sequence[str],
    *,
    interrupt: bool = False,
    probe_failures: int = 0,
    fails_with: str | None = None,
) -> None:
    """Run one attempt with a new driver and Ray head; it must fail with ``fails_with`` when given."""
    path = root / f"{name}.json"
    path.write_text(config.model_dump_json())
    faults = {"interrupt": interrupt, "probe_failures": probe_failures, "fails_with": fails_with}
    run_in_fresh_driver(f"{_MODULE}:run_attempt", str(path), json.dumps(list(allowed)), json.dumps(faults))


def _summary(root: Path, name: str) -> dict[str, object]:
    return json.loads((root / f"{name}.json.summary").read_text())


def _workspace_root(config: ResolvedVideoCutConfig) -> Path:
    """Resolve the saved workspace of the config's sources without recording a new one."""
    assert config.input.uris is not None
    bindings = resolve_bindings(config)
    attempt = reconcile_and_select_attempt(bindings, lambda: discover_items(config))
    assert attempt.workspace is not None
    return Path(attempt.workspace.root)


def assert_source_resume(config_json: str, log_dir: str) -> None:
    """Run the shared fresh-cluster scenario with a remote probe that records each source it settles."""
    config = ResolvedVideoCutConfig.model_validate_json(config_json)
    log = Path(log_dir, "probed.txt")

    fixed_stride.plan_sources = partial(
        _plan_with_probe, probe=partial(_probe, allowed=set(config.input.uris or ()), log=log, failures=0)
    )
    summary = assert_fresh_cluster_resume(
        RECIPE,
        config,
        process=pipeline.cut_sources,
        work_logs=lambda: {
            "probe": [make_source_id(uri) for uri in log.read_text().splitlines()] if log.exists() else []
        },
        num_cpus=4,
    )
    assert (summary["appended_rows"], summary["errors"]) == (8, 1)


@pytest.mark.integration
def test_fresh_cluster_resumes_source_checkpoints(config: ResolvedVideoCutConfig, tmp_path: Path) -> None:
    """The real plan filters checkpointed sources on a fresh cluster and publishes through ``run_config``."""
    run_in_fresh_driver(f"{_MODULE}:assert_source_resume", config.model_dump_json(), str(tmp_path))


@pytest.mark.integration
@pytest.mark.parametrize("probe_failures", [1, 4])
def test_probe_retries_never_checkpoint_an_inconclusive_failure(
    config: ResolvedVideoCutConfig, tmp_path: Path, probe_failures: int
) -> None:
    """Probes retry within the attempt, or remain recoverable when retries are exhausted."""
    uri = "s3://test-bucket/long.mp4"
    raw = config.model_dump(mode="json")
    raw["input"]["uris"] = [uri]
    config = resolve_config_data(raw, overrides=["execution.rows_per_checkpoint=1"])
    exhausted = probe_failures == 4
    fails_with = "injected FFprobe failure" if exhausted else None
    _attempt(config, tmp_path, "probe", (uri,), probe_failures=probe_failures, fails_with=fails_with)
    if exhausted:
        dataset = lance.dataset(config.output.spans_lance_uri)
        assert dataset.count_rows() == 0
        assert not list(Path(_workspace_root(config), "results").glob(f"*.{STAGED_FILE_EXTENSION}"))
        _attempt(config, tmp_path, "probe", (uri,))
    summary = _summary(tmp_path, "probe")
    assert (summary["errors"], summary["selected_items"], summary["appended_rows"]) == (0, 1, 3)
    probes = (tmp_path / "probes.txt").read_text().splitlines()
    assert probes == [uri] * (probe_failures + 1)


@pytest.mark.integration
def test_interrupted_publication_resumes_without_media_work_and_matches_a_clean_run(
    config: ResolvedVideoCutConfig, tmp_path: Path
) -> None:
    """Staged but uncommitted fragments are rebuilt from checkpoints; a root discovery resumes explicit URIs."""
    config = resolve_config_data(
        config.model_dump(mode="json"), overrides=["execution.rows_per_checkpoint=1", "execution.rows_per_fragment=4"]
    )
    _attempt(
        config,
        tmp_path,
        "first",
        config.input.uris or (),
        interrupt=True,
        fails_with="injected driver loss before commit",
    )
    assert json.loads((tmp_path / "staged-fragments.json").read_text()) == [4, 4]
    unpublished = lance.dataset(config.output.spans_lance_uri)
    assert (unpublished.count_rows(), len(unpublished.get_fragments())) == (0, 0)
    workspace = _workspace_root(config)
    staged = list(Path(workspace, "results").glob(f"*.{STAGED_FILE_EXTENSION}"))
    assert sorted(read_staged_table(str(path)).num_rows for path in staged) == [1, 1, 1, 1]

    expected_uris = config.input.uris or ()
    (tmp_path / "listed-sources.json").write_text(json.dumps(list(expected_uris)))
    raw = config.model_dump(mode="json")
    raw["input"] = {"root_uri": "s3://test-bucket/"}
    resumed = resolve_config_data(raw, overrides=["execution.rows_per_checkpoint=2", "execution.rows_per_fragment=5"])
    _attempt(resumed, tmp_path, "resumed", ())
    assert not workspace.exists()
    summary = _summary(tmp_path, "resumed")
    assert (summary["selected_items"], summary["appended_rows"], summary["errors"]) == (4, 8, 1)
    assert summary["input_items"] is summary["remaining_items"] is None
    assert summary["cleanup_complete"] is True
    recovered = lance.dataset(resumed.output.spans_lance_uri)
    assert sorted(fragment.count_rows() for fragment in recovered.get_fragments()) == [3, 5]
    assert sorted((tmp_path / "probes.txt").read_text().splitlines()) == sorted(expected_uris)

    reference = resolve_config_data(
        resumed.model_dump(mode="json"),
        overrides=[
            f"output.spans_lance_uri={tmp_path / 'reference.lance'}",
            f"output.staging_root_uri={tmp_path / 'reference-staging'}",
            "execution.rows_per_checkpoint=3",
        ],
    )
    _attempt(reference, tmp_path, "clean", tuple(expected_uris))
    actual = recovered.to_table().sort_by([("source_id", "ascending"), ("span_id", "ascending")])
    expected = (
        lance.dataset(reference.output.spans_lance_uri)
        .to_table()
        .sort_by([("source_id", "ascending"), ("span_id", "ascending")])
    )
    assert actual.equals(expected, check_metadata=False)
    assert len(actual) == 8
    span_ids = actual["span_id"].drop_null().to_pylist()
    assert len(set(span_ids)) == len(span_ids) == 6


@pytest.mark.integration
def test_native_pending_write_is_replayed_within_the_saved_selection(
    config: ResolvedVideoCutConfig, tmp_path: Path
) -> None:
    """An interrupted file write is discarded and its selected source is recomputed."""
    workspace = _workspace_root(config)
    uri = "s3://test-bucket/long.mp4"
    writer = CheckpointWriter.create(
        CheckpointConfig(
            id_column="source_id", checkpoint_path=str(workspace / "checkpoints"), delete_checkpoint_on_success=False
        )
    )
    row = plan_outcome(uri, config)
    writer.write_pending_checkpoint(pa.array([row["source_id"]]), "interrupted")
    Path(workspace, "results").mkdir(parents=True)
    write_staged_table(
        pa.Table.from_pylist([row], schema=resolve_bindings(config).spec.result_fields),
        str(workspace / "results" / f"interrupted_000000.{STAGED_FILE_EXTENSION}"),
    )
    _attempt(config, tmp_path, "pending", config.input.uris or ())
    assert not workspace.exists()
    assert lance.dataset(config.output.spans_lance_uri).count_rows() == 8
    assert sorted((tmp_path / "probes.txt").read_text().splitlines()) == sorted(config.input.uris or ())


@pytest.mark.integration
@pytest.mark.parametrize("rows_per_checkpoint", [1, 2])
def test_full_sized_fragments_pack_source_plans(
    config: ResolvedVideoCutConfig, tmp_path: Path, rows_per_checkpoint: int
) -> None:
    """Two 60k-span plans pack into 100k/20k fragments within and across staged files."""
    raw = config.model_dump(mode="json")
    raw["input"] = {"uris": ["s3://test-bucket/large-1.mp4", "s3://test-bucket/large-2.mp4"]}
    config = resolve_config_data(
        raw,
        overrides=[f"execution.rows_per_checkpoint={rows_per_checkpoint}", "execution.rows_per_fragment=100000"],
    )
    _attempt(config, tmp_path, "large", config.input.uris or ())
    dataset = lance.dataset(config.output.spans_lance_uri)
    assert dataset.count_rows() == 120_000
    assert sorted(fragment.count_rows() for fragment in dataset.get_fragments()) == [20_000, 100_000]
    assert dataset.version == 2
    spans = dataset.to_table(columns=["span_id", "source_id"])
    assert len(set(spans["span_id"].to_pylist())) == 120_000
    assert len(set(spans["source_id"].to_pylist())) == 2


@pytest.mark.integration
def test_empty_and_error_only_sources_publish_and_resume(config: ResolvedVideoCutConfig, tmp_path: Path) -> None:
    """A selection without spans still publishes durable outcomes through Ray."""
    raw = config.model_dump(mode="json")
    raw["input"] = {"uris": ["s3://test-bucket/short.mp4", "s3://test-bucket/error.mp4"]}
    config = resolve_config_data(raw)
    _attempt(config, tmp_path, "empty", config.input.uris or ())
    summary = _summary(tmp_path, "empty")
    assert (summary["selected_items"], summary["appended_rows"], summary["errors"]) == (2, 2, 1)
    assert summary["cleanup_complete"] is True
    dataset = lance.dataset(config.output.spans_lance_uri)
    assert (dataset.version, dataset.count_rows()) == (2, 2)
    assert dataset.to_table(columns=["span_id"])["span_id"].null_count == 2
    errors = dataset.to_table(filter="error_stage IS NOT NULL", columns=["source_uri"]).to_pylist()
    assert errors == [{"source_uri": "s3://test-bucket/error.mp4"}]
