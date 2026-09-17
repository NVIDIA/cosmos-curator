# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real local Ray execution with fresh processes and native checkpoint recovery."""

import json
import subprocess
import sys
from pathlib import Path

import lance
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from ray.data.checkpoint import CheckpointConfig
from ray.data.checkpoint.checkpoint_writer import CheckpointWriter

from cosmos_curator.next.media.ffmpeg import VideoMetadata
from cosmos_curator.next.recipes.video_cut.config import ResolvedVideoCutConfig, resolve_config_data
from cosmos_curator.next.recipes.video_cut.contracts import SOURCE_SCHEMA, make_source_id
from cosmos_curator.next.recipes.video_cut.processing import source_result
from cosmos_curator.next.recipes.video_cut.publication import completed_publication, resolve_workspace
from cosmos_curator.next.utils.identity import canonical_digest

# Each attempt has a new driver and Ray head. Only metadata probing is replaced;
# datasource filtering, file-sink checkpoints, and Lance writes/commits are real.
_ATTEMPT = """
import json
import sys
from pathlib import Path

import pyarrow as pa
import ray
from ray.data.context import DataContext

from cosmos_curator.next.media.ffmpeg import ProbeError, VideoMetadata
from cosmos_curator.core.utils.storage.s3_client import S3Prefix
from cosmos_curator.next.recipes.video_cut import discovery, pipeline
from cosmos_curator.next.recipes.video_cut.config import resolve_config
from cosmos_curator.next.recipes.video_cut.contracts import SOURCE_SCHEMA, make_source_id
from cosmos_curator.next.recipes.video_cut.processing import source_result

config = resolve_config(Path(sys.argv[1]))
allowed = set(json.loads(sys.argv[2]))
log = Path(sys.argv[3])
interrupt = sys.argv[4] == "interrupt"
probe_failures = int(sys.argv[5])

if config.input.root_uri is not None:
    class ListedSources:
        def list_recursive_with_suffixes(self, root: S3Prefix, suffixes: tuple[str, ...]) -> list[S3Prefix]:
            assert root.path == "s3://test-bucket/"
            assert suffixes == (".mp4",)
            listed = Path(sys.argv[1]).parent / "listed-sources.json"
            return [S3Prefix(uri) for uri in json.loads(listed.read_text())]
    discovery.s3_client = lambda _: ListedSources()

def plan(batch, *, split, execution):
    rows = []
    for source in batch.to_pylist():
        uri = source["source_uri"]
        assert uri in allowed, f"Checkpointed source reached probing: {uri}"
        with log.open("a") as stream:
            stream.write(uri + "\\n")
        if probe_failures:
            counter = log.with_name(make_source_id(uri) + ".attempts")
            attempts = int(counter.read_text()) if counter.exists() else 0
            counter.write_text(str(attempts + 1))
            if attempts < probe_failures:
                raise ProbeError("injected FFprobe failure")
        duration = 100 if uri.endswith(("short.mp4", "error.mp4")) else 5_000_000_000
        if "/large-" in uri:
            duration = 120_000_000_000_000
        row = source_result(uri, VideoMetadata(duration, 64, 48, 30.0, None, "h264"), 50, split)
        if uri.endswith("error.mp4"):
            row.update(status="error", error_stage="source-read", error_message="synthetic missing source")
        rows.append(row)
    return pa.Table.from_pylist(rows, schema=SOURCE_SCHEMA)

pipeline.plan_sources = plan
def interrupted_commit(candidates, **kwargs):
    import lance
    assert sorted(lance.FragmentMetadata.from_json(item).num_rows for item in candidates) == [1, 4, 4]
    assert lance.dataset(config.output.spans_lance_uri).count_rows() == 0
    raise RuntimeError("injected driver loss before commit")
if interrupt:
    pipeline.commit_fragments = interrupted_commit

ray.init(address="local", num_cpus=4,
         include_dashboard=False, log_to_driver=False, object_store_memory=256 * 1024**2)
try:
    summary = pipeline.run_config(config)
    assert DataContext.get_current().checkpoint_config is None
    # A second successful attempt must be a no-op, including the empty plan.
    allowed.clear()
    def unexpected_stage(*args, **kwargs):
        raise AssertionError("Completed publication reached fragment staging")
    pipeline.stage_fragment = unexpected_stage
    def unexpected_ray(**kwargs):
        raise AssertionError("Completed publication reached Ray initialization")
    pipeline.ensure_ray_initialized = unexpected_ray
    def unexpected_cleanup(*args, **kwargs):
        raise AssertionError("Completed publication reached workspace cleanup")
    pipeline.cleanup_workspace = unexpected_cleanup
    again = pipeline.run_config(config)
    assert again == summary | {"cleanup_complete": None}
    Path(sys.argv[1] + ".summary").write_text(json.dumps(summary))
finally:
    ray.shutdown()
"""


def _attempt(
    config: ResolvedVideoCutConfig,
    root: Path,
    name: str,
    allowed: tuple[str, ...],
    *,
    interrupt: bool = False,
    probe_failures: int = 0,
) -> subprocess.CompletedProcess[str]:
    path = root / f"{name}.json"
    path.write_text(config.model_dump_json())
    return subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-c",
            _ATTEMPT,
            str(path),
            json.dumps(allowed),
            str(root / "probes.txt"),
            "interrupt" if interrupt else "complete",
            str(probe_failures),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=240,
    )


@pytest.mark.integration
@pytest.mark.parametrize("probe_failures", [1, 4])
def test_probe_retries_never_checkpoint_an_inconclusive_failure(
    config: ResolvedVideoCutConfig, tmp_path: Path, probe_failures: int
) -> None:
    """Probes retry within the attempt, or remain recoverable when retries are exhausted."""
    raw = config.model_dump(mode="json")
    raw["input"]["uris"] = [config.input.uris[0]]
    config = resolve_config_data(raw, overrides=["execution.sources_per_checkpoint=1"])
    first = _attempt(config, tmp_path, "probe", config.input.uris, probe_failures=probe_failures)
    if probe_failures == 4:
        assert first.returncode != 0
        assert "injected FFprobe failure" in first.stderr, first.stdout + first.stderr
        dataset = lance.dataset(config.output.spans_lance_uri)
        assert completed_publication(dataset) is None
        assert dataset.count_rows() == 0
        workspace = resolve_workspace(config, canonical_digest(config.input.uris))
        assert not list(Path(workspace.results).glob("*.parquet"))
        resumed = _attempt(config, tmp_path, "probe", config.input.uris)
        assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    else:
        assert first.returncode == 0, first.stdout + first.stderr
    summary = json.loads((tmp_path / "probe.json.summary").read_text())
    assert summary["errors"] == 0
    assert summary["source_results"] == 1
    assert summary["spans"] == 3
    probes = (tmp_path / "probes.txt").read_text().splitlines()
    assert probes == [config.input.uris[0]] * (probe_failures + 1)


@pytest.mark.integration
def test_fresh_driver_recovers_source_checkpoints_and_uncommitted_fragments(
    config: ResolvedVideoCutConfig, tmp_path: Path
) -> None:
    """Recovery skips durable sources in the frozen snapshot and matches a clean run."""
    raw = config.model_dump(mode="json")
    raw["input"]["uris"].extend(["s3://test-bucket/long-2.mp4", "s3://test-bucket/long-3.mp4"])
    config = resolve_config_data(
        raw, overrides=["execution.sources_per_checkpoint=1", "execution.spans_per_fragment=4"]
    )
    first = _attempt(config, tmp_path, "first", config.input.uris, interrupt=True)
    assert first.returncode != 0
    assert "injected driver loss before commit" in first.stderr, first.stdout + first.stderr
    partial = lance.dataset(config.output.spans_lance_uri)
    assert partial.count_rows() == 0
    assert len(partial.get_fragments()) == 0
    assert len(list((Path(config.output.spans_lance_uri) / "data").glob("*.lance"))) == 3
    workspace = resolve_workspace(config, canonical_digest(config.input.uris))
    initial_files = list(Path(workspace.results).glob("*.parquet"))
    assert sum(pq.read_metadata(path).num_rows for path in initial_files) == 4
    assert all(pq.read_metadata(path).num_rows == 1 for path in initial_files)

    raw = config.model_dump(mode="json")
    expected_uris = config.input.uris
    (tmp_path / "listed-sources.json").write_text(json.dumps(expected_uris))
    # A directory selection reuses the same source checkpoints as explicit URIs.
    raw["input"] = {"root_uri": "s3://test-bucket/"}
    resumed = resolve_config_data(
        raw, overrides=["execution.sources_per_checkpoint=2", "execution.spans_per_fragment=5"]
    )
    second = _attempt(resumed, tmp_path, "resumed", ())
    assert second.returncode == 0, second.stdout + second.stderr
    assert not Path(workspace.root).exists()
    summary = json.loads((tmp_path / "resumed.json.summary").read_text())
    assert summary["selected_sources"] == 4
    assert summary["source_results"] == 4
    assert summary["errors"] == 0
    assert summary["spans"] == 9
    assert summary["cleanup_complete"] is True
    recovered = lance.dataset(resumed.output.spans_lance_uri)
    assert sorted(fragment.count_rows() for fragment in recovered.get_fragments()) == [4, 5]
    assert recovered.version == 2
    probes = (tmp_path / "probes.txt").read_text().splitlines()
    assert sorted(probes) == list(expected_uris)

    reference = resolve_config_data(
        resumed.model_dump(mode="json"),
        overrides=[
            f"output.spans_lance_uri={tmp_path / 'reference.lance'}",
            f"output.staging_root_uri={tmp_path / 'reference-staging'}",
            "execution.sources_per_checkpoint=3",
        ],
    )
    clean = _attempt(reference, tmp_path, "clean", expected_uris)
    assert clean.returncode == 0, clean.stdout + clean.stderr
    actual = lance.dataset(resumed.output.spans_lance_uri).to_table().sort_by("span_id")
    expected = lance.dataset(reference.output.spans_lance_uri).to_table().sort_by("span_id")
    assert actual.equals(expected, check_metadata=False)
    assert len(set(actual["span_id"].to_pylist())) == len(actual) == 9


@pytest.mark.integration
def test_native_pending_write_is_replayed_within_frozen_snapshot(
    config: ResolvedVideoCutConfig, tmp_path: Path
) -> None:
    """An interrupted file write is discarded and its selected source is recomputed."""
    workspace = resolve_workspace(config, canonical_digest(config.input.uris))
    uri = config.input.uris[0]
    writer = CheckpointWriter.create(
        CheckpointConfig(
            id_column="source_id", checkpoint_path=workspace.checkpoints, delete_checkpoint_on_success=False
        )
    )
    writer.write_pending_checkpoint(pa.array([make_source_id(uri)]), "interrupted")
    Path(workspace.results).mkdir(parents=True)
    unfinished = Path(workspace.results) / "interrupted_000000.parquet"
    row = source_result(uri, VideoMetadata(5_000_000_000, 64, 48, 30.0, None, "h264"), 50, config.split)
    pq.write_table(pa.Table.from_pylist([row], schema=SOURCE_SCHEMA), unfinished)
    result = _attempt(config, tmp_path, "pending", config.input.uris)
    assert result.returncode == 0, result.stdout + result.stderr
    assert not Path(workspace.root).exists()
    assert lance.dataset(config.output.spans_lance_uri).count_rows() == 3
    assert sorted((tmp_path / "probes.txt").read_text().splitlines()) == list(config.input.uris)


@pytest.mark.integration
def test_resuming_after_an_overlapping_cut_run_filters_checkpointed_sources(
    config: ResolvedVideoCutConfig, tmp_path: Path
) -> None:
    """A sequential cut run may publish some checkpointed sources before the original run resumes."""
    raw = config.model_dump(mode="json")
    raw["input"]["uris"].extend(["s3://test-bucket/long-2.mp4", "s3://test-bucket/long-3.mp4"])
    config = resolve_config_data(
        raw, overrides=["execution.sources_per_checkpoint=1", "execution.spans_per_fragment=4"]
    )
    first = _attempt(config, tmp_path, "interrupted", config.input.uris, interrupt=True)
    assert first.returncode != 0
    assert "injected driver loss before commit" in first.stderr, first.stdout + first.stderr

    raw = config.model_dump(mode="json")
    raw["input"] = {"uris": [config.input.uris[0]]}
    subset = resolve_config_data(raw)
    second = _attempt(subset, tmp_path, "subset", subset.input.uris)
    assert second.returncode == 0, second.stdout + second.stderr
    existing = [
        fragment.metadata.to_json() for fragment in lance.dataset(config.output.spans_lance_uri).get_fragments()
    ]

    third = _attempt(config, tmp_path, "resumed-overlap", ())
    assert third.returncode == 0, third.stdout + third.stderr
    dataset = lance.dataset(config.output.spans_lance_uri)
    assert dataset.count_rows() == 9
    assert len(set(dataset.to_table(columns=["span_id"])["span_id"].to_pylist())) == 9
    assert [fragment.metadata.to_json() for fragment in dataset.get_fragments()[: len(existing)]] == existing
    summary = json.loads((tmp_path / "resumed-overlap.json.summary").read_text())
    assert summary["source_results"] == 3
    assert summary["selected_sources"] == 4


@pytest.mark.integration
@pytest.mark.parametrize("sources_per_checkpoint", [1, 2])
def test_full_sized_fragments_pack_source_plans(
    config: ResolvedVideoCutConfig, tmp_path: Path, sources_per_checkpoint: int
) -> None:
    """Two 60k-span plans pack into 100k/20k fragments within and across Parquet files."""
    raw = config.model_dump(mode="json")
    raw["input"] = {"uris": ["s3://test-bucket/large-1.mp4", "s3://test-bucket/large-2.mp4"]}
    config = resolve_config_data(
        raw,
        overrides=[f"execution.sources_per_checkpoint={sources_per_checkpoint}", "execution.spans_per_fragment=100000"],
    )
    result = _attempt(config, tmp_path, "large", config.input.uris)
    assert result.returncode == 0, result.stdout + result.stderr
    dataset = lance.dataset(config.output.spans_lance_uri)
    assert dataset.count_rows() == 120_000
    assert sorted(fragment.count_rows() for fragment in dataset.get_fragments()) == [20_000, 100_000]
    assert dataset.version == 2
    spans = dataset.to_table(columns=["span_id", "source_id"])
    assert len(set(spans["span_id"].to_pylist())) == 120_000
    assert len(set(spans["source_id"].to_pylist())) == 2


@pytest.mark.integration
def test_empty_and_error_only_sources_publish_and_resume(config: ResolvedVideoCutConfig, tmp_path: Path) -> None:
    """An entirely empty expansion still publishes counts and skips all work on replay."""
    raw = config.model_dump(mode="json")
    raw["input"] = {"uris": ["s3://test-bucket/short.mp4", "s3://test-bucket/error.mp4"]}
    config = resolve_config_data(raw)
    result = _attempt(config, tmp_path, "empty", config.input.uris)
    assert result.returncode == 0, result.stdout + result.stderr
    summary = json.loads((tmp_path / "empty.json.summary").read_text())
    assert summary["source_results"] == 2
    assert summary["errors"] == 1
    assert summary["spans"] == 0
    assert summary["cleanup_complete"] is True
    errors = [json.loads(line) for line in Path(summary["errors_uri"]).read_text().splitlines()]
    assert len(errors) == 1
    assert errors[0]["source_uri"] == "s3://test-bucket/error.mp4"
    assert errors[0]["error_message"] == "synthetic missing source"
    dataset = lance.dataset(config.output.spans_lance_uri)
    assert dataset.version == 2
    assert dataset.get_fragments() == []
    assert "workspace_root" not in completed_publication(dataset)
    assert not Path(resolve_workspace(config, canonical_digest(config.input.uris)).root).exists()
