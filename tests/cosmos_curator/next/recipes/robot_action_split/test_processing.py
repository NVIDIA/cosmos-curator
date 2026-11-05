# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Chunk processing's failure policy: terminal outcomes for stable source problems, raises for the rest.

``cut_plan`` is replaced, so these run without an H.264 encoder; real cuts are
covered in ``test_pipeline_integration.py``.
"""

import json
import pickle
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import pyarrow as pa
import pytest
from botocore.exceptions import ClientError

from cosmos_curator.next.media.failures import CutError, FFmpegRunError, ProbeError, UnsupportedMediaError
from cosmos_curator.next.recipes.robot_action_split import processing
from cosmos_curator.next.recipes.robot_action_split.config import ResolvedRobotActionSplitConfig
from cosmos_curator.next.recipes.robot_action_split.contracts import CHUNK_SCHEMA, validate_chunk_results
from cosmos_curator.next.recipes.robot_action_split.discovery import discover_items
from cosmos_curator.next.utils import source_objects
from tests.cosmos_curator.next.recipes.robot_action_split.conftest import FRAMES_PER_SUBTASK, MAIN_VIEW


def _main_chunk(config: ResolvedRobotActionSplitConfig) -> dict[str, Any]:
    items = discover_items(config).to_pylist()
    return next(item for item in items if f"/{MAIN_VIEW}/" in item["chunk_mp4_uri"])


def _cut(
    _source: str,
    cuts: list[dict[str, Any]],
    *,
    on_cut: Callable[[dict[str, Any]], None],
    fail: frozenset[int] = frozenset(),
    transient: frozenset[int] = frozenset(),
    **_kwargs: object,
) -> list[dict[str, Any]]:
    """Stand in for ``cut_plan``: write each cut and hand its record to ``on_cut`` as the real one does.

    Indices in ``fail`` miss a frame; indices in ``transient`` fail as a full disk would.
    """
    records = []
    for index, cut in enumerate(cuts):
        expected = cut["endFrame"] - cut["startFrame"] + 1
        record: dict[str, Any] = {"output": cut["output"], "expected_frames": expected, "failure": None}
        if index in transient:
            failure: Exception = FFmpegRunError("failed to run cut: No space left on device")
            record |= {"success": False, "error": str(failure), "failure": failure}
        elif index in fail:
            failure = CutError(f"wrote {expected - 1}/{expected} frames")
            record |= {"success": False, "error": str(failure), "failure": failure, "written_frames": expected - 1}
        else:
            Path(cut["output"]).write_bytes(b"clip bytes")
            record |= {"success": True, "written_frames": expected}
        on_cut(record)
        records.append(record)
    return records


def _outcome(item: dict[str, Any], config: ResolvedRobotActionSplitConfig) -> dict[str, Any]:
    """Process one chunk and require a checkpointable outcome."""
    outcome = processing.cut_chunk(item, config=config)
    validate_chunk_results(
        pa.RecordBatch.from_pylist([outcome], schema=CHUNK_SCHEMA), media_root=config.output.media_root
    )
    return outcome


def test_finished_chunk_writes_every_payload_before_returning(
    config: ResolvedRobotActionSplitConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each clip, its action payload, and its sidecar are durable in the outcome's references."""
    monkeypatch.setattr(processing, "cut_plan", _cut)
    outcome = _outcome(_main_chunk(config), config)

    assert outcome["error_stage"] is None
    assert len(outcome["clips"]) == 2
    media_root = Path(config.output.media_root)
    for clip in outcome["clips"]:
        assert clip["error_stage"] is None
        assert Path(clip["clip_uri"].removeprefix("file://")).read_bytes() == b"clip bytes"
        action = pickle.loads(Path(clip["action_data_uri"].removeprefix("file://")).read_bytes())  # noqa: S301
        assert len(action["action"]) == FRAMES_PER_SUBTASK
        assert (media_root / "temp_metas" / f"{clip['clip_id']}.json").is_file()
        assert "dollies" in clip["camera_motion_annotation"].lower()


def test_cuts_run_with_the_task_cpus(config: ResolvedRobotActionSplitConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    """A chunk task reserves ``cut_cpus`` and runs that many cuts at once."""
    seen: dict[str, object] = {}

    def record(source: str, cuts: list[dict[str, Any]], **kwargs: Any) -> list[dict[str, Any]]:  # noqa: ANN401
        seen.update(kwargs)
        return _cut(source, cuts, **kwargs)

    monkeypatch.setattr(processing, "cut_plan", record)
    tuned = config.model_copy(update={"execution": config.execution.model_copy(update={"cut_cpus": 3.0})})
    processing.cut_chunk(_main_chunk(tuned), config=tuned)
    assert seen["workers"] == 3


def test_a_clip_that_misses_frames_fails_alone(
    config: ResolvedRobotActionSplitConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cut without the exact frames is terminal for that clip; its sibling still finishes."""
    monkeypatch.setattr(processing, "cut_plan", partial(_cut, fail=frozenset({1})))
    outcome = _outcome(_main_chunk(config), config)

    assert outcome["error_stage"] is None
    finished, failed = outcome["clips"]
    assert finished["error_stage"] is None
    assert (failed["error_stage"], failed["error_message"]) == ("cut", "wrote 99/100 frames")
    assert not (Path(config.output.media_root) / "temp_metas" / f"{failed['clip_id']}.json").exists()


def test_a_span_outside_the_chunk_fails_alone(
    config: ResolvedRobotActionSplitConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Source data that places a span before the chunk starts fails that clip, not the chunk or the run."""
    item = _main_chunk(config)
    spans = json.loads(item["spans_json"])
    spans[1]["episode_from_timestamp"] = -1000.0
    item["spans_json"] = json.dumps(spans)
    monkeypatch.setattr(processing, "cut_plan", _cut)
    outcome = _outcome(item, config)

    finished, failed = outcome["clips"]
    assert finished["error_stage"] is None
    assert failed["error_stage"] == "cut"
    assert "not a range in the chunk" in failed["error_message"]


def test_action_data_that_cannot_be_encoded_fails_its_clip(
    config: ResolvedRobotActionSplitConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unencodable action data is a property of the immutable source, so the clip fails without payloads."""
    monkeypatch.setattr(processing, "cut_plan", _cut)

    def unencodable(*_args: object, **_kwargs: object) -> bytes:
        msg = "field set does not match the registered spec"
        raise ValueError(msg)

    monkeypatch.setattr(processing, "_serialize_action_pickle", unencodable)
    outcome = _outcome(_main_chunk(config), config)

    assert {clip["error_stage"] for clip in outcome["clips"]} == {"action"}
    assert not (Path(config.output.media_root) / "video").exists()


def test_missing_data_parquet_fails_the_chunk(config: ResolvedRobotActionSplitConfig) -> None:
    """Without its data parquet no clip in the chunk can be produced."""
    item = _main_chunk(config)
    Path(item["data_parquet_uri"]).unlink()
    outcome = _outcome(item, config)
    assert outcome["error_stage"] == "load"
    assert {clip["error_stage"] for clip in outcome["clips"]} == {"load"}


def test_unparseable_data_parquet_fails_the_chunk(config: ResolvedRobotActionSplitConfig) -> None:
    """Corrupt parquet bytes in an immutable source never parse on a retry either."""
    item = _main_chunk(config)
    Path(item["data_parquet_uri"]).write_bytes(b"not a parquet file")
    assert _outcome(item, config)["error_stage"] == "parquet-parse"


def test_parquet_allocation_failure_stays_retryable(
    config: ResolvedRobotActionSplitConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Running out of memory while parsing says nothing about the parquet, so the chunk is not failed."""

    def out_of_memory(*_args: object, **_kwargs: object) -> None:
        msg = "malloc of size 1073741824 failed"
        raise pa.ArrowMemoryError(msg)

    monkeypatch.setattr(processing, "_load_action_arrays", out_of_memory)
    with pytest.raises(pa.ArrowMemoryError):
        _outcome(_main_chunk(config), config)


def test_missing_chunk_mp4_fails_the_chunk(config: ResolvedRobotActionSplitConfig) -> None:
    """A chunk MP4 that is not there fails every clip it would have held."""
    item = _main_chunk(config)
    Path(item["chunk_mp4_uri"]).unlink()
    assert _outcome(item, config)["error_stage"] == "chunk-stage"


def test_unindexable_chunk_fails_the_chunk(config: ResolvedRobotActionSplitConfig) -> None:
    """``cut_plan`` refuses a source whose packets do not index its frames; that does not change on a retry."""

    def unindexable(*_args: object, **_kwargs: object) -> list[dict[str, Any]]:
        msg = "packet index is not a frame index"
        raise UnsupportedMediaError(msg)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(processing, "cut_plan", unindexable)
        outcome = _outcome(_main_chunk(config), config)
    assert (outcome["error_stage"], outcome["error_message"]) == ("cut-index", "packet index is not a frame index")


def test_payload_writes_retry_then_raise(
    config: ResolvedRobotActionSplitConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A write that keeps failing leaves the chunk retryable rather than checkpointing a failure."""
    monkeypatch.setattr(processing, "cut_plan", _cut)
    attempts: list[str] = []

    def unavailable(location: str, _data: bytes, **_kwargs: object) -> None:
        attempts.append(location)
        msg = "storage unavailable"
        raise OSError(msg)

    monkeypatch.setattr(processing, "write_media", unavailable)
    tuned = config.model_copy(update={"execution": config.execution.model_copy(update={"media_write_attempts": 2})})
    monkeypatch.setattr("cosmos_curator.core.utils.misc.retry_utils.time.sleep", lambda _seconds: None)
    with pytest.raises(OSError, match="storage unavailable"):
        processing.cut_chunk(_main_chunk(tuned), config=tuned)
    assert len(attempts) == 2
    assert len(set(attempts)) == 1


def test_unclassified_read_failures_propagate(
    config: ResolvedRobotActionSplitConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only a source that stays unreadable is terminal; anything else leaves the chunk retryable."""

    def denied(*_args: object, **_kwargs: object) -> bytes:
        raise ClientError({"Error": {"Code": "AccessDenied", "Message": "denied"}}, "GetObject")

    monkeypatch.setattr(processing, "read_bytes", denied)
    with pytest.raises(ClientError):
        processing.cut_chunk(_main_chunk(config), config=config)


def test_inputs_are_read_with_the_input_profile_and_written_with_the_output_profile(
    config: ResolvedRobotActionSplitConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A split-store config reads sources and writes payloads through different profiles."""
    monkeypatch.setattr(processing, "cut_plan", _cut)
    reads: list[str | None] = []
    writes: set[str] = set()
    original = processing.get_storage_client

    def client(uri: str, *, profile_name: str = "default", **kwargs: object) -> object:
        reads.append(profile_name)
        return original(uri, profile_name=profile_name, **kwargs)

    def write(location: str, data: bytes, *, storage_profile: str) -> None:
        writes.add(storage_profile)
        Path(location).parent.mkdir(parents=True, exist_ok=True)
        Path(location).write_bytes(data)

    monkeypatch.setattr(processing, "get_storage_client", client)
    monkeypatch.setattr(processing, "write_media", write)
    split = config.model_copy(
        update={
            "input": config.input.model_copy(update={"storage_profile": "source-store"}),
            "execution": config.execution.model_copy(update={"storage_profile": "output-store"}),
        }
    )
    processing.cut_chunk(_main_chunk(config), config=split)
    assert set(reads) == {"source-store"}
    assert writes == {"output-store"}


def test_each_clip_is_uploaded_and_removed_as_its_cut_finishes(
    config: ResolvedRobotActionSplitConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Local disk holds the chunk and the clips in flight, not every clip of the chunk."""
    left_on_disk: list[bool] = []

    def cut_and_check(source: str, cuts: list[dict[str, Any]], *, on_cut: Callable[..., None], **kwargs: Any) -> Any:  # noqa: ANN401
        def check(record: dict[str, Any]) -> None:
            on_cut(record)
            left_on_disk.append(Path(record["output"]).exists())

        return _cut(source, cuts, on_cut=check, **kwargs)

    monkeypatch.setattr(processing, "cut_plan", cut_and_check)
    outcome = _outcome(_main_chunk(config), config)
    assert left_on_disk == [False, False]
    assert all(Path(clip["clip_uri"].removeprefix("file://")).is_file() for clip in outcome["clips"])


def test_an_environmental_cut_failure_raises_instead_of_publishing(
    config: ResolvedRobotActionSplitConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A full disk, timeout, or killed ffmpeg says nothing about the clip, so the chunk stays retryable."""
    monkeypatch.setattr(processing, "cut_plan", partial(_cut, transient=frozenset({0})))
    with pytest.raises(FFmpegRunError, match="No space left on device"):
        processing.cut_chunk(_main_chunk(config), config=config)


def test_an_environmental_probe_failure_raises_instead_of_failing_the_chunk(
    config: ResolvedRobotActionSplitConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only a source whose index is unusable becomes cut-index; a killed or starved probe is retried."""

    def starved(*_args: object, **_kwargs: object) -> list[dict[str, Any]]:
        msg = "packet index probe failed: Cannot allocate memory"
        raise ProbeError(msg)

    monkeypatch.setattr(processing, "cut_plan", starved)
    with pytest.raises(ProbeError):
        processing.cut_chunk(_main_chunk(config), config=config)


def test_a_raising_chunk_names_itself(config: ResolvedRobotActionSplitConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    """A chunk that keeps raising blocks its selection, so the error says which chunk it is."""
    monkeypatch.setattr(processing, "cut_plan", partial(_cut, transient=frozenset({0})))
    item = _main_chunk(config)
    with pytest.raises(FFmpegRunError) as raised:
        processing.cut_chunks(pa.Table.from_pylist([item]), config=config)
    assert any(item["chunk_id"] in note and item["chunk_mp4_uri"] in note for note in raised.value.__notes__)


class _FakeChunkClient:
    """Stand in for an S3 client: record each download's profile and fail the first ``failures`` attempts."""

    def __init__(self, source: Path, failures: list[Exception]) -> None:
        self._source = source
        self._failures = failures
        self.attempts = 0

    def download_to_path(self, _prefix: object, destination: str) -> None:
        self.attempts += 1
        if self._failures:
            raise self._failures.pop(0)
        Path(destination).write_bytes(self._source.read_bytes())


def _remote_chunk(
    config: ResolvedRobotActionSplitConfig, monkeypatch: pytest.MonkeyPatch, failures: list[Exception]
) -> tuple[dict[str, Any], _FakeChunkClient, list[str]]:
    """Point the main chunk at S3 and serve it through a fake client, recording the profiles used."""
    item = _main_chunk(config)
    client = _FakeChunkClient(Path(item["chunk_mp4_uri"]), failures)
    profiles: list[str] = []
    original = processing.get_storage_client

    def storage_client(uri: str, *, profile_name: str = "default", **kwargs: object) -> object:
        if uri.startswith("s3://"):
            profiles.append(profile_name)
            return client
        return original(uri, profile_name=profile_name, **kwargs)

    monkeypatch.setattr(processing, "get_storage_client", storage_client)
    monkeypatch.setattr(processing, "cut_plan", _cut)
    monkeypatch.setattr("cosmos_curator.core.utils.misc.retry_utils.time.sleep", lambda _seconds: None)
    return item | {"chunk_mp4_uri": "s3://source-bucket/videos/chunk-000/file-000.mp4"}, client, profiles


def test_a_remote_chunk_download_retries_with_the_input_profile(
    config: ResolvedRobotActionSplitConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A transfer that fails once is retried; the chunk is read through the input profile."""
    split = config.model_copy(update={"input": config.input.model_copy(update={"storage_profile": "source-store"})})
    item, client, profiles = _remote_chunk(split, monkeypatch, [ConnectionError("reset")])
    outcome = _outcome(item, split)
    assert outcome["error_stage"] is None
    assert client.attempts == 2
    assert profiles == ["source-store"]


def test_a_missing_remote_chunk_fails_the_chunk_after_retries(
    config: ResolvedRobotActionSplitConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bare 404 from the transfer's HEAD, in an existing bucket, is a missing chunk."""
    missing = [ClientError({"Error": {"Code": "404"}}, "HeadObject") for _ in range(3)]
    item, client, _profiles = _remote_chunk(config, monkeypatch, missing)
    monkeypatch.setattr(source_objects, "_bucket_exists", lambda *_args, **_kwargs: True)
    assert _outcome(item, config)["error_stage"] == "chunk-stage"
    assert client.attempts == config.execution.chunk_download_attempts


def test_a_denied_remote_chunk_propagates(
    config: ResolvedRobotActionSplitConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Credentials and permissions describe the environment, so the chunk stays retryable."""
    denied = [ClientError({"Error": {"Code": "AccessDenied"}}, "GetObject") for _ in range(3)]
    item, _client, _profiles = _remote_chunk(config, monkeypatch, denied)
    with pytest.raises(ClientError, match="AccessDenied"):
        processing.cut_chunk(item, config=config)
