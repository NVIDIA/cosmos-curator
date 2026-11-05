# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Consume real mp4-index output without a Ray cluster or a second video index."""

import json
import struct
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import av
import lance
import numpy as np
import pyarrow as pa
import pytest
from pydantic import ValidationError

from cosmos_curator.core.sensors.sensors import streaming_camera_sensor
from cosmos_curator.next.recipes.mp4_index.config import ResolvedMp4IndexConfig
from cosmos_curator.next.recipes.mp4_index.contracts import VIDEO_MANIFEST_SCHEMA, make_video_id
from cosmos_curator.next.recipes.mp4_index.processing import process_one_video, result_table
from cosmos_curator.next.recipes.multimodal_split import plan, split
from cosmos_curator.next.recipes.multimodal_split.config import (
    MultimodalSplitInputConfig,
    ResolvedMultimodalSplitConfig,
)
from cosmos_curator.next.recipes.multimodal_split.identities import make_split_contract_digest
from cosmos_curator.next.recipes.multimodal_split.split import split_session

from .conftest import DEVICE_ORIGIN_NS, selected_source_timestamps


@pytest.fixture
def indexed_camera(tmp_path: Path, make_session: Callable[..., Path]) -> tuple[Path, dict[str, Any], Path]:
    """Index a real camera using the producer, then expose its manifest row for fault injection."""
    session = make_session(tmp_path / "sessions", "session-000", frames=60, write_timestamps=False)
    media = session / "camera/front.mp4"
    config = ResolvedMp4IndexConfig.resolve_config_data(
        {
            "schema_version": 1,
            "kind": "mp4-index",
            "input": {"uris": [str(media)], "container_kind": "nonfragmented"},
            "output": {"path": str(tmp_path / "index")},
        }
    )
    result = process_one_video({"video_id": make_video_id(str(media)), "media_location": str(media)}, config=config)
    assert result["status"] == "success"
    row = result_table(result).to_pylist()[0]
    return session, row, tmp_path / "index/manifest.lance"


def _write_manifest(row: dict[str, Any], manifest: Path, *, duplicate: bool = False) -> None:
    lance.write_dataset(
        pa.Table.from_pylist([row] * (2 if duplicate else 1), schema=VIDEO_MANIFEST_SCHEMA), str(manifest)
    )


def _config(
    make_config: Callable[..., ResolvedMultimodalSplitConfig],
    session: Path,
    manifest: Path,
    *,
    alignment_fields: list[str] | None = None,
    whole_session: bool = False,
) -> ResolvedMultimodalSplitConfig:
    config = make_config(session.parent, session.parent.parent / "clips", whole_session=whole_session)
    data = config.model_dump(mode="json")
    data["input"]["mp4_index_uri"] = str(manifest)
    if alignment_fields is not None:
        data["sensors"]["front"]["camera_alignment_fields"] = alignment_fields
    return ResolvedMultimodalSplitConfig.model_validate(data)


def _plan(session: Path, config: ResolvedMultimodalSplitConfig) -> list[dict[str, Any]]:
    return plan.span_planner({"source_session_id": session.name, "session_uri": str(session)}, config=config)


def _forbid_direct_index(*_args: object, **_kwargs: object) -> None:
    pytest.fail("indexed planning and decoding must not index the source container")


def test_manifest_config_validates_locations(tmp_path: Path) -> None:
    """Accept a manifest location using the same storage rules as other inputs."""
    config = MultimodalSplitInputConfig(input_path_prefix=str(tmp_path), mp4_index_uri=str(tmp_path / "manifest.lance"))
    assert config.mp4_index_uri == str(tmp_path / "manifest.lance")
    with pytest.raises(ValidationError):
        MultimodalSplitInputConfig(input_path_prefix=str(tmp_path), mp4_index_uri="https://example.com/index")


def test_indexed_eof_drives_planning_and_playable_clips(
    indexed_camera: tuple[Path, dict[str, Any], Path],
    make_config: Callable[..., ResolvedMultimodalSplitConfig],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use indexed EOF through real decoding without external timestamp inputs."""
    session, row, manifest = indexed_camera
    # Deprecated input files must not affect indexed EOF reads.
    times = [DEVICE_ORIGIN_NS + round(i * 1e9 / 30) for i in range(60)]
    for frame, timestamp in zip(row["frames"], times, strict=True):
        frame["eof_ns"] = timestamp
    (session / "camera/front.mp4.timestamps.npy").write_bytes(b"broken sidecar")
    _write_manifest(row, manifest)
    monkeypatch.setattr(plan, "load_container_pts_ns", _forbid_direct_index)
    monkeypatch.setattr(plan, "load_sensor_timestamps_ns", _forbid_direct_index)
    monkeypatch.setattr("cosmos_curator.core.sensors.sensors.streaming_camera_sensor.index_video", _forbid_direct_index)
    config = _config(make_config, session, manifest)
    plans = _plan(session, config)
    assert len(plans) == 1
    assert plans[0]["timeline_ns"] == times
    assert plans[0]["spans"][0]["timestamp_source"] == "sensor"
    outcomes = split_session(plans[0], config=config)
    assert len(outcomes) == 2
    assert all(item["status"] == "success" for item in outcomes)
    for item in outcomes:
        with av.open(item["media_location"]) as container:
            assert len(list(container.decode(video=0))) == 30


@pytest.mark.parametrize("field", ["eof_ns", "moe_ns", "sof_ns", "pts_ns", "edited_pts_ns"])
def test_selected_camera_clock_drives_planning_decoding_and_provenance(
    indexed_camera: tuple[Path, dict[str, Any], Path],
    make_config: Callable[..., ResolvedMultimodalSplitConfig],
    field: str,
) -> None:
    """Every supported clock must survive the Arrow plan and drive actual frame delivery."""
    session, row, manifest = indexed_camera
    offsets = {"eof_ns": 0, "sof_ns": -1_000_000, "moe_ns": -2_000_000}
    ticks_per_frame = row["frames"][1]["dts_ticks"]
    row["metadata"]["elst_bytes"] = struct.pack(">I4sIIIihh", 28, b"elst", 0, 1, 2000, ticks_per_frame, 1, 0)
    for i, frame in enumerate(row["frames"]):
        frame["composition_offset_ticks"] = ticks_per_frame
        for column, offset in offsets.items():
            frame[column] = DEVICE_ORIGIN_NS + round(i * 1e9 / 30) + offset
    _write_manifest(row, manifest)
    config = _config(
        make_config,
        session,
        manifest,
        alignment_fields=[field, "eof_ns" if field != "eof_ns" else "moe_ns"],
        whole_session=True,
    )

    plans = pa.Table.from_pylist(_plan(session, config), schema=plan.PLANNING_RESULT_SCHEMA).to_pylist()

    assert len(plans) == 1
    assert plans[0]["camera_alignment_field"] == field
    start_ns = DEVICE_ORIGIN_NS + offsets[field] if field in offsets else 0
    if field == "pts_ns":
        start_ns = (
            ticks_per_frame * 1_000_000_000 * row["metadata"]["time_base_num"] // row["metadata"]["time_base_den"]
        )
    assert plans[0]["timeline_ns"][0] == start_ns
    outcomes = split_session(plans[0], config=config)
    assert len(outcomes) == 1
    assert all(outcome["status"] == "success" for outcome in outcomes)
    for outcome in outcomes:
        with av.open(outcome["media_location"]) as container:
            assert len(list(container.decode(video=0))) == 60
        selected = np.asarray(selected_source_timestamps(outcome))
        assert selected[0] == outcome["start_ns"]
        clip_manifest = json.loads((Path(outcome["media_location"]).parent / "clip.json").read_text())
        assert clip_manifest["camera_alignment_fields"] == {"front": field}


@pytest.mark.parametrize("partial", [False, True])
def test_camera_clock_fallback_skips_an_unavailable_column_for_the_entire_video(
    indexed_camera: tuple[Path, dict[str, Any], Path],
    make_config: Callable[..., ResolvedMultimodalSplitConfig],
    *,
    partial: bool,
) -> None:
    """Missing MOE or one missing MOE row selects the complete SOF column, without mixing clocks."""
    session, row, manifest = indexed_camera
    times = [DEVICE_ORIGIN_NS + round(i * 1e9 / 30) for i in range(60)]
    for i, frame in enumerate(row["frames"]):
        frame["eof_ns"] = times[i] + 2_000_000
        frame["sof_ns"] = times[i]
        frame["moe_ns"] = times[i] - 1_000_000 if partial and i else None
    _write_manifest(row, manifest)
    config = _config(make_config, session, manifest, alignment_fields=["moe_ns", "sof_ns", "eof_ns"])

    planned = _plan(session, config)[0]

    assert planned["camera_alignment_field"] == "sof_ns"
    assert planned["timeline_ns"] == times
    outcomes = split_session(planned, config=config)
    assert all(outcome["status"] == "success" for outcome in outcomes)
    assert selected_source_timestamps(outcomes[0]) == times[:30]


def test_invalid_preferred_camera_clock_does_not_fall_back(
    indexed_camera: tuple[Path, dict[str, Any], Path],
    make_config: Callable[..., ResolvedMultimodalSplitConfig],
) -> None:
    """A complete but non-monotonic preferred clock is a data error."""
    session, row, manifest = indexed_camera
    for i, frame in enumerate(row["frames"]):
        frame["eof_ns"] = DEVICE_ORIGIN_NS + round(i * 1e9 / 30)
        frame["moe_ns"] = DEVICE_ORIGIN_NS
    _write_manifest(row, manifest)
    config = _config(make_config, session, manifest, alignment_fields=["moe_ns", "eof_ns"])

    assert _plan(session, config) == []


def test_indexed_windows_continue_after_a_partial_encode_failure(
    indexed_camera: tuple[Path, dict[str, Any], Path],
    make_config: Callable[..., ResolvedMultimodalSplitConfig],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Indexed clips share one decode pass and preserve the next window after an encoder failure."""
    session, row, manifest = indexed_camera
    times = [DEVICE_ORIGIN_NS + round(i * 1e9 / 30) for i in range(60)]
    for frame, timestamp in zip(row["frames"], times, strict=True):
        frame["eof_ns"] = timestamp
    _write_manifest(row, manifest)
    config = _config(make_config, session, manifest)
    planned = _plan(session, config)[0]
    real_encode = split.encode_frames_to_path
    encode_calls = 0

    def fail_first_clip(frames: object, path: Path, **kwargs: object) -> int:
        nonlocal encode_calls
        encode_calls += 1
        if encode_calls == 1:
            for index, _frame in enumerate(frames):  # type: ignore[call-overload]
                if index == 5:
                    msg = "encoder failed"
                    raise RuntimeError(msg)
            return 0
        return real_encode(frames, path, **kwargs)  # type: ignore[arg-type]

    decoder = Mock(wraps=streaming_camera_sensor.StreamingVideoDecoder)
    monkeypatch.setattr(streaming_camera_sensor, "StreamingVideoDecoder", decoder)
    monkeypatch.setattr(streaming_camera_sensor, "index_video", _forbid_direct_index)
    monkeypatch.setattr(split, "encode_frames_to_path", fail_first_clip)

    outcomes = split_session(planned, config=config)

    assert [outcome["status"] for outcome in outcomes] == ["failure", "success"]
    assert selected_source_timestamps(outcomes[1]) == times[30:]
    decoder.assert_called_once()
    assert encode_calls == 2


def test_index_without_eof_refuses_deprecated_sidecar(
    indexed_camera: tuple[Path, dict[str, Any], Path],
    make_config: Callable[..., ResolvedMultimodalSplitConfig],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An obsolete external artifact cannot substitute for missing indexed EOF."""
    session, row, manifest = indexed_camera
    np.save(session / "camera/front.mp4.timestamps.npy", np.array([DEVICE_ORIGIN_NS], dtype=np.int64))
    _write_manifest(row, manifest)
    monkeypatch.setattr(plan, "load_container_pts_ns", _forbid_direct_index)
    assert _plan(session, _config(make_config, session, manifest)) == []


@pytest.mark.parametrize("allow_pts", [False, True])
def test_pts_fallback_uses_index_only_when_allowed(
    indexed_camera: tuple[Path, dict[str, Any], Path],
    make_config: Callable[..., ResolvedMultimodalSplitConfig],
    monkeypatch: pytest.MonkeyPatch,
    *,
    allow_pts: bool,
) -> None:
    """Require explicit permission before using container timestamps."""
    session, row, manifest = indexed_camera
    _write_manifest(row, manifest)
    monkeypatch.setattr(plan, "load_container_pts_ns", _forbid_direct_index)
    plans = _plan(
        session,
        _config(
            make_config, session, manifest, alignment_fields=["eof_ns", "edited_pts_ns"] if allow_pts else ["eof_ns"]
        ),
    )
    assert bool(plans) == allow_pts
    if plans:
        assert plans[0]["timeline_ns"][0] == 0
        assert plans[0]["spans"][0]["timestamp_source"] == "container_pts"


@pytest.mark.parametrize(
    "fault", ["missing", "duplicate", "stale_size", "stale_schema", "partial_eof", "unordered_eof", "count"]
)
def test_invalid_index_is_a_sensor_failure(
    indexed_camera: tuple[Path, dict[str, Any], Path],
    make_config: Callable[..., ResolvedMultimodalSplitConfig],
    fault: str,
) -> None:
    """Reject incomplete, ambiguous, or stale indexed inputs before splitting."""
    session, row, manifest = indexed_camera
    if fault == "missing":
        row["media_location"] += ".other"
    elif fault == "stale_size":
        row["media_size_bytes"] += 1
    elif fault == "stale_schema":
        row["record_schema_version"] -= 1
    elif fault == "partial_eof":
        row["frames"][0]["eof_ns"] = DEVICE_ORIGIN_NS
    elif fault == "unordered_eof":
        for frame in row["frames"]:
            frame["eof_ns"] = DEVICE_ORIGIN_NS
    elif fault == "count":
        row["frames"].pop()
    _write_manifest(row, manifest, duplicate=fault == "duplicate")
    config = _config(make_config, session, manifest)
    if fault == "stale_schema":
        with pytest.raises(ValueError, match="record_schema_version"):
            _plan(session, config)
        return
    assert _plan(session, config) == []
    assert list((Path(config.output.path) / "errors").rglob("*.json"))


def test_pts_fallback_ignores_deprecated_sidecar(
    indexed_camera: tuple[Path, dict[str, Any], Path], make_config: Callable[..., ResolvedMultimodalSplitConfig]
) -> None:
    """An obsolete file cannot override or block the explicitly requested indexed PTS fallback."""
    session, row, manifest = indexed_camera
    (session / "camera/front.mp4.timestamps.npy").write_bytes(b"obsolete timestamp file")
    _write_manifest(row, manifest)
    plans = _plan(session, _config(make_config, session, manifest, alignment_fields=["eof_ns", "edited_pts_ns"]))
    assert plans[0]["timeline_ns"][0] == 0
    assert plans[0]["spans"][0]["timestamp_source"] == "container_pts"


def test_index_input_changes_recovery_contract(
    indexed_camera: tuple[Path, dict[str, Any], Path], make_config: Callable[..., ResolvedMultimodalSplitConfig]
) -> None:
    """Changing timeline inputs must not reconcile old episodes as equivalent."""
    session, _, manifest = indexed_camera
    original = make_config(session.parent, session.parent.parent / "clips")
    indexed = _config(make_config, session, manifest)
    assert make_split_contract_digest(indexed) != make_split_contract_digest(original)


@pytest.mark.parametrize("clock", ["eof", "pts"])
def test_indexed_timeline_follows_presentation_order(
    indexed_camera: tuple[Path, dict[str, Any], Path],
    make_config: Callable[..., ResolvedMultimodalSplitConfig],
    clock: str,
) -> None:
    """Reordered samples must label frames in presentation order rather than inheriting DTS order."""
    session, row, manifest = indexed_camera
    # Decode samples 0, 1, 2 present as 1, 2, 0; later samples retain their order.
    presentation_rank = [2, 0, 1, *range(3, 60)]
    ticks_per_frame = row["frames"][1]["dts_ticks"] - row["frames"][0]["dts_ticks"]
    for sample, (frame, rank) in enumerate(zip(row["frames"], presentation_rank, strict=True)):
        frame["composition_offset_ticks"] = (rank - sample + 2) * ticks_per_frame
        if clock == "eof":
            frame["eof_ns"] = DEVICE_ORIGIN_NS + round(rank * 1e9 / 30)
    _write_manifest(row, manifest)
    config = _config(
        make_config, session, manifest, alignment_fields=["eof_ns", "edited_pts_ns"] if clock == "pts" else ["eof_ns"]
    )
    plans = _plan(session, config)
    assert len(plans) == 1
    times = plans[0]["timeline_ns"]
    assert np.all(np.diff(times) > 0)
    if clock == "eof":
        assert times[0] == DEVICE_ORIGIN_NS
    else:
        assert abs(times[0] - round(2e9 / 30)) <= 1
    assert abs(times[1] - times[0] - round(1e9 / 30)) <= 1


def test_local_uri_matches_the_producers_normalized_media_location(
    indexed_camera: tuple[Path, dict[str, Any], Path],
    make_config: Callable[..., ResolvedMultimodalSplitConfig],
) -> None:
    """The same indexed file must be found through file:// and plain path spellings."""
    session, row, manifest = indexed_camera
    for i, frame in enumerate(row["frames"]):
        frame["eof_ns"] = DEVICE_ORIGIN_NS + round(i * 1e9 / 30)
    _write_manifest(row, manifest)
    config = _config(make_config, session, manifest)
    data = config.model_dump(mode="json")
    data["input"]["input_path_prefix"] = session.parent.as_uri()
    data["input"]["mp4_index_uri"] = manifest.as_uri()
    config = ResolvedMultimodalSplitConfig.model_validate(data)
    plans = plan.span_planner({"source_session_id": session.name, "session_uri": session.as_uri()}, config=config)
    assert plans[0]["timeline_ns"][0] == DEVICE_ORIGIN_NS


def test_missing_configured_manifest_fails_before_ray_or_output_writes(
    tmp_path: Path,
    make_config: Callable[..., ResolvedMultimodalSplitConfig],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An invalid explicit input is a run error, detected before starting distributed work."""
    from cosmos_curator.next.recipes.multimodal_split import pipeline  # noqa: PLC0415

    config = _config(make_config, tmp_path / "sessions/session-000", tmp_path / "missing.lance")
    monkeypatch.setattr(pipeline, "ensure_ray_initialized", _forbid_direct_index)
    with pytest.raises(FileNotFoundError, match="Configured mp4-index manifest"):
        pipeline.run_config(config)
    assert not Path(config.output.path).exists()


@pytest.mark.integration
@pytest.mark.usefixtures("ray_local")
def test_indexed_session_publishes_lance_episodes(
    indexed_camera: tuple[Path, dict[str, Any], Path],
    make_config: Callable[..., ResolvedMultimodalSplitConfig],
) -> None:
    """Exercise indexed planning through Ray Data, publication, and a canonical Lance read."""
    from cosmos_curator.next.recipes.multimodal_split.pipeline import run_config  # noqa: PLC0415

    session, row, manifest = indexed_camera
    for i, frame in enumerate(row["frames"]):
        frame["eof_ns"] = DEVICE_ORIGIN_NS + round(i * 1e9 / 30)
    _write_manifest(row, manifest)
    config = _config(make_config, session, manifest)
    summary = run_config(config)
    assert summary["episodes_newly_published"] == 2
    assert summary["failures"] == 0
    episodes = lance.dataset(str(Path(config.output.path) / "episodes.lance")).to_table()
    assert episodes.num_rows == 2
    assert set(episodes["timestamp_source"].to_pylist()) == {"sensor"}


@pytest.mark.parametrize("supported", [False, True])
def test_eof_input_requires_supported_playback_edits(
    indexed_camera: tuple[Path, dict[str, Any], Path],
    make_config: Callable[..., ResolvedMultimodalSplitConfig],
    *,
    supported: bool,
) -> None:
    """Index construction rejects unsupported edits; supported shifts leave EOF unchanged."""
    session, row, manifest = indexed_camera
    row["metadata"]["elst_bytes"] = (
        struct.pack(">I4sIIIihh", 28, b"elst", 0, 1, 1000, 30, 1, 0) if supported else b"unsupported playback edit"
    )
    for i, frame in enumerate(row["frames"]):
        frame["eof_ns"] = DEVICE_ORIGIN_NS + round(i * 1e9 / 30)
    _write_manifest(row, manifest)
    config = _config(make_config, session, manifest)
    plans = _plan(session, config)
    if not supported:
        assert plans == []
        assert list((Path(config.output.path) / "errors").rglob("*.json"))
        return
    assert plans[0]["timeline_ns"][0] == DEVICE_ORIGIN_NS
    assert plans[0]["spans"][0]["timestamp_source"] == "sensor"
