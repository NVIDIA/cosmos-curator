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
"""Per-video indexing over real clips, compared with PyAV demuxing the whole source.

``test_processing.py`` covers the box-format and caching paths with hand-packed files.
"""

import io
import pathlib
from typing import Any

import attrs
import av
import lance
import numpy as np
import pyarrow as pa
import pytest
import smart_open  # type: ignore[import-untyped]

from cosmos_curator.core.sensors.data.video import VideoProbe
from cosmos_curator.core.sensors.sampling.grid import SamplingGrid
from cosmos_curator.core.sensors.sampling.policy import NearestTimestampPolicy
from cosmos_curator.core.sensors.sampling.spec import SamplingSpec
from cosmos_curator.core.sensors.sensors.streaming_camera_sensor import StreamingCameraSensor
from cosmos_curator.core.sensors.types.types import VideoIndexCreationMethod
from cosmos_curator.core.sensors.utils.video import video_probe
from cosmos_curator.next.recipes.mp4_index import processing
from cosmos_curator.next.recipes.mp4_index.config import ResolvedMp4IndexConfig
from cosmos_curator.next.recipes.mp4_index.contracts import (
    CONTAINER_KIND_FRAGMENTED,
    CONTAINER_KIND_NONFRAGMENTED,
    make_video_id,
    read_video_record,
)
from cosmos_curator.next.recipes.mp4_index.index_lookup import Mp4IndexLookup
from tests.cosmos_curator.core.sensors.mp4 import _real_fixtures as real_mp4
from tests.cosmos_curator.core.utils.storage.conftest import FakeStorageClient


class _NoOpExtension:
    """An extension that registers no UUID decoders, so the box walker indexes the video."""

    uuid_handlers = ()
    frame_schema = pa.schema([pa.field("eof_ns", pa.int64())])

    def frame_columns(self, extraction: processing.Mp4Extraction) -> pa.RecordBatch:
        return pa.RecordBatch.from_arrays(
            [pa.nulls(len(extraction.video_track.size), type=pa.int64())], schema=self.frame_schema
        )


def _config(tmp_path: pathlib.Path, *, container_kind: str) -> ResolvedMp4IndexConfig:
    return ResolvedMp4IndexConfig.model_validate(
        {
            "schema_version": 1,
            "kind": "mp4-index",
            "input": {"path_prefix": str(tmp_path), "globs": ["*.mp4"], "container_kind": container_kind},
            "output": {"path": str(tmp_path / "output")},
        }
    )


def _process(tmp_path: pathlib.Path, media_location: str, *, container_kind: str, **kwargs: object) -> dict[str, Any]:
    return processing.process_one_video(
        {"video_id": make_video_id(media_location), "media_location": media_location},
        config=_config(tmp_path, container_kind=container_kind),
        **kwargs,  # type: ignore[arg-type]
    )


def _write(tmp_path: pathlib.Path, data: bytes) -> str:
    media_path = tmp_path / "session-a" / "unit-00" / "video-1.mp4"
    media_path.parent.mkdir(parents=True)
    media_path.write_bytes(data)
    return str(media_path)


def _assert_frames_match_the_source(result: dict[str, Any], data: bytes) -> None:
    probe: VideoProbe = video_probe(data, index_method=VideoIndexCreationMethod.FULL_DEMUX)
    assert result["status"] == "success", result["error_message"]
    frame_table, metadata_row = read_video_record(processing.result_table(result))
    with av.open(io.BytesIO(data)) as container:
        assert metadata_row["track_id"] == container.streams.video[0].id
    np.testing.assert_array_equal(frame_table.column("dts_ticks").to_numpy(), probe.dts)
    composition_offsets = frame_table.column("composition_offset_ticks")
    if probe.composition_offset is None:
        assert composition_offsets.null_count == len(probe)
    else:
        np.testing.assert_array_equal(composition_offsets.to_numpy(), probe.composition_offset)
    np.testing.assert_array_equal(frame_table.column("is_keyframe").to_numpy(zero_copy_only=False), probe.is_keyframe)
    np.testing.assert_array_equal(frame_table.column("offset").to_numpy(), probe.offset)
    np.testing.assert_array_equal(frame_table.column("size").to_numpy(), probe.size)
    assert metadata_row["frame_count"] == len(probe)
    assert metadata_row["extradata"] == probe.extradata


_CLIPS = [
    pytest.param(real_mp4.CLIP.read_bytes(), CONTAINER_KIND_NONFRAGMENTED, id=real_mp4.CLIP.name),
    pytest.param(real_mp4.BFRAME_CLIP.read_bytes(), CONTAINER_KIND_NONFRAGMENTED, id=real_mp4.BFRAME_CLIP.name),
    pytest.param(real_mp4.fragmented_clip(), CONTAINER_KIND_FRAGMENTED, id="one moof per frame"),
]


@pytest.mark.parametrize(("data", "container_kind"), _CLIPS)
@pytest.mark.parametrize("with_extension", [False, True], ids=["standard", "extension"])
def test_process_one_video_indexes_a_real_clip_as_pyav_demuxes_the_source(
    tmp_path: pathlib.Path, data: bytes, container_kind: str, *, with_extension: bool
) -> None:
    """PyAV, and the box walker an extension runs, publish the frames a full demux of the source reads.

    The walker reads the stored values, which match libav's here because none of these clips
    has negative composition offsets or a final sample with no stored duration.
    """
    extension_factory = (lambda _media_location: _NoOpExtension()) if with_extension else None

    result = _process(
        tmp_path, _write(tmp_path, data), container_kind=container_kind, extension_factory=extension_factory
    )

    _assert_frames_match_the_source(result, data)


@pytest.mark.parametrize("with_extension", [False, True], ids=["pyav", "box_walker"])
def test_indexed_track_survives_lance_and_selects_the_decoded_video(
    tmp_path: pathlib.Path, *, with_extension: bool
) -> None:
    """The stored native track ID selects the indexed pixels, even with audio and an earlier video track."""
    data = real_mp4.audio_then_two_video_tracks_clip()
    with av.open(io.BytesIO(data)) as container:
        selected = container.streams.video[1]
        assert container.streams.best("video") is selected
        assert selected.id != selected.index != 1
        track_id = selected.id
        expected_shape = (selected.height, selected.width, 3)
    extension_factory = (lambda _media_location: _NoOpExtension()) if with_extension else None
    media_location = _write(tmp_path, data)
    result = _process(
        tmp_path, media_location, container_kind=CONTAINER_KIND_NONFRAGMENTED, extension_factory=extension_factory
    )
    assert result["status"] == "success", result["error_message"]
    manifest_uri = str(tmp_path / "manifest.lance")
    lance.write_dataset(processing.result_table(result), manifest_uri)
    lookup = Mp4IndexLookup.open(manifest_uri)
    assert lookup is not None
    record = lookup.index_for(media_location, media_size_bytes=len(data))
    assert record is not None
    index, metadata = record
    assert metadata.track_id == track_id
    sensor = StreamingCameraSensor(data, frame_index=index, metadata=metadata)
    timestamps = sensor.timestamps_ns[:2]
    start_ns, end_ns = int(timestamps[0]), int(timestamps[-1]) + 1
    spec = SamplingSpec(
        grid=SamplingGrid(start_ns, end_ns, timestamps, stride_ns=end_ns - start_ns, duration_ns=end_ns - start_ns)
    )
    frames = [
        frame
        for window in sensor.sample(spec, policy=NearestTimestampPolicy(max_delta_ns=0))
        for _reference_ns, _timestamp_ns, frame in window.frames()
    ]
    assert [frame.shape for frame in frames] == [expected_shape] * 2

    missing_track = attrs.evolve(metadata, track_id=track_id + 100)
    sensor = StreamingCameraSensor(data, frame_index=index, metadata=missing_track)
    window = next(sensor.sample(spec, policy=NearestTimestampPolicy(max_delta_ns=0)))
    with pytest.raises(ValueError, match=r"track.*not found"):
        list(window.frames())


def test_process_one_video_indexes_a_real_fragment_layout_the_box_walker_refuses(tmp_path: pathlib.Path) -> None:
    """One moof per keyframe interval, many B-frame samples per trun: PyAV reads it from the boxes alone."""
    data = real_mp4.gop_fragmented_clip()

    result = _process(tmp_path, _write(tmp_path, data), container_kind=CONTAINER_KIND_FRAGMENTED)

    _assert_frames_match_the_source(result, data)
    frame_table, _metadata_row = read_video_record(processing.result_table(result))
    assert frame_table.column("composition_offset_ticks").null_count == 0


def test_process_one_video_reads_only_boxes_from_a_remote_source_cold_and_nothing_warm(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cold, the source is read for its moov and its moof walk; warm, the cached boxes index it unopened."""
    media_location = "s3://bucket/session-a/unit-00/video-1.mp4"
    data = real_mp4.fragmented_clip()
    storage_client = FakeStorageClient({media_location: data})
    opened: list[dict[str, Any] | None] = []
    smart_open_open = smart_open.open

    def open_object_or_local_file(uri: str, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        if uri != media_location:
            return smart_open_open(uri, *args, **kwargs)
        opened.append(kwargs.get("transport_params"))
        return io.BytesIO(data)

    monkeypatch.setattr(smart_open, "open", open_object_or_local_file)
    monkeypatch.setattr(
        processing, "get_smart_open_client_params", lambda client: {"transport_params": {"client": client}}
    )

    cold = _process(
        tmp_path, media_location, container_kind=CONTAINER_KIND_FRAGMENTED, input_storage_client=storage_client
    )
    cold_opens = len(opened)
    warm = _process(
        tmp_path, media_location, container_kind=CONTAINER_KIND_FRAGMENTED, input_storage_client=storage_client
    )

    _assert_frames_match_the_source(cold, data)
    assert cold_opens
    assert all(transport_params == {"client": storage_client} for transport_params in opened)
    assert warm["status"] == "success"
    assert warm["frames"].equals(cold["frames"])
    assert len(opened) == cold_opens
