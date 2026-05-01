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
"""MCAP camera sensor."""

from collections.abc import Generator

import numpy as np
import numpy.typing as npt
from mcap.reader import McapReader
from mcap.reader import make_reader as mcap_make_reader
from mcap.records import Channel

from cosmos_curator.core.sensors.data.camera_data import CameraData
from cosmos_curator.core.sensors.data.video import VideoMetadata
from cosmos_curator.core.sensors.sampling.grid import SamplingWindow
from cosmos_curator.core.sensors.sampling.sampler import sample_window_indices
from cosmos_curator.core.sensors.sampling.spec import SamplingSpec
from cosmos_curator.core.sensors.types.types import DataSource
from cosmos_curator.core.sensors.utils.io import open_data_source
from cosmos_curator.core.sensors.utils.mcap import (
    VIDEO_METADATA_RECORD_NAME,
    channel_for_topic,
    get_metadata_record,
    iter_messages_log_time_ns,
    load_start_end_ns,
    load_timeline,
)

_RGB_CHANNELS = 3


def _rgb8_channel_dimensions(channel: Channel) -> tuple[int, int]:
    """Parse ``width`` and ``height`` from an ``rgb8`` MCAP channel record."""
    if channel.message_encoding != "rgb8":
        msg = f"expected rgb8 channel, got message_encoding={channel.message_encoding!r}"
        raise ValueError(msg)
    try:
        width = int(channel.metadata["width"])
        height = int(channel.metadata["height"])
    except (KeyError, ValueError) as e:
        msg = f"channel metadata must include integer width and height strings: {channel.metadata!r}"
        raise ValueError(msg) from e
    if width <= 0 or height <= 0:
        msg = f"invalid rgb8 dimensions width={width} height={height}"
        raise ValueError(msg)
    return width, height


def _decode_payload_rows(
    payloads: list[bytes],
    indices: npt.NDArray[np.int64],
    *,
    width: int,
    height: int,
) -> npt.NDArray[np.uint8]:
    """Decode selected payload bytes into an ``(N, H, W, 3)`` frame tensor."""
    frame_shape = (height, width, _RGB_CHANNELS)
    frames = np.empty((len(indices), height, width, _RGB_CHANNELS), dtype=np.uint8)
    for i, idx in enumerate(indices):
        frames[i] = np.frombuffer(payloads[int(idx)], dtype=np.uint8).reshape(frame_shape)
    return frames


class McapCameraSensor:
    """MCAP camera sensor.

    This is not meant to be used in production, but rather as a proof of
    concept to demonstrate how to use MCAP as a data source for a sensor.

    Reads RGB frames from an MCAP topic matching the contract emitted by
    :mod:`cosmos_curator.core.sensors.scripts.make_mcap_from_mp4`.

    - MCAP timestamps are stored in nanoseconds in log_time
    - sensor_timestamps_ns comes from log_time
    - pts_stream is also reported in nanoseconds for MCAP-backed sensors, so
      unlike CameraSensor it is not a separate stream-native unit

    This example sensor buffers all raw RGB payload bytes for one sampling
    window before applying nearest-neighbour selection. Large windows at high
    resolution can therefore use substantial memory, so this class is
    intentionally kept as example code rather than a production reader.
    """

    def __init__(self, source: DataSource, topic: str = "/camera/rgb") -> None:
        """Initialize the MCAP camera sensor."""
        self._source = source
        self._topic = topic
        self._message_log_times_ns_cache: npt.NDArray[np.int64] | None = None
        self._start_ns: int | None = None
        self._end_ns: int | None = None
        self._video_metadata: VideoMetadata | None = None
        self._empty_camera_data: CameraData | None = None

    @property
    def video_metadata(self) -> VideoMetadata:
        """Return the video metadata stored in the MCAP file."""
        if self._video_metadata is None:
            self._video_metadata = self._load_video_metadata()
        return self._video_metadata

    def _load_video_metadata(self) -> VideoMetadata:
        """Read the required MCAP video metadata record."""
        with open_data_source(self._source, mode="rb") as stream:
            reader = mcap_make_reader(stream)  # type: ignore[no-untyped-call]
            return VideoMetadata.from_string_dict(get_metadata_record(reader, VIDEO_METADATA_RECORD_NAME))

    @property
    def start_ns(self) -> int:
        """Earliest frame time on this topic, in nanoseconds."""
        self._ensure_start_end_ns_cached()
        if self._start_ns is None:
            msg = "start_ns was not loaded"
            raise ValueError(msg)
        return self._start_ns

    @property
    def end_ns(self) -> int:
        """Latest frame time on this topic, in nanoseconds."""
        self._ensure_start_end_ns_cached()
        if self._end_ns is None:
            msg = "end_ns was not loaded"
            raise ValueError(msg)
        return self._end_ns

    def _ensure_start_end_ns_cached(self) -> None:
        """Cache only the first and last message timestamps for the topic."""
        if self._start_ns is not None and self._end_ns is not None:
            return

        full = self._message_log_times_ns_cache
        if full is not None:
            self._start_ns = int(full[0])
            self._end_ns = int(full[-1])
            return

        with open_data_source(self._source, mode="rb") as stream:
            reader = mcap_make_reader(stream)  # type: ignore[no-untyped-call]
            self._start_ns, self._end_ns = load_start_end_ns(reader, self._topic)

    def _ensure_timeline_cached(self) -> npt.NDArray[np.int64]:
        """Load and cache the full ordered timeline for the topic."""
        if self._message_log_times_ns_cache is not None:
            return self._message_log_times_ns_cache

        with open_data_source(self._source, mode="rb") as stream:
            reader = mcap_make_reader(stream)  # type: ignore[no-untyped-call]
            arr = load_timeline(reader, self._topic)
        self._message_log_times_ns_cache = arr
        self._start_ns = int(arr[0])
        self._end_ns = int(arr[-1])
        return arr

    @property
    def max_gap_ns(self) -> int:
        """Return maximum expected gap duration in nanoseconds."""
        return 0

    @property
    def timestamps_ns(self) -> npt.NDArray[np.int64]:
        """Presentation times in nanoseconds (raw MCAP ``log_time``)."""
        return self._ensure_timeline_cached()

    def _resolve_topic_dimensions(self, reader: McapReader) -> tuple[int, int]:
        """Resolve and validate frame dimensions for the configured topic."""
        metadata = self.video_metadata
        summary = reader.get_summary()
        if summary is None:
            return metadata.width, metadata.height

        channel = channel_for_topic(summary, self._topic)
        if channel is None:
            return metadata.width, metadata.height

        width, height = _rgb8_channel_dimensions(channel)
        self._validate_dimensions(width, height, metadata)
        return width, height

    def _validate_dimensions(self, width: int, height: int, metadata: VideoMetadata) -> None:
        """Ensure channel dimensions match the stored video metadata."""
        if metadata.width != width or metadata.height != height:
            msg = (
                "MCAP channel dimensions do not match stored video metadata: "
                f"channel={width}x{height} metadata={metadata.width}x{metadata.height}"
            )
            raise ValueError(msg)

    def _read_window_messages(
        self,
        reader: McapReader,
        window: SamplingWindow,
        width: int,
        height: int,
    ) -> tuple[npt.NDArray[np.int64], list[bytes]]:
        """Read all topic messages whose timestamps overlap one sampling window."""
        if len(window) == 0:
            return np.empty(0, dtype=np.int64), []

        frame_bytes = width * height * _RGB_CHANNELS
        log_times: list[int] = []
        # This buffers every raw RGB payload in the window before sampling.
        # Large windows at high resolution can therefore have a steep memory cost.
        payloads: list[bytes] = []
        for _schema, _channel, message in iter_messages_log_time_ns(
            reader,
            self._topic,
            int(window.timestamps_ns[0]),
            int(window.exclusive_end_ns),
            log_time_order=True,
        ):
            if len(message.data) != frame_bytes:
                msg = (
                    f"rgb8 payload size {len(message.data)} != {frame_bytes} "
                    f"for {width}x{height} on topic {self._topic!r}"
                )
                raise ValueError(msg)
            log_times.append(int(message.log_time))
            payloads.append(message.data)

        return np.array(log_times, dtype=np.int64), payloads

    def _get_empty_camera_data(self) -> CameraData:
        """Return a cached empty batch preserving the expected frame shape."""
        if self._empty_camera_data is None:
            metadata = self.video_metadata
            empty_ts = np.empty(0, dtype=np.int64)
            empty_frames = np.empty((0, metadata.height, metadata.width, _RGB_CHANNELS), dtype=np.uint8)
            self._empty_camera_data = CameraData(
                align_timestamps_ns=empty_ts,
                sensor_timestamps_ns=empty_ts,
                pts_stream=empty_ts,
                frames=empty_frames,
                metadata=metadata,
            )
        return self._empty_camera_data

    def _sample_window(  # noqa: PLR0913
        self,
        window: SamplingWindow,
        log_times_ns: npt.NDArray[np.int64],
        payloads: list[bytes],
        *,
        width: int,
        height: int,
        spec: SamplingSpec,
    ) -> CameraData:
        """Build a ``CameraData`` batch for one window."""
        if len(window) == 0 or len(log_times_ns) == 0:
            return self._get_empty_camera_data()

        indices, _counts = sample_window_indices(log_times_ns, window, policy=spec.policy, dedup=False)
        if len(indices) == 0:
            return self._get_empty_camera_data()

        frames = _decode_payload_rows(payloads, indices, width=width, height=height)
        align_timestamps_ns = window.timestamps_ns
        sensor_timestamps_ns = log_times_ns[indices]
        sensor_timestamps_ns.flags.writeable = False
        return CameraData(
            align_timestamps_ns=align_timestamps_ns,
            sensor_timestamps_ns=sensor_timestamps_ns,
            pts_stream=sensor_timestamps_ns,
            frames=frames,
            metadata=self.video_metadata,
        )

    def sample(self, spec: SamplingSpec) -> Generator[CameraData, None, None]:
        """Sample camera frames according to the provided ``SamplingSpec``.

        Each yielded batch follows the sampling-grid half-open interval
        convention. For a window emitted by :class:`SamplingGrid`,
        ``window.exclusive_end_ns`` is the exclusive right boundary marker.

        Any reference timestamp strictly less than ``window.exclusive_end_ns``
        belongs to this batch, while a timestamp exactly equal to
        ``window.exclusive_end_ns`` belongs to the later batch, not both.
        Because ``window`` is sorted in ascending order, this means the
        current batch uses ``window.exclusive_end_ns``.

        Empty windows yield an empty :class:`CameraData` so that batch index
        ``i`` continues to correspond to the ``i`` th sampling window. Empty
        batches reuse shared read-only zero-length timestamp arrays and frame
        tensors.

        Args:
            spec: the sampling spec to use when sampling data from this
                sensor.

        Yields:
            CameraData batches

        Notes:
            The returned generator keeps the underlying source open while iteration is
            active. If iteration stops early, call ``gen.close()`` to release the
            resource promptly.

        """
        with open_data_source(self._source, mode="rb") as stream:
            reader = mcap_make_reader(stream)  # type: ignore[no-untyped-call]
            width, height = self._resolve_topic_dimensions(reader)
            for window in spec.grid:
                log_times_ns, payloads = self._read_window_messages(reader, window, width, height)
                yield self._sample_window(window, log_times_ns, payloads, width=width, height=height, spec=spec)
