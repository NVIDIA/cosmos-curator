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
"""Camera sensor for the curator package."""

from collections.abc import Generator
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from cosmos_curator.core.sensors.data.camera_data import CameraData
from cosmos_curator.core.sensors.data.extrinsics import SensorExtrinsics
from cosmos_curator.core.sensors.data.intrinsics import CameraIntrinsics
from cosmos_curator.core.sensors.data.video import VideoIndex, VideoMetadata
from cosmos_curator.core.sensors.sampling.sampler import sample_window_indices
from cosmos_curator.core.sensors.sampling.spec import SamplingSpec
from cosmos_curator.core.sensors.types.types import DataSource, VideoIndexCreationMethod
from cosmos_curator.core.sensors.utils.video import (
    DEFAULT_VIDEO_DECODE_CONFIG,
    CpuVideoDecodeConfig,
    CpuVideoDecoder,
    GpuVideoDecodeConfig,
    GpuVideoDecoder,
    VideoDecodeConfig,
    make_decode_plan,
    make_index_and_metadata,
    pts_to_ns,
)

if TYPE_CHECKING:
    from contextlib import AbstractContextManager


class CameraSensor:
    """Camera sensor.

    This camera sensor provides access to video stream metadata including timestamps
    and frame type information for videos in MP4 and MKV containers.

    Supported containers:
        - MP4 and related formats (MOV, M4A, etc.)
        - MKV (Matroska)
          - MKV has received little testing and is not well-supported.

    Rejected containers:
        - MPEG-TS


    """

    def __init__(  # noqa: PLR0913
        self,
        source: DataSource,
        stream_idx: int = 0,
        decode_config: VideoDecodeConfig = DEFAULT_VIDEO_DECODE_CONFIG,
        index_method: VideoIndexCreationMethod = VideoIndexCreationMethod.FROM_HEADER,
        intrinsics: CameraIntrinsics | None = None,
        extrinsics: SensorExtrinsics | None = None,
    ) -> None:
        """Initialize the camera sensor.

        Args:
            source: the source of the video data
            stream_idx: The index of the video stream to use (default: 0).
            decode_config: Backend configuration for frame decoding. Use
                :class:`~cosmos_curator.core.sensors.utils.video.CpuVideoDecodeConfig`
                for PyAV/CPU decode or
                :class:`~cosmos_curator.core.sensors.utils.video.GpuVideoDecodeConfig`
                for the future GPU backend.
            index_method: How ``VideoIndex`` packet metadata is built. See
                :class:`~cosmos_curator.core.sensors.types.types.VideoIndexCreationMethod`.
                Prefer ``FROM_HEADER``. Use ``FULL_DEMUX`` only for tests or rare
                validation; if production needs full demux, file an issue.
            intrinsics: Optional pre-parsed camera calibration for the decoded
                image geometry. Callers using parser objects should parse them
                before constructing ``CameraSensor``.
            extrinsics: Optional pre-parsed rigid transform from the camera frame
                to a caller-defined reference frame.

        """
        self._source = source
        self._stream_idx = stream_idx
        self._decode_config = decode_config
        self._intrinsics = intrinsics
        self._extrinsics = extrinsics
        self._video_index, self._video_metadata = make_index_and_metadata(
            self._source, self._stream_idx, index_method=index_method
        )
        if len(self._video_index.display_pts_ns) == 0:
            msg = "video stream contains no displayable frames"
            raise ValueError(msg)
        self._empty_camera_data: CameraData | None = None

    @property
    def video_index(self) -> VideoIndex:
        """Return the video index for this sensor."""
        return self._video_index

    @property
    def video_metadata(self) -> VideoMetadata:
        """Return the video metadata for this sensor."""
        return self._video_metadata

    @property
    def start_ns(self) -> int:
        """Return the canonical starting timestamp for this sensor (nanoseconds)."""
        return int(self._video_index.display_pts_ns[0])

    @property
    def end_ns(self) -> int:
        """Return the canonical ending timestamp for this sensor (nanoseconds)."""
        return int(self._video_index.display_pts_ns[-1])

    @property
    def max_gap_ns(self) -> int:
        """Return maximum expected gap duration in nanoseconds.

        This is for sensors that have regular dead periods, for example,
        rotating lidar.
        """
        return 0

    @property
    def timestamps_ns(self) -> npt.NDArray[np.int64]:
        """Get the timestamps in nanoseconds.

        Returns:
            Numpy array of timestamps in nanoseconds as int64.

        """
        return self._video_index.display_pts_ns

    @property
    def codec_name(self) -> str:
        """Get the video codec name.

        Returns:
            Codec name string (e.g., 'h264', 'hevc', 'vp9').

        """
        return self._video_metadata.codec_name

    @property
    def codec_max_bframes(self) -> int:
        """Get the max number of consecutive B-frames between I and P frames.

        This counts the maximum number of consecutive B-frames between I and P
        frames that the encoder was configured to generate.

        This is not how many B-frames are in the stream. It is possible that the
        encoder was configured to generate B-frames, but did not generate any.

        Returns:
            Maximum number of B-frames the encoder was configured to generate.

        Note:
            For authoritative B-frame detection, the entire video must be
            read and analyzed. It may be possible to perform this analysis
            without decoding the entire video, but not without reading the
            entire video.

            This is not practical at scale.

            The authoritative way to detect B-frames is to read the entire
            video and analyze the NAL units.

        """
        return self._video_metadata.codec_max_bframes

    def _get_empty_camera_data(self) -> CameraData:
        """Return a cached empty batch preserving the expected frame shape."""
        if self._empty_camera_data is None:
            empty_frames = np.empty((0, self._video_metadata.height, self._video_metadata.width, 3), dtype=np.uint8)
            empty_ts = np.empty(0, dtype=np.int64)
            self._empty_camera_data = CameraData(
                align_timestamps_ns=empty_ts,
                sensor_timestamps_ns=empty_ts,
                pts_stream=empty_ts,
                frames=empty_frames,
                metadata=self._video_metadata,
                intrinsics=self._intrinsics,
                extrinsics=self._extrinsics,
            )
        return self._empty_camera_data

    def sample(
        self,
        spec: SamplingSpec,
        stats: dict[str, float] | None = None,
    ) -> Generator[CameraData, None, None]:
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
            stats: optional dict for benchmarking instrumentation.  When
                provided, seek and convert timings are accumulated into the
                dict by the underlying decode function.  Pass ``None``
                (default) in production.

        Yields:
            CameraData batches

        """
        decoder_cm: AbstractContextManager[CpuVideoDecoder | GpuVideoDecoder]
        match self._decode_config:
            case CpuVideoDecodeConfig() as config:
                decoder_cm = CpuVideoDecoder.open(self._source, self._stream_idx, config, stats)
            case GpuVideoDecodeConfig() as config:
                decoder_cm = GpuVideoDecoder.open(self._source, self._stream_idx, config, stats)
            case _:
                msg = f"unsupported decode_config: {type(self._decode_config).__name__}"
                raise ValueError(msg)

        with decoder_cm as decoder:
            for window in spec.grid:
                if len(window) == 0:
                    yield self._get_empty_camera_data()
                    continue

                indices, counts = sample_window_indices(self.video_index.display_pts_ns, window, policy=spec.policy)
                if len(indices) == 0:
                    yield self._get_empty_camera_data()
                    continue
                sampled_pts_stream = self.video_index.display_pts_stream[indices]
                decode_plan = make_decode_plan(self.video_index.kf_pts_stream, sampled_pts_stream, counts)
                frames = decoder.decode(decode_plan)
                pts_stream_expanded = np.repeat(sampled_pts_stream, counts)

                yield CameraData(
                    align_timestamps_ns=window.timestamps_ns,
                    sensor_timestamps_ns=pts_to_ns(pts_stream_expanded, decoder.time_base),
                    pts_stream=pts_stream_expanded,
                    frames=frames,
                    metadata=self._video_metadata,
                    intrinsics=self._intrinsics,
                    extrinsics=self._extrinsics,
                )
