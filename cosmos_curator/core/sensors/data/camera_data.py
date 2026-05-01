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

"""Camera data structures for cosmos_curator.core.sensors package."""

from typing import Protocol

import attrs
import numpy as np
import numpy.typing as npt

from cosmos_curator.core.sensors.data.extrinsics import SensorExtrinsics
from cosmos_curator.core.sensors.data.intrinsics import CameraIntrinsics
from cosmos_curator.core.sensors.data.video import VideoMetadata
from cosmos_curator.core.sensors.utils.helpers import as_readonly_view, as_readonly_view_tuple
from cosmos_curator.core.sensors.utils.validation import (
    nondecreasing_int64_array,
    strictly_increasing_int64_array,
    uint8_frame_batch,
)

_MOTION_VECTOR_FRAME_NDIM = 2
_MOTION_VECTOR_COLUMNS = 10
_CAMERA_FRAME_NDIM = 4
_RGB_CHANNELS = 3


class _HasCameraFrames(Protocol):
    frames: npt.NDArray[np.uint8]


class _HasCameraBatchFields(_HasCameraFrames, Protocol):
    align_timestamps_ns: npt.NDArray[np.int64]
    sensor_timestamps_ns: npt.NDArray[np.int64]
    pts_stream: npt.NDArray[np.int64]


class _HasMetadata(Protocol):
    metadata: VideoMetadata


def _mvd_frames(
    _instance: object,
    _attribute: object,
    value: tuple[npt.NDArray[np.float64], ...],
) -> None:
    """Validate per-frame motion vector block tables."""
    for i, frame in enumerate(value):
        if frame.ndim != _MOTION_VECTOR_FRAME_NDIM:
            msg = f"motion_vectors.frames[{i}] must be 2-D, got ndim={frame.ndim}"
            raise ValueError(msg)
        if frame.shape[1] != _MOTION_VECTOR_COLUMNS:
            msg = f"motion_vectors.frames[{i}] must have shape (N, {_MOTION_VECTOR_COLUMNS}), got shape={frame.shape}"
            raise ValueError(msg)


def _motion_vectors(
    instance: _HasCameraFrames,
    _attribute: object,
    value: "MotionVectorData | None",
) -> None:
    """Validate optional motion-vector payload length against the RGB frame batch."""
    if value is None:
        return
    if len(value.frames) != len(instance.frames):
        error_msg = f"motion_vectors.frames length {len(value.frames)} != frames length {len(instance.frames)}"
        raise ValueError(error_msg)


def _batch_lengths(
    instance: _HasCameraBatchFields,
    _attribute: object,
    _value: VideoMetadata,
) -> None:
    """Validate shared row-count invariants across camera batch arrays."""
    if not (
        len(instance.align_timestamps_ns)
        == len(instance.sensor_timestamps_ns)
        == len(instance.pts_stream)
        == len(instance.frames)
    ):
        error_msg = (
            "All arrays must be the same length: "
            f"align_timestamps_ns={len(instance.align_timestamps_ns)} "
            f"sensor_timestamps_ns={len(instance.sensor_timestamps_ns)} "
            f"pts_stream={len(instance.pts_stream)} "
            f"frames={len(instance.frames)}"
        )
        raise ValueError(error_msg)


def _metadata_shape(
    instance: _HasCameraFrames,
    _attribute: object,
    value: VideoMetadata,
) -> None:
    """Validate frame geometry against metadata dimensions."""
    expected_shape = (value.height, value.width, _RGB_CHANNELS)
    if instance.frames.shape[1:] != expected_shape:
        msg = f"frames must have shape (N, {value.height}, {value.width}, {_RGB_CHANNELS}), got {instance.frames.shape}"
        raise ValueError(msg)


def _intrinsics_dimensions_match_metadata(
    instance: _HasMetadata,
    _attribute: object,
    value: CameraIntrinsics | None,
) -> None:
    """Validate optional intrinsics dimensions against the batch metadata."""
    if value is None:
        return
    metadata = instance.metadata
    if value.width != metadata.width or value.height != metadata.height:
        msg = (
            f"CameraIntrinsics dimensions ({value.width}x{value.height}) do not match "
            f"VideoMetadata dimensions ({metadata.width}x{metadata.height}). "
            "This may indicate either rig calibration errors or video encoding errors."
        )
        raise ValueError(msg)


@attrs.define(hash=False, frozen=True)
class MotionVectorData:
    """Per-frame motion vectors from H.264/HEVC (FFmpeg AVMotionVector). Variable num_blocks per frame."""

    __hash__ = None  # type: ignore[assignment]

    # List of length N; frame i has shape (num_blocks_i, 10).
    # Columns:
    # [0:2] w, h (block size)
    # [2:4] src_x, src_y
    # [4:6] dst_x, dst_y
    # [6:7] flags
    # [7:9] motion_x, motion_y (sub-pixel; divide by motion_scale for pixel delta)
    # [9]   motion_scale
    frames: tuple[npt.NDArray[np.float64], ...] = attrs.field(
        validator=_mvd_frames,
        converter=as_readonly_view_tuple,
    )


@attrs.define(hash=False, frozen=True)
class CameraData:
    """Decoded RGB video: ``N`` frames, with row ``i`` indexing the same moment across arrays.

    Satisfies ``SensorData`` (``cosmos_curator.core.sensors.data.sensor_data``).

    Attributes:
        align_timestamps_ns: 1-D alignment timeline (ns) each sample row is aligned to; length ``N``,
            row ``i`` with ``frames[i]``
        sensor_timestamps_ns: 1-D sensor-reported times (ns); may differ from ``align_timestamps_ns``
            (resampling/grid); length ``N``
        pts_stream: 1-D presentation timestamps in producer-specific int units, length ``N``;
            ``CameraSensor`` uses stream-native ``time_base`` for exact seeks; ``McapCameraSensor``
            uses nanoseconds matching ``sensor_timestamps_ns`` (example sensor only; production
            ``pts_stream`` is expected to follow the ``CameraSensor`` contract)
        frames: decoded RGB, shape ``(N, H, W, 3)``, ``uint8``; row ``i`` is the image at index ``i``
        metadata: stream geometry and related fields (``VideoMetadata``)
        motion_vectors: optional per-frame motion vectors; when set, length ``N`` matches ``frames``
        intrinsics: optional typed camera calibration for the same image geometry as ``metadata``
        extrinsics: optional rigid transform from the camera frame to a caller-defined reference frame

    """

    __hash__ = None  # type: ignore[assignment]
    align_timestamps_ns: npt.NDArray[np.int64] = attrs.field(
        converter=as_readonly_view,
        validator=strictly_increasing_int64_array,
    )
    sensor_timestamps_ns: npt.NDArray[np.int64] = attrs.field(
        converter=as_readonly_view,
        validator=nondecreasing_int64_array,
    )
    pts_stream: npt.NDArray[np.int64] = attrs.field(
        converter=as_readonly_view,
        validator=nondecreasing_int64_array,
    )
    frames: npt.NDArray[np.uint8] = attrs.field(
        converter=as_readonly_view,
        validator=uint8_frame_batch,
    )
    # Attach batch-length validation to the last required field so all batch
    # arrays are already set when attrs runs this validator.
    metadata: VideoMetadata = attrs.field(
        validator=attrs.validators.and_(
            _batch_lengths,
            _metadata_shape,
        )
    )
    motion_vectors: MotionVectorData | None = attrs.field(
        default=None,
        validator=_motion_vectors,
    )  # optional; requires decoder with export_mvs
    intrinsics: CameraIntrinsics | None = attrs.field(
        default=None,
        validator=_intrinsics_dimensions_match_metadata,
    )
    extrinsics: SensorExtrinsics | None = attrs.field(default=None)
