# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic episode-level EgoTrajectory caption context."""

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Protocol

import numpy as np
from mcap.exceptions import McapError

from cosmos_curator.core.sensors.alignment.ego_trajectory import align_ego_trajectory_nearest
from cosmos_curator.core.sensors.sampling.policy import NearestTimestampPolicy
from cosmos_curator.core.sensors.sensors.ego_trajectory_sensor import EgoTrajectorySensor


@dataclass(frozen=True)
class RelativeMotionPolicy:
    """Effective numerical policy shared by derivation and result identity."""

    displacement_decimal_places: int = 3
    heading_decimal_places: int = 1
    min_horizontal_forward_norm: float = 1e-8


RELATIVE_MOTION_POLICY = RelativeMotionPolicy()
POSE_CONVENTION = "rfu_body_local_enu_v1"
_HALF_TURN_DEG = 180.0


@dataclass(frozen=True)
class EgoTrajectoryContextPolicy:
    """Versioned behavior values for one deterministic Ego context."""

    representation_version: int
    cadence_period_ns: int
    max_delta_ns: int

    def __post_init__(self) -> None:
        """Reject invalid fixed-policy values."""
        if self.representation_version < 1:
            msg = "representation_version must be positive"
            raise ValueError(msg)
        if self.cadence_period_ns < 1:
            msg = "cadence_period_ns must be positive"
            raise ValueError(msg)
        if self.max_delta_ns < 0:
            msg = "max_delta_ns must be non-negative"
            raise ValueError(msg)


class EgoContextBuilder(Protocol):
    """Top-level serializable builder contract selected by a context definition."""

    def __call__(  # noqa: PLR0913 -- all decoder inputs are explicit identity
        self,
        mcap_bytes: bytes,
        *,
        episode_start_ns: int,
        exclusive_end_ns: int,
        topic: str,
        schema_name: str,
        protobuf_mapping: Mapping[str, Any],
        frame: str,
        policy: EgoTrajectoryContextPolicy,
    ) -> tuple[bytes, str]:
        """Build canonical context bytes and their digest."""
        ...


def build_ego_trajectory_absolute_poses(  # noqa: PLR0913 -- all decoder inputs are explicit identity
    mcap_bytes: bytes,
    *,
    episode_start_ns: int,
    exclusive_end_ns: int,
    topic: str,
    schema_name: str,
    protobuf_mapping: Mapping[str, Any],
    frame: str,
    policy: EgoTrajectoryContextPolicy,
) -> tuple[bytes, str]:
    """Decode one split MCAP and return canonical summary-and-samples bytes plus SHA-256."""
    payload = _build_ego_trajectory_payload(
        mcap_bytes,
        episode_start_ns=episode_start_ns,
        exclusive_end_ns=exclusive_end_ns,
        topic=topic,
        schema_name=schema_name,
        protobuf_mapping=protobuf_mapping,
        frame=frame,
        policy=policy,
    )
    return _serialize_context(payload)


def build_ego_trajectory_summary(  # noqa: PLR0913 -- all decoder inputs are explicit identity
    mcap_bytes: bytes,
    *,
    episode_start_ns: int,
    exclusive_end_ns: int,
    topic: str,
    schema_name: str,
    protobuf_mapping: Mapping[str, Any],
    frame: str,
    policy: EgoTrajectoryContextPolicy,
) -> tuple[bytes, str]:
    """Decode and qualify one split MCAP, then serialize only its canonical summary."""
    payload = _build_ego_trajectory_payload(
        mcap_bytes,
        episode_start_ns=episode_start_ns,
        exclusive_end_ns=exclusive_end_ns,
        topic=topic,
        schema_name=schema_name,
        protobuf_mapping=protobuf_mapping,
        frame=frame,
        policy=policy,
    )
    del payload["samples"]
    return _serialize_context(payload)


def build_ego_trajectory_absolute_poses_with_relative_motion(  # noqa: PLR0913 -- decoder and derivation identity
    mcap_bytes: bytes,
    *,
    episode_start_ns: int,
    exclusive_end_ns: int,
    topic: str,
    schema_name: str,
    protobuf_mapping: Mapping[str, Any],
    frame: str,
    policy: EgoTrajectoryContextPolicy,
) -> tuple[bytes, str]:
    """Add output-point displacement and heading offset to the qualified absolute poses."""
    payload = _build_ego_trajectory_payload(
        mcap_bytes,
        episode_start_ns=episode_start_ns,
        exclusive_end_ns=exclusive_end_ns,
        topic=topic,
        schema_name=schema_name,
        protobuf_mapping=protobuf_mapping,
        frame=frame,
        policy=policy,
    )
    samples = payload["samples"]
    headings = [_horizontal_forward(sample["orientation_xyzw"]) for sample in samples]
    forward_x, forward_y = headings[0]
    origin_x, origin_y, origin_z = samples[0]["position_m"]
    for sample, (heading_x, heading_y) in zip(samples, headings, strict=True):
        x, y, z = sample["position_m"]
        dx, dy = x - origin_x, y - origin_y
        heading_deg = math.degrees(
            math.atan2(forward_x * heading_y - forward_y * heading_x, forward_x * heading_x + forward_y * heading_y)
        )
        # Wrap after rounding too: values just below +180 can round to +180.
        heading_deg = round(heading_deg, RELATIVE_MOTION_POLICY.heading_decimal_places) or 0.0
        sample["relative_motion"] = {
            "forward_m": round(dx * forward_x + dy * forward_y, RELATIVE_MOTION_POLICY.displacement_decimal_places)
            or 0.0,
            "left_m": round(-dx * forward_y + dy * forward_x, RELATIVE_MOTION_POLICY.displacement_decimal_places)
            or 0.0,
            "up_m": round(z - origin_z, RELATIVE_MOTION_POLICY.displacement_decimal_places) or 0.0,
            "heading_change_deg": heading_deg - 360.0 if heading_deg >= _HALF_TURN_DEG else heading_deg,
        }
    return _serialize_context(payload)


def _horizontal_forward(orientation_xyzw: list[float]) -> tuple[float, float]:
    """Project quaternion-rotated body +Y into ENU's horizontal plane."""
    norm = math.hypot(*orientation_xyzw)
    x, y, z, w = (value / norm for value in orientation_xyzw)
    forward_x, forward_y = 2.0 * (x * y - z * w), 1.0 - 2.0 * (x * x + z * z)
    horizontal_norm = math.hypot(forward_x, forward_y)
    if horizontal_norm <= RELATIVE_MOTION_POLICY.min_horizontal_forward_norm:
        msg = "Relative motion requires a defined horizontal body-forward heading"
        raise ValueError(msg)
    return forward_x / horizontal_norm, forward_y / horizontal_norm


def _build_ego_trajectory_payload(  # noqa: PLR0913 -- all decoder inputs are explicit identity
    mcap_bytes: bytes,
    *,
    episode_start_ns: int,
    exclusive_end_ns: int,
    topic: str,
    schema_name: str,
    protobuf_mapping: Mapping[str, Any],
    frame: str,
    policy: EgoTrajectoryContextPolicy,
) -> dict[str, Any]:
    """Decode once and build the existing fully qualified absolute payload."""
    if exclusive_end_ns <= episode_start_ns:
        msg = "exclusive_end_ns must be greater than episode_start_ns"
        raise ValueError(msg)
    try:
        sensor = EgoTrajectorySensor(
            mcap_bytes,
            topic=topic,
            schema_name=schema_name,
            protobuf_mapping=protobuf_mapping,
            frame=frame,
        )
        source = sensor.read_all()
    except McapError as exc:
        msg = f"Failed to decode Ego MCAP container ({type(exc).__name__})"
        raise ValueError(msg) from exc
    except (TypeError, ValueError) as exc:
        # Shared sensor and protobuf-mapper diagnostics may include source
        # values. Keep durable item errors useful without persisting them.
        msg = f"Failed to decode Ego trajectory ({type(exc).__name__})"
        raise ValueError(msg) from exc
    # V1 serializes no validity flag, and any decoded row—including the
    # predecessor halo—may be selected. Reject non-all-valid artifacts.
    if not bool(np.all(source.pose_valid)):
        msg = "ego-trajectory context v1 requires pose_valid=true for every decoded row"
        raise ValueError(msg)

    in_span = (source.align_timestamps_ns >= episode_start_ns) & (source.align_timestamps_ns < exclusive_end_ns)
    positions = source.positions_m[in_span]
    if len(positions) == 0:
        msg = "ego-trajectory context requires at least one recorded in-span pose"
        raise ValueError(msg)

    if len(positions) == 1:
        path_length_m = 0.0
        net_displacement_m = 0.0
    else:
        path_length_m = float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())
        net_displacement_m = float(np.linalg.norm(positions[-1] - positions[0]))

    duration_ns = exclusive_end_ns - episode_start_ns
    sample_count = (duration_ns + policy.cadence_period_ns - 1) // policy.cadence_period_ns
    relative_timestamps = np.arange(sample_count, dtype=np.int64) * policy.cadence_period_ns
    requested_timestamps = relative_timestamps + episode_start_ns
    aligned = align_ego_trajectory_nearest(
        source,
        requested_timestamps,
        policy=NearestTimestampPolicy(max_delta_ns=policy.max_delta_ns),
    )

    return {
        "frame": frame,
        "representation_version": policy.representation_version,
        "samples": [
            {
                "orientation_xyzw": [float(value) for value in aligned.orientations_xyzw[index]],
                "position_m": [float(value) for value in aligned.positions_m[index]],
                "relative_time_ns": int(relative_timestamps[index]),
                "requested_align_timestamp_ns": int(requested_timestamps[index]),
            }
            for index in range(sample_count)
        ],
        "summary": {
            "duration_ns": duration_ns,
            "net_displacement_m": net_displacement_m,
            "path_length_m": path_length_m,
            "recorded_pose_count": len(positions),
        },
    }


def _serialize_context(payload: Mapping[str, Any]) -> tuple[bytes, str]:
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return canonical, sha256(canonical).hexdigest()
