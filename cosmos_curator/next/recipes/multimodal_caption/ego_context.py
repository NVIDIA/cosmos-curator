# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic episode-level EgoTrajectory caption context."""

import json
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
