# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
"""Postgres schemas for the video pipeline."""

import datetime
import uuid
from enum import Enum

from sqlalchemy import (
    ARRAY,
    UUID,
    BigInteger,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy import (
    Enum as SQLEnum,
)
from sqlalchemy.dialects.postgresql import BYTEA, JSONB
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy declarative models in the video pipeline."""


class SourceData(Base):
    """Source data table.

    This table contains information about the source data.
    """

    __tablename__ = "source_data"
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    session_name = Column(
        String,
        nullable=False,
        index=True,
        comment="Session name which include videos from multiple cameras",
    )
    version = Column(String, nullable=False, index=True, comment="Version of the session")
    session_url = Column(String, nullable=False, index=False)
    num_cameras = Column(Integer, nullable=False, index=True)

    __table_args__ = (UniqueConstraint("session_name", "version", name="source_data_session_name_version"),)


class Run(Base):
    """Run table.

    This table contains information about the run.
    """

    __tablename__ = "run"
    run_uuid = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    datetime = Column(DateTime, nullable=False, default=datetime.datetime.now(datetime.UTC))
    run_type = Column(String, nullable=False, index=True, comment="Run type, e.g. split")
    pipeline_version = Column(String, nullable=False, comment="Pipeline implementation Version")
    description = Column(String, comment="Run description")
    params = Column(JSONB, comment="Pipeline and other extra parameters")


class ClippedSession(Base):
    """Clipped session table.

    This table contains information about the clipped sessions.
    """

    __tablename__ = "clipped_session"
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    source_video_session_name = Column(String, nullable=False, index=True)
    source_video_version = Column(String, nullable=False, index=True)
    session_uuid = Column(UUID(as_uuid=True), nullable=False, index=True)
    version = Column(String, nullable=False, index=True)
    num_cameras = Column(Integer, nullable=False)
    split_algo_name = Column(String, nullable=False)
    encoder = Column(String, nullable=False)

    datetime = Column(DateTime, nullable=False, default=datetime.datetime.now(datetime.UTC))
    run_uuid = Column(UUID(as_uuid=True), ForeignKey("run.run_uuid"), nullable=False)

    __table_args__ = (
        UniqueConstraint(
            "session_uuid",
            "version",
            "split_algo_name",
            "encoder",
            name="clipped_session_session_uuid_version_split_algo_name_encoder",
        ),
    )


class VideoSpan(Base):
    """Video span table.

    This table contains information about the video spans for a clip.
    """

    __tablename__ = "video_span"
    id = Column(BigInteger, primary_key=True, autoincrement=True)

    clip_uuid = Column(UUID(as_uuid=True), nullable=False, index=True)
    clip_session_uuid = Column(UUID(as_uuid=True), nullable=False, index=True)
    version = Column(String, nullable=False, index=True)

    source_video = Column(String, nullable=False, index=True)
    camera_id = Column(Integer, nullable=False, index=True)
    session_uuid = Column(UUID(as_uuid=True), nullable=False, index=True)
    span_index = Column(Integer, nullable=False, index=True)

    split_algo_name = Column(String, nullable=False, index=True)
    span_start = Column(Float, nullable=False)
    span_end = Column(Float, nullable=False)
    timestamps = Column(BYTEA, nullable=False)

    encoder = Column(String, nullable=False, index=True)
    url = Column(String, nullable=False)

    byte_size = Column(BigInteger, nullable=False)
    duration = Column(Float, nullable=False)
    framerate = Column(Float, nullable=False)
    num_frames = Column(Integer, nullable=False, index=True)
    height = Column(Integer, nullable=False, index=True)
    width = Column(Integer, nullable=False, index=True)
    sha256 = Column(String(64), nullable=False)
    datatime = Column(DateTime, nullable=False, default=datetime.datetime.now(datetime.UTC))
    run_uuid = Column(UUID(as_uuid=True), ForeignKey("run.run_uuid"), nullable=False, index=True)

    __table_args__ = (
        UniqueConstraint(
            "clip_uuid",
            "version",
            "split_algo_name",
            "encoder",
            name="video_span_clip_uuid_version_split_algo_name_encoder",
        ),
    )


class ClipCaption(Base):
    """Clip caption table.

    This table contains information about the captions for a clip.
    """

    __tablename__ = "clip_caption"
    id = Column(BigInteger, primary_key=True, autoincrement=True)

    clip_uuid = Column(UUID(as_uuid=True), nullable=False, index=True)
    version = Column(String, nullable=False, index=True)
    prompt_type = Column(String, nullable=False, index=True)

    window_start_frame: Column[list[int]] = Column(ARRAY(Integer), nullable=False)
    window_end_frame: Column[list[int]] = Column(ARRAY(Integer), nullable=False)
    window_caption: Column[list[str]] = Column(ARRAY(String), nullable=False)
    t5_embedding_url = Column(String, nullable=False)

    datetime = Column(DateTime, nullable=False, default=datetime.datetime.now(datetime.UTC))
    run_uuid = Column(UUID(as_uuid=True), ForeignKey("run.run_uuid"), nullable=False, index=True)

    __table_args__ = (
        UniqueConstraint(
            "clip_uuid",
            "version",
            "prompt_type",
            name="clip_caption_clip_uuid_version_prompt_type",
        ),
    )


class ClipTrajectory(Base):
    """Clip trajectory table.

    This table contains information about the trajectory data for a clip.
    """

    __tablename__ = "clip_trajectory"
    id = Column(BigInteger, primary_key=True, autoincrement=True)

    clip_uuid = Column(UUID(as_uuid=True), nullable=False, index=True)
    version = Column(String, nullable=False, index=True)

    trajectory_url = Column(String, nullable=False)

    datetime = Column(DateTime, nullable=False, default=datetime.datetime.now(datetime.UTC))
    run_uuid = Column(UUID(as_uuid=True), ForeignKey("run.run_uuid"), nullable=False, index=True)

    __table_args__ = (
        UniqueConstraint(
            "clip_uuid",
            "version",
            name="clip_trajectory_clip_uuid_version",
        ),
    )


class ClipTag(Base):
    """Clip tag table.

    This table contains information about the tags for a clip.
    """

    __tablename__ = "clip_tag"
    id = Column(BigInteger, primary_key=True, autoincrement=True)

    clip_uuid = Column(UUID(as_uuid=True), nullable=False, index=True)
    version = Column(String, nullable=False, index=True)

    country = Column(String, nullable=False, index=True)
    traffic = Column(String, nullable=False, index=True)
    ego_speed = Column(String, nullable=False, index=True)
    ego_acceleration = Column(String, nullable=False, index=True)
    ego_curve = Column(String, nullable=False, index=True)
    ego_turn = Column(String, nullable=False, index=True)
    osm_features = Column(String, nullable=False, index=True)
    road_type = Column(String, nullable=False, index=True)
    visibility = Column(String, nullable=False, index=True)
    road_surface = Column(String, nullable=False, index=True)
    illumination = Column(String, nullable=False, index=True)

    datetime = Column(DateTime, nullable=False, default=datetime.datetime.now(datetime.UTC))
    run_uuid = Column(UUID(as_uuid=True), ForeignKey("run.run_uuid"), nullable=False, index=True)

    __table_args__ = (UniqueConstraint("clip_uuid", "version", name="clip_tag_clip_uuid_version"),)


class EgoSpeedTier(Enum):
    """Ego speed tier enum.

    This enum represents the speed tier of the ego vehicle.
    """

    high = "high"
    medium = "medium"
    low = "low"
    stand_still = "stand_still"
    unknown = "unknown"


class EgoSpeedingBehavior(Enum):
    """Ego speeding behavior enum.

    This enum represents the behavior of the ego vehicle in relation to speed limits.
    """

    speeding = "speeding"
    normal = "normal"
    underspeed = "underspeed"
    no_speed_limit = "no_speed_limit"
    unknown = "unknown"


class EgoAccelerationType(Enum):
    """Ego acceleration type enum.

    This enum represents the type of acceleration performed by the ego vehicle.
    """

    fast_accel = "fast_accel"
    slow_accel = "slow_accel"
    fast_decel = "fast_decel"
    slow_decel = "slow_decel"
    maintain = "maintain"
    brake = "brake"
    unknown = "unknown"


class EgoManeuverType(Enum):
    """Ego maneuver type enum.

    This enum represents the type of maneuver performed by the ego vehicle.
    """

    reverse = "reverse"
    change_lane_left = "lane_change_left"
    change_lane_right = "lane_change_right"
    left_turn = "left_turn"
    right_turn = "right_turn"
    curve_left = "curve_left"
    curve_right = "curve_right"
    straight = "straight"
    non_straight = "non_straight"
    unknown = "unknown"


class SessionDescription(Base):
    """Session description table.

    This table contains information about the session, including the country, city,
    daytime, and various behavioral characteristics.
    """

    __tablename__ = "session_description_dev"

    id = Column(String(255), primary_key=True)
    country = Column(String(255), nullable=False)
    city_name = Column(String(255), nullable=False)
    daytime = Column(Boolean, nullable=False)
    ego_speed_tier: Column[SQLEnum] = Column(SQLEnum(EgoSpeedTier), nullable=False)
    speed_limit_m_s = Column(Integer, nullable=False)
    ego_speeding_behavior: Column[SQLEnum] = Column(SQLEnum(EgoSpeedingBehavior), nullable=False)
    ego_acceleration_type: Column[SQLEnum] = Column(SQLEnum(EgoAccelerationType), nullable=False)
    ego_maneuver_type: Column[SQLEnum] = Column(SQLEnum(EgoManeuverType), nullable=False)
    at_intersection_with_traffic_light = Column(Boolean, nullable=False)
    at_intersection_with_stop = Column(Boolean, nullable=False)
    at_parking_lot = Column(Boolean, nullable=False)
    at_garage = Column(Boolean, nullable=False)
    at_driveways = Column(Boolean, nullable=False)
    at_tunnels = Column(Boolean, nullable=False)
    lane_type = Column(String(255), nullable=False)
    actors = Column(String(255), nullable=False)
