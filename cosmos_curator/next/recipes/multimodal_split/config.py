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

"""Typed config and resolution for Curator Next ``multimodal-split``."""

import json
import re
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal, Self
from urllib.parse import unquote, urlsplit

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from cosmos_curator.core.utils.storage.s3_validation import validate_configured_s3_location
from cosmos_curator.next.core.config import apply_dotted_overrides

_MODEL_CONFIG = ConfigDict(frozen=True, strict=True, extra="forbid")
_NS_PER_SECOND = 1_000_000_000
_SENSOR_KEY_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")

# Both defaults separate a fix that arrived late from one that never arrived.
#
# A 1 Hz receiver does not tick at exactly 1 Hz: measured across real sessions,
# consecutive fixes land within about +/-15 us of the nominal second. A dropped
# fix instead leaves a gap near two seconds. So the useful bound sits well above
# the jitter and well below a drop, and anything set at the nominal interval
# itself refuses healthy data -- which is what an interval of 1000.002 ms did.
#
# Half the gap bound, because the furthest a grid point can sit from the nearer
# of two fixes is half the distance between them.
_DEFAULT_MAX_DELTA_MS = 750.0
_DEFAULT_MAX_GAP_MS = 1500.0
_BITRATE_SUFFIXES = frozenset({"K", "M"})
_BITRATE_PATTERN = r"^[1-9][0-9]*(?:\.[0-9]+)?[KkMm]$"
_SUPPORTED_SCHEMES = frozenset({"s3", "file"})
_YAML_SUFFIXES = frozenset({".yaml", ".yml"})
_PARENT_SEGMENT = ".."
_RESERVED_PATH_SEGMENTS = frozenset({".", _PARENT_SEGMENT})
# ``urlsplit`` deletes these from a URI outright rather than rejecting it, which
# WHATWG requires and RFC 3986 does not. All three are legal in POSIX filenames
# and S3 keys.
_URI_STRIPPED_WHITESPACE = ("\t", "\n", "\r")

SchemaVersion = Literal[1]
MultimodalSplitKind = Literal["multimodal-split"]
# One member today. Supporting another encoder is adding it here and nowhere else.
MultimodalSplitVideoEncoder = Literal["libopenh264"]

# Widened as each kind lands. A config naming a kind the recipe cannot process
# yet is refused at load, rather than discovering it after a listing pass.
MultimodalSplitSensorType = Literal["camera", "egotrajectory", "gps", "imu"]

#: How a sensor's samples are placed on the clip's grid.
#:
#: ``nearest_neighbor`` selects the closest recorded sample for each grid point
#: and invents nothing, so a row is a measurement that really happened, carrying
#: whatever staleness the gap to it implies. ``interpolate`` derives a value
#: between the two fixes bracketing the grid point, which reads more smoothly
#: and is a value no instrument reported.
#:
#: Which is right depends on the source and on what a consumer does with it,
#: which is why it is configured rather than decided here.
MultimodalSplitSampling = Literal["nearest_neighbor", "interpolate"]

SENSOR_TYPE_CAMERA = "camera"

#: Kinds whose source artifact is an MCAP, so planning takes their bounds from a
#: topic rather than from a timeline sidecar. Says nothing about how they are cut.
MCAP_SENSOR_TYPES: frozenset[str] = frozenset({"egotrajectory", "gps", "imu"})

#: Kinds cut by copying records at the rate they were recorded. With
#: ``ALIGNED_SENSOR_TYPES`` this partitions ``MCAP_SENSOR_TYPES``, and the
#: splitter refuses a kind in neither rather than guessing how to cut it.
#:
#: An egotrajectory is recorded at 100 Hz and consumed at that rate -- lidar
#: motion compensation wants a pose at each point's own instant, not at a video
#: frame's. Resampling onto the clip grid would throw away two thirds of it
#: before any consumer saw it, so the clip carries what the sensor recorded and
#: ``core.sensors.alignment.ego_trajectory`` derives poses from that when a
#: consumer wants them on a grid.
#:
#: IMU is here for a sharper version of the same reason: at ~100-200 Hz,
#: nearest/interpolate (``ALIGNED_SENSOR_TYPES``'s two policies) would each
#: throw away all but one or two samples per grid interval, and neither
#: recovers what a consumer actually wants from an inertial stream -- a pose
#: delta integrated over the interval, not a point sample of one. That
#: integration is ``core.sensors.sensors.preintegrated_imu_sensor``'s job,
#: deliberately deferred to whenever a consumer needs it against whatever grid
#: it needs it on, rather than frozen into this clip against this one.
COPIED_SENSOR_TYPES: frozenset[str] = frozenset({"egotrajectory", "imu"})

#: Kinds cut by placing samples on the clip's grid, the same grid the cameras
#: are resampled onto.
#:
#: The opposite situation from a copied sensor: GPS arrives at 1 Hz against a
#: 30 Hz grid, so the grid is denser than the source and this is upsampling, not
#: discarding. Sharing the cameras' grid is what makes row *i* of the video and
#: row *i* of the position the same instant, which is the whole reason an
#: episode has one timeline.
ALIGNED_SENSOR_TYPES: frozenset[str] = frozenset({"gps"})


class MultimodalSplitInputConfig(BaseModel):
    """Candidate session selection resolved on the driver before Ray starts.

    Two discovery modes are expressed by this config. With ``input_path_prefix``
    alone, the immediate children of the prefix are the candidate sessions. When
    ``session_id_list_path`` is also set, the session IDs come from that file and
    are joined to the prefix instead.
    """

    model_config = _MODEL_CONFIG

    input_path_prefix: str = Field(
        min_length=1,
        description="Local directory or s3:// prefix holding one child per recording session.",
        examples=["s3://example-bucket/recordings"],
    )
    session_id_list_path: str | None = Field(
        default=None,
        description="Optional newline-delimited UTF-8 file of session IDs to join to input_path_prefix.",
        examples=["s3://example-bucket/sessions.txt"],
    )
    limit: int | None = Field(
        default=None,
        ge=1,
        description="Optional cap applied after session ID normalization, deduplication, and sorting.",
    )

    @field_validator("input_path_prefix")
    @classmethod
    def _validate_input_path_prefix(cls, prefix: str) -> str:
        return _validate_location(prefix, strip_trailing_slash=True)

    @field_validator("session_id_list_path")
    @classmethod
    def _validate_session_id_list_path(cls, path: str | None) -> str | None:
        if path is None:
            return None
        return _validate_location(path, strip_trailing_slash=False)


class MultimodalSplitClipConfig(BaseModel):
    """The episode timeline: how long a clip is, and at what rate it is sampled.

    Both belong to the clip because both are what every sensor obeys once it is
    placed on the grid at all -- a camera selects the nearest frame to each grid
    timestamp, a GPS interpolates onto them. So the rate is not a camera setting
    that the video happens to inherit; it is the grid, and the video inherits it.

    Not every sensor is placed on it. A copied sensor (``COPIED_SENSOR_TYPES``,
    e.g. egotrajectory, imu) is cut at the rate it was recorded and carries no
    relationship to this grid at all -- deriving a value against it, an IMU's
    preintegrated pose delta included, is deferred to whatever consumer needs
    that later, against whatever grid it needs it on.

    ``sampling_fps`` names what it is. The grid is built with it as
    ``sample_rate_hz``, and a clip plays at that rate as a consequence of having
    been sampled at it -- the old ``output_fps`` named the consequence.

    Clips are contiguous and non-overlapping: there is no stride setting, so the
    stride is the clip duration. A trailing span shorter than ``duration_s`` is
    dropped rather than kept short, because the runtime generates full spans only.
    """

    model_config = _MODEL_CONFIG

    duration_s: float = Field(
        default=10.0,
        gt=0.0,
        allow_inf_nan=False,
        description="Duration of one clip in seconds.",
    )
    sampling_fps: int = Field(
        default=30,
        ge=1,
        description="Rate the episode timeline is sampled at; every sensor is sampled or derived against it.",
    )

    @model_validator(mode="after")
    def _validate_clip_geometry(self) -> Self:
        """Check that a clip's duration is a whole number of sampled frames.

        A clip is a whole number of frames or it is not a valid clip, and 0.333 s
        at 30 fps is 9.99 of them. Refusing that config preserves the requested
        duration instead of silently dropping a partial frame at runtime. The
        delivered half-open grid remains authoritative for row count after the
        duration is normalized to integer nanoseconds.

        Every arithmetic check here goes through ``Decimal`` rather than float.
        The two diverge on ordinary durations -- ``65.693362 * 1e9`` is
        ``65693361999.99999`` in binary floating point -- which would make a
        remainder test reject a clip that is exact.

        Raises:
            ValueError: If the duration is not a whole number of sampled frames.

        """
        frames_per_clip = Decimal(str(self.duration_s)) * self.sampling_fps
        if frames_per_clip % 1 != 0:
            msg = (
                f"clip.duration_s ({self.duration_s}) at clip.sampling_fps ({self.sampling_fps}) is "
                f"{format(frames_per_clip.normalize(), 'f')} frames, which is not a whole number. A clip is a "
                f"whole number of sampled frames, so adjust the duration or the rate"
            )
            raise ValueError(msg)
        return self


class EncoderOpenH264Config(BaseModel):
    """Everything ``libopenh264`` needs to encode this recipe's clips.

    Settings live on the encoder rather than on a shared section because they are
    not shared. ``video_encoder`` is the discriminator: a second target -- NVENC
    is the expected one -- becomes another class beside this one, selected by that
    field, carrying whatever settings it actually has. NVENC has no bitrate-only
    rate control and this has no ``preset``, so a common section would be a
    section of fields half of which do not apply.

    There is no quality setting, and that is a limit of the encoder rather than a
    decision. ``libopenh264`` implements no constant-quality rate control:
    ``crf``, a bare ``qp``, and ``rc_mode=quality`` are all accepted and then
    silently ignored, encoding byte-identical output at every value. Bitrate is
    the only lever it honours, and a quality target needs a different encoder.

    There is no ``audio_mode``. Audio is a deferred capability for this pipeline,
    so there is no stream to copy or re-encode.
    """

    model_config = _MODEL_CONFIG

    video_encoder: MultimodalSplitVideoEncoder = "libopenh264"
    video_bitrate: str = Field(
        default="4M",
        pattern=_BITRATE_PATTERN,
        description="Target video bitrate as a magnitude and a K or M suffix, such as 4M or 800K.",
    )
    gop_size: int = Field(
        default=30,
        ge=1,
        description="Number of frames from one keyframe to the next, so 30 at 30 fps is a keyframe each "
        "second. Smaller seeks more precisely and costs size: measured on 4K footage a clip is roughly "
        "twice the nominal bitrate at 5, and one per second at 30.",
    )
    faststart: bool = Field(
        default=True,
        description="Write the MP4 index ahead of the media, so a reader finds it without a second fetch.",
    )

    @field_validator("video_bitrate", mode="before")
    @classmethod
    def _canonicalize_bitrate(cls, value: object) -> object:
        """Fold the spellings of one bitrate together before the pattern sees them.

        ``4M``, ``4.0M`` and ``4m`` all name the same rate, and a setting that
        reaches a stage in three spellings is three different strings to compare,
        log, or hash into an output identity. Canonicalizing here means the
        resolved config holds one of them.

        Anything this cannot parse is returned untouched, so the field pattern
        reports the bad value the operator actually wrote rather than some
        half-normalized rewrite of it.
        """
        if not isinstance(value, str):
            return value
        magnitude, suffix = value[:-1], value[-1:]
        if not magnitude or suffix.upper() not in _BITRATE_SUFFIXES:
            return value
        try:
            parsed = Decimal(magnitude)
        except ArithmeticError:
            return value
        return f"{format(parsed.normalize(), 'f')}{suffix.upper()}"


class MultimodalSplitVideoConfig(BaseModel):
    """How a camera becomes the video inside a clip.

    Camera-specific, which is what separates these from the episode timeline in
    ``clip``. The rate a clip is sampled at is shared by every modality; how far
    a *frame* may sit from a sample point is not -- the design document gives an
    IMU its own support-gap threshold and a GPS its own policy. ``max_delta_s``
    belongs to the rig rather than to a camera in it. It decides how far a
    source frame may sit from a grid timestamp and still be taken as that
    moment, which is to say it defines what "the same moment" means for a clip.
    Per-camera tolerances would let views disagree about simultaneity inside one
    multiview clip, which is not a thing a clip can be.

    Stated in the seconds every other duration on this surface uses, and
    converted once to the nanoseconds the sampling policy takes, so the unit an
    operator writes and the unit the sensor library works in never have to agree.
    """

    model_config = _MODEL_CONFIG

    max_delta_s: float = Field(
        default=0.05,
        gt=0.0,
        allow_inf_nan=False,
        description="How far in seconds a source frame may sit from a grid timestamp and still be selected.",
    )
    encoder: EncoderOpenH264Config = Field(default_factory=EncoderOpenH264Config)

    @property
    def max_delta_ns(self) -> int:
        """Return the tolerance in the nanoseconds the sampling policy expects.

        Rounded rather than truncated. The tolerance is a bound, and flooring a
        sub-nanosecond remainder would make it quietly tighter than what was
        configured.
        """
        return round(self.max_delta_s * _NS_PER_SECOND)


class MultimodalSplitSensorConfig(BaseModel):
    """One sensor inside a session, whatever kind it is.

    Sensors are one flat map rather than a block per kind. Grouping them by kind
    means every kind after the first is defined by not being a camera, which is a
    leftovers drawer rather than a taxonomy -- and it cannot express two IMUs, or
    a sensor whose artifact the recipe reads the same way but means something
    different by.

    ``sensor_type`` is declared rather than inferred. A file's suffix names its
    container, so it can say an artifact is an MCAP but never whether that MCAP
    holds IMU or GPS. Only the types this recipe can actually process are
    accepted, so a config naming one it cannot is refused at load rather than at
    the end of a discovery pass.

    ``uri_pattern`` is relative to a session URI, because discovery stops at the
    session boundary and this is the only place that knows a session has sensors
    in it. Anything absolute or upward-reaching would resolve outside the session
    it was joined to, so both are rejected rather than normalized.

    A whole path segment may be ``*``, so a config can name a sensor without
    knowing which recorder a rig put it on.
    """

    model_config = _MODEL_CONFIG

    sensor_type: MultimodalSplitSensorType = Field(
        description="What kind of sensor this is, which decides how it is cut.",
    )
    uri_pattern: str = Field(
        min_length=1,
        description="Path to the artifact relative to a session URI, such as */camera_front.mp4.",
    )
    topic: str | None = Field(
        default=None,
        min_length=1,
        description="MCAP topic to cut. Defaults to the file's only topic, and is required when it has several.",
    )
    required: bool = Field(
        default=True,
        description="Whether a session missing this sensor is refused, or produces episodes without it.",
    )
    allow_pts_fallback: bool = Field(
        default=False,
        description=(
            "Whether a camera missing its capture timeline may fall back to the container's own "
            "presentation timestamps instead of being refused as broken. Off by default: a "
            "constant-frame-rate mux regenerates PTS on a fixed grid, so a timeline read from it "
            "describes the encoder rather than the hardware."
        ),
    )
    sampling: MultimodalSplitSampling | None = Field(
        default=None,
        description=(
            "How samples are placed on the clip grid. Required for a sensor that is aligned rather than copied."
        ),
    )
    max_delta_ms: float | None = Field(
        default=None,
        gt=0.0,
        allow_inf_nan=False,
        description=(
            f"How stale a nearest-neighbour row may be before the clip is refused. Defaults to "
            f"{_DEFAULT_MAX_DELTA_MS:g}."
        ),
    )
    max_gap_ms: float | None = Field(
        default=None,
        gt=0.0,
        allow_inf_nan=False,
        description=(
            f"How far apart the two fixes bracketing an interpolated row may be before the clip is refused. "
            f"Defaults to {_DEFAULT_MAX_GAP_MS:g}."
        ),
    )

    @property
    def max_delta_ns(self) -> int:
        """Return the nearest-neighbour tolerance in the units the policy takes.

        Rounded rather than truncated, for the reason the video tolerance is:
        flooring a sub-nanosecond remainder makes the bound quietly tighter than
        what was configured.
        """
        return round((self.max_delta_ms if self.max_delta_ms is not None else _DEFAULT_MAX_DELTA_MS) * 1_000_000)

    @property
    def max_gap_ns(self) -> int:
        """Return the interpolation bracket bound in the units the policy takes, rounded."""
        return round((self.max_gap_ms if self.max_gap_ms is not None else _DEFAULT_MAX_GAP_MS) * 1_000_000)

    @property
    def sampling_bound_ns(self) -> int | None:
        """Return the bound the configured policy actually applies, or None if none does.

        One number rather than both, because only one is ever consulted. Carrying
        the unused one into an identity would let a setting that changes nothing
        about the output change what the output is called.
        """
        if self.sampling == "nearest_neighbor":
            return self.max_delta_ns
        if self.sampling == "interpolate":
            return self.max_gap_ns
        return None

    @model_validator(mode="after")
    def _validate_sampling_applies(self) -> Self:
        """Require a policy where there is a choice, and refuse one where there is not.

        A copied sensor is not placed on the grid at all, so naming a policy for
        it describes work the run will not do. An aligned sensor has no defensible
        default: ``nearest_neighbor`` reports only measurements that happened and
        ``interpolate`` reports values no instrument did, and which is right
        depends on the source's accuracy and on the consumer. Choosing silently
        would bake that into the artifact without anyone deciding.
        """
        aligned = self.sensor_type in ALIGNED_SENSOR_TYPES
        if aligned and self.sampling is None:
            msg = f"sensor_type {self.sensor_type!r} is placed on the clip grid, so sampling must be set"
            raise ValueError(msg)
        if not aligned and self.sampling is not None:
            msg = f"sensor_type {self.sensor_type!r} is not placed on the clip grid, so sampling cannot be set"
            raise ValueError(msg)

        # Each bound belongs to one policy. Accepting the other one silently
        # would let a config state a tolerance the run never applies.
        bound_for = {"nearest_neighbor": "max_delta_ms", "interpolate": "max_gap_ms"}
        for policy, field in bound_for.items():
            unused = self.sampling != policy and getattr(self, field) is not None
            if unused:
                applies = f"sampling {policy!r}" if self.sampling is not None else "an aligned sensor"
                msg = f"{field} applies to {applies}, so it cannot be set here"
                raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _validate_topic_applies(self) -> Self:
        """Reject a topic on a sensor that has none, rather than ignoring it.

        Only an MCAP carries topics. Accepting the field on a camera and reading
        it nowhere lets a config say something the run does not do, and the
        config is the only place an operator can see what was asked for.
        """
        if self.topic is not None and self.sensor_type not in MCAP_SENSOR_TYPES:
            msg = f"sensor_type {self.sensor_type!r} carries no topics, so topic cannot be set"
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _validate_pts_fallback_applies(self) -> Self:
        """Reject PTS fallback on a sensor that has no capture timeline to fall back from.

        Only a camera reads a capture timeline in the first place, so only a
        camera has anything to fall back from.
        """
        if self.allow_pts_fallback and self.sensor_type != SENSOR_TYPE_CAMERA:
            msg = f"sensor_type {self.sensor_type!r} has no capture timeline, so allow_pts_fallback cannot be set"
            raise ValueError(msg)
        return self

    @field_validator("uri_pattern")
    @classmethod
    def _validate_uri_pattern(cls, pattern: str) -> str:
        if pattern != pattern.strip():
            msg = "sensor uri_pattern cannot have leading or trailing whitespace"
            raise ValueError(msg)
        if "://" in pattern or pattern.startswith("/"):
            msg = (
                f"sensor uri_pattern {pattern!r} must be relative to a session, so it can be joined "
                f"onto any session URI. Drop the scheme or the leading slash"
            )
            raise ValueError(msg)
        _reject_parent_segments(pattern, pattern)
        for segment in pattern.split("/"):
            # A whole segment may be a wildcard, so a config need not know which
            # recorder holds a sensor. Part of one may not: that matches on file
            # names rather than on layout, and resolves differently as names change.
            if "*" in segment and segment != "*":
                msg = (
                    f"sensor uri_pattern {pattern!r} may use * as a whole path segment, not inside one. "
                    f"Write */name.mp4 rather than rec*/name.mp4"
                )
                raise ValueError(msg)
        return pattern


class MultimodalSplitOutputConfig(BaseModel):
    """Output root for camera artifacts, diagnostics, and canonical episodes."""

    model_config = _MODEL_CONFIG

    path: str = Field(
        min_length=1,
        description="Local directory or s3:// prefix containing sessions/, errors/, and episodes.lance.",
    )

    # Publication deliberately remains in schema v1: it is unconditional,
    # derives episodes.lance from output.path, and adds no setting or opt-out.
    @property
    def episodes_lance_uri(self) -> str:
        """Return the fixed canonical episode-table location beneath the output root."""
        if "://" not in self.path:
            return f"{self.path.rstrip('/')}/episodes.lance"
        # Preserve URI root separators: stripping the whole ``file:///`` string
        # would turn the filesystem root into the relative-looking ``file:``.
        parsed = urlsplit(self.path)
        return parsed._replace(path=f"{parsed.path.rstrip('/')}/episodes.lance").geturl()

    @field_validator("path")
    @classmethod
    def _validate_path(cls, path: str) -> str:
        return _validate_location(path, strip_trailing_slash=True)


class SpanPlannerResources(BaseModel):
    """What one planning task reserves.

    Planning reads a listing and one timeline per sensor and does almost nothing
    with a CPU while it waits. Measured, a task keeps 0.13 of a core busy over
    S3 and 0.29 over a local disk -- which is why this is a setting rather than
    a constant: the right figure is a property of the storage a run reads from,
    not of the code.

    It also sets memory, since Ray Data scales its default reservation by the
    CPU a task asks for. Declaring a whole CPU reserves 2.4 GiB for a task that
    holds twelve small arrays, and it is the memory rather than the cores that
    then binds: on a node with sixty-four cores and 124 GB of RAM that caps the
    run near thirty tasks, and on the same cores with far more RAM it would not.
    """

    model_config = _MODEL_CONFIG

    num_cpus: float = Field(
        default=0.125,
        gt=0.0,
        le=1.0,
        description="CPU one planning task reserves; measured at 0.13 over S3.",
    )


class SplitSessionResources(BaseModel):
    """What one splitting task reserves.

    A task decodes one sensor of one session, which is where a run's memory
    actually goes. Left unset, Ray Data invents a flat 2.4 GiB per CPU -- a
    hyperscaler rule of thumb that describes nothing here -- and caps
    concurrency by it however much headroom there is.

    Unset by default because a guess of ours would be worth no more than Ray's:
    peak depends on the footage, and the figure should come from measuring it.
    """

    model_config = _MODEL_CONFIG

    memory_gb: float | None = Field(
        default=None,
        gt=0.0,
        description="Heap one splitting task reserves, in GiB, or null to accept Ray's own default.",
    )


class MultimodalSplitExecutionConfig(BaseModel):
    """Settings that decide how a run is carried out, never what it produces.

    Deliberately separate from everything a clip's identity hashes. Nothing here
    changes a written byte, so retuning any of it must not relocate media that is
    already correct -- which is exactly what would happen if these lived beside
    the settings that do decide the output.

    ``progress`` and ``storage_profile`` mirror ``video_split`` and
    ``data_integrity``, which carry the same two, so one vocabulary covers the
    recipes an operator moves between.

    Resources are stated per stage, under the name the stage has in the code.
    What a task reserves is a property of the work it does: planning waits on
    storage and holds twelve small arrays, splitting decodes 4K video. One
    figure for both would be wrong for whichever it was not measured against.

    """

    model_config = _MODEL_CONFIG

    progress: bool = Field(
        default=False,
        description="Draw Ray Data progress bars. Off by default: a batch run writes to a log.",
    )
    storage_profile: str = Field(
        default="default",
        min_length=1,
        description="Profile name used to build storage clients for reads and writes.",
    )
    read_buffer_bytes: int | None = Field(
        default=None,
        ge=1,
        description="Bytes smart_open buffers per read from an S3 object, or null for its own 128 KiB default.",
    )
    span_planner: SpanPlannerResources = Field(default_factory=SpanPlannerResources)
    split_session: SplitSessionResources = Field(default_factory=SplitSessionResources)


class ResolvedMultimodalSplitConfig(BaseModel):
    """Canonical execution contract for Curator Next ``multimodal-split``.

    ``input`` selects the candidate sessions, ``sensors`` names the artifacts to
    read from each one, ``clip`` says how long a clip is, ``video`` describes how
    a camera becomes a clip's media, and ``output`` says where it is written.

    ``clip``, ``video`` and ``execution`` are fully defaulted, so a config written
    before any of them existed still resolves. ``sensors`` and ``output`` are not:
    there is no defensible default for which artifact inside a session holds the
    video, or for where a run should write. Both are required.
    """

    model_config = _MODEL_CONFIG

    schema_version: SchemaVersion
    kind: MultimodalSplitKind
    input: MultimodalSplitInputConfig
    sensors: dict[str, MultimodalSplitSensorConfig] = Field(
        min_length=1,
        description="Sensor key to artifact. Every entry is written for every clip.",
    )

    @model_validator(mode="after")
    def _validate_a_camera_is_configured(self) -> Self:
        """Refuse a rig with no camera, which cannot produce an episode.

        ``min_length`` on ``sensors`` carried this while a camera was the only
        kind of sensor there was. It no longer does: a config naming only a
        trajectory satisfies it, then fails per session in planning, so a run
        exits successfully having written nothing but error reports.

        The cameras define the episode's timeline -- its span, its grid, and its
        row count all come from them -- so a clip without one is not a clip with
        a modality missing, it is nothing to hang the other modalities on.
        """
        if not self.cameras:
            kinds = ", ".join(sorted({sensor.sensor_type for sensor in self.sensors.values()})) or "none"
            msg = f"sensors must include at least one camera, which defines the episode timeline; got {kinds}"
            raise ValueError(msg)
        return self

    output: MultimodalSplitOutputConfig
    clip: MultimodalSplitClipConfig = Field(default_factory=MultimodalSplitClipConfig)
    video: MultimodalSplitVideoConfig = Field(default_factory=MultimodalSplitVideoConfig)
    execution: MultimodalSplitExecutionConfig = Field(default_factory=MultimodalSplitExecutionConfig)

    @property
    def cameras(self) -> dict[str, MultimodalSplitSensorConfig]:
        """Return the sensors this recipe reads as video, keyed as configured.

        The camera/not-camera split is real inside the pipeline -- cameras define
        the grid every other sensor is cut against -- so it is derived here from
        what each sensor says it is, rather than being a shape an operator has to
        express by choosing which block to write a sensor in.
        """
        return {key: sensor for key, sensor in self.sensors.items() if sensor.sensor_type == SENSOR_TYPE_CAMERA}

    @property
    def mcap_sensors(self) -> dict[str, MultimodalSplitSensorConfig]:
        """Return the sensors whose source is an MCAP, however their clip is written.

        Planning takes their bounds from a topic rather than a timeline sidecar,
        which is all this set decides. ``COPIED_SENSOR_TYPES`` and
        ``ALIGNED_SENSOR_TYPES`` partition it by how the clip is produced.
        """
        return {key: sensor for key, sensor in self.sensors.items() if sensor.sensor_type in MCAP_SENSOR_TYPES}

    @field_validator("sensors")
    @classmethod
    def _validate_sensor_keys(
        cls, sensors: dict[str, MultimodalSplitSensorConfig]
    ) -> dict[str, MultimodalSplitSensorConfig]:
        for key in sensors:
            # "." and "." * 2 match the character class but name a directory rather than sit in one.
            if not _SENSOR_KEY_PATTERN.fullmatch(key) or key in _RESERVED_PATH_SEGMENTS:
                msg = (
                    f"sensor key {key!r} is written into output paths, so it must be a single path "
                    f"segment of letters, digits, dots, dashes, or underscores"
                )
                raise ValueError(msg)
        return sensors


_TEMPLATE_BASE: dict[str, Any] = {
    "schema_version": 1,
    "kind": "multimodal-split",
    "input": {"input_path_prefix": "s3://example-bucket/recordings"},
    "sensors": {"front": {"sensor_type": "camera", "uri_pattern": "*/camera_front.mp4"}},
    "output": {"path": "s3://example-bucket/media"},
}
# Settings that default to null are dumped away by ``exclude_none``, so the ones
# an operator would otherwise never discover are named here instead.
_TEMPLATE_PREAMBLE = """\
# Every setting with a non-null default is shown; unchanged settings may be removed.
# These default to null, so they are omitted above and are named here instead:
#   input.session_id_list_path  read session IDs from a file instead of listing the prefix
#   input.limit                 cap how many sessions are kept
#   sensors.<key>.topic         which topic to cut from an MCAP carrying more than one
#   sensors.<key>.sampling      nearest_neighbor or interpolate; required for a gps sensor
#   sensors.<key>.max_delta_ms  how stale a nearest_neighbor row may be
#   sensors.<key>.max_gap_ms    how far apart the fixes bracketing an interpolated row may be
"""


def config_template() -> dict[str, Any]:
    """Return an editable config template carrying every non-null default.

    Derived from the model rather than hand-written, so a section added to
    ``ResolvedMultimodalSplitConfig`` reaches ``cosmos-curator pipeline template``
    with no edit here and cannot disagree with what the model accepts.

    Built per call rather than as a module constant for two reasons. The
    validators it runs are defined below this point in the module, so an
    import-time constant here raises ``NameError``. Moving it past them would fix
    that but would make every import of this module validate the example ``s3://``
    prefix, which pulls in boto3 -- the cost ``_validate_s3_location`` defers its
    import to avoid. A fresh mapping per call is also what callers may edit in
    place.
    """
    return ResolvedMultimodalSplitConfig.model_validate(_TEMPLATE_BASE).model_dump(mode="json", exclude_none=True)


def config_template_yaml() -> str:
    """Render the template as YAML for ``cosmos-curator pipeline template``."""
    return _TEMPLATE_PREAMBLE + yaml.safe_dump(config_template(), sort_keys=False)


def load_config(config_path: str | Path) -> ResolvedMultimodalSplitConfig:
    """Load and validate a ``multimodal-split`` config file."""
    return resolve_config(config_path)


def resolve_config(
    config_path: str | Path,
    overrides: tuple[str, ...] | list[str] = (),
) -> ResolvedMultimodalSplitConfig:
    """Load a config file and apply ``--set`` overrides before validation.

    Overrides go through the shared dotted-override helper every Curator Next
    recipe uses, so ``--set`` behaves identically across kinds. Each has the form
    ``"path.to.key=value"``, where the value is parsed with ``yaml.safe_load`` so
    numeric, boolean, and null literals become native types instead of strings.
    That matters here because the config is strict: a quoted ``"5"`` would be
    rejected for ``input.limit``. A bare ``key=`` assigns the empty string; null
    stays reachable through YAML's own ``null`` and ``~`` spellings.

    Example::

        resolve_config("config.yaml", overrides=["input.limit=10"])
    """
    path = Path(config_path)
    if not path.exists():
        msg = f"Config file not found: {path}"
        raise FileNotFoundError(msg)
    with path.open(encoding="utf-8") as handle:
        loaded: object = yaml.safe_load(handle) if path.suffix.lower() in _YAML_SUFFIXES else json.load(handle)
    if not isinstance(loaded, dict):
        msg = f"Config file must contain a mapping at the top level, got {type(loaded).__name__}: {path}"
        raise TypeError(msg)

    raw: dict[str, object] = loaded
    apply_dotted_overrides(raw, overrides)
    return ResolvedMultimodalSplitConfig.model_validate(raw)


def _validate_location(location: str, *, strip_trailing_slash: bool) -> str:
    """Canonicalize a local path, ``file://`` URI, or ``s3://`` URI.

    ``strip_trailing_slash`` applies to prefixes, whose trailing slashes are not
    meaningful, but never to object names such as the session ID list file.
    """
    if not location or not location.strip():
        msg = "Storage locations cannot be empty"
        raise ValueError(msg)
    if location != location.strip():
        msg = "Storage locations cannot have leading or trailing whitespace"
        raise ValueError(msg)

    if "://" not in location:
        _reject_parent_segments(location, location)
        return _strip_trailing_slash(location) if strip_trailing_slash else location

    raw_scheme, remainder = location.split("://", maxsplit=1)
    scheme = raw_scheme.lower()
    if scheme not in _SUPPORTED_SCHEMES:
        msg = f"Unsupported storage scheme in {location!r}; expected local paths, file:// URIs, or s3:// URIs"
        raise ValueError(msg)

    normalized = f"{scheme}://{remainder}"
    if any(character in normalized for character in _URI_STRIPPED_WHITESPACE):
        # Unlike '?' and '#', these do not truncate: they vanish, leaving a
        # location that names a different directory and looks entirely ordinary.
        # ``file:///data/rec\tord`` canonicalizes to ``file:///data/record``.
        msg = (
            f"Storage location {location!r} contains a tab, carriage return, or newline, which URI parsing "
            f"deletes as whitespace. Percent-encode them as %09, %0D, and %0A, or pass the location as a "
            f"plain path instead of a URI"
        )
        raise ValueError(msg)

    parsed = urlsplit(normalized)
    if parsed.query or parsed.fragment:
        # ``?`` and ``#`` are URI delimiters, so ``urlsplit`` drops everything from
        # the first one onwards out of ``path``. Both are legal in POSIX directory
        # names and S3 keys, so an unencoded one does not fail: it silently names a
        # shorter location than the author wrote. ``file:///data/my#dir`` reads
        # ``/data/my``. Rejecting is the only option that cannot lose data, since a
        # bare local path and a percent-encoded URI both express the real name.
        msg = (
            f"Storage location {location!r} contains an unencoded '?' or '#'. "
            f"Percent-encode them as %3F and %23, or pass the location as a plain path instead of a URI"
        )
        raise ValueError(msg)
    if scheme == "file":
        if parsed.netloc not in {"", "localhost"} or not parsed.path:
            msg = f"Unsupported local file URI: {location}"
            raise ValueError(msg)
        _reject_parent_segments(location, unquote(parsed.path))
    else:
        _reject_parent_segments(location, parsed.path)

    canonical = _strip_trailing_slash(normalized) if strip_trailing_slash else normalized
    if scheme == "s3":
        _validate_s3_location(canonical)
    return canonical


def _validate_s3_location(location: str) -> str:
    """Run the storage layer's own bucket and key validation at config time.

    Checking here moves failures from the middle of a run to
    ``cosmos-curator pipeline validate``. The shared helper also adds the
    config-only rejection of ``*`` and ``?``; ``S3Prefix`` must accept those
    characters because listings can return real object keys containing them.
    """
    validate_configured_s3_location(location)
    return location


def _reject_parent_segments(location: str, path: str) -> None:
    """Reject ``..`` segments, which no reader here can agree on how to resolve.

    Locally, ``..`` after a symlink means two different directories at once: the
    kernel resolves the link first, so ``os.scandir`` lists the link target's
    parent, while ``os.path.abspath`` cancels the segment lexically and names the
    link's own parent. Discovery needs both -- it lists with one and builds
    ``session_uri`` with the other -- so a prefix like ``/data/latest/..`` would
    emit URIs under a directory whose children were never listed.

    In S3 there is no divergence but no meaning either: keys are opaque, so
    ``..`` is a literal name component rather than a parent reference.

    Neither reading is worth guessing at, and an absolute path always says what
    was meant, so ``..`` is rejected before any listing happens.
    """
    if _PARENT_SEGMENT in path.split("/"):
        msg = (
            f"Storage location {location!r} contains a '..' segment. "
            f"Write the location it resolves to instead: '..' names one directory before a symlink "
            f"and another after it, and is a literal key component in S3"
        )
        raise ValueError(msg)


def _strip_trailing_slash(location: str) -> str:
    """Drop trailing slashes from the location itself, never from its scheme.

    Stripping the whole string would turn ``s3://`` into ``s3:`` and ``file:///``
    into ``file:``. Neither is recognized as a remote path afterwards, so both
    would silently degrade into a path relative to the driver's working
    directory instead of failing.
    """
    scheme, separator, _ = location.partition("://")
    if not separator:
        return location.rstrip("/") or "/"

    parsed = urlsplit(location)
    if scheme != "file" and not parsed.netloc:
        msg = f"Storage location {location!r} names a scheme but no location"
        raise ValueError(msg)

    # Strip the path only. Stripping the whole string drops the authority too, so
    # ``file://localhost/`` would become ``file://localhost``, which has no path
    # and is rejected later by the very code meant to read it.
    stripped_path = parsed.path.rstrip("/")
    if scheme == "file" and not stripped_path:
        stripped_path = "/"
    return parsed._replace(path=stripped_path).geturl()
