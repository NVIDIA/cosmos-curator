# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Data Model util."""

import enum
import json
import pathlib
import sys
from collections import deque
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Self
from uuid import UUID

import attrs
import numpy as np
import numpy.typing as npt
from loguru import logger

import cosmos_curator.pipelines.video.filtering.motion.motion_vector_backend as motion
from cosmos_curator.core.interfaces.stage_interface import PipelineTask
from cosmos_curator.core.utils.data.lazy_data import LazyData
from cosmos_curator.core.utils.infra.performance_utils import StagePerfStats
from cosmos_curator.core.utils.storage import storage_client
from cosmos_curator.pipelines.video.utils.decoder_utils import extract_video_metadata, get_video_timestamps

if TYPE_CHECKING:
    from collections.abc import Iterable

try:
    import torch

    TensorType = getattr(torch, "Tensor", None)
except ImportError:
    TensorType = None


def _get_object_size(obj: object) -> int:
    """Get the size of a single object.

    Lists, tuples, sets, and frozensets return 0 since we only count contents,
    not the container itself. The calling function is expected to extract the
    contents of the container and call this function on each item.

    Arguments:
        obj: The object to get the size of.

    Returns:
        The size of the object in bytes.

    """
    if obj is None:
        return 0
    if isinstance(obj, (np.ndarray, np.generic)):
        return obj.nbytes
    if TensorType is not None and isinstance(obj, TensorType):
        if hasattr(obj, "element_size") and hasattr(obj, "nelement"):
            return int(obj.element_size() * obj.nelement())
        return sys.getsizeof(obj)  # Fallback for unexpected tensor types
    # For containers, only count contents, not the container itself
    if isinstance(obj, (dict, list, tuple, set, frozenset)):
        return 0
    return sys.getsizeof(obj)


def _add_children_to_queue(obj: object, q: deque[object], visited: set[int]) -> None:
    """Add child objects to the queue for processing."""
    children: Iterable[object] = iter([])
    # Skip transient performance tracking fields that shouldn't count toward data size
    _skip_fields = {"stage_perf"}

    if isinstance(obj, dict):
        children = obj.values()
    elif isinstance(obj, (list, tuple, set, frozenset)):
        children = obj
    elif attrs.has(obj.__class__):
        children = (getattr(obj, field.name) for field in attrs.fields(obj.__class__) if field.name not in _skip_fields)

    for child in children:
        if id(child) not in visited:
            q.append(child)


def get_major_size(obj: object) -> int:
    """Get the memory size of an attrs instance in bytes.

    This function is used to get the memory size of an attrs instance in bytes.
    It can handle circular references and nested structures. It does not count
    the size of the containers themselves, only the contents.

    Args:
        obj: The object to get the memory size of.

    Returns:
        The memory size of the object in bytes.

    """
    size = 0
    visited: set[int] = set()
    q: deque[object] = deque([obj])

    while q:
        current_obj = q.popleft()
        if id(current_obj) in visited:
            continue
        visited.add(id(current_obj))

        size += _get_object_size(current_obj)
        _add_children_to_queue(current_obj, q, visited)

    return size


@attrs.define
class TokenCounts:
    """Token usage from a single vLLM inference (prompt + output)."""

    prompt_tokens: int = 0
    output_tokens: int = 0


class CaptionOutcome(enum.StrEnum):
    """Normalized captioning outcomes written by caption backends."""

    SUCCESS = "success"
    TRUNCATED = "truncated"
    BLOCKED = "blocked"
    ERROR = "error"
    SKIPPED = "skipped"


CAPTION_OK_STATUSES = {CaptionOutcome.SUCCESS, CaptionOutcome.TRUNCATED}
CAPTION_STATUS_KEYS = tuple(outcome.value for outcome in CaptionOutcome)


type CaptionFailureReason = Literal["exception", "timeout"]
CAPTION_FAILURE_REASON_KEYS = ("exception", "timeout")
CAPTION_QUALITY_FLAG_KEYS = ("flag_length_outlier", "flag_repetition", "flag_near_duplicate")


def _zero_counts(keys: tuple[str, ...]) -> dict[str, int]:
    return dict.fromkeys(keys, 0)


def _merge_counts(target: dict[str, int], source: dict[str, int]) -> None:
    for key, value in source.items():
        target[key] = target.get(key, 0) + value


def _serialized_counts(
    counts: dict[str, int], known_keys: tuple[str, ...], *, include_unknown_keys: bool
) -> dict[str, int]:
    data = {key: counts.get(key, 0) for key in known_keys}
    if include_unknown_keys:
        for key in sorted(set(counts) - set(known_keys)):
            data[key] = counts[key]
    return data


def _count_value(data: Mapping[str, Any], key: str) -> int:
    value = data.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        msg = f"{key} must be an integer"
        raise TypeError(msg)
    if value < 0:
        msg = f"{key} must be non-negative"
        raise ValueError(msg)
    return value


def _count_map(data: Mapping[str, Any], key: str, known_keys: tuple[str, ...]) -> dict[str, int]:
    raw = data.get(key)
    if not isinstance(raw, Mapping):
        msg = f"{key} must be an object"
        raise TypeError(msg)

    unknown_keys = set(raw) - set(known_keys)
    if unknown_keys:
        unknown = ", ".join(sorted(str(item) for item in unknown_keys))
        msg = f"{key} has unknown keys: {unknown}"
        raise ValueError(msg)

    missing_keys = set(known_keys) - set(raw)
    if missing_keys:
        missing = ", ".join(sorted(missing_keys))
        msg = f"{key} is missing keys: {missing}"
        raise ValueError(msg)

    counts: dict[str, int] = {}
    for known_key in known_keys:
        value = raw[known_key]
        if not isinstance(value, int) or isinstance(value, bool):
            msg = f"{key}.{known_key} must be an integer"
            raise TypeError(msg)
        if value < 0:
            msg = f"{key}.{known_key} must be non-negative"
            raise ValueError(msg)
        counts[known_key] = value
    return counts


@attrs.define
class CaptionQualityStats:
    """Run-level caption structural-health counters."""

    SCHEMA_VERSION: ClassVar[int] = 1
    PIPELINE: ClassVar[str] = "split_video_pipeline"

    caption_windows_checked: int = 0
    caption_status_counts: dict[str, int] = attrs.Factory(lambda: _zero_counts(CAPTION_STATUS_KEYS))
    caption_failure_reason_counts: dict[str, int] = attrs.Factory(lambda: _zero_counts(CAPTION_FAILURE_REASON_KEYS))
    caption_quality_flags_evaluated_count: int = 0
    caption_quality_flag_counts: dict[str, int] = attrs.Factory(lambda: _zero_counts(CAPTION_QUALITY_FLAG_KEYS))
    empty_caption_count: int = 0
    sentinel_caption_count: int = 0

    @classmethod
    def from_dict(cls, data: object) -> Self:
        """Build stats from a serialized aggregate, validating v1 keys and counter shapes.

        Chunk-embedded stats omit ``schema_version`` and ``pipeline``; root artifacts include them.
        """
        if not isinstance(data, Mapping):
            msg = "caption_quality_stats must be an object"
            raise TypeError(msg)

        schema_version = data.get("schema_version")
        if schema_version is not None and schema_version != cls.SCHEMA_VERSION:
            msg = f"schema_version must be {cls.SCHEMA_VERSION}"
            raise ValueError(msg)

        pipeline = data.get("pipeline")
        if pipeline is not None and pipeline != cls.PIPELINE:
            msg = f"pipeline must be {cls.PIPELINE}"
            raise ValueError(msg)

        stats = cls(
            caption_windows_checked=_count_value(data, "caption_windows_checked"),
            caption_status_counts=_count_map(data, "caption_status_counts", CAPTION_STATUS_KEYS),
            caption_failure_reason_counts=_count_map(
                data,
                "caption_failure_reason_counts",
                CAPTION_FAILURE_REASON_KEYS,
            ),
            caption_quality_flags_evaluated_count=_count_value(data, "caption_quality_flags_evaluated_count"),
            caption_quality_flag_counts=_count_map(
                data,
                "caption_quality_flag_counts",
                CAPTION_QUALITY_FLAG_KEYS,
            ),
            empty_caption_count=_count_value(data, "empty_caption_count"),
            sentinel_caption_count=_count_value(data, "sentinel_caption_count"),
        )
        errors = stats.validation_errors()
        if errors:
            msg = "; ".join(errors)
            raise ValueError(msg)
        return stats

    def copy(self) -> Self:
        """Return an independent copy of the aggregate."""
        return type(self)(
            caption_windows_checked=self.caption_windows_checked,
            caption_status_counts=dict(self.caption_status_counts),
            caption_failure_reason_counts=dict(self.caption_failure_reason_counts),
            caption_quality_flags_evaluated_count=self.caption_quality_flags_evaluated_count,
            caption_quality_flag_counts=dict(self.caption_quality_flag_counts),
            empty_caption_count=self.empty_caption_count,
            sentinel_caption_count=self.sentinel_caption_count,
        )

    def combine(self, other: Self) -> None:
        """Merge another caption quality aggregate into this one."""
        self.caption_windows_checked += other.caption_windows_checked
        _merge_counts(self.caption_status_counts, other.caption_status_counts)
        _merge_counts(self.caption_failure_reason_counts, other.caption_failure_reason_counts)
        self.caption_quality_flags_evaluated_count += other.caption_quality_flags_evaluated_count
        _merge_counts(self.caption_quality_flag_counts, other.caption_quality_flag_counts)
        self.empty_caption_count += other.empty_caption_count
        self.sentinel_caption_count += other.sentinel_caption_count

    def _invariant_errors(self) -> list[str]:
        errors: list[str] = []
        if sum(self.caption_status_counts.values()) != self.caption_windows_checked:
            errors.append("caption_status_counts must sum to caption_windows_checked")

        error_count = self.caption_status_counts[CaptionOutcome.ERROR.value]
        failure_reason_total = sum(self.caption_failure_reason_counts.values())
        if failure_reason_total > error_count:
            errors.append("caption_failure_reason_counts must not exceed error status count")

        errors.extend(
            f"caption_quality_flag_counts.{key} must not exceed evaluated count"
            for key in CAPTION_QUALITY_FLAG_KEYS
            if self.caption_quality_flag_counts[key] > self.caption_quality_flags_evaluated_count
        )

        ok_count = (
            self.caption_status_counts[CaptionOutcome.SUCCESS.value]
            + self.caption_status_counts[CaptionOutcome.TRUNCATED.value]
        )
        if self.empty_caption_count + self.sentinel_caption_count > ok_count:
            errors.append("empty and sentinel counts must not exceed OK status count")

        return errors

    def validation_errors(self) -> list[str]:
        """Return contract invariant failures for this aggregate."""
        return self._invariant_errors()

    def to_dict(self, *, include_schema: bool = False, include_unknown_keys: bool = False) -> dict[str, Any]:
        """Serialize the aggregate using the v1 artifact field names."""
        data: dict[str, Any] = {}
        if include_schema:
            data["schema_version"] = self.SCHEMA_VERSION
            data["pipeline"] = self.PIPELINE

        data.update(
            {
                "caption_windows_checked": self.caption_windows_checked,
                "caption_status_counts": _serialized_counts(
                    self.caption_status_counts,
                    CAPTION_STATUS_KEYS,
                    include_unknown_keys=include_unknown_keys,
                ),
                "caption_failure_reason_counts": _serialized_counts(
                    self.caption_failure_reason_counts,
                    CAPTION_FAILURE_REASON_KEYS,
                    include_unknown_keys=include_unknown_keys,
                ),
                "caption_quality_flags_evaluated_count": self.caption_quality_flags_evaluated_count,
                "caption_quality_flag_counts": _serialized_counts(
                    self.caption_quality_flag_counts,
                    CAPTION_QUALITY_FLAG_KEYS,
                    include_unknown_keys=include_unknown_keys,
                ),
                "empty_caption_count": self.empty_caption_count,
                "sentinel_caption_count": self.sentinel_caption_count,
            }
        )
        return data


@attrs.define
class CaptionResult:
    """Normalized caption result returned by backend adapters."""

    outcome: CaptionOutcome
    text: str | None = None
    failure_reason: CaptionFailureReason | None = None


@attrs.define
class Window:
    """Container for captioning window."""

    # Start frame number of this window
    start_frame: int
    # End frame number of this window
    end_frame: int
    # MP4 bytes for this window; wrapped in LazyData for zero-copy inter-stage
    # transport via PEP 574 (bytes auto-converted to numpy by LazyData.coerce).
    mp4_bytes: LazyData[npt.NDArray[np.uint8]] = attrs.field(factory=LazyData, converter=LazyData.coerce)  # type: ignore[misc]
    # Model input for this window: model_variant -> llm_input
    model_input: dict[str, dict[str, Any]] = attrs.Factory(dict)
    # `caption: {model_name: caption}`
    caption: dict[str, str] = attrs.Factory(dict)
    enhanced_caption: dict[str, str] = attrs.Factory(dict)
    # Token counts from vLLM inference: {model_variant: TokenCounts}
    token_counts: dict[str, TokenCounts] = attrs.Factory(dict)
    # t5_xxl embeddings for this window
    t5_xxl_embedding: dict[str, npt.NDArray[np.int32]] = attrs.Factory(dict)
    # webp preview; wrapped in LazyData for zero-copy inter-stage transport
    # via PEP 574 (bytes auto-converted to numpy by LazyData.coerce).
    webp_bytes: LazyData[npt.NDArray[np.uint8]] = attrs.field(factory=LazyData, converter=LazyData.coerce)  # type: ignore[misc]
    # caption outcome; set by caption stages.
    caption_status: Literal["success", "truncated", "blocked", "error", "skipped"] | None = None
    # set only when caption_status == "error"
    caption_failure_reason: CaptionFailureReason | None = None
    flag_length_outlier: bool | None = None
    flag_repetition: bool | None = None
    flag_near_duplicate: bool | None = None
    # for debugging
    errors: dict[str, str] = attrs.Factory(dict)

    def get_major_size(self) -> int:
        """Calculate total memory size of the window.

        Returns:
            Total size in bytes.

        """
        return get_major_size(self)


@attrs.define
class Clip:
    """Container for video clip data including metadata, frames, and processing results.

    This class stores information about a video segment, including its source, timing,
    extracted frames, motion data, aesthetic scores, and generated captions.

    ``span`` is the clip's time range on the original source video (seconds).
    Each ``Window`` in ``windows`` covers a contiguous slice of native frames
    within the clip.  The relationship between the two coordinate systems::

        source timeline
        span[0]=10s |=====[w0: 10-14.7s]=====[w1: 15.3-20s]=====| span[1]=20s
                    ^clip start                                   ^clip end
                         ^window source_start  ^window source_end  (for w0)

    See :func:`~cosmos_curator.pipelines.video.utils.windowing_utils.window_source_time_bounds_from_clip`
    for the linear mapping from ``Window.start_frame`` / ``end_frame`` to source seconds.
    """

    uuid: UUID
    source_video: str
    span: tuple[float, float]
    encoded_data: LazyData[npt.NDArray[np.uint8]] = attrs.field(factory=LazyData, converter=LazyData.coerce)  # type: ignore[misc]
    # decoded frames (dict of numpy arrays keyed by extraction signature);
    # wrapped in LazyData for zero-copy inter-stage transport via PEP 574.
    # No converter: producer must compute nbytes explicitly (dict has no .nbytes attr)
    extracted_frames: LazyData[dict[str, npt.NDArray[np.uint8]]] = attrs.field(factory=LazyData)
    # motion
    decoded_motion_data: motion.DecodedData | None = None
    motion_score_global_mean: float | None = None
    motion_score_per_patch_min_256: float | None = None
    # aesthetic
    aesthetic_score: float | None = None
    # artificial text (overlay / post-production text filter)
    has_artificial_text: bool | None = None
    artificial_text_segments: list[dict[str, Any]] | None = None
    # embedding frames; wrapped in LazyData for API consistency and Phase 2
    # split-field ObjectRef potential (already numpy, so zero-copy via PEP 574).
    cosmos_embed1_frames: LazyData[npt.NDArray[np.float32]] = attrs.field(factory=LazyData, converter=LazyData.coerce)  # type: ignore[misc]
    cosmos_embed1_embedding: npt.NDArray[np.float32] | None = None
    intern_video_2_frames: LazyData[npt.NDArray[np.float32]] = attrs.field(factory=LazyData, converter=LazyData.coerce)  # type: ignore[misc]
    intern_video_2_embedding: npt.NDArray[np.float32] | None = None
    openai_embedding: npt.NDArray[np.float32] | None = None
    # captioning
    windows: list[Window] = attrs.Factory(list)
    filter_windows: list[Window] = attrs.Factory(list)
    # for testing
    cosmos_embed1_text_match: tuple[str, float] | None = None
    intern_video_2_text_match: tuple[str, float] | None = None
    # for debugging
    errors: dict[str, str] = attrs.Factory(dict)
    # Qwen video classifier: list of VIDEO_TYPE_LABELS that were "yes" in any window (for metadata/debug)
    qwen_type_classification: list[str] | None = None
    # When clip is in filtered_clips due to Qwen: "classifier" (type allow/block) or "semantic" (criteria filter)
    qwen_rejection_stage: str | None = None
    # SAM3 object tracking: list of per-prompt tracking results; populated by SAM3TrackingStage
    sam3_tracked_objects: list[dict[str, Any]] = attrs.Factory(list)
    # fraction [0,1] of frames where at least one tracked object is present; populated by SAM3TrackingStage
    sam3_foreground_coverage: float | None = None
    # SAM3 bbox stage outputs (populated by SAM3BBoxStage); None when SAM3 did not run.
    # Each entry summarises one tracked object_id across the entire clip with
    # fields object_id (int), prompt (str), start_time_s (float, wall-clock
    # seconds with ms precision), end_time_s (float), and num_frames (int).
    # Seconds (not frame indices) are used so the downstream VLM captioning
    # stage can pass the payload to a VLM unchanged.
    sam3_instances: list[dict[str, Any]] | None = None
    # Per-frame object detections: {frame_idx: [{object_id, prompt, box_xyxy}, ...]}
    sam3_objects_by_frame: dict[int, list[dict[str, Any]]] | None = None
    # Per-event VLM annotations populated by PerEventCaptionStage. The shape
    # of each entry is defined entirely by the prompt — the stage passes the
    # model's JSON output through unchanged, so downstream consumers must be
    # robust to whatever shape the configured prompt asks for.
    sam3_events: list[Any] | None = None
    # Re-encoded mp4 bytes with boxes/masks/ids/trails drawn on top (produced by AnnotatedVideoWriterStage).
    # Wrapped in LazyData for zero-copy inter-stage transport via PEP 574.
    sam3_annotated_video: LazyData[npt.NDArray[np.uint8]] = attrs.field(factory=LazyData, converter=LazyData.coerce)  # type: ignore[misc]

    def get_all_captions(self) -> list[str]:
        """Get all captions from the clip's windows.

        Returns:
            A list of all captions from the clip's windows.

        """
        captions: list[str] = []
        for window in self.windows:
            captions.extend(window.caption.values())
        return captions

    def extract_metadata(self) -> dict[str, Any] | None:
        """Extract metadata from the clip's encoded_data.

        Returns:
            A dictionary containing the extracted metadata (width, height, framerate,
            num_frames, video_codec, num_bytes) if encoded_data exists, None otherwise.

        Raises:
            Exception: Any exception from extract_video_metadata is propagated.

        """
        data = self.encoded_data.resolve()
        if data is None:
            return None

        metadata = extract_video_metadata(data)

        return {
            "width": metadata.width,
            "height": metadata.height,
            "framerate": metadata.fps,
            "num_frames": metadata.num_frames,
            "video_codec": metadata.video_codec,
            "num_bytes": data.nbytes,
        }

    @property
    def duration(self) -> float:
        """Calculate the duration of the clip.

        Returns:
            Duration of the clip in seconds.

        """
        return self.span[1] - self.span[0]

    def get_major_size(self) -> int:
        """Calculate total memory size of the clip.

        Returns:
            Total size in bytes.

        """
        total_size = len(self.uuid.bytes)
        total_size += self.encoded_data.nbytes
        total_size += self.extracted_frames.nbytes
        if self.decoded_motion_data is not None:
            total_size += self.decoded_motion_data.get_major_size()
        total_size += self.cosmos_embed1_frames.nbytes
        total_size += self.intern_video_2_frames.nbytes
        if self.intern_video_2_embedding is not None:
            total_size += self.intern_video_2_embedding.nbytes
        if self.cosmos_embed1_embedding is not None:
            total_size += self.cosmos_embed1_embedding.nbytes
        if self.openai_embedding is not None:
            total_size += self.openai_embedding.nbytes
        for window in self.windows:
            total_size += window.get_major_size()
        return total_size


@attrs.define
class ClipStats:
    """Statistics for video clips including filtering, transcoding, and captioning results.

    This class accumulates statistics about the number of clips processed through
    different stages of the video processing pipeline, including motion filtering,
    aesthetic filtering, and captioning.
    """

    num_filtered_by_motion: int = 0
    num_filtered_by_aesthetic: int = 0
    num_filtered_by_qwen_classifier: int = 0
    num_filtered_by_qwen_semantic: int = 0
    num_filtered_by_artificial_text: int = 0
    num_passed: int = 0
    num_transcoded: int = 0
    num_with_embeddings: int = 0
    num_with_caption: int = 0
    num_caption_windows: int = 0
    num_with_webp: int = 0
    total_clip_duration: float = 0.0
    max_clip_duration: float = 0.0
    total_prompt_tokens: int = 0
    total_output_tokens: int = 0
    caption_quality_stats: CaptionQualityStats | None = None

    def combine(self, other: Self) -> None:
        """Combine two ClipStats objects.

        Args:
            other: ClipStats object to combine with.

        """
        self.num_filtered_by_motion += other.num_filtered_by_motion
        self.num_filtered_by_aesthetic += other.num_filtered_by_aesthetic
        self.num_filtered_by_qwen_classifier += other.num_filtered_by_qwen_classifier
        self.num_filtered_by_qwen_semantic += other.num_filtered_by_qwen_semantic
        self.num_filtered_by_artificial_text += other.num_filtered_by_artificial_text
        self.num_passed += other.num_passed
        self.num_transcoded += other.num_transcoded
        self.num_with_embeddings += other.num_with_embeddings
        self.num_with_caption += other.num_with_caption
        self.num_caption_windows += other.num_caption_windows
        self.num_with_webp += other.num_with_webp
        self.total_clip_duration += other.total_clip_duration
        self.max_clip_duration = max(self.max_clip_duration, other.max_clip_duration)
        self.total_prompt_tokens += other.total_prompt_tokens
        self.total_output_tokens += other.total_output_tokens
        if other.caption_quality_stats is not None:
            if self.caption_quality_stats is None:
                self.caption_quality_stats = other.caption_quality_stats.copy()
            else:
                self.caption_quality_stats.combine(other.caption_quality_stats)


@attrs.define
class VideoMetadata:
    """Metadata for video content including dimensions, timing, and codec information.

    This class stores essential video properties such as resolution, frame rate,
    duration, and encoding details.
    """

    size: int | None = None
    height: int | None = None
    width: int | None = None
    framerate: float | None = None
    num_frames: int | None = None
    duration: float | None = None
    video_codec: str | None = None
    pixel_format: str | None = None
    audio_codec: str | None = None
    bit_rate_k: int | None = None
    format_name: str | None = None


@attrs.define
class Video:
    """Container for video content including metadata, frames, and processing results.

    This class stores information about a video segment, including its source, timing,
    extracted frames, motion data, aesthetic scores, and generated captions.
    """

    input_video: storage_client.StoragePrefix | pathlib.Path
    # Path relative to session/input; when non-empty, output clips preserve this structure under each clip UUID.
    relative_path: str = ""

    # encoded video data (numpy for zero-copy Ray transport via PEP 574 PickleBuffer)
    encoded_data: LazyData[npt.NDArray[np.uint8]] = attrs.field(factory=LazyData, converter=LazyData.coerce)  # type: ignore[misc]
    # video metadata
    metadata: VideoMetadata = attrs.Factory(VideoMetadata)
    # decoded video frames (numpy for zero-copy Ray transport via PEP 574 PickleBuffer)
    frame_array: LazyData[npt.NDArray[np.uint8]] = attrs.field(factory=LazyData, converter=LazyData.coerce)  # type: ignore[misc]
    # Per-frame PTS in seconds. If encoded_data is replaced, set to None and call populate_timestamps().
    timestamps: npt.NDArray[np.float32] | None = attrs.field(default=None, eq=False)
    # clips
    clips: list[Clip] = attrs.Factory(list)
    filtered_clips: list[Clip] = attrs.Factory(list)
    # for chunking clips that have one set of source videos across multiple tasks.
    num_total_clips: int = 0
    num_clip_chunks: int = 0
    clip_chunk_index: int = 0
    # for last writer stage
    clip_stats: ClipStats = attrs.Factory(ClipStats)
    # for debugging
    errors: dict[str, str] = attrs.Factory(dict)
    # True if VideoDownloader remuxed this video (mpegts → mp4). Set once per source
    # video, so filter on clip_chunk_index == 0 when aggregating to avoid double-counting
    # chunked outputs.
    was_remuxed: bool = False

    def populate_timestamps(self) -> None:
        """Extract and assign per-frame PTS timestamps from encoded_data.

        Raises:
            ValueError: If encoded_data is None.

        """
        data = self.encoded_data.resolve()
        if data is None:
            error_msg = "No video data available: encoded_data is None"
            raise ValueError(error_msg)
        self.timestamps = get_video_timestamps(data)

    def populate_metadata(self) -> None:
        """Extract and assign video metadata from encoded_data.

        This method extracts metadata from the video data in encoded_data and
        assigns it to self.metadata.

        Raises:
            ValueError: If encoded_data is None.
            Exception: Any exception from extract_video_metadata is propagated.

        """
        data = self.encoded_data.resolve()
        if data is None:
            error_msg = "No video data available: encoded_data is None"
            raise ValueError(error_msg)

        # Extract metadata using the existing function
        extracted_metadata = extract_video_metadata(data)

        # Set the size from encoded_data
        self.metadata.size = data.nbytes

        # Map the extracted metadata to our metadata object
        self.metadata.height = extracted_metadata.height
        self.metadata.width = extracted_metadata.width
        self.metadata.framerate = extracted_metadata.fps
        self.metadata.num_frames = extracted_metadata.num_frames
        self.metadata.duration = extracted_metadata.video_duration
        self.metadata.video_codec = extracted_metadata.video_codec
        self.metadata.pixel_format = extracted_metadata.pixel_format
        self.metadata.audio_codec = extracted_metadata.audio_codec
        self.metadata.bit_rate_k = extracted_metadata.bit_rate_k
        self.metadata.format_name = extracted_metadata.format_name

    @property
    def fraction(self) -> float:
        """Calculate the fraction of processed clips.

        Returns:
            Fraction of processed clips.

        """
        if self.num_total_clips == 0:
            return 1.0
        return (len(self.clips) + len(self.filtered_clips)) / self.num_total_clips

    @property
    def weight(self) -> float:
        """Calculate the weight of the video.

        Returns:
            Weight of the video.

        """
        if self.metadata.size is None:
            return 0
        # normalize to 5 min
        assert self.metadata.duration is not None
        weight = self.metadata.duration / 300
        # when clips are further chunked
        return weight * self.fraction

    @property
    def input_path(self) -> str:
        """Get the input path of the video.

        Returns:
            Input path of the video.

        """
        if isinstance(self.input_video, storage_client.StoragePrefix):
            return self.input_video.path
        return self.input_video.as_posix()

    def has_metadata(self) -> bool:
        """Check if all metadata fields are present.

        Returns:
            True if all metadata fields are present, False otherwise.

        """
        return all(
            [
                self.metadata.height,
                self.metadata.width,
                self.metadata.duration,
                self.metadata.framerate,
                self.metadata.num_frames,
                self.metadata.video_codec,
            ],
        )

    def nvdec_support(self) -> bool:
        """Heuristic function to switch between nvdec or CPU-fallback on V100/A100/H100.

        For detailed info on Video Codec SDK hardware support, see:
        https://developer.nvidia.com/video-encode-and-decode-gpu-support-matrix-new
        """
        if self.metadata.video_codec is None or self.metadata.pixel_format is None:
            return False
        if self.metadata.video_codec == "h264" and (
            "nv16" in self.metadata.pixel_format or "420p" in self.metadata.pixel_format
        ):
            # h264 decoding supported only 8-bit surface format
            return True
        if self.metadata.video_codec == "hevc" and (
            "420p" in self.metadata.pixel_format or "444p" in self.metadata.pixel_format
        ):
            # h265/hevc decoding supports yuv420p and yuv444p pixel formats
            return True
        if self.metadata.video_codec not in ("mjpeg", "av1", "vp9", "vp8"):
            # - mjpeg is not supported
            # - av1 is not supported on A100/H100
            # - VP8/9 are not exposed in PyNvVideoCodec (but supported by VideoCodecSDK)
            logger.warning(f"Encountered new video codec [{self.metadata.video_codec}], assuming no NVDEC support.")
        return False

    def is_10_bit_color(self) -> bool | None:
        """Heuristic function to determine if the input video has 10-bit color."""
        if self.metadata.pixel_format is None:
            return None
        return "10le" in self.metadata.pixel_format or "10be" in self.metadata.pixel_format

    def get_major_size(self) -> int:
        """Calculate total memory size of the video.

        Returns:
            Total size in bytes.

        """
        return get_major_size(self)


def check_clip_time_alignment(clips_per_video: list[list[Clip]]) -> list[int]:
    """Check if clips at the same index have identical spans across all videos.

    Arguments:
        clips_per_video: List of clip lists, one per video. All lists must have the same length.

    Returns:
        List of clip indices where spans are misaligned. Empty list if all aligned.

    Raises:
        ValueError: If videos have different numbers of clips.

    """
    if not clips_per_video:
        return []

    # All videos must have the same number of clips to check alignment
    clip_counts = [len(clips) for clips in clips_per_video]
    if not all(count == clip_counts[0] for count in clip_counts):
        msg = (
            f"Cannot check time alignment: videos have different clip counts {clip_counts}. "
            f"All videos must have the same number of clips."
        )
        raise ValueError(msg)

    if clip_counts[0] == 0:
        return []

    misaligned_indices = []
    num_clips = len(clips_per_video[0])

    for clip_idx in range(num_clips):
        spans = [clips[clip_idx].span for clips in clips_per_video]
        if not all(s == spans[0] for s in spans):
            misaligned_indices.append(clip_idx)

    return misaligned_indices


def assert_video_clip_alignment(videos: list[Video]) -> None:
    """Validate that all videos have synchronized clips.

    Validates time alignment across multiple videos by checking:
    1. All videos have processed the same number of clips
    2. Clips at the same index have identical time spans across all videos

    Arguments:
        videos: List of Video instances to validate.

    Raises:
        ValueError: If videos have misaligned processing or invalid state.

    """
    if not videos:
        return

    # Check 1: All videos should have processed the same number of clips
    processed_per_video = [len(v.clips) + len(v.filtered_clips) for v in videos]
    if not all(p == processed_per_video[0] for p in processed_per_video):
        msg = (
            f"Multi-cam videos have processed different numbers of clips: {processed_per_video}. "
            f"All cameras should process clips together to maintain time alignment."
        )
        raise ValueError(msg)

    # Check 2: All clips at the same index must have identical spans (time alignment)
    # Check processed clips
    clips_per_video = [v.clips for v in videos]
    misaligned_clip_indices = check_clip_time_alignment(clips_per_video)
    if misaligned_clip_indices:
        # Get spans for the first misaligned index to show in error
        idx = misaligned_clip_indices[0]
        spans = [v.clips[idx].span for v in videos]
        msg = (
            f"Multi-cam clips at index {idx} have misaligned spans: {spans}. "
            f"All cameras must process the same time spans. "
            f"Misaligned indices: {misaligned_clip_indices}"
        )
        raise ValueError(msg)

    # Check filtered clips
    filtered_clips_per_video = [v.filtered_clips for v in videos]
    misaligned_filtered_indices = check_clip_time_alignment(filtered_clips_per_video)
    if misaligned_filtered_indices:
        # Get spans for the first misaligned index to show in error
        idx = misaligned_filtered_indices[0]
        spans = [v.filtered_clips[idx].span for v in videos]
        msg = (
            f"Multi-cam filtered clips at index {idx} have misaligned spans: {spans}. "
            f"All cameras must filter the same time spans. "
            f"Misaligned indices: {misaligned_filtered_indices}"
        )
        raise ValueError(msg)


@attrs.define
class SplitPipeTask(PipelineTask):
    """The data we want to pass between stages of split-pipeline.

    Attributes:
        session_id: multi-cam session id for multi-camera tasks and the
            video path for single-camera tasks.
        videos: The list of videos in the task.
        stage_perf: The performance statistics for the task.
        video: provides single-camera support by returning `videos[0]` (the primary camera).
        errors: For tracking errors at the task level.

    """

    session_id: str
    videos: list[Video] = attrs.field(factory=list)
    stage_perf: dict[str, StagePerfStats] = attrs.Factory(dict)
    errors: dict[str, str] = attrs.Factory(dict)

    # Hidden field for single-camera support
    _init_video: Video | None = attrs.field(default=None, init=True, alias="video")

    def __attrs_post_init__(self) -> None:
        """Handle backward-compatible initialization after attrs initialization.

        This allows initialization with either `video=` or `videos=` parameter.
        """
        # If single video was provided via video= parameter, convert to list
        if self._init_video is not None:
            if self.videos:
                msg = "Cannot specify both 'video' and 'videos' parameters"
                raise ValueError(msg)
            object.__setattr__(self, "videos", [self._init_video])

        # Validate that we have at least one video
        if not self.videos:
            msg = "Must specify either 'video' or 'videos' parameter"
            raise ValueError(msg)

    @property
    def video(self) -> Video:
        """Get the primary video (first video in the list).

        This property provides single camera compatibility for code that accesses `task.video`.
        For multi-camera tasks, this returns the primary camera at index 0.

        Returns:
            The primary video.

        Raises:
            IndexError: If videos list is empty.

        """
        return self.videos[0]

    @property
    def fraction(self) -> float:
        """Calculate fraction of processed video in the task.

        Sums all processed and filtered clips across all videos and divides by total clips.

        Returns:
            Fraction of processed video (0.0 to 1.0).

        """
        total_processed = sum(len(v.clips) + len(v.filtered_clips) for v in self.videos)
        total_clips = sum(v.num_total_clips for v in self.videos)
        if total_clips == 0:
            return 1.0
        return total_processed / total_clips

    def assert_time_alignment(self) -> None:
        """Validate that all cameras have synchronized processing.

        Multi-camera stages should call this method at the end of process_data()
        to ensure clips remain time-aligned across all cameras.

        Validates:
            1. All videos have the same total number of clips
            2. All videos have processed the same number of clips
            3. Processed clips do not exceed total clips
            4. Clips at the same index have identical spans across all videos

        Raises:
            ValueError: If cameras have misaligned processing or invalid state.

        """
        assert_video_clip_alignment(self.videos)

    @property
    def weight(self) -> float:
        """Calculate weight of video in the task.

        For single-camera tasks, returns the primary video's weight.
        For multi-camera tasks, sums weights across all videos since all must be processed.

        Returns:
            Weight of video(s).

        """
        return sum(v.weight for v in self.videos)

    def get_major_size(self) -> int:
        """Calculate memory size of video(s) in the task.

        Sums the memory size across all videos in the task.

        Returns:
            Total size in bytes.

        """
        return sum(v.get_major_size() for v in self.videos)


@attrs.define
class ClipSample:
    """Container for video clip sample data including metadata, frames, and embeddings.

    This class stores information about a video clip sample, including its UUID, dimensions,
    frame count, frame rate, byte size, and metadata.
    """

    uuid: str
    width: int
    height: int
    num_frames: int
    framerate: float
    num_bytes: int
    clip_location: storage_client.StoragePrefix | pathlib.Path
    clip_metadata: dict[str, Any] = attrs.Factory(dict)
    encoded_data: LazyData[npt.NDArray[np.uint8]] = attrs.field(factory=LazyData, converter=LazyData.coerce)  # type: ignore[misc]
    t5_xxl_embeddings: list[npt.NDArray[np.int32]] = attrs.Factory(list)

    def get_major_size(self) -> int:
        """Calculate total memory size of the clip sample.

        Returns:
            Total size in bytes.

        """
        total_size = sys.getsizeof(self.clip_metadata)
        total_size += self.encoded_data.nbytes
        total_size += sum(x.nbytes for x in self.t5_xxl_embeddings)
        return total_size


@attrs.define
class ShardPipeTask(PipelineTask):
    """The data we want to pass between stages of sharding-pipeline."""

    bin_path: str
    part_num: int
    samples: list[ClipSample]
    output_tar_video: storage_client.StoragePrefix | pathlib.Path
    output_tar_metas: storage_client.StoragePrefix | pathlib.Path
    output_tar_t5_xxl: storage_client.StoragePrefix | pathlib.Path
    key_count: int
    stage_perf: dict[str, StagePerfStats] = attrs.Factory(dict)

    def get_major_size(self) -> int:
        """Calculate total memory size of all samples in the task.

        Returns:
            Total size in bytes.

        """
        total_size = 0
        for sample in self.samples:
            total_size += sample.get_major_size()
        return total_size


def assert_time_alignment(tasks: list[SplitPipeTask]) -> None:
    """Validate time alignment for a batch of multi-camera tasks.

    Convenience function for stages to validate all tasks in one call.
    Calls task.assert_time_alignment() on each task.

    Arguments:
        tasks: List of SplitPipeTask instances to validate.

    Raises:
        ValueError: If any task has misaligned cameras.

    """
    for task in tasks:
        task.assert_time_alignment()


def get_video_from_task(task: PipelineTask) -> Video:
    """Extract the Video object attached to a pipeline task.

    Args:
        task: Task expected to expose a `video` attribute.

    Returns:
        The associated `Video` instance.

    Raises:
        TypeError: If the task is missing a `video` attribute or it has an unexpected type.

    """
    video = getattr(task, "video", None)
    if not isinstance(video, Video):
        msg = f"task.video type={type(video)}, expected `Video`"
        raise TypeError(msg)
    return video


@attrs.define
class VllmSamplingConfig:
    """Configuration for vLLM sampling parameters.

    Unless otherwise specified, the vLLM default values are used.
    The defaults differ to maintain compatibility with the previous configuration.

    Args:
        presence_penalty: Penalize tokens based on their presence in the generated text.
        frequency_penalty: Penalize tokens based on their frequency in the generated text.
        repetition_penalty: Penalize tokens that have been generated previously.
        temperature: Controls randomness in sampling (higher = more random).
        top_p: Nucleus sampling threshold.
        top_k: Top-k sampling parameter (0 = disabled).
        min_p: Minimum probability threshold for sampling.
        min_tokens: Minimum tokens before EOS is allowed (0 = disabled, 16 = default for fp8 safety).
        max_tokens: Maximum number of tokens to generate (None = no limit).

    """

    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    repetition_penalty: float = 1.05  # vLLM default is 1.0
    temperature: float = 0.1  # vLLM default is 1.0
    top_p: float = 0.001  # vLLM default is 1.0
    top_k: int = 0
    min_p: float = 0.0
    min_tokens: int = attrs.field(default=16, validator=attrs.validators.ge(0))
    max_tokens: int | None = 8192  # vLLM default is None


@attrs.define
class VllmConfig:
    """Configuration for a vLLM model.

    Args:
        model_variant: Name of the model variant to use.
        prompt_variant: Type of prompt to use.
        prompt_text: Custom prompt text if provided.
        batch_size: Number of samples to process in parallel.
        fp8: Whether to enable FP8 precision.
        preprocess: Whether model handles preprocessing.
        disable_mmcache: Whether to disable model cache.
        num_gpus_per_worker: Number of GPUs to allocate per worker.
        batch_size: Number of samples to process in parallel.
        stage2_caption: Whether to enable stage 2 captioning.
        stage2_prompt_text: Custom prompt text for stage 2 captioning.
        max_retries: Number of times to retry captioning failures.
        copy_weights_to: Optional custom directory to copy model weights to before loading.
            If set, model weights will be copied from the default cache location to this
            directory before the model is loaded. This is useful for copying weights to
            faster storage (e.g., local SSD) on compute nodes.
        sampling_config: Configuration for sampling parameters.
        performance_mode: vLLM performance mode. "throughput" favors aggregate tokens/sec
            at high concurrency; "interactivity" favors low per-request latency; "balanced"
            is the vLLM default. Set to None to use the vLLM default.
        debug_save_frames: Whether to save video frames as PNGs for debugging.
        debug_frames_output_dir: Directory to save debug frame PNGs. Required if debug_save_frames is True.
        use_image_input: When True, use the image modality only: content type "image",
            multi_modal_data["image"], and limit_mm_per_prompt for image (video slot 0).
            Used by the image pipeline; video pipeline leaves this False.

    """

    model_variant: str
    use_image_input: bool = False
    prompt_variant: str = "default"
    prompt_text: str | None = None
    fp8: bool = False
    preprocess: bool = False
    disable_mmcache: bool = False
    num_cpus_for_prepare: float = 2.0
    num_gpus: int = 1
    batch_size: int = 4
    stage2_caption: bool = False
    stage2_prompt_text: str | None = None
    max_retries: int = 3
    copy_weights_to: pathlib.Path | None = None
    sampling_config: VllmSamplingConfig = attrs.Factory(VllmSamplingConfig)
    performance_mode: Literal["balanced", "interactivity", "throughput"] | None = "throughput"
    debug_save_frames: bool = False
    debug_frames_output_dir: pathlib.Path | None = None


def validate_optional_json_str_str_dict(
    _: object,
    attribute: "attrs.Attribute[str]",
    value: str,
) -> None:
    """Reject string fields that, when non-empty, are not a JSON ``dict[str, str]``.

    The empty string is treated as "field not set" and skips validation entirely,
    so callers can leave the field at its default without paying a JSON parse.
    """
    if not value:
        return
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        msg = f"{attribute.name} must be valid JSON, got {value!r}"
        raise ValueError(msg) from exc
    if not isinstance(parsed, dict):
        msg = f"{attribute.name} must be a JSON object (dict), got {type(parsed).__name__}"
        raise TypeError(msg)
    for k, v in parsed.items():
        if not isinstance(v, str):
            msg = f"{attribute.name} values must be strings, got key={k!r}, value={v!r} ({type(v).__name__})"
            raise TypeError(msg)


@attrs.define(frozen=True)
class VllmAsyncConfig:
    """Configuration for the in-process ``AsyncLLM`` engine.

    Mirrors sync's ``VllmConfig`` shape where overlapping (``fp8``,
    ``disable_mmcache``, ``preprocess``, ``max_retries``).  Use
    :meth:`to_vllm_config` when calling plugin methods that take a sync
    ``VllmConfig`` (``model_path``, ``processor``, ``make_llm_input``).
    Note: ``model_async`` takes ``VllmAsyncConfig`` directly (no
    conversion needed) and reads async-only fields such as
    ``async_scheduling`` and ``distributed_executor_backend``.

    String-typed engine knobs (``distributed_executor_backend``, ``kv_cache_dtype``,
    ``mm_encoder_tp_mode``, ``mm_processor_cache_type``) intentionally use plain
    ``str``: vLLM is the source of truth for accepted values and rejects
    unknown ones at engine init.  Cosmos-curator deliberately does NOT
    duplicate that validation, so users can pass new vLLM-supported values
    without waiting for a release.
    """

    model_variant: str
    num_gpus: int = attrs.field(
        default=1,
        validator=[attrs.validators.instance_of(int), attrs.validators.ge(1)],
    )
    data_parallel_size: int = attrs.field(
        default=1,
        validator=[attrs.validators.instance_of(int), attrs.validators.ge(1)],
    )
    fp8: bool = False
    disable_mmcache: bool = False
    enforce_eager: bool = False
    disable_log_stats: bool = True
    enable_log_requests: bool = False
    disable_chunked_mm_input: bool = False
    skip_mm_profiling: bool = True
    stream_interval: int = 9999

    # When False (default), CPU prep stage owns the deterministic resize and
    # tokenization and vLLM's processor only does rescale + normalize; this
    # mirrors sync's VllmConfig.preprocess=False (vLLM's mm_processor_kwargs
    # set do_resize/do_rescale/do_normalize to False).  Set True to let
    # vLLM's processor re-run resize + rescale + normalize on top of the
    # CPU-prepared frames (rarely needed; matches sync's behaviour).
    preprocess: bool = False

    # Per-request retry budget around engine.generate (mirrors sync's
    # VllmConfig.max_retries).  EngineDeadError is never retried - it
    # escalates so Xenna restarts the actor.
    max_retries: int = attrs.field(default=3, validator=attrs.validators.ge(1))

    max_num_seqs: int = 0
    long_prefill_token_threshold: int = 0
    mm_processor_cache_type: str = ""
    mm_encoder_tp_mode: str = "data"
    extra_env_vars: str = attrs.field(default="", validator=validate_optional_json_str_str_dict)

    distributed_executor_backend: str = "ray"
    kv_cache_dtype: str = "auto"

    gpu_memory_utilization: float | None = attrs.field(
        default=None,
        validator=attrs.validators.optional(
            attrs.validators.and_(
                attrs.validators.gt(0.0),
                attrs.validators.le(1.0),
            ),
        ),
    )

    async_scheduling: bool | None = None
    enable_chunked_prefill: bool | None = None
    sampling_config: VllmSamplingConfig = attrs.Factory(VllmSamplingConfig)

    @property
    def total_gpus(self) -> int:
        """Total GPU footprint across data-parallel replicas: ``num_gpus * data_parallel_size``."""
        return self.num_gpus * self.data_parallel_size

    def to_vllm_config(self) -> "VllmConfig":
        """Translate to ``VllmConfig`` for plugin methods that require it."""
        return VllmConfig(
            model_variant=self.model_variant,
            use_image_input=False,
            copy_weights_to=None,
            fp8=self.fp8,
            disable_mmcache=self.disable_mmcache,
            preprocess=self.preprocess,
            max_retries=self.max_retries,
        )


@attrs.define
class WindowConfig:
    """Configuration for splitting a video into windows.

    Args:
        sampling_fps: Frames per second for sampling.
        window_size: Size of each window in frames.
        remainder_threshold: Minimum frames required for a remainder window.
        preprocess_dtype: Data type for preprocessing.
        model_does_preprocess: Whether model handles preprocessing.
        use_input_bit_rate: Whether to use the input video's bit rate for processing.

    """

    window_size: int = 256
    sampling_fps: float = 2.0
    remainder_threshold: int = 128
    model_does_preprocess: bool = True
    preprocess_dtype: str = "float32"
    use_input_bit_rate: bool = False


@attrs.define
class VllmCaptionRequest:
    """A vLLM captioning task for a single clip window.

    Args:
        request_id: The request ID.
        inputs: The inputs for the vLLM model.
        caption: The caption generated by the vLLM model.
           * If caption is None, this indicates that the caption should be generated
             by the vLLM model using the inputs
        stage2_prompt: A second-stage prompt used to refine the caption
           * If stage2_prompt is set, and caption is not None, this indicates
             that the caption should be refined using the stage2_prompt.
             A new request should be created with new inputs and with stage2_prompt set to None.
           * If stage2_prompt is not set, and caption is not None, this indicates
             that the caption should be used as is.

    """

    request_id: str
    inputs: dict[str, Any]
    caption: str | None = None
    stage2_prompt: str | None = None
    prompt_tokens: int = 0
    output_tokens: int = 0
    finish_reason: str | None = None
