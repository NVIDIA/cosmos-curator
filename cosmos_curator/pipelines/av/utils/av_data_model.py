# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
"""Data model for AV pipelines."""

import sys
from typing import Any
from uuid import UUID

import attrs
import numpy as np
import numpy.typing as npt
from loguru import logger

from cosmos_curator.core.interfaces.stage_interface import PipelineTask
from cosmos_curator.core.utils.data.lazy_data import LazyData
from cosmos_curator.core.utils.infra.performance_utils import StagePerfStats
from cosmos_curator.pipelines.video.utils.decoder_utils import extract_video_metadata


@attrs.define
class VideoMetadata:
    """Metadata for a video."""

    size: int | None = None
    height: int | None = None
    width: int | None = None
    framerate: float | None = None
    num_frames: int | None = None
    duration: float | None = None
    video_codec: str | None = None
    pixel_format: str | None = None


@attrs.define
class ClipForTranscode:
    """A clip for transcode."""

    uuid: UUID
    clip_session_uuid: UUID
    span_index: int
    span_start: float
    span_end: float
    timestamps_ms: npt.NDArray[np.int64] | None = None
    encoded_data: LazyData[npt.NDArray[np.uint8]] = attrs.field(factory=LazyData, converter=LazyData.coerce)  # type: ignore[misc]
    url: str | None = None

    def get_major_size(self) -> int:
        """Get the major size of the clip."""
        total_size = self.timestamps_ms.nbytes if self.timestamps_ms is not None else 0
        total_size += self.encoded_data.nbytes
        return total_size


@attrs.define
class AvVideo:
    """A video."""

    source_video: str
    camera_id: int
    encoded_data: LazyData[npt.NDArray[np.uint8]] = attrs.field(factory=LazyData, converter=LazyData.coerce)  # type: ignore[misc]
    metadata: VideoMetadata = attrs.field(factory=VideoMetadata)
    timestamps_ms: npt.NDArray[np.int64] | None = None
    clips: list[ClipForTranscode] = attrs.field(factory=list)

    def populate_metadata(self) -> None:
        """Extract and assign video metadata from encoded_data.

        This method extracts metadata from the video data in encoded_data.

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

    def has_metadata(self) -> bool:
        """Check if the video has metadata."""
        return (
            all(
                [
                    self.metadata.size,
                    self.metadata.height,
                    self.metadata.width,
                    self.metadata.framerate,
                    self.metadata.num_frames,
                    self.metadata.duration,
                ]
            )
            and self.metadata.framerate is not None
            and self.metadata.framerate > 0
        )

    def is_10_bit_color(self) -> bool | None:
        """Heuristic function to determine if the input video has 10-bit color."""
        if self.metadata.pixel_format is None:
            return None
        return "10le" in self.metadata.pixel_format or "10be" in self.metadata.pixel_format

    def get_major_size(self) -> int:
        """Get the major size of the video."""
        total_size = self.encoded_data.nbytes
        total_size += self.timestamps_ms.nbytes if self.timestamps_ms is not None else 0
        for clip in self.clips:
            total_size += clip.get_major_size()
        return total_size


@attrs.define
class AvSessionVideoSplitTask(PipelineTask):
    """A task for splitting a video session."""

    source_video_session_name: str
    source_video_version: str
    session_uuid: UUID
    session_url: str
    num_cameras: int | None = None
    videos: list[AvVideo] = attrs.field(factory=list)
    split_algo_name: str | None = None
    encoder: str | None = None
    stage_perf: dict[str, StagePerfStats] = attrs.Factory(dict)

    @property
    def source_video_duration_s(self) -> float:
        """Get the duration of the source video.

        Returns:
            The duration of the source video.

        """
        return sum(x.metadata.duration or 0 for x in self.videos)

    def get_major_size(self) -> int:
        """Get the major size of the session video split task.

        Returns:
            The major size of the session video split task.

        """
        return sum(x.get_major_size() for x in self.videos)


@attrs.define
class AvSessionVideoIngestTask(PipelineTask):
    """A task for ingesting a video session."""

    sessions: list[AvSessionVideoSplitTask] = attrs.field(factory=list)

    def get_major_size(self) -> int:
        """Get the major size of the session video ingest task.

        Returns:
            The major size of the session video ingest task.

        """
        return sum(x.get_major_size() for x in self.sessions)


def _append[T](key: str, value: T, container: dict[str, list[T]]) -> None:
    if key not in container:
        container[key] = []
    container[key].append(value)


def _get_last[T](key: str, container: dict[str, list[T]]) -> T:
    return container[key][-1]


@attrs.define
class CaptionWindow:
    """A caption window."""

    start_frame: int
    end_frame: int
    model_input: dict[str, dict[str, Any]] = attrs.field(factory=dict)
    # caption: prompt_variant -> list of captions in the order they are generated,
    # last in list is the most important.
    captions: dict[str, list[str]] = attrs.field(factory=dict)
    t5_xxl_embeddings: dict[str, npt.NDArray[np.float32]] = attrs.field(factory=dict)

    def get_major_size(self) -> int:
        """Get the major size of the caption window.

        Returns:
            The major size of the caption window.

        """
        total_size = 0

        for prompt_variant, captions in self.captions.items():
            total_size += sys.getsizeof(prompt_variant)
            total_size += sum(sys.getsizeof(caption) for caption in captions)

        for prompt_variant, model_input in self.model_input.items():
            total_size += sys.getsizeof(prompt_variant)
            total_size += sys.getsizeof(model_input)

        for prompt_variant, t5_xxl_embedding in self.t5_xxl_embeddings.items():
            total_size += sys.getsizeof(prompt_variant)
            total_size += t5_xxl_embedding.nbytes

        return total_size

    def append_caption(self, prompt_variant: str, caption: str) -> None:
        """Append a caption to the caption window.

        Args:
            prompt_variant: The prompt variant to append the caption to.
            caption: The caption to append.

        """
        _append(prompt_variant, caption, self.captions)

    def get_last_caption(self, prompt_variant: str) -> str:
        """Get the last caption from the caption window.

        Args:
            prompt_variant: The prompt variant to get the last caption from.

        Returns:
            The last caption from the caption window.

        """
        return _get_last(prompt_variant, self.captions)

    def to_dict(
        self,
        last_caption_only: bool = False,  # noqa: FBT001, FBT002
        attr_white_list: list[str] | None = None,
        use_formatted_vri_tags: bool = False,  # noqa: FBT001, FBT002
    ) -> dict[str, Any]:
        """Convert the caption window to a dictionary.

        Filter out model_input and t5_xxl_embedding.

        Args:
            last_caption_only: If True, only include the last caption for each
                prompt variant. If False, include all captions.
            attr_white_list: A list of attribute names to include in the dictionary.
                If None, include start_frame, end_frame, and captions.
            use_formatted_vri_tags: If True, use formatted VRI tags.

        Returns:
            A dictionary representation of the caption window.

        """
        _attr_white_list = {"start_frame", "end_frame", "captions"} if attr_white_list is None else set(attr_white_list)

        data = attrs.asdict(self, filter=lambda attr, _: attr.name in _attr_white_list)

        if last_caption_only:
            _prompt_variant_blacklist = {"vri"} if use_formatted_vri_tags else set()
            data["captions"] = {
                prompt_variant: [captions[-1]] if captions else []
                for prompt_variant, captions in self.captions.items()
                if prompt_variant not in _prompt_variant_blacklist
            }
        return data


@attrs.define
class ClipForAnnotation:
    """A clip for annotation."""

    video_session_name: str
    clip_session_uuid: UUID
    uuid: UUID
    camera_id: int
    span_index: int
    url: str
    # encoded video data (numpy for zero-copy Ray transport via PEP 574 PickleBuffer)
    encoded_data: LazyData[npt.NDArray[np.uint8]] = attrs.field(factory=LazyData, converter=LazyData.coerce)  # type: ignore[misc]
    # for captioning
    caption_windows: list[CaptionWindow] = attrs.field(factory=list)
    t5_xxl_embedding_urls: dict[str, str] = attrs.field(factory=dict)
    # for continued captioning from split pipeline
    span_start: float | None = None
    span_end: float | None = None
    # for tags
    vri_tags: dict[str, str] = attrs.field(factory=dict)
    # for error tracking
    errors: dict[str, str] = attrs.field(factory=dict)

    def get_major_size(self) -> int:
        """Get the major size of the clip.

        Returns:
            The major size of the clip.

        """
        total_size = self.encoded_data.nbytes
        total_size += sum(x.get_major_size() for x in self.caption_windows)
        return total_size

    def get_vri_caption_text(self) -> str:
        """Get the VRI caption text from the clip.

        Returns:
            The VRI caption text from the clip.

        """
        for window in self.caption_windows:
            if "vri" in window.captions and len(window.captions["vri"]) > 0:
                return window.captions["vri"][-1]

        error = f"No vri caption text found for clip {self.uuid}"
        raise ValueError(error)

    def to_dict(
        self,
        last_caption_only: bool = False,  # noqa: FBT001, FBT002
        attr_white_list: list[str] | None = None,
        use_formatted_vri_tags: bool = False,  # noqa: FBT001, FBT002
    ) -> dict[str, Any]:
        """Convert the clip to a dictionary.

        Args:
            last_caption_only: If True, only include the last caption in each
                caption window. If False, include all captions.
            attr_white_list: A list of attribute names to include in the dictionary.
                If None, include all attributes except buffer.
            use_formatted_vri_tags: If True, use formatted VRI tags.

        Returns:
            A dictionary containing the clip's data, excluding attributes not
            in the white list.

        """
        if attr_white_list is None:
            _attr_white_list = {
                "video_session_name",
                "clip_session_uuid",
                "uuid",
                "camera_id",
                "span_start",
                "span_end",
                "url",
                "caption_windows",
                "t5_xxl_embedding_urls",
            }
        else:
            _attr_white_list = set(attr_white_list)

        if use_formatted_vri_tags:
            _attr_white_list.add("vri_tags")

        data = attrs.asdict(self, filter=lambda attr, _: attr.name in _attr_white_list)

        if "caption_windows" in _attr_white_list:
            data["caption_windows"] = [
                window.to_dict(
                    last_caption_only=last_caption_only,
                    use_formatted_vri_tags=use_formatted_vri_tags,
                )
                for window in self.caption_windows
            ]
        return data


@attrs.define
class AvClipAnnotationTask(PipelineTask):
    """A task for annotating a clip."""

    clips: list[ClipForAnnotation] = attrs.field(factory=list)
    video_session_name: str | None = None
    num_session_chunks: int | None = None
    session_chunk_index: int | None = None
    stage_perf: dict[str, StagePerfStats] = attrs.Factory(dict)
    # for continued captioning from split pipeline
    source_video_duration_s: float | None = None
    height: int | None = None
    width: int | None = None
    framerate: float | None = None

    @property
    def fraction(self) -> float:
        """Get the fraction of the session.

        Returns:
            The fraction of the session.

        """
        if self.num_session_chunks is None:
            return 1.0
        return 1.0 / self.num_session_chunks

    @property
    def weight(self) -> float:
        """Get the weight of the task.

        Returns:
            The weight of the task.

        """
        return 1.0 * self.fraction

    def to_dict(
        self,
        video_level: bool = False,  # noqa: FBT001, FBT002
        attr_white_list: list[str] | None = None,
    ) -> dict[str, Any]:
        """Convert the task to a dictionary.

        Args:
            attr_white_list: A list of attribute names to include in the dictionary.
                If None, include all attributes.
            video_level: If True, include the video level attributes.

        Returns:
            A dictionary representation of the task.

        """
        if attr_white_list is None:
            _attr_white_list = {
                "video_session_name",
                "num_session_chunks",
                "session_chunk_index",
                "source_video_duration_s",
                "height",
                "width",
                "frame_rate",
            }
        else:
            _attr_white_list = set(attr_white_list)

        data = attrs.asdict(self, filter=lambda attr, _: attr.name in _attr_white_list)

        if not video_level:
            data["clips"] = [str(clip.uuid) for clip in self.clips]

        return data

    def get_major_size(self) -> int:
        """Get the major size of the task.

        Returns:
            The major size of the task.

        """
        return sum(x.get_major_size() for x in self.clips)


@attrs.define
class ClipForTrajectory:
    """A clip for trajectory."""

    uuid: UUID
    camera_id: int
    span_index: int
    timestamps_ms: bytes | None = None
    # for trajectory
    trajectory: npt.NDArray[np.float32] | None = None
    trajectory_url: str | None = None

    def get_major_size(self) -> int:
        """Get the major size of the clip.

        Returns:
            The major size of the clip.

        """
        return 0 if self.timestamps_ms is None else len(self.timestamps_ms)


@attrs.define
class AvSessionTrajectoryTask(PipelineTask):
    """A task for trajectory."""

    session_url: str
    clips: list[ClipForTrajectory] = attrs.field(factory=list)
    sqlite_db: bytes | None = None
    stage_perf: dict[str, StagePerfStats] = attrs.Factory(dict)

    def get_major_size(self) -> int:
        """Get the major size of the task.

        Returns:
            The major size of the task.

        """
        return sum(x.get_major_size() for x in self.clips)


@attrs.define
class AvSample:
    """A sample."""

    clip_session_uuid: UUID
    camera_ids: list[int]
    clip_uuids: list[UUID]
    clip_urls: list[str]
    clip_timestampss_ms: list[bytes]
    window_captions: list[str]
    window_start_frames: list[int]
    window_end_frames: list[int]
    t5_urls: list[str]
    trajectory_urls: list[str] | None = None

    def get_major_size(self) -> int:
        """Get the major size of the sample.

        Returns:
            The major size of the sample.

        """
        total_size = sys.getsizeof(self.clip_session_uuid)
        total_size += sys.getsizeof(self.camera_ids)
        total_size += sys.getsizeof(self.clip_urls)
        total_size += sys.getsizeof(self.clip_timestampss_ms)
        total_size += sys.getsizeof(self.window_captions)
        total_size += sys.getsizeof(self.t5_urls)
        total_size += 0 if self.trajectory_urls is None else sys.getsizeof(self.trajectory_urls)
        return total_size


@attrs.define
class AvShardingTask(PipelineTask):
    """A task for sharding."""

    part_num: int
    samples: list[AvSample] = attrs.field(factory=list)
    tar_mappings: dict[int, dict[str, str]] = attrs.field(factory=dict)
    s3_upload_error: bool = False
    source_data_error: bool = False
    stage_perf: dict[str, StagePerfStats] = attrs.Factory(dict)

    def get_major_size(self) -> int:
        """Get the major size of the task.

        Returns:
            The major size of the task.

        """
        return sum(x.get_major_size() for x in self.samples)


def get_clip_window_mappings(
    clips: list[ClipForAnnotation],
    prompt_variant: str,
    skip_missing: str,
    verbose: bool = False,  # noqa: FBT001, FBT002
) -> list[tuple[int, int]]:
    """Map list indexes to (clip, window) indexes for clips with captions or model input.

    Clips without these are logged as warnings.

    Args:
        clips: List of clips containing caption windows
        prompt_variant: The prompt variant to check for captions or model inputs
        skip_missing: Must be one of "captions" or "model_input". Whether to skip clips without captions or model inputs
        verbose: If True, log warnings for clips without captions or model inputs

    Returns:
        List of (clip_idx, window_idx) tuples

    """
    if skip_missing not in ["captions", "model_input"]:
        error = f"skip_missing must be one of 'captions' or 'model_input', got {skip_missing}"
        raise ValueError(error)

    mappings: list[tuple[int, int]] = []
    for clip_idx, clip in enumerate(clips):
        for window_idx, window in enumerate(clip.caption_windows):
            if skip_missing == "captions" and not window.captions.get(prompt_variant):
                if verbose:
                    logger.error(f"Clip {clip.uuid} window-{window_idx} has no captions.")
                continue
            if skip_missing == "model_input" and not window.model_input.get(prompt_variant):
                if verbose:
                    logger.error(
                        f"Clip {clip.uuid} window-{window_idx} has no prepared inputs "
                        f"for prompt_variant={prompt_variant}."
                    )
                continue
            mappings.append((clip_idx, window_idx))
    return mappings


def append_captions_to_clips(
    clips: list[ClipForAnnotation],
    prompt_variant: str,
    captions: list[str],
    mappings: list[tuple[int, int]],
) -> None:
    """Append captions to the caption chain of each window in the task.

    This function processes each caption and appends it to the caption chain of the
    corresponding window in the task. It also logs the caption if the verbose flag is set.

    Args:
        clips: List of clips containing caption windows
        prompt_variant: The prompt variant to append the captions to
        captions: List of captions to append
        mappings: List of (clip_idx, window_idx) tuples for windows with captions

    Raises:
        ValueError: If the number of mappings and captions do not match

    """
    if len(mappings) != len(captions):
        error = f"Number of mappings ({len(mappings)}) does not match number of captions ({len(captions)})"
        raise ValueError(error)

    for (clip_idx, window_idx), caption in zip(mappings, captions, strict=True):
        caption_window = clips[clip_idx].caption_windows[window_idx]
        caption_window.append_caption(prompt_variant, caption)


def get_last_captions(
    clips: list[ClipForAnnotation], prompt_variant: str, mappings: list[tuple[int, int]]
) -> list[str]:
    """Extract the last caption from each window specified in the mappings.

    Args:
        clips: List of clips containing caption windows
        prompt_variant: The prompt variant to get last captions of
        mappings: List of (clip_idx, window_idx) tuples for windows with captions

    Returns:
        List of the last caption from each specified window

    """
    last_captions: list[str] = []
    for clip_idx, window_idx in mappings:
        caption_window = clips[clip_idx].caption_windows[window_idx]
        last_caption = caption_window.get_last_caption(prompt_variant)
        last_captions.append(last_caption)
    return last_captions
