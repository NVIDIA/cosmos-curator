# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
"""Module for writing cosmos-predict2 compatible datasets.

This module provides functionality for generating cosmos-predict2 compatible
dataset structures from AV captioning pipeline outputs, organizing clips by
camera view and generating the required directory structure and file formats.
"""

import pathlib
import pickle
import uuid
from typing import Any

import numpy as np
import ray
from loguru import logger

from cosmos_curator.core.interfaces.stage_interface import CuratorStage, CuratorStageResource
from cosmos_curator.core.utils.infra.performance_utils import StageTimer
from cosmos_curator.core.utils.pixi_runtime_envs import PixiRuntimeEnv
from cosmos_curator.core.utils.storage import s3_client
from cosmos_curator.core.utils.storage.s3_client import is_s3path
from cosmos_curator.core.utils.storage.writer_utils import write_bytes, write_text
from cosmos_curator.models import t5_encoder
from cosmos_curator.pipelines.av.captioning.captioning_stages import _get_prompt
from cosmos_curator.pipelines.av.utils.av_data_info import CAMERA_MAPPING
from cosmos_curator.pipelines.av.utils.av_data_model import AvClipAnnotationTask, ClipForAnnotation

# Camera ID to cosmos-predict2 view name mapping
COSMOS_PREDICT2_CAMERA_MAPPING = {
    2: "pinhole_front",  # camera_front_wide_120fov
    5: "pinhole_front_left",  # camera_cross_left_120fov
    7: "pinhole_front_right",  # camera_cross_right_120fov
    4: "pinhole_side_left",  # camera_rear_left_70fov
    8: "pinhole_side_right",  # camera_rear_right_70fov
}


def _get_cosmos_predict2_file_url(  # noqa: PLR0913
    output_prefix: str,
    dataset_name: str,
    camera_view: str,
    clip_uuid: uuid.UUID,
    file_type: str,
    extension: str,
) -> s3_client.S3Prefix | pathlib.Path:
    """Generate a URL or path for storing cosmos-predict2 dataset files.

    Args:
        output_prefix: Base path for output
        dataset_name: Name of the dataset
        camera_view: Camera view name (e.g., "pinhole_front")
        clip_uuid: UUID of the clip
        file_type: Type of file ("metas", "videos", "t5_xxl")
        extension: File extension (e.g., "txt", "mp4", "pkl")

    Returns:
        An S3Prefix object if the output_prefix is an S3 path, otherwise a
        pathlib.Path object

    """
    full_path = f"{output_prefix}/datasets/{dataset_name}/{file_type}/{camera_view}/{clip_uuid}.{extension}"
    if is_s3path(output_prefix):
        return s3_client.S3Prefix(full_path)
    return pathlib.Path(full_path)


def _get_cosmos_predict2_cache_url(
    output_prefix: str,
    dataset_name: str,
    camera_view: str,
) -> s3_client.S3Prefix | pathlib.Path:
    """Generate a URL or path for storing cosmos-predict2 prefix embeddings cache files.

    Args:
        output_prefix: Base path for output
        dataset_name: Name of the dataset
        camera_view: Camera view name (e.g., "pinhole_front")

    Returns:
        An S3Prefix object if the output_prefix is an S3 path, otherwise a
        pathlib.Path object

    """
    full_path = f"{output_prefix}/datasets/{dataset_name}/cache/prefix_t5_embeddings_{camera_view}.pkl"
    if is_s3path(output_prefix):
        return s3_client.S3Prefix(full_path)
    return pathlib.Path(full_path)


def _make_camera_directories(
    output_prefix: str,
    dataset_name: str,
    camera_views: list[str],
) -> None:
    """Create camera-specific directories for cosmos-predict2 dataset structure.

    This function only creates directories for local filesystem paths.
    For S3 paths, directories are created implicitly when files are uploaded.

    Args:
        output_prefix: Base path for output
        dataset_name: Name of the dataset
        camera_views: List of camera view names to create directories for

    """
    # Only create directories for local filesystem
    if is_s3path(output_prefix):
        logger.debug("S3 path detected - directories will be created implicitly when files are uploaded")
        return

    base_path = f"{output_prefix}/datasets/{dataset_name}"

    # Create main directories
    directories_to_create = [
        f"{base_path}/cache",
    ]

    # Create camera-specific subdirectories
    for camera_view in camera_views:
        directories_to_create.extend(
            [
                f"{base_path}/metas/{camera_view}",
                f"{base_path}/videos/{camera_view}",
                f"{base_path}/t5_xxl/{camera_view}",
            ]
        )

    # Create directories for local filesystem
    for directory in directories_to_create:
        pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
        logger.debug(f"Created local directory: {directory}")


def _rm_files(
    file_paths: list[s3_client.S3Prefix | pathlib.Path],
    s3_client_instance: s3_client.S3Client | None,  # noqa: ARG001
    *,
    verbose: bool = False,
) -> None:
    """Clean up partially written files.

    Args:
        file_paths: List of file paths to delete
        s3_client_instance: S3 client for remote operations
        verbose: Whether to enable verbose logging

    """
    for file_path in file_paths:
        try:
            if isinstance(file_path, s3_client.S3Prefix):
                # For S3 paths, we'd need to implement S3 deletion
                # For now, log that cleanup is needed but not implemented
                if verbose:
                    logger.warning(f"S3 file cleanup not implemented: {file_path}")
            # For local paths, delete the file if it exists
            elif file_path.exists():
                file_path.unlink()
                if verbose:
                    logger.debug(f"Cleaned up partial file: {file_path}")
        except Exception as cleanup_error:  # noqa: BLE001
            logger.warning(f"Failed to clean up file {file_path}: {cleanup_error}")


def generate_prefix_embeddings(
    camera_views: list[str],
    prompt_type: str = "default",
    prompt_text: str | None = None,
    *,
    verbose: bool = False,
) -> dict[str, np.ndarray[Any, np.dtype[Any]]]:
    """Generate T5 embeddings for prompt prefixes by camera view.

    This function encodes just the prompt prefix text (e.g., "Describe what you see
    in this driving scene.") for each camera view, creating reusable prefix embeddings
    for training optimization.

    Args:
        camera_views: List of camera view names to generate prefixes for
        prompt_type: Type of prompt to encode (default, visibility, etc.)
        prompt_text: Custom prompt text to use instead of predefined prompt types
        verbose: Whether to enable verbose logging

    Returns:
        Dictionary mapping camera view names to prefix embeddings

    """
    prompt_text_to_encode = _get_prompt(prompt_type, prompt_text)

    if verbose:
        logger.info(f"Generating prefix embeddings for prompt: '{prompt_text_to_encode}'")

    encoder = t5_encoder.T5Encoder(t5_encoder.ModelVariant.T5_XXL)
    encoder.setup()

    encoded_results = encoder.encode([prompt_text_to_encode], batch_size=1)
    prefix_embedding = encoded_results[0].encoded_text

    if verbose:
        logger.info(f"Generated prefix embedding with shape: {prefix_embedding.shape}")

    # Create the same prefix embedding for each camera view
    # (The prefix is the same regardless of camera view)
    result = dict.fromkeys(camera_views, prefix_embedding)

    if verbose:
        logger.info(f"Created prefix embeddings for {len(camera_views)} camera views")

    return result


def write_prefix_embeddings_cache(  # noqa: PLR0913
    prefix_embeddings_by_view: dict[str, np.ndarray[Any, np.dtype[Any]]],
    s3_client_instance: s3_client.S3Client | None,
    output_prefix: str,
    dataset_name: str,
    prompt_type: str = "default",
    *,
    verbose: bool = False,
) -> int:
    """Write prefix embeddings cache files for each camera view.

    Args:
        prefix_embeddings_by_view: Dictionary mapping camera views to prefix embeddings
        s3_client_instance: S3 client for storage operations
        output_prefix: Base path for output
        dataset_name: Name of the dataset
        prompt_type: Type of prompt that was encoded
        verbose: Whether to enable verbose logging

    Returns:
        Number of cache files successfully written

    Raises:
        Exception: If cache file write operation fails

    """
    num_written = 0

    for camera_view, prefix_embedding in prefix_embeddings_by_view.items():
        cache_path = _get_cosmos_predict2_cache_url(
            output_prefix,
            dataset_name,
            camera_view,
        )

        cache_data = {
            "camera_view": camera_view,
            "dataset_name": dataset_name,
            "prompt_type": prompt_type,
            "prefix_embedding": prefix_embedding,
            "embedding_shape": prefix_embedding.shape,
            "metadata": {
                "version": "1.0",
                "format": "cosmos-predict2",
                "type": "prefix_embedding",
            },
        }

        cache_bytes = pickle.dumps(cache_data)

        write_bytes(
            cache_bytes,
            cache_path,
            f"prefix-cache-{camera_view}",
            "prefix_embeddings_cache",
            verbose=verbose,
            client=s3_client_instance,
            overwrite=True,
        )

        num_written += 1

        if verbose:
            logger.info(f"Wrote prefix embedding cache for {camera_view} (shape: {prefix_embedding.shape})")

    return num_written


def write_cosmos_predict2_dataset(  # noqa: PLR0913, C901
    clips: list[ClipForAnnotation],
    s3_client_instance: s3_client.S3Client | None,
    output_prefix: str,
    dataset_name: str,
    supported_cameras: dict[int, str],
    camera_views_mapping: dict[int, str],
    *,
    verbose: bool = False,
) -> int:
    """Write cosmos-predict2 dataset files for a list of clips.

    This function handles the core logic for writing cosmos-predict2 compatible
    dataset files, organizing clips by camera view and generating the required
    file structure.

    If a clip is missing any of the required data, it will be skipped and the
    error will be recorded in the clip.errors dictionary.

    Args:
        clips: List of clips to process
        s3_client_instance: S3 client for storage operations
        output_prefix: Base path for output
        dataset_name: Name of the dataset
        supported_cameras: Mapping of camera IDs to camera names
        camera_views_mapping: Mapping of camera IDs to cosmos-predict2 view names
        verbose: Whether to enable verbose logging

    Returns:
        Number of clips successfully processed

    """
    num_processed_clips = 0

    for clip in clips:
        # Skip clips from unsupported cameras
        if clip.camera_id not in supported_cameras:
            if verbose:
                logger.debug(f"Skipping clip {clip.clip_session_uuid} from unsupported camera {clip.camera_id}")
            continue

        camera_view = camera_views_mapping[clip.camera_id]

        if verbose:
            logger.debug(f"Processing clip {clip.clip_session_uuid} for camera view {camera_view}")

        # Check for required data - all must be present for cosmos-predict2
        clip_valid = True

        # Check for video encoded_data
        if not clip.encoded_data:
            clip.errors["encoded_data"] = "no video encoded_data present"
            clip_valid = False

        # Check for captions
        has_captions = any(window.captions.get("default", []) for window in clip.caption_windows)
        if not has_captions:
            clip.errors["captions"] = "no captions present"
            clip_valid = False

        # Check for T5 embeddings
        has_embeddings = any(
            window.t5_xxl_embeddings.get("default", None) is not None for window in clip.caption_windows
        )
        if not has_embeddings:
            clip.errors["embedding"] = "no embedding present"
            clip_valid = False

        # Skip clip if any required data is missing
        if not clip_valid:
            if verbose:
                error_details = ", ".join(f"{k}: {v}" for k, v in clip.errors.items())
                logger.warning(
                    f"Skipping clip {clip.clip_session_uuid} {clip.camera_id} due to missing data: {error_details}"
                )
            continue

        # Write all files for this clip - any exception excludes the clip
        written_files = []  # Track successfully written files for cleanup on failure

        try:
            # Create file URLs before function calls
            video_file_url = _get_cosmos_predict2_file_url(
                output_prefix, dataset_name, camera_view, clip.clip_session_uuid, "videos", "mp4"
            )
            caption_file_url = _get_cosmos_predict2_file_url(
                output_prefix, dataset_name, camera_view, clip.clip_session_uuid, "metas", "txt"
            )
            embedding_file_url = _get_cosmos_predict2_file_url(
                output_prefix, dataset_name, camera_view, clip.clip_session_uuid, "t5_xxl", "pkl"
            )

            # Write video clip
            write_video_clip(clip, camera_view, s3_client_instance, video_file_url, verbose=verbose)
            written_files.append(video_file_url)

            # Write caption text
            write_caption_text(clip, camera_view, s3_client_instance, caption_file_url, "default", verbose=verbose)
            written_files.append(caption_file_url)

            # Write T5 embedding
            write_t5_embedding(clip, camera_view, s3_client_instance, embedding_file_url, "default", verbose=verbose)
            written_files.append(embedding_file_url)

            num_processed_clips += 1

        except Exception as e:  # noqa: BLE001
            # Clean up any files that were successfully written
            _rm_files(written_files, s3_client_instance, verbose=verbose)

            clip.errors["write_failure"] = f"file write failed: {e}"

            if verbose:
                logger.error(
                    f"Failed to write files for clip {clip.clip_session_uuid} {clip.camera_id}: {e} - "
                    f"excluding from dataset"
                )

    if verbose:
        logger.info(f"Processed {num_processed_clips} clips for cosmos-predict2 dataset")

    return num_processed_clips


def write_video_clip(
    clip: ClipForAnnotation,
    camera_view: str,
    s3_client_instance: s3_client.S3Client | None,
    dest_url: s3_client.S3Prefix | pathlib.Path,
    *,
    verbose: bool = False,
) -> None:
    """Write a video clip to storage.

    Args:
        clip: Clip containing video data to write
        camera_view: Camera view name (e.g., "pinhole_front")
        s3_client_instance: S3 client for storage operations
        dest_url: Destination URL/path for the video file
        verbose: Whether to enable verbose logging

    Raises:
        ValueError: If clip has no encoded_data data
        Exception: If storage write operation fails

    """
    data = clip.encoded_data.resolve()
    if data is None:
        error_msg = f"Clip {clip.uuid} has no encoded_data data"
        raise ValueError(error_msg)

    write_bytes(
        data,
        dest_url,
        f"clip-{clip.clip_session_uuid}",
        "video_clip",
        verbose=verbose,
        client=s3_client_instance,
        overwrite=True,
    )

    if verbose:
        logger.debug(f"Successfully wrote video clip {clip.clip_session_uuid} to {camera_view}")


def write_caption_text(  # noqa: PLR0913
    clip: ClipForAnnotation,
    camera_view: str,
    s3_client_instance: s3_client.S3Client | None,
    dest_url: s3_client.S3Prefix | pathlib.Path,
    prompt_type: str = "default",
    *,
    verbose: bool = False,
) -> None:
    """Write caption text for a clip to storage.

    Args:
        clip: Clip containing caption data to write
        camera_view: Camera view name (e.g., "pinhole_front")
        s3_client_instance: S3 client for storage operations
        dest_url: Destination URL/path for the caption file
        prompt_type: Type of prompt to extract caption from
        verbose: Whether to enable verbose logging

    Raises:
        ValueError: If clip has no caption data
        Exception: If storage write operation fails

    """
    # Extract caption text from clip
    caption_text = ""
    for window in clip.caption_windows:
        captions = window.captions.get(prompt_type, [])
        if captions:
            caption_text = captions[-1]  # Use the last (most recent) caption
            break

    if not caption_text:
        error_msg = f"Clip {clip.clip_session_uuid} has no {prompt_type} caption"
        raise ValueError(error_msg)

    write_text(
        caption_text,
        dest_url,
        f"caption-{clip.clip_session_uuid}",
        "caption_text",
        verbose=verbose,
        client=s3_client_instance,
        overwrite=True,
    )

    if verbose:
        logger.debug(f"Successfully wrote caption for clip {clip.clip_session_uuid} to {camera_view}")


def write_t5_embedding(  # noqa: PLR0913
    clip: ClipForAnnotation,
    camera_view: str,
    s3_client_instance: s3_client.S3Client | None,
    dest_url: s3_client.S3Prefix | pathlib.Path,
    prompt_type: str = "default",
    *,
    verbose: bool = False,
) -> None:
    """Write T5 embedding for a clip to storage.

    Args:
        clip: Clip containing T5 embedding data to write
        camera_view: Camera view name (e.g., "pinhole_front")
        s3_client_instance: S3 client for storage operations
        dest_url: Destination URL/path for the embedding file
        prompt_type: Type of prompt to extract embedding from
        verbose: Whether to enable verbose logging

    Raises:
        ValueError: If clip has no embedding data
        Exception: If storage write operation fails

    """
    # Extract T5 embedding from clip
    embedding = None
    for window in clip.caption_windows:
        embeddings = window.t5_xxl_embeddings.get(prompt_type, None)
        if embeddings is not None:
            embedding = embeddings
            break

    if embedding is None:
        error_msg = f"Clip {clip.clip_session_uuid} has no {prompt_type} T5 embedding"
        raise ValueError(error_msg)

    # Serialize the embedding to bytes
    embedding_bytes = pickle.dumps([embedding])

    write_bytes(
        embedding_bytes,
        dest_url,
        f"t5-embedding-{clip.clip_session_uuid}",
        "t5_embedding",
        verbose=verbose,
        client=s3_client_instance,
        overwrite=True,
    )

    if verbose:
        logger.debug(f"Successfully wrote T5 embedding for clip {clip.clip_session_uuid} to {camera_view}")


class CosmosPredict2WriterStage(CuratorStage):
    """Stage for writing cosmos-predict2 compatible datasets.

    This stage generates cosmos-predict2 compatible dataset structures from
    AV captioning pipeline outputs, organizing clips by camera view and
    creating the required directory structure and file formats.
    """

    def __init__(
        self,
        output_prefix: str,
        dataset_name: str,
        camera_format_id: str,
        *,
        verbose: bool = False,
        log_stats: bool = False,
    ) -> None:
        """Initialize the CosmosPredict2WriterStage.

        Args:
            output_prefix: Base path for output files
            dataset_name: Name for the dataset (used in directory structure)
            camera_format_id: Camera configuration ("U" or "L") - used to determine supported cameras
            verbose: Whether to enable verbose logging
            log_stats: Whether to log performance statistics

        Raises:
            ValueError: If camera_format_id is not supported
            ValueError: If no supported cameras found in configuration

        """
        super().__init__()
        self._timer = StageTimer(self)
        self._output_prefix = output_prefix.rstrip("/")
        self._dataset_name = dataset_name
        self._verbose = verbose
        self._log_stats = log_stats
        self._s3_client: s3_client.S3Client | None = None

        # Validate camera format ID
        if camera_format_id not in CAMERA_MAPPING:
            error_msg = f"Unsupported camera_format_id: {camera_format_id}"
            raise ValueError(error_msg)

        # Get camera configuration and filter for supported cameras
        camera_config = CAMERA_MAPPING[camera_format_id]
        self._camera_id_mapping = camera_config["camera_name_mapping_cosmos"]

        # Filter for cameras that have cosmos-predict2 mappings
        self._supported_cameras = {
            camera_id: camera_name
            for camera_id, camera_name in self._camera_id_mapping.items()
            if camera_id in COSMOS_PREDICT2_CAMERA_MAPPING
        }

        if not self._supported_cameras:
            error_msg = f"No supported cameras found for camera_format_id: {camera_format_id}"
            raise ValueError(error_msg)

        # Create list of camera views for directory creation
        self._camera_views = [COSMOS_PREDICT2_CAMERA_MAPPING[camera_id] for camera_id in self._supported_cameras]

        # Create directories once during initialization (for local filesystem only)
        _make_camera_directories(
            self._output_prefix,
            self._dataset_name,
            self._camera_views,
        )

        if self._verbose:
            logger.info(f"CosmosPredict2WriterStage initialized for dataset: {dataset_name}")
            logger.info(f"Supported cameras: {list(self._supported_cameras.values())}")
            logger.info(f"Camera views: {self._camera_views}")

    @property
    def resources(self) -> CuratorStageResource:
        """Get the resource requirements for this stage.

        Returns:
            The resource requirements for this stage.

        """
        return CuratorStageResource(
            cpus=4.0,  # Higher CPU count for parallel file operations
            gpus=0,  # No GPU required for file writing
        )

    @property
    def conda_env_name(self) -> str:
        """Get the conda environment name.

        Returns:
            The conda environment name.

        """
        return "unified"

    def stage_setup(self) -> None:
        """Set up the CosmosPredict2WriterStage.

        This method initializes the S3 client for file operations.
        Directory creation was handled during initialization.
        """
        super().stage_setup()

        # Initialize S3 client
        self._s3_client = s3_client.create_s3_client(
            target_path=self._output_prefix,
            can_overwrite=True,
        )

    def process_data(self, tasks: list[AvClipAnnotationTask]) -> list[AvClipAnnotationTask] | None:  # type: ignore[override]
        """Process and write cosmos-predict2 dataset files.

        Args:
            tasks: Tasks containing clip annotations to process

        Returns:
            List containing the input tasks if successful

        """
        return [self._process_data(task) for task in tasks]

    def _process_data(self, task: AvClipAnnotationTask) -> AvClipAnnotationTask:
        """Process a single AvClipAnnotationTask.

        Args:
            task: Task containing clips to process

        Returns:
            The processed task

        """
        self._timer.reinit(self, task.get_major_size())

        with self._timer.time_process():
            num_processed_clips = write_cosmos_predict2_dataset(
                clips=task.clips,
                s3_client_instance=self._s3_client,
                output_prefix=self._output_prefix,
                dataset_name=self._dataset_name,
                supported_cameras=self._supported_cameras,
                camera_views_mapping=COSMOS_PREDICT2_CAMERA_MAPPING,
                verbose=self._verbose,
            )

            if self._verbose:
                logger.info(f"CosmosPredict2WriterStage processed {num_processed_clips} clips")

        if self._log_stats:
            stage_name, stage_perf_stats = self._timer.log_stats()
            task.stage_perf[stage_name] = stage_perf_stats

        return task

    def destroy(self) -> None:
        """Clean up resources used by the stage."""
        super().destroy()
        # Additional cleanup if needed


@ray.remote(num_gpus=1, runtime_env=PixiRuntimeEnv("unified"))
def generate_cosmos_predict2_prefix_cache(  # noqa: PLR0913
    output_prefix: str,
    dataset_name: str,
    camera_format_id: str,
    prompt_type: str,
    prompt_text: str | None,
    verbose: bool,  # noqa: FBT001, ray doesn't support keyword-only arguments
) -> None:
    """Generate prefix embeddings cache files after pipeline completion.

    This function runs ONCE per pipeline run, after all clip processing is complete.
    It loads the T5 encoder, encodes the prompt prefix for the specified prompt type,
    validates existing cache files, and regenerates if needed.

    Args:
        output_prefix: Base path for output files
        dataset_name: Name of the dataset
        camera_format_id: Camera configuration ID to determine supported camera views
        prompt_type: Single prompt type to generate cache files for
        prompt_text: Custom prompt text to use instead of predefined prompt types
        verbose: Whether to enable verbose logging

    Raises:
        ValueError: If camera_format_id is not supported
        Exception: If T5 encoder initialization fails

    """
    if verbose:
        logger.info(f"Starting prefix embeddings cache generation for dataset: {dataset_name}")
        logger.info(f"Prompt type: {prompt_type}")

    # Validate camera format ID
    if camera_format_id not in CAMERA_MAPPING:
        error_msg = f"Unsupported camera_format_id: {camera_format_id}"
        raise ValueError(error_msg)

    # Get camera configuration and filter for supported cameras
    camera_config = CAMERA_MAPPING[camera_format_id]
    camera_id_mapping = camera_config["camera_name_mapping_cosmos"]

    # Filter for cameras that have cosmos-predict2 mappings
    supported_cameras = {
        camera_id: camera_name
        for camera_id, camera_name in camera_id_mapping.items()
        if camera_id in COSMOS_PREDICT2_CAMERA_MAPPING
    }

    if not supported_cameras:
        error_msg = f"No supported cameras found for camera_format_id: {camera_format_id}"
        raise ValueError(error_msg)

    # Get camera views for cache generation
    camera_views = [COSMOS_PREDICT2_CAMERA_MAPPING[camera_id] for camera_id in supported_cameras]

    if verbose:
        logger.info(f"Generating prefix cache for camera views: {camera_views}")

    # Initialize S3 client for storage operations
    s3_client_instance = s3_client.create_s3_client(
        target_path=output_prefix,
        can_overwrite=True,
    )

    try:
        # For now, always regenerate cache files (validate + regenerate logic can be added later)
        if verbose:
            logger.info(f"Processing prompt type: {prompt_type}")

        # Generate prefix embeddings for this prompt type
        prefix_embeddings_by_view = generate_prefix_embeddings(
            camera_views=camera_views,
            prompt_type=prompt_type,
            prompt_text=prompt_text,
            verbose=verbose,
        )

        # Write cache files
        num_cache_files = write_prefix_embeddings_cache(
            prefix_embeddings_by_view=prefix_embeddings_by_view,
            s3_client_instance=s3_client_instance,
            output_prefix=output_prefix,
            dataset_name=dataset_name,
            prompt_type=prompt_type,
            verbose=verbose,
        )

        if verbose:
            logger.info(f"Generated {num_cache_files} cache files for prompt type '{prompt_type}'")
            logger.info("Prefix cache generation completed successfully")

    except Exception as cache_error:
        # Log error and re-raise to notify caller
        logger.exception(f"Failed to generate cache for prompt type '{prompt_type}': {cache_error}")
        raise
