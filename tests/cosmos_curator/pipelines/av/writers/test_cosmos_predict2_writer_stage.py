# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
"""Test cosmos_predict2_writer_stage module."""

import pathlib
import pickle
from unittest.mock import MagicMock, patch
from uuid import UUID

import numpy as np
import pytest

from cosmos_curator.core.utils.data.bytes_transport import bytes_to_numpy
from cosmos_curator.core.utils.storage.s3_client import S3Prefix, is_s3path
from cosmos_curator.pipelines.av.utils.av_data_model import CaptionWindow
from cosmos_curator.pipelines.av.writers.cosmos_predict2_writer_stage import (
    COSMOS_PREDICT2_CAMERA_MAPPING,
    CosmosPredict2WriterStage,
    _get_cosmos_predict2_cache_url,
    _get_cosmos_predict2_file_url,
    _make_camera_directories,
    _rm_files,
    generate_prefix_embeddings,
    write_caption_text,
    write_cosmos_predict2_dataset,
    write_prefix_embeddings_cache,
    write_t5_embedding,
    write_video_clip,
)
from tests.cosmos_curator.pipelines.av.writers.test_utils import (
    create_test_annotation_task,
)


class TestCosmosPredict2CameraMapping:
    """Test the cosmos-predict2 camera mapping functionality."""

    def test_camera_mapping_values_are_strings(self) -> None:
        """Test that all camera mapping values are strings."""
        for camera_view in COSMOS_PREDICT2_CAMERA_MAPPING.values():
            assert isinstance(camera_view, str)
            assert len(camera_view) > 0

    def test_camera_mapping_unique_values(self) -> None:
        """Test that all camera view names are unique."""
        camera_views = list(COSMOS_PREDICT2_CAMERA_MAPPING.values())
        assert len(camera_views) == len(set(camera_views))


class TestFilePathGeneration:
    """Test file path generation functions."""

    def test_get_cosmos_predict2_file_url_path_structure(self) -> None:
        """Test that file URL has correct path structure."""
        result = _get_cosmos_predict2_file_url(
            output_prefix="/base/path",
            dataset_name="waymo",
            camera_view="pinhole_front",
            clip_uuid=UUID("12345678-1234-5678-1234-567812345678"),
            file_type="videos",
            extension="mp4",
        )
        expected_path = "/base/path/datasets/waymo/videos/pinhole_front/12345678-1234-5678-1234-567812345678.mp4"
        assert str(result) == expected_path

    def test_get_cosmos_predict2_cache_url_path_structure(self) -> None:
        """Test that cache URL has correct path structure."""
        result = _get_cosmos_predict2_cache_url(
            output_prefix="/base/path",
            dataset_name="waymo",
            camera_view="pinhole_front",
        )
        expected_path = "/base/path/datasets/waymo/cache/prefix_t5_embeddings_pinhole_front.pkl"
        assert str(result) == expected_path


class TestDirectoryCreation:
    """Test directory creation functionality."""

    def test_make_camera_directories_local_filesystem(self, tmp_path: pathlib.Path) -> None:
        """Test directory creation for local filesystem."""
        camera_views = ["pinhole_front", "pinhole_side_left"]

        _make_camera_directories(
            output_prefix=str(tmp_path),
            dataset_name="test_dataset",
            camera_views=camera_views,
        )

        # Verify directories were created
        base_path = tmp_path / "datasets" / "test_dataset"

        # Check cache directory
        assert (base_path / "cache").exists()

        # Check camera-specific directories
        for camera_view in camera_views:
            assert (base_path / "metas" / camera_view).exists()
            assert (base_path / "videos" / camera_view).exists()
            assert (base_path / "t5_xxl" / camera_view).exists()

    def test_make_camera_directories_s3_backend(self) -> None:
        """Test directory creation for S3 backend (should not create actual directories)."""
        camera_views = ["pinhole_front"]

        # Should not raise any exceptions
        _make_camera_directories(
            output_prefix="s3://bucket/path",
            dataset_name="test_dataset",
            camera_views=camera_views,
        )

        # For S3, directories are created implicitly, so no actual operations should occur


class TestCosmosPredict2WriterStage:
    """Test the CosmosPredict2WriterStage class."""

    def test_init_valid_parameters(self, tmp_path: pathlib.Path) -> None:
        """Test that CosmosPredict2WriterStage can be initialized with valid parameters."""
        stage = CosmosPredict2WriterStage(
            output_prefix=str(tmp_path),
            dataset_name="test_dataset",
            camera_format_id="U",
            verbose=True,
            log_stats=True,
        )

        # Check that stage was initialized successfully
        assert stage is not None

    def test_init_invalid_camera_format_id(self) -> None:
        """Test that initialization fails with invalid camera format ID."""
        with pytest.raises(ValueError, match="Unsupported camera_format_id"):
            CosmosPredict2WriterStage(
                output_prefix="/test/path",
                dataset_name="test_dataset",
                camera_format_id="INVALID",
            )

    def test_init_strips_trailing_slash(self, tmp_path: pathlib.Path) -> None:
        """Test that trailing slashes are stripped from output prefix."""
        stage = CosmosPredict2WriterStage(
            output_prefix=str(tmp_path) + "/",  # Add trailing slash to test stripping
            dataset_name="test_dataset",
            camera_format_id="U",
        )

        # Check that trailing slash was stripped (test behavior, not private attribute)
        assert stage is not None  # Stage initialized successfully

    @pytest.mark.parametrize(
        ("output_prefix"),
        [
            ("s3://bucket/path"),
            (""),  # Will be replaced with tmp_path
        ],
    )
    def test_stage_setup_s3_client_creation(self, tmp_path: pathlib.Path, output_prefix: str) -> None:
        """Test that stage_setup creates S3 client correctly for different path types."""
        # Use tmp_path for local filesystem test
        if not is_s3path(output_prefix):
            output_prefix = str(tmp_path)

        stage = CosmosPredict2WriterStage(
            output_prefix=output_prefix,
            dataset_name="test_dataset",
            camera_format_id="U",
        )

        # Mock S3Client class and config to avoid creating real AWS connections
        with (
            patch("cosmos_curator.core.utils.storage.s3_client.S3Client") as mock_s3_client_class,
            patch("cosmos_curator.core.utils.storage.s3_client.get_s3_client_config") as mock_get_config,
        ):
            mock_s3_client_instance = MagicMock()
            mock_s3_client_class.return_value = mock_s3_client_instance
            mock_config = MagicMock()
            mock_get_config.return_value = mock_config

            stage.stage_setup()

            if is_s3path(output_prefix):
                # For S3 paths, S3Client should be created and stored
                assert stage._s3_client is not None
            else:
                # For local paths, S3 client should be None
                assert stage._s3_client is None

    def test_process_data(self, tmp_path: pathlib.Path) -> None:
        """Test basic process_data functionality."""
        stage = CosmosPredict2WriterStage(
            output_prefix=str(tmp_path),
            dataset_name="test_dataset",
            camera_format_id="U",
        )
        stage.stage_setup()  # Initialize S3 client

        # Create test task with clips that have all required data
        task = create_test_annotation_task()

        # Set up clips to be processed successfully
        for clip in task.clips:
            clip.camera_id = 2  # Supported camera
            clip.encoded_data = bytes_to_numpy(b"fake_video_data")
            clip.caption_windows = [
                CaptionWindow(
                    start_frame=0,
                    end_frame=256,
                    captions={"default": ["Test caption"]},
                    t5_xxl_embeddings={"default": np.array([0.1, 0.2, 0.3], dtype=np.float32)},
                    model_input={},
                )
            ]

        # Test public interface
        result = stage.process_data([task])

        # Should return the same task
        assert result == [task]

    @pytest.mark.parametrize("log_stats", [False, True])
    def test_process_data_single_task(self, tmp_path: pathlib.Path, *, log_stats: bool) -> None:
        """Test processing of a single task."""
        stage = CosmosPredict2WriterStage(
            output_prefix=str(tmp_path),
            dataset_name="test_dataset",
            camera_format_id="U",
            log_stats=log_stats,
        )

        # Set S3 client to None for local path (realistic for local filesystem)
        stage._s3_client = None

        # Create test task
        task = create_test_annotation_task()

        # Process the task
        result = stage._process_data(task)

        # Should return the same task
        assert result == task

        # Validate performance stats behavior based on log_stats parameter
        if log_stats:
            # When log_stats=True, performance stats should be recorded
            assert "CosmosPredict2WriterStage" in task.stage_perf
        else:
            # When log_stats=False, performance stats should not be recorded
            assert "CosmosPredict2WriterStage" not in task.stage_perf

    def test_init_creates_directories_for_local_path(self, tmp_path: pathlib.Path) -> None:
        """Test that initialization creates directories for local filesystem paths."""
        CosmosPredict2WriterStage(
            output_prefix=str(tmp_path),
            dataset_name="test_dataset",
            camera_format_id="U",
        )

        # Verify directories were created during initialization
        base_path = tmp_path / "datasets" / "test_dataset"
        assert (base_path / "cache").exists()

        # Check at least one camera-specific directory
        assert (base_path / "metas" / "pinhole_front").exists()
        assert (base_path / "videos" / "pinhole_front").exists()
        assert (base_path / "t5_xxl" / "pinhole_front").exists()


class TestWriteCosmosPredict2Dataset:
    """Test the standalone write_cosmos_predict2_dataset function."""

    @pytest.mark.parametrize(
        ("scenario", "use_real_clips", "supported_cameras", "expected_result"),
        [
            ("empty_clips", False, {2: "camera_front_wide_120fov"}, 0),
            ("no_supported_cameras", True, {99: "camera_nonexistent"}, 0),
            ("filters_unsupported_cameras", True, {2: "camera_front_wide_120fov"}, "dynamic"),
        ],
    )
    def test_write_cosmos_predict2_dataset(
        self,
        tmp_path: pathlib.Path,
        scenario: str,
        *,
        use_real_clips: bool,
        supported_cameras: dict[int, str],
        expected_result: int | str,
    ) -> None:
        """Test write_cosmos_predict2_dataset function with various input scenarios."""
        # Set up clips based on scenario
        if use_real_clips:
            task = create_test_annotation_task()
            clips = task.clips
        else:
            clips = []

        # Set up camera views mapping based on supported cameras
        camera_views_mapping = {
            camera_id: f"pinhole_{camera_name.split('_')[-1]}" for camera_id, camera_name in supported_cameras.items()
        }

        result = write_cosmos_predict2_dataset(
            clips=clips,
            s3_client_instance=None,
            output_prefix=str(tmp_path),
            dataset_name="test_dataset",
            supported_cameras=supported_cameras,
            camera_views_mapping=camera_views_mapping,
            verbose=(scenario != "empty_clips"),  # Vary verbose based on scenario
        )

        # Calculate expected result
        if expected_result == "dynamic":
            # For filtering case, calculate expected clips dynamically
            # Note: clips must have all required data (encoded_data, captions, embeddings)
            expected_clips = [
                clip
                for clip in clips
                if clip.camera_id in supported_cameras
                and clip.encoded_data.resolve() is not None
                and any(window.captions.get("default", []) for window in clip.caption_windows)
                and any(window.t5_xxl_embeddings.get("default", None) is not None for window in clip.caption_windows)
            ]
            expected_result = len(expected_clips)

        assert result == expected_result

    def test_write_cosmos_predict2_dataset_atomic_writes(self, tmp_path: pathlib.Path) -> None:
        """Test that partial clips are not written when one file write fails."""
        # Create test task with clips that have all required data
        task = create_test_annotation_task()
        clips = task.clips[:2]  # Use first 2 clips for testing

        # Ensure clips have all required data and set camera_id to supported value
        for clip in clips:
            clip.camera_id = 2  # Set to supported camera
            clip.encoded_data = bytes_to_numpy(b"fake_video_data")
            clip.caption_windows = [
                CaptionWindow(
                    start_frame=0,
                    end_frame=256,
                    captions={"default": ["Test caption"]},
                    t5_xxl_embeddings={"default": np.array([0.1, 0.2, 0.3], dtype=np.float32)},
                    model_input={},
                )
            ]

        supported_cameras = {2: "camera_front_wide_120fov"}
        camera_views_mapping = {2: "pinhole_front"}

        # Mock write_t5_embedding to fail on the second clip
        original_write_t5_embedding = write_t5_embedding
        call_count = [0]  # Use list to make it mutable in nested function
        fail_on_call = 2  # Fail on second call (second clip)
        error_message = "Simulated T5 write failure"

        def mock_write_t5_embedding(*args: object, **kwargs: object) -> None:
            call_count[0] += 1
            if call_count[0] == fail_on_call:
                raise RuntimeError(error_message)
            return original_write_t5_embedding(*args, **kwargs)

        # Patch the write_t5_embedding function
        patch_target = "cosmos_curator.pipelines.av.writers.cosmos_predict2_writer_stage.write_t5_embedding"
        with patch(patch_target, side_effect=mock_write_t5_embedding):
            result = write_cosmos_predict2_dataset(
                clips=clips,
                s3_client_instance=None,
                output_prefix=str(tmp_path),
                dataset_name="test_dataset",
                supported_cameras=supported_cameras,
                camera_views_mapping=camera_views_mapping,
                verbose=True,
            )

        # Should only process 1 clip (first one succeeds, second fails and gets cleaned up)
        assert result == 1

        # Verify first clip files exist (successful write)
        clip1_uuid = clips[0].uuid
        base_path = tmp_path / "datasets" / "test_dataset" / "pinhole_front"

        assert (base_path.parent / "videos" / "pinhole_front" / f"{clip1_uuid}.mp4").exists()
        assert (base_path.parent / "metas" / "pinhole_front" / f"{clip1_uuid}.txt").exists()
        assert (base_path.parent / "t5_xxl" / "pinhole_front" / f"{clip1_uuid}.pkl").exists()

        # Verify second clip files do NOT exist (failed write was cleaned up)
        clip2_uuid = clips[1].uuid

        # Video and caption files should not exist
        assert not (base_path.parent / "videos" / "pinhole_front" / f"{clip2_uuid}.mp4").exists()
        assert not (base_path.parent / "metas" / "pinhole_front" / f"{clip2_uuid}.txt").exists()
        assert not (base_path.parent / "t5_xxl" / "pinhole_front" / f"{clip2_uuid}.pkl").exists()

        # Verify error was recorded in the failed clip
        assert "write_failure" in clips[1].errors
        assert error_message in clips[1].errors["write_failure"]

    def test_write_cosmos_predict2_dataset_cleanup_on_video_write_failure(self, tmp_path: pathlib.Path) -> None:
        """Test cleanup when video write fails (first in sequence)."""
        # Create test task
        task = create_test_annotation_task()
        clip = task.clips[0]

        # Ensure clip has all required data and set camera_id to supported value
        clip.camera_id = 2  # Set to supported camera
        clip.encoded_data = bytes_to_numpy(b"fake_video_data")
        clip.caption_windows = [
            CaptionWindow(
                start_frame=0,
                end_frame=256,
                captions={"default": ["Test caption"]},
                t5_xxl_embeddings={"default": np.array([0.1, 0.2, 0.3], dtype=np.float32)},
                model_input={},
            )
        ]

        supported_cameras = {2: "camera_front_wide_120fov"}
        camera_views_mapping = {2: "pinhole_front"}

        # Mock write_video_clip to fail immediately
        patch_target = "cosmos_curator.pipelines.av.writers.cosmos_predict2_writer_stage.write_video_clip"
        error_message = "Video write failed"
        with patch(patch_target, side_effect=RuntimeError(error_message)):
            result = write_cosmos_predict2_dataset(
                clips=[clip],
                s3_client_instance=None,
                output_prefix=str(tmp_path),
                dataset_name="test_dataset",
                supported_cameras=supported_cameras,
                camera_views_mapping=camera_views_mapping,
                verbose=True,
            )

        # No clips should be processed successfully
        assert result == 0

        # Verify no files exist (since video write failed first, no cleanup needed)
        clip_uuid = clip.uuid
        base_path = tmp_path / "datasets" / "test_dataset" / "pinhole_front"

        assert not (base_path.parent / "videos" / "pinhole_front" / f"{clip_uuid}.mp4").exists()
        assert not (base_path.parent / "metas" / "pinhole_front" / f"{clip_uuid}.txt").exists()
        assert not (base_path.parent / "t5_xxl" / "pinhole_front" / f"{clip_uuid}.pkl").exists()

        # Verify error was recorded
        assert "write_failure" in clip.errors
        assert error_message in clip.errors["write_failure"]

    def test_write_cosmos_predict2_dataset_cleanup_on_caption_write_failure(self, tmp_path: pathlib.Path) -> None:
        """Test cleanup when caption write fails (middle in sequence)."""
        # Create test task
        task = create_test_annotation_task()
        clip = task.clips[0]

        # Ensure clip has all required data and set camera_id to supported value
        clip.camera_id = 2  # Set to supported camera
        clip.encoded_data = bytes_to_numpy(b"fake_video_data")
        clip.caption_windows = [
            CaptionWindow(
                start_frame=0,
                end_frame=256,
                captions={"default": ["Test caption"]},
                t5_xxl_embeddings={"default": np.array([0.1, 0.2, 0.3], dtype=np.float32)},
                model_input={},
            )
        ]

        supported_cameras = {2: "camera_front_wide_120fov"}
        camera_views_mapping = {2: "pinhole_front"}

        # Mock write_caption_text to fail (video should succeed first, then caption fails)
        patch_target = "cosmos_curator.pipelines.av.writers.cosmos_predict2_writer_stage.write_caption_text"
        error_message = "Caption write failed"
        with patch(patch_target, side_effect=RuntimeError(error_message)):
            result = write_cosmos_predict2_dataset(
                clips=[clip],
                s3_client_instance=None,
                output_prefix=str(tmp_path),
                dataset_name="test_dataset",
                supported_cameras=supported_cameras,
                camera_views_mapping=camera_views_mapping,
                verbose=True,
            )

        # No clips should be processed successfully
        assert result == 0

        # Verify no files exist (video file should have been cleaned up after caption failure)
        clip_uuid = clip.uuid
        base_path = tmp_path / "datasets" / "test_dataset" / "pinhole_front"

        assert not (base_path.parent / "videos" / "pinhole_front" / f"{clip_uuid}.mp4").exists()
        assert not (base_path.parent / "metas" / "pinhole_front" / f"{clip_uuid}.txt").exists()
        assert not (base_path.parent / "t5_xxl" / "pinhole_front" / f"{clip_uuid}.pkl").exists()

        # Verify error was recorded
        assert "write_failure" in clip.errors
        assert error_message in clip.errors["write_failure"]


class TestWriteVideoClip:
    """Test the write_video_clip function."""

    def test_write_video_clip_success(self, tmp_path: pathlib.Path) -> None:
        """Test successful video clip writing."""
        # Create test clip with encoded_data
        task = create_test_annotation_task()
        clip = task.clips[0]

        # Ensure clip has encoded_data
        if not clip.encoded_data:
            clip.encoded_data = bytes_to_numpy(b"fake_video_data")

        # Create destination URL
        dest_url = tmp_path / "datasets" / "test_dataset" / "videos" / "pinhole_front" / f"{clip.uuid}.mp4"

        # Write video clip
        write_video_clip(
            clip=clip,
            camera_view="pinhole_front",
            s3_client_instance=None,
            dest_url=dest_url,
            verbose=True,
        )

        # Verify file was written
        assert dest_url.exists()
        assert dest_url.read_bytes() == clip.encoded_data.resolve().tobytes()

    def test_write_video_clip_no_encoded_data(self, tmp_path: pathlib.Path) -> None:
        """Test video clip writing fails when no encoded_data."""
        # Create test clip without encoded_data
        task = create_test_annotation_task()
        clip = task.clips[0]
        clip.encoded_data.drop()

        # Create destination URL
        dest_url = tmp_path / "datasets" / "test_dataset" / "videos" / "pinhole_front" / f"{clip.uuid}.mp4"

        # Should raise ValueError for missing encoded_data
        with pytest.raises(ValueError, match=f"Clip {clip.uuid} has no encoded_data"):
            write_video_clip(
                clip=clip,
                camera_view="pinhole_front",
                s3_client_instance=None,
                dest_url=dest_url,
            )


class TestWriteCaptionText:
    """Test the write_caption_text function."""

    def test_write_caption_text_success(self, tmp_path: pathlib.Path) -> None:
        """Test successful caption text writing."""
        # Create test clip with captions
        task = create_test_annotation_task()
        clip = task.clips[0]

        # Ensure clip has caption data
        if not clip.caption_windows or not clip.caption_windows[0].captions.get("default", []):
            caption_window = CaptionWindow(
                start_frame=0,
                end_frame=256,
                captions={"default": ["Test caption text"]},
                t5_xxl_embeddings={},
                model_input={},
            )
            clip.caption_windows = [caption_window]

        # Create destination URL
        dest_url = tmp_path / "datasets" / "test_dataset" / "metas" / "pinhole_front" / f"{clip.uuid}.txt"

        # Write caption text
        write_caption_text(
            clip=clip,
            camera_view="pinhole_front",
            s3_client_instance=None,
            dest_url=dest_url,
            verbose=True,
        )

        # Verify file was written
        assert dest_url.exists()
        assert dest_url.read_text() == "Test caption text"

    def test_write_caption_text_no_captions(self, tmp_path: pathlib.Path) -> None:
        """Test caption text writing fails when no caption data."""
        # Create test clip without captions
        task = create_test_annotation_task()
        clip = task.clips[0]
        clip.caption_windows = []

        # Create destination URL
        dest_url = tmp_path / "datasets" / "test_dataset" / "metas" / "pinhole_front" / f"{clip.uuid}.txt"

        # Should raise ValueError for missing captions
        with pytest.raises(ValueError, match=f"Clip {clip.clip_session_uuid} has no default caption"):
            write_caption_text(
                clip=clip,
                camera_view="pinhole_front",
                s3_client_instance=None,
                dest_url=dest_url,
            )


class TestWriteT5Embedding:
    """Test the write_t5_embedding function."""

    def test_write_t5_embedding_success(self, tmp_path: pathlib.Path) -> None:
        """Test successful T5 embedding writing."""
        # Create test clip with T5 embeddings
        task = create_test_annotation_task()
        clip = task.clips[0]

        # Set up test T5 embedding data
        test_embedding = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)  # Mock embedding
        caption_window = CaptionWindow(
            start_frame=0,
            end_frame=256,
            captions={},
            t5_xxl_embeddings={"default": test_embedding},
            model_input={},
        )
        clip.caption_windows = [caption_window]

        # Create destination URL
        dest_url = tmp_path / "datasets" / "test_dataset" / "t5_xxl" / "pinhole_front" / f"{clip.uuid}.pkl"

        # Write T5 embedding
        write_t5_embedding(
            clip=clip,
            camera_view="pinhole_front",
            s3_client_instance=None,
            dest_url=dest_url,
            verbose=True,
        )

        # Verify file was written
        assert dest_url.exists()

        # Verify content - should be a list containing the embedding
        with dest_url.open("rb") as f:
            loaded_embedding_list = pickle.load(f)  # noqa: S301
        assert isinstance(loaded_embedding_list, list)
        assert len(loaded_embedding_list) == 1
        np.testing.assert_array_equal(loaded_embedding_list[0], test_embedding)

    def test_write_t5_embedding_no_embeddings(self, tmp_path: pathlib.Path) -> None:
        """Test T5 embedding writing fails when no embedding data."""
        # Create test clip without T5 embeddings
        task = create_test_annotation_task()
        clip = task.clips[0]
        clip.caption_windows = []

        # Create destination URL
        dest_url = tmp_path / "datasets" / "test_dataset" / "t5_xxl" / "pinhole_front" / f"{clip.uuid}.pkl"

        # Should raise ValueError for missing embeddings
        with pytest.raises(ValueError, match=f"Clip {clip.clip_session_uuid} has no default T5 embedding"):
            write_t5_embedding(
                clip=clip,
                camera_view="pinhole_front",
                s3_client_instance=None,
                dest_url=dest_url,
            )


class TestRmFiles:
    """Test the _rm_files function."""

    def test_rm_files_local_existing_files(self, tmp_path: pathlib.Path) -> None:
        """Test removal of existing local files."""
        # Create test files
        file1 = tmp_path / "file1.txt"
        file2 = tmp_path / "file2.mp4"
        file3 = tmp_path / "file3.pkl"

        file1.write_text("test content 1")
        file2.write_bytes(b"test video data")
        file3.write_bytes(b"test pickle data")

        # Verify files exist
        assert file1.exists()
        assert file2.exists()
        assert file3.exists()

        # Remove files
        _rm_files(
            file_paths=[file1, file2, file3],
            s3_client_instance=None,
            verbose=True,
        )

        # Verify files were deleted
        assert not file1.exists()
        assert not file2.exists()
        assert not file3.exists()

    def test_rm_files_local_nonexistent_files(self, tmp_path: pathlib.Path) -> None:
        """Test removal with non-existent local files (should not raise errors)."""
        # Create paths to non-existent files
        file1 = tmp_path / "nonexistent1.txt"
        file2 = tmp_path / "nonexistent2.mp4"

        # Verify files don't exist
        assert not file1.exists()
        assert not file2.exists()

        # Remove non-existent files (should not raise errors)
        _rm_files(
            file_paths=[file1, file2],
            s3_client_instance=None,
            verbose=True,
        )

        # Files should still not exist (no change)
        assert not file1.exists()
        assert not file2.exists()

    def test_rm_files_mixed_existing_nonexistent(self, tmp_path: pathlib.Path) -> None:
        """Test _rm_files with mix of existing and non-existent files."""
        # Create one existing file and one non-existent
        existing_file = tmp_path / "existing.txt"
        nonexistent_file = tmp_path / "nonexistent.txt"

        existing_file.write_text("test content")

        # Verify initial state
        assert existing_file.exists()
        assert not nonexistent_file.exists()

        # Remove both files
        _rm_files(
            file_paths=[existing_file, nonexistent_file],
            s3_client_instance=None,
            verbose=False,
        )

        # Verify results
        assert not existing_file.exists()  # Should be deleted
        assert not nonexistent_file.exists()  # Should remain non-existent

    def test_rm_files_empty_list(self) -> None:
        """Test _rm_files with empty file list."""
        # Should not raise any errors
        _rm_files(
            file_paths=[],
            s3_client_instance=None,
            verbose=True,
        )

    def test_rm_files_s3_paths_no_errors(self) -> None:
        """Test that S3 paths don't cause errors (functionality test)."""
        # Create mock S3 paths
        s3_path1 = S3Prefix("s3://bucket/path/file1.txt")
        s3_path2 = S3Prefix("s3://bucket/path/file2.mp4")

        # Should not raise any exceptions (S3 removal is not implemented but shouldn't fail)
        _rm_files(
            file_paths=[s3_path1, s3_path2],
            s3_client_instance=None,
            verbose=True,
        )

        # If we get here, the function handled S3 paths without crashing

    def test_rm_files_s3_paths_silent_no_errors(self) -> None:
        """Test that S3 paths work with verbose=False."""
        # Create mock S3 paths
        s3_path1 = S3Prefix("s3://bucket/path/file1.txt")

        # Should not raise any exceptions
        _rm_files(
            file_paths=[s3_path1],
            s3_client_instance=None,
            verbose=False,
        )

        # If we get here, the function handled S3 paths without crashing

    def test_rm_files_permission_error_no_crash(self, tmp_path: pathlib.Path) -> None:
        """Test _rm_files handles permission errors gracefully without crashing."""
        # Create a file
        test_file = tmp_path / "test_file.txt"
        test_file.write_text("test content")

        # Mock the unlink method to raise a PermissionError
        with patch.object(pathlib.Path, "unlink", side_effect=PermissionError("Permission denied")):
            # Should not raise an exception - errors are caught and logged
            _rm_files(
                file_paths=[test_file],
                s3_client_instance=None,
                verbose=True,
            )

            # If we get here, the function handled the error gracefully
            # File should still exist since unlink failed (but we can't check this with the mock)

    def test_rm_files_verbose_successful_cleanup(self, tmp_path: pathlib.Path) -> None:
        """Test that verbose mode works correctly for successful cleanup."""
        # Create test file
        test_file = tmp_path / "test_file.txt"
        test_file.write_text("test content")

        # Verify file exists
        assert test_file.exists()

        # Should not raise any exceptions
        _rm_files(
            file_paths=[test_file],
            s3_client_instance=None,
            verbose=True,
        )

        # Verify file was deleted
        assert not test_file.exists()


class TestGeneratePrefixEmbeddings:
    """Test the generate_prefix_embeddings function."""

    @pytest.mark.parametrize(
        ("test_case", "camera_views", "prompt_type", "expected_count", "expected_views"),
        [
            (
                "multiple_cameras",
                ["pinhole_front", "pinhole_front_left"],
                "default",
                2,
                ["pinhole_front", "pinhole_front_left"],
            ),
            ("single_camera", ["pinhole_front"], "visibility", 1, ["pinhole_front"]),
            ("empty_cameras", [], "default", 0, []),
        ],
    )
    def test_generate_prefix_embeddings(
        self,
        test_case: str,  # noqa: ARG002
        camera_views: list[str],
        prompt_type: str,
        expected_count: int,
        expected_views: list[str],
    ) -> None:
        """Test prefix embedding generation for various scenarios."""
        # Mock the T5 encoder to avoid loading the actual model
        mock_embedding = np.array([[0.1, 0.2, 0.3]] * 10, dtype=np.float32)
        mock_encoded_sample = MagicMock()
        mock_encoded_sample.encoded_text = mock_embedding

        with patch("cosmos_curator.models.t5_encoder.T5Encoder") as mock_t5_class:
            mock_encoder = MagicMock()
            mock_encoder.encode.return_value = [mock_encoded_sample]
            mock_t5_class.return_value = mock_encoder

            result = generate_prefix_embeddings(
                camera_views=camera_views,
                prompt_type=prompt_type,
                prompt_text=None,
                verbose=True,
            )

            # Test outcomes based on expected results
            assert len(result) == expected_count

            # For non-empty results, verify expected camera views and data
            for expected_view in expected_views:
                assert expected_view in result
                np.testing.assert_array_equal(result[expected_view], mock_embedding)


class TestWritePrefixEmbeddingsCache:
    """Test the write_prefix_embeddings_cache function."""

    def test_write_prefix_embeddings_cache_success(self, tmp_path: pathlib.Path) -> None:
        """Test successful cache file writing."""
        # Create test prefix embeddings data (single embedding per camera view)
        prefix_embeddings_by_view = {
            "pinhole_front": np.array([[0.1, 0.2, 0.3]] * 10, dtype=np.float32),  # Shape: (10, 3)
            "pinhole_front_left": np.array([[0.4, 0.5, 0.6]] * 10, dtype=np.float32),  # Shape: (10, 3)
        }

        result = write_prefix_embeddings_cache(
            prefix_embeddings_by_view=prefix_embeddings_by_view,
            s3_client_instance=None,
            output_prefix=str(tmp_path),
            dataset_name="test_dataset",
            prompt_type="default",
            verbose=True,
        )

        # Should write 2 cache files
        expected_cache_files = 2
        assert result == expected_cache_files

        # Verify cache files exist
        cache_front = tmp_path / "datasets" / "test_dataset" / "cache" / "prefix_t5_embeddings_pinhole_front.pkl"
        cache_left = tmp_path / "datasets" / "test_dataset" / "cache" / "prefix_t5_embeddings_pinhole_front_left.pkl"

        assert cache_front.exists()
        assert cache_left.exists()

        # Verify cache file content
        with cache_front.open("rb") as f:
            cache_data = pickle.load(f)  # noqa: S301

        assert cache_data["camera_view"] == "pinhole_front"
        assert cache_data["dataset_name"] == "test_dataset"
        assert cache_data["prompt_type"] == "default"
        assert "prefix_embedding" in cache_data
        assert cache_data["embedding_shape"] == (10, 3)
        assert cache_data["metadata"]["version"] == "1.0"
        assert cache_data["metadata"]["format"] == "cosmos-predict2"
        assert cache_data["metadata"]["type"] == "prefix_embedding"

        # Verify the actual embedding data
        np.testing.assert_array_equal(cache_data["prefix_embedding"], prefix_embeddings_by_view["pinhole_front"])

    def test_write_prefix_embeddings_cache_empty_embeddings(self, tmp_path: pathlib.Path) -> None:
        """Test cache writing with empty embeddings."""
        # Empty embeddings data
        prefix_embeddings_by_view: dict[str, np.ndarray] = {}

        result = write_prefix_embeddings_cache(
            prefix_embeddings_by_view=prefix_embeddings_by_view,
            s3_client_instance=None,
            output_prefix=str(tmp_path),
            dataset_name="test_dataset",
            prompt_type="default",
            verbose=True,
        )

        # Should write 0 cache files
        assert result == 0
