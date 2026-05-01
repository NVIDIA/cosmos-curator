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
"""Tests for data_model utility functions."""

import pathlib
import sys
import uuid
from collections import deque
from typing import Any
from unittest.mock import MagicMock, patch

import attrs
import numpy as np
import numpy.typing as npt
import pytest

from cosmos_curator.core.utils.data.bytes_transport import bytes_to_numpy
from cosmos_curator.pipelines.video.utils.data_model import (
    Clip,
    SplitPipeTask,
    Video,
    VideoMetadata,
    _add_children_to_queue,
    _get_object_size,
    assert_time_alignment,
    assert_video_clip_alignment,
    check_clip_time_alignment,
    get_major_size,
    get_video_from_task,
)


class TestGetObjectSize:
    """Test _get_object_size function."""

    def test_numpy_array(self) -> None:
        """Test size calculation for numpy arrays."""
        arr = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        assert _get_object_size(arr) == arr.nbytes

    def test_numpy_scalar(self) -> None:
        """Test size calculation for numpy scalars."""
        scalar = np.int32(42)
        assert _get_object_size(scalar) == scalar.nbytes

    def test_torch_tensor(self) -> None:
        """Test size calculation for torch tensors."""
        # Mock torch tensor
        mock_tensor = MagicMock()
        element_size = 4
        num_elements = 10
        expected_size = element_size * num_elements
        mock_tensor.element_size.return_value = element_size
        mock_tensor.nelement.return_value = num_elements

        # Create a mock tensor type
        mock_tensor_type = type(mock_tensor)

        with patch("cosmos_curator.pipelines.video.utils.data_model.TensorType", mock_tensor_type):
            result = _get_object_size(mock_tensor)
            assert result == expected_size

    def test_regular_object(self) -> None:
        """Test size calculation for regular Python objects."""
        obj = "test string"
        expected_size = sys.getsizeof(obj)
        assert _get_object_size(obj) == expected_size

    def test_dict(self) -> None:
        """Test size calculation for dictionaries."""
        obj = {"key": "value"}
        # Dictionaries return 0 since we only count contents, not the container
        expected_size = 0
        assert _get_object_size(obj) == expected_size

    def test_list(self) -> None:
        """Test size calculation for lists."""
        obj = [np.array([1, 2, 3, 4, 5])]
        # lists return 0 since we only count contents, not the container
        expected_size = 0
        assert _get_object_size(obj) == expected_size

    def test_none(self) -> None:
        """Test size calculation for None."""
        obj = None
        # None should return 0 since it represents absence of data
        expected_size = 0
        assert _get_object_size(obj) == expected_size


class TestAddChildrenToQueue:
    """Test _add_children_to_queue function."""

    def test_dict_children(self) -> None:
        """Test adding children from dictionary."""
        obj = {"a": 1, "b": 2}
        q: deque[object] = deque()
        visited: set[int] = set()

        _add_children_to_queue(obj, q, visited)

        expected_count = 2
        assert len(q) == expected_count
        assert obj["a"] in q
        assert obj["b"] in q

    def test_list_children(self) -> None:
        """Test adding children from list."""
        obj = [1, 2, 3]
        q: deque[object] = deque()
        visited: set[int] = set()

        _add_children_to_queue(obj, q, visited)

        expected_count = 3
        assert len(q) == expected_count
        assert obj[0] in q
        assert obj[1] in q
        assert obj[2] in q

    def test_tuple_children(self) -> None:
        """Test adding children from tuple."""
        obj = (1, 2, 3)
        q: deque[object] = deque()
        visited: set[int] = set()

        _add_children_to_queue(obj, q, visited)

        expected_count = 3
        assert len(q) == expected_count
        assert obj[0] in q
        assert obj[1] in q
        assert obj[2] in q

    def test_attrs_object_children(self) -> None:
        """Test adding children from attrs object."""

        @attrs.define
        class TestClass:
            field1: int = 1
            field2: str = "test"
            stage_perf: dict[str, str] = attrs.Factory(dict)  # Should be skipped

        obj = TestClass()
        q: deque[object] = deque()
        visited: set[int] = set()

        _add_children_to_queue(obj, q, visited)

        expected_count = 2
        assert len(q) == expected_count
        assert 1 in q
        assert "test" in q

    def test_visited_objects_skipped(self) -> None:
        """Test that already visited objects are skipped."""
        shared_obj = "shared"
        obj = {"a": shared_obj, "b": shared_obj}
        q: deque[object] = deque()
        visited: set[int] = {id(shared_obj)}

        _add_children_to_queue(obj, q, visited)

        # Should not add shared_obj since it's already visited
        assert len(q) == 0

    def test_non_attrs_object(self) -> None:
        """Test handling of non-attrs objects."""
        obj = "simple string"
        q: deque[object] = deque()
        visited: set[int] = set()

        _add_children_to_queue(obj, q, visited)

        # Should not add any children for simple objects
        assert len(q) == 0


class TestEstimateMajorSize:
    """Test get_major_size function."""

    def test_simple_object(self) -> None:
        """Test size estimation for simple objects."""
        obj = "test string"
        expected_size = sys.getsizeof(obj)
        assert get_major_size(obj) == expected_size

    def test_numpy_array(self) -> None:
        """Test size estimation for numpy arrays."""
        arr = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        expected_size = arr.nbytes
        assert get_major_size(arr) == expected_size

    def test_dict_with_values(self) -> None:
        """Test size estimation for dictionaries with values."""
        obj = {"key1": "value1", "key2": "value2"}
        # Should only include size of values, not the dict itself
        value1_size = sys.getsizeof("value1")
        value2_size = sys.getsizeof("value2")
        expected_size = value1_size + value2_size

        assert get_major_size(obj) == expected_size

    def test_list_with_items(self) -> None:
        """Test size estimation for lists with items."""
        obj = [1, 2, 3]
        # Should only include size of items, not the list itself
        item_size = sys.getsizeof(1)  # All items are same size
        expected_size = item_size * 3

        assert get_major_size(obj) == expected_size

    def test_attrs_object(self) -> None:
        """Test size estimation for attrs objects."""

        @attrs.define
        class TestClass:
            field1: int = 42
            field2: str = "test"
            stage_perf: dict[str, str] = attrs.Factory(dict)  # Should be skipped

        obj = TestClass()
        # Should include size of the attrs object itself plus field1 and field2, but not stage_perf
        obj_size = sys.getsizeof(obj)
        field1_size = sys.getsizeof(42)
        field2_size = sys.getsizeof("test")
        expected_size = obj_size + field1_size + field2_size

        assert get_major_size(obj) == expected_size

    def test_nested_structures(self) -> None:
        """Test size estimation for nested data structures."""
        inner_dict = {"inner": "value"}
        outer_dict = {"outer": inner_dict}

        # Should only include size of the inner value, not the dicts themselves
        value_size = sys.getsizeof("value")
        expected_size = value_size

        assert get_major_size(outer_dict) == expected_size

    def test_circular_reference(self) -> None:
        """Test that circular references don't cause infinite loops."""
        obj: dict[str, object] = {}
        obj["self"] = obj  # Create circular reference

        # Should not hang and should return 0 since dicts return 0
        size = get_major_size(obj)
        assert size == 0
        assert isinstance(size, int)

    def test_mixed_data_types(self) -> None:
        """Test size estimation for mixed data types."""
        arr = np.array([1, 2, 3])
        obj = {"array": arr, "string": "test", "number": 42, "nested": {"inner": "value"}}

        # Should include sizes of all components (not the containers themselves)
        arr_size = arr.nbytes
        string_size = sys.getsizeof("test")
        number_size = sys.getsizeof(42)
        inner_value_size = sys.getsizeof("value")
        expected_size = arr_size + string_size + number_size + inner_value_size

        size = get_major_size(obj)
        assert size == expected_size

    def test_empty_containers(self) -> None:
        """Test size estimation for empty containers."""
        empty_dict: dict[str, str] = {}
        empty_list: list[str] = []
        empty_tuple: tuple[()] = ()
        dict_size = get_major_size(empty_dict)
        list_size = get_major_size(empty_list)
        tuple_size = get_major_size(empty_tuple)

        assert dict_size == 0
        assert list_size == 0
        assert tuple_size == 0

    def test_large_numpy_array(self) -> None:
        """Test size estimation for large numpy arrays."""
        rng = np.random.default_rng(42)
        large_arr = rng.random((1000, 1000))
        expected_size = large_arr.nbytes

        assert get_major_size(large_arr) == expected_size

    def test_torch_tensor_integration(self) -> None:
        """Test size estimation with torch tensors."""
        # Mock torch tensor
        mock_tensor = MagicMock()
        element_size = 4
        num_elements = 100
        expected_size = element_size * num_elements
        mock_tensor.element_size.return_value = element_size
        mock_tensor.nelement.return_value = num_elements

        # Create a mock tensor type
        mock_tensor_type = type(mock_tensor)

        with patch("cosmos_curator.pipelines.video.utils.data_model.TensorType", mock_tensor_type):
            size = get_major_size(mock_tensor)
            assert size == expected_size


class TestSplitPipeTask:
    """Test SplitPipeTask initialization and methods."""

    @pytest.mark.parametrize(
        ("num_videos", "use_video_param"),
        [
            (1, True),  # Single video with video= param
            (2, False),  # Two videos with videos= param
            (3, False),  # Three videos with videos= param
        ],
    )
    def test_task_initialization(self, num_videos: int, *, use_video_param: bool) -> None:
        """Test task initialization with single and multiple videos."""
        videos = [
            Video(
                input_video=pathlib.Path(f"cam{i}.mp4"),
                metadata=VideoMetadata(duration=100.0, size=1000),
            )
            for i in range(num_videos)
        ]

        task = (
            SplitPipeTask(session_id="test-session", video=videos[0])
            if use_video_param
            else SplitPipeTask(session_id="test-session", videos=videos)
        )

        assert len(task.videos) == num_videos
        assert task.videos[0] is videos[0]
        assert task.video is videos[0]  # Property should return first video
        if num_videos > 1:
            assert task.videos[1] is videos[1]

    @pytest.mark.parametrize(
        ("num_videos", "expected_weight_multiplier"),
        [
            (1, 1.0),  # Single video: weight = video.weight
            (2, 2.0),  # Two videos: weight = sum of both
            (3, 3.0),  # Three videos: weight = sum of all three
        ],
    )
    def test_weight_aggregation(self, num_videos: int, expected_weight_multiplier: float) -> None:
        """Test weight calculation for single and multi-camera tasks."""
        videos = [
            Video(
                input_video=pathlib.Path(f"cam{i}.mp4"),
                metadata=VideoMetadata(duration=300.0, size=5000),
            )
            for i in range(num_videos)
        ]

        task = (
            SplitPipeTask(session_id="test-session", video=videos[0])
            if num_videos == 1
            else SplitPipeTask(session_id="test-session", videos=videos)
        )

        # Weight should be sum of all video weights
        expected_weight = videos[0].weight * expected_weight_multiplier
        assert task.weight == expected_weight

    @pytest.mark.parametrize(
        "num_videos",
        [1, 2, 3],
    )
    def test_get_major_size_aggregation(self, num_videos: int) -> None:
        """Test memory size calculation for single and multi-camera tasks."""
        videos = [
            Video(
                input_video=pathlib.Path(f"cam{i}.mp4"),
                encoded_data=bytes_to_numpy(f"camera {i} data".encode()),
                metadata=VideoMetadata(duration=100.0, size=1000),
            )
            for i in range(num_videos)
        ]

        task = (
            SplitPipeTask(session_id="test-session", video=videos[0])
            if num_videos == 1
            else SplitPipeTask(session_id="test-session", videos=videos)
        )

        # Size should be sum of all video sizes
        expected_size = sum(v.get_major_size() for v in videos)
        assert task.get_major_size() == expected_size
        assert task.get_major_size() > 0

    def test_fraction_property(self) -> None:
        """Test fraction property returns primary video's fraction."""
        video1 = Video(
            input_video=pathlib.Path("cam1.mp4"),
            metadata=VideoMetadata(duration=100.0, size=1000),
            clips=[],
            num_total_clips=10,
        )

        task = SplitPipeTask(session_id="test-session", videos=[video1])

        # Fraction should be calculated from all videos (just one in this case)
        # Same result: sum of clips / sum of total = video1.fraction
        assert task.fraction == video1.fraction

    @pytest.mark.parametrize(
        ("video_kwarg", "videos_kwarg", "expected_error"),
        [
            (None, None, "Must specify either 'video' or 'videos'"),
            ("video_obj", ["video_obj"], "Cannot specify both 'video' and 'videos'"),
        ],
    )
    def test_initialization_errors(
        self, video_kwarg: str | None, videos_kwarg: list[str] | None, expected_error: str
    ) -> None:
        """Test that appropriate errors are raised for invalid initialization."""
        video = Video(
            input_video=pathlib.Path("test.mp4"),
            metadata=VideoMetadata(duration=100.0, size=1000),
        )

        kwargs: dict[str, Any] = {"session_id": "test-session"}
        if video_kwarg is not None:
            kwargs["video"] = video
        if videos_kwarg is not None:
            kwargs["videos"] = [video]

        with pytest.raises(ValueError, match=expected_error):
            SplitPipeTask(**kwargs)

    def test_get_video_from_task_compatibility(self) -> None:
        """Test that get_video_from_task utility works with new structure."""
        video = Video(
            input_video=pathlib.Path("test.mp4"),
            metadata=VideoMetadata(duration=100.0, size=1000),
        )

        task = SplitPipeTask(session_id="test-session", video=video)

        # get_video_from_task should work with the property accessor
        retrieved_video = get_video_from_task(task)
        assert retrieved_video is video
        assert isinstance(retrieved_video, Video)

    @pytest.mark.parametrize(
        "stage_perf_value",
        [
            None,  # Default initialization
            {"stage1": "mock"},  # Custom stage_perf
        ],
    )
    def test_stage_perf_initialization(self, stage_perf_value: dict[str, str] | None) -> None:
        """Test stage_perf property initialization."""
        video = Video(
            input_video=pathlib.Path("test.mp4"),
            metadata=VideoMetadata(duration=100.0, size=1000),
        )

        if stage_perf_value is None:
            task = SplitPipeTask(session_id="test-session", video=video)
            assert task.stage_perf == {}
        else:
            custom_perf = {"stage1": MagicMock()}
            task = SplitPipeTask(session_id="test-session", video=video, stage_perf=custom_perf)
            assert task.stage_perf is custom_perf

    @pytest.mark.parametrize(
        ("scenario", "video1_config", "video2_config", "expected_error"),
        [
            # Aligned: should not raise
            (
                "aligned",
                {"clips": 0, "filtered": 0, "total": 10, "clip_spans": [], "filtered_spans": []},
                {"clips": 0, "filtered": 0, "total": 10, "clip_spans": [], "filtered_spans": []},
                None,
            ),
            # Check 1: Different processed counts
            (
                "different_processed",
                {"clips": 0, "filtered": 0, "total": 10, "clip_spans": [], "filtered_spans": []},
                {"clips": 1, "filtered": 0, "total": 10, "clip_spans": [(0.0, 10.0)], "filtered_spans": []},
                "processed different numbers of clips",
            ),
            # Check 2: Misaligned clip spans
            (
                "misaligned_spans",
                {"clips": 1, "filtered": 0, "total": 10, "clip_spans": [(0.0, 10.0)], "filtered_spans": []},
                {"clips": 1, "filtered": 0, "total": 10, "clip_spans": [(0.0, 15.0)], "filtered_spans": []},
                "clips at index .* have misaligned spans",
            ),
            # Misaligned filtered clip spans
            (
                "misaligned_filtered",
                {"clips": 0, "filtered": 1, "total": 10, "clip_spans": [], "filtered_spans": [(20.0, 30.0)]},
                {"clips": 0, "filtered": 1, "total": 10, "clip_spans": [], "filtered_spans": [(20.0, 35.0)]},
                "filtered clips at index .* have misaligned spans",
            ),
        ],
    )
    def test_time_alignment_validation(
        self,
        scenario: str,
        video1_config: dict[str, int | list[tuple[float, float]]],
        video2_config: dict[str, int | list[tuple[float, float]]],
        expected_error: str | None,
    ) -> None:
        """Test assert_time_alignment validates all alignment conditions."""

        def build_video(name: str, config: dict[str, int | list[tuple[float, float]]]) -> Video:
            """Build a video from config."""
            clip_spans = config["clip_spans"]
            filtered_spans = config["filtered_spans"]
            assert isinstance(clip_spans, list)
            assert isinstance(filtered_spans, list)
            clips = [Clip(uuid=uuid.uuid4(), source_video=name, span=span) for span in clip_spans]
            filtered_clips = [Clip(uuid=uuid.uuid4(), source_video=name, span=span) for span in filtered_spans]
            return Video(
                input_video=pathlib.Path(name),
                metadata=VideoMetadata(duration=100.0, size=1000),
                clips=clips,
                filtered_clips=filtered_clips,
                num_total_clips=config["total"],
            )

        video1 = build_video("cam1.mp4", video1_config)
        video2 = build_video("cam2.mp4", video2_config)
        task = SplitPipeTask(session_id="test-session", videos=[video1, video2])

        if expected_error is None:
            # Should not raise
            task.assert_time_alignment()
            # Also test fraction calculation for aligned case
            if scenario == "aligned":
                assert task.fraction == 0.0
        else:
            # Should raise with expected error
            with pytest.raises(ValueError, match=expected_error):
                task.assert_time_alignment()

            # For different_processed case, also test that fraction calculation still works
            if scenario == "different_processed":
                # video1: 0/10, video2: 1/10 → total: 1/20 = 0.05
                assert abs(task.fraction - 0.05) < 1e-9

    def test_batch_assert_time_alignment(self) -> None:
        """Test batch validation function for multiple tasks."""
        # Create aligned tasks
        task1 = SplitPipeTask(
            session_id="test-session-1",
            videos=[
                Video(
                    input_video=pathlib.Path("cam1.mp4"),
                    metadata=VideoMetadata(duration=100.0, size=1000),
                    clips=[],
                    filtered_clips=[],
                    num_total_clips=10,
                ),
                Video(
                    input_video=pathlib.Path("cam2.mp4"),
                    metadata=VideoMetadata(duration=100.0, size=1000),
                    clips=[],
                    filtered_clips=[],
                    num_total_clips=10,
                ),
            ],
        )

        task2 = SplitPipeTask(
            session_id="test-session-2",
            videos=[
                Video(
                    input_video=pathlib.Path("cam3.mp4"),
                    metadata=VideoMetadata(duration=100.0, size=1000),
                    clips=[],
                    filtered_clips=[],
                    num_total_clips=5,
                ),
                Video(
                    input_video=pathlib.Path("cam4.mp4"),
                    metadata=VideoMetadata(duration=100.0, size=1000),
                    clips=[],
                    filtered_clips=[],
                    num_total_clips=5,
                ),
            ],
        )

        # Should not raise for aligned tasks
        assert_time_alignment([task1, task2])

        # Create misaligned task (different processed counts)
        task3 = SplitPipeTask(
            session_id="test-session-3",
            videos=[
                Video(
                    input_video=pathlib.Path("cam5.mp4"),
                    metadata=VideoMetadata(duration=100.0, size=1000),
                    clips=[],
                    filtered_clips=[],
                    num_total_clips=10,
                ),
                Video(
                    input_video=pathlib.Path("cam6.mp4"),
                    metadata=VideoMetadata(duration=100.0, size=1000),
                    clips=[Clip(uuid=uuid.uuid4(), source_video="cam6.mp4", span=(0.0, 10.0))],
                    filtered_clips=[],
                    num_total_clips=10,
                ),
            ],
        )

        # Should raise when any task is misaligned
        with pytest.raises(ValueError, match=r".*processed different numbers of clips.*"):
            assert_time_alignment([task1, task2, task3])

    def test_assert_video_clip_alignment_helper(self) -> None:
        """Test assert_video_clip_alignment can be used independently."""
        # Test with aligned videos
        aligned_videos = [
            Video(
                input_video=pathlib.Path("cam1.mp4"),
                metadata=VideoMetadata(duration=100.0, size=1000),
                clips=[Clip(uuid=uuid.uuid4(), source_video="cam1.mp4", span=(0.0, 10.0))],
                filtered_clips=[],
                num_total_clips=10,
            ),
            Video(
                input_video=pathlib.Path("cam2.mp4"),
                metadata=VideoMetadata(duration=100.0, size=1000),
                clips=[Clip(uuid=uuid.uuid4(), source_video="cam2.mp4", span=(0.0, 10.0))],
                filtered_clips=[],
                num_total_clips=10,
            ),
        ]

        # Should not raise for aligned videos
        assert_video_clip_alignment(aligned_videos)

        # Test with empty list (should not raise)
        assert_video_clip_alignment([])

        # Test with misaligned videos (different processed counts)
        misaligned_videos = [
            Video(
                input_video=pathlib.Path("cam3.mp4"),
                metadata=VideoMetadata(duration=100.0, size=1000),
                clips=[],
                filtered_clips=[],
                num_total_clips=10,
            ),
            Video(
                input_video=pathlib.Path("cam4.mp4"),
                metadata=VideoMetadata(duration=100.0, size=1000),
                clips=[Clip(uuid=uuid.uuid4(), source_video="cam4.mp4", span=(0.0, 10.0))],
                filtered_clips=[],
                num_total_clips=10,
            ),
        ]

        # Should raise for misaligned videos
        with pytest.raises(ValueError, match=r".*processed different numbers of clips.*"):
            assert_video_clip_alignment(misaligned_videos)

    def test_check_clip_time_alignment_edge_cases(self) -> None:
        """Test check_clip_time_alignment with edge cases for full coverage."""
        # Test with empty list
        assert check_clip_time_alignment([]) == []

        # Test with list of empty clip lists
        assert check_clip_time_alignment([[], []]) == []

        # Test with different clip counts (should raise)
        different_counts = [
            [Clip(uuid=uuid.uuid4(), source_video="cam1.mp4", span=(0.0, 10.0))],
            [
                Clip(uuid=uuid.uuid4(), source_video="cam2.mp4", span=(0.0, 10.0)),
                Clip(uuid=uuid.uuid4(), source_video="cam2.mp4", span=(10.0, 20.0)),
            ],
        ]
        with pytest.raises(ValueError, match=r".*videos have different clip counts.*"):
            check_clip_time_alignment(different_counts)

        # Test with aligned clips
        aligned_clips = [
            [Clip(uuid=uuid.uuid4(), source_video="cam1.mp4", span=(0.0, 10.0))],
            [Clip(uuid=uuid.uuid4(), source_video="cam2.mp4", span=(0.0, 10.0))],
        ]
        assert check_clip_time_alignment(aligned_clips) == []

        # Test with misaligned clips (returns misaligned indices)
        misaligned_clips = [
            [
                Clip(uuid=uuid.uuid4(), source_video="cam1.mp4", span=(0.0, 10.0)),
                Clip(uuid=uuid.uuid4(), source_video="cam1.mp4", span=(10.0, 20.0)),
            ],
            [
                Clip(uuid=uuid.uuid4(), source_video="cam2.mp4", span=(0.0, 15.0)),  # Misaligned!
                Clip(uuid=uuid.uuid4(), source_video="cam2.mp4", span=(10.0, 20.0)),
            ],
        ]
        assert check_clip_time_alignment(misaligned_clips) == [0]


class TestVideoGetMajorSize:
    """Test Video.get_major_size() accounts for both clips and filtered_clips."""

    def test_filtered_clips_included(self) -> None:
        """get_major_size() includes memory from filtered_clips."""
        clip_data = bytes_to_numpy(b"x" * 1024)
        clip = Clip(uuid=uuid.uuid4(), source_video="v.mp4", span=(0.0, 5.0), encoded_data=clip_data)

        video_no_filtered = Video(
            input_video=pathlib.Path("v.mp4"),
            clips=[clip],
        )
        video_with_filtered = Video(
            input_video=pathlib.Path("v.mp4"),
            filtered_clips=[clip],
        )

        assert video_with_filtered.get_major_size() == video_no_filtered.get_major_size()

    def test_both_clips_and_filtered_clips(self) -> None:
        """get_major_size() includes memory from both clips and filtered_clips."""
        bytes_a = bytes_to_numpy(b"a" * 512)
        bytes_b = bytes_to_numpy(b"b" * 512)
        clip_a = Clip(uuid=uuid.uuid4(), source_video="v.mp4", span=(0.0, 5.0), encoded_data=bytes_a)
        clip_b = Clip(uuid=uuid.uuid4(), source_video="v.mp4", span=(5.0, 10.0), encoded_data=bytes_b)

        only_clips = Video(input_video=pathlib.Path("v.mp4"), clips=[clip_a])
        only_filtered = Video(input_video=pathlib.Path("v.mp4"), filtered_clips=[clip_b])
        both = Video(input_video=pathlib.Path("v.mp4"), clips=[clip_a], filtered_clips=[clip_b])

        assert both.get_major_size() > only_clips.get_major_size()
        assert both.get_major_size() > only_filtered.get_major_size()

    def test_empty_filtered_clips(self) -> None:
        """get_major_size() unchanged when filtered_clips is empty."""
        video = Video(input_video=pathlib.Path("v.mp4"))
        assert video.get_major_size() == video.get_major_size()  # idempotent sanity
        assert video.get_major_size() >= 0

    def test_propagates_through_split_pipe_task(self) -> None:
        """SplitPipeTask.get_major_size() reflects filtered_clips via Video."""
        clip_data = bytes_to_numpy(b"z" * 256)
        filtered_clip = Clip(uuid=uuid.uuid4(), source_video="v.mp4", span=(0.0, 5.0), encoded_data=clip_data)

        video_with = Video(
            input_video=pathlib.Path("v.mp4"),
            metadata=VideoMetadata(duration=100.0, size=1000),
            filtered_clips=[filtered_clip],
        )
        video_without = Video(
            input_video=pathlib.Path("v.mp4"),
            metadata=VideoMetadata(duration=100.0, size=1000),
        )

        task_with = SplitPipeTask(session_id="s1", video=video_with)
        task_without = SplitPipeTask(session_id="s2", video=video_without)

        assert task_with.get_major_size() > task_without.get_major_size()
        assert task_with.get_major_size() == video_with.get_major_size()


class TestVideoPopulateTimestamps:
    """Tests for Video.populate_timestamps() and timestamps field."""

    def test_populate_timestamps_sets_field(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """populate_timestamps() stores the result of get_video_timestamps on video.timestamps."""
        expected = np.array([0.0, 0.033, 0.066], dtype=np.float32)
        monkeypatch.setattr(
            "cosmos_curator.pipelines.video.utils.data_model.get_video_timestamps",
            lambda _data: expected,
        )
        video = Video(
            input_video=pathlib.Path("test.mp4"),
            encoded_data=bytes_to_numpy(b"fake"),
        )
        video.populate_timestamps()
        assert video.timestamps is not None
        assert np.array_equal(video.timestamps, expected)

    def test_populate_timestamps_raises_without_encoded_data(self) -> None:
        """populate_timestamps() raises ValueError when encoded_data is None."""
        video = Video(input_video=pathlib.Path("test.mp4"))
        with pytest.raises(ValueError, match="encoded_data is None"):
            video.populate_timestamps()

    @pytest.mark.parametrize(
        ("timestamps", "expect_nonzero"),
        [
            (np.array([0.0, 0.033, 0.066], dtype=np.float32), True),
            (None, False),
        ],
    )
    def test_get_major_size_with_and_without_timestamps(
        self, timestamps: npt.NDArray[np.float32] | None, *, expect_nonzero: bool
    ) -> None:
        """get_major_size() includes timestamps.nbytes when set, unaffected when None."""
        base_size = Video(input_video=pathlib.Path("test.mp4")).get_major_size()
        video = Video(input_video=pathlib.Path("test.mp4"), timestamps=timestamps)
        size = video.get_major_size()
        if expect_nonzero:
            assert timestamps is not None
            assert size >= base_size + timestamps.nbytes
        else:
            # None timestamps add nothing to size
            assert size == base_size
