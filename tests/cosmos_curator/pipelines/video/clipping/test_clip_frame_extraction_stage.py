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
"""Tests for clip frame extraction stage backends and wiring."""

import argparse
from pathlib import Path
from uuid import UUID, uuid4

import numpy as np
import numpy.typing as npt
import pytest

from cosmos_curator.core.utils.misc.stage_compare import run_stage_compare
from cosmos_curator.core.utils.misc.stage_replay import DirectStageExecutor, PickleTaskSerializer
from cosmos_curator.pipelines.video.clipping.clip_frame_extraction_stages import ClipFrameExtractionStage
from cosmos_curator.pipelines.video.clipping.clipping_builders import (
    FrameExtractionConfig,
    build_frame_extraction_stages,
)
from cosmos_curator.pipelines.video.splitting_pipeline import _assemble_stages, _setup_parser
from cosmos_curator.pipelines.video.utils.data_model import Clip, SplitPipeTask, Video
from cosmos_curator.pipelines.video.utils.decoder_utils import FrameExtractionPolicy, FrameExtractionSignature


def _make_task(tmp_path: Path, clip_bytes: bytes, *, clip_uuid: UUID | None = None) -> SplitPipeTask:
    """Create a minimal split task with one real clip fixture for stage testing."""
    clip = Clip(
        uuid=clip_uuid or uuid4(),
        source_video="source.mp4",
        span=(0.0, 10.0),
        encoded_data=np.frombuffer(clip_bytes, dtype=np.uint8).copy(),
    )
    video = Video(
        input_video=tmp_path / "video.mp4",
        clips=[clip],
    )
    return SplitPipeTask(session_id="session-a", video=video)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    _setup_parser(parser)
    return parser


def _frame_signature(fps: float) -> str:
    return FrameExtractionSignature(
        extraction_policy=FrameExtractionPolicy.sequence,
        target_fps=fps,
    ).to_str()


def _raise_unexpected_backend(*_args: object, **_kwargs: object) -> npt.NDArray[np.uint8]:
    msg = "unexpected backend invocation"
    raise AssertionError(msg)


def test_clip_frame_extraction_stage_defaults_to_extract_frames_backend(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, sample_clip_data: bytes
) -> None:
    """Default mode should use the existing extract_frames backend."""
    task = _make_task(tmp_path, sample_clip_data)
    stage = ClipFrameExtractionStage(target_fps=[2.0])

    expected = np.ones((2, 4, 4, 3), dtype=np.uint8)

    def _return_expected(*_args: object, **_kwargs: object) -> npt.NDArray[np.uint8]:
        return expected

    monkeypatch.setattr(stage, "_extract_frames_default", _return_expected)
    monkeypatch.setattr(stage, "_sample_with_camera_sensor", _raise_unexpected_backend)

    result = stage.process_data([task])
    assert result is not None
    extracted = task.video.clips[0].extracted_frames.resolve()
    assert extracted is not None
    assert set(extracted) == {_frame_signature(2.0)}
    np.testing.assert_array_equal(extracted[_frame_signature(2.0)], expected)
    assert task.video.clips[0].extracted_frames.nbytes == expected.nbytes


def test_clip_frame_extraction_stage_camera_sensor_backend_selected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, sample_clip_data: bytes
) -> None:
    """Explicit camera_sensor mode should route through the alternate backend."""
    task = _make_task(tmp_path, sample_clip_data)
    stage = ClipFrameExtractionStage(target_fps=[2.0], decoder_mode=ClipFrameExtractionStage.CAMERA_SENSOR_DECODER_MODE)

    expected = np.full((3, 4, 4, 3), 7, dtype=np.uint8)

    def _return_expected(*_args: object, **_kwargs: object) -> npt.NDArray[np.uint8]:
        return expected

    monkeypatch.setattr(stage, "_extract_frames_default", _raise_unexpected_backend)
    monkeypatch.setattr(stage, "_sample_with_camera_sensor", _return_expected)

    stage.process_data([task])
    extracted = task.video.clips[0].extracted_frames.resolve()
    assert extracted is not None
    np.testing.assert_array_equal(extracted[_frame_signature(2.0)], expected)


def test_clip_frame_extraction_stage_lcm_signatures_and_subsampling(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, sample_clip_data: bytes
) -> None:
    """Multiple integer FPS targets should preserve LCM decode reuse semantics."""
    task = _make_task(tmp_path, sample_clip_data)
    stage = ClipFrameExtractionStage(target_fps=[1, 2])

    lcm_frames = np.arange(4 * 2 * 2 * 3, dtype=np.uint8).reshape(4, 2, 2, 3)

    def _fake_default(
        _data: bytes | npt.NDArray[np.uint8], policy: FrameExtractionPolicy, fps: float
    ) -> npt.NDArray[np.uint8]:
        assert policy is FrameExtractionPolicy.sequence
        assert fps == 2
        return lcm_frames

    monkeypatch.setattr(stage, "_extract_frames_default", _fake_default)

    stage.process_data([task])
    extracted = task.video.clips[0].extracted_frames.resolve()
    assert extracted is not None
    assert set(extracted) == {_frame_signature(1), _frame_signature(2)}
    np.testing.assert_array_equal(extracted[_frame_signature(2)], lcm_frames)
    np.testing.assert_array_equal(extracted[_frame_signature(1)], lcm_frames[::2])


def test_clip_frame_extraction_stage_camera_sensor_decodes_each_target_fps(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, sample_clip_data: bytes
) -> None:
    """CameraSensor mode should sample each requested FPS independently instead of using LCM striding."""
    task = _make_task(tmp_path, sample_clip_data)
    stage = ClipFrameExtractionStage(
        target_fps=[1, 2],
        decoder_mode=ClipFrameExtractionStage.CAMERA_SENSOR_DECODER_MODE,
    )
    called_fps: list[float] = []

    def _fake_camera_sensor(_data: bytes | npt.NDArray[np.uint8], fps: float) -> npt.NDArray[np.uint8]:
        called_fps.append(fps)
        frame_count = int(fps)
        return np.full((frame_count, 2, 2, 3), frame_count, dtype=np.uint8)

    monkeypatch.setattr(stage, "_sample_with_camera_sensor", _fake_camera_sensor)
    monkeypatch.setattr(stage, "_extract_frames_default", _raise_unexpected_backend)

    stage.process_data([task])

    assert called_fps == [1, 2]
    extracted = task.video.clips[0].extracted_frames.resolve()
    assert extracted is not None
    assert set(extracted) == {_frame_signature(1), _frame_signature(2)}
    assert extracted[_frame_signature(1)].shape == (1, 2, 2, 3)
    assert extracted[_frame_signature(2)].shape == (2, 2, 2, 3)


def test_clip_frame_extraction_stage_camera_sensor_rejects_non_sequence_policy(sample_clip_data: bytes) -> None:
    """CameraSensor mode should preserve the explicit sequence-only policy guard."""
    stage = ClipFrameExtractionStage(
        extraction_policies=(FrameExtractionPolicy.middle,),
        target_fps=[2.0],
        decoder_mode=ClipFrameExtractionStage.CAMERA_SENSOR_DECODER_MODE,
    )

    with pytest.raises(NotImplementedError, match="CameraSensor clip frame extraction only supports"):
        stage._extract_frames_for_policy(sample_clip_data, FrameExtractionPolicy.middle)


def test_build_frame_extraction_stages_passes_decoder_mode() -> None:
    """Builder should thread decoder_mode into ClipFrameExtractionStage."""
    stages = build_frame_extraction_stages(
        FrameExtractionConfig(
            target_fps=[2],
            decoder_mode=ClipFrameExtractionStage.CAMERA_SENSOR_DECODER_MODE,
        )
    )
    assert len(stages) == 1
    stage = stages[0]
    assert isinstance(stage, ClipFrameExtractionStage)
    assert stage._decoder_mode == ClipFrameExtractionStage.CAMERA_SENSOR_DECODER_MODE


def test_split_parser_and_assemble_stages_pass_clip_extraction_decoder_mode() -> None:
    """Split CLI parser and stage assembly should preserve the selected decoder mode."""
    parser = _parser()
    input_path = Path.cwd() / "tmp-input"
    output_path = Path.cwd() / "tmp-output"
    args = parser.parse_args(
        [
            "--input-video-path",
            input_path.as_posix(),
            "--output-clip-path",
            output_path.as_posix(),
            "--no-generate-embeddings",
            "--aesthetic-threshold",
            "3.5",
            "--clip-extraction-decoder-mode",
            "camera_sensor",
        ]
    )
    assert args.clip_extraction_decoder_mode == "camera_sensor"

    stages = _assemble_stages(args)
    extraction_stages = [stage for stage in stages if isinstance(stage, ClipFrameExtractionStage)]
    assert len(extraction_stages) == 1
    assert extraction_stages[0]._decoder_mode == ClipFrameExtractionStage.CAMERA_SENSOR_DECODER_MODE


@pytest.mark.env("cosmos-curator")
def test_clip_frame_extraction_stage_compare_camera_sensor_matches_default(
    tmp_path: Path, sample_clip_data: bytes
) -> None:
    """Stage compare should validate CameraSensor output against default output on the real clip fixture."""
    serializer = PickleTaskSerializer()
    input_dir = tmp_path / "tasks" / "ClipFrameExtractionStage-input"
    golden_dir = tmp_path / "tasks" / "ClipFrameExtractionStage-golden"
    input_dir.mkdir(parents=True)
    golden_dir.mkdir(parents=True)

    clip_uuid = uuid4()
    default_task = _make_task(tmp_path, sample_clip_data, clip_uuid=clip_uuid)
    camera_task = _make_task(tmp_path, sample_clip_data, clip_uuid=clip_uuid)

    default_stage = ClipFrameExtractionStage(target_fps=[2.0])
    camera_stage = ClipFrameExtractionStage(
        target_fps=[2.0],
        decoder_mode=ClipFrameExtractionStage.CAMERA_SENSOR_DECODER_MODE,
    )

    serializer.save(input_dir / "batch_000.task.pkl", [camera_task])
    serializer.save(golden_dir / "batch_000.task.pkl", [default_stage.process_data([default_task])[0]])

    report_path = tmp_path / "compare" / "ClipFrameExtractionStage" / "report.json"
    result = run_stage_compare(
        [camera_stage],
        input_dir,
        golden_dir,
        atol=0.0,
        limit=0,
        pass_threshold=1.0,
        report_path=report_path,
        executor=DirectStageExecutor(),
        serializer=serializer,
        backend="serial",
    )

    assert result.passed
    assert report_path.exists()


@pytest.mark.env("cosmos-curator")
def test_clip_frame_extraction_camera_sensor_matches_default_backend_exactly(
    tmp_path: Path, sample_clip_data: bytes
) -> None:
    """CameraSensor backend should exactly match extracted_frames from the default backend at 2 FPS."""
    clip_uuid = uuid4()

    default_task = _make_task(tmp_path, sample_clip_data, clip_uuid=clip_uuid)
    camera_task = _make_task(tmp_path, sample_clip_data, clip_uuid=clip_uuid)

    default_stage = ClipFrameExtractionStage(
        extraction_policies=(FrameExtractionPolicy.sequence,),
        target_fps=[2.0],
        decoder_mode=ClipFrameExtractionStage.DEFAULT_DECODER_MODE,
    )
    camera_stage = ClipFrameExtractionStage(
        extraction_policies=(FrameExtractionPolicy.sequence,),
        target_fps=[2.0],
        decoder_mode=ClipFrameExtractionStage.CAMERA_SENSOR_DECODER_MODE,
    )

    default_stage.process_data([default_task])
    camera_stage.process_data([camera_task])

    golden = default_task.video.clips[0].extracted_frames.resolve()
    candidate = camera_task.video.clips[0].extracted_frames.resolve()

    assert golden is not None
    assert candidate is not None
    assert set(golden) == {_frame_signature(2.0)}
    assert set(golden) == set(candidate)

    for key in sorted(golden):
        assert golden[key].dtype == np.uint8
        assert candidate[key].dtype == np.uint8
        assert golden[key].shape == candidate[key].shape
        np.testing.assert_array_equal(golden[key], candidate[key])


@pytest.mark.env("cosmos-curator")
def test_clip_frame_extraction_camera_sensor_matches_default_backend_exactly_multi_fps(
    tmp_path: Path, sample_clip_data: bytes
) -> None:
    """CameraSensor backend should exactly match extracted_frames for the shared multi-FPS path."""
    clip_uuid = uuid4()

    default_task = _make_task(tmp_path, sample_clip_data, clip_uuid=clip_uuid)
    camera_task = _make_task(tmp_path, sample_clip_data, clip_uuid=clip_uuid)

    default_stage = ClipFrameExtractionStage(
        extraction_policies=(FrameExtractionPolicy.sequence,),
        target_fps=[1, 2],
        decoder_mode=ClipFrameExtractionStage.DEFAULT_DECODER_MODE,
    )
    camera_stage = ClipFrameExtractionStage(
        extraction_policies=(FrameExtractionPolicy.sequence,),
        target_fps=[1, 2],
        decoder_mode=ClipFrameExtractionStage.CAMERA_SENSOR_DECODER_MODE,
    )

    default_stage.process_data([default_task])
    camera_stage.process_data([camera_task])

    golden = default_task.video.clips[0].extracted_frames.resolve()
    candidate = camera_task.video.clips[0].extracted_frames.resolve()

    assert golden is not None
    assert candidate is not None
    assert set(golden) == {_frame_signature(1), _frame_signature(2)}
    assert set(golden) == set(candidate)

    for key in sorted(golden):
        assert golden[key].dtype == np.uint8
        assert candidate[key].dtype == np.uint8
        assert golden[key].shape == candidate[key].shape
        np.testing.assert_array_equal(golden[key], candidate[key])


@pytest.mark.env("cosmos-curator")
def test_clip_frame_extraction_camera_sensor_matches_default_backend_exactly_with_resize(
    tmp_path: Path, sample_clip_data: bytes
) -> None:
    """CameraSensor backend should exactly match resized extracted_frames from the default backend."""
    clip_uuid = uuid4()

    default_task = _make_task(tmp_path, sample_clip_data, clip_uuid=clip_uuid)
    camera_task = _make_task(tmp_path, sample_clip_data, clip_uuid=clip_uuid)

    default_stage = ClipFrameExtractionStage(
        extraction_policies=(FrameExtractionPolicy.sequence,),
        target_fps=[2.0],
        target_res=(8, 8),
        decoder_mode=ClipFrameExtractionStage.DEFAULT_DECODER_MODE,
    )
    camera_stage = ClipFrameExtractionStage(
        extraction_policies=(FrameExtractionPolicy.sequence,),
        target_fps=[2.0],
        target_res=(8, 8),
        decoder_mode=ClipFrameExtractionStage.CAMERA_SENSOR_DECODER_MODE,
    )

    default_stage.process_data([default_task])
    camera_stage.process_data([camera_task])

    golden = default_task.video.clips[0].extracted_frames.resolve()
    candidate = camera_task.video.clips[0].extracted_frames.resolve()

    assert golden is not None
    assert candidate is not None
    assert set(golden) == {_frame_signature(2.0)}
    assert set(golden) == set(candidate)

    for key in sorted(golden):
        assert golden[key].dtype == np.uint8
        assert candidate[key].dtype == np.uint8
        assert golden[key].shape == candidate[key].shape
        assert golden[key].shape[1:] == (8, 8, 3)
        np.testing.assert_array_equal(golden[key], candidate[key])
