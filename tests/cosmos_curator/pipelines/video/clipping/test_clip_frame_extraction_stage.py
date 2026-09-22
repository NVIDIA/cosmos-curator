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
"""Tests for the CameraSensor-only clip frame extraction stage."""

import argparse
from fractions import Fraction
from pathlib import Path
from uuid import UUID, uuid4

import numpy as np
import numpy.typing as npt
import pytest

from cosmos_curator.core.sensors.data.camera_data import CameraData, MotionVectorData, MotionVectorFrameData
from cosmos_curator.core.sensors.data.video import VideoMetadata
from cosmos_curator.core.sensors.sampling.policy import NearestTimestampPolicy
from cosmos_curator.core.sensors.sampling.spec import SamplingSpec
from cosmos_curator.core.sensors.utils.video import CpuVideoDecodeConfig
from cosmos_curator.pipelines.video.clipping.clip_frame_extraction_stages import (
    CameraSensorMotionVectorConfig,
    ClipFrameExtractionStage,
    motion_sampling_stop_ns,
)
from cosmos_curator.pipelines.video.embedding.openai_embedding_stage import OpenAIEmbeddingStage
from cosmos_curator.pipelines.video.filtering.aesthetics.aesthetic_filter_stages import AestheticFilterStage
from cosmos_curator.pipelines.video.filtering.motion.motion_filter_stages import (
    MotionFilterStage,
)
from cosmos_curator.pipelines.video.splitting_pipeline import _assemble_stages, _setup_parser
from cosmos_curator.pipelines.video.utils.data_model import Clip, SplitPipeTask, Video
from cosmos_curator.pipelines.video.utils.decoder_utils import FrameExtractionPolicy, FrameExtractionSignature
from tests.cosmos_curator.pipelines.stage_test_utils import stage_object


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


_SEC = 1_000_000_000


def test_camera_sensor_motion_config_defaults_match_legacy_frame_floor() -> None:
    """The motion config should default to the legacy 10-frame floor."""
    assert CameraSensorMotionVectorConfig().min_motion_frames == 10


@pytest.mark.parametrize(
    ("duration_s", "expected_window_s"),
    [
        # Long clips: duration*ratio dominates the 10-frame floor (matches legacy decode_for_motion).
        (20.0, 10.0),
        # Boundary (the equivalence fixture): duration*ratio == floor window == 5s.
        (10.0, 5.0),
        # Short clips: the max(10, ...) floor holds a 5s window where duration*ratio would give less.
        (6.0, 5.0),
        # Very short clip: floor window exceeds the clip, so the whole clip is sampled.
        (2.0, 2.0),
    ],
)
def test_motion_sampling_stop_ns_honors_frame_floor(duration_s: float, expected_window_s: float) -> None:
    """CameraSensor motion sampling must reproduce legacy's max(10, ...) frame-floor window."""
    start_ns = 3 * _SEC
    end_ns = start_ns + round(duration_s * _SEC)

    stop_ns = motion_sampling_stop_ns(
        start_ns,
        end_ns,
        target_fps=2.0,
        target_duration_ratio=0.5,
        min_motion_frames=10,
    )

    assert stop_ns == start_ns + round(expected_window_s * _SEC)


def test_motion_sampling_stop_ns_short_clip_widens_beyond_ratio_window() -> None:
    """On a short clip the floor must sample a wider window than duration*ratio alone."""
    start_ns = 0
    end_ns = 6 * _SEC
    ratio = 0.5

    stop_ns = motion_sampling_stop_ns(
        start_ns, end_ns, target_fps=2.0, target_duration_ratio=ratio, min_motion_frames=10
    )

    naive_ratio_window_ns = round((end_ns - start_ns) * ratio)  # 3s — the pre-fix behavior
    assert stop_ns - start_ns > naive_ratio_window_ns


def test_motion_sampling_stop_ns_zero_duration_returns_end() -> None:
    """A zero-length clip should collapse to an empty window at the clip end."""
    assert (
        motion_sampling_stop_ns(5 * _SEC, 5 * _SEC, target_fps=2.0, target_duration_ratio=0.5, min_motion_frames=10)
        == 5 * _SEC
    )


@pytest.mark.parametrize(
    ("target_duration_ratio", "min_motion_frames"),
    [(-0.5, -3), (-1.0, 0), (0.5, -10)],
)
def test_motion_sampling_stop_ns_never_returns_before_start(
    target_duration_ratio: float, min_motion_frames: int
) -> None:
    """Negative ratio/floor inputs must clamp to a non-negative window (stop within [start, end])."""
    start_ns = 2 * _SEC
    end_ns = 8 * _SEC

    stop_ns = motion_sampling_stop_ns(
        start_ns,
        end_ns,
        target_fps=2.0,
        target_duration_ratio=target_duration_ratio,
        min_motion_frames=min_motion_frames,
    )

    assert start_ns <= stop_ns <= end_ns


def _make_video_metadata(height: int = 256, width: int = 256) -> VideoMetadata:
    return VideoMetadata(
        codec_name="h264",
        has_bframes=False,
        codec_profile="High",
        container_format="mp4",
        height=height,
        width=width,
        avg_frame_rate=Fraction(30, 1),
        pix_fmt="yuv420p",
        bit_rate_bps=1_000,
    )


def _make_motion_vector_frame() -> MotionVectorFrameData:
    return MotionVectorFrameData(
        source=np.array([-1], dtype=np.int32),
        w=np.array([16], dtype=np.int32),
        h=np.array([16], dtype=np.int32),
        src_x=np.array([1], dtype=np.int32),
        src_y=np.array([2], dtype=np.int32),
        dst_x=np.array([32], dtype=np.int32),
        dst_y=np.array([32], dtype=np.int32),
        flags=np.array([0], dtype=np.int64),
        motion_x=np.array([4], dtype=np.int32),
        motion_y=np.array([0], dtype=np.int32),
        motion_scale=np.array([1], dtype=np.int32),
    )


def _make_camera_data(motion_vectors: MotionVectorData | None, *, height: int = 256, width: int = 256) -> CameraData:
    timestamps = np.array([0], dtype=np.int64)
    return CameraData(
        align_timestamps_ns=timestamps,
        sensor_timestamps_ns=timestamps,
        pts_stream=timestamps,
        frames=np.zeros((1, height, width, 3), dtype=np.uint8),
        metadata=_make_video_metadata(height=height, width=width),
        motion_vectors=motion_vectors,
    )


def test_clip_frame_extraction_stage_samples_via_camera_sensor(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, sample_clip_data: bytes
) -> None:
    """The stage should sample clip frames through CameraSensor by default."""
    task = _make_task(tmp_path, sample_clip_data)
    stage = ClipFrameExtractionStage(target_fps=[2.0])

    expected = np.full((3, 4, 4, 3), 7, dtype=np.uint8)

    def _return_expected(*_args: object, **_kwargs: object) -> npt.NDArray[np.uint8]:
        return expected

    monkeypatch.setattr(stage, "_sample_with_camera_sensor", _return_expected)

    result = stage.process_data([task])
    assert result is not None
    extracted = task.video.clips[0].extracted_frames.resolve()
    assert extracted is not None
    assert set(extracted) == {_frame_signature(2.0)}
    np.testing.assert_array_equal(extracted[_frame_signature(2.0)], expected)
    assert task.video.clips[0].extracted_frames.nbytes == expected.nbytes


def test_clip_frame_extraction_stage_lcm_signatures_and_subsampling(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, sample_clip_data: bytes
) -> None:
    """Multiple integer FPS targets should decode once at the LCM rate and stride-subsample.

    The mock returns frames tagged with the decode rate, so a non-reuse implementation that
    decoded the 1 FPS target separately would yield frames tagged ``1`` instead of the
    stride-2 view of the ``2``-tagged LCM decode.
    """
    task = _make_task(tmp_path, sample_clip_data)
    stage = ClipFrameExtractionStage(target_fps=[1, 2])

    def _fake_camera_sensor(_data: bytes | npt.NDArray[np.uint8], fps: float) -> npt.NDArray[np.uint8]:
        return np.full((4, 2, 2, 3), int(fps), dtype=np.uint8)

    monkeypatch.setattr(stage, "_sample_with_camera_sensor", _fake_camera_sensor)

    stage.process_data([task])

    extracted = task.video.clips[0].extracted_frames.resolve()
    assert extracted is not None
    assert set(extracted) == {_frame_signature(1), _frame_signature(2)}
    lcm_decode = np.full((4, 2, 2, 3), 2, dtype=np.uint8)
    np.testing.assert_array_equal(extracted[_frame_signature(2)], lcm_decode)
    np.testing.assert_array_equal(extracted[_frame_signature(1)], lcm_decode[::2])


def test_clip_frame_extraction_stage_non_integer_fps_decodes_each_target(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, sample_clip_data: bytes
) -> None:
    """Non-integer FPS targets fall outside LCM reuse and are sampled independently.

    The mock returns a distinct frame count per decode rate, so each signature reflects a
    decode at its own target FPS rather than a single shared decode.
    """
    task = _make_task(tmp_path, sample_clip_data)
    stage = ClipFrameExtractionStage(target_fps=[1.5, 2.0])

    def _fake_camera_sensor(_data: bytes | npt.NDArray[np.uint8], fps: float) -> npt.NDArray[np.uint8]:
        frame_count = round(fps * 2)
        return np.full((frame_count, 2, 2, 3), frame_count, dtype=np.uint8)

    monkeypatch.setattr(stage, "_sample_with_camera_sensor", _fake_camera_sensor)

    stage.process_data([task])

    extracted = task.video.clips[0].extracted_frames.resolve()
    assert extracted is not None
    assert set(extracted) == {_frame_signature(1.5), _frame_signature(2.0)}
    # 1.5 FPS -> 3 frames, 2.0 FPS -> 4 frames: each signature carries its own decode.
    assert extracted[_frame_signature(1.5)].shape == (3, 2, 2, 3)
    assert extracted[_frame_signature(2.0)].shape == (4, 2, 2, 3)


def test_clip_frame_extraction_stage_exports_camera_sensor_motion_vectors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, sample_clip_data: bytes
) -> None:
    """CameraSensor motion export should populate clip.decoded_motion_data."""
    task = _make_task(tmp_path, sample_clip_data)
    captured: dict[str, object] = {}
    motion_vectors = MotionVectorData(frames=(_make_motion_vector_frame(),))

    class FakeCameraSensor:
        start_ns = 0
        end_ns = 1_000_000_000

        def __init__(self, source: bytes, *, decode_config: CpuVideoDecodeConfig | None = None) -> None:
            captured["source"] = source
            captured["decode_config"] = decode_config

        def sample(self, spec: object, *, policy: NearestTimestampPolicy) -> list[CameraData]:
            captured["spec"] = spec
            captured["policy"] = policy
            return [_make_camera_data(motion_vectors)]

    monkeypatch.setattr(
        "cosmos_curator.pipelines.video.clipping.clip_frame_extraction_stages.CameraSensor",
        FakeCameraSensor,
    )
    stage = ClipFrameExtractionStage(
        target_fps=[],
        motion_vectors=CameraSensorMotionVectorConfig(target_fps=3.0, target_duration_ratio=0.25),
        num_cpus_per_worker=1.0,
    )

    stage.process_data([task])

    clip = task.video.clips[0]
    assert clip.decoded_motion_data is not None
    assert len(clip.decoded_motion_data.frames) == 1
    assert tuple(clip.decoded_motion_data.frame_size) == (256, 256, 3)
    decode_config = captured["decode_config"]
    assert isinstance(decode_config, CpuVideoDecodeConfig)
    assert decode_config.export_mvs is True
    assert decode_config.thread_count == 2
    spec = captured["spec"]
    assert isinstance(spec, SamplingSpec)
    assert isinstance(captured["policy"], NearestTimestampPolicy)
    assert spec.grid.start_ns == 0
    # This 1s clip is shorter than the min_motion_frames floor window (10 frames / 3fps = 3.3s),
    # so the whole clip is sampled (4 frames at 3fps) rather than just the first duration*ratio = 250ms.
    assert spec.grid.exclusive_end_ns == 1_166_666_666
    np.testing.assert_array_equal(
        spec.grid.timestamps_ns,
        np.array([0, 333_333_333, 666_666_666, 999_999_999], dtype=np.int64),
    )
    extracted = clip.extracted_frames.resolve()
    assert extracted == {}


def test_clip_frame_extraction_stage_marks_empty_camera_sensor_motion_vectors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, sample_clip_data: bytes
) -> None:
    """Empty sensor motion-vector payloads should map to no_motion_frames."""
    task = _make_task(tmp_path, sample_clip_data)
    motion_vectors = MotionVectorData(frames=(MotionVectorFrameData.empty(),))

    class FakeCameraSensor:
        start_ns = 0
        end_ns = 1_000_000_000

        def __init__(self, _source: bytes, *, decode_config: CpuVideoDecodeConfig | None = None) -> None:
            assert decode_config is not None
            assert decode_config.export_mvs is True

        def sample(self, _spec: object, *, policy: NearestTimestampPolicy) -> list[CameraData]:
            assert isinstance(policy, NearestTimestampPolicy)
            return [_make_camera_data(motion_vectors)]

    monkeypatch.setattr(
        "cosmos_curator.pipelines.video.clipping.clip_frame_extraction_stages.CameraSensor",
        FakeCameraSensor,
    )
    stage = ClipFrameExtractionStage(
        target_fps=[],
        motion_vectors=CameraSensorMotionVectorConfig(),
    )

    stage.process_data([task])

    clip = task.video.clips[0]
    assert clip.decoded_motion_data is None
    assert clip.errors["motion_decode"] == "no_motion_frames"


def test_clip_frame_extraction_stage_frame_failure_still_records_motion_decode(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, sample_clip_data: bytes
) -> None:
    """A frame-decode failure must still attempt motion export and record its own motion_decode error.

    The standalone MotionVectorDecodeStage set motion_decode independently before the two stages
    were fused; the fused extraction stage must not silently skip motion export when frame
    extraction fails.
    """
    task = _make_task(tmp_path, sample_clip_data)

    class FailingCameraSensor:
        def __init__(self, _source: bytes, **_kwargs: object) -> None:
            msg = "corrupt clip bytes"
            raise RuntimeError(msg)

    monkeypatch.setattr(
        "cosmos_curator.pipelines.video.clipping.clip_frame_extraction_stages.CameraSensor",
        FailingCameraSensor,
    )
    stage = ClipFrameExtractionStage(
        target_fps=[2.0],
        motion_vectors=CameraSensorMotionVectorConfig(),
    )

    stage.process_data([task])

    clip = task.video.clips[0]
    assert clip.errors["frame_extraction"] == "video_decode_failed"
    assert clip.errors["motion_decode"] == "decode_failed"
    assert clip.decoded_motion_data is None


def test_split_assemble_motion_filter_extracts_before_filtering() -> None:
    """Motion filtering should insert a single CameraSensor extraction stage before the filter."""
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
            "--motion-filter",
            "score-only",
            "--motion-decode-target-fps",
            "3.0",
            "--motion-decode-target-duration-ratio",
            "0.25",
        ]
    )

    stages = [stage_object(stage) for stage in _assemble_stages(args)]

    extraction_stages = [stage for stage in stages if isinstance(stage, ClipFrameExtractionStage)]
    assert len(extraction_stages) == 1
    motion_filter_index = next(i for i, stage in enumerate(stages) if isinstance(stage, MotionFilterStage))
    extraction_index = next(i for i, stage in enumerate(stages) if isinstance(stage, ClipFrameExtractionStage))
    assert extraction_index < motion_filter_index


def test_split_assemble_single_extraction_serves_motion_and_aesthetics() -> None:
    """One extraction stage should coordinate motion export and aesthetic frame sampling."""
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
            "--motion-filter",
            "score-only",
            "--aesthetic-threshold",
            "3.5",
        ]
    )

    stages = [stage_object(stage) for stage in _assemble_stages(args)]

    extraction_stages = [stage for stage in stages if isinstance(stage, ClipFrameExtractionStage)]
    # Exactly one extraction stage serves both the motion export and the aesthetics consumer.
    assert len(extraction_stages) == 1
    motion_filter_index = next(i for i, stage in enumerate(stages) if isinstance(stage, MotionFilterStage))
    extraction_index = next(i for i, stage in enumerate(stages) if isinstance(stage, ClipFrameExtractionStage))
    assert extraction_index < motion_filter_index


def test_split_assemble_no_motion_filter_when_motion_disabled() -> None:
    """No motion filter stage should be added when motion filtering is disabled (the default)."""
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
        ]
    )

    stages = [stage_object(stage) for stage in _assemble_stages(args)]

    assert not any(isinstance(stage, MotionFilterStage) for stage in stages)


@pytest.mark.parametrize(
    ("sampling_args", "expected_target_fps", "signatures_are_shared"),
    [
        ([], [1.0, 2.0], False),
        (["--embedding-sampling-fps", "1"], [1.0, 1.0], True),
        (["--embedding-sampling-fps", "1.5"], [1.0, 1.5], False),
    ],
    ids=["default-rate", "shared-one-fps-rate", "fractional-rate"],
)
def test_split_assemble_aesthetics_configures_frame_ownership_for_embedding(
    sampling_args: list[str],
    expected_target_fps: list[float | int],
    *,
    signatures_are_shared: bool,
) -> None:
    """Aesthetics should precede embedding and preserve only an intentionally shared frame entry."""
    input_path = Path.cwd() / "tmp-input"
    output_path = Path.cwd() / "tmp-output"
    args = _parser().parse_args(
        [
            "--input-video-path",
            input_path.as_posix(),
            "--output-clip-path",
            output_path.as_posix(),
            "--no-generate-captions",
            "--embedding-algorithm",
            "openai",
            "--aesthetic-threshold",
            "3.5",
            *sampling_args,
        ]
    )

    stages = [stage_object(stage) for stage in _assemble_stages(args)]

    extraction_stage = next(stage for stage in stages if isinstance(stage, ClipFrameExtractionStage))
    aesthetic_stage = next(stage for stage in stages if isinstance(stage, AestheticFilterStage))
    embedding_stage = next(stage for stage in stages if isinstance(stage, OpenAIEmbeddingStage))
    assert extraction_stage._target_fps == expected_target_fps
    assert aesthetic_stage._preserve_extracted_frames is signatures_are_shared
    assert (
        aesthetic_stage._frame_extraction_signature == embedding_stage._frame_extraction_signature
    ) is signatures_are_shared
    assert stages.index(extraction_stage) < stages.index(aesthetic_stage) < stages.index(embedding_stage)


def test_split_assemble_rejects_distinct_rates_that_alias_aesthetics_signature() -> None:
    """A shared dictionary key cannot represent distinct aesthetics and embedding sampling rates."""
    input_path = Path.cwd() / "tmp-input"
    output_path = Path.cwd() / "tmp-output"
    args = _parser().parse_args(
        [
            "--input-video-path",
            input_path.as_posix(),
            "--output-clip-path",
            output_path.as_posix(),
            "--no-generate-captions",
            "--embedding-algorithm",
            "openai",
            "--aesthetic-threshold",
            "3.5",
            "--embedding-sampling-fps",
            "1.0005",
        ]
    )

    with pytest.raises(ValueError, match="both map to the same frame extraction signature"):
        _assemble_stages(args)


def test_split_assemble_accepts_signature_alias_rate_without_aesthetics() -> None:
    """Signature collision validation should apply only when another consumer requests the same key."""
    input_path = Path.cwd() / "tmp-input"
    output_path = Path.cwd() / "tmp-output"
    args = _parser().parse_args(
        [
            "--input-video-path",
            input_path.as_posix(),
            "--output-clip-path",
            output_path.as_posix(),
            "--no-generate-captions",
            "--embedding-algorithm",
            "openai",
            "--embedding-sampling-fps",
            "1.0005",
        ]
    )

    stages = [stage_object(stage) for stage in _assemble_stages(args)]

    extraction_stage = next(stage for stage in stages if isinstance(stage, ClipFrameExtractionStage))
    embedding_stage = next(stage for stage in stages if isinstance(stage, OpenAIEmbeddingStage))
    assert extraction_stage._target_fps == [1.0005]
    assert embedding_stage._frame_extraction_signature == _frame_signature(1.0005)
    assert not any(isinstance(stage, AestheticFilterStage) for stage in stages)


def test_split_assemble_extraction_without_motion_when_aesthetics_only() -> None:
    """With motion filtering off but aesthetics on, a single extraction stage (no motion export) runs."""
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
        ]
    )

    stages = [stage_object(stage) for stage in _assemble_stages(args)]

    extraction_stages = [stage for stage in stages if isinstance(stage, ClipFrameExtractionStage)]
    aesthetic_stage = next(stage for stage in stages if isinstance(stage, AestheticFilterStage))
    assert len(extraction_stages) == 1
    assert extraction_stages[0]._target_fps == [1.0]
    # Motion filtering is off, so the shared extraction must not export motion vectors.
    assert extraction_stages[0]._motion_vector_config is None
    assert aesthetic_stage._preserve_extracted_frames is False
    assert not any(isinstance(stage, MotionFilterStage) for stage in stages)


@pytest.mark.env("default")
def test_clip_frame_extraction_stage_camera_sensor_real_fixture(tmp_path: Path, sample_clip_data: bytes) -> None:
    """The CameraSensor path should produce non-empty uint8 frames on the real clip fixture."""
    task = _make_task(tmp_path, sample_clip_data)
    stage = ClipFrameExtractionStage(target_fps=[2.0])

    stage.process_data([task])

    extracted = task.video.clips[0].extracted_frames.resolve()
    assert extracted is not None
    assert set(extracted) == {_frame_signature(2.0)}
    frames = extracted[_frame_signature(2.0)]
    assert frames.dtype == np.uint8
    assert frames.ndim == 4
    assert len(frames) > 0


@pytest.mark.env("default")
def test_clip_frame_extraction_stage_camera_sensor_lcm_stride_real_fixture(
    tmp_path: Path, sample_clip_data: bytes
) -> None:
    """On the real fixture, the 1 FPS target should be the stride-2 view of the 2 FPS decode."""
    task = _make_task(tmp_path, sample_clip_data)
    stage = ClipFrameExtractionStage(target_fps=[1, 2])

    stage.process_data([task])

    extracted = task.video.clips[0].extracted_frames.resolve()
    assert extracted is not None
    assert set(extracted) == {_frame_signature(1), _frame_signature(2)}
    frames_2fps = extracted[_frame_signature(2)]
    frames_1fps = extracted[_frame_signature(1)]
    assert frames_2fps.dtype == np.uint8
    assert frames_1fps.dtype == np.uint8
    assert len(frames_2fps) > 0
    np.testing.assert_array_equal(frames_1fps, frames_2fps[::2])


@pytest.mark.env("default")
def test_clip_frame_extraction_stage_camera_sensor_applies_target_res(tmp_path: Path, sample_clip_data: bytes) -> None:
    """The CameraSensor path should resize frames to target_res on the real clip fixture."""
    task = _make_task(tmp_path, sample_clip_data)
    stage = ClipFrameExtractionStage(target_fps=[2.0], target_res=(8, 8))

    stage.process_data([task])

    extracted = task.video.clips[0].extracted_frames.resolve()
    assert extracted is not None
    assert set(extracted) == {_frame_signature(2.0)}
    frames = extracted[_frame_signature(2.0)]
    assert frames.dtype == np.uint8
    assert len(frames) > 0
    assert frames.shape[1:] == (8, 8, 3)
