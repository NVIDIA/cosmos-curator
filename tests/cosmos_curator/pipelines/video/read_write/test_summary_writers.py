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
"""Tests for split summary writing."""

import pathlib
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

from cosmos_curator.pipelines.video.read_write import summary_writers
from cosmos_curator.pipelines.video.read_write.summary_writers import write_split_summary
from cosmos_curator.pipelines.video.utils.data_model import CaptionQualityStats, SplitPipeTask, Video


def _make_video(*, was_remuxed: bool, clip_chunk_index: int) -> Video:
    v = Video(input_video=pathlib.Path("test.ts"))
    v.was_remuxed = was_remuxed
    v.clip_chunk_index = clip_chunk_index
    return v


def _caption_quality_chunk() -> dict[str, Any]:
    stats = CaptionQualityStats()
    stats.caption_windows_checked = 1
    stats.caption_status_counts["success"] = 1
    # Some tests mutate this serialized form to exercise invalid chunk inputs.
    return stats.to_dict()


def _processed_video_data(chunk_stats: object) -> dict[str, summary_writers.ProcessedVideoMetadata]:
    return {
        "video.mp4": summary_writers.ProcessedVideoMetadata(
            video_metadata={"num_clip_chunks": 1},
            clip_chunks=[{"caption_quality_stats": chunk_stats}],
        )
    }


def _patch_caption_quality_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[list[dict[str, Any]], list[str]]:
    writes: list[dict[str, Any]] = []
    warnings: list[str] = []

    def fake_write_json(data: dict[str, Any], *_args: object, **_kwargs: object) -> None:
        writes.append(data)

    def fake_warning(message: str, *args: object) -> None:
        warnings.append(message.format(*args) if args else message)

    monkeypatch.setattr(summary_writers, "write_json", fake_write_json)
    monkeypatch.setattr(summary_writers, "logger", SimpleNamespace(info=lambda *_args: None, warning=fake_warning))
    return writes, warnings


def _caption_quality_options(
    *,
    generate_captions: bool = True,
    caption_quality_stats_enabled: bool = True,
    caption_models: list[str] | None = None,
    multi_cam: bool = False,
) -> summary_writers.CaptionQualityStatsWriteOptions:
    return summary_writers.CaptionQualityStatsWriteOptions(
        generate_captions=generate_captions,
        caption_quality_stats_enabled=caption_quality_stats_enabled,
        caption_models=caption_models if caption_models is not None else ["qwen"],
        multi_cam=multi_cam,
    )


def test_num_remuxed_videos_no_double_count() -> None:
    """clip_chunk_index == 0 guard prevents double-counting chunked videos.

    Two Video objects represent the same source video split into two chunks.
    Both have was_remuxed=True, but only the chunk-0 object should be counted,
    so num_remuxed_videos must be 1, not 2.
    """
    chunk0 = _make_video(was_remuxed=True, clip_chunk_index=0)
    chunk1 = _make_video(was_remuxed=True, clip_chunk_index=1)

    task0 = SplitPipeTask(session_id="s", video=chunk0)
    task1 = SplitPipeTask(session_id="s", video=chunk1)

    with patch("cosmos_curator.pipelines.video.read_write.summary_writers._write_split_result_summary") as mock_write:
        write_split_summary(
            input_path="/in",
            input_videos_relative=["test.ts"],
            num_input_videos_selected=1,
            output_path="/out",
            output_s3_profile_name="default",
            output_tasks=[task0, task1],
            embedding_algorithm="internvideo2",
            limit=0,
        )

    mock_write.assert_called_once()
    _, kwargs = mock_write.call_args
    assert kwargs["num_remuxed_videos"] == 1, (
        f"Expected 1 remuxed video (chunk-0 only), got {kwargs['num_remuxed_videos']}"
    )


def test_caption_quality_stats_writes_artifact_with_schema_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    """Successful aggregation writes root artifact schema fields and counters."""
    writes, warnings = _patch_caption_quality_outputs(monkeypatch)

    summary_writers._write_caption_quality_stats(
        output_path="/output",
        client_output=None,
        all_video_data=_processed_video_data(_caption_quality_chunk()),
        options=_caption_quality_options(),
    )

    assert warnings == []
    assert writes == [
        {
            "schema_version": 1,
            "pipeline": "split_video_pipeline",
            "caption_windows_checked": 1,
            "caption_status_counts": {
                "success": 1,
                "truncated": 0,
                "blocked": 0,
                "error": 0,
                "skipped": 0,
            },
            "caption_failure_reason_counts": {
                "exception": 0,
                "timeout": 0,
            },
            "caption_quality_flags_evaluated_count": 0,
            "caption_quality_flag_counts": {
                "flag_length_outlier": 0,
                "flag_repetition": 0,
                "flag_near_duplicate": 0,
            },
            "empty_caption_count": 0,
            "sentinel_caption_count": 0,
        }
    ]


def test_caption_quality_stats_ignores_unprocessed_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unprocessed inputs match summary.json semantics and contribute no counters."""
    writes, warnings = _patch_caption_quality_outputs(monkeypatch)
    all_video_data = _processed_video_data(_caption_quality_chunk())
    all_video_data["unprocessed.mp4"] = summary_writers.ProcessedVideoMetadata()

    summary_writers._write_caption_quality_stats(
        output_path="/output",
        client_output=None,
        all_video_data=all_video_data,
        options=_caption_quality_options(),
    )

    assert warnings == []
    assert writes[0]["caption_windows_checked"] == 1
    assert writes[0]["caption_status_counts"]["success"] == 1


def test_caption_quality_stats_omits_for_single_model_violation(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-single subject caption model set omits with a warning."""
    writes, warnings = _patch_caption_quality_outputs(monkeypatch)

    summary_writers._write_caption_quality_stats(
        output_path="/output",
        client_output=None,
        all_video_data=_processed_video_data(_caption_quality_chunk()),
        options=_caption_quality_options(caption_models=["qwen", "openai"]),
    )

    assert writes == []
    assert warnings == ["Skipping caption_quality_stats.json: expected exactly one caption model, got 2"]


def test_caption_quality_stats_omits_for_missing_chunk_aggregate(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing per-chunk aggregate omits with a warning."""
    writes, warnings = _patch_caption_quality_outputs(monkeypatch)
    all_video_data = {
        "video.mp4": summary_writers.ProcessedVideoMetadata(
            video_metadata={"num_clip_chunks": 1},
            clip_chunks=[{}],
        )
    }

    summary_writers._write_caption_quality_stats(
        output_path="/output",
        client_output=None,
        all_video_data=all_video_data,
        options=_caption_quality_options(),
    )

    assert writes == []
    assert warnings == ["Skipping caption_quality_stats.json: video.mp4 clip chunk is missing caption_quality_stats"]


def test_caption_quality_stats_omits_for_unmapped_failure_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unknown failure-reason keys are rejected by summary parsing."""
    writes, warnings = _patch_caption_quality_outputs(monkeypatch)
    chunk = _caption_quality_chunk()
    chunk["caption_failure_reason_counts"]["oom"] = 1

    summary_writers._write_caption_quality_stats(
        output_path="/output",
        client_output=None,
        all_video_data=_processed_video_data(chunk),
        options=_caption_quality_options(),
    )

    assert writes == []
    assert warnings == [
        "Skipping caption_quality_stats.json: video.mp4 clip chunk has invalid caption_quality_stats: "
        "caption_failure_reason_counts has unknown keys: oom"
    ]


def test_caption_quality_stats_omits_for_invariant_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Invalid aggregate invariants omit with a warning."""
    writes, warnings = _patch_caption_quality_outputs(monkeypatch)
    chunk = _caption_quality_chunk()
    chunk["empty_caption_count"] = 2

    summary_writers._write_caption_quality_stats(
        output_path="/output",
        client_output=None,
        all_video_data=_processed_video_data(chunk),
        options=_caption_quality_options(),
    )

    assert writes == []
    assert warnings == [
        "Skipping caption_quality_stats.json: video.mp4 clip chunk has invalid caption_quality_stats: "
        "empty and sentinel counts must not exceed OK status count"
    ]


def test_caption_quality_stats_omits_for_unmapped_status(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unknown writer-side statuses are rejected by summary parsing."""
    writes, warnings = _patch_caption_quality_outputs(monkeypatch)
    chunk = _caption_quality_chunk()
    chunk["caption_status_counts"]["partial"] = 1

    summary_writers._write_caption_quality_stats(
        output_path="/output",
        client_output=None,
        all_video_data=_processed_video_data(chunk),
        options=_caption_quality_options(),
    )

    assert writes == []
    assert warnings == [
        "Skipping caption_quality_stats.json: video.mp4 clip chunk has invalid caption_quality_stats: "
        "caption_status_counts has unknown keys: partial"
    ]


def test_caption_quality_stats_omits_for_unmapped_quality_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unknown quality-flag keys are rejected by summary parsing."""
    writes, warnings = _patch_caption_quality_outputs(monkeypatch)
    chunk = _caption_quality_chunk()
    chunk["caption_quality_flag_counts"]["flag_new_metric"] = 1

    summary_writers._write_caption_quality_stats(
        output_path="/output",
        client_output=None,
        all_video_data=_processed_video_data(chunk),
        options=_caption_quality_options(),
    )

    assert writes == []
    assert warnings == [
        "Skipping caption_quality_stats.json: video.mp4 clip chunk has invalid caption_quality_stats: "
        "caption_quality_flag_counts has unknown keys: flag_new_metric"
    ]


def test_caption_quality_stats_omits_silently_when_captions_disabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    """Caption disabled runs clear stale stats without warning or writing."""
    writes, warnings = _patch_caption_quality_outputs(monkeypatch)
    stale_path = tmp_path / "caption_quality_stats.json"
    stale_path.write_text("{}")

    summary_writers._write_caption_quality_stats(
        output_path=tmp_path.as_posix(),
        client_output=None,
        all_video_data=_processed_video_data(_caption_quality_chunk()),
        options=_caption_quality_options(generate_captions=False),
    )

    assert writes == []
    assert warnings == []
    assert not stale_path.exists()


def test_caption_quality_stats_omits_silently_when_artifact_disabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    """Explicitly disabled artifact clears stale stats without warning or writing."""
    writes, warnings = _patch_caption_quality_outputs(monkeypatch)
    stale_path = tmp_path / "caption_quality_stats.json"
    stale_path.write_text("{}")

    summary_writers._write_caption_quality_stats(
        output_path=tmp_path.as_posix(),
        client_output=None,
        all_video_data=_processed_video_data(_caption_quality_chunk()),
        options=_caption_quality_options(caption_quality_stats_enabled=False),
    )

    assert writes == []
    assert warnings == []
    assert not stale_path.exists()


def test_caption_quality_stats_omits_for_multicam(monkeypatch: pytest.MonkeyPatch) -> None:
    """Multi-camera summaries do not emit a healthy zero stats artifact."""
    writes, warnings = _patch_caption_quality_outputs(monkeypatch)

    summary_writers._write_caption_quality_stats(
        output_path="/output",
        client_output=None,
        all_video_data={},
        options=_caption_quality_options(multi_cam=True),
    )

    assert writes == []
    assert warnings == [
        "Skipping caption_quality_stats.json: multi-camera caption quality aggregation is not supported"
    ]


def test_split_result_summary_uses_caption_windows_for_token_averages(monkeypatch: pytest.MonkeyPatch) -> None:
    """Token averages are per caption window, not per clip with any caption."""
    captured_summary: dict[str, Any] = {}
    info_calls: list[tuple[str, tuple[object, ...]]] = []

    def fake_write_json(data: dict[str, Any], *_args: object, **_kwargs: object) -> None:
        captured_summary.update(data)

    def fake_info(message: str, *args: object) -> None:
        info_calls.append((message, args))

    monkeypatch.setattr(
        summary_writers,
        "logger",
        SimpleNamespace(info=fake_info),
    )
    monkeypatch.setattr(summary_writers.storage_utils, "get_storage_client", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(summary_writers, "get_files_relative", lambda *_args, **_kwargs: ["video.mp4"])
    monkeypatch.setattr(summary_writers, "write_json", fake_write_json)
    monkeypatch.setattr(
        summary_writers,
        "_read_all_video_metadata_parallel",
        lambda *_args, **_kwargs: {
            "video.mp4": summary_writers.ProcessedVideoMetadata(
                video_metadata={
                    "video_uuid": "video-id",
                    "num_clip_chunks": 1,
                    "num_total_clips": 1,
                    "duration": 5.0,
                },
                clip_chunks=[
                    {
                        "num_clips_passed": 1,
                        "num_clips_transcoded": 1,
                        "num_clips_with_caption": 1,
                        "num_caption_windows": 2,
                        "total_prompt_tokens": 30,
                        "total_output_tokens": 12,
                        "clips": ["clip-id"],
                        "filtered_clips": [],
                    }
                ],
            )
        },
    )

    summary_writers._write_split_result_summary(
        input_path="/input",
        input_videos_relative=["video.mp4"],
        num_input_videos_selected=1,
        output_path="/output",
        output_s3_profile_name="default",
        embedding_algorithm="internvideo2",
        limit=0,
        pipeline_run_time=1.0,
        write_all_caption_json=False,
    )

    assert captured_summary["total_num_clips_with_caption"] == 1
    assert captured_summary["total_num_caption_windows"] == 2
    assert captured_summary["video.mp4"]["num_caption_windows"] == 2

    throughput_args = next(args for message, args in info_calls if "Captioning throughput" in message)
    assert throughput_args[2] == 2
    assert throughput_args[3] == 15
    assert throughput_args[4] == 6
