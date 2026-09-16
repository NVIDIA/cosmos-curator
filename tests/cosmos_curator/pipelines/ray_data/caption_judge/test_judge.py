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
"""Tests for the caption judge Ray Data actor."""

from pathlib import Path

import pyarrow as pa
import pytest

from cosmos_curator.pipelines.ray_data.caption_judge import _judge as judge_module
from cosmos_curator.pipelines.ray_data.caption_judge._judge import (
    CaptionJudgeActor,
    extract_window_mp4,
    parse_judge_response,
)
from cosmos_curator.pipelines.ray_data.caption_judge.config import CaptionJudgePipelineConfig
from cosmos_curator.pipelines.ray_data.caption_judge.result_model import JUDGE_JOB_SCHEMA


def _config() -> CaptionJudgePipelineConfig:
    return CaptionJudgePipelineConfig(
        schema_version=1,
        kind="caption_judge",
        input={
            "baseline": "output-a",
            "candidate": "output-b",
        },
        output={"report_path": "report.json"},
    )


def _job_table() -> pa.Table:
    return pa.Table.from_pylist(
        [
            {
                "video_uuid": "video-1",
                "clip_uuid": "clip-1",
                "start_ns": 0,
                "end_ns": 1_000_000_000,
                "clip_location": "/clips/b.mp4",
                "caption_model_baseline": "qwen",
                "caption_model_candidate": "cosmos3_nano",
                "baseline_caption": "baseline",
                "candidate_caption": "candidate",
            }
        ],
        schema=JUDGE_JOB_SCHEMA,
    )


def test_parse_judge_response_accepts_fenced_json() -> None:
    """The parser accepts common fenced JSON responses."""
    parsed = parse_judge_response(
        '```json\n{"winner":"a","confidence":0.8,"reason":"better","a_errors":[],"b_errors":["miss"]}\n```'
    )

    assert parsed.winner == "a"
    assert parsed.confidence == 0.8
    assert parsed.b_errors == ["miss"]


def test_parse_judge_response_rejects_missing_json_object() -> None:
    """A response with no JSON object is invalid."""
    with pytest.raises(ValueError, match="did not contain"):
        parse_judge_response("winner is a")


def test_extract_window_mp4_uses_ns_bound_windowing_util(monkeypatch: pytest.MonkeyPatch) -> None:
    """Window extraction delegates clip-relative ns bounds to shared windowing utils."""
    captured: dict[str, object] = {}

    def fake_extract(clip_bytes: bytes, *, start_ns: int, end_ns: int) -> bytes:
        captured["clip_bytes"] = clip_bytes
        captured["start_ns"] = start_ns
        captured["end_ns"] = end_ns
        return b"window"

    monkeypatch.setattr(judge_module, "extract_window_mp4_from_clip_relative_ns_bounds", fake_extract)

    result = extract_window_mp4(b"clip", start_ns=1_000_000_000, end_ns=1_500_000_000)

    assert result == b"window"
    assert captured == {
        "clip_bytes": b"clip",
        "start_ns": 1_000_000_000,
        "end_ns": 1_500_000_000,
    }


def test_extract_window_mp4_propagates_windowing_util_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """Invalid metadata bounds still surface as extraction failures."""

    def fake_extract(_clip_bytes: bytes, *, start_ns: int, end_ns: int) -> bytes:
        msg = f"No decoded frames fell within clip-relative bounds start_ns={start_ns}, end_ns={end_ns}"
        raise ValueError(msg)

    monkeypatch.setattr(judge_module, "extract_window_mp4_from_clip_relative_ns_bounds", fake_extract)

    with pytest.raises(ValueError, match="No decoded frames"):
        extract_window_mp4(b"clip", start_ns=1_000_000_000, end_ns=1_500_000_000)


def test_cached_clip_bytes_reads_location_inside_input_root(tmp_path: Path) -> None:
    """A clip_location nested under input_root is read normally."""
    root = tmp_path / "candidate"
    clip = root / "clips" / "a.mp4"
    clip.parent.mkdir(parents=True)
    clip.write_bytes(b"data")

    data = judge_module._cached_clip_bytes(str(clip), {}, storage_clients={}, input_root=str(root))

    assert data == b"data"


def test_cached_clip_bytes_rejects_location_outside_input_root(tmp_path: Path) -> None:
    """A tampered clip_location outside input_root is rejected before it is read."""
    root = tmp_path / "candidate"
    root.mkdir()
    outside = tmp_path / "outside" / "secret.mp4"
    outside.parent.mkdir()
    outside.write_bytes(b"secret")

    with pytest.raises(ValueError, match="outside"):
        judge_module._cached_clip_bytes(str(outside), {}, storage_clients={}, input_root=str(root))


def test_actor_calls_cached_clip_bytes_with_candidate_root(monkeypatch: pytest.MonkeyPatch) -> None:
    """The actor validates clip_location against the config's candidate root."""

    class FakeJudge:
        def __init__(self, _config: CaptionJudgePipelineConfig) -> None:
            pass

        def setup(self) -> None:
            pass

        def judge(self, *, prompt: str, video_bytes: bytes) -> str:
            del prompt, video_bytes
            return '{"winner":"tie","confidence":0.5,"reason":"same","a_errors":[],"b_errors":[]}'

    captured: dict[str, object] = {}

    def fake_cached_clip_bytes(
        _clip_location: str,
        _clip_cache: dict[str, bytes],
        *,
        input_root: str,
        **_kwargs: object,
    ) -> bytes:
        captured["input_root"] = input_root
        return b"clip"

    monkeypatch.setattr(judge_module, "_OpenAIJudgeClient", FakeJudge)
    monkeypatch.setattr(judge_module, "_cached_clip_bytes", fake_cached_clip_bytes)
    monkeypatch.setattr(judge_module, "extract_window_mp4", lambda *_args, **_kwargs: b"window")

    CaptionJudgeActor(config=_config())(_job_table())

    assert captured["input_root"] == "output-b"


def test_actor_emits_regression_when_judge_prefers_baseline(monkeypatch: pytest.MonkeyPatch) -> None:
    """The actor emits a regression issue when the judge picks the non-candidate side."""

    class FakeJudge:
        def __init__(self, _config: CaptionJudgePipelineConfig) -> None:
            pass

        def setup(self) -> None:
            pass

        def judge(self, *, prompt: str, video_bytes: bytes) -> str:
            assert "Caption A:" in prompt
            assert "Caption B:" in prompt
            assert "candidate output" not in prompt.lower()
            assert "qwen" not in prompt
            assert "cosmos3_nano" not in prompt
            assert video_bytes == b"window"
            return '{"winner":"a","confidence":0.95,"reason":"A is more accurate","a_errors":[],"b_errors":["miss"]}'

    monkeypatch.setattr(judge_module, "_OpenAIJudgeClient", FakeJudge)
    monkeypatch.setattr(judge_module, "_cached_clip_bytes", lambda *_args, **_kwargs: b"clip")
    monkeypatch.setattr(judge_module, "extract_window_mp4", lambda *_args, **_kwargs: b"window")

    result = CaptionJudgeActor(config=_config())(_job_table())

    rows = result.to_pylist()
    assert len(rows) == 1
    assert rows[0]["code"] == "caption_judge_prefers_baseline"
    assert rows[0]["winner"] == "baseline"
    assert rows[0]["output"] == "candidate"
    assert rows[0]["candidate_errors"] == ["miss"]


def test_actor_retries_judge_setup_after_setup_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """A transient setup failure must not cache a partially initialized judge."""

    class FakeJudge:
        setup_calls = 0

        def __init__(self, _config: CaptionJudgePipelineConfig) -> None:
            self.setup_done = False

        def setup(self) -> None:
            type(self).setup_calls += 1
            if type(self).setup_calls == 1:
                msg = "setup boom"
                raise RuntimeError(msg)
            self.setup_done = True

        def judge(self, *, prompt: str, video_bytes: bytes) -> str:
            assert prompt
            assert video_bytes == b"window"
            if not self.setup_done:
                msg = "judge used before setup"
                raise RuntimeError(msg)
            return '{"winner":"b","confidence":0.95,"reason":"B is better","a_errors":[],"b_errors":[]}'

    monkeypatch.setattr(judge_module, "_OpenAIJudgeClient", FakeJudge)
    monkeypatch.setattr(judge_module, "_cached_clip_bytes", lambda *_args, **_kwargs: b"clip")
    monkeypatch.setattr(judge_module, "extract_window_mp4", lambda *_args, **_kwargs: b"window")

    actor = CaptionJudgeActor(config=_config())
    with pytest.raises(RuntimeError, match="setup boom"):
        actor(_job_table())

    result = actor(_job_table())

    assert FakeJudge.setup_calls == 2
    assert result.to_pylist() == []


def test_actor_records_invalid_judge_response(monkeypatch: pytest.MonkeyPatch) -> None:
    """Invalid provider JSON is preserved as an invalid-response issue."""

    class FakeJudge:
        def __init__(self, _config: CaptionJudgePipelineConfig) -> None:
            pass

        def setup(self) -> None:
            pass

        def judge(self, *, prompt: str, video_bytes: bytes) -> str:
            assert prompt
            assert video_bytes
            return "not json"

    monkeypatch.setattr(judge_module, "_OpenAIJudgeClient", FakeJudge)
    monkeypatch.setattr(judge_module, "_cached_clip_bytes", lambda *_args, **_kwargs: b"clip")
    monkeypatch.setattr(judge_module, "extract_window_mp4", lambda *_args, **_kwargs: b"window")

    result = CaptionJudgeActor(config=_config())(_job_table())

    row = result.to_pylist()[0]
    assert row["code"] == "caption_judge_invalid_response"
    assert row["raw_response"] == "not json"
