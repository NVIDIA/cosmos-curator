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

"""Tests for the per-event VLM caption stage.

These tests intentionally avoid loading a real Qwen or Gemini backend; the
``_call_qwen`` / ``_call_gemini`` methods are monkeypatched so the suite stays
CPU-only and runs in the default cosmos-curator environment.
"""

import json
from pathlib import Path
from uuid import uuid4

import pytest

from cosmos_curator.core.utils.config import config as config_module
from cosmos_curator.core.utils.config.config import ConfigFileData, Gemini
from cosmos_curator.core.utils.data.bytes_transport import bytes_to_numpy
from cosmos_curator.pipelines.video.captioning.per_event_caption_stage import (
    PerEventCaptionStage,
    _build_instances_block,
    _build_prompt,
    _extract_events_payload,
)
from cosmos_curator.pipelines.video.utils.data_model import Clip, SplitPipeTask, Video

# ---------------------------------------------------------------------------
# _extract_events_payload — JSON-shape robustness
# ---------------------------------------------------------------------------


def test_extract_events_payload_wrapped_dict() -> None:
    """Top-level ``{"events": [...]}`` is unwrapped."""
    raw = '{"events": [{"event_id": "e0"}, {"event_id": "e1"}]}'
    assert _extract_events_payload(raw) == [{"event_id": "e0"}, {"event_id": "e1"}]


def test_extract_events_payload_bare_array() -> None:
    """A bare top-level array is accepted as-is."""
    raw = '[{"event_id": "e0"}]'
    assert _extract_events_payload(raw) == [{"event_id": "e0"}]


def test_extract_events_payload_dict_extra_keys_ignored() -> None:
    """Extra sibling keys next to ``events`` are tolerated."""
    raw = '{"meta": "blah", "events": [{"e": 1}], "trailing": [1, 2]}'
    assert _extract_events_payload(raw) == [{"e": 1}]


def test_extract_events_payload_dict_without_events_returns_empty() -> None:
    """A JSON object missing ``events`` returns an empty list."""
    assert _extract_events_payload('{"foo": [1, 2, 3]}') == []


def test_extract_events_payload_dict_events_not_list_returns_empty() -> None:
    """``events`` of the wrong type returns an empty list rather than crashing."""
    assert _extract_events_payload('{"events": "oops"}') == []


def test_extract_events_payload_strips_markdown_json_fence() -> None:
    """Markdown ```json fences are stripped before parsing."""
    raw = '```json\n{"events": [{"event_id": "e0"}]}\n```'
    assert _extract_events_payload(raw) == [{"event_id": "e0"}]


def test_extract_events_payload_strips_bare_fence() -> None:
    """Markdown ``` fences without a language tag are also stripped."""
    raw = '```\n[{"event_id": "e0"}]\n```'
    assert _extract_events_payload(raw) == [{"event_id": "e0"}]


def test_extract_events_payload_recovers_object_after_prose() -> None:
    """The greedy regex fallback recovers a JSON object embedded in prose."""
    raw = 'Here is the result:\n{"events": [{"event_id": "e0"}]}\n\nThanks!'
    assert _extract_events_payload(raw) == [{"event_id": "e0"}]


def test_extract_events_payload_returns_empty_on_garbage() -> None:
    """Non-JSON / empty input returns ``[]`` instead of raising."""
    assert _extract_events_payload("not json at all") == []
    assert _extract_events_payload("") == []


def test_extract_events_payload_tolerates_unescaped_newlines() -> None:
    r"""``strict=False`` lets us recover when a VLM forgets to escape ``\n`` in a caption."""
    raw = '{"events": [{"event_caption": "line1\nline2"}]}'
    out = _extract_events_payload(raw)
    assert len(out) == 1
    assert "line1" in out[0]["event_caption"]
    assert "line2" in out[0]["event_caption"]


# ---------------------------------------------------------------------------
# _build_instances_block / _build_prompt — pure helpers
# ---------------------------------------------------------------------------


def _make_clip(span: tuple[float, float] = (0.0, 5.0), instances: list[dict] | None = None) -> Clip:
    clip = Clip(uuid=uuid4(), source_video="src.mp4", span=span)
    clip.sam3_instances = instances
    return clip


def test_build_instances_block_filters_non_int_object_ids() -> None:
    """Entries with a non-int ``object_id`` are dropped from the prompt block."""
    clip = _make_clip(
        instances=[
            {"object_id": 1, "prompt": "a car", "start_time_s": 0.0, "end_time_s": 1.0},
            {"object_id": "garbage", "prompt": "a bus", "start_time_s": 0.0, "end_time_s": 1.0},
            {"object_id": 7, "prompt": "a cat"},
        ],
    )
    payload = json.loads(_build_instances_block(clip))
    assert [e["object_id"] for e in payload["instances"]] == [1, 7]
    assert payload["instances"][0]["class"] == "a car"
    # ``class`` falls through to the prompt; missing time fields surface as None.
    assert payload["instances"][1]["class"] == "a cat"
    assert payload["instances"][1]["start_time_s"] is None


def test_build_instances_block_handles_missing_instances() -> None:
    """``sam3_instances=None`` produces an empty instances payload."""
    clip = _make_clip(instances=None)
    payload = json.loads(_build_instances_block(clip))
    assert payload == {"instances": []}


def test_build_prompt_appends_duration_and_instances() -> None:
    """The user template is followed by clip duration and the instances block."""
    clip = _make_clip(
        span=(0.0, 5.0),
        instances=[{"object_id": 24, "prompt": "a car", "start_time_s": 0.5, "end_time_s": 4.5}],
    )
    out = _build_prompt("USER PROMPT", clip)
    assert "USER PROMPT" in out
    assert "CLIP DURATION: 5.00 seconds" in out
    assert "[0.0, 5.00]" in out
    assert '"object_id": 24' in out


# ---------------------------------------------------------------------------
# PerEventCaptionStage._looks_truncated
# ---------------------------------------------------------------------------


def test_looks_truncated_completed_object() -> None:
    """A response that ends in ``}`` is treated as complete."""
    assert PerEventCaptionStage._looks_truncated('{"events": []}') is False


def test_looks_truncated_completed_array() -> None:
    """A response that ends in ``]`` is treated as complete."""
    assert PerEventCaptionStage._looks_truncated("[1, 2, 3]") is False


def test_looks_truncated_dangling_object() -> None:
    """A response that ends mid-object is flagged as truncated."""
    assert PerEventCaptionStage._looks_truncated('{"events": [{"x":') is True


def test_looks_truncated_empty_string() -> None:
    """An empty response is not flagged as truncated (it's ``empty_or_unparseable``)."""
    assert PerEventCaptionStage._looks_truncated("") is False


# ---------------------------------------------------------------------------
# PerEventCaptionStage construction-time validation
# ---------------------------------------------------------------------------


def _patch_gemini_config(monkeypatch: pytest.MonkeyPatch, *, api_key: str | None = "key-xyz") -> None:
    """Stub ``load_config`` so Gemini construction doesn't read ~/.config."""
    config = ConfigFileData(gemini=Gemini(api_key=api_key)) if api_key is not None else ConfigFileData()
    monkeypatch.setattr(config_module, "load_config", lambda: config)


def test_invalid_backend_raises() -> None:
    """An unknown backend name fails fast with ``ValueError``."""
    with pytest.raises(ValueError, match="backend"):
        PerEventCaptionStage(backend="bogus")  # type: ignore[arg-type]


def test_invalid_media_resolution_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """``gemini_media_resolution`` outside {low,medium,high} raises ``ValueError``."""
    _patch_gemini_config(monkeypatch)
    with pytest.raises(ValueError, match="media_resolution"):
        PerEventCaptionStage(backend="gemini", gemini_media_resolution="ultra")


def test_gemini_backend_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Constructing the gemini backend without a configured key raises ``RuntimeError``."""
    _patch_gemini_config(monkeypatch, api_key=None)
    with pytest.raises(RuntimeError, match="Gemini API key missing"):
        PerEventCaptionStage(backend="gemini")


def test_gemini_backend_loads_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the config contains a key, the gemini backend records it and stays CPU-only."""
    _patch_gemini_config(monkeypatch)
    stage = PerEventCaptionStage(backend="gemini")
    assert stage._gemini_api_key == "key-xyz"
    assert stage.resources.gpus == 0
    assert stage.resources.cpus >= 1.0
    assert stage._qwen_model is None


# ---------------------------------------------------------------------------
# _process_clip orchestration (Gemini-backed for cheap construction)
# ---------------------------------------------------------------------------


def _make_clip_with_annotated(
    instances: list[dict] | None,
    annotated_bytes: bytes | None,
) -> tuple[SplitPipeTask, Clip]:
    """Build a ``SplitPipeTask`` whose only clip optionally carries SAM3 outputs."""
    clip = Clip(uuid=uuid4(), source_video="src.mp4", span=(0.0, 5.0))
    if instances is not None:
        clip.sam3_instances = instances
    if annotated_bytes is not None:
        clip.sam3_annotated_video = bytes_to_numpy(annotated_bytes)  # type: ignore[assignment]
    video = Video(input_video=Path("src.mp4"), clips=[clip])
    task = SplitPipeTask(session_id="t", videos=[video])
    return task, clip


def _gemini_stage(monkeypatch: pytest.MonkeyPatch) -> PerEventCaptionStage:
    """Construct a Gemini-backed stage with a stubbed API key."""
    _patch_gemini_config(monkeypatch)
    return PerEventCaptionStage(backend="gemini", verbose=True)


def _stub_call_gemini(stage: PerEventCaptionStage, monkeypatch: pytest.MonkeyPatch, response: str) -> None:
    """Replace ``_call_gemini`` with a stub returning ``response`` regardless of args."""
    monkeypatch.setattr(stage, "_call_gemini", lambda _mp4, _prompt, _uuid: response)


def test_process_clip_skips_when_no_instances(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clips with ``sam3_instances=None`` are skipped silently (no errors recorded)."""
    stage = _gemini_stage(monkeypatch)
    _, clip = _make_clip_with_annotated(instances=None, annotated_bytes=b"\x00")
    stage._process_clip(clip)
    assert clip.sam3_events is None
    assert clip.errors == {}


def test_process_clip_skips_on_empty_instances(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clips with an empty instances list are skipped silently."""
    stage = _gemini_stage(monkeypatch)
    _, clip = _make_clip_with_annotated(instances=[], annotated_bytes=b"\x00")
    stage._process_clip(clip)
    assert clip.sam3_events is None
    assert clip.errors == {}


def test_process_clip_records_missing_annotated_video(monkeypatch: pytest.MonkeyPatch) -> None:
    """Instances without an annotated video record ``missing_annotated_video`` and skip."""
    stage = _gemini_stage(monkeypatch)
    _, clip = _make_clip_with_annotated(
        instances=[{"object_id": 1, "prompt": "a car", "start_time_s": 0.0, "end_time_s": 1.0}],
        annotated_bytes=None,
    )
    stage._process_clip(clip)
    assert clip.errors["per_event_caption"] == "missing_annotated_video"
    assert clip.sam3_events is None


def test_process_clip_populates_events_on_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """A well-formed VLM response populates ``clip.sam3_events`` and records no errors."""
    stage = _gemini_stage(monkeypatch)
    _, clip = _make_clip_with_annotated(
        instances=[{"object_id": 1, "prompt": "a car", "start_time_s": 0.0, "end_time_s": 1.0}],
        annotated_bytes=b"fake-mp4-bytes",
    )
    _stub_call_gemini(
        stage,
        monkeypatch,
        '{"events": [{"event_id": "event_000000", "category": "collision"}]}',
    )
    stage._process_clip(clip)
    assert clip.sam3_events == [{"event_id": "event_000000", "category": "collision"}]
    assert clip.errors == {}


def test_process_clip_records_empty_response(monkeypatch: pytest.MonkeyPatch) -> None:
    """A complete-but-eventless response records ``empty_or_unparseable_response``."""
    stage = _gemini_stage(monkeypatch)
    _, clip = _make_clip_with_annotated(
        instances=[{"object_id": 1, "prompt": "a car", "start_time_s": 0.0, "end_time_s": 1.0}],
        annotated_bytes=b"fake-mp4-bytes",
    )
    # Valid JSON dict that ends in ``}`` (not truncated) but has no ``events`` key.
    _stub_call_gemini(stage, monkeypatch, '{"foo": "bar"}')
    stage._process_clip(clip)
    assert clip.sam3_events == []
    assert clip.errors["per_event_caption"] == "empty_or_unparseable_response"


def test_process_clip_records_truncated_response(monkeypatch: pytest.MonkeyPatch) -> None:
    """A response chopped mid-JSON records ``truncated_response``."""
    stage = _gemini_stage(monkeypatch)
    _, clip = _make_clip_with_annotated(
        instances=[{"object_id": 1, "prompt": "a car", "start_time_s": 0.0, "end_time_s": 1.0}],
        annotated_bytes=b"fake-mp4-bytes",
    )
    # No closing brace — trips ``_looks_truncated``.
    _stub_call_gemini(stage, monkeypatch, '{"events": [{"x":')
    stage._process_clip(clip)
    assert clip.sam3_events == []
    assert clip.errors["per_event_caption"] == "truncated_response"


def test_process_clip_records_api_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exceptions from the backend bubble into ``clip.errors`` as ``api_error``."""
    stage = _gemini_stage(monkeypatch)
    _, clip = _make_clip_with_annotated(
        instances=[{"object_id": 1, "prompt": "a car", "start_time_s": 0.0, "end_time_s": 1.0}],
        annotated_bytes=b"fake-mp4-bytes",
    )

    def _boom(_mp4: bytes, _prompt: str, _uuid: str) -> str:
        msg = "kaboom"
        raise RuntimeError(msg)

    monkeypatch.setattr(stage, "_call_gemini", _boom)
    stage._process_clip(clip)
    assert "api_error" in clip.errors["per_event_caption"]
    assert "kaboom" in clip.errors["per_event_caption"]
    assert clip.sam3_events is None


def test_process_data_iterates_all_clips(monkeypatch: pytest.MonkeyPatch) -> None:
    """``process_data`` should call ``_process_clip`` once per clip across all tasks."""
    stage = _gemini_stage(monkeypatch)
    task, _ = _make_clip_with_annotated(
        instances=[{"object_id": 1, "prompt": "a car", "start_time_s": 0.0, "end_time_s": 1.0}],
        annotated_bytes=b"fake-mp4-bytes",
    )
    second_clip = Clip(uuid=uuid4(), source_video="src.mp4", span=(0.0, 5.0))
    second_clip.sam3_instances = [{"object_id": 2, "prompt": "a bus", "start_time_s": 0.0, "end_time_s": 1.0}]
    second_clip.sam3_annotated_video = bytes_to_numpy(b"second")  # type: ignore[assignment]
    task.videos[0].clips.append(second_clip)

    calls: list[str] = []

    def _stub(_mp4: bytes, _prompt: str, uuid: str) -> str:
        calls.append(uuid)
        return '{"events": [{"event_id": "e0"}]}'

    monkeypatch.setattr(stage, "_call_gemini", _stub)
    stage.process_data([task])
    assert len(calls) == 2
    for clip in task.videos[0].clips:
        assert clip.sam3_events == [{"event_id": "e0"}]
