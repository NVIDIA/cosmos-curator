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
"""Ray Data actor that sends caption windows to a hosted judge."""

import json
from typing import Any, Literal

import pyarrow as pa
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from cosmos_curator.core.utils.storage import storage_utils
from cosmos_curator.core.utils.storage.storage_client import StorageClient
from cosmos_curator.pipelines.ray_data.caption_judge.config import CaptionJudgePipelineConfig
from cosmos_curator.pipelines.ray_data.caption_judge.result_model import ISSUE_SCHEMA, Issue, make_issue
from cosmos_curator.pipelines.video.captioning.openai_caption_stage import OpenAICaptionStage
from cosmos_curator.pipelines.video.utils.windowing_utils import extract_window_mp4_from_clip_relative_ns_bounds

_MAX_RAW_RESPONSE_CHARS = 2000


class JudgeResponse(BaseModel):
    """Validated response from the caption judge."""

    model_config = ConfigDict(extra="ignore")

    winner: Literal["a", "b", "tie"]
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str
    a_errors: list[str] = Field(default_factory=list)
    b_errors: list[str] = Field(default_factory=list)


class _OpenAIJudgeClient:
    """Small wrapper around the existing OpenAI-compatible video request stage."""

    def __init__(self, config: CaptionJudgePipelineConfig) -> None:
        judge_config = config.judge
        self._stage = OpenAICaptionStage(
            model_name=judge_config.model_name,
            model_variant="caption_judge",
            max_output_tokens=judge_config.max_output_tokens,
            max_caption_retries=judge_config.max_retries,
            retry_delay_seconds=judge_config.retry_delay_seconds,
            batch_size=1,
            endpoint_key=judge_config.endpoint_key,
        )

    def setup(self) -> None:
        self._stage.stage_setup()

    def judge(self, *, prompt: str, video_bytes: bytes) -> str:
        return self._stage.caption_single(prompt, video_bytes)


class CaptionJudgeActor:
    """Ray Data actor for extracting window MP4s and calling the hosted judge."""

    conda_env_name = "default"

    def __init__(self, *, config: CaptionJudgePipelineConfig) -> None:
        """Bind run config and defer provider setup until the actor receives work."""
        self._config = config
        self._judge: _OpenAIJudgeClient | None = None
        self._storage_clients: dict[str, StorageClient | None] = {}

    def __call__(self, batch: pa.Table) -> pa.Table:
        """Process one Ray Data batch and return issue rows."""
        self._ensure_setup()
        assert self._judge is not None
        clip_cache: dict[str, bytes] = {}
        issues: list[Issue] = []
        for row in batch.to_pylist():
            try:
                clip_bytes = _cached_clip_bytes(
                    row["clip_location"],
                    clip_cache,
                    storage_clients=self._storage_clients,
                    input_root=self._config.candidate,
                    allow_unsafe_clip_location=self._config.execution.allow_unsafe_clip_location,
                )
                window_mp4 = extract_window_mp4(
                    clip_bytes,
                    start_ns=row["start_ns"],
                    end_ns=row["end_ns"],
                )
            except Exception as exc:  # noqa: BLE001 -- media failures become structured not-comparable rows.
                issues.append(
                    _issue_for_job(
                        row,
                        "caption_window_not_comparable",
                        f"Failed to extract candidate window MP4: {exc}",
                        output="candidate",
                    )
                )
                continue
            prompt = build_judge_prompt(row)
            try:
                raw = self._judge.judge(prompt=prompt, video_bytes=window_mp4)
            except Exception as exc:  # noqa: BLE001 -- hosted provider failures are surfaced in the report.
                issues.append(
                    _issue_for_job(
                        row,
                        "caption_judge_failed",
                        f"Caption judge request failed: {exc}",
                        output="candidate",
                    )
                )
                continue
            try:
                response = parse_judge_response(raw)
            except (json.JSONDecodeError, TypeError, ValueError, ValidationError) as exc:
                issues.append(
                    _issue_for_job(
                        row,
                        "caption_judge_invalid_response",
                        f"Caption judge returned an invalid response: {exc}",
                        output="candidate",
                        reason=str(exc),
                        raw_response=_truncate(raw),
                    )
                )
                continue
            if response.winner == "a":
                issues.append(
                    _issue_for_job(
                        row,
                        "caption_judge_prefers_baseline",
                        "Judge preferred baseline over candidate",
                        output="candidate",
                        winner="baseline",
                        confidence=response.confidence,
                        reason=response.reason,
                        baseline_errors=response.a_errors,
                        candidate_errors=response.b_errors,
                    )
                )
        return pa.Table.from_pylist(issues, schema=ISSUE_SCHEMA)

    def _ensure_setup(self) -> None:
        if self._judge is not None:
            return
        judge = _OpenAIJudgeClient(self._config)
        judge.setup()
        self._judge = judge


def _cached_clip_bytes(
    clip_location: str,
    clip_cache: dict[str, bytes],
    *,
    storage_clients: dict[str, StorageClient | None],
    input_root: str,
    allow_unsafe_clip_location: bool = False,
) -> bytes:
    cached = clip_cache.get(clip_location)
    if cached is not None:
        return cached
    resolved_location = (
        clip_location
        if allow_unsafe_clip_location
        else str(storage_utils.resolve_clip_location(clip_location, input_root))
    )
    if resolved_location not in storage_clients:
        storage_clients[resolved_location] = storage_utils.get_storage_client(resolved_location)
    data = storage_utils.read_bytes(resolved_location, storage_clients[resolved_location])
    clip_cache[clip_location] = data
    return data


def extract_window_mp4(clip_bytes: bytes, *, start_ns: int, end_ns: int) -> bytes:
    """Return the captioning window MP4 matching clip-relative nanosecond bounds."""
    return extract_window_mp4_from_clip_relative_ns_bounds(
        clip_bytes,
        start_ns=start_ns,
        end_ns=end_ns,
    )


def build_judge_prompt(row: dict[str, Any]) -> str:
    """Build the deterministic JSON-only caption judging prompt."""
    return "\n".join(
        [
            "You are judging two captions for the attached video window.",
            "Choose the better caption for the visible media. Use tie when both captions are equivalently correct.",
            "Set confidence to a probability-style number from 0.0 (low) to 1.0 (high).",
            "Return only one JSON object with this exact shape:",
            '{"winner":"a|b|tie","confidence":0.9,"reason":"short explanation","a_errors":[],"b_errors":[]}',
            "",
            "Caption A:",
            str(row["baseline_caption"]),
            "",
            "Caption B:",
            str(row["candidate_caption"]),
        ]
    )


def parse_judge_response(raw: str) -> JudgeResponse:
    """Parse a hosted judge response, accepting plain JSON or fenced JSON."""
    payload = _extract_json_object(raw)
    data = json.loads(payload)
    if not isinstance(data, dict):
        msg = "Judge response JSON must be an object"
        raise TypeError(msg)
    return JudgeResponse.model_validate(data)


def _extract_json_object(raw: str) -> str:  # noqa: C901
    start = raw.find("{")
    if start < 0:
        msg = "Judge response did not contain a JSON object"
        raise ValueError(msg)
    depth = 0
    in_string = False
    escaped = False
    for index, char in enumerate(raw[start:], start=start):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return raw[start : index + 1]
    msg = "Judge response JSON object was not closed"
    raise ValueError(msg)


def _issue_for_job(  # noqa: PLR0913 -- passes through the persisted issue schema fields.
    row: dict[str, Any],
    code: str,
    message: str,
    *,
    output: str | None = None,
    winner: str | None = None,
    confidence: float | None = None,
    reason: str | None = None,
    baseline_errors: list[str] | None = None,
    candidate_errors: list[str] | None = None,
    raw_response: str | None = None,
) -> Issue:
    return make_issue(
        code,  # type: ignore[arg-type]
        message,
        video_uuid=row["video_uuid"],
        clip_uuid=row["clip_uuid"],
        start_ns=row["start_ns"],
        end_ns=row["end_ns"],
        output=output,
        caption_model_baseline=row["caption_model_baseline"],
        caption_model_candidate=row["caption_model_candidate"],
        winner=winner,
        confidence=confidence,
        reason=reason,
        baseline_errors=baseline_errors,
        candidate_errors=candidate_errors,
        baseline_caption=row["baseline_caption"],
        candidate_caption=row["candidate_caption"],
        raw_response=raw_response,
    )


def _truncate(raw: str) -> str:
    if len(raw) <= _MAX_RAW_RESPONSE_CHARS:
        return raw
    return raw[:_MAX_RAW_RESPONSE_CHARS] + "...<truncated>"
