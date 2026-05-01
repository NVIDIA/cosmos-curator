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

"""Shared helpers for OpenAI/Gemini API-backed caption stages."""

from typing import Any

from cosmos_curator.core.utils.config.config import resolve_model_name_auto
from cosmos_curator.pipelines.video.utils.data_model import CaptionFailureReason, CaptionOutcome, CaptionResult


class NonRetryableGeminiError(RuntimeError):
    """Error raised when retrying a Gemini request will not succeed."""


def create_openai_client_and_resolve_model(
    openai_module: Any,  # noqa: ANN401
    *,
    api_key: str,
    base_url: str | None,
    model_name: str,
    endpoint_label: str,
) -> tuple[Any, str]:
    """Create an OpenAI-compatible client and resolve ``auto`` model names."""
    client_kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    client = openai_module.OpenAI(**client_kwargs)
    resolved_model_name = resolve_model_name_auto(client, model_name, endpoint_label=endpoint_label)
    return client, resolved_model_name


def normalize_openai_response_with_detail(response: object) -> tuple[CaptionResult, str | None]:
    """Map an OpenAI chat completion response to a caption result."""
    choices = getattr(response, "choices", None)
    if not choices:
        return (
            CaptionResult(outcome=CaptionOutcome.ERROR, failure_reason="exception"),
            "OpenAI-compatible API returned no choices.",
        )

    choice = choices[0]
    finish_reason = getattr(choice, "finish_reason", None)
    content = getattr(choice.message, "content", None)
    text = content.strip() if isinstance(content, str) and content.strip() else None

    if finish_reason == "content_filter":
        return CaptionResult(outcome=CaptionOutcome.BLOCKED), None
    if content is None:
        return (
            CaptionResult(outcome=CaptionOutcome.ERROR, failure_reason="exception"),
            f"OpenAI-compatible API returned null content. finish_reason={finish_reason!r}",
        )
    if finish_reason == "length":
        outcome = CaptionOutcome.TRUNCATED if text is not None else CaptionOutcome.ERROR
        failure_reason: CaptionFailureReason | None = None if text is not None else "exception"
        detail = None
        if text is None:
            detail = (
                "OpenAI-compatible API returned no caption text after truncation. "
                f"finish_reason={finish_reason!r}, content={content!r}"
            )
        return CaptionResult(outcome=outcome, text=text, failure_reason=failure_reason), detail
    if text is not None:
        return CaptionResult(outcome=CaptionOutcome.SUCCESS, text=text), None
    return (
        CaptionResult(outcome=CaptionOutcome.ERROR, failure_reason="exception"),
        f"OpenAI-compatible API returned empty caption text. finish_reason={finish_reason!r}, content={content!r}",
    )


def normalize_openai_response(response: object) -> CaptionResult:
    """Map an OpenAI chat completion response to a caption result."""
    result, _detail = normalize_openai_response_with_detail(response)
    return result


def openai_error_result_from_exception(
    exc: BaseException,
    *,
    timeout_error_type: type[BaseException] | None,
) -> tuple[CaptionResult, str]:
    """Map an OpenAI exception to an error result and detail string."""
    failure_reason: CaptionFailureReason = (
        "timeout" if timeout_error_type is not None and isinstance(exc, timeout_error_type) else "exception"
    )
    return CaptionResult(outcome=CaptionOutcome.ERROR, failure_reason=failure_reason), str(exc)


def collect_gemini_response_text(response: object) -> tuple[str | None, set[str]]:
    """Collect response text and candidate finish reasons from a Gemini response."""
    candidates = getattr(response, "candidates", None) or []
    collected: list[str] = []
    finish_reasons: set[str] = set()
    for candidate in candidates:
        finish_reason = getattr(candidate, "finish_reason", None)
        if finish_reason:
            finish_reasons.add(str(finish_reason))
        content = getattr(candidate, "content", None) or candidate
        parts = getattr(content, "parts", None)
        if parts:
            for part in parts:
                part_text = getattr(part, "text", None)
                if isinstance(part_text, str) and part_text.strip():
                    collected.append(part_text.strip())

    response_text = getattr(response, "text", None)
    text = response_text.strip() if isinstance(response_text, str) and response_text.strip() else None
    if text is None:
        joined = "\n".join(collected).strip()
        text = joined or None
    return text, finish_reasons


def normalize_gemini_response_with_detail(response: object) -> tuple[CaptionResult, str | None]:
    """Map a Gemini response object to a caption result."""
    prompt_feedback = getattr(response, "prompt_feedback", None)
    block_reason = getattr(prompt_feedback, "block_reason", None)
    if block_reason:
        return CaptionResult(outcome=CaptionOutcome.BLOCKED), None

    text, finish_reasons = collect_gemini_response_text(response)

    if finish_reasons.intersection({"SAFETY", "RECITATION"}):
        return CaptionResult(outcome=CaptionOutcome.BLOCKED), None
    if "MAX_TOKENS" in finish_reasons:
        if text is not None:
            return CaptionResult(outcome=CaptionOutcome.TRUNCATED, text=text), None
        return (
            CaptionResult(outcome=CaptionOutcome.ERROR, failure_reason="exception"),
            f"Gemini returned MAX_TOKENS without caption text. finish_reasons={sorted(finish_reasons)}",
        )
    if text is not None:
        return CaptionResult(outcome=CaptionOutcome.SUCCESS, text=text), None
    detail = "Gemini response did not contain caption text."
    if finish_reasons:
        detail = f"{detail} finish_reasons={sorted(finish_reasons)}"
    return CaptionResult(outcome=CaptionOutcome.ERROR, failure_reason="exception"), detail


def normalize_gemini_response(response: object) -> CaptionResult:
    """Map a Gemini response object to a caption result."""
    result, _detail = normalize_gemini_response_with_detail(response)
    return result


def handle_gemini_client_exception(exc: BaseException) -> BaseException:
    """Wrap Gemini client exceptions with clearer messaging when appropriate."""
    message = str(exc).lower()
    if "api key not found" in message or "api_key_invalid" in message:
        msg = (
            "Gemini rejected the API key (API_KEY_INVALID). "
            "Update the cosmos-curator config with a valid Gemini API key."
        )
        return NonRetryableGeminiError(msg)
    return exc


def should_retry_gemini_exception(exc: BaseException) -> bool:
    """Decide whether the Gemini request should be retried."""
    return not isinstance(exc, NonRetryableGeminiError)


def gemini_error_result_from_exception(
    exc: BaseException,
    *,
    timeout_error_type: type[BaseException],
) -> tuple[CaptionResult, str]:
    """Map a Gemini exception to an error result and detail string."""
    failure_reason: CaptionFailureReason = "timeout" if isinstance(exc, timeout_error_type) else "exception"
    return CaptionResult(outcome=CaptionOutcome.ERROR, failure_reason=failure_reason), str(exc)
