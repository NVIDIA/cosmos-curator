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

"""Shared CPU postprocessing helpers for semantic filtering/classification."""

import json
import pathlib
import re

import attrs
from loguru import logger

MALFORMED_MODEL_OUTPUT = "malformed_model_output"


def parse_comma_separated_types(cs: str | None) -> list[str]:
    """Return list of non-empty stripped tokens from a comma-separated string."""
    if not cs:
        return []
    return [s.strip() for s in cs.split(",") if s.strip()]


def read_categories_file(path: str | None) -> str | None:
    """Read a newline-separated categories file and return a comma-separated string."""
    if path is None:
        return None
    categories = [line.strip() for line in pathlib.Path(path).read_text().splitlines() if line.strip()]
    if not categories:
        msg = f"Categories file {path!r} is empty."
        raise ValueError(msg)
    return ",".join(categories)


def custom_categories_union(type_allow: str | None, type_block: str | None) -> str | None:
    """Return comma-separated sorted union of allow and block types, or None if empty."""
    categories = set(parse_comma_separated_types(type_allow)) | set(parse_comma_separated_types(type_block))
    return ",".join(sorted(categories)) if categories else None


def _clean_json_string(output_text: str) -> dict[str, str] | None:
    """Clean and fix common JSON formatting issues."""
    last_think_end = output_text.rfind("</think>")
    if last_think_end != -1:
        output_text = output_text[last_think_end + len("</think>") :]

    output_text = re.sub(r"</?answer>", "", output_text)
    output_text = output_text.replace('\\"', "'")
    output_text = output_text.lstrip("\ufeff\n\r\t ")
    output_text = output_text.replace("\r\n", "\n")
    output_text = re.sub(r'"\s*\n\s*"', '",\n"', output_text)

    start = output_text.find("{")
    if start == -1:
        logger.error(f"No JSON object found in output: {output_text!r}")
        return None

    depth = 0
    end = -1
    for i in range(start, len(output_text)):
        if output_text[i] == "{":
            depth += 1
        elif output_text[i] == "}":
            depth -= 1
            if depth == 0:
                end = i
                break
    if end == -1:
        logger.error(f"Unmatched braces in output: {output_text!r}")
        return None

    output_text = output_text[start : end + 1].strip()
    try:
        return json.loads(output_text)  # type: ignore[no-any-return]
    except json.JSONDecodeError as e:
        logger.error(f"JSON Error: {e}")
        logger.error(f"Output text: {output_text}")
        return None


def parse_results(caption: str) -> dict[str, str] | None:
    """Parse a model caption into a normalized dictionary."""
    result = _clean_json_string(caption)
    if result is None:
        return None
    return {k.replace(" ", "_"): v for k, v in result.items()}


def evaluate_semantic_window_results(
    window_results: list[tuple[int, str]],
    *,
    filter_criteria: list[str],
    rejection_threshold: float,
    score_only: bool,
) -> tuple[bool, set[str], dict[int, dict[str, str]], dict[int, str]]:
    """Evaluate semantic filter captions and return pass/fail plus window rejection details."""
    all_issues: set[str] = set()
    rejected_windows: set[int] = set()
    per_window_reasons: dict[int, dict[str, str]] = {}
    per_window_errors: dict[int, str] = {}

    for window_idx, result in window_results:
        filtering_dict = parse_results(result)
        if filtering_dict is None:
            per_window_errors[window_idx] = MALFORMED_MODEL_OUTPUT
            continue
        window_specific_issues: dict[str, str] = {}
        for criterion in filter_criteria:
            criterion_key = criterion.replace(" ", "_")
            if filtering_dict.get(criterion_key, "no") == "yes":
                all_issues.add(criterion)
                rejected_windows.add(window_idx)
                window_specific_issues[criterion] = "yes"
        per_window_reasons[window_idx] = window_specific_issues

    clip_should_pass = True
    if not score_only and window_results:
        all_malformed = len(per_window_errors) == len(window_results)
        if all_malformed or (len(rejected_windows) / len(window_results)) > rejection_threshold:
            clip_should_pass = False
    return clip_should_pass, all_issues, per_window_reasons, per_window_errors


@attrs.define(frozen=True)
class ClassifierEvaluationConfig:
    """Configuration for classifier window-result evaluation."""

    type_allow: list[str]
    type_block: list[str]
    custom_categories: bool
    valid_type_labels: tuple[str, ...]
    rejection_threshold: float


def _collect_classifier_matches(
    filtering_dict: dict[str, str],
    *,
    type_allow: list[str],
    type_block: list[str],
) -> tuple[dict[str, str], bool, str | None]:
    """Collect per-window rejection reasons and classifier matches."""
    rejection_reasons: dict[str, str] = {}
    matched_allow = False
    matched_block: str | None = None

    for label in type_block:
        if filtering_dict.get(label, "no") == "yes":
            rejection_reasons[label] = "yes"
            if matched_block is None:
                matched_block = label

    for label in type_allow:
        value = filtering_dict.get(label, "no")
        if value == "yes":
            matched_allow = True
        elif value == "no":
            rejection_reasons[label] = "no"

    return rejection_reasons, matched_allow, matched_block


def evaluate_classifier_window_results(
    window_results: list[tuple[int, str]],
    *,
    config: ClassifierEvaluationConfig,
) -> tuple[bool, set[str], dict[int, dict[str, str]], dict[int, str], list[str]]:
    """Evaluate classifier captions and return pass/fail plus labels and rejection details."""
    has_allowed = not bool(config.type_allow)
    any_parsed = False
    all_types_yes: set[str] = set()
    all_issues: set[str] = set()
    rejected_windows: set[int] = set()
    per_window_reasons: dict[int, dict[str, str]] = {}
    per_window_errors: dict[int, str] = {}

    for window_idx, result in window_results:
        filtering_dict = parse_results(result)
        if filtering_dict is None:
            per_window_errors[window_idx] = MALFORMED_MODEL_OUTPUT
            continue
        any_parsed = True
        all_types_yes.update(
            key for key, value in filtering_dict.items() if value == "yes" and key in config.valid_type_labels
        )
        rejection_reasons, matched_allow, matched_block = _collect_classifier_matches(
            filtering_dict,
            type_allow=config.type_allow,
            type_block=config.type_block,
        )
        per_window_reasons[window_idx] = rejection_reasons
        if config.type_allow and matched_allow:
            has_allowed = True
        if matched_block is not None:
            all_issues.add(matched_block)
            rejected_windows.add(window_idx)

    if config.type_allow and not any_parsed and not per_window_errors:
        has_allowed = True

    allow_ok = has_allowed
    block_ok = True
    if config.type_block and window_results:
        block_ok = (len(rejected_windows) / len(window_results)) <= config.rejection_threshold

    classification = sorted(all_types_yes) if all_types_yes else (["unclassified"] if config.custom_categories else [])
    return bool(allow_ok and block_ok), all_issues, per_window_reasons, per_window_errors, classification
