# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic egocentric caption prompt rendering (pure; no I/O, Lance, or model).

The prompt is rendered ONCE, on the Lance scan, and travels as a ``prompt_text``
column. Rendering it again closer to the engine would put two normalizers on the
same fields, which is how one table ends up holding two subtly different prompts
under a single contract digest.
"""

from collections.abc import Mapping, Sequence
from typing import Final

import attrs

PROMPT_VERSION: Final = "egocentric_v1"


@attrs.frozen
class PromptContextField:
    """One optional prompt context column and the label it renders under.

    Attributes:
        column: Source column name, also the projection key.
        label: Prefix rendered as ``label: value``.

    """

    column: str
    label: str


# Every context column this prompt knows how to render, in render order. One
# definition drives projection, rendering, diagnostics, and the digest, so a
# column cannot be projected without a label or labelled without being read.
#
# Which of these a given run actually uses is resolved per snapshot against the
# source schema (``table.resolve_context_fields``); a column absent from the table
# is simply not part of that run's contract. Listing a column here therefore does
# not change any digest on its own - landing it upstream does, because the
# resolved subset grows. That is intended: those captions really were produced
# without it.
#
# ``task_description`` sits directly after ``task_name`` because it elaborates the
# same label. Reordering this tuple is a deliberate digest change.
#
# No per-field length cap, and do not add one on the assumption that these are short
# label columns: only task_name is. subtask_name and camera_motion_annotation are
# free-form prose whose length grows with clip content. Caps stay off because the whole
# rendered prompt is small beside the video frames in the same request - but that is a
# property of the input data, not a guarantee of this code. A prompt that outgrows the
# bound fails in one of two ways, and neither is silent. Exceed
# generation.max_model_len
# outright and the request raises: the engine stage pins should_continue_on_error=False,
# so that ends the whole run instead of producing one terminal error row. Merely eat
# into the output headroom and nothing raises at all - the generation stops at the
# context limit, which contracts.caption_status catches by measuring the prompt plus
# the generation against generation.max_model_len, so the shortened caption
# publishes as truncated rather than as complete.
# Re-measure before trusting the bound on new inputs; see
# docs/curator/design/curator-next-robot-action-caption.md.
SUPPORTED_CONTEXT_FIELDS: Final[tuple[PromptContextField, ...]] = (
    PromptContextField(column="task_name", label="Task"),
    PromptContextField(column="task_description", label="Task description"),
    PromptContextField(column="subtask_name", label="Current action"),
    PromptContextField(column="camera_motion_annotation", label="Camera motion"),
)

SYSTEM_INSTRUCTION: Final = (
    "You are annotating egocentric video recorded during robot manipulation data collection. "
    "Describe only what is visible in the video."
)

OUTPUT_INSTRUCTION: Final = (
    "Describe this clip in two to four sentences. Cover the manipulated objects, the hand or "
    "gripper motion, and how the scene changes from start to end. Write plain declarative prose "
    "with no preamble and no bullet points. Treat the context above as background only: do not "
    "restate it, and do not mention the camera, the recording, or the annotation task."
)


def normalize_context_value(value: object) -> str | None:
    """Return a single-line context value, or ``None`` when absent.

    Collapsing whitespace keeps a multi-line value from breaking the
    ``Label: value`` line structure, which is also what stops free text from
    injecting its own fake label lines into the prompt.

    Args:
        value: Raw column value; anything non-string is treated as absent.

    Returns:
        The normalized value, or ``None`` when it is absent or blank.

    """
    if not isinstance(value, str):
        return None
    collapsed = " ".join(value.split())
    if not collapsed:
        return None
    return collapsed


def render_prompt(context: Mapping[str, object], fields: Sequence[PromptContextField]) -> str:
    """Render the egocentric caption prompt for one clip's context.

    Args:
        context: Row values keyed by the resolved fields' column names. Missing,
            null, or blank values are omitted from the rendered prompt.
        fields: The context fields this run resolved against its source snapshot,
            in render order. Passed in rather than read from
            :data:`SUPPORTED_CONTEXT_FIELDS` so that one attempt renders exactly
            the subset its digest was computed over.

    Returns:
        The prompt text, byte-identical for byte-identical input.

    """
    lines = [SYSTEM_INSTRUCTION, ""]
    rendered = [
        f"{field.label}: {value}"
        for field in fields
        if (value := normalize_context_value(context.get(field.column))) is not None
    ]
    if rendered:
        # An absent field is dropped entirely rather than rendered as a bare
        # "Camera motion:" line. An empty labelled line teaches the model that
        # absence is itself a value, and it reliably provokes commentary about the
        # missing annotation instead of about the video.
        lines.extend(rendered)
        lines.append("")
    lines.append(OUTPUT_INSTRUCTION)
    return "\n".join(lines)
