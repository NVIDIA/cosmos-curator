# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Define caption columns, terminal states, and the semantic digest.

The five flat, nullable columns follow the existing Curate convention on
``clips.lance`` rather than the caption struct used by ``video_caption`` on
``video_split``.

The digest answers whether an output remains valid under this run's semantics. It
covers result-defining values and excludes execution policy, so concurrency or
retry tuning cannot discard completed GPU work through pending or recovery gates.

Its five non-constant inputs come from their owning boundaries::

    RegisteredModel      <- the shipped registry (models.registry)
    context fields       <- resolved against the SOURCE SNAPSHOT (table.py)
    generation           <- operator configuration (config.generation)
    video                <- operator configuration (config.video)
    media                <- operator configuration (config.media)

Adding a supported source context column grows the resolved subset and recaptions
the configured input scope. That is intentional because existing captions
omitted that context, but it must be scheduled rather than treated as incidental.
"""

import enum
import types
from collections.abc import Sequence
from typing import Any, Final

import pyarrow as pa

from cosmos_curator.models.registry import RegisteredModel
from cosmos_curator.next.recipes.robot_action_caption import prompt as prompt_contract
from cosmos_curator.next.recipes.robot_action_caption.config import (
    RobotActionCaptionGenerationConfig,
    RobotActionCaptionMediaConfig,
    RobotActionCaptionVideoConfig,
)
from cosmos_curator.next.recipes.robot_action_caption.prompt import (
    OUTPUT_INSTRUCTION,
    SYSTEM_INSTRUCTION,
    PromptContextField,
)
from cosmos_curator.next.utils.identity import canonical_digest

# Version of terminal classification policy: comparison boundaries, fallback
# rules, and the deterministic-fault-versus-systemic-failure boundary. Numeric
# truncation bounds are also recorded explicitly in the digest.
TERMINAL_POLICY_VERSION: Final = 1

CAPTION_COLUMN: Final = "caption"
CAPTION_STATUS_COLUMN: Final = "caption_status"
CAPTION_ERROR_COLUMN: Final = "caption_error"
CAPTION_DIGEST_COLUMN: Final = "caption_contract_digest"
CAPTION_MODEL_COLUMN: Final = "caption_model"

CAPTION_COLUMNS: Final[tuple[str, ...]] = (
    CAPTION_COLUMN,
    CAPTION_STATUS_COLUMN,
    CAPTION_ERROR_COLUMN,
    CAPTION_DIGEST_COLUMN,
    CAPTION_MODEL_COLUMN,
)

# Every field is nullable so the whole set can be added to an existing table as a
# metadata-only `add_columns`, with no data-file rewrite and no backfill.
CAPTION_FIELDS: Final = pa.schema(
    [
        pa.field(CAPTION_COLUMN, pa.large_string(), nullable=True),
        pa.field(CAPTION_STATUS_COLUMN, pa.string(), nullable=True),
        pa.field(CAPTION_ERROR_COLUMN, pa.large_string(), nullable=True),
        pa.field(CAPTION_DIGEST_COLUMN, pa.string(), nullable=True),
        pa.field(CAPTION_MODEL_COLUMN, pa.string(), nullable=True),
    ]
)


class CaptionStatus(enum.StrEnum):
    """The terminal verdict a published caption row carries.

    The three values are a persisted format contract: they are written to
    ``caption_status`` and read back by operator queries over the table. NULL is
    not a member and never will be - it means no run has claimed the row yet and
    is one of the full caption-group pending predicate's fail-closed states.
    """

    SUCCESS = "success"
    TRUNCATED = "truncated"
    ERROR = "error"


def validate_terminal_values(
    *,
    status: object,
    caption: object,
    error: object,
) -> CaptionStatus:
    """Validate and normalize one terminal caption verdict.

    Args:
        status: Stored status string or in-process enum member.
        caption: Generated caption value.
        error: Failure description value.

    Returns:
        The normalized terminal status.

    Raises:
        ValueError: If the status is not terminal or its caption/error null
            pairing is invalid.

    """
    if isinstance(status, CaptionStatus):
        terminal_status = status
    elif isinstance(status, str):
        try:
            terminal_status = CaptionStatus(status)
        except ValueError as e:
            msg = f"expected a terminal caption status, got {status!r}"
            raise ValueError(msg) from e
    else:
        msg = f"expected a terminal caption status, got {status!r}"
        # This is durable-row validation, not typed API dispatch. One refusal
        # category lets publication wrap every malformed stored value uniformly.
        raise ValueError(msg)  # noqa: TRY004

    if terminal_status is CaptionStatus.ERROR:
        if caption is not None:
            msg = "an error caption row requires a null caption"
            raise ValueError(msg)
        if not isinstance(error, str) or not error.strip():
            msg = "an error caption row requires a nonblank string error"
            raise ValueError(msg)
        return terminal_status

    if not isinstance(caption, str) or not caption.strip():
        msg = f"a {terminal_status} caption row requires a nonblank string caption"
        raise ValueError(msg)
    if error is not None:
        msg = f"a {terminal_status} caption row requires a null error"
        raise ValueError(msg)
    return terminal_status


# The shape of the engine turn. These live HERE, beside the digest that stamps
# them, and are imported by `request.py`, `media.py`, `inference.py`, and
# `engine.py` rather than restated there. The direction is the whole guarantee:
# the digest decides which published rows are still valid, so a value it stamps
# has to be the same object the run used. Move one back down into the module
# that consumes it and the digest is stamping a copy - editing that copy would
# then change every caption while the digest stayed byte-identical, leaving two
# populations in one table with nothing to tell them apart.
REQUEST_ROLE: Final = "user"

# Qwen3 templates prefill a thinking turn into the generation prompt, so the
# trace arrives as caption text without an opening tag to remove. Templates
# without this switch ignore it. The digest reads this same mapping so captions
# produced before the policy changed are not reused.
CHAT_TEMPLATE_KWARGS: Final = types.MappingProxyType({"enable_thinking": False})

VIDEO_CONTENT_TYPE: Final = "video_url"
TEXT_CONTENT_TYPE: Final = "text"

# Render ORDER, not merely membership: the model reads the clip before the
# instruction text, and swapping the two changes the generated caption.
REQUEST_CONTENT_TYPES: Final[tuple[str, ...]] = (VIDEO_CONTENT_TYPE, TEXT_CONTENT_TYPE)

# The RFC 2397 media type of the base64 `data:` URL a clip travels in. Fixed
# because the upstream splitter writes MP4 exclusively, so any other container
# would change the split contract.
MEDIA_TYPE: Final = "video/mp4"

# One clip per request. Passed to the engine explicitly so a model default that
# admitted a second video could not silently change what a caption describes.
VIDEO_ITEMS_PER_PROMPT: Final = 1

# The processor's media preparation stage is what turns a `video_url` in a
# message into engine input at all; without it the request reaches the engine as
# a placeholder-bearing prompt carrying no video.
PREPARE_MULTIMODAL: Final = True

# Without this the model processor re-samples frames we already selected. A short
# clip is the visible case: it arrives with every frame retained and is then
# sampled again, so the frames the caption describes are not the frames we
# budgeted or digested.
DO_SAMPLE_FRAMES: Final = False


def normalized_caption_contract(
    model: RegisteredModel,
    context_fields: Sequence[PromptContextField],
    *,
    generation: RobotActionCaptionGenerationConfig,
    video: RobotActionCaptionVideoConfig,
    media: RobotActionCaptionMediaConfig,
) -> dict[str, Any]:
    """Return every result-defining value in a JSON-compatible, order-stable shape.

    Args:
        model: The resolved model identity.
        context_fields: The context subset resolved against this run's source
            snapshot, in render order.
        generation: Result-defining sampling and context limits.
        video: Result-defining video settings, digested as the exact mappings
            the engine receives.
        media: Result-defining media admission policy.

    Returns:
        The digest payload. Absent from it, deliberately: engine replica count,
        concurrent-batch count, media batch size, CPU reservations, retry budgets
        (``execution.media_attempts``), and the staged field set and staging
        protocol, which ``workspace.resolve_workspace`` hashes into the recovery
        namespace instead - none of which move the generated text.

    """
    return {
        "model": {"id": model.model_id, "revision": model.revision, "precision": model.precision},
        "prompt": {
            "version": prompt_contract.PROMPT_VERSION,
            "system": SYSTEM_INSTRUCTION,
            "output": OUTPUT_INSTRUCTION,
            "normalization": "collapse_whitespace_drop_null_or_blank",
            "context": [{"column": field.column, "label": field.label} for field in context_fields],
        },
        # The request skeleton is digested because it decides what the model is
        # shown, not merely how the bytes get there: dropping the video item or
        # reordering content changes the generated text.
        "request": {
            "role": REQUEST_ROLE,
            "content": list(REQUEST_CONTENT_TYPES),
            "media_type": MEDIA_TYPE,
            "video_items_per_prompt": VIDEO_ITEMS_PER_PROMPT,
            "prepare_multimodal": PREPARE_MULTIMODAL,
            "chat_template_kwargs": dict(CHAT_TEMPLATE_KWARGS),
        },
        # ``max_model_len`` is sampling rather than execution because it is the
        # other bound on where a generation stops - see the field's decision
        # constraint in config.py for the mode in which it cuts a caption without
        # anything failing.
        "sampling": {
            "temperature": generation.temperature,
            "top_p": generation.top_p,
            "max_tokens": generation.max_output_tokens,
            "max_model_len": generation.max_model_len,
        },
        # Digest the two mappings AS PRODUCED, not a reconstructed dictionary.
        # ``processor_config`` consumes these same methods, so removing, adding,
        # or renaming an engine-facing key changes identity with the executed
        # semantics. Either mapping may be empty; that records backend/artifact
        # ownership of the omitted override group without inventing a value.
        "video": {
            "media_io_kwargs": video.media_io_kwargs(),
            "mm_processor_kwargs": video.mm_processor_kwargs(),
            "do_sample_frames": DO_SAMPLE_FRAMES,
        },
        "media": {"max_clip_bytes": media.max_clip_bytes},
        "terminal": {
            "policy_version": TERMINAL_POLICY_VERSION,
            # Sorted by ``.value`` rather than definition order, so renaming a
            # member is a deliberate digest change while a cosmetic reorder is
            # not. ``.value`` is taken explicitly so the hashed text stays
            # identical if the enum base ever changes.
            "states": sorted(status.value for status in CaptionStatus),
            # Both bounds `caption_status` measures against, either of which makes
            # a row `truncated`. The context length appears here as well as under
            # `sampling` because the two entries record different facts - that the
            # engine was built with it, and that the terminal verdict is computed
            # against it - so a run under a different cap or context length stamps
            # differently.
            #
            # These entries cover the bounds, never the comparison drawn against
            # them. Flipping `>=` to `>`, swapping the `min` in `caption_status`,
            # or changing the classifier's handling of counts re-verdicts every
            # caption while the digest stays byte-identical. Such a change has to
            # increment TERMINAL_POLICY_VERSION or leave the configured scope
            # needing a recaption to stay comparable.
            "truncated_at_or_above_generated_tokens": generation.max_output_tokens,
            "truncated_at_or_above_prompt_plus_generated_tokens": generation.max_model_len,
            "empty_generation": CaptionStatus.ERROR.value,
            "deterministic_media_fault": CaptionStatus.ERROR.value,
        },
    }


def caption_contract_digest(
    model: RegisteredModel,
    context_fields: Sequence[PromptContextField],
    *,
    generation: RobotActionCaptionGenerationConfig,
    video: RobotActionCaptionVideoConfig,
    media: RobotActionCaptionMediaConfig,
) -> str:
    """Return the digest identifying the semantics that produced a caption row.

    Args:
        model: The resolved model identity.
        context_fields: The context subset resolved against this run's snapshot.
        generation: Result-defining sampling and context limits.
        video: Result-defining video settings passed to the engine.
        media: Result-defining media admission policy.

    Returns:
        A 64-character lowercase hex sha256.

    """
    return canonical_digest(
        normalized_caption_contract(
            model,
            context_fields,
            generation=generation,
            video=video,
            media=media,
        )
    )


def terminal_row(
    *,
    status: CaptionStatus,
    digest: str,
    model_id: str,
    caption: str | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    """Build one terminal caption row, enforcing the published null-invariants.

    The shared validator rejects malformed rows here at construction and again at
    publication after a durable Parquet round trip.

    Args:
        status: The verdict to publish.
        digest: Contract digest for this run.
        model_id: Resolved model id, recorded for triage without parsing the digest.
        caption: Generated text. Required unless ``status`` is ``error``.
        error: Failure description. Required when ``status`` is ``error``.

    Returns:
        A mapping keyed by exactly :data:`CAPTION_COLUMNS`.

    Raises:
        ValueError: If the caption/error fields do not match ``status``.

    """
    status = validate_terminal_values(status=status, caption=caption, error=error)
    return {
        CAPTION_COLUMN: caption,
        CAPTION_STATUS_COLUMN: status,
        CAPTION_ERROR_COLUMN: error,
        CAPTION_DIGEST_COLUMN: digest,
        CAPTION_MODEL_COLUMN: model_id,
    }


def caption_status(
    *,
    generated_token_count: int,
    prompt_token_count: int,
    text: str | None,
    generation: RobotActionCaptionGenerationConfig,
) -> CaptionStatus:
    """Classify one generation into a terminal status.

    A generation is truncated when it reached either bound that can stop it - the
    sampling cap, or the context length net of the prompt. Both ask the same
    question, so both are measured as one ceiling on the generated count.

    Which bound was reached is inferred from the counts because the Ray Data LLM
    engine stage does not surface ``finish_reason``. A generation that happens
    to end naturally on a bound is indistinguishable from one the bound cut off,
    so an exact tie is conservatively reported as truncated.

    Args:
        generated_token_count: Tokens generated.
        prompt_token_count: Tokens occupied by the prompt.
        text: Generated text.
        generation: Sampling and context limits used for the request.

    Returns:
        The verdict for this generation.

    """
    if not text or not text.strip():
        return CaptionStatus.ERROR
    # One ceiling from two bounds: the sampling cap bounds the generated count
    # directly, while the context length bounds prompt PLUS generation, which is
    # the same bound on the generated count once the prompt's share is taken out.
    #
    # The prompt tally is the engine's own. If it counts multimodal placeholders
    # before expansion, subtraction leaves the ceiling above the real headroom
    # and this classifier can miss context-bound truncations. Do not replace that
    # observed tally with an unverified estimate: an estimate below the real
    # headroom would instead invent truncations.
    ceiling = min(generation.max_output_tokens, generation.max_model_len - prompt_token_count)
    # Equality is conservative only at this decision: without finish_reason, an
    # exact bound hit cannot be distinguished from a natural completion.
    if generated_token_count >= ceiling:
        return CaptionStatus.TRUNCATED
    return CaptionStatus.SUCCESS
