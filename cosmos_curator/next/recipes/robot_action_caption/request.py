# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build engine requests and canonicalize one terminal staged schema.

The preprocess hook pre-seeds output keys because Ray bypasses postprocess for
failed rows. The postprocess hook re-emits that exact key set, and
``canonicalize`` alone authors published caption values. This module is pure:
it owns neither Ray execution nor Lance publication.

See ``docs/curator/design/curator-next-robot-action-caption.md`` for the bypass
flow, schema invariant, and single-writer rationale.
"""

from typing import Any, Final

import pyarrow as pa

from cosmos_curator.next.recipes.robot_action_caption.config import RobotActionCaptionGenerationConfig
from cosmos_curator.next.recipes.robot_action_caption.contracts import (
    CAPTION_COLUMNS,
    CAPTION_FIELDS,
    DO_SAMPLE_FRAMES,
    REQUEST_CONTENT_TYPES,
    REQUEST_ROLE,
    TEXT_CONTENT_TYPE,
    VIDEO_CONTENT_TYPE,
    CaptionStatus,
    caption_status,
    terminal_row,
)
from cosmos_curator.next.recipes.robot_action_caption.datasource import (
    CHECKPOINT_ID_COLUMN,
    FRAGMENT_ID_COLUMN,
    PROMPT_COLUMN,
)
from cosmos_curator.next.recipes.robot_action_caption.media import (
    MEDIA_ERROR_COLUMN,
    MEDIA_URL_COLUMN,
)
from cosmos_curator.next.recipes.robot_action_caption.table import CLIP_ID_COLUMN
from cosmos_curator.next.utils.identity import canonical_digest

# Ray's LLM processor names the failure channel. Also the flag the processor's own
# stages read to decide whether to bypass a row, so setting it in preprocess is
# how a media fault skips the GPU entirely instead of being sent to the engine.
# Routing a media fault into it is what makes a row-local media fault and a
# row-local inference fault travel one channel and be published by one code path.
INFERENCE_ERROR_COLUMN: Final = "__inference_error__"

# The vLLM engine stage's generation output keys. The prompt tally is one of them
# because the terminal classifier needs it and not because anything here reads
# it: the context length bounds prompt PLUS generation, so the generated count on
# its own cannot say whether a generation ran out of room.
GENERATED_TEXT_COLUMN: Final = "generated_text"
GENERATED_TOKENS_COLUMN: Final = "num_generated_tokens"
INPUT_TOKENS_COLUMN: Final = "num_input_tokens"

# Row identity that must reach the staged Parquet. ``__checkpoint_id`` is the one
# that is load-bearing rather than merely useful: Ray marks a row complete by
# reading that column back off the written output, so dropping it here would
# checkpoint nothing while the run appeared to work.
_IDENTITY_COLUMNS: Final[tuple[str, ...]] = (CLIP_ID_COLUMN, FRAGMENT_ID_COLUMN, CHECKPOINT_ID_COLUMN)

# The exact key set that survives ``build_request`` and that ``read_generation``
# re-emits. Defined once because it is the same set on both engine outcomes; two
# definitions is precisely how the error and success shapes drift apart.
_STAGED_KEYS: Final[frozenset[str]] = frozenset(
    {
        *_IDENTITY_COLUMNS,
        *CAPTION_COLUMNS,
        GENERATED_TEXT_COLUMN,
        GENERATED_TOKENS_COLUMN,
        INPUT_TOKENS_COLUMN,
        INFERENCE_ERROR_COLUMN,
    }
)

STAGED_SCHEMA: Final = pa.schema(
    [
        pa.field(CLIP_ID_COLUMN, pa.string(), nullable=False),
        pa.field(FRAGMENT_ID_COLUMN, pa.int64(), nullable=False),
        pa.field(CHECKPOINT_ID_COLUMN, pa.uint64(), nullable=False),
        *CAPTION_FIELDS,
    ]
)

_EMPTY_GENERATION_ERROR: Final = "the engine returned no caption text"


def staged_schema_fingerprint() -> str:
    """Hash the ordered persisted field contract for recovery identity."""
    return canonical_digest(
        [
            {
                "name": field.name,
                "type": str(field.type),
                "nullable": field.nullable,
            }
            for field in STAGED_SCHEMA
        ]
    )


def build_request(
    row: dict[str, Any],
    *,
    generation: RobotActionCaptionGenerationConfig,
) -> dict[str, Any]:
    """Build one engine request, or mark the row so it bypasses inference.

    Args:
        row: One scanned-and-fetched clip row. Mutated: see below.
        generation: Sampling values recorded in the semantic contract.

    Returns:
        The engine keys plus the pre-seeded output columns, or - when the media
        stage reported a fault - only the pre-seeded columns with the failure
        channel set.

    Raises:
        TypeError: If a row without a media fault carries no media URL string,
            which means the media stage broke its own contract.

    """
    identity = {name: row[name] for name in _IDENTITY_COLUMNS}
    prompt_text = row[PROMPT_COLUMN]
    media_url = row.get(MEDIA_URL_COLUMN)
    media_fault = row.get(MEDIA_ERROR_COLUMN)

    # ``wrap_preprocess`` sets the processor's data dict to THIS object and then
    # merges the return value into it, so a scan column that is neither dropped
    # here nor returned still reaches the output - and because an error row's
    # output IS that dict, it would appear on error rows and be absent from
    # success rows. Dropping everything outside the staged key set is what keeps
    # the two shapes one schema. It also removes the duplicate top-level alias of
    # the base64 URL; the message below retains that URL until the multimodal
    # preparation stage consumes it.
    for name in [name for name in row if name not in _STAGED_KEYS]:
        del row[name]

    # Seeded as NULL rather than as a plausible-looking status: status NULL is an
    # explicit arm of the pending predicate, so a row that reached publication
    # without passing through canonicalize stays pending instead of carrying a
    # verdict nothing computed. Load-bearing on BOTH engine outcomes, because
    # canonicalize is
    # the sole writer of the caption columns: a bypassed row's output IS this
    # dict, so without the seed an error row would reach the Parquet write five
    # columns narrower than a success row.
    seeded = {
        **identity,
        **dict.fromkeys(CAPTION_COLUMNS),
        GENERATED_TEXT_COLUMN: None,
        GENERATED_TOKENS_COLUMN: None,
        INPUT_TOKENS_COLUMN: None,
    }
    if media_fault is not None:
        return {**seeded, INFERENCE_ERROR_COLUMN: str(media_fault)}
    if not isinstance(media_url, str):
        msg = f"clip {identity[CLIP_ID_COLUMN]!r} has no media fault and no {MEDIA_URL_COLUMN} string"
        raise TypeError(msg)
    # Each part is looked up under the shared order rather than written out in
    # order, so the sequence the model reads is the sequence the digest stamps.
    # A part's own type names its payload key, which is the chat content-part
    # convention the engine expects, so one constant spells both. A type added to
    # REQUEST_CONTENT_TYPES with no payload here raises ``KeyError``, and that is
    # the loud half of the coupling: the alternative is a request that reaches
    # the engine quietly missing a content part.
    payloads: dict[str, Any] = {
        VIDEO_CONTENT_TYPE: {"url": media_url},
        TEXT_CONTENT_TYPE: prompt_text,
    }
    return {
        **seeded,
        INFERENCE_ERROR_COLUMN: "",
        "messages": [
            {
                "role": REQUEST_ROLE,
                "content": [{"type": name, name: payloads[name]} for name in REQUEST_CONTENT_TYPES],
            }
        ],
        "sampling_params": {
            "temperature": generation.temperature,
            "top_p": generation.top_p,
            "max_tokens": generation.max_output_tokens,
        },
        # The frame BUDGET is not here: Ray's multimodal stage decodes and samples
        # from the engine config's media_io_kwargs, so the configured frame count
        # belongs to the processor build. This flag stops the model processor
        # re-sampling frames that stage already selected.
        "mm_processor_kwargs": {"do_sample_frames": DO_SAMPLE_FRAMES},
    }


def read_generation(row: dict[str, Any]) -> dict[str, Any]:
    """Map one successful generation onto the same key set an error row carries.

    Args:
        row: The processor's data dict after the engine stages.

    Returns:
        Exactly the keys :func:`build_request` pre-seeded. ``__inference_error__``
        is included and empty on purpose: a bypassed row carries it, so omitting
        it here would make the success and error shapes differ by one column.

    """
    return {
        **{name: row[name] for name in _IDENTITY_COLUMNS},
        **{name: row.get(name) for name in CAPTION_COLUMNS},
        GENERATED_TEXT_COLUMN: row[GENERATED_TEXT_COLUMN],
        GENERATED_TOKENS_COLUMN: row[GENERATED_TOKENS_COLUMN],
        INPUT_TOKENS_COLUMN: row[INPUT_TOKENS_COLUMN],
        INFERENCE_ERROR_COLUMN: "",
    }


def canonicalize(
    batch: pa.Table,
    *,
    digest: str,
    model_id: str,
    generation: RobotActionCaptionGenerationConfig,
) -> pa.Table:
    """Turn both engine row shapes into the one staged schema.

    Args:
        batch: Rows carrying the key set :func:`build_request` established.
        digest: The run's contract digest, stamped on every terminal row.
        model_id: The resolved model id, stamped on every terminal row.
        generation: Sampling and context bounds used by the engine.

    Returns:
        A :data:`STAGED_SCHEMA` table: row identity plus the five caption columns.

    Raises:
        TypeError: If a clip id is not a string or a successful token count is
            not a non-bool integer.
        ValueError: If a required identity column is missing, a clip id is
            blank, or a successful token count is negative.

    """
    missing = sorted(set(_IDENTITY_COLUMNS) - set(batch.schema.names))
    if missing:
        msg = f"canonicalize is missing required identity column(s): {', '.join(missing)}"
        raise ValueError(msg)
    raws = batch.to_pylist()
    rows = [
        {
            CLIP_ID_COLUMN: _clip_id(raw[CLIP_ID_COLUMN]),
            FRAGMENT_ID_COLUMN: int(raw[FRAGMENT_ID_COLUMN]),
            CHECKPOINT_ID_COLUMN: int(raw[CHECKPOINT_ID_COLUMN]),
            **_terminal_values(
                raw,
                digest=digest,
                model_id=model_id,
                generation=generation,
            ),
        }
        for raw in raws
    ]
    return pa.Table.from_pylist(rows, schema=STAGED_SCHEMA)


def _terminal_values(
    raw: dict[str, Any],
    *,
    digest: str,
    model_id: str,
    generation: RobotActionCaptionGenerationConfig,
) -> dict[str, Any]:
    """Classify one row into its terminal caption values."""
    failure = str(raw.get(INFERENCE_ERROR_COLUMN) or "")
    if failure:
        return terminal_row(status=CaptionStatus.ERROR, digest=digest, model_id=model_id, error=failure)
    generated_token_count = _token_count(raw, GENERATED_TOKENS_COLUMN)
    prompt_token_count = _token_count(raw, INPUT_TOKENS_COLUMN)
    text = raw.get(GENERATED_TEXT_COLUMN)
    caption = text if isinstance(text, str) else None
    status = caption_status(
        generated_token_count=generated_token_count,
        prompt_token_count=prompt_token_count,
        text=caption,
        generation=generation,
    )
    if status is CaptionStatus.ERROR:
        return terminal_row(
            status=CaptionStatus.ERROR,
            digest=digest,
            model_id=model_id,
            error=_EMPTY_GENERATION_ERROR,
        )
    return terminal_row(status=status, digest=digest, model_id=model_id, caption=caption)


def _clip_id(value: object) -> str:
    """Return a nonblank string clip id without coercion or normalization."""
    if not isinstance(value, str):
        msg = f"{CLIP_ID_COLUMN} must be a non-empty string, got {type(value).__name__}"
        raise TypeError(msg)
    if not value.strip():
        msg = f"{CLIP_ID_COLUMN} must be a non-empty string"
        raise ValueError(msg)
    return value


def _token_count(row: dict[str, Any], column: str) -> int:
    """Return one required non-negative engine token count."""
    value = row[column]
    if not isinstance(value, int) or isinstance(value, bool):
        msg = f"{column} must be a non-negative integer, got {type(value).__name__}"
        raise TypeError(msg)
    if value < 0:
        msg = f"{column} must be a non-negative integer, got {value}"
        raise ValueError(msg)
    return value
