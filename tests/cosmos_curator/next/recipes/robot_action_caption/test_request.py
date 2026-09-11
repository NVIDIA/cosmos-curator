# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify request construction and terminal-row canonicalization."""

from collections.abc import Callable
from typing import Any

import pyarrow as pa
import pytest

from cosmos_curator.next.recipes.robot_action_caption import request as request_module
from cosmos_curator.next.recipes.robot_action_caption.config import RobotActionCaptionGenerationConfig
from cosmos_curator.next.recipes.robot_action_caption.contracts import (
    CAPTION_COLUMN,
    CAPTION_COLUMNS,
    CAPTION_DIGEST_COLUMN,
    CAPTION_ERROR_COLUMN,
    CAPTION_FIELDS,
    CAPTION_MODEL_COLUMN,
    CAPTION_STATUS_COLUMN,
    CaptionStatus,
)
from cosmos_curator.next.recipes.robot_action_caption.datasource import (
    CHECKPOINT_ID_COLUMN,
    FRAGMENT_ID_COLUMN,
    PROMPT_COLUMN,
)
from cosmos_curator.next.recipes.robot_action_caption.media import MEDIA_ERROR_COLUMN, MEDIA_URL_COLUMN
from cosmos_curator.next.recipes.robot_action_caption.request import (
    GENERATED_TEXT_COLUMN,
    GENERATED_TOKENS_COLUMN,
    INFERENCE_ERROR_COLUMN,
    INPUT_TOKENS_COLUMN,
    STAGED_SCHEMA,
    build_request,
    canonicalize,
    read_generation,
)
from cosmos_curator.next.recipes.robot_action_caption.table import CLIP_ID_COLUMN, CLIP_URI_COLUMN

DIGEST = "test-digest"
MODEL_ID = "test/model"
GENERATION = RobotActionCaptionGenerationConfig(
    temperature=0.17,
    top_p=0.63,
    max_output_tokens=73,
    max_model_len=509,
)

type FetchedRowFactory = Callable[..., dict[str, Any]]
type StagedRowFactory = Callable[..., dict[str, Any]]


@pytest.fixture
def fetched_row() -> FetchedRowFactory:
    """Return one row as the media stage leaves it."""

    def build(
        *,
        media_url: str | None = "data:video/mp4;base64,QUJD",
        media_error: str | None = None,
        prompt: str = "describe this clip",
    ) -> dict[str, Any]:
        return {
            CLIP_ID_COLUMN: "c0",
            FRAGMENT_ID_COLUMN: 1,
            CHECKPOINT_ID_COLUMN: (1 << 32) | 4,
            CLIP_URI_COLUMN: "s3://b/c0.mp4",
            PROMPT_COLUMN: prompt,
            MEDIA_URL_COLUMN: media_url,
            MEDIA_ERROR_COLUMN: media_error,
        }

    return build


@pytest.fixture
def staged_success(fetched_row: FetchedRowFactory) -> StagedRowFactory:
    """Return a successful engine row ready for canonicalization."""

    def build(
        *,
        text: str | None = "a caption",
        tokens: int | None = 12,
        prompt_tokens: int | None = 64,
    ) -> dict[str, Any]:
        request = build_request(fetched_row(), generation=GENERATION)
        return read_generation(
            {
                **request,
                GENERATED_TEXT_COLUMN: text,
                GENERATED_TOKENS_COLUMN: tokens,
                INPUT_TOKENS_COLUMN: prompt_tokens,
            }
        )

    return build


@pytest.fixture
def staged_bypass(fetched_row: FetchedRowFactory) -> dict[str, Any]:
    """Return a media-fault row that bypasses inference."""
    return build_request(
        fetched_row(media_url=None, media_error="clip object is unavailable"),
        generation=GENERATION,
    )


def _canonicalize_input(rows: list[dict[str, Any]]) -> pa.Table:
    """Build a table with production identity types."""
    types = {CHECKPOINT_ID_COLUMN: pa.uint64(), FRAGMENT_ID_COLUMN: pa.int64()}
    names = sorted({name for row in rows for name in row})
    return pa.table({name: pa.array([row.get(name) for row in rows], types.get(name)) for name in names})


def _canonicalize(rows: list[dict[str, Any]]) -> pa.Table:
    """Canonicalize rows under the test contract."""
    return canonicalize(
        _canonicalize_input(rows),
        digest=DIGEST,
        model_id=MODEL_ID,
        generation=GENERATION,
    )


def test_build_request_pre_seeds_every_output_column(fetched_row: FetchedRowFactory) -> None:
    """Give successful and bypassed paths the same output columns."""
    request = build_request(fetched_row(), generation=GENERATION)

    assert set(CAPTION_COLUMNS) <= set(request)
    assert all(request[name] is None for name in CAPTION_COLUMNS)
    assert request[CHECKPOINT_ID_COLUMN] == (1 << 32) | 4


def test_build_request_places_media_prompt_and_sampling_in_the_engine_request(
    fetched_row: FetchedRowFactory,
) -> None:
    """Translate one fetched row into the engine's chat request."""
    request = build_request(fetched_row(), generation=GENERATION)

    assert request["messages"] == [
        {
            "role": "user",
            "content": [
                {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,QUJD"}},
                {"type": "text", "text": "describe this clip"},
            ],
        }
    ]
    assert request["sampling_params"] == {"temperature": 0.17, "top_p": 0.63, "max_tokens": 73}
    assert request["mm_processor_kwargs"] == {"do_sample_frames": False}


def test_build_request_drops_source_and_media_transport_columns(fetched_row: FetchedRowFactory) -> None:
    """Keep only the shared staged keys beside the engine request."""
    request = build_request(fetched_row(), generation=GENERATION)

    assert CLIP_URI_COLUMN not in request
    assert PROMPT_COLUMN not in request
    assert MEDIA_URL_COLUMN not in request
    assert MEDIA_ERROR_COLUMN not in request


def test_build_request_marks_a_media_fault_for_engine_bypass(fetched_row: FetchedRowFactory) -> None:
    """Route a row-local media fault through the processor failure channel."""
    request = build_request(
        fetched_row(media_url=None, media_error="object unavailable"),
        generation=GENERATION,
    )

    assert request[INFERENCE_ERROR_COLUMN] == "object unavailable"
    assert "messages" not in request


def test_build_request_rejects_missing_media_without_a_fault(fetched_row: FetchedRowFactory) -> None:
    """Reject a row that violates the media stage's outcome contract."""
    with pytest.raises(TypeError, match="no media fault"):
        build_request(fetched_row(media_url=None), generation=GENERATION)


def test_build_request_leaves_prompt_admission_to_vllm(fetched_row: FetchedRowFactory) -> None:
    """Dispatch an arbitrarily long prompt without recipe-local token inference."""
    prompt = "x" * 100_000

    request = build_request(fetched_row(prompt=prompt), generation=GENERATION)

    assert request["messages"][0]["content"][1]["text"] == prompt
    assert request[INFERENCE_ERROR_COLUMN] == ""


def test_success_and_bypass_rows_share_one_key_set(
    staged_success: StagedRowFactory,
    staged_bypass: dict[str, Any],
) -> None:
    """Keep both engine outcomes compatible with one Arrow block schema."""
    assert set(staged_success()) == set(staged_bypass)


@pytest.mark.parametrize(
    ("tokens", "prompt_tokens", "expected"),
    [
        pytest.param(72, 64, CaptionStatus.SUCCESS, id="below-output-cap"),
        pytest.param(73, 64, CaptionStatus.TRUNCATED, id="at-output-cap"),
        pytest.param(12, 497, CaptionStatus.TRUNCATED, id="fills-context"),
    ],
)
def test_canonicalize_classifies_success_and_truncation(
    staged_success: StagedRowFactory,
    tokens: int,
    prompt_tokens: int,
    expected: CaptionStatus,
) -> None:
    """Classify terminal status from observed engine token counts."""
    table = _canonicalize([staged_success(tokens=tokens, prompt_tokens=prompt_tokens)])

    assert table.column(CAPTION_STATUS_COLUMN).to_pylist() == [expected]


def test_canonicalize_publishes_blank_generation_as_an_error(staged_success: StagedRowFactory) -> None:
    """Reject a successful engine row with no caption text."""
    table = _canonicalize([staged_success(text="")])

    assert table.column(CAPTION_STATUS_COLUMN).to_pylist() == [CaptionStatus.ERROR]
    assert table.column(CAPTION_COLUMN).to_pylist() == [None]
    assert table.column(CAPTION_ERROR_COLUMN).to_pylist() == ["the engine returned no caption text"]


def test_canonicalize_carries_a_bypassed_rows_error(staged_bypass: dict[str, Any]) -> None:
    """Publish a row-local media fault without reading absent token counts."""
    table = _canonicalize([staged_bypass])

    assert table.column(CAPTION_STATUS_COLUMN).to_pylist() == [CaptionStatus.ERROR]
    assert table.column(CAPTION_ERROR_COLUMN).to_pylist() == ["clip object is unavailable"]


def test_canonicalize_stamps_contract_identity_on_every_row(
    staged_success: StagedRowFactory,
    staged_bypass: dict[str, Any],
) -> None:
    """Stamp one digest and model identity across both terminal outcomes."""
    table = _canonicalize([staged_success(), staged_bypass])

    assert table.column(CAPTION_DIGEST_COLUMN).to_pylist() == [DIGEST, DIGEST]
    assert table.column(CAPTION_MODEL_COLUMN).to_pylist() == [MODEL_ID, MODEL_ID]
    assert table.column(CHECKPOINT_ID_COLUMN).type == pa.uint64()
    assert table.column(CHECKPOINT_ID_COLUMN).to_pylist() == [(1 << 32) | 4, (1 << 32) | 4]


def test_canonicalize_names_a_missing_identity_column(staged_success: StagedRowFactory) -> None:
    """Reject a terminal batch missing checkpoint identity."""
    row = staged_success()
    del row[CHECKPOINT_ID_COLUMN]

    with pytest.raises(ValueError, match=CHECKPOINT_ID_COLUMN):
        _canonicalize([row])


@pytest.mark.parametrize(
    "column",
    [GENERATED_TEXT_COLUMN, GENERATED_TOKENS_COLUMN, INPUT_TOKENS_COLUMN],
)
def test_read_generation_requires_each_engine_output_key(
    fetched_row: FetchedRowFactory,
    column: str,
) -> None:
    """Fail at the engine boundary when a required output is absent."""
    row = {
        **build_request(fetched_row(), generation=GENERATION),
        GENERATED_TEXT_COLUMN: "caption",
        GENERATED_TOKENS_COLUMN: 2,
        INPUT_TOKENS_COLUMN: 3,
    }
    del row[column]

    with pytest.raises(KeyError, match=column):
        read_generation(row)


@pytest.mark.parametrize(
    ("column", "value", "error"),
    [
        pytest.param(GENERATED_TOKENS_COLUMN, True, TypeError, id="bool-generated"),
        pytest.param(GENERATED_TOKENS_COLUMN, -1, ValueError, id="negative-generated"),
        pytest.param(INPUT_TOKENS_COLUMN, 1.5, TypeError, id="float-prompt"),
        pytest.param(INPUT_TOKENS_COLUMN, -1, ValueError, id="negative-prompt"),
    ],
)
def test_canonicalize_rejects_invalid_engine_token_counts(
    staged_success: StagedRowFactory,
    column: str,
    value: object,
    error: type[Exception],
) -> None:
    """Require non-bool, non-negative integer engine token counts."""
    row = staged_success()
    row[column] = value

    with pytest.raises(error, match=column):
        _canonicalize([row])


@pytest.mark.parametrize(
    ("clip_id", "error"),
    [
        pytest.param(7, TypeError, id="non-string"),
        pytest.param("   ", ValueError, id="blank"),
    ],
)
def test_canonicalize_rejects_invalid_clip_ids(
    staged_success: StagedRowFactory,
    clip_id: object,
    error: type[Exception],
) -> None:
    """Require a nonblank string clip identity."""
    row = staged_success()
    row[CLIP_ID_COLUMN] = clip_id

    with pytest.raises(error, match=CLIP_ID_COLUMN):
        _canonicalize([row])


def test_staged_schema_is_the_literal_phase_a_contract() -> None:
    """Pin the ordered Arrow schema written by Phase A."""
    assert (
        pa.schema(
            [
                pa.field(CLIP_ID_COLUMN, pa.string(), nullable=False),
                pa.field(FRAGMENT_ID_COLUMN, pa.int64(), nullable=False),
                pa.field(CHECKPOINT_ID_COLUMN, pa.uint64(), nullable=False),
                *CAPTION_FIELDS,
            ]
        )
        == STAGED_SCHEMA
    )


def test_staged_schema_fingerprint_changes_with_the_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bind recovery identity to the complete staged schema."""
    original = request_module.staged_schema_fingerprint()
    mutated = STAGED_SCHEMA.append(pa.field("new_field", pa.string()))

    monkeypatch.setattr(request_module, "STAGED_SCHEMA", mutated)

    assert request_module.staged_schema_fingerprint() != original
