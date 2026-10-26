# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Video-cut's append contract: its bindings, terminal source outcomes, and span expansion.

Shared publication and recovery behavior is covered by ``test_conformance.py``.
"""

from typing import Any

import pyarrow as pa
import pytest

from cosmos_curator.next.append.contracts import canonical_results
from cosmos_curator.next.media.ffmpeg import VideoMetadata
from cosmos_curator.next.recipes.video_cut.bindings import resolve_bindings
from cosmos_curator.next.recipes.video_cut.config import ResolvedVideoCutConfig
from cosmos_curator.next.recipes.video_cut.contracts import (
    SOURCE_SCHEMA,
    SPAN_SCHEMA,
    expand_sources,
    make_source_id,
    media_fields,
    validate_source_results,
)
from cosmos_curator.next.recipes.video_cut.fixed_stride import planned_source
from tests.cosmos_curator.next.recipes.video_cut.conftest import plan_outcome


def test_bindings_bind_spans_to_the_cutting_contract(config: ResolvedVideoCutConfig) -> None:
    """The cutting contract is the result contract, and source plans stage exactly as Parquet reads them."""
    spec = resolve_bindings(config).spec
    assert (spec.owner, spec.id_column) == ("video-cut", "source_id")
    assert spec.result_contract == config.split.cutting_contract
    assert not spec.staged_schema.equals(SOURCE_SCHEMA, check_metadata=True)
    assert spec.staged_schema.field("spans").type.value_field.name == "element"
    assert spec.output_fields.equals(SPAN_SCHEMA)
    assert spec.error_column == "error_stage"


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"stream_index": 1}, "first video stream"),
        ({"source_duration_ns": None}, "probed plans or described terminal errors"),
        ({"error_message": "bad"}, "probed plans or described terminal errors"),
        ({"error_stage": "source-probe"}, "probed plans or described terminal errors"),
        ({"error_stage": "source-probe", "error_message": "bad"}, "probed plans or described terminal errors"),
    ],
    ids=["stream", "unprobed-success", "message-without-stage", "stage-without-message", "error-with-spans"],
)
def test_only_terminal_outcomes_under_the_contract_publish(
    config: ResolvedVideoCutConfig, change: dict[str, Any], message: str
) -> None:
    """Checkpointed source plans must plan the first video stream and be terminal."""
    row = plan_outcome("s3://test-bucket/long.mp4", config) | change
    batch = pa.RecordBatch.from_pylist([row], schema=SOURCE_SCHEMA)
    with pytest.raises(ValueError, match=message):
        validate_source_results(batch)


def test_expansion_keeps_spans_with_their_sources_across_chunks(config: ResolvedVideoCutConfig) -> None:
    """Empty plans, errors, and chunk boundaries never shift a span to another source."""
    rows = [
        planned_source(
            {
                "source_id": make_source_id(f"s3://test-bucket/{index}.mp4"),
                "source_uri": f"s3://test-bucket/{index}.mp4",
            }
            | media_fields(VideoMetadata(duration, 64 + index, 48, 30.0, None, "h264"), 50),
            config.split,
        )
        for index, duration in enumerate([100, 5_000_000_000, 100, 3_000_000_000])
    ]
    rows[2] |= {"error_stage": "source-probe", "error_message": "bad media"}
    spec = resolve_bindings(config).spec
    outcomes = canonical_results(pa.Table.from_pylist(rows, schema=spec.staged_schema), spec)
    expanded = list(expand_sources(outcomes, max_rows=2))
    assert [batch.num_rows for batch in expanded] == [2, 2, 2, 1]
    assert expanded[0]["error_stage"].to_pylist() == [None, "source-probe"]
    assert expanded[0]["span_id"].null_count == 2
    assert expanded[0]["error_message"].to_pylist() == [None, "bad media"]
    expected = [
        {field.name: span[field.name] if field.name in span else row.get(field.name) for field in SPAN_SCHEMA}
        for row in rows
        if row["error_stage"] is None
        for span in row["spans"]
    ]
    assert pa.concat_tables(expanded[1:]).equals(pa.Table.from_pylist(expected, schema=SPAN_SCHEMA))
