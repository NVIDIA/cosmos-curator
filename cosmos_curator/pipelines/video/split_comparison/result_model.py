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
"""Result-model types for split output comparison: ``Issue`` rows and ``Report``.

The pipeline's output contract: every comparator emits :class:`Issue` rows
that conform to the Arrow ``ISSUE_SCHEMA`` (the canonical issue contract),
constructed via :func:`make_issue`; the driver assembles those rows into a
single :class:`Report`. Pairs with :mod:`...config`, which owns the input
contract. See ``docs/curator/design/split-comparison.md``.
"""

import json
from typing import Any, Literal, TypedDict

import attrs
import pyarrow as pa

from cosmos_curator.pipelines.video.split_comparison.config import SplitComparisonConfig

ISSUE_SCHEMA: pa.Schema = pa.schema(
    [
        ("code", pa.dictionary(pa.int16(), pa.string())),
        ("message", pa.string()),
        ("feature", pa.string()),
        ("video", pa.string()),
        ("clip", pa.string()),
        ("output", pa.string()),
        ("field", pa.string()),
        ("details", pa.string()),  # JSON-encoded
    ],
)

# Per-video identity sidecar: one row per video in the union of both
# summaries, carrying only the join key + per-side presence flags. Full
# source paths are reconstructed by consumers as ``Report.source_a + video_key``
# (or ``source_b + video_key``); the comparator derives those roots by
# stripping each summary's ``source_video`` of its ``video_key`` suffix and
# asserting the layout is uniform per side (see
# ``driver._derive_videos_and_sources``). When the assertion fails for a
# side, that side's source root is empty and a
# ``summary_source_layout_inconsistent`` issue is emitted.
VIDEOS_SCHEMA: pa.Schema = pa.schema(
    [
        ("video_key", pa.string()),
        ("in_a", pa.bool_()),
        ("in_b", pa.bool_()),
    ],
)

# Universe of issue codes. Add new codes here when a comparator emits one.
IssueCode = Literal[
    "summary_field_mismatch",
    "summary_load_failed",
    "summary_source_layout_inconsistent",
    "summary_video_only_in_a",
    "summary_video_only_in_b",
    "summary_video_processed_state_mismatch",
    "summary_video_field_mismatch",
    "summary_clip_uuid_set_mismatch",
    "metadata_one_sided",
    "metadata_unreadable",
    "metadata_value_invalid_type",
    "metadata_value_one_sided",
    "aesthetic_score_mismatch",
    "motion_score_mismatch",
    "caption_similarity_below_threshold",
    "clip_mp4_missing",
    "clip_mp4_unreadable",
    "clip_mp4_header_index_unavailable",
    "clip_mp4_index_mismatch",
    "clip_mp4_index_dtype_mismatch",
    "clip_mp4_metadata_mismatch",
    "clip_mp4_comparison_failed",
]


class Issue(TypedDict, total=False):
    """Row shape for the Arrow issue table.

    Carries no methods; the table is the canonical issue representation. Construct
    rows via :func:`make_issue` so keyword names are checked and ``details`` is
    JSON-encoded at the call site.
    """

    code: str
    message: str
    feature: str | None
    video: str | None
    clip: str | None
    output: str | None
    field: str | None
    details: str | None  # JSON-encoded


def make_issue(  # noqa: PLR0913 -- ISSUE_SCHEMA has 8 columns; helper mirrors them as kwargs
    code: IssueCode,
    message: str,
    *,
    feature: str | None = None,
    video: str | None = None,
    clip: str | None = None,
    output: str | None = None,
    field: str | None = None,
    details: dict[str, Any] | None = None,
) -> Issue:
    """Build a schema-compatible :class:`Issue` row.

    Keyword-only args enforce field names; ``details`` is JSON-encoded so the
    resulting row fits ``ISSUE_SCHEMA`` directly.
    """
    return Issue(
        code=code,
        message=message,
        feature=feature,
        video=video,
        clip=clip,
        output=output,
        field=field,
        details=json.dumps(details, sort_keys=True) if details is not None else None,
    )


def empty_issues() -> pa.Table:
    """Return an empty issue table that still carries ``ISSUE_SCHEMA``.

    Used in places that need to express "no issues emitted by this stage" while
    keeping :func:`pyarrow.concat_tables` schema-aligned downstream.
    """
    return pa.Table.from_pylist([], schema=ISSUE_SCHEMA)


def empty_videos() -> pa.Table:
    """Return an empty videos table that still carries ``VIDEOS_SCHEMA``.

    Used as the default for :attr:`Report.videos` and as the load-failure
    fallback when :func:`compare_split_outputs` can't read either summary.
    """
    return pa.Table.from_pylist([], schema=VIDEOS_SCHEMA)


@attrs.define(frozen=True)
class Report:
    """Result of one comparison run.

    Attributes:
        issues: Arrow table conforming to :data:`ISSUE_SCHEMA`. Empty when the run
            passed.
        videos: Arrow table conforming to :data:`VIDEOS_SCHEMA`. One row per
            video encountered (union of both outputs' summaries), carrying
            ``video_key`` plus per-side presence booleans. Source paths are
            *not* stored per row -- reconstruct as
            ``f"{report.source_a}{row['video_key']}"`` (or ``source_b``).
            Empty when the load-failure short-circuit fires.
        source_a / source_b: Source-video roots derived per side from the
            corresponding summary. The comparator strips ``video_key`` off
            each entry's ``source_video`` and asserts every row in that side
            yields the same root; if the assertion fails the value is ``""``
            and a ``summary_source_layout_inconsistent`` issue is emitted.
        passed: ``True`` iff ``issues.num_rows == 0``. Provided alongside the table
            so callers don't have to import pyarrow just to ask.
        stages_run: Set of stage names that actually executed. A ``passed`` report
            with ``stages_run={"summary", "metadata"}`` reads honestly as
            "metadata agrees" rather than "everything was checked" -- relevant
            when ``config.compare_video_index`` is False.
        clip_count: Number of clips actually compared (post ``config.video_key``
            filter and ``config.clip_limit`` slice). Drives
            ``total_clips_compared`` in the persisted report summary; counts
            each clip once across both outputs (set union, not sum).
        clips_in_a / clips_in_b / clips_in_both: Per-output breakdown over the same
            filtered/sliced clip table, so the summary can disambiguate the union
            count from one-sided membership.
        output_a / output_b: The two output roots that were compared, echoed
            back into the persisted report.
        runtime_sec: Wall-clock seconds spent inside ``compare_split_outputs``
            (driver + stages); persisted under ``summary.runtime_sec``. Zero
            when callers (e.g. tests) construct a Report directly.
        config: The ``SplitComparisonConfig`` that produced this report; serialized
            into the persisted output so a reader can tell which flags / tolerances
            / filters were in effect. ``None`` only when callers (e.g. tests)
            build a Report directly without going through ``compare_split_outputs``.

    """

    issues: pa.Table = attrs.field(eq=False)
    passed: bool
    stages_run: frozenset[str]
    videos: pa.Table = attrs.field(eq=False, factory=empty_videos)
    clip_count: int = 0
    clips_in_a: int = 0
    clips_in_b: int = 0
    clips_in_both: int = 0
    output_a: str = ""
    output_b: str = ""
    source_a: str = ""
    source_b: str = ""
    runtime_sec: float = 0.0
    config: SplitComparisonConfig | None = None
