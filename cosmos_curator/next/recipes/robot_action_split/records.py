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

"""Arrow contract for published clips, and a plain-dict projection for failures.

Mirrors ``video_split.records`` exactly: ``CLIP_SCHEMA`` holds only successful,
canonical clip rows (append-only, one row per ``clip_id``) so a retried failure
never collides with an already-committed row under the idempotent append
protocol in ``lance_fragment_recovery``. Failures never reach Lance at all —
``error_records`` projects them to plain dicts for a replaced-each-run
``errors.json`` report, the same shape ``video_split`` writes.
"""

from typing import Any

import pyarrow as pa

CLIP_SCHEMA = pa.schema(
    [
        pa.field("record_schema_version", pa.int32(), nullable=False),
        pa.field("media_contract_version", pa.int32(), nullable=False),
        pa.field("clip_id", pa.string(), nullable=False),
        pa.field("span_group_id", pa.string(), nullable=False),
        pa.field("view_name", pa.string(), nullable=False),
        pa.field("source_id", pa.string(), nullable=False),
        pa.field("source_dataset", pa.string(), nullable=False),
        pa.field("episode_id", pa.string(), nullable=False),
        pa.field("episode_index", pa.int32(), nullable=False),
        pa.field("subtask_index", pa.int32(), nullable=False),
        pa.field("subtask_name", pa.string(), nullable=False),
        pa.field("task_index", pa.int32(), nullable=False),
        pa.field("task_name", pa.string(), nullable=False),
        pa.field("task_description", pa.large_string(), nullable=True),
        pa.field("frame_start", pa.int32(), nullable=False),
        pa.field("frame_end", pa.int32(), nullable=False),
        pa.field("start_ns", pa.int64(), nullable=False),
        pa.field("end_ns", pa.int64(), nullable=False),
        pa.field("native_fps", pa.float64(), nullable=False),
        pa.field("episode_from_timestamp", pa.float64(), nullable=False),
        pa.field("clip_uri", pa.large_string(), nullable=False),
        pa.field("action_data_uri", pa.large_string(), nullable=False),
        # Genuinely optional even for a successful clip: absent when the source
        # has no camera-trajectory data to describe, not a failure signal.
        pa.field("camera_motion_annotation", pa.large_string()),
    ]
)

_BASE_FIELD_NAMES = tuple(
    name
    for name in CLIP_SCHEMA.names
    if name not in ("record_schema_version", "media_contract_version", "clip_uri", "action_data_uri")
)

# error_records omits camera_motion_annotation too: every failure path sets it to
# None (processing.py never computes motion for a clip that didn't finish
# cutting), so the field would carry no information in errors.json.
_ERROR_FIELD_NAMES = tuple(name for name in _BASE_FIELD_NAMES if name != "camera_motion_annotation")


def clip_table(outcomes: list[dict[str, Any]], *, record_schema_version: int, media_contract_version: int) -> pa.Table:
    """Project successful outcome dicts onto the canonical clip schema."""
    rows = [
        {
            "record_schema_version": record_schema_version,
            "media_contract_version": media_contract_version,
            **{field: outcome[field] for field in _BASE_FIELD_NAMES},
            "clip_uri": outcome["clip_uri"],
            "action_data_uri": outcome["action_data_uri"],
        }
        for outcome in outcomes
        if outcome["status"] == "success"
    ]
    return pa.Table.from_pylist(rows, schema=CLIP_SCHEMA)


def error_records(outcomes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Project failed outcome dicts to plain JSON-serializable rows.

    Not a Lance schema — these never reach the canonical table. Written as a
    replaced-each-run ``errors.json`` report, matching ``video_split``.
    """
    return [
        {
            **{field: outcome[field] for field in _ERROR_FIELD_NAMES},
            "error_stage": outcome["error_stage"],
            "error_message": outcome["error_message"],
        }
        for outcome in outcomes
        if outcome["status"] != "success"
    ]
