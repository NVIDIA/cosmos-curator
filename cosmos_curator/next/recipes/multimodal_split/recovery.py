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

"""Pinned-snapshot reconciliation for canonical multimodal episodes."""

from collections import defaultdict
from collections.abc import Iterable
from typing import Any

import lance

from cosmos_curator.next.utils.lance_fragment_recovery import sql_string_literal

_RECONCILIATION_COLUMNS = (
    "record_schema_version",
    "media_contract_version",
    "episode_id",
    "source_id",
    "sampling_fps",
    "timestamp_source",
    "split_contract_digest",
    "camera_media",
)

type ReconciliationMap = dict[str, dict[str, dict[str, Any]]]


def read_reconciliation(dataset: lance.LanceDataset, source_ids: Iterable[str]) -> ReconciliationMap:
    """Read existing base facts for discovered sources from one pinned version."""
    unique_source_ids = sorted(set(source_ids))
    if not unique_source_ids:
        return {}
    literals = ",".join(sql_string_literal(source_id) for source_id in unique_source_ids)
    table = dataset.to_table(columns=list(_RECONCILIATION_COLUMNS), filter=f"source_id IN ({literals})")
    grouped: defaultdict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    episode_ids: set[str] = set()
    for row in table.to_pylist():
        episode_id = str(row["episode_id"])
        if episode_id in episode_ids:
            msg = f"Canonical episode table contains duplicate episode_id {episode_id!r}"
            raise ValueError(msg)
        episode_ids.add(episode_id)
        grouped[str(row["source_id"])][episode_id] = row
    return {source_id: dict(rows) for source_id, rows in grouped.items()}


def reconcile_planned_episode(
    expected: dict[str, Any],
    *,
    existing_by_episode_id: dict[str, dict[str, Any]],
    camera_sources: dict[str, str],
) -> bool:
    """Return whether an exact episode is canonical, failing incompatible matches."""
    existing = existing_by_episode_id.get(str(expected["episode_id"]))
    if existing is None:
        return False
    expected_contract = (
        int(expected["record_schema_version"]),
        int(expected["media_contract_version"]),
        str(expected["split_contract_digest"]),
        int(expected["sampling_fps"]),
        str(expected["timestamp_source"]),
        tuple(sorted(camera_sources.items())),
    )
    existing_sources = tuple(
        sorted((str(camera["camera_key"]), str(camera["source_media_location"])) for camera in existing["camera_media"])
    )
    existing_contract = (
        int(existing["record_schema_version"]),
        int(existing["media_contract_version"]),
        str(existing["split_contract_digest"]),
        int(existing["sampling_fps"]),
        str(existing["timestamp_source"]),
        existing_sources,
    )
    if existing_contract != expected_contract:
        msg = (
            f"Canonical episode {expected['episode_id']} conflicts with the current multimodal-split contract or "
            "resolved camera sources"
        )
        raise ValueError(msg)
    return True
