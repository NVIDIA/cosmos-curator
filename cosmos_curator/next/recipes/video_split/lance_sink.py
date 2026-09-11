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

"""Schema bootstrap, distributed fragment staging, and incremental appends.

Thin ``video-split``-schema binding over the generic canonical-table primitives
in ``cosmos_curator.next.utils.lance_fragment_recovery``.
"""

import lance
import pyarrow as pa

from cosmos_curator.next.recipes.video_split.records import CLIP_SCHEMA
from cosmos_curator.next.utils import lance_fragment_recovery
from cosmos_curator.next.utils.lance_utils import LANCE_DATA_STORAGE_VERSION

_KIND = "video-split"
_APPEND_OPERATION = "append-clips"


def open_or_create_clip_table(*, uri: str, storage_profile: str) -> lance.LanceDataset:
    """Open the canonical table or atomically bootstrap its zero-row schema."""
    return lance_fragment_recovery.open_or_create_table(
        uri=uri,
        storage_profile=storage_profile,
        schema=CLIP_SCHEMA,
        data_storage_version=LANCE_DATA_STORAGE_VERSION,
        kind=_KIND,
    )


def validate_clip_table(dataset: lance.LanceDataset, *, uri: str) -> None:
    """Require the splitting-owned schema while allowing nullable curation fields."""
    lance_fragment_recovery.validate_table(
        dataset,
        uri=uri,
        schema=CLIP_SCHEMA,
        data_storage_version=LANCE_DATA_STORAGE_VERSION,
    )


def write_clip_fragment(clips: pa.Table, *, uri: str, storage_profile: str) -> str | None:
    """Stage one canonical clip batch and serialize its recovery candidate."""
    return lance_fragment_recovery.write_row_fragment(
        clips,
        uri=uri,
        storage_profile=storage_profile,
        schema=CLIP_SCHEMA,
        id_column="clip_id",
        data_storage_version=LANCE_DATA_STORAGE_VERSION,
    )


def append_clip_fragment(
    candidate_json: str,
    *,
    uri: str,
    storage_profile: str,
    attempts: int,
) -> int:
    """Idempotently append one staged fragment and return a version containing it."""
    return lance_fragment_recovery.append_row_fragment(
        candidate_json,
        uri=uri,
        storage_profile=storage_profile,
        schema=CLIP_SCHEMA,
        id_column="clip_id",
        kind=_KIND,
        operation=_APPEND_OPERATION,
        data_storage_version=LANCE_DATA_STORAGE_VERSION,
        attempts=attempts,
    )
