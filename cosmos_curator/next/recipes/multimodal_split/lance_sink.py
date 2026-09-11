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

"""Canonical episode-table bootstrap, staging, and append bindings."""

import lance
import pyarrow as pa

from cosmos_curator.core.utils.storage.storage_utils import get_lance_storage_options
from cosmos_curator.next.recipes.multimodal_split.records import EPISODE_SCHEMA
from cosmos_curator.next.utils import lance_fragment_recovery
from cosmos_curator.next.utils.lance_utils import LANCE_DATA_STORAGE_VERSION, open_dataset_or_raise

_KIND = "multimodal-split"
_APPEND_OPERATION = "append-episodes"


def open_or_create_episode_table(*, uri: str, storage_profile: str) -> lance.LanceDataset:
    """Open the canonical episode table or atomically bootstrap its schema."""
    return lance_fragment_recovery.open_or_create_table(
        uri=uri,
        storage_profile=storage_profile,
        schema=EPISODE_SCHEMA,
        data_storage_version=LANCE_DATA_STORAGE_VERSION,
        kind=_KIND,
    )


def open_episode_table(*, uri: str, storage_profile: str) -> lance.LanceDataset:
    """Open and validate an existing canonical episode table."""
    storage_options = get_lance_storage_options(uri, profile_name=storage_profile)
    dataset = open_dataset_or_raise(uri, storage_options=storage_options)
    validate_episode_table(dataset, uri=uri)
    return dataset


def validate_episode_table(dataset: lance.LanceDataset, *, uri: str) -> None:
    """Require the episode base schema while allowing nullable enrichments."""
    lance_fragment_recovery.validate_table(
        dataset,
        uri=uri,
        schema=EPISODE_SCHEMA,
        data_storage_version=LANCE_DATA_STORAGE_VERSION,
    )


def write_episode_fragment(episodes: pa.Table, *, uri: str, storage_profile: str) -> str | None:
    """Stage one canonical episode batch and serialize its recovery candidate."""
    return lance_fragment_recovery.write_row_fragment(
        episodes,
        uri=uri,
        storage_profile=storage_profile,
        schema=EPISODE_SCHEMA,
        id_column="episode_id",
        data_storage_version=LANCE_DATA_STORAGE_VERSION,
    )


def append_episode_fragment(candidate_json: str, *, uri: str, storage_profile: str, attempts: int) -> int:
    """Idempotently append one staged episode fragment."""
    return lance_fragment_recovery.append_row_fragment(
        candidate_json,
        uri=uri,
        storage_profile=storage_profile,
        schema=EPISODE_SCHEMA,
        id_column="episode_id",
        kind=_KIND,
        operation=_APPEND_OPERATION,
        data_storage_version=LANCE_DATA_STORAGE_VERSION,
        attempts=attempts,
    )
