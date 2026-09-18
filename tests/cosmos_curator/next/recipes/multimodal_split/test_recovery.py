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

"""Tests for pinned canonical-episode reconciliation reads."""

from unittest.mock import MagicMock

import pyarrow as pa
import pytest

from cosmos_curator.next.recipes.multimodal_split.recovery import read_reconciliation


def test_duplicate_canonical_episode_ids_fail_closed() -> None:
    """A duplicate identity must not silently overwrite the first canonical row."""
    row = {
        "record_schema_version": 2,
        "media_contract_version": 3,
        "episode_id": "episode",
        "source_id": "source",
        "sampling_fps": 30,
        "timestamp_source": "sensor",
        "split_contract_digest": "contract",
        "camera_media": [{"camera_key": "front", "source_media_location": "/input/front.mp4"}],
    }
    dataset = MagicMock()
    dataset.to_table.return_value = pa.Table.from_pylist([row, row])

    with pytest.raises(ValueError, match="duplicate episode_id 'episode'"):
        read_reconciliation(dataset, ["source"])
