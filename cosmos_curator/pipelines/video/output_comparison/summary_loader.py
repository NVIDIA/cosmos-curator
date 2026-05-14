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
"""Storage-backed loading for split pipeline summaries."""

import json
import pathlib
from typing import cast

import smart_open  # type: ignore[import-untyped]

from cosmos_curator.core.utils.storage import storage_utils
from cosmos_curator.core.utils.storage.storage_client import StoragePrefix
from cosmos_curator.pipelines.video.output_comparison.json_types import JsonDictObject
from cosmos_curator.pipelines.video.output_comparison.summary_schema import OutputSummary

OutputRoot = str | pathlib.Path | StoragePrefix


def load_summary(
    output_root: OutputRoot,
    *,
    profile_name: str,
) -> OutputSummary:
    """Load ``summary.json`` from an output root.

    Args:
        output_root: Split pipeline output root.
        profile_name: Storage profile used for remote paths.

    Returns:
        Loaded typed summary.

    """
    summary_path = storage_utils.get_full_path(output_root, "summary.json")
    client = storage_utils.get_storage_client(str(summary_path), profile_name=profile_name)
    client_params = storage_utils.get_smart_open_client_params(client) if client is not None else {}
    with smart_open.open(str(summary_path), "r", encoding="utf-8", **client_params) as fp:
        data = json.load(fp)
    if not isinstance(data, dict) or not all(isinstance(key, str) for key in data):
        error_msg = "summary.json must contain a JSON object with string keys"
        raise ValueError(error_msg)
    return OutputSummary.from_json_dict(cast("JsonDictObject", data))
