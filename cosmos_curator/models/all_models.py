# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Model management for curator."""

import json

from cosmos_curator.core.utils.environment import CONTAINER_PATHS_CODE_DIR


def get_all_models() -> dict[str, dict[str, str | list[str] | None]]:
    """Get all models available in the curator.

    Returns:
        A dictionary containing model names as keys and their configurations as values.

    """
    model_json = CONTAINER_PATHS_CODE_DIR / "cosmos_curator" / "configs" / "all_models.json"
    all_models: dict[str, dict[str, str | list[str] | None]] = json.loads(model_json.read_text("utf-8"))
    return all_models


def get_all_models_by_id() -> dict[str, dict[str, str | list[str] | None]]:
    """Get all models available in the curator indexed by model ID.

    Returns:
        A dictionary containing model IDs as keys and their configurations as values.

    """
    all_models = get_all_models()
    all_models_by_id: dict[str, dict[str, str | list[str] | None]] = {}
    for model_detail in all_models.values():
        assert "model_id" in model_detail
        assert isinstance(model_detail["model_id"], str)
        model_id: str = model_detail["model_id"]
        all_models_by_id[model_id] = model_detail
    return all_models_by_id
