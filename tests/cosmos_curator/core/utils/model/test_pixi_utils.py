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

"""Tests for Pixi environment detection."""

import pytest

from cosmos_curator.core.utils.environment import PIXI_ENVIRONMENT_NAME_VAR_NAME
from cosmos_curator.core.utils.model import pixi_utils


def test_pixi_env_name_is_current_env_name(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pixi env names are reported as the current environment."""
    monkeypatch.setenv(PIXI_ENVIRONMENT_NAME_VAR_NAME, "default")

    assert pixi_utils.get_env_name() == "default"
    assert pixi_utils.is_running_in_env("default")
    assert not pixi_utils.is_running_in_env("unified")


def test_named_pixi_env_does_not_match_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Named Pixi envs do not also match default."""
    monkeypatch.setenv(PIXI_ENVIRONMENT_NAME_VAR_NAME, "unified")

    assert pixi_utils.get_env_name() == "unified"
    assert pixi_utils.is_running_in_env("unified")
    assert not pixi_utils.is_running_in_env("default")


def test_missing_pixi_env_name_returns_empty_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing Pixi env names are treated as no active logical env."""
    monkeypatch.delenv(PIXI_ENVIRONMENT_NAME_VAR_NAME, raising=False)

    assert pixi_utils.get_env_name() == ""
    assert not pixi_utils.is_running_in_env("default")
