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
"""test models/prompts.py."""

from contextlib import AbstractContextManager, nullcontext
from typing import Any

import pytest

from cosmos_curator.models.prompts import (
    _DEFAULT_STAGE2_PROMPT,
    _ENHANCE_PROMPTS,
    _PROMPTS,
    get_enhance_prompt,
    get_prompt,
    get_stage2_prompt,
)


class TestGetPrompt:
    """Test cases for get_prompt function."""

    @pytest.mark.parametrize(
        ("variant", "expected_type", "raises"),
        [(v, str, nullcontext()) for v in _PROMPTS] + [("unknown", str, pytest.raises(ValueError, match=r".*"))],
    )
    @pytest.mark.parametrize("verbose", [True, False])
    def test_get_prompt(
        self, variant: str, expected_type: type, raises: AbstractContextManager[Any], *, verbose: bool
    ) -> None:
        """Test getting default prompt variant."""
        with raises:
            result = get_prompt(variant, None, verbose=verbose)
            assert isinstance(result, expected_type)

    @pytest.mark.parametrize("verbose", [True, False])
    def test_get_prompt_custom_text(self, *, verbose: bool) -> None:
        """Test getting prompt with custom prompt_text."""
        custom_text = "This is a custom prompt for testing."
        result = get_prompt("any_variant", custom_text, verbose=verbose)
        assert result == custom_text


class TestGetEnhancePrompt:
    """Test cases for get_enhance_prompt function."""

    @pytest.mark.parametrize(
        ("variant", "expected_type", "raises"),
        [(v, str, nullcontext()) for v in _ENHANCE_PROMPTS]
        + [("unknown", str, pytest.raises(ValueError, match=r".*"))],
    )
    @pytest.mark.parametrize("verbose", [True, False])
    def test_get_enhance_prompt(
        self, variant: str, expected_type: type, raises: AbstractContextManager[Any], *, verbose: bool
    ) -> None:
        """Test getting default prompt variant."""
        with raises:
            result = get_enhance_prompt(variant, None, verbose=verbose)
            assert isinstance(result, expected_type)

    @pytest.mark.parametrize("verbose", [True, False])
    def test_get_enhance_prompt_custom_text(self, *, verbose: bool) -> None:
        """Test getting prompt with custom prompt_text."""
        custom_text = "This is a custom prompt for testing."
        result = get_enhance_prompt("any_variant", custom_text, verbose=verbose)
        assert result == custom_text


class TestGetStage2Prompt:
    """Test cases for get_stage2_prompt function."""

    @pytest.mark.parametrize("prompt", [None, "This is a custom stage 2 prompt for testing."])
    def test_get_stage2_prompt(self, prompt: str | None) -> None:
        """Test getting default stage 2 prompt."""
        result = get_stage2_prompt(prompt)

        if prompt is None:
            assert _DEFAULT_STAGE2_PROMPT.strip() in result
        else:
            assert result is prompt
