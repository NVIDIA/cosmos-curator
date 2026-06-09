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
"""Tests for caption_embedding helpers that need no model weights."""

from typing import cast

import pytest

from cosmos_curator.pipelines.video.split_comparison.caption_embedding import cosine_similarity_batch


def test_cosine_similarity_batch_rejects_mismatched_lengths() -> None:
    """Unequal input lengths raise before encode (would otherwise misalign pairs)."""
    # The length guard runs before any model.encode call, so no real model is needed.
    with pytest.raises(ValueError, match="paired inputs"):
        cosine_similarity_batch(cast("object", None), ["a cat", "a dog"], ["a cat"], batch_size=8)  # type: ignore[arg-type]
