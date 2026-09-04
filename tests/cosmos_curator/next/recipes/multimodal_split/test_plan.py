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

"""Tests for deciding which spans of a session every sensor can serve."""

import numpy as np
import pytest

from cosmos_curator.next.recipes.multimodal_split.plan import first_uncovered

_MS = 1_000_000
_EPOCH = 1_700_000_000 * 1_000_000_000


@pytest.mark.parametrize(
    ("offset_ns", "covered"),
    [
        (49 * _MS, True),
        # Exactly at the tolerance. The comparison is ``worst > max``, so equal
        # is covered -- and a flip to ``>=`` would pass every other test here,
        # which all sit well clear of the boundary.
        (50 * _MS, True),
        (50 * _MS + 1, False),
        (51 * _MS, False),
    ],
)
def test_the_tolerance_is_inclusive_at_its_boundary(offset_ns: int, *, covered: bool) -> None:
    """A frame exactly ``max_delta_s`` from a reference still serves that moment.

    Which way the boundary falls decides whether a rig at the edge of its
    tolerance yields clips or is refused, and nothing else in the suite pins it:
    every other case is a comfortable margin either side.
    """
    max_delta_ns = 50 * _MS
    references = np.array([_EPOCH, _EPOCH + 100 * _MS], dtype=np.int64)
    timeline = references + offset_ns

    missing = first_uncovered(references, {"front": timeline}, max_delta_ns)

    assert (missing is None) is covered
    if missing is not None:
        assert missing == ("front", offset_ns)
