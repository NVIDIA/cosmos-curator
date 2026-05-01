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

"""Tests for UUID utility functions."""

from cosmos_curator.core.utils.misc.uuid_utils import is_uuid


def test_is_uuid() -> None:
    """Test UUID string validation."""
    assert is_uuid("550e8400-e29b-41d4-a716-446655440000") is True
    assert is_uuid("a1b2c3d4-e5f6-7890-abcd-ef1234567890") is True
    assert is_uuid("not-a-uuid") is False
    assert is_uuid("subdir") is False
