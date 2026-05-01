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

"""Utilities shared across ``cosmos_curator.core.sensors`` (data models, devices, etc.)."""

from typing import Any

import numpy.typing as npt


def as_readonly_view(array: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """Return a read-only view without mutating the caller-owned array."""
    readonly_view = array.view()
    readonly_view.flags.writeable = False
    return readonly_view


def as_readonly_view_tuple(
    arrays: tuple[npt.NDArray[Any], ...] | list[npt.NDArray[Any]],
) -> tuple[npt.NDArray[Any], ...]:
    """Return a tuple of read-only views without mutating the caller-owned arrays."""
    return tuple(as_readonly_view(array) for array in arrays)
