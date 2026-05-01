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
"""Timestamp sampling specification."""

import attrs

from cosmos_curator.core.sensors.sampling.grid import SamplingGrid
from cosmos_curator.core.sensors.sampling.policy import SamplingPolicy


@attrs.define(frozen=True, hash=False)
class SamplingSpec:
    """Timestamp sampling specification.

    ``policy=None`` means no sampling policy is applied. When a
    :class:`SamplingPolicy` is provided, its rules constrain sampling.
    """

    __hash__ = None  # type: ignore[assignment]
    grid: SamplingGrid
    policy: SamplingPolicy | None = None
