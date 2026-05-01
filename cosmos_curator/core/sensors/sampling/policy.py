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
"""Timestamp sampling policy."""

import attrs


@attrs.define(frozen=True, hash=False)
class SamplingPolicy:
    """Timestamp sampling policy.

    Attributes:
        tolerance_ns: the maximum allowed time delta between a reference
            timestamp and the chosen canonical sample (per-sensor or global;
            exact semantics are implementation-defined).

    """

    __hash__ = None  # type: ignore[assignment]
    tolerance_ns: int = attrs.field(default=0, validator=attrs.validators.ge(0))
