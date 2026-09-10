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
"""Timestamp sampling policies."""

import attrs


@attrs.define(frozen=True, hash=False)
class NearestTimestampPolicy:
    """Nearest-timestamp sampling policy.

    Attributes:
        max_delta_ns: the maximum allowed time delta between a reference
            timestamp and the chosen canonical sample. ``None`` disables only
            this maximum-delta check.

    """

    __hash__ = None  # type: ignore[assignment]
    max_delta_ns: int | None = attrs.field(default=None, validator=attrs.validators.optional(attrs.validators.ge(0)))


@attrs.define(frozen=True, hash=False)
class GpsInterpolationPolicy:
    """GPS position interpolation policy.

    Attributes:
        max_gap_ns: maximum permitted duration between the lower and upper
            source fixes. ``None`` disables only this maximum-gap check.

    """

    __hash__ = None  # type: ignore[assignment]
    max_gap_ns: int | None = attrs.field(default=None, validator=attrs.validators.optional(attrs.validators.ge(0)))


GpsSamplingPolicy = NearestTimestampPolicy | GpsInterpolationPolicy


@attrs.define(frozen=True, hash=False)
class NoSamplingPolicy:
    """Explicit no-operation policy for sensors whose current sampling is policy-independent."""

    __hash__ = None  # type: ignore[assignment]


def require_nearest_timestamp_policy(policy: object, *, sensor_name: str) -> NearestTimestampPolicy:
    """Return *policy* when it is a nearest-timestamp policy, otherwise raise a configuration error."""
    if isinstance(policy, NearestTimestampPolicy):
        return policy
    msg = f"{sensor_name} requires NearestTimestampPolicy, got {type(policy).__name__}"
    raise TypeError(msg)


def require_gps_sampling_policy(policy: object, *, sensor_name: str) -> GpsSamplingPolicy:
    """Return a supported GPS policy or raise a configuration error."""
    if isinstance(policy, (NearestTimestampPolicy, GpsInterpolationPolicy)):
        return policy
    msg = f"{sensor_name} requires NearestTimestampPolicy or GpsInterpolationPolicy, got {type(policy).__name__}"
    raise TypeError(msg)


def require_no_sampling_policy(policy: object, *, sensor_name: str) -> NoSamplingPolicy:
    """Return *policy* when it is the explicit no-op policy, otherwise raise a configuration error."""
    if isinstance(policy, NoSamplingPolicy):
        return policy
    msg = f"{sensor_name} requires NoSamplingPolicy, got {type(policy).__name__}"
    raise TypeError(msg)
