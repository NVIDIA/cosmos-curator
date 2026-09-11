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

"""Deterministic integer-timeline span contracts for Curator Next media.

Video, robot action, and AV remain independent streams during incubation.
Each recipe using these helpers owns its output and recovery compatibility.
Changes to rounding, interval boundaries, or tail retention require reviewing
affected callers and their respective contract versions. Video-cut's
``behavior_version`` applies to its own cutting contract.
"""

from dataclasses import dataclass
from decimal import ROUND_HALF_EVEN, Decimal

_NANOSECONDS_PER_SECOND = Decimal(1_000_000_000)


@dataclass(frozen=True)
class Span:
    """Inclusive start and exclusive end on a source video's timeline."""

    start_ns: int
    end_ns: int

    def __post_init__(self) -> None:  # noqa: D105
        if self.start_ns < 0:
            msg = f"Span.start_ns must be non-negative, got {self.start_ns}"
            raise ValueError(msg)
        if self.end_ns <= self.start_ns:
            msg = f"Span.end_ns must be greater than start_ns, got end_ns={self.end_ns} <= start_ns={self.start_ns}"
            raise ValueError(msg)

    @property
    def duration_ns(self) -> int:
        """Return the span duration in integer nanoseconds."""
        return self.end_ns - self.start_ns


def seconds_to_nanoseconds(seconds: float | str | Decimal) -> int:
    """Convert config/probe seconds to integer nanoseconds with one defined rounding rule."""
    value = (seconds if isinstance(seconds, Decimal) else Decimal(str(seconds))) * _NANOSECONDS_PER_SECOND
    return int(value.to_integral_value(rounding=ROUND_HALF_EVEN))


def nanoseconds_to_ffmpeg_timestamp(nanoseconds: int) -> str:
    """Render integer nanoseconds as an exact decimal timestamp accepted by FFmpeg."""
    seconds = Decimal(nanoseconds) / _NANOSECONDS_PER_SECOND
    return format(seconds, "f")


def fixed_stride_spans(
    source_duration_ns: int,
    *,
    duration_ns: int,
    stride_ns: int,
    min_duration_ns: int,
) -> tuple[Span, ...]:
    """Generate deterministic fixed-stride spans, retaining a sufficiently long tail."""
    if source_duration_ns < 0:
        msg = f"source_duration_ns must be non-negative, got {source_duration_ns}"
        raise ValueError(msg)
    if duration_ns <= 0 or stride_ns <= 0 or min_duration_ns <= 0:
        msg = "duration_ns, stride_ns, and min_duration_ns must all be positive"
        raise ValueError(msg)
    if min_duration_ns > duration_ns:
        msg = "min_duration_ns cannot be greater than duration_ns"
        raise ValueError(msg)

    spans: list[Span] = []
    start_ns = 0
    while start_ns < source_duration_ns:
        end_ns = min(start_ns + duration_ns, source_duration_ns)
        if end_ns - start_ns >= min_duration_ns:
            spans.append(Span(start_ns=start_ns, end_ns=end_ns))
        start_ns += stride_ns
    return tuple(spans)
