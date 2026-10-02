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

"""Find a non-fragmented file's ``moov`` box and return its bytes, box header included."""

import struct

from loguru import logger  # type: ignore[import-untyped]

from cosmos_curator.core.sensors.mp4.box_primitives import (
    EXTENDED_HEADER_BYTES,
    EXTENDED_SIZE_MARKER,
    STANDARD_HEADER_BYTES,
    BoxNotFoundError,
    read_box_header,
)
from cosmos_curator.core.sensors.mp4.byte_range import FetchRange

#: size (4) + type (4) + optional 64-bit largesize (8) -- enough to read any box header.
_HEADER_PROBE_BYTES = 16

#: Fetched on the first attempt, head or tail alike.
_DEFAULT_INITIAL_GUESS_BYTES = 8 * 1024 * 1024

#: Multiplier applied to the guess size on the one bounded retry, head or tail alike.
_DEFAULT_GUESS_RETRY_MULTIPLIER = 4


def get_moov_full_walk(file_size: int, fetch_range: FetchRange) -> bytes:
    """Find ``moov`` by walking every top-level box header from offset 0, and return its bytes.

    Raises:
        BoxNotFoundError: no top-level ``moov`` box in the file.

    """
    position = 0
    while position < file_size:
        header = fetch_range(position, _HEADER_PROBE_BYTES)
        if len(header) < STANDARD_HEADER_BYTES:
            break
        size32 = struct.unpack_from(">I", header, 0)[0]
        header_bytes_needed = EXTENDED_HEADER_BYTES if size32 == EXTENDED_SIZE_MARKER else STANDARD_HEADER_BYTES
        if len(header) < header_bytes_needed:
            break
        found_type, box_size, _ = read_box_header(header, 0, file_size - position)
        if found_type == "moov":
            if box_size <= len(header):
                return header[:box_size]
            return header + fetch_range(position + len(header), box_size - len(header))
        if box_size <= 0:
            break
        position += box_size
    msg = f"no top-level moov box found (walked to file offset {position} of {file_size})"
    raise BoxNotFoundError(msg)


def _trailing_bytes_are_well_formed_boxes(tail: bytes, offset: int, tail_start: int, file_size: int) -> bool:
    """Confirm ``tail[offset:]`` parses as a clean run of box headers consuming exactly to *file_size*.

    Files don't always end with ``moov`` as the last byte -- trailing ``skip`` boxes have been
    observed -- so this tolerates that while still ruling out a coincidental ``"moov"`` byte match
    in preceding binary content.
    """
    remaining = file_size - (tail_start + offset)
    if remaining == 0:
        return True
    if remaining < 0:
        return False
    while remaining > 0:
        if offset + STANDARD_HEADER_BYTES > len(tail):
            return False  # not enough tail buffered to confirm the rest -- don't guess
        size32 = struct.unpack_from(">I", tail, offset)[0]
        header_bytes_needed = EXTENDED_HEADER_BYTES if size32 == EXTENDED_SIZE_MARKER else STANDARD_HEADER_BYTES
        if offset + header_bytes_needed > len(tail):
            return False  # extended-size box's header isn't fully buffered yet -- don't guess
        box_type, box_size, header_len = read_box_header(tail, offset, len(tail))
        if box_size < header_len or box_size > remaining or not box_type.isascii() or not box_type.isprintable():
            return False
        offset += box_size
        remaining -= box_size
    return True


def _find_trailing_moov_in_tail(tail: bytes, tail_start: int, file_size: int) -> tuple[int, int] | None:
    """Find ``moov`` in a possibly non-box-aligned tail slice, assuming it is near the file's end.

    Validates each ``moov``-matching position until one passes or none remain.
    """
    search_end = len(tail)  # scan backward: the genuine trailing moov is closest to file_size
    while (hit := tail.rfind(b"moov", 0, search_end)) != -1:
        search_end = hit
        header_start = hit - 4
        if header_start < 0 or header_start + STANDARD_HEADER_BYTES > len(tail):
            continue
        size = struct.unpack_from(">I", tail, header_start)[0]
        if size < STANDARD_HEADER_BYTES:
            continue
        absolute_start = tail_start + header_start
        if _trailing_bytes_are_well_formed_boxes(tail, header_start + size, tail_start, file_size):
            return absolute_start, absolute_start + size
    return None


def get_moov_tail_guess(
    file_size: int,
    fetch_range: FetchRange,
    *,
    initial_tail_bytes: int = _DEFAULT_INITIAL_GUESS_BYTES,
    retry_multiplier: int = _DEFAULT_GUESS_RETRY_MULTIPLIER,
) -> bytes:
    """Find ``moov`` optimized for a file that trails it, and return its bytes.

    Args:
        file_size: total size of the file.
        fetch_range: see :data:`~.byte_range.FetchRange`.
        initial_tail_bytes: bytes fetched on the first, common-case attempt.
        retry_multiplier: multiplier applied to the tail size on the one retry.

    Raises:
        BoxNotFoundError: no top-level ``moov`` box in the file.

    """
    tail_bytes = initial_tail_bytes
    for _ in range(2):
        tail_bytes = min(tail_bytes, file_size)
        tail_start = file_size - tail_bytes
        tail = fetch_range(tail_start, tail_bytes)
        found = _find_trailing_moov_in_tail(tail, tail_start, file_size)
        if found is not None:
            box_start, box_end = found
            return tail[box_start - tail_start : box_end - tail_start]
        if tail_bytes >= file_size:
            break  # already fetched the whole file; a bigger retry would only repeat it
        tail_bytes *= retry_multiplier
    return get_moov_full_walk(file_size, fetch_range)


def _find_leading_moov_in_head(head: bytes) -> tuple[int, int] | None:
    """Find ``moov`` by walking top-level box headers from offset 0 within *head*.

    Returns ``None`` (not raising) when the walk can't complete within *head*'s own bytes --
    ``moov`` may be further in, or the file may trail it instead.
    """
    position = 0
    while position + STANDARD_HEADER_BYTES <= len(head):
        size32 = struct.unpack_from(">I", head, position)[0]
        header_bytes_needed = EXTENDED_HEADER_BYTES if size32 == EXTENDED_SIZE_MARKER else STANDARD_HEADER_BYTES
        if position + header_bytes_needed > len(head):
            return None  # extended-size box's header isn't fully buffered yet -- inconclusive
        box_type, box_size, _header_len = read_box_header(head, position, len(head))
        if box_size <= 0:
            return None  # malformed -- don't guess further
        if box_type == "moov":
            if position + box_size > len(head):
                return None  # moov starts within head but its content isn't fully buffered yet
            return position, position + box_size
        position += box_size
    return None


def get_moov_head_guess(
    file_size: int,
    fetch_range: FetchRange,
    *,
    initial_head_bytes: int = _DEFAULT_INITIAL_GUESS_BYTES,
    retry_multiplier: int = _DEFAULT_GUESS_RETRY_MULTIPLIER,
    source_name: str = "<unknown>",
) -> bytes | None:
    """Find ``moov`` optimized for a file that leads with it, or ``None`` if the guess is inconclusive.

    Never falls back to the full walk itself -- a miss here should try the tail guess next, not
    jump straight to the expensive fallback; the head-then-tail cascade owns that decision
    instead.

    Args:
        file_size: total size of the file.
        fetch_range: see :data:`~.byte_range.FetchRange`.
        initial_head_bytes: bytes fetched on the first, common-case attempt.
        retry_multiplier: multiplier applied to the head size on the one retry.
        source_name: source file for debug logging.

    Returns:
        ``None`` if neither guess was conclusive.

    """
    head_bytes = initial_head_bytes
    for attempt in range(2):
        head_bytes = min(head_bytes, file_size)
        head = fetch_range(0, head_bytes)
        found = _find_leading_moov_in_head(head)
        if found is not None:
            box_start, box_end = found
            if attempt > 0:
                logger.info(
                    "{}: moov found on retry at a {} B head (initial guess was too small)", source_name, head_bytes
                )
            logger.debug("{}: moov is {} B", source_name, box_end - box_start)
            return head[box_start:box_end]
        if head_bytes >= file_size:
            break  # already fetched the whole file; a bigger retry would only repeat it
        head_bytes *= retry_multiplier
    return None


def get_moov_auto(file_size: int, fetch_range: FetchRange, *, source_name: str = "<unknown>") -> bytes:
    """Find ``moov`` without assuming where it sits -- head guess, then tail guess, then a full walk.

    Runs sequentially, not in parallel: concurrency across many videos may happen one layer
    up, so a second attempt costs something only on a miss, not on every video.

    Args:
        file_size: total size of the file.
        fetch_range: see :data:`~.byte_range.FetchRange`.
        source_name: source file for debug logging.

    Raises:
        BoxNotFoundError: no top-level ``moov`` box in the file -- raised by
            the tail guess's own full-walk fallback, the only one of the
            three that can prove absence.

    """
    found = get_moov_head_guess(file_size, fetch_range, source_name=source_name)
    if found is not None:
        return found
    return get_moov_tail_guess(file_size, fetch_range)
