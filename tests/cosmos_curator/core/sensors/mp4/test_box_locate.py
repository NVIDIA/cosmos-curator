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
"""Tests for finding a non-fragmented file's moov box and returning its bytes.

Focus is on the bytes-back contract: get_moov_tail_guess/get_moov_full_walk
return moov's own content directly, never leaving a caller to make a second
fetch for bytes already in hand.
"""

import struct

import pytest

from cosmos_curator.core.sensors.mp4 import box_locate as bl
from cosmos_curator.core.sensors.mp4.box_primitives import BoxNotFoundError
from tests.cosmos_curator.core.sensors.mp4._box_fixtures import pack_box as _box


class _CountingFetcher:
    """A stand-in remote store: serves ranges from an in-memory buffer, counting calls."""

    def __init__(self, buf: bytes) -> None:
        self._buf = buf
        self.calls: list[tuple[int, int]] = []

    def __call__(self, offset: int, length: int) -> bytes:
        self.calls.append((offset, length))
        return self._buf[offset : offset + length]


# ---------------------------------------------------------------------------
# get_moov_full_walk -- always-correct header walk, returns bytes
# ---------------------------------------------------------------------------


def test_get_moov_full_walk_returns_moov_bytes() -> None:
    """The walk returns moov's own bytes directly, not offsets a caller has to fetch itself."""
    buf = _box("ftyp", b"isom") + _box("mdat", b"data" * 1000) + _box("moov", b"payload")
    fetcher = _CountingFetcher(buf)

    result = bl.get_moov_full_walk(len(buf), fetcher)

    assert result == _box("moov", b"payload")


def test_get_moov_full_walk_never_fetches_mdat_content() -> None:
    """Walking past a giant mdat costs one header-sized fetch, not its content."""
    buf = _box("ftyp", b"isom") + _box("mdat", b"x" * 100_000) + _box("moov", b"payload")
    fetcher = _CountingFetcher(buf)

    bl.get_moov_full_walk(len(buf), fetcher)

    header_calls = [call for call in fetcher.calls if call[1] == 16]
    assert len(header_calls) >= 2  # at least ftyp's and mdat's headers, before the final content fetch
    assert all(length <= 16 for _offset, length in fetcher.calls[:-1])


def test_get_moov_full_walk_raises_when_absent() -> None:
    """No matching box at all still raises."""
    buf = _box("ftyp", b"isom") + _box("mdat", b"data")
    fetcher = _CountingFetcher(buf)

    with pytest.raises(BoxNotFoundError):
        bl.get_moov_full_walk(len(buf), fetcher)


def test_get_moov_full_walk_raises_cleanly_when_an_extended_size_header_is_truncated_at_eof() -> None:
    """A truncated file whose last box declares the extended (64-bit) size form doesn't crash the walk.

    Only 10 bytes are available for a box whose full extended header needs 16 -- confirms the walk
    treats this the same as any other incomplete trailing box (BoxNotFoundError, the documented
    outcome for "no moov found"), not a raw struct.error escaping the walk.
    """
    prefix = _box("ftyp", b"isom") + _box("mdat", b"data")
    truncated_extended_header = struct.pack(">I", 1) + b"free" + b"\x00\x00"  # only 2 of 8 size64 bytes present
    buf = prefix + truncated_extended_header
    fetcher = _CountingFetcher(buf)

    with pytest.raises(BoxNotFoundError):
        bl.get_moov_full_walk(len(buf), fetcher)


# ---------------------------------------------------------------------------
# get_moov_tail_guess -- tail-guess-first, falls back to the full walk
# ---------------------------------------------------------------------------


def test_get_moov_tail_guess_hits_on_the_first_fetch_and_makes_no_second_fetch() -> None:
    """A moov that fits inside the initial tail guess costs exactly one fetch -- no follow-up."""
    buf = _box("ftyp", b"isom") + _box("mdat", b"data" * 1000) + _box("moov", b"payload")
    fetcher = _CountingFetcher(buf)

    result = bl.get_moov_tail_guess(len(buf), fetcher, initial_tail_bytes=64)

    assert result == _box("moov", b"payload")
    assert len(fetcher.calls) == 1


def test_get_moov_tail_guess_needs_the_retry_when_the_first_guess_is_too_small() -> None:
    """A moov bigger than the initial guess is still found -- on the second, larger fetch."""
    moov = _box("moov", b"p" * 100)
    buf = _box("ftyp", b"isom") + _box("mdat", b"data" * 1000) + moov
    fetcher = _CountingFetcher(buf)

    result = bl.get_moov_tail_guess(len(buf), fetcher, initial_tail_bytes=32, retry_multiplier=8)

    assert result == moov
    assert len(fetcher.calls) == 2


def test_get_moov_tail_guess_falls_back_to_the_full_walk_for_a_leading_box() -> None:
    """A moov before mdat isn't in any tail guess -- both tail fetches miss, then the walk finds it."""
    buf = _box("ftyp", b"isom") + _box("moov", b"payload") + _box("mdat", b"data" * 1000)
    fetcher = _CountingFetcher(buf)

    result = bl.get_moov_tail_guess(len(buf), fetcher, initial_tail_bytes=16, retry_multiplier=2)

    assert result == _box("moov", b"payload")
    # 2 tail attempts (both miss) + however many header hops the walk fallback took, + one content fetch.
    assert len(fetcher.calls) > 3


def test_get_moov_tail_guess_ignores_a_coincidental_type_code_byte_match() -> None:
    """A literal 'moov' substring elsewhere in the tail is rejected: what follows it isn't well-formed boxes."""
    decoy_content = b"xx" + b"moov" + b"xx"
    buf = _box("ftyp", b"isom") + _box("free", decoy_content) + _box("moov", b"payload")
    fetcher = _CountingFetcher(buf)

    result = bl.get_moov_tail_guess(len(buf), fetcher, initial_tail_bytes=len(buf))

    assert result == _box("moov", b"payload")


def test_get_moov_tail_guess_accepts_moov_trailed_by_a_well_formed_box() -> None:
    """Moov need not be the file's literal last box -- real unit output trails it with one.

    Measured against a real corpus: a multi-GB file whose ``moov`` ends
    several MB before file_size, with a
    ``skip``-typed box filling the remainder. An exact end-of-file match
    would always miss this shape and fall back to a full walk on every such
    file -- expensive when (also measured) the file is built from many small
    top-level box pairs.
    """
    trailing = _box("skip", b"p" * 64)
    buf = _box("ftyp", b"isom") + _box("mdat", b"data" * 1000) + _box("moov", b"payload") + trailing
    fetcher = _CountingFetcher(buf)

    result = bl.get_moov_tail_guess(len(buf), fetcher, initial_tail_bytes=len(buf))

    assert result == _box("moov", b"payload")


def test_get_moov_tail_guess_prefers_the_rightmost_match_over_an_earlier_lookalike() -> None:
    """When two candidates could both validate, the one nearer file_size wins -- checked first, not last.

    A forward scan that accepted the first validating candidate risked a
    coincidental earlier match validating *by accident* (its own declared,
    wrong end position happening to line up with where the real trailing
    boxes actually start) before the scan ever reached the genuine, later
    moov. Scanning from the end backward means the real one is considered
    first and returned, without ever depending on the earlier lookalike
    failing to validate.
    """
    real_moov = _box("moov", b"payload")
    # A byte sequence containing the literal "moov" type code, sized so that if it were (wrongly)
    # treated as a real box, its declared end would land exactly on real_moov's own start --
    # the exact coincidence that made a forward, first-match scan risky.
    lookalike = struct.pack(">I", 16) + b"moov" + b"\x00" * 8
    buf = _box("ftyp", b"isom") + _box("free", lookalike) + real_moov
    fetcher = _CountingFetcher(buf)

    result = bl.get_moov_tail_guess(len(buf), fetcher, initial_tail_bytes=len(buf))

    assert result == real_moov


def test_get_moov_tail_guess_raises_when_absent_everywhere() -> None:
    """No matching box at all still raises, via the walk fallback, not a silent wrong answer."""
    buf = _box("ftyp", b"isom") + _box("mdat", b"data")
    fetcher = _CountingFetcher(buf)

    with pytest.raises(BoxNotFoundError):
        bl.get_moov_tail_guess(len(buf), fetcher, initial_tail_bytes=8)


def test_get_moov_tail_guess_falls_back_cleanly_when_trailing_validation_hits_a_truncated_extended_header() -> None:
    """A malformed trailing structure that happens to declare the extended size form doesn't crash validation.

    Real moov's own trailing-well-formedness check would need to read a 16-byte extended header,
    but only 10 bytes remain to file_size -- self-contradictory, and the straddle this test forces
    onto _trailing_bytes_are_well_formed_boxes's own extended-size handling. The tail guess must
    reject this candidate gracefully (not crash) and fall through to the full-walk fallback, which
    finds the real moov directly without ever reaching the malformed trailing bytes.
    """
    real_moov = _box("moov", b"payload")
    malformed_trailing = struct.pack(">I", 1) + b"skip" + b"\x00\x00"  # only 2 of 8 size64 bytes present
    buf = _box("ftyp", b"isom") + _box("mdat", b"data" * 1000) + real_moov + malformed_trailing
    fetcher = _CountingFetcher(buf)

    result = bl.get_moov_tail_guess(len(buf), fetcher, initial_tail_bytes=len(buf))

    assert result == real_moov


# ---------------------------------------------------------------------------
# get_moov_head_guess -- head-guess-first, returns None (never raises) on a miss
# ---------------------------------------------------------------------------


def test_get_moov_head_guess_hits_on_the_first_fetch_and_makes_no_second_fetch() -> None:
    """A moov right after ftyp costs exactly one fetch -- no follow-up."""
    buf = _box("ftyp", b"isom") + _box("moov", b"payload") + _box("mdat", b"data" * 1000)
    fetcher = _CountingFetcher(buf)

    result = bl.get_moov_head_guess(len(buf), fetcher, initial_head_bytes=64)

    assert result == _box("moov", b"payload")
    assert len(fetcher.calls) == 1


def test_get_moov_head_guess_needs_the_retry_when_the_first_guess_is_too_small() -> None:
    """A moov bigger than the initial guess is still found -- on the second, larger fetch."""
    moov = _box("moov", b"p" * 100)
    buf = _box("ftyp", b"isom") + moov + _box("mdat", b"data" * 1000)
    fetcher = _CountingFetcher(buf)

    result = bl.get_moov_head_guess(len(buf), fetcher, initial_head_bytes=32, retry_multiplier=8)

    assert result == moov
    assert len(fetcher.calls) == 2


def test_get_moov_head_guess_stays_inconclusive_when_an_extended_size_header_is_split_by_the_fetch_size() -> None:
    """A head fetch ending 8-15 bytes into an extended-size box's header returns None, not a crash.

    initial_head_bytes lands exactly in the middle of an earlier box's 64-bit size field -- enough
    for the existing 8-byte check to pass, not enough for the 8 more bytes the extended form needs.
    """
    prefix = _box("ftyp", b"isom")
    content = b"y" * 50
    extended_box = struct.pack(">I", 1) + b"free" + struct.pack(">Q", 16 + len(content)) + content
    buf = prefix + extended_box + _box("moov", b"payload")
    head_bytes = len(prefix) + 10  # 10 of the box's 16 header bytes -- the straddle this test forces
    fetcher = _CountingFetcher(buf)

    result = bl.get_moov_head_guess(len(buf), fetcher, initial_head_bytes=head_bytes, retry_multiplier=1)

    assert result is None


def test_get_moov_head_guess_returns_none_for_a_trailing_box() -> None:
    """A moov after mdat is not in any head guess -- returns None, doesn't raise, doesn't fall back itself."""
    buf = _box("ftyp", b"isom") + _box("mdat", b"data" * 1000) + _box("moov", b"payload")
    fetcher = _CountingFetcher(buf)

    result = bl.get_moov_head_guess(len(buf), fetcher, initial_head_bytes=16, retry_multiplier=2)

    assert result is None


def test_get_moov_head_guess_returns_none_rather_than_raising_when_absent_everywhere() -> None:
    """No matching box anywhere still returns None -- this function never raises, unlike get_moov_tail_guess."""
    buf = _box("ftyp", b"isom") + _box("mdat", b"data")
    fetcher = _CountingFetcher(buf)

    result = bl.get_moov_head_guess(len(buf), fetcher, initial_head_bytes=8)

    assert result is None


# ---------------------------------------------------------------------------
# get_moov_auto -- head guess, then tail guess (which falls back to the full walk)
# ---------------------------------------------------------------------------


def test_get_moov_auto_uses_the_head_guess_for_a_leading_moov() -> None:
    """A leading moov is found via the head guess alone -- one fetch, no tail fallback triggered.

    A single fetch call is proof, not just a plausible guess: had get_moov_auto
    tried the tail guess first (wrongly) on this leading-moov buffer, that
    guess would miss, forcing at least a retry and the full-walk fallback --
    strictly more than one call.
    """
    buf = _box("ftyp", b"isom") + _box("moov", b"payload") + _box("mdat", b"data" * 1000)
    fetcher = _CountingFetcher(buf)

    result = bl.get_moov_auto(len(buf), fetcher)

    assert result == _box("moov", b"payload")
    assert len(fetcher.calls) == 1


def test_get_moov_auto_falls_back_to_the_tail_guess_for_a_trailing_moov(monkeypatch: pytest.MonkeyPatch) -> None:
    """A head-guess miss falls through to the tail guess that follows it.

    The head guess's own default size (8 MiB) would swallow this whole small
    test buffer in one fetch regardless of where moov actually sits, so a
    real miss can't be forced without an unwieldy multi-MB buffer -- monkeypatching
    get_moov_head_guess to return None directly exercises the real fallback
    wiring instead: get_moov_auto must call get_moov_tail_guess next, not
    stop or raise on the first None.
    """
    buf = _box("ftyp", b"isom") + _box("mdat", b"data" * 1000) + _box("moov", b"payload")
    fetcher = _CountingFetcher(buf)
    monkeypatch.setattr(bl, "get_moov_head_guess", lambda *_args, **_kwargs: None)

    result = bl.get_moov_auto(len(buf), fetcher)

    assert result == _box("moov", b"payload")


def test_get_moov_auto_raises_when_absent_everywhere() -> None:
    """No matching box at all still raises, via the tail guess's own full-walk fallback."""
    buf = _box("ftyp", b"isom") + _box("mdat", b"data")
    fetcher = _CountingFetcher(buf)

    with pytest.raises(BoxNotFoundError):
        bl.get_moov_auto(len(buf), fetcher)
