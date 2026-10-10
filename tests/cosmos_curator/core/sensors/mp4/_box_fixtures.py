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
"""Shared ISO-BMFF box-packing helpers for building synthetic MP4 fixtures in tests.

Test infrastructure, not recipe domain logic -- for any test needing a real,
minimal box layout rather than mocking box_locate/box_primitives/moof_walk
directly. Lives here, not in one test module, because multiple test files
need the identical fragmented (:func:`pack_traf`/:func:`pack_moof`) and
non-fragmented (:func:`synthetic_moov` and its own helpers -- ``avc1_stsd``,
``mvhd_box``, ``tkhd_box``, ...) box shapes.
"""

import struct

#: The real, hand-decoded High-profile avcC bytes from a real corpus file's own extradata --
#: SPS/PPS content is never decoded by moov.py, so their bytes are arbitrary here; only the
#: fixed header and extension fields matter.
_REAL_SPS_BYTES = bytes(range(21))
_REAL_PPS_BYTES = bytes([0xAA, 0xBB, 0xCC, 0xDD])
_REAL_AVCC_EXTENSION_BYTES = bytes([0xFD, 0xF8, 0xF8, 0x00])

TKHD_MATRIX_IDENTITY = (0x00010000, 0, 0, 0, 0x00010000, 0, 0, 0, 0x40000000)


def pack_box(box_type: str, content: bytes) -> bytes:
    """Pack one standard (32-bit size) box."""
    return struct.pack(">I", 8 + len(content)) + box_type.encode("ascii") + content


def pack_full_box(box_type: str, version: int, content_after_flags: bytes) -> bytes:
    """Pack one ``FullBox`` (version byte + 3 reserved flag bytes prepended to *content_after_flags*)."""
    return pack_box(box_type, bytes([version]) + b"\x00\x00\x00" + content_after_flags)


def pack_traf(
    tfhd: bytes = b"tfhd-placeholder", tfdt: bytes = b"tfdt-placeholder", trun: bytes = b"trun-placeholder"
) -> bytes:
    """Pack a ``traf`` box with placeholder ``tfhd``/``tfdt``/``trun`` children -- structure real, content arbitrary."""
    return pack_box("traf", pack_box("tfhd", tfhd) + pack_box("tfdt", tfdt) + pack_box("trun", trun))


def pack_moof(traf_content: bytes | None = None, *, siblings: bytes = b"") -> bytes:
    """Pack a ``moof`` box around *traf_content* (default: :func:`pack_traf`'s own), plus any sibling boxes."""
    return pack_box("moof", (traf_content if traf_content is not None else pack_traf()) + siblings)


# tfhd/trun flags (ISO/IEC 14496-12 §8.8.7/§8.8.8) -- see fragment_table.py's own identical constants.
_TFHD_DEFAULT_SAMPLE_DURATION_PRESENT = 0x000008
_TFHD_DEFAULT_BASE_IS_MOOF = 0x020000
_TRUN_DATA_OFFSET_PRESENT = 0x000001
_TRUN_SAMPLE_DURATION_PRESENT = 0x000100
_TRUN_SAMPLE_SIZE_PRESENT = 0x000200
_TRUN_SAMPLE_FLAGS_PRESENT = 0x000400
_TRUN_SAMPLE_COMPOSITION_TIME_OFFSETS_PRESENT = 0x000800
_SAMPLE_IS_NON_SYNC_SAMPLE_BIT = 0x00010000


def real_tfhd(
    *, track_id: int = 1, default_sample_duration: int | None = 3000, default_base_is_moof: bool = True
) -> bytes:
    """Pack a ``tfhd`` matching the flags observed against a real corpus (``0x020008``, see ``fragment_table.py``)."""
    flags = 0
    if default_base_is_moof:
        flags |= _TFHD_DEFAULT_BASE_IS_MOOF
    content = struct.pack(">I", track_id)
    if default_sample_duration is not None:
        flags |= _TFHD_DEFAULT_SAMPLE_DURATION_PRESENT
        content += struct.pack(">I", default_sample_duration)
    return pack_box("tfhd", bytes([0]) + flags.to_bytes(3, "big") + content)


def real_tfdt(*, base_media_decode_time: int, version: int = 1) -> bytes:
    """Pack a ``tfdt`` (version-0 32-bit or version-1 64-bit ``baseMediaDecodeTime``)."""
    if version == 1:
        return pack_box("tfdt", bytes([1]) + b"\x00\x00\x00" + struct.pack(">Q", base_media_decode_time))
    return pack_box("tfdt", bytes([0]) + b"\x00\x00\x00" + struct.pack(">I", base_media_decode_time))


def real_trun(
    *,
    data_offset: int,
    sample_size: int,
    is_keyframe: bool = True,
    sample_count: int = 1,
    composition_offset: int | None = None,
    sample_duration: int | None = None,
) -> bytes:
    """Pack a ``trun`` matching the flags observed against a real corpus (``0x000601``, see ``fragment_table.py``)."""
    flags = _TRUN_DATA_OFFSET_PRESENT | _TRUN_SAMPLE_SIZE_PRESENT | _TRUN_SAMPLE_FLAGS_PRESENT
    sample_flags = 0 if is_keyframe else _SAMPLE_IS_NON_SYNC_SAMPLE_BIT
    content = struct.pack(">I", sample_count) + struct.pack(">i", data_offset)
    if sample_duration is not None:
        flags |= _TRUN_SAMPLE_DURATION_PRESENT
        content += struct.pack(">I", sample_duration)
    content += struct.pack(">I", sample_size) + struct.pack(">I", sample_flags)
    if composition_offset is not None:
        flags |= _TRUN_SAMPLE_COMPOSITION_TIME_OFFSETS_PRESENT
        content += struct.pack(">i", composition_offset)
    return pack_box("trun", bytes([0]) + flags.to_bytes(3, "big") + content)


def real_traf(
    *,
    decode_time: int,
    data_offset: int,
    sample_size: int,
    is_keyframe: bool = True,
    default_sample_duration: int | None = 3000,
    default_base_is_moof: bool = True,
    sample_count: int = 1,
    composition_offset: int | None = None,
    trun_sample_duration: int | None = None,
) -> bytes:
    """Pack a real, byte-accurate ``traf`` -- the shape :mod:`fragment_table` actually parses.

    *default_sample_duration* is ``tfhd``'s own fallback; *trun_sample_duration* is this one
    sample's own ``trun``-level override -- named separately since real files can carry either,
    both, or neither, and :mod:`fragment_table` resolves them with different precedence.
    """
    return pack_box(
        "traf",
        real_tfhd(default_sample_duration=default_sample_duration, default_base_is_moof=default_base_is_moof)
        + real_tfdt(base_media_decode_time=decode_time)
        + real_trun(
            data_offset=data_offset,
            sample_size=sample_size,
            is_keyframe=is_keyframe,
            sample_count=sample_count,
            composition_offset=composition_offset,
            sample_duration=trun_sample_duration,
        ),
    )


def trex_box(*, default_sample_duration: int = 0, default_sample_size: int = 0, default_sample_flags: int = 0) -> bytes:
    """Pack a minimal ``mvex/trex`` box -- this recipe's own fallback-defaults source."""
    trex_content = struct.pack(">5I", 1, 1, default_sample_duration, default_sample_size, default_sample_flags)
    return pack_box("mvex", pack_full_box("trex", 0, trex_content))


def synthetic_fragmented_moov(
    *,
    fourcc: str = "avc1",
    extradata: bytes | None = None,
    timescale: int = 30,
    duration: int = 0,
    mvhd_version: int = 0,
    tkhd_version: int = 0,
    include_elst: bool = False,
    include_trex: bool = True,
    trex_sample_duration: int = 0,
    minf_extra_children: bytes = b"",
    mdia_extra_children: bytes = b"",
) -> bytes:
    """Pack a full, real, minimal fragmented ``moov`` box -- real ``stsd``, empty ``stbl`` around it.

    Matches a real corpus file's own shape: ``mvhd``/``tkhd``/``mdhd``/``stsd`` are all real and
    ordinary (codec description doesn't depend on fragmentation); ``stbl`` carries only ``stsd``
    -- no ``stts``/``stsz``/``stco``/``stsc``, since every sample's own data lives in its own
    fragment instead. ``mvex/trex`` is a sibling of ``trak``, present by default (a real file
    carries one), omittable to exercise the "no trex" fallback path.
    """
    width, height = 640, 480
    if extradata is None:
        extradata = bytes([1, 0x42, 0xC0, 0x1E, 0xFF, 0xE1, 0, 0])
    stsd = avc1_stsd(width, height, extradata) if fourcc == "avc1" else hevc_stsd(width, height)
    stbl = pack_box("stbl", stsd)
    minf = pack_box("minf", stbl + minf_extra_children)
    mdhd = pack_full_box("mdhd", 0, struct.pack(">III", 0, 0, timescale) + struct.pack(">I", duration))
    mdia = pack_box("mdia", mdhd + minf + mdia_extra_children)
    tkhd = tkhd_box(version=tkhd_version)
    edts = pack_box("edts", elst_box()) if include_elst else b""
    trak = pack_box("trak", tkhd + mdia + edts)
    mvhd = mvhd_box(version=mvhd_version)
    mvex = trex_box(default_sample_duration=trex_sample_duration) if include_trex else b""
    return pack_box("moov", mvhd + trak + mvex)


def avcc_extradata(profile_idc: int, *, include_extension: bool = True) -> bytes:
    """Build avcC extradata matching a real, hand-decoded corpus example.

    ``lengthSizeMinusOne=3`` (-> nal_length_size=4), one SPS, one PPS, and
    (when *include_extension*) the profile>=100 extension bytes
    ``chroma_format_idc=1, bit_depth_luma_minus8=0, bit_depth_chroma_minus8=0``.
    """
    header = bytes([0x01, profile_idc, 0x00, 0x1F, 0xFF, 0xE1])
    body = header + struct.pack(">H", len(_REAL_SPS_BYTES)) + _REAL_SPS_BYTES
    body += bytes([0x01]) + struct.pack(">H", len(_REAL_PPS_BYTES)) + _REAL_PPS_BYTES
    if include_extension:
        body += _REAL_AVCC_EXTENSION_BYTES
    return body


def avc1_stsd(width: int, height: int, extradata: bytes, *, extra_children: bytes = b"", entry_count: int = 1) -> bytes:
    """Pack a full ``stsd`` box wrapping one ``avc1`` sample entry around *extradata*'s own ``avcC``."""
    avcc = pack_box("avcC", extradata)
    fixed = struct.pack(">HH", 0, 0) + struct.pack(">III", 0, 0, 0) + struct.pack(">HH", width, height)
    fixed += struct.pack(">II", 0x00480000, 0x00480000) + struct.pack(">I", 0) + struct.pack(">H", 1)
    fixed += b"\x00" * 32 + struct.pack(">H", 0x0018) + struct.pack(">h", -1)
    entry = struct.pack(">6sH", b"\x00" * 6, 1) + fixed + avcc + extra_children  # reserved(6)+data_ref_index(2)
    avc1 = pack_box("avc1", entry)
    return pack_full_box("stsd", 0, struct.pack(">I", entry_count) + avc1)


def hevc_stsd(width: int, height: int) -> bytes:
    """Pack a full ``stsd`` box wrapping one ``hev1`` sample entry with no ``hvcC`` at all."""
    fixed = struct.pack(">HH", 0, 0) + struct.pack(">III", 0, 0, 0) + struct.pack(">HH", width, height)
    fixed += struct.pack(">II", 0x00480000, 0x00480000) + struct.pack(">I", 0) + struct.pack(">H", 1)
    fixed += b"\x00" * 32 + struct.pack(">H", 0x0018) + struct.pack(">h", -1)
    entry = struct.pack(">6sH", b"\x00" * 6, 1) + fixed  # no hvcC at all -- exercises the "box not found" path
    hev1 = pack_box("hev1", entry)
    return pack_full_box("stsd", 0, struct.pack(">I", 1) + hev1)


def mvhd_box(*, version: int = 0, creation_time: int = 0, modification_time: int = 0) -> bytes:
    """Pack a ``mvhd`` box -- only creation/modification time are read by any current parser."""
    if version == 1:
        content = bytes([1]) + b"\x00\x00\x00" + struct.pack(">QQ", creation_time, modification_time)
    else:
        content = bytes([0]) + b"\x00\x00\x00" + struct.pack(">II", creation_time, modification_time)
    return pack_box("mvhd", content)


def tkhd_box(
    *,
    version: int = 0,
    creation_time: int = 0,
    modification_time: int = 0,
    matrix: tuple[int, ...] = TKHD_MATRIX_IDENTITY,
    width: int = 640 << 16,
    height: int = 480 << 16,
) -> bytes:
    """Pack a ``tkhd`` box with a real, spec-shaped matrix/width/height tail."""
    if version == 1:
        head = bytes([1]) + b"\x00\x00\x00" + struct.pack(">QQ", creation_time, modification_time)
        head += struct.pack(">I", 1) + b"\x00\x00\x00\x00" + struct.pack(">Q", 0)
    else:
        head = bytes([0]) + b"\x00\x00\x00" + struct.pack(">II", creation_time, modification_time)
        head += struct.pack(">I", 1) + b"\x00\x00\x00\x00" + struct.pack(">I", 0)
    tail = b"\x00" * 16  # reserved(8)+layer(2)+alternate_group(2)+volume(2)+reserved(2)
    matrix_bytes = struct.pack(">9i", *matrix)
    return pack_box("tkhd", head + tail + matrix_bytes + struct.pack(">II", width, height))


def elst_box() -> bytes:
    """Pack a minimal one-entry, version-0 ``elst`` box -- content is never decoded, only preserved verbatim."""
    return pack_full_box("elst", 0, struct.pack(">I", 1) + struct.pack(">Iii", 1000, 0, 0x00010000))


def edit_list_box(
    *,
    entries: list[tuple[int, int, int, int]] | None = None,
    version: int = 0,
) -> bytes:
    """Pack an ``elst`` box from explicit ``(segment_duration, media_time, rate_integer, rate_fraction)`` entries.

    Unlike :func:`elst_box` (a fixed, minimal normal-rate single entry), lets a caller build
    the shapes ``_edit_list_media_time_ticks`` (``cosmos_curator.core.sensors.data.mp4``)
    refuses -- multiple entries, an empty edit (``media_time=-1``), or a non-1x rate -- plus
    version-1 (64-bit) entries.
    """
    entry_list = entries if entries is not None else [(1000, 0, 1, 0)]
    body = struct.pack(">I", len(entry_list))
    for segment_duration, media_time, rate_integer, rate_fraction in entry_list:
        if version == 1:
            body += struct.pack(">Qq", segment_duration, media_time)
        else:
            body += struct.pack(">Ii", segment_duration, media_time)
        body += struct.pack(">hh", rate_integer, rate_fraction)
    return pack_full_box("elst", version, body)


def uuid_box(extension_type: bytes, payload: bytes) -> bytes:
    """Pack a ``uuid`` box from a 16-byte *extension_type* and arbitrary *payload*."""
    return pack_box("uuid", extension_type + payload)


def synthetic_moov(
    *,
    frame_count: int,
    keyframe_index: int | None,
    ctts_offset: int | None,
    ctts_sample_count: int | None = None,
    fourcc: str = "avc1",
    extradata: bytes | None = None,
    stsd_extra_children: bytes = b"",
    stsd_entry_count: int = 1,
    stbl_extra_children: bytes = b"",
    minf_extra_children: bytes = b"",
    mdia_extra_children: bytes = b"",
    moov_extra_children: bytes = b"",
    mvhd_version: int = 0,
    tkhd_version: int = 0,
    include_elst: bool = False,
    elst_content: bytes | None = None,
    stsz_sample_count: int | None = None,
    stsc_samples_per_chunk: int | None = None,
    timescale: int = 12288,
    tkhd_width: int = 640 << 16,
    tkhd_height: int = 480 << 16,
) -> bytes:
    """Pack a full, real, minimal non-fragmented ``moov`` box for ``parse_video_track`` tests."""
    width, height = 640, 480
    if extradata is None:
        extradata = bytes([1, 0x42, 0xC0, 0x1E, 0xFF, 0xE1, 0, 0])  # avcC header: profile=0x42, level=0x1e
    stsd = (
        avc1_stsd(width, height, extradata, extra_children=stsd_extra_children, entry_count=stsd_entry_count)
        if fourcc == "avc1"
        else hevc_stsd(width, height)
    )

    stts = pack_full_box("stts", 0, struct.pack(">I", 1) + struct.pack(">II", frame_count, 512))
    stss = (
        pack_full_box("stss", 0, struct.pack(">I", 1) + struct.pack(">I", keyframe_index + 1))
        if keyframe_index is not None
        else b""
    )
    # stsz_sample_count defaults to frame_count (a conformant file); a caller passes a
    # different value deliberately, to build a stsz/stts sample-count mismatch.
    sample_count = frame_count if stsz_sample_count is None else stsz_sample_count
    sizes = [1000 + i for i in range(sample_count)]
    stsz = pack_full_box("stsz", 0, struct.pack(">II", 0, sample_count) + struct.pack(f">{sample_count}I", *sizes))
    # stsc_samples_per_chunk defaults to frame_count (one chunk holding every sample); a
    # caller passes a different value deliberately, to build a stsc/stts offset-count mismatch.
    samples_per_chunk = frame_count if stsc_samples_per_chunk is None else stsc_samples_per_chunk
    stsc = pack_full_box("stsc", 0, struct.pack(">I", 1) + struct.pack(">III", 1, samples_per_chunk, 1))
    stco = pack_full_box("stco", 0, struct.pack(">I", 1) + struct.pack(">I", 5000))

    stbl_children = stsd + stts + stss + stsz + stsc + stco
    if ctts_offset is not None:
        # ctts_sample_count defaults to frame_count (a conformant file); a caller
        # passes a different value deliberately, to build a non-conformant one.
        entry_sample_count = frame_count if ctts_sample_count is None else ctts_sample_count
        stbl_children += pack_full_box(
            "ctts", 0, struct.pack(">I", 1) + struct.pack(">Ii", entry_sample_count, ctts_offset)
        )
    stbl = pack_box("stbl", stbl_children + stbl_extra_children)

    minf = pack_box("minf", stbl + minf_extra_children)
    mdhd = pack_full_box("mdhd", 0, struct.pack(">III", 0, 0, timescale) + struct.pack(">I", frame_count * 512))
    mdia = pack_box("mdia", mdhd + minf + mdia_extra_children)
    tkhd = tkhd_box(version=tkhd_version, width=tkhd_width, height=tkhd_height)
    edts = pack_box("edts", elst_content if elst_content is not None else elst_box()) if include_elst else b""
    trak = pack_box("trak", tkhd + mdia + edts)
    mvhd = mvhd_box(version=mvhd_version)
    return pack_box("moov", mvhd + trak + moov_extra_children)
