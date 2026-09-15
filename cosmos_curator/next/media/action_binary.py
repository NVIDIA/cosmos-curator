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

"""ACT2 binary codec for per-frame robot action data.

Binary layout::

    bytes[0:4]    magic b"ACT2"
    bytes[4:8]    little-endian uint32 JSON length
    bytes[8:1024] UTF-8 JSON header padded to 1024 bytes total (including prefix)
    bytes[1024:]  fixed-stride frame records then per-clip tail
"""

import json
from collections.abc import Iterable, Mapping
from math import prod
from typing import Any

import attrs
import numpy as np

MAGIC: bytes = b"ACT2"
_VERSION: int = 2
_HEADER_SIZE: int = 1024
_HEADER_PREFIX_SIZE: int = 8
_DTYPE = np.dtype("<f4")


@attrs.frozen
class ActionFieldSpec:
    """One named array within a frame or clip record."""

    name: str
    shape: tuple[int, ...]

    @property
    def value_count(self) -> int:
        """Total number of float32 values in this field."""
        return prod(self.shape)


@attrs.frozen
class ActionBinarySpec:
    """Reusable fixed-stride action layout for one dataset family."""

    per_frame_fields: tuple[ActionFieldSpec, ...]
    per_clip_fields: tuple[ActionFieldSpec, ...] = ()

    @property
    def per_frame_value_count(self) -> int:
        """Total number of values in one frame record."""
        return sum(field.value_count for field in self.per_frame_fields)

    @property
    def frame_record_size_bytes(self) -> int:
        """Byte size of one frame record."""
        return self.per_frame_value_count * _DTYPE.itemsize

    @property
    def per_clip_value_count(self) -> int:
        """Total number of values in the per-clip tail."""
        return sum(field.value_count for field in self.per_clip_fields)

    @property
    def per_clip_size_bytes(self) -> int:
        """Byte size of the per-clip tail."""
        return self.per_clip_value_count * _DTYPE.itemsize


ACTION_BINARY_SPECS: dict[str, ActionBinarySpec] = {
    "libero": ActionBinarySpec(
        per_frame_fields=(
            ActionFieldSpec(name="action", shape=(7,)),
            ActionFieldSpec(name="state", shape=(8,)),
        ),
    ),
    "mecka": ActionBinarySpec(
        per_frame_fields=(
            ActionFieldSpec(name="hand_left_cam", shape=(63,)),
            ActionFieldSpec(name="hand_right_cam", shape=(63,)),
            ActionFieldSpec(name="hand_left_cam_rotation", shape=(84,)),
            ActionFieldSpec(name="hand_right_cam_rotation", shape=(84,)),
            ActionFieldSpec(name="camera_position", shape=(3,)),
            ActionFieldSpec(name="camera_rotation", shape=(4,)),
        ),
        per_clip_fields=(ActionFieldSpec(name="intrinsics", shape=(8,)),),
    ),
    "droid_lerobot": ActionBinarySpec(
        per_frame_fields=(
            ActionFieldSpec(name="action", shape=(7,)),
            ActionFieldSpec(name="state", shape=(7,)),
        ),
    ),
    "robomind_franka": ActionBinarySpec(
        per_frame_fields=(
            ActionFieldSpec(name="action", shape=(8,)),
            ActionFieldSpec(name="state_joint_position", shape=(8,)),
            ActionFieldSpec(name="state_end_effector", shape=(6,)),
        ),
    ),
    "robomind_franka_dual": ActionBinarySpec(
        per_frame_fields=(
            ActionFieldSpec(name="action", shape=(16,)),
            ActionFieldSpec(name="state_joint_position", shape=(16,)),
            ActionFieldSpec(name="state_end_effector", shape=(12,)),
        ),
    ),
    "robomind_ur": ActionBinarySpec(
        per_frame_fields=(ActionFieldSpec(name="action", shape=(7,)),),
    ),
}

ACTION_BINARY_SPEC_BY_DATASET: dict[str, str] = {
    "libero_10": "libero",
    "libero_90": "libero",
    "feb_08_500hr_lerobot_no_bframes": "mecka",
    "mar_6_ego_dexterous_lerobot_no_bframes": "mecka",
    "feb_08_500hr_lerobot_updated": "mecka",
    "feb_15_1500hr_lerobot_no_bframes": "mecka",
    "feb_23_1000hr_lerobot_no_bframes": "mecka",
    "mar_02_1000hr_lerobot": "mecka",
    "mar_09_4000hr_lerobot": "mecka",
    "mar_16_7000hr_lerobot": "mecka",
    "mar_30_9000hr_lerobot": "mecka",
    "apr_03_lerobot": "mecka",
    "apr_06_10000hr_lerobot": "mecka",
    "20260509_46000h_everyday_mono_no_wrist_lerobot": "mecka",
    "droid_lerobot": "droid_lerobot",
    "droid_plus_lerobot_640x360_20260412": "droid_lerobot",
    "droid_plus_lerobot_640x360_20260412_success": "droid_lerobot",
    "robomind-franka": "robomind_franka",
    "robomind_franka": "robomind_franka",
    "robomind-franka-dual": "robomind_franka_dual",
    "robomind_franka_dual": "robomind_franka_dual",
    "robomind-ur": "robomind_ur",
    "robomind_ur": "robomind_ur",
}


def get_action_binary_spec(source_dataset: str) -> ActionBinarySpec:
    """Return the fixed-stride spec for *source_dataset*.

    Raises ValueError for unregistered dataset names.
    """
    try:
        spec_name = ACTION_BINARY_SPEC_BY_DATASET[source_dataset]
        return ACTION_BINARY_SPECS[spec_name]
    except KeyError as exc:
        msg = f"No action-binary spec registered for source dataset {source_dataset!r}."
        raise ValueError(msg) from exc


def _build_header(
    spec_name: str,
    spec: ActionBinarySpec,
    source_dataset: str,
    num_frames: int,
) -> bytes:
    """Build the 1024-byte ACT2 binary header."""
    per_frame_fields: list[dict[str, Any]] = []
    byte_offset = 0
    for field in spec.per_frame_fields:
        per_frame_fields.append({"name": field.name, "shape": list(field.shape), "byte_offset": byte_offset})
        byte_offset += field.value_count * _DTYPE.itemsize

    per_clip_fields: list[dict[str, Any]] = []
    byte_offset = 0
    for field in spec.per_clip_fields:
        per_clip_fields.append({"name": field.name, "shape": list(field.shape), "byte_offset": byte_offset})
        byte_offset += field.value_count * _DTYPE.itemsize

    per_clip_offset = _HEADER_SIZE + num_frames * spec.frame_record_size_bytes
    header_obj = {
        "dtype": _DTYPE.str,
        "format_version": _VERSION,
        "frame_data_offset": _HEADER_SIZE,
        "frame_record_size_bytes": spec.frame_record_size_bytes,
        "layout": "frame_major_with_per_clip_tail",
        "num_frames": num_frames,
        "per_clip_fields": per_clip_fields,
        "per_clip_offset": per_clip_offset,
        "per_clip_size_bytes": spec.per_clip_size_bytes,
        "per_frame_fields": per_frame_fields,
        "source_dataset": source_dataset,
        "spec_name": spec_name,
    }
    header_bytes = json.dumps(header_obj, separators=(",", ":"), sort_keys=True).encode("utf-8")
    if len(header_bytes) > _HEADER_SIZE - _HEADER_PREFIX_SIZE:
        msg = f"ACT2 header is too large: {len(header_bytes)} bytes"
        raise ValueError(msg)
    prefix = MAGIC + len(header_bytes).to_bytes(4, "little", signed=False)
    return prefix + header_bytes + bytes(_HEADER_SIZE - len(prefix) - len(header_bytes))


def encode_action_bin(action_data: Mapping[str, Any], source_dataset: str) -> bytes:
    """Return an ACT2 payload for *action_data* using the spec for *source_dataset*.

    Validates that the provided field names match the registered spec exactly,
    then concatenates the fixed-size header, frame-major body, and per-clip tail.
    """
    try:
        spec_name = ACTION_BINARY_SPEC_BY_DATASET[source_dataset]
        spec = ACTION_BINARY_SPECS[spec_name]
    except KeyError as exc:
        msg = f"No action-binary spec registered for source dataset {source_dataset!r}."
        raise ValueError(msg) from exc

    expected_names = {field.name for field in (*spec.per_frame_fields, *spec.per_clip_fields)}
    if set(action_data) != expected_names:
        msg = (
            f"Action fields for {source_dataset!r} must be exactly {sorted(expected_names)}, got {sorted(action_data)}."
        )
        raise ValueError(msg)

    # Build frame-major body.
    columns: list[np.ndarray[Any, Any]] = []
    num_frames: int | None = None
    for field in spec.per_frame_fields:
        array = np.asarray(action_data[field.name], dtype=_DTYPE)
        if array.ndim != len(field.shape) + 1 or array.shape[1:] != field.shape:
            msg = f"Action field {field.name!r} must have shape (frames, {field.shape}), got {array.shape}."
            raise ValueError(msg)
        if num_frames is None:
            num_frames = array.shape[0]
        elif array.shape[0] != num_frames:
            msg = f"Action field {field.name!r} has {array.shape[0]} frames; expected {num_frames}."
            raise ValueError(msg)
        columns.append(array.reshape(array.shape[0], field.value_count))

    if num_frames is None:
        num_frames = 0
    frame_data = np.concatenate(columns, axis=1) if columns else np.empty((0, 0), dtype=_DTYPE)

    # Build per-clip tail.
    per_clip_columns: list[np.ndarray[Any, Any]] = []
    for field in spec.per_clip_fields:
        array = np.asarray(action_data[field.name], dtype=_DTYPE)
        if array.shape != field.shape:
            msg = f"Per-clip field {field.name!r} must have shape {field.shape}, got {array.shape}."
            raise ValueError(msg)
        per_clip_columns.append(array.reshape(field.value_count))
    per_clip_data = np.concatenate(per_clip_columns) if per_clip_columns else np.empty((0,), dtype=_DTYPE)

    return (
        _build_header(spec_name, spec, source_dataset, num_frames)
        + np.ascontiguousarray(frame_data, dtype=_DTYPE).tobytes()
        + np.ascontiguousarray(per_clip_data, dtype=_DTYPE).tobytes()
    )


def _require(header: Mapping[str, Any], key: str) -> Any:  # noqa: ANN401 - returns a raw JSON value
    """Return ``header[key]`` or raise ``ValueError`` naming the missing key.

    Every header field is read through this accessor so a corrupt artifact
    surfaces as the documented ``ValueError`` (naming ACT2 and the field) rather
    than as an opaque ``KeyError`` that a caller following the ``Raises: ValueError``
    contract would not catch.
    """
    try:
        return header[key]
    except KeyError as exc:
        msg = f"ACT2 header is missing required key {key!r}"
        raise ValueError(msg) from exc


def _as_int(value: Any, field_label: str) -> int:  # noqa: ANN401 - coerces a raw JSON value
    """Coerce a header field to ``int`` or raise the module's ``ValueError``.

    ``int(...)`` raises ``TypeError`` for a JSON ``null`` (or any non-numeric
    value), which would escape the ``ValueError`` contract every malformed-artifact
    path in this module honours. Re-raise it as ``ValueError`` naming ACT2 and the
    field so a corrupt integer field is caught by the same handler as every other
    malformed header.
    """
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        msg = f"ACT2 header field {field_label!r} must be an integer, got {value!r}"
        raise ValueError(msg) from exc


def _shape_value_count(shape: Any, field_label: str) -> int:  # noqa: ANN401 - coerces a raw JSON value
    """Return ``prod(shape)`` as ``int`` after validating every dimension.

    ``prod(...)`` alone accepts a malformed shape that still multiplies to a
    plausible width - a negative dimension flips the sign, a float dimension makes
    it fractional - so the decoder would read the wrong byte count instead of
    failing. Each member must be a real non-negative ``int``; ``bool`` is an
    ``int`` subclass but not a valid dimension, so it is rejected too. A
    non-iterable / non-numeric shape re-raises as the module's ``ValueError`` so a
    malformed shape stays on the documented contract rather than escaping as an
    opaque ``TypeError``.
    """
    try:
        dims = list(shape)
    except TypeError as exc:
        msg = f"ACT2 header field {field_label!r} has an invalid shape {shape!r}"
        raise ValueError(msg) from exc
    for dim in dims:
        if isinstance(dim, bool) or not isinstance(dim, int) or dim < 0:
            msg = f"ACT2 header field {field_label!r} has an invalid shape {shape!r} (dimension {dim!r})"
            raise ValueError(msg)
    return int(prod(dims))


def _dtype_from_header(header: Mapping[str, Any]) -> np.dtype[Any]:
    """Resolve the header ``dtype`` string, mapping an unknown dtype to ``ValueError``."""
    raw = _require(header, "dtype")
    try:
        return np.dtype(str(raw))
    except TypeError as exc:
        msg = f"ACT2 header has an unknown dtype {raw!r}"
        raise ValueError(msg) from exc


def _parse_header(data: bytes) -> dict[str, Any]:
    """Parse and fully validate the 1024-byte ACT2 header from *data*.

    Validates both the *framing* (magic, JSON length prefix, JSON syntax,
    ``format_version``, overall length) and the *contents*: a JSON object with
    every declared field present (via :func:`_require`), the ``dtype`` this
    format version packs, and self-consistent offsets (per-clip tail after the
    frame body, no field spill).

    Returns:
        The parsed header mapping (spec_name, source_dataset, num_frames, offsets).

    Raises:
        ValueError: If the magic, header JSON, version, a required field, the
            dtype, the offsets, or the overall length is invalid.

    """
    if len(data) < _HEADER_SIZE:
        msg = f"ACT2 payload too short: {len(data)} bytes < {_HEADER_SIZE}-byte header"
        raise ValueError(msg)
    if data[: len(MAGIC)] != MAGIC:
        msg = f"not an ACT2 payload: magic {data[: len(MAGIC)]!r} != {MAGIC!r}"
        raise ValueError(msg)
    json_len = int.from_bytes(data[len(MAGIC) : _HEADER_PREFIX_SIZE], "little", signed=False)
    if json_len > _HEADER_SIZE - _HEADER_PREFIX_SIZE:
        msg = f"ACT2 header length {json_len} exceeds the {_HEADER_SIZE - _HEADER_PREFIX_SIZE}-byte header capacity"
        raise ValueError(msg)
    try:
        parsed: Any = json.loads(data[_HEADER_PREFIX_SIZE : _HEADER_PREFIX_SIZE + json_len])
    except json.JSONDecodeError as exc:
        msg = f"ACT2 header is not valid JSON: {exc}"
        raise ValueError(msg) from exc
    if not isinstance(parsed, dict):
        # ValueError (not TypeError) keeps the whole header contract uniform: every
        # malformed-artifact path in this module raises ValueError, which the action
        # leg catches to drop one row. See this function's ``Raises: ValueError``.
        msg = f"ACT2 header must be a JSON object, got {type(parsed).__name__}"
        raise ValueError(msg)  # noqa: TRY004
    header: dict[str, Any] = parsed
    version = header.get("format_version")
    if version != _VERSION:
        msg = f"unsupported ACT2 format_version {version!r}; this decoder supports {_VERSION}"
        raise ValueError(msg)
    _validate_header_layout(header, data)
    return header


def _validate_field_offsets(
    fields: Iterable[object],
    *,
    itemsize: int,
    region_size_bytes: int,
    field_kind: str,
    region_label: str,
) -> None:
    """Reject a malformed field entry or one whose ``byte_offset`` is invalid.

    Shared by the per-frame record and the per-clip tail: both are fixed-stride
    regions whose fields must be mappings sitting at a non-negative, dtype-item
    aligned ``byte_offset`` that fits inside the region. ``field_kind`` /
    ``region_label`` only shape the error text (``"per-frame"`` / ``"record"``
    vs ``"per-clip"`` / ``"tail"``).
    """
    for field in fields:
        # A non-mapping entry (e.g. a JSON list or scalar where an object was
        # expected) would make _require's subscript raise TypeError, which
        # escapes the ValueError contract decode() relies on - reject it here.
        if not isinstance(field, Mapping):
            msg = f"ACT2 header is self-inconsistent: {field_kind} field entry {field!r} is not a mapping"
            # ValueError (not TypeError) is the decode contract the per-row drop
            # handler catches; a raw TypeError would escape it.
            raise ValueError(msg)  # noqa: TRY004
        field_name = str(_require(field, "name"))
        # int()/prod() over raw JSON raise TypeError for a null or non-iterable
        # value; _shape_value_count / _as_int re-map that to the module's
        # ValueError so a malformed shape or offset stays on the decode contract.
        width_bytes = (
            _shape_value_count(_require(field, "shape"), f"{field_kind} field {field_name!r} shape") * itemsize
        )
        byte_offset = _as_int(_require(field, "byte_offset"), f"{field_kind} field {field_name!r} byte_offset")
        # A negative offset is a multiple of itemsize for the alignment check
        # (e.g. -4 % 4 == 0) and can pass the upper-bound check, but decode's
        # ``byte_offset // itemsize`` index would then slice from the tail of the
        # body array - silent misalignment. Reject it before the modulo test.
        if byte_offset < 0:
            msg = (
                f"ACT2 header is self-inconsistent: {field_kind} field {field_name!r} "
                f"byte_offset {byte_offset} is negative"
            )
            raise ValueError(msg)
        if byte_offset % itemsize:
            msg = (
                f"ACT2 header is self-inconsistent: {field_kind} field {field_name!r} "
                f"byte_offset {byte_offset} is not aligned to {itemsize}-byte dtype items"
            )
            raise ValueError(msg)
        if byte_offset + width_bytes > region_size_bytes:
            msg = (
                f"ACT2 header is self-inconsistent: {field_kind} field {field_name!r} "
                f"spans [{byte_offset}, {byte_offset + width_bytes}) past the {region_size_bytes}-byte {region_label}"
            )
            raise ValueError(msg)


def _validate_header_layout(header: Mapping[str, Any], data: bytes) -> None:
    """Check the header's field offsets are self-consistent and fit the buffer.

    Converts failure classes that would otherwise surface as opaque ``reshape`` /
    ``frombuffer`` errors (or silent corruption) into one clear message: a header
    declaring a dtype other than the one this format version packs, a header
    that disagrees with itself (frame data begins inside the fixed header, the
    per-clip tail does not begin where the frame body ends, a field spills past
    its record or tail, or a field entry is not a mapping or has a negative /
    dtype-misaligned ``byte_offset``) and a payload shorter than the header
    describes.
    """
    dtype = _dtype_from_header(header)
    # A zero-itemsize dtype (e.g. "V0") would make decode's
    # ``frame_record_size_bytes // dtype.itemsize`` divide by zero - a
    # ZeroDivisionError the per-row drop handler does not catch, so it must be
    # rejected here as the documented ValueError instead.
    if dtype.itemsize <= 0:
        msg = f"ACT2 header dtype {dtype.str!r} has zero itemsize; cannot decode fixed-stride records"
        raise ValueError(msg)
    # This format version packs one dtype only, so any other must be refused
    # rather than honored: a same-width dtype (e.g. "<i4") satisfies every
    # stride, offset and length check below, and decode's np.frombuffer would
    # then reinterpret the float32 payload under it - garbage numbers with no
    # error raised anywhere. Checked after the itemsize guard so a zero-itemsize
    # dtype keeps its own, more specific diagnosis.
    if dtype != _DTYPE:
        msg = (
            f"ACT2 header dtype {dtype.str!r} is not the {_DTYPE.str!r} fixed by format_version "
            f"{_VERSION}; refusing to reinterpret the payload"
        )
        raise ValueError(msg)
    num_frames = _as_int(_require(header, "num_frames"), "num_frames")
    frame_data_offset = _as_int(_require(header, "frame_data_offset"), "frame_data_offset")
    # Frame data must begin at or after the fixed header. An offset that points
    # inside the header would make decode's np.frombuffer read header / JSON
    # bytes as frame records - silent corruption that the tail and length checks
    # do NOT catch, because a smaller offset only shrinks the expected tail and
    # the required payload length.
    if frame_data_offset < _HEADER_SIZE:
        msg = (
            f"ACT2 header is self-inconsistent: frame_data_offset {frame_data_offset} points inside the "
            f"{_HEADER_SIZE}-byte fixed header; frame data must begin at or after the header"
        )
        raise ValueError(msg)
    frame_record_size_bytes = _as_int(_require(header, "frame_record_size_bytes"), "frame_record_size_bytes")
    if frame_record_size_bytes % dtype.itemsize:
        msg = (
            f"ACT2 header frame_record_size_bytes {frame_record_size_bytes} is not a whole "
            f"number of {dtype.itemsize}-byte dtype items"
        )
        raise ValueError(msg)
    per_clip_offset = _as_int(_require(header, "per_clip_offset"), "per_clip_offset")
    per_clip_size_bytes = _as_int(_require(header, "per_clip_size_bytes"), "per_clip_size_bytes")

    expected_tail = frame_data_offset + num_frames * frame_record_size_bytes
    if per_clip_offset != expected_tail:
        msg = (
            f"ACT2 header is self-inconsistent: per_clip_offset {per_clip_offset} != frame_data_offset "
            f"{frame_data_offset} + num_frames {num_frames} * frame_record_size_bytes "
            f"{frame_record_size_bytes} = {expected_tail}"
        )
        raise ValueError(msg)

    _validate_field_offsets(
        _require(header, "per_frame_fields"),
        itemsize=dtype.itemsize,
        region_size_bytes=frame_record_size_bytes,
        field_kind="per-frame",
        region_label="record",
    )
    _validate_field_offsets(
        _require(header, "per_clip_fields"),
        itemsize=dtype.itemsize,
        region_size_bytes=per_clip_size_bytes,
        field_kind="per-clip",
        region_label="tail",
    )

    required_len = per_clip_offset + per_clip_size_bytes
    if len(data) < required_len:
        msg = f"ACT2 payload truncated: {len(data)} bytes < expected {required_len}"
        raise ValueError(msg)


def read_action_bin_header(data: bytes) -> dict[str, Any]:
    """Return the parsed ACT2 header without decoding the arrays.

    Cheap way to learn which layout family an artifact uses (``spec_name``) and
    where it came from (``source_dataset``) without materializing every field.

    Args:
        data: The full ACT2 payload bytes.

    Returns:
        The parsed header mapping.

    Raises:
        ValueError: If the payload is not a valid ACT2 buffer.

    """
    return _parse_header(data)


@attrs.frozen(eq=False)  # numpy-array dict field: a generated __eq__/__hash__ would raise on it
class ActionArtifact:
    """One decoded ACT2 payload: its identity plus its named arrays.

    Returned by :func:`decode_action_artifact` so a caller needing both the arrays
    *and* a header field obtains them from a single header parse.

    Attributes:
        spec_name: The ACT2 layout family (e.g. ``"mecka"``); self-describing, so
            it is the authoritative record of the layout that produced it.
        source_dataset: The dataset the artifact was exported from.
        num_frames: Per-frame record count.
        arrays: Field name -> array. Per-frame fields have shape
            ``(num_frames, *field_shape)``; per-clip fields have shape
            ``field_shape``. Arrays are read-only views over ``data``.

    """

    spec_name: str
    source_dataset: str
    num_frames: int
    arrays: dict[str, np.ndarray[Any, Any]]


def decode_action_artifact(data: bytes) -> ActionArtifact:
    """Decode an ACT2 payload into an :class:`ActionArtifact` (identity + arrays).

    The inverse of :func:`encode_action_bin`, plus the header identity fields.
    Reconstruction is driven purely by the self-describing header, so an artifact
    stays decodable even if the spec registry is later edited or a dataset
    renamed.

    Returns:
        The decoded artifact. Its ``arrays`` are read-only views over ``data``.

    Raises:
        ValueError: If the payload is not a valid ACT2 buffer.

    """
    header = _parse_header(data)
    dtype = _dtype_from_header(header)
    num_frames = int(_require(header, "num_frames"))
    frame_record_size_bytes = int(_require(header, "frame_record_size_bytes"))
    values_per_record = frame_record_size_bytes // dtype.itemsize

    body = np.frombuffer(
        data,
        dtype=dtype,
        count=num_frames * values_per_record,
        offset=int(_require(header, "frame_data_offset")),
    ).reshape(num_frames, values_per_record)

    out: dict[str, np.ndarray[Any, Any]] = {}
    for field in _require(header, "per_frame_fields"):
        shape = tuple(_require(field, "shape"))
        start = int(_require(field, "byte_offset")) // dtype.itemsize
        width = int(prod(shape))
        out[_require(field, "name")] = body[:, start : start + width].reshape(num_frames, *shape)

    tail_offset = int(_require(header, "per_clip_offset"))
    for field in _require(header, "per_clip_fields"):
        shape = tuple(_require(field, "shape"))
        width = int(prod(shape))
        start = tail_offset + int(_require(field, "byte_offset"))
        out[_require(field, "name")] = np.frombuffer(data, dtype=dtype, count=width, offset=start).reshape(*shape)

    return ActionArtifact(
        spec_name=str(_require(header, "spec_name")),
        source_dataset=str(_require(header, "source_dataset")),
        num_frames=num_frames,
        arrays=out,
    )


def decode_action_bin(data: bytes) -> dict[str, np.ndarray[Any, Any]]:
    """Decode an ACT2 payload into its named per-frame and per-clip arrays.

    Thin wrapper over :func:`decode_action_artifact` for callers that need only
    the arrays. See that function for the field-shape and read-only-view contract.

    Raises:
        ValueError: If the payload is not a valid ACT2 buffer.

    """
    return decode_action_artifact(data).arrays
