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

"""Serve a stream's timeline out of an ``mp4-index`` manifest instead of the video itself.

Two differences from :class:`CameraSensor` are visible in the measurements:

* The timeline is ``dts_ns``, in decode order, where ``CameraSensor`` serves sorted
  presentation timestamps. The two agree when the file has no ``ctts`` box. When
  per-frame composition offsets vary, the measured cadence is the decoder's schedule,
  not the display's. Decode timestamps always increase, so the ordering metric cannot
  report a problem against them, and a PASS there says less than one taken from the
  media. Readers tell the two apart by :attr:`TimestampSource.CONTAINER_DTS`.
* ``FRAME_INDEX_SCHEMA`` has no ``is_discard`` column, so every sample counts as
  displayable where ``CameraSensor`` reports only ``display_pts_ns``. A stream with
  discarded packets reports more samples here, and that is not detectable from the
  index alone.
"""

from collections.abc import Iterable, Iterator
from fractions import Fraction

import lance  # type: ignore[import-untyped]
import numpy as np
import numpy.typing as npt
import pyarrow as pa
from loguru import logger

from cosmos_curator.core.sensors.data.mp4 import Mp4FrameIndex, Mp4Metadata
from cosmos_curator.core.sensors.types.types import TimestampSource
from cosmos_curator.core.utils.storage_cli import get_lance_storage_options
from cosmos_curator.next.recipes.mp4_index.contracts import RECORD_SCHEMA_VERSION, read_video_record
from cosmos_curator.next.utils.lance_fragment_recovery import sql_string_literal
from cosmos_curator.next.utils.lance_utils import open_dataset
from cosmos_curator.next.utils.uris import normalize_location

#: The columns one lookup projects. ``media_size_bytes`` is compared against the
#: object's current size before the row is used (see :meth:`Mp4IndexLookup.sensor_for`).
_PROJECTION = ("media_location", "media_size_bytes", "metadata", "frames")

#: Schemes :func:`normalize_location` can spell. An ``az://`` source is never in a
#: manifest, and normalizing one resolves it as a relative local path, which would
#: then be looked up as if it were a real key.
_INDEXABLE_SCHEMES = ("s3://",)


class IndexRecordUnusableError(ValueError):
    """A manifest row exists but cannot be used for this stream, so the caller opens the media.

    Raised when the row has no frames, or when its frame count disagrees with the sensor
    timeline beside the video.
    """


class Mp4IndexSensor:
    """A stream's timeline and container facts, read from one ``mp4-index`` manifest row.

    Implements ``IntegritySensor`` and nothing more. It cannot decode frames; use
    :class:`~cosmos_curator.core.sensors.sensors.camera_sensor.CameraSensor` for that.
    """

    def __init__(
        self,
        index: Mp4FrameIndex,
        metadata: Mp4Metadata,
        *,
        sensor_timestamps_ns: npt.NDArray[np.int64] | None = None,
    ) -> None:
        """Build the sensor from one video's frame index and box metadata.

        Args:
            index: the video's frame index. Only ``dts_ns`` and the composition offsets
                are read.
            metadata: the video's box metadata, for the scalar facts the engine reports.
            sensor_timestamps_ns: the recording's own clock, one entry per frame, or
                ``None`` to report the container's decode timestamps. When given it is
                the reported timeline and the index contributes only the scalar facts.

        Raises:
            IndexRecordUnusableError: if the index has no frames, or if
                *sensor_timestamps_ns* describes a different number of frames.

        """
        if len(index) == 0:
            msg = "mp4-index record contains no frames"
            raise IndexRecordUnusableError(msg)
        self._codec_name = metadata.codec_name
        self._avg_frame_rate = _avg_frame_rate(metadata)
        # An all-null column means the ctts box is absent; an all-zero one means it is
        # present and says there is no reordering.
        offsets = index.composition_offset_ticks
        self._has_bframes = offsets is not None and bool(np.any(offsets != 0))
        if sensor_timestamps_ns is None:
            self._timeline_ns = index.dts_ns
            self._timestamp_source = TimestampSource.CONTAINER_DTS
        else:
            self._timeline_ns = _require_matching_length(sensor_timestamps_ns, len(index))
            self._timestamp_source = TimestampSource.SENSOR

    @property
    def codec_name(self) -> str:
        """Video codec name as the container's own boxes spell it."""
        return self._codec_name

    @property
    def has_bframes(self) -> bool:
        """Whether the container carries composition offsets, i.e. signals frame reordering."""
        return self._has_bframes

    @property
    def avg_frame_rate(self) -> Fraction:
        """The container's nominal frame rate, or ``Fraction(0)`` when it declares none."""
        return self._avg_frame_rate

    @property
    def timestamp_source(self) -> TimestampSource:
        """Which clock :attr:`timestamps_ns` is on."""
        return self._timestamp_source

    @property
    def start_ns(self) -> int:
        """First timestamp in nanoseconds."""
        return int(self._timeline_ns[0])

    @property
    def end_ns(self) -> int:
        """Last timestamp in nanoseconds."""
        return int(self._timeline_ns[-1])

    @property
    def timestamps_ns(self) -> npt.NDArray[np.int64]:
        """The whole timeline in ``int64`` nanoseconds."""
        return self._timeline_ns

    def stream_timestamps(self, batch_size: int = 0) -> Iterator[npt.NDArray[np.int64]]:
        """Yield the timeline in ``int64`` ns batches.

        Args:
            batch_size: ``0`` yields one batch spanning the timeline; ``> 0`` yields
                consecutive non-overlapping batches of that length, the last possibly
                shorter.

        Yields:
            Timestamps in nanoseconds as ``int64`` arrays.

        Raises:
            ValueError: if *batch_size* is negative.

        """
        if batch_size < 0:
            msg = f"batch_size must be non-negative, got {batch_size}"
            raise ValueError(msg)
        timestamps = self._timeline_ns
        # The constructor rejects an empty index, so step is always >= 1.
        step = batch_size or len(timestamps)
        return (timestamps[start : start + step] for start in range(0, len(timestamps), step))


def _avg_frame_rate(metadata: Mp4Metadata) -> Fraction:
    """Rebuild the nominal frame rate, or ``Fraction(0)`` when the container declares none."""
    if metadata.avg_frame_rate_den == 0:
        return Fraction(0)
    return Fraction(metadata.avg_frame_rate_num, metadata.avg_frame_rate_den)


def _require_matching_length(sensor_timestamps_ns: npt.NDArray[np.int64], frames: int) -> npt.NDArray[np.int64]:
    """Return *sensor_timestamps_ns* unchanged, or reject the record if it has a different count.

    The counts can differ legitimately: the index counts every sample, while a sensor
    timeline has one entry per *displayable* frame. Raising
    :class:`IndexRecordUnusableError` falls back to opening the video.
    """
    if len(sensor_timestamps_ns) != frames:
        msg = (
            f"sensor timeline describes {len(sensor_timestamps_ns)} observations "
            f"but the mp4-index record has {frames} frames"
        )
        raise IndexRecordUnusableError(msg)
    return sensor_timestamps_ns


def _manifest_key(source: str) -> str | None:
    """Normalize *source* into the ``media_location`` value ``mp4-index`` would have stored.

    Returns ``None`` if *source* is not a location the manifest could hold. Uses
    ``mp4-index``'s own normalizer so both spell a location the same way.
    """
    if not source.startswith(_INDEXABLE_SCHEMES) and "://" in source:
        return None
    try:
        return normalize_location(source)
    except ValueError:
        return None


class Mp4IndexLookup:
    """An open ``manifest.lance``, queried for one video's timeline.

    Opened once and held across a session's streams. Each lookup is a point query
    against the manifest's BTREE index on ``media_location``. Misses are expected, since
    a dataset is often only partly indexed, so any failure to find a usable row returns
    ``None`` and the caller opens the media.
    """

    def __init__(self, dataset: lance.LanceDataset) -> None:
        """Wrap an already-open manifest dataset."""
        self._dataset = dataset
        # Warn once per process rather than once per stream: if one metadata lookup
        # fails, they all will.
        self._warned_unverifiable = False

    @classmethod
    def open(
        cls,
        manifest_uri: str,
        *,
        s3_profile_name: str | None = None,
        endpoint_url: str | None = None,
    ) -> "Mp4IndexLookup | None":
        """Open *manifest_uri*, or return ``None`` when there is no dataset there.

        A missing manifest is not an error. The index is optional, so a run pointed at
        one that has not been built yet opens the videos instead.

        Raises:
            ValueError: if the manifest's rows predate the current
                ``RECORD_SCHEMA_VERSION``.

        """
        storage_options = get_lance_storage_options(
            manifest_uri, s3_profile_name=s3_profile_name, endpoint_url=endpoint_url
        )
        dataset = open_dataset(manifest_uri, storage_options=storage_options)
        if dataset is None:
            return None
        _reject_stale_schema(dataset, manifest_uri)
        return cls(dataset)

    def sensor_for(
        self,
        source: str,
        *,
        media_size_bytes: int | None,
        sensor_timestamps_ns: npt.NDArray[np.int64] | None = None,
    ) -> Mp4IndexSensor | None:
        """Return a sensor for *source*, or ``None`` if the manifest has no usable row for it.

        Args:
            source: the video's location, however the caller spelled it.
            media_size_bytes: the object's current size, or ``None`` if it could not be
                read. Compared against the indexed size, because ``mp4-index`` keys its
                rows on location alone, so a replaced video keeps its old row.
            sensor_timestamps_ns: the recording's own clock, if one sits beside the
                video; see :class:`Mp4IndexSensor`.

        Returns:
            A sensor reading the indexed timeline, or ``None`` if there is no row for
            *source*, its size does not match, or the row cannot be used for this stream.

        """
        key = _manifest_key(source)
        if key is None:
            return None
        # One try block for all three steps: whatever goes wrong here, the answer is to
        # open the video. Letting a malformed row or a transport error propagate would
        # record the stream as one whose video could not be read.
        try:
            table = self._dataset.to_table(columns=list(_PROJECTION), filter=_media_location_filter([key]))
            if table.num_rows == 0:
                return None
            if not self._describes_current_bytes(source, table, media_size_bytes):
                return None
            index, metadata = _unwrap(table)
            return Mp4IndexSensor(index, metadata, sensor_timestamps_ns=sensor_timestamps_ns)
        except IndexRecordUnusableError:
            # Expected, and harmless, so log at debug only.
            logger.opt(exception=True).debug("mp4-index row for {} cannot answer; opening the video", source)
            return None
        except (ValueError, OSError):
            # Unexpected: if the manifest fails this way for every row, the run is slow
            # and nothing explains why. Warn so there is something to read.
            logger.opt(exception=True).warning("mp4-index row for {} unusable; opening the video", source)
            return None

    def _describes_current_bytes(self, source: str, table: pa.Table, media_size_bytes: int | None) -> bool:
        """Whether the indexed row still matches the object now at *source*.

        Warns and returns ``False`` when the indexed size differs from the current one, and
        when the current size could not be read. The caller opens the video in both cases.
        """
        if media_size_bytes is None:
            if not self._warned_unverifiable:
                self._warned_unverifiable = True
                logger.warning(
                    "cannot read the size of {} to check the mp4-index against it; opening videos "
                    "directly for the rest of this run",
                    source,
                )
            return False
        indexed_size_bytes = table.column("media_size_bytes")[0].as_py()
        if indexed_size_bytes != media_size_bytes:
            logger.warning(
                "mp4-index describes {} as {} bytes but it is {} now; the index is stale for this "
                "video, opening it directly (re-run mp4-index to refresh it)",
                source,
                indexed_size_bytes,
                media_size_bytes,
            )
            return False
        return True


def _unwrap(table: pa.Table) -> tuple[Mp4FrameIndex, Mp4Metadata]:
    """Convert one manifest row into the sensor library's ``Mp4FrameIndex`` and ``Mp4Metadata``.

    Does not apply the edit list. This sensor reports ``dts_ns``, which an ``elst`` never
    shifts, so an unsupported edit list must not fail the check.
    """
    frame_table, metadata_row = read_video_record(table)
    metadata = Mp4Metadata.from_row(metadata_row)
    index = Mp4FrameIndex.from_frame_index_table(
        frame_table,
        time_base_num=metadata.time_base_num,
        time_base_den=metadata.time_base_den,
        elst_bytes=None,
    )
    return index, metadata


def _reject_stale_schema(dataset: lance.LanceDataset, manifest_uri: str) -> None:
    """Raise unless every row is at the record schema version this reader understands.

    Checked once on open. This fails the run rather than falling back, because a
    manifest built by an older ``mp4-index`` means the run is misconfigured.

    Raises:
        ValueError: if any row is at a different ``record_schema_version``.

    """
    stale = dataset.count_rows(filter=f"record_schema_version != {RECORD_SCHEMA_VERSION}")
    if stale:
        msg = (
            f"{manifest_uri} has {stale} row(s) at a record_schema_version other than "
            f"{RECORD_SCHEMA_VERSION}; re-run mp4-index to rebuild it"
        )
        raise ValueError(msg)


def _media_location_filter(keys: Iterable[str]) -> str:
    """Build a ``media_location IN (...)`` predicate over the manifest's BTREE-indexed column."""
    literals = ",".join(sql_string_literal(key) for key in keys)
    return f"media_location IN ({literals})"
