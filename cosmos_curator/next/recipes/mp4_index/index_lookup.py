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

"""Read one video's frame index back out of a ``manifest.lance`` written by this recipe.

The reader sits beside :mod:`.lance_sink`, which writes the rows, and hands back the
sensor library's own :class:`Mp4FrameIndex` and :class:`Mp4Metadata`. Interpreting those
values is the library's job; unpacking the stored representation is this module's.

Two things a caller gets from an index that it would not get from the video:

* The timeline is ``dts_ns``, in decode order, where a sensor built over the media
  serves sorted presentation timestamps. The two agree when the file has no ``ctts``
  box. When per-frame composition offsets vary, the measured cadence is the decoder's
  schedule, not the display's. Decode timestamps always increase, so an ordering metric
  cannot report a problem against them, and a PASS there says less than one taken from
  the media. Readers tell the two apart by :attr:`TimestampSource.CONTAINER_DTS`.
* ``FRAME_INDEX_SCHEMA`` has no ``is_discard`` column, so every sample counts as
  displayable where a media-backed sensor reports only displayable frames. A stream with
  discarded packets reports more samples here, and that is not detectable from the index
  alone.
"""

from collections.abc import Iterable

import lance  # type: ignore[import-untyped]
import pyarrow as pa
from loguru import logger

from cosmos_curator.core.sensors.data.mp4 import Mp4FrameIndex, Mp4Metadata
from cosmos_curator.core.utils.storage_cli import get_lance_storage_options
from cosmos_curator.next.recipes.mp4_index.contracts import RECORD_SCHEMA_VERSION, read_video_record
from cosmos_curator.next.utils.lance_fragment_recovery import sql_string_literal
from cosmos_curator.next.utils.lance_utils import open_dataset
from cosmos_curator.next.utils.uris import normalize_location

#: The columns one lookup projects. ``media_size_bytes`` is compared against the
#: object's current size before the row is used (see :meth:`Mp4IndexLookup.index_for`).
_PROJECTION = ("media_location", "media_size_bytes", "metadata", "frames")

#: Schemes :func:`normalize_location` can spell. An ``az://`` source is never in a
#: manifest, and normalizing one resolves it as a relative local path, which would
#: then be looked up as if it were a real key.
_INDEXABLE_SCHEMES = ("s3://",)


class Mp4IndexLookup:
    """An open ``manifest.lance``, queried for one video's frame index.

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

    def index_for(self, source: str, *, media_size_bytes: int | None) -> tuple[Mp4FrameIndex, Mp4Metadata] | None:
        """Return the frame index for *source*, or ``None`` if the manifest has no usable row.

        Args:
            source: the video's location, however the caller spelled it.
            media_size_bytes: the object's current size, or ``None`` if it could not be
                read. Compared against the indexed size, because this recipe keys its
                rows on location alone, so a replaced video keeps its old row.

        Returns:
            The video's frame index and box metadata, or ``None`` if there is no row for
            *source* or its size does not match.

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
            return _unwrap(table)
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


def _manifest_key(source: str) -> str | None:
    """Normalize *source* into the ``media_location`` value this recipe would have stored.

    Returns ``None`` if *source* is not a location the manifest could hold. Uses the
    recipe's own normalizer so a writer and a reader spell a location the same way.
    """
    if not source.startswith(_INDEXABLE_SCHEMES) and "://" in source:
        return None
    try:
        return normalize_location(source)
    except ValueError:
        return None


def _unwrap(table: pa.Table) -> tuple[Mp4FrameIndex, Mp4Metadata]:
    """Convert one manifest row into the sensor library's ``Mp4FrameIndex`` and ``Mp4Metadata``.

    Does not apply the edit list. A caller reading ``dts_ns`` is reading the one timeline
    an ``elst`` never shifts, so an unsupported edit list must not fail the read.
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
