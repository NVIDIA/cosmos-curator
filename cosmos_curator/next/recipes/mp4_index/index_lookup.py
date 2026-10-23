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

"""Read video frame indexes from a ``manifest.lance`` written by this recipe.

The reader sits beside :mod:`.lance_sink`, which writes the rows, and hands back the
sensor library's own :class:`Mp4FrameIndex` and :class:`Mp4Metadata`. Interpreting those
values is the library's job; unpacking the stored representation is this module's.
Consumers choose which timestamps to use.

Source videos and manifests must remain immutable during a run. Size checks catch
changed-length replacements, not equal-size replacements.

``FRAME_INDEX_SCHEMA`` has no ``is_discard`` column, so the index includes samples
that a decoder may discard. This difference cannot be detected from the index alone.
"""

from collections.abc import Iterable
from typing import Literal, Self, overload

import lance  # type: ignore[import-untyped]
import pyarrow as pa
from loguru import logger

from cosmos_curator.core.sensors.data.mp4 import Mp4FrameIndex, Mp4Metadata
from cosmos_curator.core.utils.storage.s3_client import S3Client, S3Prefix
from cosmos_curator.core.utils.storage.storage_utils import (
    get_lance_storage_options,
    get_storage_client,
    is_remote_path,
)
from cosmos_curator.core.utils.storage_cli import get_lance_storage_options as get_cli_lance_storage_options
from cosmos_curator.next.recipes.mp4_index.contracts import RECORD_SCHEMA_VERSION, read_video_record
from cosmos_curator.next.utils.lance_fragment_recovery import sql_string_literal
from cosmos_curator.next.utils.lance_utils import open_dataset
from cosmos_curator.next.utils.local_uri import local_path
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
    against the manifest's BTREE index on ``media_location``. Optional lookups return
    ``None`` for unusable records; required lookups raise.
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
        storage_profile: str | None = None,
    ) -> Self | None:
        """Open *manifest_uri*, or return ``None`` when there is no dataset there.

        Use :meth:`open_required` when the manifest must exist.
        ``storage_profile`` selects recipe profiles; otherwise use CLI credentials.

        Raises:
            ValueError: if the manifest's rows predate the current
                ``RECORD_SCHEMA_VERSION``.

        """
        storage_options = (
            get_lance_storage_options(manifest_uri, profile_name=storage_profile)
            if storage_profile is not None
            else get_cli_lance_storage_options(manifest_uri, s3_profile_name=s3_profile_name, endpoint_url=endpoint_url)
        )
        dataset = open_dataset(manifest_uri, storage_options=storage_options)
        if dataset is None:
            return None
        _reject_stale_schema(dataset, manifest_uri)
        return cls(dataset)

    @classmethod
    def open_required(
        cls,
        manifest_uri: str,
        *,
        storage_profile: str = "default",
    ) -> Self:
        """Open a required manifest using recipe profiles, raising FileNotFoundError if absent."""
        lookup = cls.open(normalize_location(manifest_uri), storage_profile=storage_profile)
        if lookup is None:
            msg = f"Configured mp4-index manifest does not exist: {manifest_uri}"
            raise FileNotFoundError(msg)
        return lookup

    def require_index_for(self, source: str, *, storage_profile: str) -> tuple[Mp4FrameIndex, Mp4Metadata]:
        """Read the source's current size and require a matching indexed record."""
        location = normalize_location(source)
        if is_remote_path(location):
            client = get_storage_client(location, profile_name=storage_profile)
            if not isinstance(client, S3Client):
                msg = f"Cannot check the indexed source size for {source}"
                raise ValueError(msg)
            media_size = client.stat(S3Prefix(location)).size_bytes
        else:
            media_size = local_path(location).stat().st_size
        return self.index_for(location, media_size_bytes=media_size, required=True)

    @overload
    def index_for(
        self, source: str, *, media_size_bytes: int | None, required: Literal[True]
    ) -> tuple[Mp4FrameIndex, Mp4Metadata]: ...

    @overload
    def index_for(
        self, source: str, *, media_size_bytes: int | None, required: bool = False
    ) -> tuple[Mp4FrameIndex, Mp4Metadata] | None: ...

    def index_for(
        self, source: str, *, media_size_bytes: int | None, required: bool = False
    ) -> tuple[Mp4FrameIndex, Mp4Metadata] | None:
        """Return the frame index for *source*, or ``None`` if the manifest has no usable row.

        Args:
            source: the video's location, however the caller spelled it.
            media_size_bytes: the object's current size, or ``None`` if it could not be
                read. Compared against the indexed size, because this recipe keys its
                rows on location alone, so a replaced video keeps its old row.
            required: Raise on missing, ambiguous, stale, or invalid records rather
                than returning None. For consumers that require indexed input.

        Returns:
            The video's frame index and box metadata, or ``None`` if there is no row for
            *source* or its size does not match.

        """
        key = _manifest_key(source)
        if key is None:
            if required:
                msg = f"Cannot look up indexed source {source}"
                raise ValueError(msg)
            return None
        # Optional consumers can read the media when the stored row is unusable.
        # Required consumers propagate the original error.
        try:
            table = self._dataset.to_table(columns=list(_PROJECTION), filter=_media_location_filter([key]))
            if table.num_rows != 1:
                if required:
                    msg = f"mp4-index must contain exactly one record for {source}; found {table.num_rows}"
                    raise ValueError(msg)  # noqa: TRY301 -- strict errors propagate; optional reads fall back
                return None
            if not self._describes_current_bytes(source, table, media_size_bytes):
                if required:
                    msg = f"mp4-index size is stale or unverifiable for {source}; re-run mp4-index"
                    raise ValueError(msg)  # noqa: TRY301 -- strict errors propagate; optional reads fall back
                return None
            return _unwrap(table)
        except (ValueError, OSError):
            if required:
                raise
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

    The sensor library computes media and edited PTS; unsupported edits raise
    during frame-index construction.
    """
    frame_table, metadata_row = read_video_record(table)
    metadata = Mp4Metadata.from_row(metadata_row)
    index = Mp4FrameIndex.from_frame_index_table(
        frame_table,
        time_base_num=metadata.time_base_num,
        time_base_den=metadata.time_base_den,
        elst_bytes=metadata.elst_bytes,
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
