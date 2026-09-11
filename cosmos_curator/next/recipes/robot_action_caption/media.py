# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fetch encoded clips or name why an individual row has no media.

Base64 data URLs keep decoded frames out of Ray's object store. Deterministic
media faults remain row-local, while systemic and attempt-exhausted transport
faults fail the attempt. This stage never authors caption verdicts.

See ``docs/curator/design/curator-next-robot-action-caption.md`` for the
transport flow, schema ownership, and error taxonomy.
"""

from concurrent.futures import ThreadPoolExecutor
from typing import Final

import pyarrow as pa
import pybase64
from loguru import logger

from cosmos_curator.next.media.ffmpeg import MediaError, probe_video_bytes
from cosmos_curator.next.recipes.robot_action_caption.contracts import MEDIA_TYPE
from cosmos_curator.next.recipes.robot_action_caption.datasource import SCAN_SCHEMA
from cosmos_curator.next.recipes.robot_action_caption.table import CLIP_ID_COLUMN, CLIP_URI_COLUMN
from cosmos_curator.next.utils import media_reader

MEDIA_URL_COLUMN: Final = "video_data_url"
MEDIA_ERROR_COLUMN: Final = "media_error"

# What the media stage hands the engine boundary: the scan, plus exactly one of a
# media URL and a fault message. Both columns travel on every row so ONE schema
# leaves this stage; ``request.build_request`` decides which one is live.
FETCHED_SCHEMA: Final = pa.schema(
    [
        *SCAN_SCHEMA,
        pa.field(MEDIA_URL_COLUMN, pa.large_string(), nullable=True),
        pa.field(MEDIA_ERROR_COLUMN, pa.large_string(), nullable=True),
    ]
)

# Built from the digested constant rather than spelled out, so the type the
# engine is handed and the type the contract digest stamps cannot drift apart.
# Why the type is fixed at all lives with the constant, in ``contracts``.
_DATA_URL_PREFIX: Final = f"data:{MEDIA_TYPE};base64,"

_NULL_URI_ERROR: Final = "clip_uri is null: the source row names no media location"


class ClipMediaFetch:
    """Fetch one batch of clips and emit one uniform schema whatever the outcome.

    An actor rather than a task so the storage client is resolved once per worker
    and reused across every batch that worker handles. Within a batch the
    transfers overlap on a thread pool ``fetch_workers`` wide, so an actor holds
    that many clips in flight and the caller must reserve the matching share of
    the node's logical IO budget.

    ::

        warm every backend      [calling thread, once per backend]
                |
                v
        fetch rows in parallel  [fetch_workers threads]
                |
                v
        results in submission order --> one table in FETCHED_SCHEMA
    """

    def __init__(self, *, storage_profile: str, attempts: int, max_clip_bytes: int, fetch_workers: int) -> None:
        """Bind the run's transport and media admission policy to this actor.

        Args:
            storage_profile: Credential profile for the clip reads.
            attempts: Transport read attempts per clip before the attempt fails.
                Owned by ``execution.media_attempts``, so an operator facing a
                degraded store can raise it without a code change.
            max_clip_bytes: Per-object byte cap; a larger clip becomes a fault
                rather than a caption. Digested by the caller, so the cap that
                refuses a clip is the same one the published row is stamped with.
            fetch_workers: Clips read at once. The caller reserves one logical IO
                slot per worker, so this actor must never exceed it.

        Raises:
            ValueError: If ``fetch_workers`` is below one.

        """
        if fetch_workers < 1:
            msg = f"fetch_workers must be at least 1, got {fetch_workers}"
            raise ValueError(msg)
        self._reader = media_reader.MediaObjectReader(storage_profile=storage_profile, max_attempts=attempts)
        self._max_clip_bytes = max_clip_bytes
        self._fetch_workers = fetch_workers

    def __call__(self, batch: pa.Table) -> pa.Table:
        """Fetch one batch of clips, or record why each has no bytes.

        Args:
            batch: Rows in :data:`SCAN_SCHEMA`.

        Returns:
            The same rows in :data:`FETCHED_SCHEMA`.

        Raises:
            media_reader.MediaSystemicError: If the read path itself is unusable.
            media_reader.MediaTransportError: If a clip exhausted its attempt
                budget. Both are propagated deliberately: every clip would
                plausibly fail the same way, so one row per clip would bury a
                degraded read path in data. The checkpoint makes that affordable
                - a later attempt keeps the captions this one finished.

        """
        scanned = batch.cast(SCAN_SCHEMA).to_pylist()
        # Resolve every backend this batch touches from THIS thread before any
        # worker starts. An unresolvable profile is wrong for every row, so
        # warming makes that fault deterministic and keeps client construction
        # off the worker threads. The reader's guarded cache miss provides the
        # thread-safety independently; after the first URI per backend this is a
        # dict hit.
        for row in scanned:
            clip_uri = row[CLIP_URI_COLUMN]
            if clip_uri is not None:
                self._reader.warm_client(str(clip_uri))
        rows = self._fetched_batch(scanned)
        return pa.Table.from_pylist(rows, schema=FETCHED_SCHEMA)

    def _fetched_batch(self, scanned: list[dict[str, object]]) -> list[dict[str, object]]:
        """Fetch every scanned row, overlapping the transfers, preserving order."""
        workers = min(self._fetch_workers, len(scanned))
        if workers <= 1:
            # No pool for a single row: one thread hand-off would cost more than
            # the overlap it cannot provide.
            return [self._fetched(row) for row in scanned]
        # Threads, not processes: the transfer blocks in the storage client's
        # socket and releases the GIL for the wait that dominates this stage,
        # while a process boundary would copy every clip's bytes twice for
        # nothing.
        #
        # A context manager, NOT the explicit `shutdown(cancel_futures=True)`
        # that a fail-fast fan-out normally wants. Cancelling can only drop work
        # that has not started, and nothing here ever waits: the pool is sized to
        # the batch just above, and Ray's contract for `map_batches` bounds a
        # batch at `batch_size` from above, so one thread is running per row from
        # the first submit. With an empty queue by construction, that form would
        # be this one with extra words.
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="clip-fetch") as pool:
            submitted = [pool.submit(self._fetched, row) for row in scanned]
            # Awaited in submission order, so a propagating exception belongs to
            # the earliest failing ROW rather than to whichever thread lost the
            # race. That makes a leg-fatal fault reproducible across attempts
            # instead of naming a different clip each run.
            return [future.result() for future in submitted]

    def _fetched(self, row: dict[str, object]) -> dict[str, object]:
        """Return one scanned row widened with its media URL or its fault."""
        clip_id = row[CLIP_ID_COLUMN]
        clip_uri = row[CLIP_URI_COLUMN]
        if clip_uri is None:
            # Named as its own cause rather than handed to the reader: rendering
            # a null location into a string yields "None", which the reader then
            # refuses as a missing local object, and the published
            # "media object is missing: None" would equally describe a clip
            # really named "None". Charged to the ROW, because a source table can
            # hold one such row among millions of good ones.
            logger.warning(f"Clip {clip_id!r} is not captionable: {_NULL_URI_ERROR}")
            return row | {MEDIA_URL_COLUMN: None, MEDIA_ERROR_COLUMN: _NULL_URI_ERROR}
        try:
            data = self._reader.read(str(clip_uri), max_bytes=self._max_clip_bytes)
        except media_reader.MediaObjectError as e:
            # Reader diagnostics embed the raw URI, which may carry credentials
            # or log-control characters. Keep the stable exception category and
            # clip identity, but never persist or log the exception text or URI.
            detail = f"{type(e).__name__}: clip_id={clip_id!r}"
            logger.warning(f"Clip {clip_id!r} is not captionable: {detail}")
            return row | {MEDIA_URL_COLUMN: None, MEDIA_ERROR_COLUMN: detail}
        try:
            probe_video_bytes(data)
        except MediaError as e:
            # A CPU-side probe of a clip that would otherwise reach the engine
            # intact, and the only place this fault can be charged to one row.
            #
            # The selected engine backend needs a usable container frame rate;
            # without this probe, invalid timing fails inside inference rather
            # than remaining attributable to one row. The engine stage re-raises
            # that failure, so one bad clip would end an attempt that has already
            # paid for every caption before it. See the design document's media
            # fault contract for the cross-stage constraint.
            #
            # Charged to the row for the same reason the reader's faults are: a
            # source table can hold one clip with broken timing among millions of
            # sound ones, and `probe_video_bytes` answering at all means a retry
            # would get the identical answer.
            #
            # Redacted on the same terms as the reader above: an ffprobe
            # diagnostic is media-derived text, so the stable exception category
            # and the clip identity are persisted and its message is not.
            detail = f"{type(e).__name__}: clip_id={clip_id!r}"
            logger.warning(f"Clip {clip_id!r} has no captionable media timing: {detail}")
            return row | {MEDIA_URL_COLUMN: None, MEDIA_ERROR_COLUMN: detail}
        # The encoded string exists only until preprocess consumes it, and the raw
        # bytes are not retained past this expression. Evaluating it is the one
        # point where both representations coexist - the bytes, the encoded
        # string, and the concatenation result - so a clip briefly costs several
        # times its own size here, whatever the steady-state figure at
        # ``max_clip_bytes``.
        url = _DATA_URL_PREFIX + pybase64.b64encode_as_string(data)
        return row | {MEDIA_URL_COLUMN: url, MEDIA_ERROR_COLUMN: None}
