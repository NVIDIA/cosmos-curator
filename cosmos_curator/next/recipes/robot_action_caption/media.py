# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fetch encoded clips or name why an individual row has no media.

Base64 data URLs keep decoded frames out of Ray's object store. Deterministic
media faults remain row-local, while systemic and attempt-exhausted transport
faults fail the attempt. This stage never authors caption verdicts.

See ``docs/curator/design/curator-next-robot-action-caption.md`` for the
transport flow, schema ownership, and error taxonomy.
"""

import statistics
import time
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

# Batches one actor completes between fetch-timing summaries. Per-batch logging
# would emit tens of thousands of lines on a full shard, and per-row far more,
# while the question this answers - what rate the store sustained and how much of
# the stage was transport rather than CPU - is only meaningful over many reads.
_FETCH_REPORT_EVERY_BATCHES: Final = 100

# Successful reads a summary needs before it quotes a distribution.
# ``statistics.quantiles`` interpolates between points and so refuses fewer than
# two; an interval whose every read failed leaves none at all, which
# ``statistics.median`` refuses in turn.
_MIN_LATENCY_SAMPLES: Final = 2

# Deciles, of which the ninth and last cut point is the 90th percentile.
# ``statistics.quantiles`` returns ``n - 1`` cut points, so the index is two
# below the bucket count and moves with it.
_PERCENTILE_BUCKETS: Final = 10
_P90_CUT_POINT: Final = _PERCENTILE_BUCKETS - 2


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
        # Per-actor counters for the three quantities the reader cannot know: it
        # sees individual reads, not the batches they arrived in nor the wall
        # time a batch spent overlapping them. What each read COST is asked of
        # the reader instead of re-timed here, and must stay that way: ``read``
        # retries internally, so a clock wrapped around the call charges every
        # failed attempt and every backoff interval to "transport" and reports a
        # degraded store as a slow one.
        #
        # Nothing else in the phase measures the store, so without this summary a
        # fetch-bound run and a fetch-idle one look identical in the driver log:
        # the stage's own queue depth is set by downstream backpressure, not by
        # how fast the transport ran.
        self._fetch_batches = 0
        self._fetch_rows = 0
        self._wall_seconds = 0.0

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
        started = time.monotonic()
        rows = self._fetched_batch(scanned)
        self._record_batch(len(scanned), time.monotonic() - started)
        return pa.Table.from_pylist(rows, schema=FETCHED_SCHEMA)

    def _record_batch(self, rows: int, wall_s: float) -> None:
        """Accumulate one batch's fetch timing, reporting periodically."""
        self._fetch_batches += 1
        self._fetch_rows += rows
        self._wall_seconds += wall_s
        if self._fetch_batches % _FETCH_REPORT_EVERY_BATCHES:
            return
        # ONE snapshot for the whole line. The reader's totals are cumulative
        # since this actor built it, which is the same span the batch, row and
        # wall counters above cover, so every figure here reads "since this actor
        # started" and none is diffed against an earlier reading. Taking a second
        # snapshot for the distribution would let two halves of one sentence
        # describe two instants, and the attempt count would then be compared
        # against a delivered-read count it does not belong to.
        #
        # The snapshot's per-read sequence grows for this actor's whole life,
        # and this stage is where that is bounded, because only here is the
        # rate that fills it known: the stage feeds a GPU-bound engine in one
        # plan under backpressure, so the pool reads at the engine's row rate
        # and a node added to the run adds a media actor and engine throughput
        # together. Reads per actor per second is therefore invariant to corpus
        # size, shard count and cluster size, leaving run duration as the only
        # axis - and the allocation's time limit bounds that. Measured figures
        # and the actor lifetime they assume are in the design document's
        # fetch-timing note; re-measure them if this stage ever stops feeding a
        # slower consumer, which is what fixes the rate.
        stats = self._reader.transport_stats()
        # Derived, never counted separately: the reader appends exactly one
        # duration per delivered read, so a counter kept here could only ever
        # disagree with the sequence the latencies are summarised from.
        delivered_reads = len(stats.transport_durations)
        # Wall time can exceed the summed read time when reads overlap on the
        # pool, and fall below it for the same reason, so the two are reported
        # separately rather than differenced into a "CPU" figure this cannot see.
        megabytes = stats.delivered_bytes / (1024 * 1024)
        transport_seconds = sum(stats.transport_durations)
        logger.info(
            f"Media fetch after {self._fetch_batches} batch(es): {self._fetch_rows} row(s), "
            f"{megabytes:,.0f} MiB read in {transport_seconds:,.1f} transport-second(s) across "
            f"{self._wall_seconds:,.1f} batch-wall-second(s) at width {self._fetch_workers}. "
            f"{megabytes / self._wall_seconds if self._wall_seconds else 0.0:,.1f} MiB/s per actor; "
            f"{stats.attempts} attempt(s) delivered {delivered_reads} read(s) "
            f"with {stats.undelivered_seconds:,.1f} undelivered-second(s); "
            f"per-clip read {self._latency_summary(latencies=stats.transport_durations)}. "
            f"Transport seconds over wall seconds is the overlap this actor achieved, not a duty cycle. "
            f"Attempts above delivered reads are retried transport faults and refused objects, and their "
            f"seconds are the undelivered term rather than the transport rate."
        )

    def _latency_summary(self, *, latencies: tuple[float, ...]) -> str:
        """Return the per-read latency distribution, or say there is not one yet.

        Args:
            latencies: One entry per read that DELIVERED bytes, as the reader
                recorded it. Reads that delivered none contribute no entry, so
                the count rendered here is the population the quantiles describe
                rather than the number of reads attempted.

        """
        if len(latencies) < _MIN_LATENCY_SAMPLES:
            # Named as absent rather than rendered as zeros. A p50 of 0.00s reads
            # as an instantaneous store, which is the opposite of what too few
            # samples means, and this branch is reachable on a real run: an actor
            # whose every read so far failed, or that has been handed only rows
            # naming no location, carries no transport measurement at all.
            return f"not summarised over {len(latencies)} successful read(s)"
        # The stdlib sorts internally, so the sample is kept in arrival order and
        # no caller owes it a pre-sort.
        #
        # ``method="inclusive"`` rather than the default: these latencies ARE the
        # whole population of this actor's delivered reads, not a sample drawn
        # from a wider distribution, and the default "exclusive" method
        # extrapolates past the observed range on a small one. Measured on two
        # points at 2.0s and 8.0s it reports a 90th percentile of 12.20s, which
        # would print beside a max of 8.00s and leave the line contradicting
        # itself. Inclusive never leaves the observed range.
        percentiles = statistics.quantiles(latencies, n=_PERCENTILE_BUCKETS, method="inclusive")
        return (
            f"p50={statistics.median(latencies):.2f}s "
            f"p90={percentiles[_P90_CUT_POINT]:.2f}s "
            f"max={max(latencies):.2f}s over {len(latencies)} successful read(s)"
        )

    def _fetched_batch(self, scanned: list[dict[str, object]]) -> list[dict[str, object]]:
        """Fetch every row, overlapping the transfers and preserving input order."""
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
        # that a fail-fast fan-out normally wants. Cancelling can only drop
        # work that has not started, and nothing here ever waits: the pool is
        # sized to the batch just above, and Ray's contract for `map_batches`
        # bounds a batch at `batch_size` from above, so one thread is running
        # per row from the first submit. With an empty queue by construction,
        # that form would be this one with extra words.
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="clip-fetch") as pool:
            submitted = [pool.submit(self._fetched, row) for row in scanned]
            # Awaited in submission order, so a propagating exception belongs
            # to the earliest failing ROW rather than to whichever thread
            # lost the race. That makes a leg-fatal fault reproducible across
            # attempts instead of naming a different clip each run.
            return [future.result() for future in submitted]

    def _fetched(self, row: dict[str, object]) -> dict[str, object]:
        """Return one scanned row widened with its media URL or its fault.

        Runs on a pool thread and accumulates nothing here: what each read cost
        is charged inside the reader, under the lock that makes concurrent reads
        safe to measure.
        """
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
