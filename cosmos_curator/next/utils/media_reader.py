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

"""Capped, retrying reads of per-row media objects, plus the fault taxonomy for them.

Reach for :mod:`cosmos_curator.next.utils.local_uri` to resolve what filesystem
path a location names, and :mod:`cosmos_curator.next.utils.storage` to move bytes
to and from one: those helpers are stateless and single-shot, and a backend
failure reaches the caller as whatever the backend raised.

Reach for this module when the bytes are a per-row media object pulled by a
pipeline worker. That read carries three obligations a plain download helper must
not take on, which is why it sits behind its own boundary:

* a size ceiling enforced from the store's metadata BEFORE the transfer, so one
  mis-populated URI cannot exhaust the worker's heap and cost every other row
  that worker was holding;
* a verdict on every failure, as one of the :class:`MediaReadError` subclasses,
  so the caller can choose between dropping this row, spending a retry, and
  failing the whole run without re-deriving each backend's error vocabulary;
* a measurement of what those reads cost, because the retries happen INSIDE one
  call: a timer wrapped around :meth:`MediaObjectReader.read` reports a clip
  that succeeded on its third attempt as one slow transfer, and cannot see the
  attempt count at all. Only the reader can separate the two, so it does, and
  :meth:`MediaObjectReader.transport_stats` hands the terms back separately.
  Nothing here logs them: a shared reader cannot know the right cadence for
  every recipe on it, so reporting stays with the caller.

Serving those obligations also makes a reader stateful and long-lived - it
retries, it caches a storage client per backend across calls, and it accumulates
what those calls cost - which the location and byte helpers next door
deliberately are not.
"""

import errno
import threading
import time
from typing import Final
from urllib.parse import urlsplit

import attrs
import tenacity
from azure.core.exceptions import ClientAuthenticationError, ResourceNotFoundError
from botocore.exceptions import ClientError, NoCredentialsError

from cosmos_curator.core.utils.storage.storage_client import StorageClient
from cosmos_curator.core.utils.storage.storage_utils import (
    get_storage_client,
    is_missing_object_error,
    is_remote_path,
    path_to_prefix,
)
from cosmos_curator.next.utils.local_uri import local_media_path

# Errno values that report the URI itself is unusable as a location, rather than
# a transfer that went wrong: too long to be a path, a path component that is
# not a directory, a shape the syscall rejects outright. Selected by errno and
# never by ``OSError``, because a transient network fault is an ``OSError`` too.
_UNUSABLE_LOCATION_ERRNOS: Final = frozenset({errno.ENAMETOOLONG, errno.ENOTDIR, errno.EINVAL})

# S3 error codes that name one key rather than the bucket or credentials. Both
# boto's modeled names and the generic HeadObject spellings are object-local.
_OBJECT_S3_ERROR_CODES: Final = frozenset({"404", "NoSuchKey", "NotFound"})

# S3 error codes that describe the whole request path rather than one key:
# rejected or expired credentials, a bucket that does not exist, a disabled
# account. Retrying any of them re-sends an identically doomed request.
_SYSTEMIC_S3_ERROR_CODES: Final = frozenset(
    {
        "AccessDenied",
        "AllAccessDisabled",
        "AuthorizationHeaderMalformed",
        "ExpiredToken",
        "InvalidAccessKeyId",
        "InvalidToken",
        "NoSuchBucket",
        "SignatureDoesNotMatch",
        "UnrecognizedClientException",
    }
)

# Azure distinguishes a missing blob from a missing container with the error
# code on ResourceNotFoundError. A code-less exception keeps the Azure client's
# established missing-object fallback; only the explicit container code is
# systemic.
_OBJECT_AZURE_ERROR_CODES: Final = frozenset({"BlobNotFound"})
_SYSTEMIC_AZURE_ERROR_CODES: Final = frozenset({"ContainerNotFound"})

# Backoff for transport faults only. The schedule is not a contract: a caller
# observes how many attempts a read got and how long the waiting between them
# cost in total, never which interval was spent where.
_TRANSPORT_RETRY_MULTIPLIER_S: Final = 0.5
_TRANSPORT_RETRY_MAX_WAIT_S: Final = 8.0


# Which subclass a failure gets is a correctness boundary, not a diagnostic
# nicety. Classify a failure as an object fault only when repeating the read
# cannot change the answer for THIS object, and as systemic only when it is
# true of every object. Draw systemic too wide and one corrupt clip fails the
# run; draw it too narrow and one rejected credential becomes a permanent error
# row per object, which fails silently because a re-run skips every row that
# already carries a verdict. A caller may also record the subclass NAME in a
# stored error column, so renaming one of these edits already-written data
# rather than refactoring code: rows saved either side of the rename describe
# one condition under two names, and neither knows about the other.
class MediaReadError(Exception):
    """Base class for a media read that produced no usable bytes.

    The three subclasses partition every read failure by how it must be handled,
    which is the only distinction a caller needs: one is deterministic for this
    object, one is worth another attempt, one is true of every object. Consumers
    map them onto their own domain vocabulary rather than re-deriving the
    classification from backend exception types.
    """


class MediaObjectError(MediaReadError):
    """Refuse one object deterministically, so a retry would fail identically."""


class MediaTransportError(MediaReadError):
    """The transfer failed for a reason another attempt may not hit."""


class MediaSystemicError(MediaReadError):
    """The read path itself is unusable, so every object would fail the same way."""


@attrs.frozen
class MediaTransportStats:
    """What one reader's reads have cost, as of the instant it was asked.

    Splits each call's wall time into the attempt that delivered bytes and
    everything else, which is the split a caller wrapping :meth:`read` cannot
    make for itself: a clip that succeeded on its third attempt spent most of
    that call waiting, and folding the waiting into the transport rate makes a
    degraded store look like a slow one.

    Attributes:
        delivered_bytes: Bytes returned by the reads that succeeded.
        transport_durations: One entry per delivered read, in completion order,
            covering that attempt alone - its metadata lookup and its transfer.
            Left unreduced so each caller picks its own summary and so several
            readers' sequences concatenate into a corpus-wide one; the cost is
            one float retained per delivered read, which only the caller's own
            read rate and reader lifetime bound.
        attempts: Attempts consumed by every read, delivered or not. A corpus
            that quietly costs three transfers per clip is visible here and
            nowhere else.
        undelivered_seconds: Time in those calls that produced no bytes: every
            attempt that raised, plus the backoff waited between them.

    """

    delivered_bytes: int = 0
    transport_durations: tuple[float, ...] = ()
    attempts: int = 0
    undelivered_seconds: float = 0.0


class _TransportMeter:
    """Accumulate one reader's transport measurements under concurrent reads.

    Mutable, which is exactly why it is not the type handed out:
    :meth:`snapshot` copies into a frozen :class:`MediaTransportStats` so a
    caller can neither mutate the totals nor watch them move underneath it.
    """

    def __init__(self) -> None:
        # Guards every accumulation below, and is NOT redundant with the GIL.
        # ``x += n`` on an attribute is a load-modify-store rather than one
        # bytecode, and the GIL serialises bytecodes individually: two threads
        # finishing together can both load the same total and one write is then
        # lost, silently and with no error anywhere. The known consumer calls
        # ``read`` from a ThreadPoolExecutor, so that is the normal case rather
        # than a rare one.
        self._lock = threading.Lock()
        self._attempts = 0
        self._delivered_bytes = 0
        self._undelivered_seconds = 0.0
        # A list, though the snapshot publishes a tuple: appending to a tuple
        # rebuilds it, which would make a reader's Nth read cost O(N).
        #
        # Deliberately uncapped. A window here would make this layer choose the
        # summary it exists to leave open, and would desynchronize the scalar
        # totals beside it from the sequence a caller sums, so two figures on
        # one line would describe different populations. A caller for whom the
        # retention is material bounds it in its own summary, where its read
        # rate and reader lifetime are both known.
        self._transport_durations: list[float] = []

    def record_failed_attempt(self) -> None:
        """Charge one attempt that ran and delivered nothing."""
        with self._lock:
            self._attempts += 1

    def record_delivered(self, *, nbytes: int, transport_seconds: float, undelivered_seconds: float) -> None:
        """Charge the one attempt that delivered ``nbytes``, plus that call's other time."""
        with self._lock:
            self._attempts += 1
            self._delivered_bytes += nbytes
            self._transport_durations.append(transport_seconds)
            self._undelivered_seconds += undelivered_seconds

    def record_undelivered(self, *, seconds: float) -> None:
        """Charge a whole call that ended with no bytes, whose attempts are already counted."""
        with self._lock:
            self._undelivered_seconds += seconds

    def snapshot(self) -> MediaTransportStats:
        """Return the totals so far as a value later reads cannot change."""
        with self._lock:
            return MediaTransportStats(
                delivered_bytes=self._delivered_bytes,
                # Copied, not aliased: handing out the live list would let a
                # caller's summary change under it mid-computation, and let a
                # later read rewrite a snapshot already taken. The copy is O(N)
                # in delivered reads and is the cheapest thing any caller does
                # with the result, since summarising the sequence sorts it.
                transport_durations=tuple(self._transport_durations),
                attempts=self._attempts,
                undelivered_seconds=self._undelivered_seconds,
            )


@attrs.frozen
class MediaObjectReader:
    """Read a stored media object into memory under a hard byte ceiling.

    Owns the transport for one storage profile: it resolves a storage client per
    backend, caches it for its own lifetime, refuses an oversized object before
    the bytes are pulled in, retries only transient transport faults, reports
    every failure as one of the three :class:`MediaReadError` subclasses, and
    measures what the reads cost (:meth:`transport_stats`).

    A reader is picklable whether or not it has already read: the client cache
    and the accumulated measurements are both excluded from the pickled state and
    restored empty. For the cache that costs nothing, because every process
    rebuilds it on demand from the profile; for the measurements it is the
    intended reading, since each worker then reports the transport IT saw rather
    than a total carrying whatever the process that shipped it had already done.

    One reader may be shared across threads. Both mutable holders are guarded -
    the client cache on its miss path, the measurements on every accumulation -
    so concurrent first use resolves each backend once rather than once per
    thread, and reads finishing together lose no samples. :meth:`warm_client` is
    therefore an optimisation and a determinism aid, not a safety prerequisite.

    Attributes:
        storage_profile: Credential profile used for remote reads.
        max_attempts: Total attempts allowed per read, including the first. Only
            :class:`MediaTransportError` consumes one; a deterministic object
            fault and an unusable read path both raise on the first attempt,
            because repeating either only multiplies the wasted transfer.

    """

    storage_profile: str = "default"
    max_attempts: int = attrs.field(default=1, validator=attrs.validators.ge(1))
    # A dict is the mutable holder a frozen class needs for a lazy cache. Excluded
    # from equality, repr, and the pickled state: it is derived state, and two
    # readers configured identically are interchangeable however many backends
    # each has touched.
    _clients: dict[str, StorageClient] = attrs.field(factory=dict, init=False, eq=False, repr=False)
    # Guards the MISS path of ``_client`` so a shared reader resolves each backend
    # once. A hit never takes it.
    _client_lock: threading.Lock = attrs.field(factory=threading.Lock, init=False, eq=False, repr=False)
    # Per-process transport measurements, excluded from equality and repr for the
    # same reason as the cache: two readers configured identically are
    # interchangeable however many bytes each has already moved.
    _meter: _TransportMeter = attrs.field(factory=_TransportMeter, init=False, eq=False, repr=False)

    # Both hooks below must stay, and neither may be deleted as redundant. attrs
    # generates a getstate/setstate PAIR for every slotted class, and declaring
    # either one suppresses BOTH generated methods, not just the one declared.
    # Keeping only __getstate__ therefore leaves no setstate at all, and pickle's
    # fallback assigns the state into __dict__, which a slotted class does not
    # have, so the unpickle itself fails.

    def __getstate__(self) -> dict[str, object]:
        """Return every configuration field, and no derived state."""
        return {
            field.name: getattr(self, field.name)
            for field in attrs.fields(type(self))
            # All three are rebuilt by ``__setstate__``. The cache and the meter
            # are process-local derived state, and neither a bare lock nor the
            # meter holding one can be pickled.
            if field.name not in {"_clients", "_client_lock", "_meter"}
        }

    def __setstate__(self, state: dict[str, object]) -> None:
        """Restore the configuration from ``state`` and rebuild the derived state."""
        # A frozen class rejects plain assignment, so the slots are written
        # through the same escape hatch attrs' own generated setstate uses.
        for name, value in state.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "_clients", {})
        object.__setattr__(self, "_client_lock", threading.Lock())
        object.__setattr__(self, "_meter", _TransportMeter())

    def warm_client(self, uri: str) -> None:
        """Resolve and cache this URI's backend client without reading the object.

        Lets a caller that reads a batch concurrently resolve every backend it
        will touch from one thread first. This makes an unresolvable profile
        raise once and deterministically instead of racing in every worker, and
        keeps client construction off the worker threads. The guarded cache miss
        in :meth:`_client` makes concurrent :meth:`read` safe even without this
        warm-up.

        A local path is a no-op because it reaches no client at all.

        Args:
            uri: A URI this reader may later be asked to read.

        Raises:
            MediaSystemicError: If the profile resolves to no usable client.

        """
        if is_remote_path(uri):
            self._client(uri)

    def transport_stats(self) -> MediaTransportStats:
        """Return what this reader's reads have cost so far in this process.

        A fresh value each call, so a caller may hold one reading beside a later
        one and compare them.
        """
        return self._meter.snapshot()

    def read(self, uri: str, *, max_bytes: int) -> bytes:
        """Return the object's complete bytes, recording what the read cost.

        Args:
            uri: Local path, ``file://``, ``s3://``, or ``az://`` object URI.
            max_bytes: Ceiling on the object's size.

        Returns:
            The object's bytes, at least one and at most ``max_bytes``.

        Raises:
            ValueError: If ``max_bytes`` is less than one, so no object could
                satisfy it.
            MediaObjectError: If the object is missing, empty, larger than
                ``max_bytes``, or named by a URI this reader cannot turn into a
                readable object - an unsupported or malformed scheme included.
            MediaTransportError: If every attempt failed with a transient fault.
            MediaSystemicError: If a remote backend rejected the credentials,
                the bucket or container does not exist, the account is disabled,
                the storage profile resolved to no client at all, or a local
                path was denied without proof the denial is this object's.

        """
        # Deliberately NOT a MediaReadError: no object is at fault, and every
        # object would be refused identically. Classified as an object fault it
        # would reach the caller as a per-row terminal verdict, and a re-run
        # skips rows that already carry one - so a caller's bad argument would
        # mark the whole corpus permanently unreadable while the run exits
        # cleanly. Raising here fails the batch instead, which is the only
        # proportionate answer to a limit no object can satisfy. Checked before
        # the retry wrapper so an unsatisfiable limit costs no round trip.
        if max_bytes < 1:
            msg = f"max_bytes must be at least 1, got {max_bytes}"
            raise ValueError(msg)
        retrying = tenacity.Retrying(
            stop=tenacity.stop_after_attempt(self.max_attempts),
            wait=tenacity.wait_exponential(
                multiplier=_TRANSPORT_RETRY_MULTIPLIER_S,
                max=_TRANSPORT_RETRY_MAX_WAIT_S,
            ),
            retry=tenacity.retry_if_exception_type(MediaTransportError),
            reraise=True,
        )
        # Started here rather than inside the retry wrapper, because the whole
        # point is the difference between the two: this clock spans every
        # attempt and every backoff, the inner one spans a single attempt.
        started = time.monotonic()
        try:
            data, transport_seconds = retrying(self._timed_read_once, uri, max_bytes=max_bytes)
        except MediaReadError:
            # No attempt delivered bytes, so the entire call is undelivered
            # time. The attempts themselves were charged one at a time as each
            # one raised, which is why none is added here.
            self._meter.record_undelivered(seconds=time.monotonic() - started)
            raise
        # The second term is a RESIDUAL, not a second measurement: subtracting
        # the delivering attempt from the call's wall time leaves the failed
        # attempts, the backoff between them, and the retry bookkeeping, with no
        # interval counted twice and none unaccounted for. Deriving it instead
        # of timing it is what keeps the two terms from disagreeing.
        self._meter.record_delivered(
            nbytes=len(data),
            transport_seconds=transport_seconds,
            undelivered_seconds=time.monotonic() - started - transport_seconds,
        )
        return data

    def _timed_read_once(self, uri: str, *, max_bytes: int) -> tuple[bytes, float]:
        """Perform one attempt, returning its bytes beside how long that attempt took.

        Args:
            uri: The object URI, as :meth:`read` received it.
            max_bytes: Ceiling on the object's size.

        Returns:
            The delivered bytes, and the seconds this one attempt spent on them.

        Raises:
            MediaReadError: Whatever the attempt raised, after charging it.

        """
        started = time.monotonic()
        try:
            data = self._read_once(uri, max_bytes=max_bytes)
        except Exception:
            # Broader than this module's own taxonomy on purpose. The count
            # means "an attempt ran and delivered nothing", which is true of any
            # exception, so a raise some later edit forgets to classify still
            # shows up in the attempt total instead of vanishing from it.
            self._meter.record_failed_attempt()
            raise
        # The duration and the bytes leave in ONE expression, which is what
        # makes success-only counting structural rather than a matter of
        # statement order: an attempt that raised never reaches this line, so no
        # later rearrangement can hand its duration to the transport term.
        return data, time.monotonic() - started

    def _read_once(self, uri: str, *, max_bytes: int) -> bytes:
        """Perform one full attempt: size check, then transfer, then verify."""
        size_bytes = self._size_bytes(uri)
        if size_bytes is not None:
            # The cap is enforced HERE, off the store's own metadata, so a
            # mis-populated URI naming a whole source recording is refused for a
            # few hundred bytes of round trip. Checking it after the transfer
            # would let that one row exhaust the worker's heap, and an
            # OOM-killed worker costs every other row it was holding.
            _require_usable_size(uri, size_bytes, max_bytes)
        data = self._download(uri)
        # Re-checked against the delivered bytes because the size above is
        # advisory: a backend may omit it, and the object may have been replaced
        # between the metadata lookup and the transfer.
        _require_usable_size(uri, len(data), max_bytes)
        return data

    def _size_bytes(self, uri: str) -> int | None:
        """Return the object's size from one metadata lookup, or ``None`` if unreported."""
        if is_remote_path(uri):
            client = self._client(uri)
            try:
                return client.stat(path_to_prefix(uri)).size_bytes
            except Exception as e:
                raise _classified(uri, e) from e
        try:
            return local_media_path(uri).stat().st_size
        except Exception as e:
            raise _classified(uri, e) from e

    def _download(self, uri: str) -> bytes:
        """Transfer the object's bytes, classifying any failure."""
        if is_remote_path(uri):
            client = self._client(uri)
            try:
                return client.download_object_as_bytes(path_to_prefix(uri))
            except Exception as e:
                raise _classified(uri, e) from e
        try:
            return local_media_path(uri).read_bytes()
        except Exception as e:
            raise _classified(uri, e) from e

    def _client(self, uri: str) -> StorageClient:
        """Return the cached client for this URI's backend, building it on first use."""
        key = _backend_key(uri)
        cached = self._clients.get(key)
        if cached is not None:
            return cached
        # The build runs INSIDE the lock rather than under a double-check that
        # releases it first. That serializes concurrent first use of one backend,
        # which is the intent: the alternative resolves the same credentials once
        # per waiting thread and reports one systemic misconfiguration as many.
        # The cost is bounded to the first miss per backend per process.
        with self._client_lock:
            cached = self._clients.get(key)
            if cached is not None:
                return cached
            # Resolution sits OUTSIDE the per-object fault handling on purpose: a
            # profile with no matching credentials is wrong for every object, so
            # it must be reported as systemic rather than retried per object.
            try:
                client = get_storage_client(uri, profile_name=self.storage_profile)
            except Exception as e:
                msg = f"storage profile {self.storage_profile!r} could not be resolved for {uri}: {e}"
                raise MediaSystemicError(msg) from e
            if client is None:
                msg = f"no storage client available for {uri} under profile {self.storage_profile!r}"
                raise MediaSystemicError(msg)
            self._clients[key] = client
            return client


def _require_usable_size(uri: str, size_bytes: int, max_bytes: int) -> None:
    """Raise :class:`MediaObjectError` unless the size is within (0, ``max_bytes``]."""
    if size_bytes == 0:
        msg = f"media object is empty: {uri}"
        raise MediaObjectError(msg)
    if size_bytes > max_bytes:
        msg = f"media object is {size_bytes} bytes, above the {max_bytes}-byte limit: {uri}"
        raise MediaObjectError(msg)


def _classified(uri: str, error: Exception) -> MediaReadError:
    """Map a backend exception onto the object / transport / systemic trichotomy.

    Ordering is part of the contract: local permissions, remote authorization,
    provider resource codes, generic missing objects, unusable locations, then
    unknown transport faults.

    An unrecognized failure is reported as transport. After retries are
    exhausted, :meth:`MediaObjectReader.read` raises
    :class:`MediaTransportError`; callers decide whether that propagated error
    ends an attempt or receives another policy. This reader never creates a
    terminal row.
    """
    credential_error = _credential_error_of(error)
    if credential_error is not None:
        return _classified_credential_error(uri, credential_error)

    client_error = _client_error_of(error)
    azure_error = _resource_not_found_error_of(error)
    provider_failure = _classified_provider_resource(uri, client_error, azure_error)
    if provider_failure is not None:
        return provider_failure

    # A recognized provider exception with an unknown resource code must reach
    # the transport fallback, even when a wrapper happens to inherit from
    # FileNotFoundError. Otherwise an unrecognized bucket/account response would
    # be persisted as a terminal row-local verdict.
    if client_error is None and azure_error is None and is_missing_object_error(error):
        msg = f"media object is missing: {uri}"
        return MediaObjectError(msg)
    if isinstance(error, IsADirectoryError):
        msg = f"media URI names a directory, not an object: {uri}"
        return MediaObjectError(msg)

    # A URI that names no readable object at all - an unsupported or malformed
    # scheme, a location the syscall refuses outright - is charged to the ROW and
    # is terminal. One unusable location must cost one refused object rather than
    # the run, and the answer is a property of the URI, so another attempt
    # re-sends an identically doomed request and buys nothing but backoff.
    # This check sits BELOW the credential, bucket, and account checks above, and
    # must stay there: those faults are true of every object, and reporting one of
    # them per object would hand the caller a verdict for every row it holds.
    # Where a stored verdict is what marks a row done, that is the silent
    # failure - the run reports success, and the re-run after the fix has nothing
    # left that still looks unprocessed.
    if _names_no_readable_object(error):
        msg = f"media URI does not name a readable object: {uri} ({type(error).__name__}: {error})"
        return MediaObjectError(msg)
    msg = f"media read failed transiently for {uri}: {type(error).__name__}: {error}"
    return MediaTransportError(msg)


def _classified_credential_error(
    uri: str,
    error: PermissionError | NoCredentialsError | ClientAuthenticationError,
) -> MediaReadError:
    """Return the fault class for a local permission or remote credential error.

    A local ``PermissionError`` reaches the same verdict as a remote credential
    failure: neither exception carries a signal that the denial is one object's.
    """
    # PermissionError names a denial, not which inode caused it. A parent
    # directory, mount, or process identity that cannot read any object raises
    # the same exception as a single file's mode bits, so a row-local verdict
    # here would stamp every clip sharing the unusable path and the re-run after
    # the fix would find nothing left that still looks unprocessed. A successful
    # stat() earlier in the attempt does NOT narrow it: stat() needs only search
    # permission on the parents, so a mount option or a process identity that
    # denies reading every file in the tree still stats every one of them.
    # Split this only for a signal true of this entry and false of its siblings.
    return MediaSystemicError(f"media read was not authorized: {uri} ({error})")


def _classified_provider_resource(
    uri: str,
    client_error: ClientError | None,
    azure_error: ResourceNotFoundError | None,
) -> MediaReadError | None:
    """Return the classification for a recognized provider resource code."""
    if client_error is not None:
        code = client_error.response.get("Error", {}).get("Code")
        if code in _SYSTEMIC_S3_ERROR_CODES:
            msg = f"media read rejected with {code} for {uri}; the read path itself is unusable"
            return MediaSystemicError(msg)
        if code in _OBJECT_S3_ERROR_CODES:
            msg = f"media object is missing: {uri} (S3 {code})"
            return MediaObjectError(msg)

    if azure_error is not None:
        code = getattr(azure_error, "error_code", None)
        if code in _SYSTEMIC_AZURE_ERROR_CODES:
            msg = f"media read rejected with {code} for {uri}; the read path itself is unusable"
            return MediaSystemicError(msg)
        if code is None or code in _OBJECT_AZURE_ERROR_CODES:
            detail = "without an error code" if code is None else str(code)
            msg = f"media object is missing: {uri} (Azure {detail})"
            return MediaObjectError(msg)
    return None


def _names_no_readable_object(error: Exception) -> bool:
    """Return whether the failure reports the URI itself as unusable as a location."""
    if isinstance(error, ValueError):
        return True
    return isinstance(error, OSError) and error.errno in _UNUSABLE_LOCATION_ERRNOS


def _direct_and_chained_errors(error: BaseException) -> tuple[BaseException | None, ...]:
    """Return the outer error followed by its direct cause and context."""
    return error, error.__cause__, error.__context__


def _credential_error_of(
    error: BaseException,
) -> PermissionError | NoCredentialsError | ClientAuthenticationError | None:
    """Return a credential error, unwrapping one level of chaining."""
    for candidate in _direct_and_chained_errors(error):
        if isinstance(candidate, PermissionError | NoCredentialsError | ClientAuthenticationError):
            return candidate
    return None


def _client_error_of(error: BaseException) -> ClientError | None:
    """Return the ``ClientError`` behind ``error``, unwrapping one level of chaining.

    Storage clients surface a boto failure directly, but a wrapper (smart_open,
    a retry helper) may re-raise it as a plain ``OSError`` and keep the original
    only on the cause chain, which is where the code that decides the fault class
    lives.
    """
    for candidate in _direct_and_chained_errors(error):
        if isinstance(candidate, ClientError):
            return candidate
    return None


def _resource_not_found_error_of(error: BaseException) -> ResourceNotFoundError | None:
    """Return an Azure resource error, unwrapping one level of chaining."""
    for candidate in _direct_and_chained_errors(error):
        if isinstance(candidate, ResourceNotFoundError):
            return candidate
    return None


def _backend_key(uri: str) -> str:
    """Return the cache key identifying one storage backend (scheme plus authority)."""
    parsed = urlsplit(uri)
    return f"{parsed.scheme.lower()}://{parsed.netloc.lower()}"
