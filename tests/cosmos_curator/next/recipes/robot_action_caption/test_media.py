# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Media stage policy: which failures become a published row and which fail the attempt.

::

    fetched row   video_data_url set, media_error null    -> reaches the engine
    faulted row   media_error set, video_data_url null    -> bypasses the GPU
    raised        MediaTransportError / MediaSystemicError -> fails the attempt

``ClipMediaFetch`` is exercised as a plain object - its batch method is called
directly - because every behaviour here is the stage's own arithmetic over one
Arrow batch. Deploying it as a Ray actor would make the same checks slower
without making them better checked.

WHICH backend failure becomes which ``MediaReadError`` subclass is the reader's
contract, covered in ``tests/cosmos_curator/next/utils/test_media_reader.py``.
What is covered here is what the STAGE does with each subclass, so an object
fault is exercised once rather than once per flavour of unreadable object.

The object-fault cases run against real files under ``tmp_path``, so they take
the same code path production takes for a local clip URI. The transport and
systemic cases have no filesystem equivalent, so they are driven by replacing
the reader's transfer with the failure under test.

Cases that must reach a FETCHED row hold a real encoded clip, because the stage
probes container timing before publishing a URL: a placeholder byte string is
refused as unreadable media, so a fixture built from one would assert the
fetched path while exercising the faulted one.
"""

import base64
import pathlib
from collections.abc import Callable

import pyarrow as pa
import pytest
from loguru import logger

from cosmos_curator.next.media import ffmpeg
from cosmos_curator.next.recipes.robot_action_caption import media
from cosmos_curator.next.recipes.robot_action_caption.datasource import (
    CHECKPOINT_ID_COLUMN,
    FRAGMENT_ID_COLUMN,
    PROMPT_COLUMN,
    SCAN_SCHEMA,
)
from cosmos_curator.next.recipes.robot_action_caption.table import CLIP_ID_COLUMN, CLIP_URI_COLUMN
from cosmos_curator.next.utils import media_reader
from tests.cosmos_curator.next.recipes.robot_action_caption import caption_harness

# Builds a ``SCAN_SCHEMA`` batch, one row per clip URI given; see ``scanned_batch``.
ScannedBatch = Callable[..., pa.Table]

# Builds a ``ClipMediaFetch``; see ``media_fetch``.
MediaFetch = Callable[..., media.ClipMediaFetch]

# The fixture batch's fragment, and the high half of every row address in it.
# Deliberately non-zero so the checkpoint ids sit above 2**32: an id narrowed to
# 32 bits anywhere in the stage would still look plausible below that.
_FRAGMENT_ID = 3
_ROW_ADDRESS_FRAGMENT_SHIFT = 32

# The cap for the tests that are not ABOUT the cap: far above any fixture clip,
# so nothing admits or refuses a clip by accident.
_AMPLE_CAP_BYTES = 4096

# A cap no default, constant, or other fixture in this recipe produces, so a
# stage that ignored its configured value could not coincide with it.
_CONFIGURED_CAP_BYTES = 3 * 1024 * 1024 + 7

# A remote URI for the cases whose reader call is replaced: it never reaches a
# backend, and naming a scheme keeps it off the local-path branch.
_REMOTE_CLIP_URI = "s3://bucket/clips/a.mp4"


@pytest.fixture
def scanned_batch() -> ScannedBatch:
    """Return a builder for a ``SCAN_SCHEMA`` batch, one row per positional clip URI.

    Pass ``None`` for a row that names no media location. ``pa.Table.from_pylist``
    does not enforce nullability, which is exactly why a null can reach the stage
    in production too.
    """

    def build(*clip_uris: str | None) -> pa.Table:
        rows = [
            {
                CLIP_ID_COLUMN: f"c{index}",
                FRAGMENT_ID_COLUMN: _FRAGMENT_ID,
                CLIP_URI_COLUMN: clip_uri,
                PROMPT_COLUMN: f"describe c{index}",
                CHECKPOINT_ID_COLUMN: (_FRAGMENT_ID << _ROW_ADDRESS_FRAGMENT_SHIFT) | index,
            }
            for index, clip_uri in enumerate(clip_uris)
        ]
        return pa.Table.from_pylist(rows, schema=SCAN_SCHEMA)

    return build


class _UnusableClient:
    """A client that resolves but refuses to transfer, naming what reached it."""

    def __getattr__(self, name: str) -> object:
        """Fail the test rather than let an unstubbed transfer proceed."""
        pytest.fail(f"an unstubbed transfer reached the storage client: .{name}")


@pytest.fixture(autouse=True)
def _resolvable_remote_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """Let a remote clip URI resolve to a client without credentials present.

    The stage warms every backend before fanning out, so a remote case reaches
    client resolution even when its transfer is replaced. Unstubbed, each such
    case fails on absent credentials and reports that instead of the fault it was
    written to pin.

    Resolution is all this grants. The client itself refuses every call, so a
    test that reaches a real transfer fails on the stand-in rather than passing
    quietly, and a local case is untouched because it resolves no client at all.
    """
    monkeypatch.setattr(
        media_reader.MediaObjectReader,
        "_client",
        lambda self, uri: _UnusableClient(),  # noqa: ARG005 - signature must match the method.
        raising=True,
    )


@pytest.fixture
def media_fetch() -> MediaFetch:
    """Return a builder for the stage under test.

    Keyword arguments: ``attempts`` (default one, so a spent transport budget
    costs no backoff), ``max_clip_bytes``, and ``fetch_workers`` (default one,
    so a test that does not care about overlap runs the single-threaded path and
    keeps its assertion failures on the calling thread's traceback).
    """

    def build(
        *,
        attempts: int = 1,
        max_clip_bytes: int = _AMPLE_CAP_BYTES,
        fetch_workers: int = 1,
    ) -> media.ClipMediaFetch:
        return media.ClipMediaFetch(
            storage_profile="default",
            attempts=attempts,
            max_clip_bytes=max_clip_bytes,
            fetch_workers=fetch_workers,
        )

    return build


def outcome(fetched: pa.Table) -> tuple[str | None, str | None]:
    """Return the single row's ``(video_data_url, media_error)`` pair."""
    row = fetched.to_pylist()[0]
    return row[media.MEDIA_URL_COLUMN], row[media.MEDIA_ERROR_COLUMN]


def test_a_fetched_clip_becomes_a_data_url_carrying_its_exact_bytes(
    media_fetch: MediaFetch,
    scanned_batch: ScannedBatch,
    tmp_path: pathlib.Path,
) -> None:
    """The encoded payload decodes back to the clip byte for byte."""
    video = caption_harness.synthetic_clip_bytes("exact-bytes")
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(video)

    url, fault = outcome(media_fetch(max_clip_bytes=len(video))(scanned_batch(str(clip))))

    assert fault is None
    assert url is not None
    payload = url.removeprefix("data:video/mp4;base64,")
    # An unchanged string would mean the prefix was never written, which the
    # engine reads as an opaque URL rather than as inline video.
    assert payload != url
    assert base64.b64decode(payload) == video


def test_a_faulted_row_travels_beside_a_fetched_one_in_one_schema(
    media_fetch: MediaFetch,
    scanned_batch: ScannedBatch,
    tmp_path: pathlib.Path,
) -> None:
    """Both outcomes leave the stage in ``FETCHED_SCHEMA``, one column live per row.

    One schema per stage is what lets the engine boundary read a single shape:
    two key sets out of one operator produce a heterogeneous block that fails at
    the Parquet write, far from the clip that caused it.
    """
    video = caption_harness.synthetic_clip_bytes("mixed-batch")
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(video)

    fetched = media_fetch(max_clip_bytes=len(video))(scanned_batch(str(clip), str(tmp_path / "absent.mp4")))
    rows = fetched.to_pylist()

    assert fetched.schema == media.FETCHED_SCHEMA
    assert [row[media.MEDIA_URL_COLUMN] is not None for row in rows] == [True, False]
    assert [row[media.MEDIA_ERROR_COLUMN] is not None for row in rows] == [False, True]


def test_an_object_fault_becomes_a_terminal_error_row_named_by_class(
    media_fetch: MediaFetch,
    scanned_batch: ScannedBatch,
    tmp_path: pathlib.Path,
) -> None:
    """A deterministic object fault names its class and stable clip identity.

    ``media_error`` is published as ``caption_error``, so the prefix leaves the
    process. It is a constant there - ``MediaObjectError`` is the only class the
    stage catches - and the clip id identifies the affected row without exposing
    a storage location that may contain credentials.

    The expected prefix is spelled out rather than read from
    ``MediaObjectError.__name__``. Deriving it would look like decoupling, but
    both sides would then re-derive through the same attribute the stage uses, so
    a rename would keep this green while changing what is stored. Updating the
    literal is the cost that makes such a rename a deliberate migration.
    """
    absent = tmp_path / "absent.mp4"

    url, fault = outcome(media_fetch()(scanned_batch(str(absent))))

    assert url is None
    assert fault is not None
    assert fault.startswith("MediaObjectError: ")
    assert "clip_id='c0'" in fault
    assert str(absent) not in fault


def test_an_object_fault_exposes_only_the_stable_clip_id(
    media_fetch: MediaFetch,
    monkeypatch: pytest.MonkeyPatch,
    scanned_batch: ScannedBatch,
) -> None:
    """Neither the persisted fault nor its warning exposes a credential-bearing URI."""
    sensitive_uri = (
        "s3://access-key:secret@bucket/private.mp4"
        "?X-Amz-Credential=credential&X-Amz-Signature=signature\nforged-warning"
    )

    def raise_object_fault(_self: object, uri: str, *, max_bytes: int) -> bytes:
        del max_bytes
        message = f"media object is missing: {uri}"
        raise media_reader.MediaObjectError(message)

    monkeypatch.setattr(media_reader.MediaObjectReader, "_read_once", raise_object_fault)
    warnings: list[str] = []
    handler_id = logger.add(warnings.append, format="{message}", level="WARNING")
    try:
        url, fault = outcome(media_fetch()(scanned_batch(sensitive_uri)))
    finally:
        logger.remove(handler_id)

    warning = "".join(warnings)
    assert url is None
    assert fault == "MediaObjectError: clip_id='c0'"
    assert "Clip 'c0' is not captionable" in warning
    assert sensitive_uri not in fault
    assert sensitive_uri not in warning
    assert "X-Amz-Credential" not in warning
    assert "X-Amz-Signature" not in warning
    assert "forged-warning" not in warning


def test_a_clip_whose_container_carries_no_usable_timing_becomes_an_error_row(
    media_fetch: MediaFetch,
    scanned_batch: ScannedBatch,
    tmp_path: pathlib.Path,
) -> None:
    """A clip the engine's frame selection could not survive is charged to the row.

    This is the fault the stage's probe exists for, and the reason it cannot be
    left to the engine: the video loaders derive a frame count from the
    container's declared rate, and a container declaring none is a raised
    exception inside the engine stage rather than an error row. That stage is
    pinned to re-raise, so a single such clip ends an attempt that has already
    paid for every caption before it.

    Which exception the engine would raise is a property of the CHECKPOINT and is
    not asserted here: at pinned vLLM 0.27.1 a per-frame-family backend refuses a
    non-positive rate outright while a whole-video-family one divides by it. The
    point of the probe is that neither reaches the engine.
    """
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"\x00\x01\xfe\xff")

    url, fault = outcome(media_fetch()(scanned_batch(str(clip))))

    assert url is None
    assert fault is not None
    assert "clip_id='c0'" in fault
    # The reader admitted the object, so this is the probe's verdict and not a
    # read failure wearing the same shape.
    assert not fault.startswith("MediaObjectError: ")


def test_a_probe_fault_exposes_only_the_stable_clip_id(
    media_fetch: MediaFetch,
    monkeypatch: pytest.MonkeyPatch,
    scanned_batch: ScannedBatch,
    tmp_path: pathlib.Path,
) -> None:
    """The persisted probe fault carries no ffprobe diagnostic text.

    An ffprobe message is media-derived: it quotes the container's own metadata
    and the path it was given, so publishing it would put attacker-influenced
    text into a column operators read and into the log line beside it.
    """
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"not a video")
    leaked = "moov atom not found\nforged-warning"

    def raise_probe_fault(_data: bytes) -> object:
        raise ffmpeg.ProbeError(leaked)

    monkeypatch.setattr(media, "probe_video_bytes", raise_probe_fault)
    warnings: list[str] = []
    handler_id = logger.add(warnings.append, format="{message}", level="WARNING")
    try:
        url, fault = outcome(media_fetch()(scanned_batch(str(clip))))
    finally:
        logger.remove(handler_id)

    warning = "".join(warnings)
    assert url is None
    assert fault == "ProbeError: clip_id='c0'"
    assert leaked not in warning
    assert "forged-warning" not in warning


def test_a_null_clip_uri_becomes_a_terminal_error_row(
    media_fetch: MediaFetch,
    scanned_batch: ScannedBatch,
) -> None:
    """A source row with no location says so, rather than reporting a path called "None".

    ``SCAN_SCHEMA`` cannot pin this away: ``pa.Table.from_pylist`` ignores its
    nullability, so the null arrives at the stage. Rendered through ``str()`` it
    would become the literal path "None" and publish "media object is missing:
    None", which triage cannot tell from a clip genuinely named "None". The
    negative assertion is what keeps that fallback out.
    """
    warnings: list[str] = []
    handler_id = logger.add(warnings.append, format="{message}")
    try:
        url, fault = outcome(media_fetch()(scanned_batch(None)))
    finally:
        logger.remove(handler_id)

    assert url is None
    assert fault is not None
    assert "clip_uri is null" in fault
    assert "is missing" not in fault
    assert "Clip 'c0' is not captionable" in "".join(warnings)


def test_the_configured_cap_reaches_the_media_reader(
    media_fetch: MediaFetch,
    monkeypatch: pytest.MonkeyPatch,
    scanned_batch: ScannedBatch,
) -> None:
    """The cap the actor was built with is the cap the reader enforces.

    Asserted on the value handed to the reader rather than on a refusal message,
    because the cap is now a config field that reaches the contract digest: a
    stage enforcing some other number would judge the corpus under semantics no
    published row records.
    """
    caps: list[int] = []

    def record_cap(_self: object, _uri: str, *, max_bytes: int) -> bytes:
        caps.append(max_bytes)
        return b"clip bytes"

    monkeypatch.setattr(media_reader.MediaObjectReader, "read", record_cap)

    media_fetch(max_clip_bytes=_CONFIGURED_CAP_BYTES)(scanned_batch(_REMOTE_CLIP_URI))

    assert caps == [_CONFIGURED_CAP_BYTES]


def test_a_clip_of_exactly_the_cap_is_fetched(
    media_fetch: MediaFetch,
    scanned_batch: ScannedBatch,
    tmp_path: pathlib.Path,
) -> None:
    """The cap is the largest admitted size, not the smallest refused one.

    The boundary is asserted rather than assumed because both readings publish a
    plausible-looking corpus: an exclusive cap silently turns every clip of
    exactly the configured size into a terminal error row.
    """
    video = caption_harness.synthetic_clip_bytes("exactly-the-cap")
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(video)

    url, fault = outcome(media_fetch(max_clip_bytes=len(video))(scanned_batch(str(clip))))

    assert fault is None
    assert url is not None


def test_a_spent_transport_budget_fails_the_attempt(
    media_fetch: MediaFetch,
    monkeypatch: pytest.MonkeyPatch,
    scanned_batch: ScannedBatch,
) -> None:
    """An exhausted retry budget ends the attempt instead of publishing a row.

    The reader has already spent its whole budget by the time this class escapes,
    and a budget that runs out for one clip is far more often a degraded read
    path than a genuinely unreadable object. Absorbing it would record an outage
    as data: a published error row is no longer null, so the re-run after the fix
    skips every clip the outage touched.

    The escaping class is pinned deliberately. ``MediaTransportError`` messages
    embed the inner exception type, and this repo enables Ray Data's
    ``retried_map_errors`` with substrings including ``OSError`` and
    ``TimeoutError`` (``next/core/ray_runtime.py``), so during a real outage the
    whole block can be retried and each retry re-spends the reader's own
    attempts. That multiplication is accepted, and is why an attempt count in the
    dozens during an outage is not a bug.

    One attempt is configured so the budget is spent without paying tenacity's
    backoff; how many attempts a fault gets is the reader's contract, covered in
    ``test_media_reader.py``.
    """

    def raise_transport_fault(*_args: object, **_kwargs: object) -> bytes:
        message = "media read failed transiently for the clip under test"
        raise media_reader.MediaTransportError(message)

    monkeypatch.setattr(media_reader.MediaObjectReader, "_read_once", raise_transport_fault)

    with pytest.raises(media_reader.MediaTransportError, match="failed transiently for the clip under test"):
        media_fetch(attempts=1)(scanned_batch(_REMOTE_CLIP_URI))


def test_a_systemic_fault_fails_the_attempt_instead_of_producing_a_row(
    media_fetch: MediaFetch,
    monkeypatch: pytest.MonkeyPatch,
    scanned_batch: ScannedBatch,
) -> None:
    """A failure true of every clip must not be buried as millions of per-row errors.

    This is the one media outcome whose wrong behaviour is silent. Absorbing it
    into a row would report a successful run over a table of permanent error
    rows, and a re-run then skips each of them because the status column is no
    longer null - so the outage is recorded as data and never retried.
    """

    def raise_systemic_fault(*_args: object, **_kwargs: object) -> bytes:
        message = "media read was not authorized for the clip under test"
        raise media_reader.MediaSystemicError(message)

    monkeypatch.setattr(media_reader.MediaObjectReader, "_read_once", raise_systemic_fault)

    with pytest.raises(media_reader.MediaSystemicError, match="not authorized for the clip under test"):
        media_fetch()(scanned_batch(_REMOTE_CLIP_URI))


def test_row_routing_identity_survives_fetch_on_success_and_failure(
    media_fetch: MediaFetch,
    scanned_batch: ScannedBatch,
    tmp_path: pathlib.Path,
) -> None:
    """Every stable identity field arrives unchanged, on both outcomes.

    Ray marks a row complete by reading this column back off the staged output,
    while the staged sink partitions by fragment and publication joins by clip.
    A fetch failure must therefore retain all three fields together. The fixture
    ids sit above 2**32, which makes a checkpoint-id narrowing visible.
    """
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(caption_harness.synthetic_clip_bytes("checkpoint-id"))
    scanned = scanned_batch(str(clip), str(tmp_path / "absent.mp4"))

    fetched = media_fetch()(scanned)

    identity_columns = [CLIP_ID_COLUMN, FRAGMENT_ID_COLUMN, CHECKPOINT_ID_COLUMN]
    assert fetched.select(identity_columns).to_pylist() == scanned.select(identity_columns).to_pylist()
    assert fetched.column(media.MEDIA_ERROR_COLUMN).to_pylist()[1] is not None
