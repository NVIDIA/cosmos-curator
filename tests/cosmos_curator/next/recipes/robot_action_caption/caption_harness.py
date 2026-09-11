# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Builders for running a whole caption attempt locally with a fake engine.

``build_vllm_transform`` cannot run in a test process: it calls ``ray.init``,
downloads tokenizer files, and imports ``vllm``. The seam used here is the module
attribute ``inference.build_vllm_transform``, replaced by a builder that accepts
the same ``(config, preprocess=, postprocess=)`` call and returns the same
``Dataset -> Dataset`` transform. Everything else in an attempt is real: the Lance
table, the media reads, the Ray Data plan, Ray's own checkpoint filter, the staged
Parquet, and the Lance commit.

The fake stands in for the ENGINE STAGES only, and it reproduces the one processor
behaviour the recipe's row shapes depend on: when ``__inference_error__`` is
non-empty the processor returns the row's accumulated data verbatim and never
calls the postprocess hook. Both recipe hooks are the production ones, so no
recipe logic is restated here.

::

    read (real) --> media fetch (real) --> FAKE engine --> canonicalize (real)
                                               |               |
                                               |               v
                                               |        write_parquet (real,
                                               |        checkpointed by Ray)
                                               +-- records every clip it saw,
                                                   where the driver can read it

Invocations are recorded as files rather than in a counter, because the engine
runs in a Ray worker process where a driver-side counter would stay at zero.

The fixtures that own the temporary directories and the monkeypatching live in
this package's ``conftest.py``; everything here is a plain builder.
"""

import functools
import pathlib
import shutil
import subprocess
import tempfile
import time
import uuid
from collections.abc import Callable, Collection, Sequence
from typing import Any, Final

import attrs
import lance
import pyarrow as pa
import pyarrow.dataset as pads
import pytest

from cosmos_curator.models.registry import resolve_registered_model
from cosmos_curator.next.recipes.robot_action_caption.config import (
    DEFAULT_MODEL_VARIANT,
    ResolvedRobotActionCaptionConfig,
    resolve_config_data,
)
from cosmos_curator.next.recipes.robot_action_caption.contracts import (
    CAPTION_COLUMN,
    CAPTION_DIGEST_COLUMN,
    CAPTION_FIELDS,
)
from cosmos_curator.next.recipes.robot_action_caption.datasource import (
    CHECKPOINT_ID_COLUMN,
    FRAGMENT_ID_COLUMN,
)
from cosmos_curator.next.recipes.robot_action_caption.request import (
    GENERATED_TEXT_COLUMN,
    GENERATED_TOKENS_COLUMN,
    INFERENCE_ERROR_COLUMN,
    INPUT_TOKENS_COLUMN,
)
from cosmos_curator.next.recipes.robot_action_caption.table import (
    CLIP_ID_COLUMN,
    CLIP_URI_COLUMN,
    CURATE_SELECTED_VALUE,
    CURATE_SELECTION_REASON,
)
from cosmos_curator.next.utils.lance_utils import LANCE_DATA_STORAGE_VERSION

# Comfortably clear of both truncation bounds - well below the output cap, and
# leaving the context length nearly untouched - so every fake generation
# classifies as ``success`` and no caption's status depends on this harness's
# arithmetic.
_GENERATED_TOKENS: Final = 8
_PROMPT_TOKENS: Final = 64

# How long a refused clip waits for the clips that already finished to become
# durable. See :func:`_await_a_committed_checkpoint`.
_CHECKPOINT_BARRIER_TIMEOUT_S: Final = 180.0
_CHECKPOINT_BARRIER_POLL_S: Final = 0.05

# Uncommitted first half of Ray's two-phase checkpoint commit. A file carrying it
# names rows whose staged shard may still be removed on recovery, so counting it
# would report progress that is not durable.
_PENDING_CHECKPOINT_SUFFIX: Final = ".pending.parquet"

# What the fake engine stage emits: the exact key set ``build_request`` seeds plus
# the generation columns ``read_generation`` reads. One schema for both engine
# outcomes, which is the property ``canonicalize`` relies on.
_ENGINE_SCHEMA: Final = pa.schema(
    [
        pa.field(CLIP_ID_COLUMN, pa.string(), nullable=False),
        pa.field(FRAGMENT_ID_COLUMN, pa.int64(), nullable=False),
        pa.field(CHECKPOINT_ID_COLUMN, pa.uint64(), nullable=False),
        *CAPTION_FIELDS,
        pa.field(GENERATED_TEXT_COLUMN, pa.large_string(), nullable=True),
        pa.field(GENERATED_TOKENS_COLUMN, pa.int64(), nullable=True),
        pa.field(INPUT_TOKENS_COLUMN, pa.int64(), nullable=True),
        pa.field(INFERENCE_ERROR_COLUMN, pa.string(), nullable=True),
    ]
)

_SOURCE_SCHEMA: Final = pa.schema(
    [
        pa.field(CLIP_ID_COLUMN, pa.string(), nullable=False),
        pa.field(CLIP_URI_COLUMN, pa.large_string(), nullable=False),
        pa.field("task_name", pa.string(), nullable=False),
        pa.field("subtask_name", pa.string(), nullable=False),
        pa.field("camera_motion_annotation", pa.large_string(), nullable=True),
        pa.field(CURATE_SELECTION_REASON, pa.string(), nullable=True),
    ]
)

# The smallest real video the media stage will accept, since its probe reads the
# container's declared timing. Kept tiny because nothing here decodes it: the
# frames only have to exist for ffprobe to report a frame rate.
_SYNTHETIC_CLIP_SIZE: Final = "64x64"
_SYNTHETIC_CLIP_FPS: Final = 4
_SYNTHETIC_CLIP_SECONDS: Final = 1

# The repository encodes with the LGPL encoder rather than x264/x265. A fixture
# has no licensing story of its own, but matching the production encoder keeps a
# fixture clip probeable by exactly what production clips are probeable by.
_SYNTHETIC_CLIP_ENCODER: Final = "libopenh264"


def require_video_encoder() -> str:
    """Return the ``ffmpeg`` binary able to write a fixture clip, or skip the test.

    Returns:
        Path to an ``ffmpeg`` carrying the LGPL H.264 encoder this repository
        builds against.

    """
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("ffmpeg is not available")
    encoders = subprocess.run(  # noqa: S603
        [ffmpeg, "-encoders"], capture_output=True, text=True, timeout=30, check=False
    )
    if _SYNTHETIC_CLIP_ENCODER not in encoders.stdout:
        pytest.skip(f"{_SYNTHETIC_CLIP_ENCODER} encoder is not available")
    return ffmpeg


@functools.cache
def synthetic_clip_bytes(tag: str) -> bytes:
    """Return one distinct playable video, cached per tag for the whole session.

    The media stage probes a clip's container timing before encoding it into a
    data URL, so a fixture payload has to be a real video: a placeholder byte
    string is - correctly - refused as unreadable media, which would leave every
    fixture clip a media error and every downstream assertion about a caption
    unreachable.

    Args:
        tag: Distinguishes one fixture clip's bytes from another's. Written into
            the container as metadata so two clips never share a payload, which
            is what lets a test tell which clip a data URL came from.

    Returns:
        The encoded file's bytes.

    """
    ffmpeg = require_video_encoder()
    with tempfile.TemporaryDirectory(prefix="caption_fixture_clip_") as tmp_dir:
        path = pathlib.Path(tmp_dir) / "clip.mp4"
        subprocess.run(  # noqa: S603
            [
                ffmpeg,
                "-y",
                "-f",
                "lavfi",
                "-i",
                f"testsrc=duration={_SYNTHETIC_CLIP_SECONDS}:size={_SYNTHETIC_CLIP_SIZE}:rate={_SYNTHETIC_CLIP_FPS}",
                "-c:v",
                _SYNTHETIC_CLIP_ENCODER,
                "-pix_fmt",
                "yuv420p",
                "-metadata",
                f"comment={tag}",
                str(path),
            ],
            check=True,
            capture_output=True,
            timeout=60,
        )
        return path.read_bytes()


@attrs.frozen
class InferenceLedger:
    """Which clips reached the fake engine, readable from the driver after a run.

    Attributes:
        directory: Holds one file per engine invocation, named ``<clip_id>.<nonce>``.

    """

    directory: pathlib.Path

    @property
    def clip_ids(self) -> list[str]:
        """Return the distinct clips the engine was asked to caption, sorted."""
        # Split from the RIGHT: the nonce is the suffix, and a clip id may itself
        # carry dots, so splitting at the first one merges two clips that share a
        # prefix into a single entry and understates the work the engine did.
        return sorted({path.name.rsplit(".", 1)[0] for path in self.directory.iterdir()})

    @property
    def calls(self) -> int:
        """Return how many times the engine was invoked, duplicates included."""
        return len(list(self.directory.iterdir()))


@attrs.frozen
class StagedRow:
    """One terminal row a Phase A attempt left under its workspace.

    Attributes:
        clip_id: The clip it captions.
        caption: The generated text, carrying the producing attempt's tag.
        digest: The contract digest the attempt stamped on it.

    """

    clip_id: str
    caption: str | None
    digest: str


# Installs the fake engine and returns the ledger recording its invocations.
# Keyword arguments: ``fails_on`` (clip ids the engine refuses), ``caption_tag``
# (prefix on the generated text, identifying which attempt produced a caption),
# and ``barrier_root`` (staging root polled before a refusal).
FakeEngine = Callable[..., InferenceLedger]

# Builds a local ``clips.lance`` plus its clip objects and returns its URI.
# Keyword arguments: ``rows``, ``rows_per_file``, ``selection``, and
# ``omit_selection``.
ClipsTable = Callable[..., str]

# Builds a resolved config for one table. Keyword arguments: ``max_clip_bytes``,
# ``media_attempts``, ``geometry_rows``, ``progress``, and ``selected_only``.
CaptionConfig = Callable[..., ResolvedRobotActionCaptionConfig]


def stage_model_weights(root: pathlib.Path) -> pathlib.Path:
    """Create a minimal model directory accepted by the registry validator.

    The fake engine never loads these bytes; the fixture only preserves the
    production driver's config-plus-weight precondition.

    Args:
        root: Weight-cache root the resolved model reads.

    Returns:
        ``root``, so the fixture can install it as the environment cache root.

    """
    model_dir = root / resolve_registered_model(DEFAULT_MODEL_VARIANT).model_id
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (model_dir / "model.safetensors").write_bytes(b"not a real weight")
    return root


def build_clips_table(
    root: pathlib.Path,
    *,
    rows: int,
    rows_per_file: int,
    selection: Sequence[str | None] | None = None,
    omit_selection: bool = False,
) -> str:
    """Write a local clip table whose clip objects exist on disk, and return its URI.

    ``rows_per_file`` decides the durable Lance fragment geometry. Phase A may
    split each fragment into smaller fetch blocks. A checkpoint identifies the
    individual rows it covers, but the unit that becomes durable is one WRITE
    BUNDLE, so a fragment split into several blocks becomes durable in pieces.

    Args:
        root: Directory to hold the table and the clip objects.
        rows: Clips to write, all pending (the caption columns are absent, so the
            attempt's own widening is exercised).
        rows_per_file: Rows per Lance fragment.
        selection: Per-row curation verdicts, or ``None`` to mark every row
            selected.
        omit_selection: Omit the curation column for pre-curation table tests.

    Returns:
        The clip table URI.

    """
    media = root / "media"
    media.mkdir(exist_ok=True)
    uris = []
    for index in range(rows):
        clip = media / f"c{index}.mp4"
        clip.write_bytes(synthetic_clip_bytes(f"c{index}"))
        uris.append(str(clip))
    columns = {
        CLIP_ID_COLUMN: pa.array([f"c{index}" for index in range(rows)], pa.string()),
        CLIP_URI_COLUMN: pa.array(uris, pa.large_string()),
        "task_name": pa.array([f"task{index}" for index in range(rows)], pa.string()),
        "subtask_name": pa.array([f"subtask{index}" for index in range(rows)], pa.string()),
        "camera_motion_annotation": pa.array([f"pan {index}" for index in range(rows)], pa.large_string()),
        CURATE_SELECTION_REASON: pa.array(
            list(selection) if selection is not None else [CURATE_SELECTED_VALUE] * rows,
            pa.string(),
        ),
    }
    schema = _SOURCE_SCHEMA
    if omit_selection:
        columns.pop(CURATE_SELECTION_REASON)
        schema = pa.schema([field for field in schema if field.name != CURATE_SELECTION_REASON])
    table = pa.table(columns, schema=schema)
    uri = str(root / "clips.lance")
    lance.write_dataset(table, uri, max_rows_per_file=rows_per_file, data_storage_version=LANCE_DATA_STORAGE_VERSION)
    return uri


def build_caption_config(
    uri: str,
    *,
    staging_root: pathlib.Path,
    max_clip_bytes: int | None = None,
    media_attempts: int | None = None,
    geometry_rows: int | None = None,
    progress: bool | None = None,
    selected_only: bool | None = None,
) -> ResolvedRobotActionCaptionConfig:
    """Return a resolved config for one attempt inside a test process.

    ``geometry_rows`` shrinks the three row-count knobs together when a recovery
    test needs several durable groups from a fixture-scale corpus. It is omitted
    by default so tests concerned with production geometry see shipped values.

    Args:
        uri: The clip table to caption.
        staging_root: Durable root for the workspace.
        max_clip_bytes: Media byte cap, or ``None`` for the shipped default. Set
            it only to exercise the cap itself: it reaches the digest, so a test
            that changes it is captioning under different semantics.
        media_attempts: Transport read attempts per clip, or ``None`` for the
            shipped default. Unlike the cap it never reaches the digest, so a
            test may set it freely.
        geometry_rows: Optional shared inference-batch, Lance-read-batch, and
            staged-file row count for fixture-scale recovery tests.
        progress: Ray Data progress output, or ``None`` for the shipped default.
            Execution-only like ``media_attempts``, so it is free to set.
        selected_only: Whether to restrict work to Curate-selected rows, or
            ``None`` for the shipped default.

    Returns:
        The resolved configuration.

    """
    media = {} if max_clip_bytes is None else {"max_clip_bytes": max_clip_bytes}
    execution: dict[str, Any] = {"storage_profile": "default", "inference_concurrency": 1}
    if media_attempts is not None:
        execution["media_attempts"] = media_attempts
    if geometry_rows is not None:
        execution.update(
            inference_batch_size=geometry_rows,
            lance_read_batch_size=geometry_rows,
            parquet_rows_per_file=geometry_rows,
        )
    if progress is not None:
        execution["progress"] = progress
    input_config: dict[str, Any] = {"clips_lance_uri": uri}
    if selected_only is not None:
        input_config["selected_only"] = selected_only
    return resolve_config_data(
        {
            "schema_version": 1,
            "kind": "robot-action-caption",
            "input": input_config,
            "output": {"staging_root_uri": str(staging_root)},
            "media": media,
            "execution": execution,
        }
    )


def fake_transform_builder(
    *,
    ledger: InferenceLedger,
    fails_on: Collection[str] = (),
    caption_tag: str = "attempt",
    barrier_root: pathlib.Path | None = None,
) -> Callable[..., Callable[[Any], Any]]:
    """Return a stand-in for ``build_vllm_transform`` with the same call shape.

    Args:
        ledger: Where each invocation is recorded.
        fails_on: Clips the engine refuses, which fails the whole attempt.
        caption_tag: Prefix on the generated text, naming the producing attempt.
        barrier_root: Staging root polled before a refusal, so a crash scenario
            does not depend on which write task the scheduler finishes first.
            Requires the caller's finished set to be one write bundle; see
            :func:`_await_a_committed_checkpoint`.

    Returns:
        A callable accepting the production builder's arguments.

    """
    refused = frozenset(fails_on)

    def build(
        config: Any,  # noqa: ANN401 - mirrors the production positional config parameter
        *,
        preprocess: Callable[[dict[str, Any]], dict[str, Any]],
        postprocess: Callable[[dict[str, Any]], dict[str, Any]],
    ) -> Callable[[Any], Any]:
        batch_size = int(config.batch_size)

        def engine(batch: pa.Table) -> pa.Table:
            generated = []
            for scanned in batch.to_pylist():
                request = preprocess(dict(scanned))
                if request.get(INFERENCE_ERROR_COLUMN):
                    # A row that failed before the engine is passed through
                    # untouched; the processor does not call postprocess for it.
                    generated.append(request)
                    continue
                clip_id = str(request[CLIP_ID_COLUMN])
                (ledger.directory / f"{clip_id}.{uuid.uuid4().hex}").write_bytes(b"")
                if clip_id in refused:
                    _await_a_committed_checkpoint(barrier_root)
                    msg = f"fake engine refused clip {clip_id}"
                    raise RuntimeError(msg)
                generated.append(
                    postprocess(
                        {
                            **request,
                            GENERATED_TEXT_COLUMN: f"{caption_tag} caption for {clip_id}",
                            GENERATED_TOKENS_COLUMN: _GENERATED_TOKENS,
                            INPUT_TOKENS_COLUMN: _PROMPT_TOKENS,
                        }
                    )
                )
            return pa.Table.from_pylist(generated, schema=_ENGINE_SCHEMA)

        def transform(dataset: Any) -> Any:  # noqa: ANN401 - a ray.data.Dataset either way
            return dataset.map_batches(engine, batch_format="pyarrow", batch_size=batch_size, num_cpus=0.25)

        return transform

    return build


def staged_rows(staging_root: str) -> list[StagedRow]:
    """Return every terminal row staged under a staging root, across all namespaces.

    Args:
        staging_root: The configured staging root.

    Returns:
        One entry per staged row, empty when nothing has been staged.

    Raises:
        ValueError: If a staged row has no contract digest.

    """
    # Ray's Parquet data sink writes results/ files straight to their final
    # path; ".pending.parquet" appears only under checkpoints/, so the commit
    # filter belongs there and not here.
    files = [str(path) for path in sorted(pathlib.Path(staging_root).rglob("results/**/*.parquet"))]
    if not files:
        return []
    table = pads.dataset(files, format="parquet").to_table(
        columns=[CLIP_ID_COLUMN, CAPTION_COLUMN, CAPTION_DIGEST_COLUMN]
    )
    staged = []
    for row in table.to_pylist():
        digest = row[CAPTION_DIGEST_COLUMN]
        if digest is None:
            msg = f"staged row for clip_id {row[CLIP_ID_COLUMN]!r} has null {CAPTION_DIGEST_COLUMN}"
            raise ValueError(msg)
        staged.append(
            StagedRow(
                clip_id=str(row[CLIP_ID_COLUMN]),
                caption=row[CAPTION_COLUMN],
                digest=str(digest),
            )
        )
    return staged


def published_captions(uri: str, *, version: int | None = None) -> dict[str, str | None]:
    """Return each clip's published caption in the table at one version.

    Raises:
        ValueError: If a clip appears on more than one row. Keying by clip id
            would otherwise drop the earlier rows, so a publication that
            APPENDED a duplicate instead of replacing the row would still look
            like one caption per clip to every caller of this helper.

    """
    dataset = lance.dataset(uri, version=version)
    table = dataset.to_table(columns=[CLIP_ID_COLUMN, CAPTION_COLUMN])
    captions: dict[str, str | None] = {}
    for row in table.to_pylist():
        clip_id = str(row[CLIP_ID_COLUMN])
        if clip_id in captions:
            msg = f"caption table {uri} at v{int(dataset.version)} has duplicate {CLIP_ID_COLUMN} {clip_id!r}"
            raise ValueError(msg)
        captions[clip_id] = row[CAPTION_COLUMN]
    return captions


def published_digests(uri: str, *, version: int | None = None) -> set[str]:
    """Return the distinct contract digests stamped on every row at one version.

    Raises:
        ValueError: If any row remains unpublished without a contract digest.

    """
    dataset = lance.dataset(uri, version=version)
    table = dataset.to_table(columns=[CAPTION_DIGEST_COLUMN])
    digests = set()
    for value in table.column(CAPTION_DIGEST_COLUMN).to_pylist():
        if value is None:
            msg = (
                f"caption table {uri} at v{int(dataset.version)} has null {CAPTION_DIGEST_COLUMN}; "
                f"published_digests requires every row to carry {CAPTION_DIGEST_COLUMN}"
            )
            raise ValueError(msg)
        digests.add(str(value))
    return digests


def namespaces(staging_root: str) -> list[str]:
    """Return the recovery namespace directories present under a staging root, sorted."""
    inference_root = pathlib.Path(staging_root) / "inference"
    if not inference_root.is_dir():
        return []
    return sorted(str(path) for path in inference_root.iterdir() if path.is_dir())


def _await_a_committed_checkpoint(barrier_root: pathlib.Path | None) -> None:
    """Block until one checkpoint is committed, so a crash test is not a race.

    A refusal is only interesting once the clips that succeeded are recoverable,
    and which write task finishes first is a scheduling decision. Waiting here
    makes the scenario the test describes the one the test runs, on any machine.

    What this guarantees is ONE durable bundle, which is weaker than it looks.
    Ray checkpoints per WRITE TASK -- the checkpoint id is that task's data-file
    prefix -- so a committed file covers one write BUNDLE, not one source
    fragment and not the whole corpus. A caller must therefore either size its
    finished set to a single bundle, or assert against whatever became durable
    instead of against a fixed set. A caller that names a fixed multi-bundle set
    would pass this check with only the first bundle committed, and the refusal
    aborts the executor, so the rest would be cancelled unstaged.

    Returns immediately when no root is given, which is the case for a test that
    needs only the refusal.

    Args:
        barrier_root: Staging root to poll, or ``None`` to refuse immediately.

    Raises:
        TimeoutError: If nothing became durable within the bound. Raised rather
            than fallen through: a refusal reached without durable progress is a
            different scenario from the one the caller asked for, and the faked
            engine's refusal would still produce the plausible-looking failure
            every depending test asserts, so those tests would pass having
            measured no recovery.

    """
    if barrier_root is None:
        return
    deadline = time.monotonic() + _CHECKPOINT_BARRIER_TIMEOUT_S
    while time.monotonic() < deadline:
        if _has_a_committed_checkpoint(barrier_root):
            return
        time.sleep(_CHECKPOINT_BARRIER_POLL_S)
    msg = (
        f"no checkpoint was committed under {barrier_root} within {_CHECKPOINT_BARRIER_TIMEOUT_S}s, so this "
        f"refusal would simulate a crash that lost generations the caller expects a retry to recover"
    )
    raise TimeoutError(msg)


def _has_a_committed_checkpoint(barrier_root: pathlib.Path) -> bool:
    """Report whether any committed checkpoint exists under a staging root.

    A pending checkpoint is written BEFORE its data file and renamed only after
    the write succeeds, so the suffix is what separates durable progress from a
    write still in flight.
    """
    return any(
        not path.name.endswith(_PENDING_CHECKPOINT_SUFFIX) for path in barrier_root.rglob("checkpoints/*.parquet")
    )
