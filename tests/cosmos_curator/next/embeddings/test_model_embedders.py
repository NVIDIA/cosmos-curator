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

"""CPU (non-``env``) coverage for the text and image embedder legs.

The two model-backed legs have no non-``env`` coverage otherwise: their smoke
tests need staged weights and a GPU. These tests stub the heavy model out (a fake
``sentence_transformers`` module for text; a monkeypatched ``_embed`` for image)
so the row-level contract runs on CPU without weights.

Both legs are pure compute over positions: they read no identity column and emit
only their group's fields, one row per input row in input order (a per-row failure
emitting an all-NULL group instead of dropping the row). That is what lets the fill
worker attach ``clip_id`` positionally, so these tests pin row count, order, and
null masking rather than any key. Each embedder is constructed directly, so nothing
here touches the recipe's ``_stage_weights`` staging order.

The image leg's concurrent frame reads belong to ``ClipFrameReader`` and are covered
in ``test_frame_reader.py``; what is pinned here is only what the embedder does with
the position-keyed frames it gets back.
"""

import pathlib
import sys
import types
import zlib
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pytest

import cosmos_curator.next.embeddings.image.frame_reader as frame_reader_module
from cosmos_curator.next.embeddings.image.embedder import HfVisionImageEmbedder
from cosmos_curator.next.embeddings.image.frame_reader import ClipFrameReader
from cosmos_curator.next.embeddings.model_specs import DEFAULT_IMAGE_MODEL, DEFAULT_TEXT_MODEL, TextModelSpec
from cosmos_curator.next.embeddings.schemas import (
    EMBED_SOURCE_COLUMNS,
    IMAGE_COLUMN_GROUP,
    IMAGE_DIM,
    TEXT_COLUMN_GROUP,
    TEXT_DIM,
)
from cosmos_curator.next.embeddings.text.embedder import SentenceTransformerTextEmbedder


class _StubEncoder:
    """Stand-in for ``SentenceTransformer``: returns zeros of the right shape."""

    def __init__(self, weights_dir: str, *, local_files_only: bool = True, device: str = "cpu") -> None:
        pass

    def encode(self, texts: list[str], **_kwargs: object) -> npt.NDArray[np.float32]:
        return np.zeros((len(texts), TEXT_DIM), dtype=np.float32)


def _stub_text_embedder(monkeypatch: pytest.MonkeyPatch) -> SentenceTransformerTextEmbedder:
    """Build a text embedder whose model is the stub encoder (no weights, no torch load)."""
    fake_module = types.ModuleType("sentence_transformers")
    fake_module.SentenceTransformer = _StubEncoder  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)
    return SentenceTransformerTextEmbedder(spec=DEFAULT_TEXT_MODEL, result_fingerprint="stub-fp", device="cpu")


class _NormAwareEncoder:
    """Encoder stub whose output direction depends on the input text.

    Unlike ``_StubEncoder`` (which returns zeros), this returns a deterministic,
    per-text, far-from-unit-norm vector, and applies L2 normalization ONLY when
    ``normalize_embeddings=True`` is passed - exactly as ``SentenceTransformer``
    does. Two properties make the text pins non-vacuous on CPU without weights:

      - the raw vectors are scaled well away from unit-norm, so a leg that stops
        requesting normalization emits non-unit vectors (the norm pin bites);
      - the direction is seeded from the text bytes, so distinct task / subtask
        strings map to distinct directions that survive normalization (the
        dual-vector pin bites), whereas a constant vector would collapse to the
        same direction after L2 and hide a task==subtask regression.
    """

    _RAW_SCALE = 5.0  # push raw magnitude far from 1.0 so normalization is observable

    def __init__(self, weights_dir: str, *, local_files_only: bool = True, device: str = "cpu") -> None:
        pass

    def encode(
        self, texts: list[str], *, normalize_embeddings: bool = False, **_kwargs: object
    ) -> npt.NDArray[np.float32]:
        if not texts:
            return np.zeros((0, TEXT_DIM), dtype=np.float32)
        matrix = np.empty((len(texts), TEXT_DIM), dtype=np.float32)
        for index, text in enumerate(texts):
            seed = zlib.crc32(text.encode("utf-8"))
            matrix[index] = np.random.default_rng(seed).standard_normal(TEXT_DIM).astype(np.float32) * self._RAW_SCALE
        if normalize_embeddings:
            matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix


def _norm_aware_text_embedder(monkeypatch: pytest.MonkeyPatch) -> SentenceTransformerTextEmbedder:
    """Build a text embedder backed by the direction-preserving, norm-aware stub."""
    fake_module = types.ModuleType("sentence_transformers")
    fake_module.SentenceTransformer = _NormAwareEncoder  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)
    spec = TextModelSpec("bge_small_en_v1_5", "bge-test-model-id", TEXT_DIM)
    return SentenceTransformerTextEmbedder(spec=spec, result_fingerprint="bge-test-fingerprint", device="cpu")


def _image_embedder_without_weights(*, read_concurrency: int = 1) -> HfVisionImageEmbedder:
    """Build an ``HfVisionImageEmbedder`` without running its weight-loading constructor.

    The row-level contract (frame read, NULL-on-failure path, scatter-back) needs
    no model; only ``_embed`` does, and tests stub that separately. The result
    fingerprint is set independently of the spec so the provenance pin asserts a
    distinctive value. ``read_concurrency`` is explicit rather than inherited from
    the config, so a test that reads a batch concurrently states the width it
    needs and no test depends on a production default.
    """
    embedder = object.__new__(HfVisionImageEmbedder)
    embedder._spec = DEFAULT_IMAGE_MODEL
    embedder._result_fingerprint = "stub"
    embedder._reader = ClipFrameReader(storage_profile="default", read_concurrency=read_concurrency)
    return embedder


def _first_pixel_embed(frames: list[npt.NDArray[np.uint8]]) -> npt.NDArray[np.float32]:
    """Encode each frame's fill value into its vector, so a vector names the row it came from.

    The all-ones stub cannot tell a correct assembly from a permuted one - every
    vector is identical. Carrying the frame's own marker through makes the pairing
    between an input row and its embedding observable.
    """
    matrix = np.zeros((len(frames), IMAGE_DIM), dtype=np.float32)
    for index, frame in enumerate(frames):
        matrix[index, 0] = float(frame[0, 0, 0])
    return matrix


def _marker_frame(marker: int) -> npt.NDArray[np.uint8]:
    """Return a small RGB frame filled with ``marker``, recoverable by ``_first_pixel_embed``."""
    return np.full((4, 4, 3), marker, dtype=np.uint8)


class _DescendingKeyReader:
    """Reader stand-in returning its position-keyed frames in descending key order.

    The reader builds its result in whatever order the reads finish in, so the
    embedder may not use iteration order to decide where a frame lands. Handing it
    the least favourable order makes any such dependency observable: an embedder
    that paired frames with rows by iteration order would invert the batch. Each
    frame's marker is its own URI, so a vector names the row it came from.
    """

    def read_many(self, uris: Sequence[str | None]) -> dict[int, npt.NDArray[np.uint8]]:
        """Return one marker frame per URI, inserted highest position first."""
        return {position: _marker_frame(int(uri)) for position, uri in reversed(list(enumerate(uris))) if uri}


def test_text_embedder_emits_one_row_per_input_row(monkeypatch: pytest.MonkeyPatch) -> None:
    """The text leg never drops a row: N source rows produce N group rows and no key column."""
    embedder = _stub_text_embedder(monkeypatch)
    batch = pa.table({"task_name": ["t1", "t2", "t3"], "subtask_name": ["s1", "s2", "s3"]})
    out = embedder(batch)
    assert out.num_rows == 3
    assert out.schema.names == list(TEXT_COLUMN_GROUP.field_names)


def test_text_embedder_returns_empty_table_for_empty_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    """A zero-row batch yields a zero-row table, not a vector-shape error from the builder."""
    embedder = _stub_text_embedder(monkeypatch)
    empty = pa.table({"task_name": pa.array([], pa.string()), "subtask_name": pa.array([], pa.string())})
    out = embedder(empty)
    assert out.num_rows == 0


def test_text_embedder_counts_blank_instructions(
    monkeypatch: pytest.MonkeyPatch, loguru_records: list[dict[str, Any]]
) -> None:
    """Blank instructions are kept but counted, so the false 'exact duplicate' signal is observable."""
    embedder = _stub_text_embedder(monkeypatch)
    batch = pa.table({"task_name": ["real task", "   "], "subtask_name": [None, "real subtask"]})
    out = embedder(batch)
    assert out.num_rows == 2
    warnings = [record["message"] for record in loguru_records if "blank instruction" in record["message"]]
    assert warnings
    assert "2 blank instruction(s)" in warnings[0]


def test_image_embedder_keeps_readable_and_nulls_unreadable(
    monkeypatch: pytest.MonkeyPatch, make_clip: Callable[..., None], tmp_path: pathlib.Path
) -> None:
    """A readable clip is embedded and an unreadable one becomes a NULL row (cardinality preserved).

    The old leg dropped the unreadable clip; the fill path must keep it as an
    all-NULL group so a rebuild replaces the row with NULL rather than leaving a
    stale prior vector. It also has to stay at its own position, because the worker
    attaches ``clip_id`` positionally.
    """
    good = tmp_path / "good.mp4"
    make_clip(good)
    embedder = _image_embedder_without_weights()

    def stub_embed(frames: list[npt.NDArray[np.uint8]]) -> npt.NDArray[np.float32]:
        return np.ones((len(frames), IMAGE_DIM), dtype=np.float32)

    monkeypatch.setattr(embedder, "_embed", stub_embed)
    batch = pa.table({"clip_uri": [str(good), str(tmp_path / "nope.mp4")]})
    out = embedder(batch)
    assert out.num_rows == 2
    assert out.schema.names == list(IMAGE_COLUMN_GROUP.field_names)
    assert out.column("embedding_image").is_valid().to_pylist() == [True, False]
    assert out.column("embedding_image_model_id").to_pylist() == ["stub", None]


def test_image_embedder_scatters_survivors_to_their_input_positions(
    monkeypatch: pytest.MonkeyPatch, make_clip: Callable[..., None], tmp_path: pathlib.Path
) -> None:
    """A readable clip surrounded by an unreadable one keeps its vector at its own row.

    Pins the scatter-back-to-position step: only the readable clips run through the
    backbone, so their vectors must land at their original row indices, not be
    packed contiguously at the front.
    """
    first = tmp_path / "first.mp4"
    third = tmp_path / "third.mp4"
    make_clip(first)
    make_clip(third)
    embedder = _image_embedder_without_weights()

    def stub_embed(frames: list[npt.NDArray[np.uint8]]) -> npt.NDArray[np.float32]:
        return np.ones((len(frames), IMAGE_DIM), dtype=np.float32)

    monkeypatch.setattr(embedder, "_embed", stub_embed)
    batch = pa.table({"clip_uri": [str(first), str(tmp_path / "missing.mp4"), str(third)]})
    out = embedder(batch)
    assert out.column("embedding_image").is_valid().to_pylist() == [True, False, True]
    assert out.column("embedding_image_model_id").to_pylist() == ["stub", None, "stub"]


def test_image_embedder_scatters_by_returned_position_not_by_iteration_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each frame lands at the position it is keyed under, whatever order it arrives in.

    The caller attaches ``clip_id`` positionally, so pairing frames with rows by the
    order the reader happened to build its result would pair every embedding with
    the wrong clip - a corruption that raises nothing, fails no schema check, and is
    invisible in the row counts. The fake reader returns its positions in descending
    order, so an embedder that trusted iteration order would invert the batch.
    """
    embedder = _image_embedder_without_weights()
    monkeypatch.setattr(embedder, "_reader", _DescendingKeyReader())
    monkeypatch.setattr(embedder, "_embed", _first_pixel_embed)

    out = embedder(pa.table({"clip_uri": ["1", "2", "3", "4"]}))

    markers = [vector[0] for vector in out.column("embedding_image").to_pylist()]
    assert markers == [1.0, 2.0, 3.0, 4.0]


def test_image_embedder_emits_a_null_group_per_row_when_no_clip_has_a_uri(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A batch with no readable URI yields one all-NULL row each.

    The leg is cardinality-preserving even when there is nothing to read, so the
    group batch still has to be built at full width from an empty read result.
    """
    embedder = _image_embedder_without_weights(read_concurrency=4)
    monkeypatch.setattr(embedder, "_embed", _first_pixel_embed)

    out = embedder(pa.table({"clip_uri": [None, ""]}))

    assert out.num_rows == 2
    assert out.column("embedding_image").to_pylist() == [None, None]


def test_image_embedder_output_is_identical_serial_and_concurrent(
    monkeypatch: pytest.MonkeyPatch, make_clip: Callable[..., None], tmp_path: pathlib.Path
) -> None:
    """The same batch yields the same table whether it is read serially or concurrently.

    Concurrency is a throughput change only: the two paths must agree on every
    emitted value, including which rows are NULL, so raising the width can never
    alter what a run writes.
    """
    clips = [tmp_path / f"clip{index}.mp4" for index in range(3)]
    for clip in clips:
        make_clip(clip)
    batch = pa.table({"clip_uri": [str(clips[0]), str(tmp_path / "missing.mp4"), str(clips[2])]})

    def embed_batch(read_concurrency: int) -> pa.Table:
        embedder = _image_embedder_without_weights(read_concurrency=read_concurrency)
        monkeypatch.setattr(embedder, "_embed", _first_pixel_embed)
        return embedder(batch)

    assert embed_batch(1).equals(embed_batch(len(clips)))


def test_image_embedder_nulls_an_unreadable_clip_at_its_own_row_when_reading_concurrently(
    monkeypatch: pytest.MonkeyPatch, make_clip: Callable[..., None], tmp_path: pathlib.Path
) -> None:
    """An unreadable clip in a concurrently read batch still occupies its own row, all-NULL.

    Cardinality and position survive the pool: the drop leaves a gap in the survivor
    set, and it is the input index - not the position within the survivors - that
    decides where each vector lands.
    """
    first = tmp_path / "first.mp4"
    third = tmp_path / "third.mp4"
    make_clip(first)
    make_clip(third)
    embedder = _image_embedder_without_weights(read_concurrency=3)
    monkeypatch.setattr(embedder, "_embed", _first_pixel_embed)

    out = embedder(pa.table({"clip_uri": [str(first), str(tmp_path / "missing.mp4"), str(third)]}))

    assert out.column("embedding_image").is_valid().to_pylist() == [True, False, True]
    assert out.column("embedding_image_model_id").to_pylist() == ["stub", None, "stub"]


def test_clip_frame_reader_drop_log_names_uri_and_carries_traceback(
    loguru_records: list[dict[str, Any]], tmp_path: pathlib.Path
) -> None:
    """The frame reader's drop site names the failing URI and attaches a traceback."""
    reader = ClipFrameReader(storage_profile="default")
    uri = str(tmp_path / "does-not-exist.mp4")
    assert reader.read(uri) is None
    drops = [record for record in loguru_records if "image open failed" in record["message"]]
    assert drops
    assert all(uri in record["message"] for record in drops)
    assert all(record["exception"] is not None for record in drops)


def test_clip_frame_reader_caches_transport_params_per_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each backend gets its own cached smart_open params; same-backend reads reuse."""
    resolved_uris: list[str] = []

    def fake_get_smart_open_params(uri: str, *, profile_name: str) -> dict[str, Any]:
        del profile_name
        resolved_uris.append(uri)
        return {"marker": uri}

    def fake_smart_open(uri: str, mode: str, **params: object) -> object:
        del mode, params
        # FileNotFoundError (an OSError subclass) is what is_missing_object_error
        # classifies as an expected missing object, so read() drops the row
        # (returns None) instead of re-raising. A bare OSError would be treated as
        # a systemic fault and propagate, which is not this test's concern.
        msg = f"stub open failed for {uri}"
        raise FileNotFoundError(msg)

    monkeypatch.setattr(frame_reader_module, "get_smart_open_params", fake_get_smart_open_params)
    monkeypatch.setattr(frame_reader_module.smart_open, "open", fake_smart_open)

    reader = ClipFrameReader(storage_profile="default")
    assert reader.read("s3://bucket-a/clip1.mp4") is None
    assert reader.read("s3://bucket-a/clip2.mp4") is None
    assert reader.read("s3://bucket-b/clip1.mp4") is None

    assert resolved_uris == ["s3://bucket-a/clip1.mp4", "s3://bucket-b/clip1.mp4"]


def test_unbridgeable_backend_fails_loudly_not_per_row(
    monkeypatch: pytest.MonkeyPatch, make_clip: Callable[..., None], tmp_path: pathlib.Path
) -> None:
    """A backend smart_open cannot bridge raises from the leg rather than nulling every row.

    ``ClipFrameReader`` resolves its transport params outside the per-clip drop
    handler, so a systemic misconfiguration fails the whole leg instead of
    silently emitting an all-NULL image group.
    """
    clip = tmp_path / "c.mp4"
    make_clip(clip)
    embedder = _image_embedder_without_weights()

    def unsupported(*_args: object, **_kwargs: object) -> dict[str, Any]:
        msg = "Unsupported StorageClient type for smart_open bridge: GCS"
        raise TypeError(msg)

    monkeypatch.setattr(frame_reader_module, "get_smart_open_params", unsupported)
    with pytest.raises(TypeError, match="Unsupported StorageClient"):
        embedder(pa.table({"clip_uri": [str(clip)]}))


def test_image_embedder_pins_torchvision_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """The DINOv2 processor loads with an explicit ``backend='torchvision'``.

    Left unpinned, the image processor's resize/resample backend is whichever
    library happens to be importable in the runtime env, so one ``model_id``
    yields different preprocessing -- and therefore different embeddings -- across
    environments. Running the real constructor against a fake ``transformers``
    module lets this assert the pin on CPU, with no weights.
    """
    recorded: dict[str, Any] = {}

    class _FakeProcessor:
        @staticmethod
        def from_pretrained(_weights_dir: str, **kwargs: object) -> "_FakeProcessor":
            recorded.update(kwargs)
            return _FakeProcessor()

    class _FakeModel:
        @staticmethod
        def from_pretrained(_weights_dir: str, **_kwargs: object) -> "_FakeModel":
            return _FakeModel()

        def to(self, _device: object) -> "_FakeModel":
            return self

        def eval(self) -> "_FakeModel":
            return self

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoImageProcessor = _FakeProcessor  # type: ignore[attr-defined]
    fake_transformers.AutoModel = _FakeModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    HfVisionImageEmbedder(spec=DEFAULT_IMAGE_MODEL, result_fingerprint="stub-fp", device="cpu")

    assert recorded.get("backend") == "torchvision"
    assert recorded.get("local_files_only") is True


@pytest.mark.parametrize(
    "columns",
    [SentenceTransformerTextEmbedder.SOURCE_COLUMNS, HfVisionImageEmbedder.SOURCE_COLUMNS],
)
def test_leg_source_columns_are_subset_of_read_projection(columns: tuple[str, ...]) -> None:
    """Each leg reads only columns the recipe projects from Lance.

    A leg column absent from the projection raises ``KeyError`` inside a GPU
    actor after the source is materialized -- the most expensive place to catch a
    typo; this pins it on the driver instead.
    """
    assert set(columns) <= set(EMBED_SOURCE_COLUMNS)


def test_text_leg_emits_unit_norm_task_and_subtask_vectors(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every emitted task and subtask vector is L2-unit-norm to float32 tolerance.

    Leg-boundary pin (rows in -> embedded rows out): asserts the emitted vectors,
    never a table URI / write mode / Lance schema, so it survives the storage
    change. Non-vacuous because ``_NormAwareEncoder`` returns far-from-unit vectors
    and only normalizes when the leg passes ``normalize_embeddings=True``; drop
    that flag and these norms stop being 1.0.
    """
    embedder = _norm_aware_text_embedder(monkeypatch)
    batch = pa.table(
        {
            "task_name": ["pick up the red block", "stack the two cubes"],
            "subtask_name": ["grasp", "lift and place"],
        }
    )
    out = embedder(batch)
    assert out.num_rows == 2
    for column in ("embedding_text_subtask", "embedding_text_task"):
        matrix = np.asarray(out.column(column).to_pylist(), dtype=np.float32)
        assert matrix.shape == (2, TEXT_DIM)
        np.testing.assert_allclose(np.linalg.norm(matrix, axis=1), np.ones(2), atol=1e-5)


def test_text_leg_emits_distinct_task_and_subtask_vectors_per_row(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each row carries two vectors, and the subtask vector differs from its task vector.

    Pins the two-vectors-per-clip contract at the leg boundary: a regression that
    embedded a single string for both fields (or swapped the columns onto one
    source) would collapse the two directions and trip this.
    """
    embedder = _norm_aware_text_embedder(monkeypatch)
    batch = pa.table({"task_name": ["pick up the red block"], "subtask_name": ["grasp the block firmly"]})
    out = embedder(batch)
    assert out.num_rows == 1
    subtask = np.asarray(out.column("embedding_text_subtask")[0].as_py(), dtype=np.float32)
    task = np.asarray(out.column("embedding_text_task")[0].as_py(), dtype=np.float32)
    assert not np.allclose(subtask, task)


def test_text_leg_records_its_result_fingerprint_on_every_row(monkeypatch: pytest.MonkeyPatch) -> None:
    """The text leg stamps the fingerprint it was constructed with on every emitted row.

    The stored value is what a later run's selection predicate compares to decide
    whether a row is current, so a blanked or omitted stamp makes a complete row
    indistinguishable from a stale one. It comes from the constructor argument
    rather than from the spec, which is why the fixture's fingerprint and its
    ``model_id`` are deliberately allowed to differ.
    """
    embedder = _norm_aware_text_embedder(monkeypatch)
    batch = pa.table({"task_name": ["t1", "t2", "t3"], "subtask_name": ["s1", "s2", "s3"]})
    out = embedder(batch)
    assert out.column("embedding_text_model_id").to_pylist() == ["bge-test-fingerprint"] * 3


def test_image_leg_records_its_result_fingerprint_on_every_embedded_row(
    monkeypatch: pytest.MonkeyPatch, make_clip: Callable[..., None], tmp_path: pathlib.Path
) -> None:
    """The image leg stamps the fingerprint it was constructed with on every survivor row.

    The image embedding's L2 normalization lives inside ``_embed`` (torch), so it
    is pinned by the env-gated real-model test, not here; this CPU pin covers the
    untested provenance seam by stubbing ``_embed`` and checking the per-row
    producer on the readable clips.
    """
    first = tmp_path / "first.mp4"
    second = tmp_path / "second.mp4"
    make_clip(first)
    make_clip(second)
    embedder = _image_embedder_without_weights()

    def stub_embed(frames: list[npt.NDArray[np.uint8]]) -> npt.NDArray[np.float32]:
        return np.ones((len(frames), IMAGE_DIM), dtype=np.float32)

    monkeypatch.setattr(embedder, "_embed", stub_embed)
    batch = pa.table({"clip_uri": [str(first), str(second)]})
    out = embedder(batch)
    assert out.num_rows == 2
    assert out.column("embedding_image_model_id").to_pylist() == ["stub", "stub"]
