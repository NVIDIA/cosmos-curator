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

"""Coverage for the driver-side translation of config into a modality's fill spec.

A modality's tuning knobs are validated on the driver but consumed inside a Ray
worker, so a value that stops at the config object is silently ignored by the run
that was tuned. These tests assert the translation itself - a set value reaches the
worker's constructor arguments or its actor shape - and deliberately never assert a
default, which is a tuning decision that should be free to change.

``ModalityFill`` types the embedder as a bare ``type`` and its arguments as
``dict[str, Any]``, so the ``embedder_cls(**embedder_kwargs)`` the worker performs
is unchecked by mypy. One test per modality therefore binds the shipped kwargs
against the named constructor's real signature, which is the only place that
mismatch can be caught before a Ray worker raises it.

The result-fingerprint tests are a different kind: the fingerprint is stamped on
every row of a corpus and later compared to decide whether that row must be
recomputed, so it is a published value rather than an internal one. Its
expectations are therefore hex literals, and a legitimate re-pin of the weights
is SUPPOSED to turn them red - that red is the reminder that the corpus needs
rebuilding.
"""

import inspect
from collections.abc import Callable

import attrs
import numpy as np
import pytest

from cosmos_curator.models.registry import RegisteredModel
from cosmos_curator.next.embeddings.action.pca import PcaArtifact
from cosmos_curator.next.embeddings.action.wrist_motion import DESCRIPTOR_DIM, DESCRIPTOR_VERSION
from cosmos_curator.next.embeddings.model_specs import (
    DEFAULT_IMAGE_MODEL,
    DEFAULT_TEXT_MODEL,
    ModelSpec,
    VisionPooling,
)
from cosmos_curator.next.embeddings.schemas import (
    ACTION_COLUMN_GROUP,
    ACTION_DIM,
    IMAGE_COLUMN_GROUP,
    TEXT_COLUMN_GROUP,
)
from cosmos_curator.next.recipes.embeddings.config import EmbeddingPipelineConfig, Modality
from cosmos_curator.next.recipes.embeddings.modalities import (
    ModalityFill,
    WorkerResources,
    _model_result_fingerprint,
    build_action_fill,
    build_image_fill,
    build_text_fill,
)

from .conftest import EmbeddingsConfigFactory

# The fingerprints the shipped specs and registry produce today. Literals, never
# re-derived from the payload under test: an expectation computed the same way as
# the value proves only that the function is deterministic, and would follow any
# change to the payload silently - which is the one thing this value must not do.
_TEXT_RESULT_FINGERPRINT = "baa06ef6e74c42d0c4809fba8cf3782fc30d3f2bec668c7a66d2e478f4037acf"
_IMAGE_RESULT_FINGERPRINT = "918d00b3d79653a582796851942b617ad395a71bf31ca5273bd370d548587bbb"

# A test-owned registry record, so the per-input tests below vary one registry
# fact at a time without depending on what the shipped registry currently pins.
_REGISTERED = RegisteredModel(key="stub_key", model_id="stub/model", revision="0" * 40, precision="FP32")


def _synthetic_pca() -> PcaArtifact:
    """Return a zero basis of the right shape; the binding test never projects with it."""
    return PcaArtifact(
        mean=np.zeros(DESCRIPTOR_DIM, dtype=np.float64),
        components=np.zeros((ACTION_DIM, DESCRIPTOR_DIM), dtype=np.float64),
    )


def _fingerprint(monkeypatch: pytest.MonkeyPatch, spec: ModelSpec, registered: RegisteredModel) -> str:
    """Return ``spec``'s fingerprint with the registry resolving to ``registered``.

    Patches the name ``modalities`` itself calls, with ``raising=True`` so a rename
    of the resolver fails here rather than leaving these tests measuring the real
    registry while claiming to control it.
    """
    monkeypatch.setattr(
        "cosmos_curator.next.recipes.embeddings.modalities.resolve_registered_model",
        lambda _key: registered,
        raising=True,
    )
    return _model_result_fingerprint(spec)


_FILL_BUILDERS: dict[str, Callable[[EmbeddingPipelineConfig], ModalityFill]] = {
    "action": lambda config: build_action_fill(config, _synthetic_pca()),
    "image": build_image_fill,
    "text": build_text_fill,
}


def test_image_fill_carries_the_configured_read_width_to_the_worker(
    make_embeddings_config: EmbeddingsConfigFactory,
) -> None:
    """The configured read concurrency arrives in the embedder's constructor arguments.

    The knob only has an effect inside the worker, so a value that never leaves the
    config would leave every run reading serially while reporting the tuned width.
    """
    config = make_embeddings_config(image={"read_concurrency": 7})

    assert build_image_fill(config).embedder_kwargs["read_concurrency"] == 7


def test_action_fill_carries_the_configured_read_width_to_the_worker(
    make_embeddings_config: EmbeddingsConfigFactory,
) -> None:
    """The configured read concurrency arrives in the action reader's constructor config.

    The width is consumed inside the worker's extract call, so a value that stops
    at the recipe config would leave every run fetching artifacts one at a time
    while reporting the tuned width.
    """
    config = make_embeddings_config(action={"read_concurrency": 7})

    fill = build_action_fill(config, _synthetic_pca())

    assert fill.embedder_kwargs["config"].read_concurrency == 7


def test_image_fill_carries_the_configured_cpu_reservation_to_the_actor_shape(
    make_embeddings_config: EmbeddingsConfigFactory,
) -> None:
    """The configured CPU reservation arrives in the worker's resource request.

    Only a value that reaches ``WorkerResources`` is passed to Ray. Left in the
    config the actor would be scheduled with no CPU reservation at all, which is
    the oversubscription the field exists to prevent.
    """
    config = make_embeddings_config(image={"num_cpus": 3.0})

    assert build_image_fill(config).resources.num_cpus == 3.0


@pytest.mark.parametrize("modality", sorted(_FILL_BUILDERS))
def test_fill_kwargs_satisfy_the_embedder_constructor(
    modality: str, make_embeddings_config: EmbeddingsConfigFactory
) -> None:
    """Every kwarg a fill ships binds to a real parameter of the embedder it names.

    The worker calls ``embedder_cls(**embedder_kwargs)`` through untyped fields, so
    a rename on either side is invisible until a Ray actor raises ``TypeError``
    after loading its model. Binding the signature covers every kwarg the fill
    carries, including ones added later, rather than one name at a time.
    """
    fill = _FILL_BUILDERS[modality](make_embeddings_config())

    inspect.signature(fill.embedder_cls).bind(**fill.embedder_kwargs)


def test_the_text_result_fingerprint_is_the_pinned_value() -> None:
    """The shipped text spec and registry entry produce exactly the recorded digest.

    Red here means the value stamped on text rows has changed, so every existing
    text row is now stale. That is a legitimate outcome of a deliberate change
    (a re-pin, a reformatter, a ``result_version`` bump) and requires a corpus
    rebuild; it is a defect only if nothing about the computation changed.
    """
    assert _model_result_fingerprint(DEFAULT_TEXT_MODEL) == _TEXT_RESULT_FINGERPRINT


def test_the_image_result_fingerprint_is_the_pinned_value() -> None:
    """The shipped image spec and registry entry produce exactly the recorded digest.

    See the text counterpart: red means every existing image row became stale.
    The two digests are pinned separately because a change must be attributable
    to one modality - a single combined pin could not say which moved.
    """
    assert _model_result_fingerprint(DEFAULT_IMAGE_MODEL) == _IMAGE_RESULT_FINGERPRINT


def test_a_changed_registry_model_id_moves_the_fingerprint(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pointing a weights name at a different repository changes the fingerprint.

    This is the case the fingerprint replaced a bare model id to keep covering:
    different weights must never read as the same computation.
    """
    pinned = _fingerprint(monkeypatch, DEFAULT_TEXT_MODEL, _REGISTERED)

    moved = _fingerprint(monkeypatch, DEFAULT_TEXT_MODEL, attrs.evolve(_REGISTERED, model_id="other/model"))

    assert moved != pinned


def test_a_changed_registry_revision_moves_the_fingerprint(monkeypatch: pytest.MonkeyPatch) -> None:
    """Re-pinning the same repository to a new commit changes the fingerprint.

    The whole reason ``revision`` is read from the registry rather than the spec:
    a re-pin is made by someone editing model staging, not embedding compute, so
    no manual lever in this package would be reached. Only this input notices.
    """
    pinned = _fingerprint(monkeypatch, DEFAULT_TEXT_MODEL, _REGISTERED)

    moved = _fingerprint(monkeypatch, DEFAULT_TEXT_MODEL, attrs.evolve(_REGISTERED, revision="1" * 40))

    assert moved != pinned


def test_a_changed_pooling_moves_the_image_fingerprint(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reducing the same backbone's output differently changes the fingerprint.

    Same weights, same revision, entirely different vectors - the case no
    checkpoint identifier can distinguish.
    """
    pinned = _fingerprint(monkeypatch, DEFAULT_IMAGE_MODEL, _REGISTERED)

    moved = _fingerprint(monkeypatch, attrs.evolve(DEFAULT_IMAGE_MODEL, pooling=VisionPooling.CLS_TOKEN), _REGISTERED)

    assert moved != pinned


def test_a_changed_result_version_moves_the_fingerprint(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bumping the manual lever changes the fingerprint with every other input fixed.

    It is the only input that can express a code-only semantic change - a new text
    format, a different representative frame - so if it did not move the digest,
    no such change could ever trigger a rebuild.
    """
    pinned = _fingerprint(monkeypatch, DEFAULT_TEXT_MODEL, _REGISTERED)

    moved = _fingerprint(monkeypatch, attrs.evolve(DEFAULT_TEXT_MODEL, result_version=2), _REGISTERED)

    assert moved != pinned


def test_a_changed_weights_name_moves_the_fingerprint(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two registry keys resolving to one model id and revision must not share a fingerprint.

    The registry admits exactly that pair: ``seedvr2_7b`` and ``seedvr2_7b_sharp``
    agree on model_id, version and precision and name different weight files. The
    registry record is held FIXED here, so only the spec's key can move the
    digest - which is the case a sibling swap would present.

    It is the key that selects the files: both embedders load
    ``get_local_dir_for_weights_name(spec.weights_name)``, so two keys are two
    directories however much of the registry entry they share.
    """
    pinned = _fingerprint(monkeypatch, DEFAULT_TEXT_MODEL, _REGISTERED)

    moved = _fingerprint(monkeypatch, attrs.evolve(DEFAULT_TEXT_MODEL, weights_name="sibling_key"), _REGISTERED)

    assert moved != pinned


def test_the_registry_precision_leaves_the_fingerprint_unmoved(monkeypatch: pytest.MonkeyPatch) -> None:
    """The one registry fact the embedders never read stays out of the fingerprint.

    Not a defended permissiveness: neither embedder passes a dtype, so including
    it would invalidate a whole corpus for a change that cannot alter one vector.
    If an embedder ever starts requesting a precision, this test is what fails and
    says to re-include the field.
    """
    pinned = _fingerprint(monkeypatch, DEFAULT_TEXT_MODEL, _REGISTERED)

    assert _fingerprint(monkeypatch, DEFAULT_TEXT_MODEL, attrs.evolve(_REGISTERED, precision="FP16")) == pinned


@pytest.mark.parametrize(
    ("builder", "provenance_column"),
    [
        (build_text_fill, TEXT_COLUMN_GROUP.provenance_columns[0]),
        (build_image_fill, IMAGE_COLUMN_GROUP.provenance_columns[0]),
    ],
)
def test_a_model_fill_selects_on_the_same_value_its_worker_stamps(
    builder: Callable[[EmbeddingPipelineConfig], ModalityFill],
    provenance_column: str,
    make_embeddings_config: EmbeddingsConfigFactory,
) -> None:
    """The identity the predicate compares and the value the rows carry are one value.

    Two independent derivations would agree today and could drift later, and the
    drift is silent in the worst possible way: every row would be written with a
    value the next run's predicate does not recognize, so the corpus would be
    recomputed in full on every run while reporting success.
    """
    fill = builder(make_embeddings_config())

    assert fill.result_identity[provenance_column] == fill.embedder_kwargs["result_fingerprint"]


def test_the_action_fill_identity_names_the_descriptor_version_and_the_bound_basis(
    make_embeddings_config: EmbeddingsConfigFactory,
) -> None:
    """Action's identity is its two existing facts, taken from the basis it was given.

    The fingerprint comes from the artifact argument rather than from a re-load, so
    a run cannot select rows against one basis and then project with another.
    """
    pca = _synthetic_pca()

    fill = build_action_fill(make_embeddings_config(), pca)

    assert fill.result_identity == {
        "embedding_action_descriptor_version": DESCRIPTOR_VERSION,
        "embedding_action_pca_fingerprint": pca.fingerprint,
    }


def test_a_fill_whose_identity_misses_a_provenance_column_is_refused() -> None:
    """An identity that does not cover every provenance column is rejected at construction.

    A missing key would narrow the selection predicate to the columns that happen
    to be present, so a row stale only in the uncovered column would read as
    current forever. Uses the action group because it is the one with two columns,
    which is the only shape where a partial identity is expressible.
    """
    with pytest.raises(ValueError, match="must be exactly"):
        ModalityFill(
            modality=Modality.ACTION,
            group=ACTION_COLUMN_GROUP,
            source_columns=("clip_id",),
            result_identity={"embedding_action_descriptor_version": DESCRIPTOR_VERSION},
            applicability_filter=None,
            embedder_cls=object,
            embedder_kwargs={},
            resources=WorkerResources(scan_batch_size=1),
            weights_name=None,
        )
