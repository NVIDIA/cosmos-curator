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

"""Which PCA basis a run binds, across the three places one can come from.

The binding is the action leg's only pre-compute decision, and getting it wrong is
not an error the run reports: every worker projects onto whatever basis it is
handed, so a second basis in one group yields vectors silently incomparable with
the first basis's. These tests therefore pin the BINDING, not the projection.

None of them fit a basis. A fresh fit reads action artifacts through a Ray actor
pool, and all that needs pinning here is that resolution REACHED the fit instead
of binding something it should not have. A table whose clips carry no
``action_data_uri`` makes that observable as ``None``, because a bound basis always
returns an ``ActionPca``.

The artifact store is real. A stub would let a test pass while the fingerprint it
claims to have loaded names no artifact, and "the referenced basis is missing" is
one of the outcomes under test.
"""

import pathlib

import lance
import numpy as np
import pytest

from cosmos_curator.next.embeddings.action.pca import PcaArtifact, PcaArtifactStore, fit_action_pca
from cosmos_curator.next.embeddings.action.wrist_motion import DESCRIPTOR_DIM, DESCRIPTOR_VERSION
from cosmos_curator.next.embeddings.schemas import ACTION_DIM
from cosmos_curator.next.recipes.embeddings.action_pca import ActionPca, resolve_action_pca

from .conftest import EmbeddingsConfigFactory, vector_of, write_clips_dataset

_OTHER_DESCRIPTOR_VERSION = "wrist-motion-from-a-previous-contract"

# A syntactically valid fingerprint naming no artifact in any store.
_UNPERSISTED_FINGERPRINT = "0" * 64


def _fitted(seed: int) -> PcaArtifact:
    """Fit a small basis from deterministic pseudo-random descriptors."""
    generator = np.random.default_rng(seed)
    return fit_action_pca(generator.standard_normal((ACTION_DIM * 2, DESCRIPTOR_DIM)).astype(np.float32))


def _two_fingerprints() -> tuple[str, str]:
    """Return the two fixture bases' fingerprints, in fit order."""
    return (_fitted(0).fingerprint, _fitted(1).fingerprint)


@pytest.fixture
def basis_root(tmp_path: pathlib.Path) -> str:
    """Return an artifact root holding both fixture bases, loadable."""
    root = str(tmp_path / "bases")
    store = PcaArtifactStore(root, storage_profile="default")
    for seed in (0, 1):
        store.save_if_absent(_fitted(seed))
    return root


def _unembeddable_table(tmp_path: pathlib.Path) -> str:
    """Write a table no clip of which can be embedded, so a fresh fit has nothing to fit."""
    return write_clips_dataset(tmp_path, rows=2, action_uris=["", ""])


def _table_with_action_rows(
    tmp_path: pathlib.Path,
    *,
    fingerprints: list[str | None],
    descriptor_versions: list[str | None],
    action_uris: list[str | None] | None = None,
) -> str:
    """Write a clips table whose action group carries the given provenance per row."""
    rows = len(fingerprints)
    return write_clips_dataset(
        tmp_path,
        rows=rows,
        action_uris=action_uris,
        action={
            "action": [vector_of(index, ACTION_DIM) for index in range(rows)],
            "descriptor_version": descriptor_versions,
            "fingerprint": fingerprints,
        },
    )


def _resolve(
    uri: str, make_config: EmbeddingsConfigFactory, *, root_uri: str, staged: tuple[str, ...]
) -> ActionPca | None:
    """Resolve the basis a run over ``uri`` would bind."""
    return resolve_action_pca(
        lance.dataset(uri),
        make_config(clips_lance_uri=uri),
        root_uri=root_uri,
        staged_fingerprints=staged,
    )


def test_a_staged_fingerprint_binds_the_basis_without_a_fit(
    tmp_path: pathlib.Path, basis_root: str, make_embeddings_config: EmbeddingsConfigFactory
) -> None:
    """An attempt that crashed after staging must re-project onto the basis it staged.

    Those staged vectors are terminal - the next attempt publishes exactly them -
    so fitting a fresh basis would put two coordinate systems into one group with
    nothing reporting it.

    The group here is EMPTY, which is the state that would otherwise fit fresh, so
    only the staged fingerprint can produce this answer.
    """
    staged, _other = _two_fingerprints()
    uri = write_clips_dataset(tmp_path, rows=2)

    resolved = _resolve(uri, make_embeddings_config, root_uri=basis_root, staged=(staged,))

    assert resolved is not None
    assert resolved.fingerprint == staged
    # None marks a LOADED basis; an integer would mean this run refit.
    assert resolved.samples_used is None


def test_two_staged_fingerprints_refuse(
    tmp_path: pathlib.Path, basis_root: str, make_embeddings_config: EmbeddingsConfigFactory
) -> None:
    """Two bases have no safe automatic winner, so the run stops rather than picking one.

    Picking either would leave part of the staged set incomparable with the rest,
    and only the operator knows which half is worth keeping.
    """
    uri = write_clips_dataset(tmp_path, rows=2)

    with pytest.raises(ValueError, match="reference more than one action PCA basis"):
        _resolve(uri, make_embeddings_config, root_uri=basis_root, staged=_two_fingerprints())


def test_a_staged_and_a_canonical_basis_that_disagree_refuse(
    tmp_path: pathlib.Path, basis_root: str, make_embeddings_config: EmbeddingsConfigFactory
) -> None:
    """The staged rows and the published rows must agree before either can be trusted.

    Reachable when a basis was refit between two attempts. Publishing the staged
    rows anyway would mix projections inside one group, which is what the
    single-basis rule exists to prevent.
    """
    canonical, staged = _two_fingerprints()
    uri = _table_with_action_rows(
        tmp_path, fingerprints=[canonical, canonical], descriptor_versions=[DESCRIPTOR_VERSION] * 2
    )

    with pytest.raises(ValueError, match="disagree with the basis"):
        _resolve(uri, make_embeddings_config, root_uri=basis_root, staged=(staged,))


def test_a_staged_and_a_canonical_basis_that_agree_proceed(
    tmp_path: pathlib.Path, basis_root: str, make_embeddings_config: EmbeddingsConfigFactory
) -> None:
    """The ordinary wave case: a partial run resumes against the basis it already used.

    The negative control for the disagreement refusal above. Were the two sources
    compared by presence rather than by value, that test would still pass while
    this one - the common path - refused every resumed wave.
    """
    bound, _other = _two_fingerprints()
    uri = _table_with_action_rows(tmp_path, fingerprints=[bound, bound], descriptor_versions=[DESCRIPTOR_VERSION] * 2)

    resolved = _resolve(uri, make_embeddings_config, root_uri=basis_root, staged=(bound,))

    assert resolved is not None
    assert resolved.fingerprint == bound


def test_a_canonical_basis_alone_binds_the_basis(
    tmp_path: pathlib.Path, basis_root: str, make_embeddings_config: EmbeddingsConfigFactory
) -> None:
    """A group already published under one basis keeps it, so later rows stay comparable.

    Nothing is staged here, which is the state of any run starting cleanly against
    a partly embedded group.
    """
    bound, _other = _two_fingerprints()
    uri = _table_with_action_rows(tmp_path, fingerprints=[bound, None], descriptor_versions=[DESCRIPTOR_VERSION, None])

    resolved = _resolve(uri, make_embeddings_config, root_uri=basis_root, staged=())

    assert resolved is not None
    assert resolved.fingerprint == bound


def test_two_canonical_bases_at_the_current_contract_refuse(
    tmp_path: pathlib.Path, basis_root: str, make_embeddings_config: EmbeddingsConfigFactory
) -> None:
    """Nothing in the write path stops a group referencing two bases, so this read is the gate.

    It replaced a separate group-validation pass, so it carries that pass's whole
    remaining job: one basis is bindable and more than one is not.
    """
    first, second = _two_fingerprints()
    uri = _table_with_action_rows(tmp_path, fingerprints=[first, second], descriptor_versions=[DESCRIPTOR_VERSION] * 2)

    with pytest.raises(ValueError, match="reference more than one action PCA basis"):
        _resolve(uri, make_embeddings_config, root_uri=basis_root, staged=())


def test_rows_at_an_older_descriptor_version_neither_bind_nor_refuse(
    tmp_path: pathlib.Path, basis_root: str, make_embeddings_config: EmbeddingsConfigFactory
) -> None:
    """A descriptor-contract bump makes existing rows stale work, not a run-stopping fault.

    Two claims: the basis those rows name must NOT be bound - it belongs to
    descriptors this code no longer produces - and their presence must not raise,
    because they are rows an ordinary re-run recomputes.

    ``None`` is what proves the first half: the fingerprint IS loadable from the
    store, so binding it would have returned an artifact.
    """
    bound, _other = _two_fingerprints()
    uri = _table_with_action_rows(
        tmp_path,
        fingerprints=[bound, bound],
        descriptor_versions=[_OTHER_DESCRIPTOR_VERSION] * 2,
        action_uris=["", ""],
    )

    assert _resolve(uri, make_embeddings_config, root_uri=basis_root, staged=()) is None


def test_a_staged_fingerprint_naming_no_artifact_refuses(
    tmp_path: pathlib.Path, make_embeddings_config: EmbeddingsConfigFactory
) -> None:
    """A staged basis that is gone cannot be re-projected, and must not be quietly refit.

    Refitting would look like recovery while producing a different coordinate
    system for rows that are already staged and will be published as they are.
    """
    uri = _unembeddable_table(tmp_path)

    with pytest.raises(ValueError, match="is missing at"):
        _resolve(
            uri,
            make_embeddings_config,
            root_uri=str(tmp_path / "empty-bases"),
            staged=(_UNPERSISTED_FINGERPRINT,),
        )
