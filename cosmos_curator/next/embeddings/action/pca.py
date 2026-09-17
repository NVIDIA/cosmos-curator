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

"""Action-descriptor PCA basis: pure numpy SVD fit + a content-addressed store.

Reduces the ``DESCRIPTOR_DIM`` (600) dual-wrist descriptor to ``ACTION_DIM`` (97)
with a table-wide basis. ``fit_action_pca`` builds a ``PcaArtifact`` (pure numpy,
no scikit-learn) and ``PcaArtifact.project`` applies it.

``PcaArtifactStore`` persists each basis as an IMMUTABLE, content-addressed
object at ``<root>/<fingerprint>.npz`` (``action_pca_root_uri`` derives the root
directory beside ``clips.lance``). Content addressing decouples the artifact
write from the Lance commit: the basis is persisted before the commit and each
``clips.lance`` row references it by fingerprint, so a failed commit only leaves
a harmless unreferenced object - never new basis bytes paired with old action
vectors. An append reuses the fingerprint the existing rows already carry; a
rebuild fits a new basis under a new fingerprint. The basis records the
``descriptor_version`` it was fit under, so a later run whose descriptor
semantics differ refuses to reuse it.
"""

import datetime
import hashlib
import io

import attrs
import numpy as np
import numpy.typing as npt
from loguru import logger

from cosmos_curator.core.utils.storage.storage_client import StorageClient
from cosmos_curator.core.utils.storage.storage_utils import (
    StorageWriter,
    get_storage_client,
    path_exists,
    read_bytes,
)
from cosmos_curator.next.embeddings.action.wrist_motion import DESCRIPTOR_DIM, DESCRIPTOR_VERSION
from cosmos_curator.next.embeddings.schemas import ACTION_DIM

# A singular value below this fraction of the leading one spans a data-support-free
# (null-space) direction; used to report effective rank and reject a rank-deficient fit.
_RANK_REL_TOL = 1e-10

_EXPECTED_MEAN_SHAPE = (DESCRIPTOR_DIM,)
_MATRIX_NDIM = 2

# A retained-variance fraction below this floor means the fit population is likely
# unusual, not that ACTION_DIM is wrong. It sits below the ~90% design target that
# ACTION_DIM was chosen to hit on a reference population, so an ordinary fit does
# not warn: this is a "something looks wrong" floor, not a restatement of target.
_MIN_EXPLAINED_VARIANCE_WARN = 0.80

# The keys today's save() writes. explained_variance_ratio is deliberately NOT
# listed: it is optional on load so a pre-change 7-key artifact still validates
# (a required eighth key would silently reject every existing basis).
_REQUIRED_KEYS = (
    "mean",
    "components",
    "descriptor_dim",
    "n_components",
    "n_fit_rows",
    "descriptor_version",
    "fit_timestamp",
)

# Suffix appended to the clips lance_uri to name the sibling PCA artifact
# DIRECTORY. Owned here, beside the artifacts it names, so the module that fits and
# persists a basis is also the one that decides where the bases live.
# Each fitted basis is one immutable ``<fingerprint>.npz`` object in this directory.
ACTION_PCA_ROOT_SUFFIX = "__action_pca"


def action_pca_root_uri(clips_lance_uri: str) -> str:
    """Derive the content-addressed PCA artifact directory beside the clips table.

    The clips URI is slash-normalized first: without the ``rstrip`` a URI ending
    in ``/`` would nest the directory *inside* the clips dataset
    (``clips.lance/__action_pca``) instead of beside it (``clips.lance__action_pca``).
    """
    return f"{clips_lance_uri.rstrip('/')}{ACTION_PCA_ROOT_SUFFIX}"


def action_pca_artifact_uri(root_uri: str, fingerprint: str) -> str:
    """Return the immutable artifact URI ``<root>/<fingerprint>.npz`` for one basis."""
    return f"{root_uri.rstrip('/')}/{fingerprint}.npz"


@attrs.frozen(eq=False)  # numpy array fields: an attrs __eq__ would compare arrays and raise on truth value
class PcaArtifact:
    """Fitted action-PCA basis (numpy SVD; no scikit-learn object).

    Attributes:
        mean: ``(descriptor_dim,)`` descriptor mean subtracted before projection.
        components: ``(n_components, descriptor_dim)`` basis (one vector per row).
        descriptor_dim: Descriptor width the basis was fit on.
        n_components: Retained component count (rows of ``components``).
        n_fit_rows: Number of descriptors the basis was fit on.
        descriptor_version: Fingerprint of the descriptor SEMANTICS; a loader
            refuses a basis whose version differs from its own.
        fit_timestamp: ISO-8601 UTC time of the fit (empty when unknown).
        explained_variance_ratio: Cumulative variance the retained components
            capture, in ``[0, 1]``; ``nan`` for a pre-change artifact.
        fingerprint: Content hash of the basis (``mean`` + ``components`` bytes +
            ``descriptor_version``), derived once at construction and NEVER
            persisted. Stamped per row so a group accidentally built from two
            bases is detectable by its distinct fingerprint values. Two artifacts fingerprint equal
            iff they would project identically; any element or version change
            differs.

    """

    mean: npt.NDArray[np.float64]
    components: npt.NDArray[np.float64]
    descriptor_dim: int = DESCRIPTOR_DIM
    n_components: int = ACTION_DIM
    n_fit_rows: int = 0
    descriptor_version: str = DESCRIPTOR_VERSION
    fit_timestamp: str = ""
    explained_variance_ratio: float = float("nan")
    # Derived, not a constructor arg: computed in __attrs_post_init__ from the
    # other fields. A property would recompute per access; slotted attrs.frozen
    # cannot cache_property, so the value is materialized once here instead.
    fingerprint: str = attrs.field(init=False, default="")

    def __attrs_post_init__(self) -> None:
        """Freeze the numpy arrays and materialize the derived fingerprint once.

        One artifact instance is shared read-only by every worker that projects
        against it; ``attrs.frozen`` blocks rebinding the fields but not mutating
        the arrays they point at. Both construction paths (fit and load) run this
        hook, so both freeze the arrays and get the same fingerprint. The frozen
        ``fingerprint`` field is set via ``object.__setattr__`` (the only way to
        assign a frozen attrs field from post-init).
        """
        self.mean.setflags(write=False)
        self.components.setflags(write=False)
        object.__setattr__(self, "fingerprint", self._compute_fingerprint())

    def _compute_fingerprint(self) -> str:
        """Hash the C-contiguous ``mean`` / ``components`` bytes plus the version.

        Sequential ``update`` calls keep the mean/components boundary in the hash
        state, so two bases never collide by shifting a byte across it. The arrays
        are made C-contiguous first so a non-contiguous view (e.g. a slice) hashes
        to the same value as its dense equivalent.
        """
        digest = hashlib.sha256()
        digest.update(np.ascontiguousarray(self.mean).tobytes())
        digest.update(np.ascontiguousarray(self.components).tobytes())
        digest.update(self.descriptor_version.encode("utf-8"))
        return digest.hexdigest()

    def project(self, descriptors: npt.NDArray[np.floating]) -> npt.NDArray[np.float64]:
        """Project descriptors onto this basis (mean-centre, then rotate).

        Args:
            descriptors: A single ``(descriptor_dim,)`` descriptor or a batch
                ``(N, descriptor_dim)``.

        Returns:
            ``(n_components,)`` for a single descriptor, else ``(N, n_components)``.

        Raises:
            ValueError: If the shape is not ``(descriptor_dim,)`` or
                ``(N, descriptor_dim)``; a shape that merely broadcasts against
                the mean is rejected rather than projected to a wrong result.

        """
        matrix = np.asarray(descriptors, dtype=np.float64)
        if matrix.ndim not in {1, _MATRIX_NDIM} or matrix.shape[-1] != DESCRIPTOR_DIM:
            msg = f"descriptors must be ({DESCRIPTOR_DIM},) or (N, {DESCRIPTOR_DIM}), got shape {matrix.shape}"
            raise ValueError(msg)
        projected: npt.NDArray[np.float64] = (matrix - self.mean) @ self.components.T
        return projected


def fit_action_pca(descriptors: npt.NDArray[np.floating], n_components: int = ACTION_DIM) -> PcaArtifact:
    """Fit an ``n_components`` PCA basis over stacked descriptors via numpy SVD.

    Args:
        descriptors: ``(K, DESCRIPTOR_DIM)`` matrix of wrist descriptors (K clips).
        n_components: Principal components to keep (default ``ACTION_DIM``).

    Returns:
        The fitted ``PcaArtifact`` (logs retained variance, rank, and any low-variance warning).

    Raises:
        ValueError: If the request or sample cannot yield a valid basis (bad
            ``n_components``, wrong shape, too few rows, non-finite values, or
            zero/deficient rank).

    """
    if not 1 <= n_components <= DESCRIPTOR_DIM:
        msg = f"n_components must be in [1, {DESCRIPTOR_DIM}] (the descriptor width), got {n_components}"
        raise ValueError(msg)
    matrix = np.ascontiguousarray(descriptors, dtype=np.float64)
    if matrix.ndim != _MATRIX_NDIM:
        msg = f"descriptors must be a (K, {DESCRIPTOR_DIM}) 2-D matrix, got shape {matrix.shape}"
        raise ValueError(msg)
    num_rows, dim = matrix.shape
    if dim != DESCRIPTOR_DIM:
        msg = f"descriptors must be (K, {DESCRIPTOR_DIM}), got shape {matrix.shape}"
        raise ValueError(msg)
    if num_rows <= n_components:
        msg = (
            f"need > {n_components} descriptors to fit a {n_components}-component PCA "
            f"(mean-centering costs one degree of freedom), got {num_rows}"
        )
        raise ValueError(msg)
    if not np.isfinite(matrix).all():
        # Caught before the SVD so a non-finite input fails with an accurate
        # message rather than surfacing downstream as a misleading "rank-deficient"
        # error (a NaN sample yields NaN variance, which slips past the zero check).
        msg = f"descriptors contain non-finite values (NaN/inf) across {num_rows} rows; cannot fit a PCA basis"
        raise ValueError(msg)

    mean = matrix.mean(axis=0)
    _u, singular_values, right_vectors = np.linalg.svd(matrix - mean, full_matrices=False)
    # num_rows > n_components >= 1 guarantees num_rows >= 2, so the DOF divisor is >= 1.
    variances = (singular_values**2) / (num_rows - 1)
    total_variance = float(variances.sum())
    if total_variance <= 0.0:
        msg = f"action PCA fit is degenerate: {num_rows} descriptors have zero total variance (all rows identical)"
        raise ValueError(msg)
    leading = float(singular_values[0])
    rank = int(np.count_nonzero(singular_values > leading * _RANK_REL_TOL))
    smallest_retained = float(singular_values[n_components - 1])
    if rank < n_components:
        msg = (
            f"action PCA fit is rank-deficient: effective rank {rank} < {n_components} components from {num_rows} "
            f"descriptors (smallest retained singular value {smallest_retained:.3e}); the retained tail would be "
            "null-space noise, not signal"
        )
        raise ValueError(msg)

    # Clamp to 1.0: the ratio is <= 1 in exact arithmetic (a subset sum over the
    # same non-negative terms), but pairwise float summation can overshoot by
    # ~1e-13 on a near-full-rank retained fit, so pin the documented [0, 1] bound.
    explained = float(min(variances[:n_components].sum() / total_variance, 1.0))
    components = right_vectors[:n_components]
    # LAPACK fixes the singular subspaces but not each vector's sign; pin the
    # largest-magnitude entry of every component positive so a deliberate re-fit
    # of the same sample reproduces the same basis across BLAS builds for distinct
    # singular values (svd_flip; repeated/degenerate values are not pinned by sign).
    signs = np.sign(components[np.arange(n_components), np.abs(components).argmax(axis=1)])
    components = components * signs[:, np.newaxis]
    if explained < _MIN_EXPLAINED_VARIANCE_WARN:
        logger.warning(
            f"action PCA fit retains only {explained:.1%} variance in {n_components} components from {num_rows} "
            f"descriptors (below the {_MIN_EXPLAINED_VARIANCE_WARN:.0%} floor); the fit population may be unusual."
        )
    logger.info(
        f"action PCA fit: {n_components} components capture {explained:.1%} variance from {num_rows} descriptors "
        f"(target ~90%); effective rank {rank}, smallest retained singular value {smallest_retained:.3e}."
    )
    return PcaArtifact(
        mean=mean,
        components=np.ascontiguousarray(components),
        descriptor_dim=dim,
        n_components=n_components,
        n_fit_rows=num_rows,
        descriptor_version=DESCRIPTOR_VERSION,
        fit_timestamp=datetime.datetime.now(datetime.UTC).isoformat(),
        explained_variance_ratio=explained,
    )


def _from_archive(archive: np.lib.npyio.NpzFile) -> PcaArtifact:
    """Build a validated ``PcaArtifact`` from a loaded ``.npz`` archive.

    Converts an incompatible or foreign persisted basis into a driver-side
    ``ValueError`` naming what is wrong, before any Ray task projects a vector
    against it: a missing key, a descriptor width or component count that
    disagrees with this code, mismatched array shapes, or a stale descriptor
    version.

    Raises:
        ValueError: If a required key is absent; if ``descriptor_dim`` or
            ``n_components`` disagree with this code; if ``mean`` / ``components``
            do not match the expected shapes; if either array contains non-finite
            values; or if the recorded descriptor version differs from the loader's.

    """
    missing = [key for key in _REQUIRED_KEYS if key not in archive.files]
    if missing:
        msg = f"PCA artifact is missing required key(s) {missing}; present: {sorted(archive.files)}"
        raise ValueError(msg)

    stored_dim = int(archive["descriptor_dim"])
    if stored_dim != DESCRIPTOR_DIM:
        msg = (
            f"PCA artifact descriptor-width mismatch: this code produces {DESCRIPTOR_DIM}-d "
            f"descriptors but the basis records descriptor_dim={stored_dim}"
        )
        raise ValueError(msg)

    mean = archive["mean"]
    components = archive["components"]
    stored_components = int(archive["n_components"])
    if stored_components != ACTION_DIM:
        msg = (
            f"PCA artifact component-count mismatch: this code produces {ACTION_DIM}-d action embeddings "
            f"but the basis retains {stored_components} components"
        )
        raise ValueError(msg)
    expected_components_shape = (stored_components, DESCRIPTOR_DIM)
    if mean.shape != _EXPECTED_MEAN_SHAPE or components.shape != expected_components_shape:
        msg = (
            f"PCA artifact shape mismatch: expected mean {_EXPECTED_MEAN_SHAPE} and components "
            f"{expected_components_shape} (from recorded n_components={stored_components}), got mean {mean.shape} "
            f"and components {components.shape}"
        )
        raise ValueError(msg)

    stored_version = str(archive["descriptor_version"])
    if stored_version != DESCRIPTOR_VERSION:
        msg = (
            f"PCA artifact descriptor-version mismatch: this code produces {DESCRIPTOR_VERSION!r} descriptors "
            f"but the basis was fit under {stored_version!r}. Its vectors are dimensionally valid but "
            "semantically incompatible; re-fit the basis or point at the matching artifact."
        )
        raise ValueError(msg)
    if not np.isfinite(mean).all():
        msg = "PCA artifact validation failed: mean contains non-finite values (NaN/inf)"
        raise ValueError(msg)
    if not np.isfinite(components).all():
        msg = "PCA artifact validation failed: components contain non-finite values (NaN/inf)"
        raise ValueError(msg)
    ratio = float(archive["explained_variance_ratio"]) if "explained_variance_ratio" in archive.files else float("nan")
    return PcaArtifact(
        mean=mean,
        components=components,
        descriptor_dim=stored_dim,
        n_components=stored_components,
        n_fit_rows=int(archive["n_fit_rows"]),
        descriptor_version=stored_version,
        fit_timestamp=str(archive["fit_timestamp"]),
        explained_variance_ratio=ratio,
    )


def _from_archive_bytes(data: bytes) -> PcaArtifact:
    """Build a validated ``PcaArtifact`` from serialized ``.npz`` bytes.

    One decode-and-validate path shared by every read (load and save read-back),
    so the archive format has a single place to evolve.
    """
    with np.load(io.BytesIO(data)) as archive:
        return _from_archive(archive)


class PcaArtifactStore:
    """Content-addressed reader/writer for immutable ``.npz`` PCA bases under one root.

    Each basis lives at ``<root>/<fingerprint>.npz`` (``action_pca_artifact_uri``),
    so a basis is named by exactly the content that would project identically. The
    store owns the ``(root_uri, storage_profile)`` pair and a lazily-resolved
    storage client reused across every artifact under the root.
    """

    def __init__(self, root_uri: str, *, storage_profile: str = "default") -> None:
        """Initialize the store for one artifact root directory.

        Args:
            root_uri: Directory URI holding the ``<fingerprint>.npz`` bases (local,
                s3/gs/az); see ``action_pca_root_uri``.
            storage_profile: Profile for remote access.

        """
        self._root_uri = root_uri
        self._storage_profile = storage_profile
        self._client: StorageClient | None = None
        self._resolved = False

    def _get_client(self) -> StorageClient | None:
        """Resolve (once) the storage client for the root (``None`` for local).

        Every artifact under the root shares one backend, so one client resolved
        from the root URI serves every ``load`` / ``save_if_absent`` call.
        """
        if not self._resolved:
            self._client = get_storage_client(self._root_uri, profile_name=self._storage_profile)
            self._resolved = True
        return self._client

    def load(self, fingerprint: str) -> PcaArtifact:
        """Load and validate the immutable basis named by ``fingerprint``.

        Args:
            fingerprint: Content fingerprint a ``clips.lance`` action row references.

        Returns:
            The validated ``PcaArtifact`` at ``<root>/<fingerprint>.npz``.

        Raises:
            ValueError: If no artifact exists at that URI (a referenced basis was
                deleted or never persisted); if the archive is incompatible (see
                ``_from_archive``); or if the loaded basis's own content
                fingerprint does not equal ``fingerprint`` (the object's bytes do
                not match the name that referenced them).

        """
        uri = action_pca_artifact_uri(self._root_uri, fingerprint)
        client = self._get_client()
        if not path_exists(uri, client):
            msg = (
                f"action PCA basis {fingerprint!r} is missing at {uri}; the immutable artifact was deleted "
                "or never persisted. Restore it - the name is the hash of its own bytes, so any copy of "
                "those bytes is the same basis. Refitting is not an alternative while rows still reference "
                "this one: a fresh basis would project them into an incomparable space. If the reference "
                "came from an interrupted run's staging workspace rather than from the published rows, "
                "discarding that workspace drops it."
            )
            raise ValueError(msg)
        pca = _from_archive_bytes(read_bytes(uri, client))
        if pca.fingerprint != fingerprint:
            msg = (
                f"action PCA basis at {uri} has content fingerprint {pca.fingerprint!r}, which does not match the "
                f"requested {fingerprint!r}; the artifact bytes do not match the name that referenced them."
            )
            raise ValueError(msg)
        return pca

    def save_if_absent(self, pca: PcaArtifact) -> str:
        """Persist ``pca`` at its content-addressed URI unless already present; return the URI.

        Content addressing makes the write idempotent and safe to precede the Lance
        commit: an already-present object is loaded and its fingerprint verified (a
        matching basis is a no-op; a divergent one at the same name is a corruption
        error), and a new object is written, read back, and fully validated before
        any Lance row can reference it.

        Returns:
            The ``<root>/<fingerprint>.npz`` URI the action rows should reference.

        Raises:
            ValueError: If an object already exists at the URI with a different
                content fingerprint, or a fresh write does not round-trip.

        """
        uri = action_pca_artifact_uri(self._root_uri, pca.fingerprint)
        client = self._get_client()
        if path_exists(uri, client):
            existing = _from_archive_bytes(read_bytes(uri, client))
            if existing.fingerprint != pca.fingerprint:
                msg = (
                    f"action PCA artifact at {uri} already exists with a different content fingerprint "
                    f"{existing.fingerprint!r} than the basis being saved ({pca.fingerprint!r}); refusing to overwrite."
                )
                raise ValueError(msg)
            logger.info(f"action PCA basis {pca.fingerprint[:12]} already present at {uri}; reusing")
            return uri
        self._write(uri, pca)
        readback = _from_archive_bytes(read_bytes(uri, client))
        if readback.fingerprint != pca.fingerprint:
            msg = (
                f"action PCA artifact write to {uri} did not round-trip: read-back fingerprint "
                f"{readback.fingerprint!r} != {pca.fingerprint!r}"
            )
            raise ValueError(msg)
        logger.info(
            f"wrote action PCA basis {pca.fingerprint[:12]} (components {pca.components.shape}, "
            f"version {pca.descriptor_version!r}, fit on {pca.n_fit_rows} rows) to {uri}"
        )
        return uri

    def _write(self, uri: str, pca: PcaArtifact) -> None:
        """Serialize ``pca`` (arrays + provenance) as a single ``.npz`` object at ``uri``.

        Provenance (dims, fit population, descriptor version, timestamp) rides in
        the same archive so the basis is self-describing and a later load can
        reject an incompatible basis without a sidecar file.
        """
        buffer = io.BytesIO()
        np.savez(
            buffer,
            mean=pca.mean,
            components=pca.components,
            descriptor_dim=np.int64(pca.descriptor_dim),
            n_components=np.int64(pca.n_components),
            n_fit_rows=np.int64(pca.n_fit_rows),
            descriptor_version=np.asarray(pca.descriptor_version),
            fit_timestamp=np.asarray(pca.fit_timestamp),
            explained_variance_ratio=np.float64(pca.explained_variance_ratio),
        )
        StorageWriter(uri, profile_name=self._storage_profile).write(buffer.getvalue())
