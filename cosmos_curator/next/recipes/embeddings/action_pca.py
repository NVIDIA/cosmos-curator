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

"""Action PCA basis: bind one basis per run before compute, then judge what the fill produced.

Action is the only modality needing a decision BEFORE any worker exists - which PCA
basis to project onto - and the only one whose outcome is judged afterwards. Both
are driver-side functions over a dataset the CALLER opened::

    resolve_action_pca()  ->  ActionPca | None      (driver, once per run)
        |
        |  reuse: take the one fingerprint referenced by the STAGED rows of an
        |         interrupted attempt or by the group's CANONICAL rows at the
        |         current descriptor contract, and load that exact basis
        |
        |  first fit: narrow action_data_uri scan -> rank by sha256(uri) ->
        |         extract only the bounded candidate set -> keep the
        |         smallest-rank valid descriptors -> fit -> save
        v
    the fill projects every selected row through that one basis
        |
        v
    check_action_outcome()  ->  per-ROW total-outage error (the fill itself
                                refuses the per-FRAGMENT one)

Every worker in a run must project onto the SAME basis or the group's vectors are
mutually incomparable. That single-basis rule is why the choice is made once here, on
the driver, and then travels as data inside the fill spec.

- Applicability is Mecka-only: a clip is applicable if it carries an
  ``action_data_uri`` (a Mecka ACT2 action artifact). There is no dataset-registry
  gate - a non-dexterous or malformed artifact is not filtered out but rejected per
  row by the extractor's geometry checks, yielding a NULL group that is retried next
  run rather than being silently excluded from the denominator.
- Basis selection is a predicate on what already exists, not a mode the operator
  picks. A basis referenced by rows STAGED but not yet published counts exactly as
  one referenced by the group's CANONICAL rows, because staged rows are terminal:
  an attempt that crashed between staging and publication will publish exactly the
  vectors it staged, so a group that looks empty may already be committed. Two
  bases across those sources refuse; only with none is a fresh one fit and
  persisted under its own fingerprint. Reading staged rows this way rests on the
  single-writer precondition stated in ``workspace.py``. ``max_fragments`` does
  not bias the basis: the candidate scan reads the whole table's
  ``action_data_uri`` column, so a truncated run fits from the same population an
  untruncated one would.
- Rows recorded under an older ``descriptor_version`` neither supply a basis nor
  refuse the run: they are stale results, which the ordinary pending filter selects
  for recompute. The canonical fingerprint read is therefore scoped to the CURRENT
  contract, so an old contract's basis can neither be loaded nor block a fresh fit.

Sampling by URI BEFORE extraction (rather than extracting the whole corpus and then
sampling) is what removes the old full-corpus materialization barrier. Because some
sampled artifacts fail to read, the fitted population - and therefore the fingerprint
- is not the one the previous full-corpus approach would have produced. Bases are
immutable and content-addressed, so that is a different basis, not a corrupted one.
"""

import hashlib
import heapq
from collections.abc import Iterable, Iterator, Sequence
from typing import cast

import attrs
import lance
import numpy as np
import numpy.typing as npt
import pyarrow as pa
import ray
from loguru import logger
from ray.data import ActorPoolStrategy

from cosmos_curator.core.utils.pixi_runtime_envs import ray_data_gpu_runtime_env
from cosmos_curator.next.embeddings.action.embedder import (
    DualWristMotionDescriptorExtractor,
    DualWristMotionReadConfig,
    decode_descriptors,
)
from cosmos_curator.next.embeddings.action.pca import PcaArtifact, PcaArtifactStore, fit_action_pca
from cosmos_curator.next.embeddings.action.wrist_motion import DESCRIPTOR_DIM, DESCRIPTOR_VERSION
from cosmos_curator.next.embeddings.schemas import ACTION_COLUMN_GROUP, ACTION_DIM
from cosmos_curator.next.recipes.embeddings.columns import count_filled, scan_column
from cosmos_curator.next.recipes.embeddings.config import EmbeddingPipelineConfig
from cosmos_curator.next.recipes.embeddings.modalities import (
    ACTION_APPLICABILITY_FILTER,
    ModalityResult,
    WorkerResources,
)
from cosmos_curator.next.recipes.embeddings.publication import staged_distinct_values
from cosmos_curator.next.recipes.embeddings.workspace import EmbeddingWorkspace
from cosmos_curator.next.utils.identity import canonical_digest
from cosmos_curator.next.utils.lance_fragment_recovery import sql_string_literal
from cosmos_curator.next.utils.lance_utils import distinct_non_null_values

# The action group's two provenance columns, unpacked once by position from the
# column-group definition so their order is asserted at import (a reordering there
# fails loudly here rather than silently swapping version and fingerprint reads).
_DESCRIPTOR_VERSION_COLUMN, _PCA_FINGERPRINT_COLUMN = ACTION_COLUMN_GROUP.provenance_columns

# Distinct provenance values read when binding the basis. Two is the whole question:
# one basis is bindable and more than one is not, so a third value would change no
# decision while costing a wider scan. This read IS the multi-producer gate - nothing
# in the write path prevents a group from referencing two bases.
_PROVENANCE_VALUES_READ = 2

# Rows per batch of the driver's narrow ``action_data_uri`` candidate scan. Larger
# than a compute batch because the batch holds only URI strings and never enters a
# model; it only bounds how much of the scan is buffered at a time.
_URI_SCAN_BATCH_SIZE = 16_384

# Candidate URIs extracted per requested fit sample. The candidate set is an
# OVERSAMPLE: sampling happens before extraction, so an unreadable or geometrically
# rejected artifact removes a candidate from the fit. Extracting twice the target
# absorbs that loss without a second pass, and costs at most one extra artifact read
# per kept descriptor.
_PCA_CANDIDATE_OVERSAMPLE = 2


# eq=False: PcaArtifact holds numpy arrays and is itself eq=False, so an
# attrs-generated __eq__ here would compare two bases by object identity while
# reading like a content comparison. Identity is what this wrapper actually means.
@attrs.frozen(eq=False)
class ActionPca:
    """The one PCA basis a run binds, and how it came to be bound.

    Attributes:
        artifact: The immutable basis every worker of the run projects onto.
        samples_used: Distinct descriptors the basis was fit on, or ``None`` when
            the basis was LOADED rather than fit. That single field is the whole
            encoding of the first-fit / reuse distinction the operator is shown.

    """

    artifact: PcaArtifact
    samples_used: int | None

    @property
    def fingerprint(self) -> str:
        """Return the basis's content fingerprint, stamped on every row it projects."""
        return self.artifact.fingerprint


def action_recovery_contract() -> str:
    """Return the digest keying action's staging namespace, knowable BEFORE a basis is bound.

    Action cannot key its namespace on its result identity the way text and image
    do: that identity carries the basis fingerprint, and the fingerprint is the
    very thing a pre-basis resume reads the namespace to FIND. The descriptor
    contract is the part of the identity that is fixed by code rather than
    discovered during the run - what a descriptor MEANS plus both widths it is
    projected between - so it separates incompatible staged rows with no
    circularity. Two runs at one contract may still bind different bases; the
    single-basis check, not the path, is what refuses that.
    """
    return canonical_digest(
        {
            "descriptor_version": DESCRIPTOR_VERSION,
            "action_dim": ACTION_DIM,
            "descriptor_dim": DESCRIPTOR_DIM,
        }
    )


def staged_action_fingerprints(workspace: EmbeddingWorkspace) -> tuple[str, ...]:
    """Return the distinct PCA bases an interrupted attempt's staged rows reference.

    Separate from :func:`resolve_action_pca` because the two answer different
    questions: this one reads a workspace the DRIVER resolved, while the resolver
    decides what to bind from values alone. Keeping the read here is what keeps
    the fingerprint column and the two-value ceiling owned by the module that
    knows why two is enough - a caller passing its own ceiling could silently
    widen or defeat the single-basis check.

    Args:
        workspace: Action's namespace for this run's descriptor contract. One
            that was never staged into yields no values.

    """
    return staged_distinct_values(workspace, _PCA_FINGERPRINT_COLUMN, max_values=_PROVENANCE_VALUES_READ)


def resolve_action_pca(
    dataset: lance.LanceDataset,
    config: EmbeddingPipelineConfig,
    *,
    root_uri: str,
    staged_fingerprints: tuple[str, ...],
) -> ActionPca | None:
    """Bind the one PCA basis every action worker projects onto: reuse a bound one, or fit it.

    Args:
        dataset: The clips table, already open at the version to read.
        config: Storage profile plus the action sampling knobs.
        root_uri: Content-addressed directory holding every fitted basis.
        staged_fingerprints: Distinct fingerprints found in the run's staged rows,
            bounded by the caller. Empty on a first attempt.

    Returns:
        The bound basis, or ``None`` when nothing is staged, the group is empty AND
        no clip carries an ``action_data_uri``: nothing to load and nothing to fit
        from, so the caller skips action without ever starting an actor pool.

    Raises:
        ValueError: If two bases are in play, the referenced basis is missing or
            incompatible, or too few distinct descriptors survived a fresh fit.

    """
    store = PcaArtifactStore(root_uri, storage_profile=config.storage_profile)
    bound = _bound_fingerprint(dataset, staged_fingerprints)
    if bound is not None:
        # samples_used stays None: this run fit nothing, it re-projects onto the
        # basis the staged or canonical rows already reference.
        return ActionPca(artifact=_load_bound_basis(store, bound), samples_used=None)
    candidates = _candidate_uris(dataset, config)
    if not candidates:
        logger.info("no clip carries an action_data_uri; skipping the action group (no basis to fit)")
        return None
    sample, used = _fit_sample(config, candidates)
    if used <= ACTION_DIM:
        msg = (
            f"{used} distinct action span(s) survived the validity gates, but fitting the {ACTION_DIM}-component "
            f"PCA basis needs more than {ACTION_DIM} (mean-centering costs one degree of freedom). Provide more "
            "dexterous clips, or lower the action embedding dimension."
        )
        raise ValueError(msg)
    logger.info(f"fitting action PCA on {used} de-duplicated descriptors")
    pca = fit_action_pca(sample)
    uri = store.save_if_absent(pca)
    logger.info(f"action PCA basis {pca.fingerprint[:12]} persisted at {uri}")
    return ActionPca(artifact=pca, samples_used=used)


def check_action_outcome(result: ModalityResult, dataset: lance.LanceDataset) -> None:
    """Fail the run when every action row it published came back empty.

    The recipe's only POST-commit judgement, which the condition itself forces:
    "rows were published and every one of them landed NULL" is not decidable until
    the commit publishing them exists. Nothing is rolled back - those NULLs are the
    correct state for rows that failed, and they keep the rows owed - so this is a
    verdict on the run rather than an attempt to undo it.

    A partial per-row loss is NOT judged here: a row that failed stayed all-NULL
    and is re-selected by the ordinary missing-or-stale predicate next run, and the
    count is already in the run's own summary. Only the total outage is actionable,
    because it cannot be told apart from a misconfiguration by looking at the table.

    Args:
        result: The action modality's result; its ``failed`` count is the per-row loss.
        dataset: The clips table re-opened after the commit, so the "group holds
            nothing at all" test sees this run's own writes.

    Raises:
        ValueError: If every row the run visited failed AND the group holds no
            non-null ``embedding_action`` at all - a total outage that leaves
            nothing to embed, indistinguishable from a misconfiguration.

    """
    # A zero selection means the modality owed no work: a staging attempt that lost
    # rows raises before publication rather than returning zeroed counts, so this
    # early return cannot swallow that case.
    if result.selected == 0:
        return
    if result.failed >= result.selected and count_filled(dataset, ACTION_COLUMN_GROUP) == 0:
        msg = (
            f"action embedding failed on all {result.selected} row(s) it published and the group holds no "
            "embeddings: every artifact failed to read or was geometrically rejected, so there is nothing to "
            "embed. Re-run before investigating (transient S3 errors cause this without map retries), then "
            "check the export if it persists."
        )
        raise ValueError(msg)


def _bound_fingerprint(dataset: lance.LanceDataset, staged_fingerprints: tuple[str, ...]) -> str | None:
    """Return the one basis fingerprint this run is bound to, or ``None`` to fit fresh.

    The rule is ONE basis across both sources, not a precedence between them:
    disagreement raises, so which source is read first changes no outcome. Staged
    rows count as a source at all because they are already projected and are
    terminal - an attempt that crashed between staging and publication will
    publish exactly those vectors, so a group that looks empty may already be
    committed to a basis. Both present and AGREEING is the ordinary
    ``max_fragments`` wave case and proceeds.

    Args:
        dataset: The clips table at the version to read.
        staged_fingerprints: Distinct fingerprints the staged rows carry.

    Returns:
        The fingerprint to load, or ``None`` when nothing is bound yet.

    Raises:
        ValueError: If two fingerprints are in play from any combination of
            sources. Two bases have no safe automatic winner: choosing either
            silently makes part of the group incomparable with the rest, and only
            the operator knows which half is worth keeping.

    """
    staged = _single_fingerprint(staged_fingerprints, source="the staged rows")
    canonical = _single_fingerprint(
        _canonical_fingerprints(dataset),
        source=f"the action group at descriptor version {DESCRIPTOR_VERSION!r}",
    )
    if staged is not None and canonical is not None and staged != canonical:
        msg = (
            f"action rows staged under PCA basis {staged!r} disagree with the basis {canonical!r} the "
            f"published rows of {dataset.uri} carry; the two projections are mutually incomparable and "
            "neither can be chosen automatically. Discard the staging workspace: the published basis is "
            "the one the rest of the corpus already sits in, and a re-run re-stages against it."
        )
        raise ValueError(msg)
    bound = staged or canonical
    if bound is not None:
        # The source is logged, not because it changes the binding, but because a
        # basis bound from staged rows alone means an earlier attempt got as far as
        # staging - which nothing else in the run's output says.
        source = "staged rows" if staged is not None else "the published group"
        logger.info(f"action PCA basis {bound[:12]} bound from {source}; re-projecting, not refitting")
    return bound


def _canonical_fingerprints(dataset: lance.LanceDataset) -> tuple[str, ...]:
    """Return the basis fingerprints the published rows carry at the CURRENT contract.

    An older contract's rows are stale work rather than a refusal, so their
    fingerprint must neither be loaded nor counted towards the single-basis check;
    the pending filter re-selects those rows for recompute. Scoping the read is
    what makes both true at once.
    """
    if _PCA_FINGERPRINT_COLUMN not in dataset.schema.names:
        # A group whose columns do not exist yet carries no basis, which is a
        # complete answer rather than a failure. Checked here so binding a STAGED
        # basis does not depend on the caller having already widened the table -
        # matching ``count_filled``, which reads an absent group as zero filled.
        return ()
    # DataFusion SQL, not the scanner-filter dialect: single quotes are literals,
    # which is what sql_string_literal produces.
    return distinct_non_null_values(
        dataset,
        _PCA_FINGERPRINT_COLUMN,
        max_values=_PROVENANCE_VALUES_READ,
        where=f"{_DESCRIPTOR_VERSION_COLUMN} = {sql_string_literal(DESCRIPTOR_VERSION)}",
    )


def _single_fingerprint(fingerprints: tuple[str, ...], *, source: str) -> str | None:
    """Return the one fingerprint in ``fingerprints``, ``None`` if empty, else raise.

    Raises:
        ValueError: If more than one is present. Reachable through a group written
            by two producers, which nothing in the write path rules out.

    """
    if not fingerprints:
        return None
    if len(fingerprints) > 1:
        msg = (
            f"{source} reference more than one action PCA basis ({sorted(fingerprints)}); vectors projected "
            "onto different bases are mutually incomparable, so neither is a safe automatic winner. Two "
            "bases at one descriptor version means two runs each fit their own before either published, so "
            f"run one embeddings leg at a time over the {ACTION_COLUMN_GROUP.name} group."
        )
        raise ValueError(msg)
    return fingerprints[0]


def _load_bound_basis(store: PcaArtifactStore, fingerprint: str) -> PcaArtifact:
    """Load the bound basis from the artifact store, logging what it was fit on.

    Raises:
        ValueError: If the referenced basis is missing or incompatible (via
            ``store.load``).

    """
    pca = store.load(fingerprint)
    logger.info(f"loaded action PCA basis {fingerprint[:12]} (fit on {pca.n_fit_rows} rows)")
    return pca


def _candidate_uris(dataset: lance.LanceDataset, config: EmbeddingPipelineConfig) -> list[str]:
    """Return the bounded, rank-ordered ``action_data_uri`` set the fit may draw from.

    A narrow column scan that reads no artifact: it is how the fit becomes bounded
    without extracting the corpus first. The set is an OVERSAMPLE of the requested
    sample size, because sampling before extraction means some candidates will fail
    to read or be geometrically rejected.
    """
    return _ranked_distinct_uris(
        scan_column(
            dataset,
            "action_data_uri",
            row_filter=ACTION_APPLICABILITY_FILTER,
            batch_size=_URI_SCAN_BATCH_SIZE,
        ),
        limit=config.action.pca_sample_size * _PCA_CANDIDATE_OVERSAMPLE,
    )


def _fit_sample(config: EmbeddingPipelineConfig, candidates: Sequence[str]) -> tuple[npt.NDArray[np.float32], int]:
    """Extract the candidate URIs and keep the smallest-rank valid descriptors.

    The second of the two bounded passes: only the ranked candidate set is read, so
    the fit never touches an artifact outside the sample.

    Returns:
        ``(sample, used)`` - the ``(used, DESCRIPTOR_DIM)`` float32 fit matrix and
        the distinct-span count it holds.

    """
    if not candidates:
        return np.empty((0, DESCRIPTOR_DIM), dtype=np.float32), 0
    logger.info(f"extracting {len(candidates)} candidate descriptor(s) to fit the action PCA basis")
    return _collect_pca_sample(_extract_candidates(config, candidates), config.action.pca_sample_size)


def _extract_candidates(config: EmbeddingPipelineConfig, uris: Sequence[str]) -> Iterator[pa.Table]:
    """Extract descriptors for the candidate URIs only, across an actor pool.

    A bounded Ray Data pass over the candidate list: the same extractor the workers
    use, but driven from URIs rather than from a fragment scan, so the fit never
    touches an artifact outside the sample. The extractor is scheduled as a class so
    each actor keeps its storage clients across batches.

    Two independent widths apply, and they multiply. WITHIN an actor,
    ``read_concurrency`` overlaps the artifact fetches; ACROSS actors, the pool
    scales with the work available and the CPUs allocated. Neither divides the
    wrist geometry, which is GIL-held inside each actor and is therefore the
    per-actor ceiling - only the pool moves it, because separate actors are
    separate processes. Unlike a fill, this pass builds its own work items from a
    URI list rather than from fragments, so it is not bounded by the table's
    fragment geometry.
    """
    read_config = DualWristMotionReadConfig(
        storage_profile=config.storage_profile,
        read_concurrency=config.action.read_concurrency,
    )
    # This pass runs the action fill's own extractor, so it must land in the same
    # interpreter the fill workers use; the shared worker shape supplies that name.
    resources = WorkerResources(scan_batch_size=config.action.batch_size, num_cpus=config.action.num_cpus)
    descriptors = ray.data.from_items([{"action_data_uri": uri} for uri in uris]).map_batches(
        DualWristMotionDescriptorExtractor,
        fn_constructor_kwargs={"config": read_config},
        batch_format="pyarrow",
        batch_size=config.action.batch_size,
        num_cpus=config.action.num_cpus,
        compute=ActorPoolStrategy(min_size=1),
        runtime_env=ray_data_gpu_runtime_env(resources.env_name),
        scheduling_strategy="DEFAULT",
    )
    for batch in descriptors.iter_batches(batch_format="pyarrow"):
        yield cast("pa.Table", batch)


def _sample_rank(uri: str) -> int:
    """Map an action URI to a deterministic 128-bit rank (first half of its SHA-256).

    The rank is a pure function of the URI, so the SET of the ``sample_size``
    smallest-rank URIs is identical no matter how Ray partitions or orders the scan.
    Scan-order sampling would instead let a repartition silently change which spans
    train the basis. The set is the invariant, not the fingerprint: the sample's row
    order still follows arrival, so a re-fit yields an equivalent basis under a new
    name (see ``_collect_pca_sample``).
    """
    return int.from_bytes(hashlib.sha256(uri.encode("utf-8")).digest()[:16], "big")


def _ranked_distinct_uris(batches: Iterable[pa.Table], limit: int) -> list[str]:
    """Return the ``limit`` smallest-rank distinct URIs from a narrow URI scan.

    A bounded heap keyed on ``-_sample_rank(uri)`` keeps only the current best
    candidates, so driver memory is O(limit) strings no matter how large the corpus
    is. The result is sorted by rank ascending, which is the order the fit sample
    then keeps from.
    """
    # Bounded min-heap over -rank, so the root is the LARGEST kept rank (the next to
    # evict). An evicted URI's rank sits monotonically above the shrinking largest
    # kept rank, so it can never re-enter and ``kept`` need not remember it.
    heap: list[tuple[int, str]] = []
    kept: set[str] = set()
    for batch in batches:
        for uri in batch.column("action_data_uri").to_pylist():
            if not uri or uri in kept:
                continue
            neg_rank = -_sample_rank(uri)
            if len(heap) < limit:
                heapq.heappush(heap, (neg_rank, uri))
                kept.add(uri)
            elif neg_rank > heap[0][0]:
                _evicted_rank, evicted_uri = heapq.heappushpop(heap, (neg_rank, uri))
                kept.discard(evicted_uri)
                kept.add(uri)
    return [uri for _neg_rank, uri in sorted(heap, key=lambda entry: -entry[0])]


def _collect_pca_sample(batches: Iterable[pa.Table], sample_size: int) -> tuple[npt.NDArray[np.float32], int]:
    """Keep a bounded, de-duplicated, DETERMINISTIC descriptor sample for the PCA fit.

    De-duplicates on ``action_data_uri`` so a multi-view span (whose views share one
    action artifact) contributes its descriptor once, not once per view (which would
    over-weight it in the basis). Among the distinct spans it keeps the
    ``sample_size`` with the smallest ``_sample_rank`` - a total order derived purely
    from the URI - so repartitioning or extraction order can never change the fitted
    population. NULL descriptors (per-row extract failures) never enter the sample.

    ::

        distinct span --rank=sha256(uri)[:128]--> bounded heap (size sample_size)
                                                    keeps the smallest-rank spans

    What is invariant is the selected SET, not the row order of the returned
    matrix: the heap's internal order follows arrival, so a wider read or a
    different partitioning permutes the rows. That leaves the basis mathematically
    equivalent - the SVD of the mean-centred rows is permutation-invariant and its
    sign is pinned - but float64 summation is not associative, so the bytes shift
    in their last places and the content-addressed fingerprint changes completely.
    A deliberate re-fit is therefore expected to produce a NEW fingerprint over an
    identical sample with identical fit statistics; that is a different name for an
    equivalent basis, not a corrupted one.

    Returns:
        ``(sample, used)`` - the ``(used, DESCRIPTOR_DIM)`` float32 fit matrix and
        the distinct-span count it holds (``used <= sample_size``).

    """
    # Bounded min-heap over -rank, so the root is the LARGEST kept rank (the next to
    # evict). ``uri`` breaks any rank tie (unreachable for a 128-bit SHA-256) so a
    # comparison never reaches the ndarray descriptor.
    heap: list[tuple[int, str, npt.NDArray[np.float32]]] = []
    in_heap: set[str] = set()
    for batch in batches:
        uris = batch.column("action_data_uri").to_pylist()
        # ``decode_descriptors`` returns the per-row valid mask plus a compact matrix
        # of only the present descriptors, walked in row order by a cursor.
        valid, present = decode_descriptors(batch)
        cursor = 0
        for index, uri in enumerate(uris):
            if not valid[index]:
                continue
            descriptor = present[cursor]
            cursor += 1
            if uri in in_heap:
                continue
            neg_rank = -_sample_rank(uri)
            if len(heap) < sample_size:
                # A fresh copy drops the reference to the batch matrix so the whole
                # decoded batch can be freed once its rows are consumed.
                heapq.heappush(heap, (neg_rank, uri, np.array(descriptor, dtype=np.float32)))
                in_heap.add(uri)
            elif neg_rank > heap[0][0]:
                evicted = heapq.heappushpop(heap, (neg_rank, uri, np.array(descriptor, dtype=np.float32)))
                in_heap.discard(evicted[1])
                in_heap.add(uri)
    used = len(heap)
    sample = np.empty((used, DESCRIPTOR_DIM), dtype=np.float32)
    for row, (_neg_rank, _uri, descriptor) in enumerate(heap):
        sample[row] = descriptor
    return sample, used
