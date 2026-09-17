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

"""Per-modality fill specification, the worker shape it runs in, and the result it returns.

A modality is DATA, not a subclass. Text and image differ from each other only in
what they compute and where it lands, so each modality is described by one frozen
``ModalityFill`` built by a small function. Action is the only modality that differs
in BEHAVIOUR - it must resolve one PCA basis before any worker starts, and it guards
how much of its group actually landed - and that behaviour lives in the driver's
explicit action branch rather than in a lifecycle hook here.

::

    build_text_fill(config) -----+
    build_image_fill(config) ----+--> ModalityFill --> fill --> ModalityResult
    build_action_fill(cfg, pca) -+     what + where     scan      counts and the
                                                        compute   committed version
                                                        write

The embedder a fill names is PURE COMPUTE over positions: it is constructed once per
worker, called with an Arrow batch of ``source_columns``, and returns exactly the
group's columns - one row per input row, in input order, a per-row failure emitting
an all-NULL group. The batch it is handed does carry ``clip_id``, because the fill
joins the computed group back on it, but the embedder must not READ it: the key is
re-attached positionally, which is only sound because of that cardinality- and
order-preserving contract.
"""

from typing import Any

import attrs

from cosmos_curator.models.registry import resolve_registered_model
from cosmos_curator.next.embeddings.action.embedder import DualWristMotionEmbedder, DualWristMotionReadConfig
from cosmos_curator.next.embeddings.action.pca import PcaArtifact
from cosmos_curator.next.embeddings.action.wrist_motion import DESCRIPTOR_VERSION
from cosmos_curator.next.embeddings.image.embedder import HfVisionImageEmbedder
from cosmos_curator.next.embeddings.model_specs import (
    DEFAULT_IMAGE_MODEL,
    DEFAULT_TEXT_MODEL,
    ModelSpec,
    VisionModelSpec,
)
from cosmos_curator.next.embeddings.schemas import (
    ACTION_COLUMN_GROUP,
    IMAGE_COLUMN_GROUP,
    TEXT_COLUMN_GROUP,
    EmbeddingColumnGroup,
)
from cosmos_curator.next.embeddings.text.embedder import SentenceTransformerTextEmbedder
from cosmos_curator.next.recipes.embeddings.config import EmbeddingPipelineConfig, Modality
from cosmos_curator.next.utils.identity import canonical_digest

# The Pixi environment whose interpreter holds torch / transformers /
# sentence_transformers - and the pyarrow / lance the fill worker itself needs.
# Every embedding worker is pinned to it, including the CPU action worker: the
# driver process is not guaranteed to be running that interpreter, and an
# unpinned worker would inherit whichever one Ray happens to start.
_EMBED_ENV_NAME = "default"

# Bytes per gigabyte, so the config's ``memory_gb`` maps to Ray Data's ``memory=``
# (bytes) without a magic ``1024**3`` at the call site.
_BYTES_PER_GB = 1024**3

# Only clips that carry a media URI can be decoded and embedded. The predicate is
# pushed into the per-fragment scan so undecodable-by-construction rows never reach
# the GPU; rows that pass but fail to decode are still emitted as NULL image-group
# rows by the embedder to preserve cardinality.
_IMAGE_APPLICABILITY_FILTER = "clip_uri IS NOT NULL AND clip_uri != ''"

ACTION_APPLICABILITY_FILTER = "action_data_uri IS NOT NULL AND action_data_uri != ''"
"""Rows the action modality can embed: those carrying a Mecka ACT2 action artifact.

The action modality is Mecka-only, so a clip is applicable if its
``action_data_uri`` is non-empty. The predicate pushes down into the per-fragment
scan, so the modality visits exactly the rows the extractor will attempt. A row
whose artifact turns out non-dexterous or malformed is not excluded here but
rejected per row by the extractor's geometry checks (a NULL group, retried next
run).

Public because ``action_pca`` bounds its PCA candidate scan with the SAME
predicate: the basis must be fit from exactly the population the fill will visit.
Two copies could drift, and a drift would fit the basis on one population while
embedding another - a difference nothing type-checks and no test would catch.
"""


@attrs.frozen
class WorkerResources:
    """The Ray actor shape and scan granularity for one modality's fill workers.

    Lets the fill issue a single ``map_batches`` for every modality with no
    per-modality branch: each modality sets only the fields it has a reason to
    request. The fields are independent, not alternatives - a modality that
    holds a GPU and also does host-side work sets both ``num_gpus`` and
    ``num_cpus`` - and only a modality with a measured heap constraint sets
    ``memory_bytes``.

    Attributes:
        scan_batch_size: Rows per Arrow batch handed to the embedder. One value
            drives three things that have to agree: the repartition target that
            splits an owed fragment into blocks, Ray's ``map_batches`` batch size,
            and hence the embedder's input width. Raising it therefore coarsens
            block geometry as well as widening the batch, and block geometry is
            one of the two row minimums that size what one checkpoint makes
            durable (the other is ``parquet_rows_per_file``).
        env_name: Pixi environment the worker's interpreter runs in.
        num_gpus: GPUs reserved per worker, or ``None`` for a CPU modality. A
            fractional value lets Ray pack several workers per physical GPU.
        num_cpus: CPUs reserved per worker, or ``None`` to accept Ray's default.
        memory_bytes: Per-worker heap reservation (Ray ``memory=``), or ``None``.
            Set it only where a measured figure exists; it must cover the decoded
            inputs plus one fragment's worth of computed vectors in flight.

    """

    scan_batch_size: int
    env_name: str = _EMBED_ENV_NAME
    num_gpus: float | None = None
    num_cpus: float | None = None
    memory_bytes: int | None = None


# eq=False is required, not stylistic: embedder_kwargs can hold a numpy-bearing
# PcaArtifact, and an attrs-generated __eq__ would compare those arrays
# element-wise and then raise on the ambiguous truth value of the resulting
# boolean array. Restoring the default eq turns every comparison into a crash.
@attrs.frozen(eq=False)
class ModalityFill:
    """One modality's fill specification: what to compute, and where the result lands.

    Inert data end to end. The spec is built on the driver but its embedder is
    constructed inside a Ray worker, so nothing live (no loaded model, no storage
    client, no dataset handle) may be held here - the class and its kwargs have to
    survive pickling to that worker.

    Attributes:
        modality: The modality this spec embeds.
        group: The ``clips.lance`` column group it fills (vectors + provenance).
            Its ``primary_vector`` defines "pending"; its ``provenance_columns``
            define staleness.
        source_columns: Columns the worker's fragment scan projects; a subset of
            ``EMBED_SOURCE_ROW``. It must include ``clip_id`` - the embedder never
            reads it, but the fill joins the computed group back on it.
        applicability_filter: SQL predicate selecting the rows this modality can
            embed, or ``None`` when every row is applicable.
        result_identity: ``{provenance_column: value}`` naming the computation this
            run performs, covering every one of ``group.provenance_columns``. It is
            the single answer to "which computation produced these columns": the
            selection predicate re-computes a row whose stored values differ, and
            the rows this run writes carry these very values. A row holding a
            different value is therefore selected work, not a refusal.
        embedder_cls: The callable class the worker constructs once and calls per
            batch. Its ``__call__`` takes an Arrow batch of ``source_columns`` and
            returns exactly ``group``'s columns, one row per input row in input
            order.
        embedder_kwargs: Keyword arguments the worker constructs ``embedder_cls``
            with; picklable, and holding no live handle.
        resources: Ray actor shape and scan batch size for this modality's workers.
        weights_name: ``all_models.json`` key whose checkpoint must be staged before
            Ray starts, or ``None`` for a modality that loads no checkpoint.

    """

    modality: Modality
    group: EmbeddingColumnGroup
    source_columns: tuple[str, ...]
    applicability_filter: str | None
    result_identity: dict[str, str]
    embedder_cls: type
    embedder_kwargs: dict[str, Any]
    resources: WorkerResources
    weights_name: str | None

    def __attrs_post_init__(self) -> None:
        """Refuse an identity that does not cover exactly the group's provenance columns.

        The selection predicate iterates ``group.provenance_columns`` and reads
        ``result_identity`` for each. A missing key would silently narrow the
        predicate to the columns that happen to be present, so a row stale in the
        uncovered column would read as current and never be recomputed - a wrong
        answer that no later stage can detect. An extra key is the same mistake
        seen from the other side: a value nothing compares.
        """
        expected = set(self.group.provenance_columns)
        if set(self.result_identity) != expected:
            msg = (
                f"{self.modality} result_identity keys {sorted(self.result_identity)} "
                f"must be exactly {sorted(expected)}"
            )
            raise ValueError(msg)


@attrs.frozen(kw_only=True)
class ModalityResult:
    """Per-modality accounting for one run, for the operator-facing summary.

    Keyword-only so that inserting a count cannot silently re-bind an existing
    positional argument at a call site.

    Attributes:
        modality: Which modality produced this result.
        result_identity: The computation this run published, copied from the
            spec it ran. Reported because it is the one fact that answers "did my
            model change take effect" against a group whose counts look the same
            either way. Empty only when no computation could be identified at
            all, which is not the same as a run that identified one and owed
            nothing.
        selected: Rows this run published, summed from what the publication
            tasks streamed.
        filled: Rows whose group became complete this run (non-null primary vector).
        committed_version: Table version after the group's ``Update`` commit.
            ``None`` means NOTHING was committed, and it is the only encoding of
            that state - do not re-derive it from the counts.

    """

    modality: Modality
    result_identity: dict[str, str]
    selected: int
    filled: int
    committed_version: int | None

    @property
    def failed(self) -> int:
        """Return the rows that stayed all-NULL after a per-row failure (retryable)."""
        return self.selected - self.filled

    @classmethod
    def nothing_to_do(cls, modality: Modality, result_identity: dict[str, str]) -> "ModalityResult":
        """Return the result of a modality that owed no work and therefore committed nothing.

        Named for the success it reports, not for the empty counts it carries: a
        modality whose work all FAILED also leaves every count at zero, and that
        one raises rather than returning, so this constructor cannot express it.

        Args:
            modality: The modality that owed nothing.
            result_identity: The computation it would have performed. Pass an
                empty mapping only when none could be identified - a modality
                whose rows are all current still has one, and reporting it is
                what tells an operator WHICH computation they are current with.

        """
        return cls(
            modality=modality,
            result_identity=result_identity,
            selected=0,
            filled=0,
            committed_version=None,
        )


def _model_result_fingerprint(spec: ModelSpec) -> str:
    """Hash the facts that decide what this model's vectors contain.

    A checkpoint id alone answers "which weights" and not "which computation", so
    it cannot separate two runs that pooled the same backbone differently or
    formatted the same text differently. Hashing the registry key and its
    resolved id and revision together with the pooling reduction and
    ``result_version`` gives one value that moves whenever the vectors would
    differ - which is what makes a stored value usable both as this run's
    provenance and as the staleness test.

    ``revision`` comes from the registry rather than the spec because that is
    where the weights are actually pinned: a re-pin behind an unchanged
    ``weights_name`` must move the fingerprint.
    """
    registered = resolve_registered_model(spec.weights_name)
    return canonical_digest(
        {
            # The registry key, and NOT a redundant restatement of the two facts
            # below it. It is what both embedders pass to
            # ``get_local_dir_for_weights_name``, so it selects the DIRECTORY the
            # weights load from, and the registry admits two keys that agree on
            # everything else: seedvr2_7b and seedvr2_7b_sharp share one model_id,
            # one revision and one precision while naming different .pth files.
            # Digested from the spec rather than from ``registered``, because the
            # spec's value is the one that reaches the filesystem.
            "weights_name": spec.weights_name,
            "model_id": registered.model_id,
            "revision": registered.revision,
            # isinstance rather than getattr: it narrows the union for mypy, so the
            # payload stays typed instead of becoming Any at the one place whose
            # exact contents define every stored fingerprint.
            "pooling": spec.pooling if isinstance(spec, VisionModelSpec) else None,
            "result_version": spec.result_version,
        }
    )


def build_text_fill(config: EmbeddingPipelineConfig) -> ModalityFill:
    """Build the text fill: a SentenceTransformer over each clip's task / subtask text.

    Text applies to every row, so it declares no applicability filter, and the leg
    never fails per row: every pending row it visits comes back complete.
    """
    # Computed once and spent twice - as the value rows must already hold to count
    # as current, and as the value this run's rows will carry. Two derivations
    # could drift; one local cannot.
    fingerprint = _model_result_fingerprint(DEFAULT_TEXT_MODEL)
    return ModalityFill(
        modality=Modality.TEXT,
        group=TEXT_COLUMN_GROUP,
        source_columns=SentenceTransformerTextEmbedder.SOURCE_COLUMNS,
        applicability_filter=None,
        result_identity={TEXT_COLUMN_GROUP.provenance_columns[0]: fingerprint},
        embedder_cls=SentenceTransformerTextEmbedder,
        embedder_kwargs={
            "spec": DEFAULT_TEXT_MODEL,
            "result_fingerprint": fingerprint,
            "encode_batch_size": config.text.batch_size,
        },
        # No memory_bytes: unlike image, the text worker's heap is dominated by the
        # model weights rather than by decoded inputs, and no measured figure exists
        # to reserve.
        resources=WorkerResources(scan_batch_size=config.text.batch_size, num_gpus=config.text.num_gpus),
        weights_name=DEFAULT_TEXT_MODEL.weights_name,
    )


def build_image_fill(config: EmbeddingPipelineConfig) -> ModalityFill:
    """Build the image fill: an HF vision backbone over one representative frame per clip.

    Only clips with a non-empty ``clip_uri`` are applicable; the embedder still
    returns one group row per scanned row, emitting a NULL group for a clip whose
    media cannot be decoded so that cardinality is preserved.
    """
    # Computed once and spent twice; see build_text_fill.
    fingerprint = _model_result_fingerprint(DEFAULT_IMAGE_MODEL)
    return ModalityFill(
        modality=Modality.IMAGE,
        group=IMAGE_COLUMN_GROUP,
        source_columns=HfVisionImageEmbedder.SOURCE_COLUMNS,
        applicability_filter=_IMAGE_APPLICABILITY_FILTER,
        result_identity={IMAGE_COLUMN_GROUP.provenance_columns[0]: fingerprint},
        embedder_cls=HfVisionImageEmbedder,
        # Both fields are the frame reader's, forwarded through the embedder that
        # builds it: only the profile NAME and a plain int travel, and the reader
        # resolves its transport lazily on first use, so no client handle and no
        # thread pool has to survive pickling.
        embedder_kwargs={
            "spec": DEFAULT_IMAGE_MODEL,
            "result_fingerprint": fingerprint,
            "storage_profile": config.storage_profile,
            "read_concurrency": config.image.read_concurrency,
        },
        # Image is the one modality that reserves memory: decoded RGB frames, not the
        # weights, are its binding constraint. Its CPU reservation is a FIXED budget
        # rather than one that tracks read_concurrency (design doc section 4.2);
        # left to Ray's default the actor would be unreserved entirely. Ray documents
        # requesting both num_cpus and num_gpus on one map op as experimental and
        # warns on it, and image is the only modality that pairs them, so if a Ray
        # upgrade regresses actor placement it will regress here first.
        resources=WorkerResources(
            scan_batch_size=config.image.batch_size,
            num_gpus=config.image.num_gpus,
            num_cpus=config.image.num_cpus,
            memory_bytes=int(config.image.memory_gb * _BYTES_PER_GB),
        ),
        weights_name=DEFAULT_IMAGE_MODEL.weights_name,
    )


def build_action_fill(config: EmbeddingPipelineConfig, pca: PcaArtifact) -> ModalityFill:
    """Build the action fill: dual-wrist descriptors projected onto a resolved PCA basis.

    Taking the basis as a REQUIRED argument makes "no basis bound yet" unrepresentable,
    so this builder carries no runtime guard against it - the check is not missing, it
    is structurally unreachable. The basis then travels as data rather than as a
    fingerprint each worker re-loads: it is a frozen wrapper over read-only arrays of a
    few hundred KB, so pickling it into the spec is cheaper than one artifact-store read
    per worker and removes any chance of two workers binding different bases.

    Taking the artifact itself, rather than the driver's wrapper around it, is also
    what keeps this module free of an import cycle: the basis resolver imports
    ``ModalityResult`` from here, and depending only on the narrower type means
    nothing here has to reach back into it.
    """
    return ModalityFill(
        modality=Modality.ACTION,
        group=ACTION_COLUMN_GROUP,
        source_columns=DualWristMotionEmbedder.SOURCE_COLUMNS,
        applicability_filter=ACTION_APPLICABILITY_FILTER,
        # Action needs no digest: both of its facts are already the identity. The
        # descriptor version names the semantics and the artifact fingerprint names
        # the exact basis, so the pair is what a resume has to match - which is why
        # the caller binds the basis before this builder runs rather than letting a
        # worker discover one.
        result_identity={
            "embedding_action_descriptor_version": DESCRIPTOR_VERSION,
            "embedding_action_pca_fingerprint": pca.fingerprint,
        },
        embedder_cls=DualWristMotionEmbedder,
        embedder_kwargs={
            "config": DualWristMotionReadConfig(
                storage_profile=config.storage_profile,
                read_concurrency=config.action.read_concurrency,
            ),
            "pca": pca,
        },
        # CPU only: action reads and derives, it runs no model.
        resources=WorkerResources(
            scan_batch_size=config.action.batch_size,
            num_cpus=config.action.num_cpus,
        ),
        # Nothing to stage: the action leg loads no checkpoint.
        weights_name=None,
    )
