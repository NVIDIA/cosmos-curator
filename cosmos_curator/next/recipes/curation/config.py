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

"""Typed config for the Curate leg: one flat model plus two invariant carriers.

::

    CurateConfig            the whole operator-settable surface
      +- weights            ModalityWeights  - must sum to 1
      +- target             SelectionTarget   - at most one form set

A sub-model exists here if and only if it carries a CROSS-FIELD invariant. Both
that survive do: the fused vector is unit-norm only when the weights sum to 1,
and a target expressed twice has no single answer. Everything else is a leaf on
``CurateConfig``, because grouping leaves by theme buys nothing a name does not
already say and costs every caller an extra hop.

Strict, frozen, and ``extra="forbid"``, which is what makes a migration safe
rather than silent: a config still naming a removed key such as ``staging_root``
fails at parse time with the offending key named, instead of being ignored and
running with a default the operator did not choose.

A resolved config is also the run's IDENTITY: ``result_defining_digest`` hashes
every field that decides an outcome, and the write stamps that digest on the
commit so a committed table version can name the rules that produced it. The
excluded fields are a deny-list, so a field added here joins the identity by
default rather than needing to be remembered.

``resolve_config`` is the single entry point every config-driven surface goes
through - ``pipeline validate``, ``pipeline render``, and ``run-pipeline`` all
resolve here - so one file plus its ``--set`` overrides can never mean two
different things depending on which command read it. ``strict=True`` makes the
override parsing load-bearing rather than incidental: values are read as YAML, so
``target_mean_cluster_rows=200000`` resolves to an int and passes, while a quoted
``"200000"`` is rejected instead of being coerced.

See docs/curator/design/curator-next-curation.md.
"""

import hashlib
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Any, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from cosmos_curator.next.core.config import apply_dotted_overrides
from cosmos_curator.next.recipes.curation import fairness
from cosmos_curator.next.recipes.curation.columns import (
    FUSED_BLOCKS,
    TASK_COLUMN,
    TASK_VECTOR_COLUMN,
    CurateReason,
    WithinGroupOrder,
    block_width,
)

_MODEL_CONFIG = ConfigDict(frozen=True, strict=True, extra="forbid")
_YAML_SUFFIXES = frozenset({".yaml", ".yml"})

SchemaVersion = Literal[1]
CurateKind = Literal["curate"]

# Fields excluded from a run's identity. Named as a DENY-list, and that direction
# is the point: a field added to CurateConfig later is covered by default. An
# allow-list of result-defining fields would silently omit the new one, so two
# runs applying genuinely different rules would report the same identity - the
# failure a digest exists to prevent. This way the worst case is a digest that
# changes when it need not, which can waste a comparison but cannot mislead one.
#
#   clips_lance_uri    the table the digest is published ON, so naming it inside
#                      the identity is circular, and a table copied to a second
#                      URI would otherwise report a different identity for rules
#                      that are the same
#   storage_profile    selects credentials and endpoint, never a rule
#   dedup_concurrency  scheduling only; the retention result is invariant to it
_NON_RESULT_DEFINING: frozenset[str] = frozenset({"clips_lance_uri", "storage_profile", "dedup_concurrency"})

# Digest payload key holding the code-defined contract. The deny-list above only
# reaches fields an operator writes, but the verdicts also depend on declarations
# no config names - see contract_fingerprint below. A dunder cannot collide with a
# serialized field, so no runtime guard exists: is_valid_field_name rejects every
# name starting with an underscore, so model_dump() can never emit this key, and
# extra="forbid" closes the route through extras. The reason is NOT that pydantic
# files a name of this shape away as a private attribute - is_valid_privateattr_name
# requires a single leading underscore and excludes a dunder as well - it is that no
# such name can be a field at all. Both predicates read from pydantic 2.13.4,
# pydantic/_internal/_fields.py.
_CONTRACT_KEY: str = "__contract__"


def contract_fingerprint() -> dict[str, object]:
    """Return the code-defined half of a run's identity, as JSON-ready data.

    The config says what an operator asked for; this says what the code meant by
    it. Both halves are hashed into ``CurateConfig.result_defining_digest``,
    because a release that reorders the fused blocks, rewidths one, renames a
    persisted reason, repoints either column the task grouping reads, or folds
    task labels differently changes verdicts without changing any config file -
    and two such runs must not report the same identity.

    Assembled here rather than in ``columns`` because the contract spans two
    layers: ``columns`` declares the blocks, the reasons and both task columns,
    ``fairness`` owns the fold applied to one of them, and a declarations module
    cannot import a kernel that imports it. Every part is derived from the thing
    it identifies rather than named by a version constant.

    Returns:
        The fused block order as ``[weight field, vector column, width]`` triples,
        every persisted ``CurateReason`` value, the label-canonicalization rule's
        probe/result pairs, and the two columns level-1 fairness reads: the label
        it groups by (``task_label``) and the vector that merges those labels
        (``task_vector``), each keyed by what it holds rather than by its constant.

    """
    return {
        "canonical_task": fairness.canonicalization_contract(),
        "fused_blocks": [[field, group.primary_vector, block_width(group)] for group, field in FUSED_BLOCKS],
        "reasons": [reason.value for reason in CurateReason],
        "task_label": TASK_COLUMN,
        "task_vector": TASK_VECTOR_COLUMN,
    }


# Duplicate-threshold floor for dedup_eps: one float32 step below 1.0. dedup
# compares float32 scores against float32(1 - eps), and float32 carries a 24-bit
# mantissa, so the largest value it can represent under 1.0 is 1 - 2**-24. An eps
# at least that large names a threshold a whole step clear of 1.0; a far smaller
# one rounds 1 - eps back to exactly 1.0f, which no cosine exceeds, so the whole
# GPU retention pass runs to flag nothing but rows whose similarity overshot 1.0
# on float error.
#
# The step, deliberately, and not the exact round-off boundary: that sits an
# octave lower, just above 2**-25, half-way between two float32 neighbours, so a
# floor there would rest on how one float64 subtraction rounds. Every eps this
# turns away is within an ulp of inert.
_MIN_DEDUP_EPS: float = 2.0**-24

# The fused vector is unit-norm, and its cosine equals the weighted sum of the
# per-block cosines, only when the weights sum to exactly 1. Validating on the
# driver turns a broken weight vector into a config error rather than an identity
# that silently stops holding inside a GPU task.
_WEIGHT_SUM_TOL: float = 1e-6


class ModalityWeights(BaseModel):
    """Per-block fusion weights: an interface, not a tuning knob.

    These define the distance the whole corpus clusters and de-duplicates on, so
    changing one re-clusters everything and makes two runs incomparable. They are
    chosen TOGETHER with ``CurateConfig.dedup_eps``: a duplicate needs the
    weighted sum of per-block distances to fall below ``eps``.

    Attributes:
        subtask: Weight of the subtask text block. ``subtask`` funds the subtask
            text vector only; the task vector is never a fused block.
        image: Weight of the image block.
        action: Weight of the action block. Funds motion similarity and
            near-duplicate separation only, never a task question.

    """

    model_config = _MODEL_CONFIG

    # The 0.6 default is a design choice - the subtask instruction is the
    # intended dominant curation term - and it carries a PRECONDITION: it assumes
    # subtask_name holds real language. On a corpus whose labels are still
    # identifiers ("subtask_6186") the text block contributes identifier
    # collision distance and then dominates the metric. The corpus this design was
    # measured on is no longer such a corpus - its subtask labels score AUC 0.853
    # against the task, i.e. real language - so the escape below is a contract for
    # other corpora, not a workaround the measured one needs. Measurement and
    # apparatus: docs/curator/design/curator-next-curation.md, "Why each modality
    # is present". subtask=0.0 is that escape and needs no code
    # change: a zero-weight block contributes a zero sub-vector, keeps the fused
    # vector unit-norm, and drops out of the eligibility predicate, so the
    # modality leaves the metric AND stops being required of a row.
    subtask: float = Field(default=0.6, ge=0.0, le=1.0)
    image: float = Field(default=0.2, ge=0.0, le=1.0)
    # The action block carries NO task signal, and that is measured rather than
    # assumed: the task-separation ratio is 0.990 - different-task pairs sit no
    # farther apart than same-task pairs - and same-task-closer AUC is 0.543,
    # against 0.853 for text and 0.879 for image on the same rows and the same
    # grouping, so this is a property of the modality rather than of one corpus's
    # labels. Measurement and apparatus: the design doc section cited above.
    # Gross dual-wrist kinematics do not
    # distinguish woodworking from cleaning_shoes. So this weight buys
    # motion-redundancy separation and dedup locality: it is what stops "same
    # instruction, same scene, different execution" from being scored a
    # duplicate. It can never make the fused distance answer "which task is
    # this" - task fairness reads the canonical label strings instead.
    action: float = Field(default=0.2, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _weights_sum_to_one(self) -> Self:
        """Reject weights whose sum departs from 1 by more than the tolerance."""
        total = self.subtask + self.image + self.action
        if abs(total - 1.0) > _WEIGHT_SUM_TOL:
            msg = f"modality weights must sum to 1 within {_WEIGHT_SUM_TOL}, got {total}"
            raise ValueError(msg)
        return self


class SelectionTarget(BaseModel):
    """How many de-duplication SURVIVORS to keep: a count, a fraction, or all.

    The denominator is the survivor population, not the corpus and not the
    eligible rows, so ``target_fraction=0.5`` means half of the clips that
    survived de-duplication; duplicates, invalid rows, and rows the run never
    claimed are all outside it. At most one form is set, and neither means "keep
    every survivor".

    Attributes:
        target_count: Absolute keep-count, clamped to the population.
        target_fraction: Fraction of the population to keep.

    """

    model_config = _MODEL_CONFIG

    target_count: int | None = Field(default=None, ge=0)
    target_fraction: float | None = Field(default=None, gt=0.0, le=1.0)

    @model_validator(mode="after")
    def _at_most_one_target(self) -> Self:
        """Reject setting both target forms at once; either alone (or neither) is legal."""
        if self.target_count is not None and self.target_fraction is not None:
            msg = "SelectionTarget accepts at most one of target_count / target_fraction"
            raise ValueError(msg)
        return self

    def resolve(self, survivor_count: int) -> int:
        """Return the concrete keep-count for a population of ``survivor_count``.

        An empty population resolves to zero under every form. Otherwise
        ``target_count`` clamps to the population, ``target_fraction`` rounds
        half-up and clamps to ``[1, survivor_count]``, and an unset target keeps
        every survivor.
        """
        if survivor_count <= 0:
            return 0
        if self.target_count is not None:
            return min(self.target_count, survivor_count)
        if self.target_fraction is not None:
            # Explicit half-up: Python's round() is banker's rounding, which
            # would make the keep-count depend on parity rather than on the
            # fraction.
            scaled = math.floor(self.target_fraction * survivor_count + 0.5)
            return max(1, min(scaled, survivor_count))
        return survivor_count


class CurateConfig(BaseModel):
    """The whole operator-settable surface of a Curate run.

    ``clips_lance_uri`` is both the input and the output - Curate widens the same
    table it read - which is why it is not called a source.

    A resolved config is also the run's IDENTITY: the commit stamps
    ``result_defining_digest``, so a version can name the rules that produced it.
    That digest is an equality check, not a description, which is why the config
    is a FILE - the version says WHICH rules, the file says what they were.

    The tuning fields - the two k controls, the fit budget, the seed, the dedup
    threshold and concurrency, and the merge theta - are documented beside their
    declarations instead of here, because each one's quantified reasoning belongs
    next to the constant it justifies.

    Attributes:
        schema_version: Config schema generation.
        kind: Pipeline discriminator the generic CLI routes on.
        clips_lance_uri: The clips table to read and widen.
        storage_profile: Named storage credentials / endpoint profile.
        weights: Per-block fusion weights; see ``ModalityWeights``.
        target: How many survivors to keep; see ``SelectionTarget``.
        within_group_order: Which survivors win inside a funded group.

    """

    model_config = _MODEL_CONFIG

    # Both envelope fields are REQUIRED rather than defaulted. A default would
    # make an unversioned or misrouted file run under this generation's field
    # meanings instead of being rejected, which is the one failure this pair
    # exists to prevent.
    schema_version: SchemaVersion = Field(
        description="Config schema generation. Pinned rather than defaulted so a config written against a future "
        "generation is rejected instead of being read under this one's field meanings.",
    )
    kind: CurateKind = Field(
        description="Pipeline discriminator the generic CLI routes on; it must match the registered kind name.",
    )
    clips_lance_uri: str
    storage_profile: str = "default"
    weights: ModalityWeights = Field(default_factory=ModalityWeights)
    target: SelectionTarget = Field(default_factory=SelectionTarget)
    within_group_order: WithinGroupOrder = "farthest"

    # The sole cluster-count control: k is ceil(eligible_rows / this). Tuned for
    # the 250M-500M envelope, so on a small corpus it resolves to k == 1. That is
    # benign for de-duplication (one exhaustive cluster has perfect duplicate
    # recall and no cluster-boundary false negatives) but it does leave
    # curate_cluster_id carrying no information, so the run reports it.
    target_mean_cluster_rows: int = Field(default=200_000, ge=1)

    # Row budget for the single-GPU k-means fit sample. The ceiling is a device
    # property rather than a corpus one, and this module deliberately does not
    # restate it: pipeline._fit_sample_fragments owns the device-size assumption,
    # evaluates the ceiling for the configured value, and WARNS with the largest
    # fitting row count when the estimated device peak does not fit one card.
    # A number copied here would have no mechanism keeping it equal to that one.
    fit_sample_rows: int = Field(default=4_000_000, ge=1)

    # Seed of both k-means fits, and so RESULT-DEFINING: it decides the centroids,
    # hence every curate_cluster_id, every distance, and the within-group order
    # the cut spends its quota in. Two runs differing only here are therefore not
    # comparable. It does not invalidate a persisted basis - the centroids
    # artifact records the seed that produced it, so an artifact stays readable
    # against the version it was written for.
    kmeans_random_state: int = 42

    # Seeds the residual order at BOTH fairness levels, and so RESULT-DEFINING:
    # it decides which tasks win the leftover clips and, inside each funded task,
    # which of its subtask cells do. At a target below the task count the residual
    # IS the whole allocation, so it decides which tasks appear at all.
    # Deliberately NOT kmeans_random_state: that seed keys a persisted centroids
    # artifact, so sharing it would make "draw a different subset" require
    # refitting the basis. Two runs differing only here are both valid and not
    # comparable.
    fairness_residual_seed: int = 0

    # A row is a duplicate when its similarity to some strictly earlier row in
    # its cluster exceeds 1 - eps. Chosen TOGETHER with weights: under the
    # weighted identity a duplicate needs the weighted sum of per-block distances
    # to stay below eps, so at subtask=0.6, eps=0.01 the text distance alone must
    # fall under 0.017. The effective rule is a CONJUNCTION - duplicates describe
    # the same subtask AND look alike AND move alike - not a blend.
    #
    # The floor stays STRICT on the float branch, because eps=0.0 is a no-op
    # wearing a threshold's clothes: the retention test is a strict score >
    # 1 - eps, and a byte-identical pair scores exactly 1.0, so zero drops nothing
    # at all while still paying for the whole GPU pass. _MIN_DEDUP_EPS then lifts
    # that floor to one float32 step under 1.0, since a value beneath it is at
    # best an ulp away from wearing the same clothes for the same reason - see
    # _validate_dedup_eps_survives_float32.
    #
    # None is how a run asks for no de-duplication, and it is the escape the
    # strict floor points at: the stage is not run, no row can be marked
    # duplicate, and every row a retention pass would have flagged reaches the
    # fairness cut as a selection candidate instead. The fit still runs, because
    # curate_cluster_id and the within-group ordering come from it either way.
    dedup_eps: Annotated[float, Field(gt=0.0, le=1.0)] | None = 0.01

    # Cap on how many whole-GPU de-duplication tasks run at once: a scheduling
    # knob only, because the retention result is invariant to it. None lets the
    # scheduler use every free GPU.
    dedup_concurrency: int | None = Field(default=None, ge=1)

    # Similarity above which two TASK labels become one fairness group, so wording
    # variants of one instruction do not each draw their own quota. Level 1 only:
    # the distinct-task count is bounded by the annotation schema and does not move
    # with the row count, so a pairwise merge over the distinct labels is affordable
    # there whatever the corpus size. Level 2 is not merged at all - see
    # subtask_clusters.
    merge_theta_task: float = Field(default=0.95, ge=0.0, le=1.0)

    # The level-2 fairness group count: how many subtask clusters one k-means over
    # embedding_text_subtask partitions the corpus into. A BOUND, and that is the
    # whole point of the field - the level-2 key it replaced was annotator prose
    # resolved per shard, whose distinct-label count grew in proportion to N, so
    # the group count G grew with the corpus and the mean group held barely more
    # than one clip. A quota clamps to its group's capacity, so at that
    # cardinality no target below
    # the group count could fund any group twice and the allocation was decided
    # entirely by the tier's seeded residual order.
    #
    # TWO INDEPENDENT CEILINGS bound this value, and which one binds depends on
    # the target rather than on the corpus:
    #
    #   k <= target / merged_tasks   arithmetic: a parent whose budget is under
    #                                its child count has a level-2 fill line of
    #                                zero, so the seed alone picks its funded
    #                                cells (see fairness._water_fill)
    #   k << ~40                     semantic: one task is ESTIMATED to hold ~40
    #                                distinct subtask spellings, and a partition
    #                                finer than that ENUMERATES spellings instead
    #                                of grouping them - which is the free-form
    #                                key this field replaced, reached by another
    #                                route. Unlike the arithmetic bound this one
    #                                is not measured and the run cannot check it,
    #                                since curation never reads subtask_name
    #
    # 16 is chosen against the 250M-500M envelope, the same way
    # target_mean_cluster_rows is, and it fixes G at T * 16 - bounded by the task
    # vocabulary rather than by the corpus. At 250M rows that leaves every level-2
    # group capacity a quota can actually divide, and the arithmetic ceiling sits
    # in the thousands, so the semantic one is what holds 16 down. Raising the
    # value at all is unsupported:
    # the ~40 marks where the argument definitively breaks, not a licence to reach
    # it, and nothing establishes a benefit above the default.
    #
    # On a SMALL corpus 16 is thin: it leaves each group only a handful of clips,
    # and a scarce target then leaves a large share of those groups with nothing,
    # so which of them are funded is decided by fairness_residual_seed. Whether the
    # arithmetic ceiling binds there is a property of the TARGET, not of the small
    # corpus: it can sit above 16 on a small corpus at a generous target and below
    # it on a larger corpus at a scarce one - worked cases are tabulated in the
    # design doc cited below. The run warns once more
    # than a fifth of the level-2 groups are unfunded, and reports the count
    # and share at every value below that - but the warning is a backstop, not a
    # substitute for the arithmetic, because a k modestly above the ceiling lands
    # in a band that is degenerate and still silent. This is the same
    # small-corpus honesty target_mean_cluster_rows already has - a default tuned
    # to the target scale, with the degenerate regime reported rather than
    # hidden. See docs/curator/design/curator-next-curation.md.
    subtask_clusters: int = Field(default=16, ge=1)

    @model_validator(mode="after")
    def _validate_dedup_eps_survives_float32(self) -> Self:
        """Reject a ``dedup_eps`` smaller than one float32 step below 1.0.

        Kept apart from the ``gt=0.0`` bound because the two mistakes want
        opposite advice: ``0.0`` means the operator wanted ``None``, while a tiny
        positive value means they wanted de-duplication and would otherwise pay
        for the whole GPU pass to get almost none of it.

        Raises:
            ValueError: If ``dedup_eps`` is positive but under ``_MIN_DEDUP_EPS``.

        """
        if self.dedup_eps is not None and self.dedup_eps < _MIN_DEDUP_EPS:
            msg = (
                f"dedup_eps={self.dedup_eps!r} is below {_MIN_DEDUP_EPS!r}, one float32 step under 1.0 "
                f"and the smallest threshold this run accepts. Near it the cast sends 1 - eps back to "
                f"1.0f and the retention pass can mark nothing, so raise dedup_eps, or set it to null "
                f"to skip de-duplication."
            )
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _validate_clips_uri(self) -> Self:
        """Reject a blank, whitespace-only, or whitespace-padded table URI.

        Surrounding whitespace is rejected rather than silently trimmed so the
        stored value is exactly what the later table open sees; a padded URI
        would otherwise pass the emptiness check and open the wrong path.
        """
        stripped = self.clips_lance_uri.strip()
        if not stripped:
            msg = "clips_lance_uri must not be blank or whitespace-only"
            raise ValueError(msg)
        if stripped != self.clips_lance_uri:
            msg = "clips_lance_uri must not have leading or trailing whitespace"
            raise ValueError(msg)
        return self

    def result_defining_json(self) -> str:
        """Return the rules this run applies, as canonical JSON.

        Every field except those named in ``_NON_RESULT_DEFINING``, plus the
        code-defined contract under ``_CONTRACT_KEY``, with sorted keys and no
        incidental whitespace, so two runs agreeing on the rules produce
        byte-identical text whatever order their config files listed.

        The contract half is what stops a release that reorders the fused blocks
        or renames a reason from reporting the old and new runs as one identity;
        see ``contract_fingerprint``.

        Returns:
            Canonical JSON text; the input to ``result_defining_digest``.

        """
        # Every field is a primitive or a nested model today, so mode="json" is
        # byte-identical to the default and no test can pin it. It guards the
        # first field that is not - a Path, an enum, a datetime - which json.dumps
        # would reject outright rather than encode differently, so dropping it
        # fails loudly instead of re-basing the digest in silence.
        #
        # exclude takes a mutable set; the constant stays frozen.
        payload: dict[str, Any] = self.model_dump(mode="json", exclude=set(_NON_RESULT_DEFINING))
        payload[_CONTRACT_KEY] = contract_fingerprint()
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    def result_defining_digest(self) -> str:
        """Return ``sha256:<hex>`` over ``result_defining_json``.

        This is the run's identity: two versions carrying different digests are
        not comparable, and two carrying the same digest applied the same RULES -
        the config fields plus the declarations the contract half enumerates.
        That is narrower than "selected for the same thing": the fit-sample prefix
        rule decides which rows the locality basis is fitted on, so it moves every
        cluster id, and no config field or contract entry names it. The COMMIT
        separates two such runs anyway, on the ``centroids_fingerprint`` stamped
        beside this digest - a different prefix fits a different basis, whose
        bytes hash differently - so the two properties are read together; see
        ``pipeline._commit_properties``.

        The text this hashes is archived beside the table, so a reader can
        recompute the value and see WHERE two runs differ rather than only that
        they do.

        Returns:
            The digest, prefixed with its algorithm so a later change of hash is
            legible rather than silent.

        """
        digest = hashlib.sha256(self.result_defining_json().encode("utf-8")).hexdigest()
        return f"sha256:{digest}"


def _load_config_data(config_path: Path) -> dict[str, Any]:
    """Read one YAML or JSON config file into a mapping.

    Args:
        config_path: File to read; the suffix selects the parser.

    Returns:
        The file's top-level mapping, unvalidated.

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If the path is not a file, the file is not well-formed YAML or
            JSON, or its top level is not a mapping.

    """
    if not config_path.exists():
        msg = f"Config file not found: {config_path}"
        raise FileNotFoundError(msg)
    if not config_path.is_file():
        msg = f"Config path must be a file: {config_path}"
        raise ValueError(msg)
    # A parser's own error type is re-raised as ValueError because every caller
    # that presents config faults cleanly catches ValueError, not yaml's
    # exception tree. The routed CLI path never sees the difference -- it parses
    # the file once already, to read `kind` -- but a direct caller does.
    with config_path.open(encoding="utf-8") as handle:
        try:
            loaded: object = (
                yaml.safe_load(handle) if config_path.suffix.lower() in _YAML_SUFFIXES else json.load(handle)
            )
        except (yaml.YAMLError, json.JSONDecodeError) as error:
            msg = f"Config file is not well-formed: {config_path}: {error}"
            raise ValueError(msg) from error
    if not isinstance(loaded, dict):
        msg = f"Config file must contain a mapping at the top level, got {type(loaded).__name__}: {config_path}"
        raise ValueError(msg)  # noqa: TRY004 -- config callers catch ValueError, not TypeError
    return loaded


def resolve_config(
    config_path: str | Path,
    *,
    overrides: Sequence[str] = (),
) -> CurateConfig:
    """Load a config file, apply ``--set`` overrides, and validate the result.

    Overrides are applied to the loaded mapping BEFORE validation, so an
    overridden value is checked by the same rules as a written one - including
    the cross-field invariants, which a post-validation patch of a frozen model
    could not re-run. A dotted path reaches into a sub-model, so
    ``target.target_fraction=0.25`` sets the nested field.

    Args:
        config_path: YAML or JSON config file.
        overrides: ``PATH=VALUE`` assignments, e.g. ``["dedup_eps=0.02"]``.

    Returns:
        The validated run configuration.

    Raises:
        FileNotFoundError: If the config file does not exist.
        TypeError: If an override path passes through a non-mapping key.
        ValueError: If the path is not a regular file (for example a directory),
            the file is not well-formed YAML or JSON, its top level is not a
            mapping, an override is malformed, or validation fails
            (``ValidationError`` is a ``ValueError``).

    """
    data = _load_config_data(Path(config_path))
    apply_dotted_overrides(data, overrides)
    return CurateConfig.model_validate(data)
