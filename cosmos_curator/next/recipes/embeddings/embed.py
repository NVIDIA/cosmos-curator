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

"""Direct-``clips.lance`` embedding driver: plan every modality, then stage and publish each.

The ORDER below is this module's whole contribution, and it is a correctness
requirement rather than a tidy arrangement. Planning must precede weight
acquisition so a current table pays for neither; weight acquisition must precede
Ray initialization because ``download_models`` forks a child that runs and tears
down its own Ray, which a process that has already initialized Ray cannot do;
and Ray initialization must precede every Ray Data execution because the AWS
compatibility mode a worker needs to write a GCS-backed staging root is a
per-process environment variable set at ``ray.init`` time::

    open clips.lance -> ensure_embedding_columns(enabled groups) -> report shape
        |
        v
    PLAN ALL (driver-side Lance only: one filtered count per fragment)
        |                                       text and image
        v
    _stage_weights(owed text / image fills)      ONCE, pre-fork
        |
        v
    ensure_ray_initialized(s3 compatibility for the table AND the staging root)
        |
        v
    for text, then image:  owed? -> workspace -> stage -> publish -> cleanup
        |
        v
    action:  workspace(descriptor contract) -> staged fingerprints -> basis
        |        -> fill -> plan -> stage -> publish -> cleanup -> outcome check
        v
    EmbeddingRunResult(one ModalityResult per enabled modality, the basis, and
                       the widening's version when it added anything)

A modality is DATA (a ``ModalityFill`` built by a small function), so text and
image share one loop with no branch. Action is the one modality with genuinely
different BEHAVIOUR - it must bind one PCA basis before any worker exists, and
that basis is only discoverable from the staging namespace its own descriptor
contract keys - so it gets the explicit branch above rather than a lifecycle hook.

Modalities stay sequential. Ray Data already fans each one across the cluster,
and Lance treats two ``Update`` commits touching one fragment as conflicting even
on disjoint column groups, so each modality opens its own handle AFTER the
previous one committed. That re-open is why the driver passes an open dataset
around rather than a URI: which version a step reads is visible at its call site.

Nothing here is a run MODE. An ordinary run computes the rows whose group is
missing or whose stored identity names a different computation, so a top-up
embeds the new clips, a model change rebuilds the rows it invalidated, and an
unchanged re-run plans nothing, starts nothing, and commits nothing.
"""

import time
from collections.abc import Callable, Sequence

import attrs
import lance
from loguru import logger

from cosmos_curator.core.interfaces.pipeline_interface import download_models
from cosmos_curator.core.utils.storage.storage_utils import get_lance_storage_options
from cosmos_curator.next.core.ray_runtime import configure_ray_data_stability, ensure_ray_initialized
from cosmos_curator.next.embeddings.action.pca import action_pca_root_uri
from cosmos_curator.next.embeddings.schemas import ACTION_COLUMN_GROUP, EmbeddingColumnGroup
from cosmos_curator.next.recipes.embeddings.action_pca import (
    ActionPca,
    action_recovery_contract,
    check_action_outcome,
    resolve_action_pca,
    staged_action_fingerprints,
)
from cosmos_curator.next.recipes.embeddings.columns import ensure_embedding_columns, pending_filter
from cosmos_curator.next.recipes.embeddings.config import EmbeddingPipelineConfig, Modality
from cosmos_curator.next.recipes.embeddings.datasource import EmbeddingScanPlan, plan_embedding_scan
from cosmos_curator.next.recipes.embeddings.fill import stage_embedding_group
from cosmos_curator.next.recipes.embeddings.modalities import (
    ModalityFill,
    ModalityResult,
    build_action_fill,
    build_image_fill,
    build_text_fill,
)
from cosmos_curator.next.recipes.embeddings.publication import publish_embedding_group
from cosmos_curator.next.recipes.embeddings.workspace import (
    EmbeddingWorkspace,
    cleanup_workspace,
    default_staging_root,
    resolve_workspace,
)
from cosmos_curator.next.utils.arrow_utils import s3_compatibility_env_vars
from cosmos_curator.next.utils.identity import canonical_digest
from cosmos_curator.next.utils.lance_utils import open_dataset_or_raise, report_table_shape

# The modalities whose whole run is described by their spec, in execution order.
# Action is absent by construction, not by omission: its fill cannot be built
# until a basis is resolved, so it needs the explicit branch below rather than a
# row in this table.
_GENERIC_FILL_BUILDERS: tuple[tuple[Modality, Callable[[EmbeddingPipelineConfig], ModalityFill]], ...] = (
    (Modality.TEXT, build_text_fill),
    (Modality.IMAGE, build_image_fill),
)


@attrs.frozen(eq=False)
class _PlannedModality:
    """One spec-driven modality and the work it owes, decided before any weights or Ray.

    Pairing them is what lets weights be staged for exactly the modalities that
    will run: the decision needs both the plan (is anything owed) and the fill
    (which checkpoint). Holding them apart in two lists would let the two drift
    out of alignment.

    ``eq=False`` for the same reason as ``ModalityFill``: a fill's kwargs can hold
    a numpy-bearing basis, and a generated ``__eq__`` would compare those arrays.

    Attributes:
        fill: What this modality computes and where it lands.
        plan: The owed fragments at the planning version. Empty means the
            modality is already current.

    """

    fill: ModalityFill
    plan: EmbeddingScanPlan


@attrs.frozen(kw_only=True)
class EmbeddingRunResult:
    """Verification summary for one run: per-modality counts, the versions committed, and the action basis.

    Keyword-only so that inserting a field cannot silently re-bind an existing
    positional argument at a call site.

    Attributes:
        modalities: One result per enabled modality, in text -> image -> action
            order (including any that embedded nothing).
        action: The PCA basis the action modality used for staging or publication
            this run, or ``None`` when action was not enabled, no clip carried
            action data, or the owed-work plan was empty. Carried here rather
            than on a ``ModalityResult`` because the fingerprint and the fit /
            reuse distinction are action-specific.
        schema_commit_version: The version the schema widening committed, or
            ``None`` when every enabled group was already present so nothing was
            added. Always the run's FIRST commit, hence never later than a
            modality's.

    """

    modalities: tuple[ModalityResult, ...]
    action: ActionPca | None
    schema_commit_version: int | None

    @property
    def ending_version(self) -> int | None:
        """Return the version of this run's last commit, or ``None`` if it committed nothing.

        Projected from the commits already recorded rather than stored beside
        them, so it cannot drift from them. The widening precedes every fill, so
        the last fill's commit is the run's last whenever a fill committed, and
        the widening's is when none did.
        """
        committed = [result.committed_version for result in self.modalities if result.committed_version is not None]
        return committed[-1] if committed else self.schema_commit_version

    @property
    def committed_only_the_widening(self) -> bool:
        """Return whether the schema widening was this run's one and only commit."""
        return self.schema_commit_version is not None and all(
            result.committed_version is None for result in self.modalities
        )


def run_embedding_pipeline(config: EmbeddingPipelineConfig) -> EmbeddingRunResult:
    """Bring every enabled embedding column group of the shared ``clips.lance`` up to date.

    Args:
        config: Fully resolved run configuration.

    Returns:
        A verification summary (per-modality counts, the versions committed, and
        the action basis; never vectors).

    Raises:
        ValueError: If the table is absent or unreadable, a group's stored shape
            differs from the one it declares, the staging root is unusable, two
            action bases are in play, a worker violated the embedder contract, a
            staged partition is unusable, or a commit fails.

    """
    storage_options = get_lance_storage_options(config.clips_lance_uri, profile_name=config.storage_profile)
    # Derived rather than required, so a crashed run resumes without having been
    # configured to; the config field only moves the transient rows elsewhere.
    staging_root = config.staging_root_uri or default_staging_root(config.clips_lance_uri)
    fills = _generic_fills(config)
    logger.info(
        f"embedding clips at {config.clips_lance_uri}; modalities={list(config.modalities)}; "
        f"staging under {staging_root}"
    )

    dataset = _open_clips(config, storage_options=storage_options)
    added, schema_commit_version = ensure_embedding_columns(dataset, _enabled_groups(fills, config))
    logger.info(f"ensured embedding columns for {list(config.modalities)}: {added} field(s) added")
    # Logged because staging reads one Ray Data block per owed fragment, so this
    # geometry is the ceiling on staging concurrency and its largest fragment is
    # the one every other actor waits on. Read after the widening, so the reported
    # version and field count name the schema the modalities scan.
    report_table_shape(dataset, f"embedding source {config.clips_lance_uri}")

    planned = tuple(
        _PlannedModality(fill=fill, plan=_plan_modality(dataset, fill, max_fragments=config.max_fragments))
        for fill in fills
    )
    _stage_weights([entry.fill for entry in planned if entry.plan.fragments], config)
    if _will_execute(planned, config):
        ensure_ray_initialized(
            # Both locations, not just the staging root: the table is configured
            # independently of it and a worker scans the table's fragments itself,
            # so it can reach an endpoint the staging root does not.
            runtime_env_vars=s3_compatibility_env_vars(
                config.clips_lance_uri,
                staging_root,
                storage_profile=config.storage_profile,
            ),
        )
        # The publication phase is the reason this is not optional. A staging
        # read fault becomes a refusal payload and a compute fault fails the
        # attempt for the checkpoint to resume, but a transient fault while
        # writing a fragment's column file has no row to become, so without
        # the retried-error family it ends a phase that had already succeeded
        # everywhere else.
        #
        # Both optional knobs are left at Ray's settings, unlike the media legs
        # that disable the high-memory detector: image is the one modality here
        # declaring a memory reservation, and a warning when it exceeds that is
        # worth having. A task is one batch or one fragment either way, so Ray's
        # no-progress window reads a wedged task rather than a slow one.
        configure_ray_data_stability()

    results = [
        _run_planned_modality(entry, config=config, staging_root=staging_root, storage_options=storage_options)
        for entry in planned
    ]
    action: ActionPca | None = None
    if Modality.ACTION in config.modalities:
        action_result, action = _run_action_modality(config, staging_root=staging_root, storage_options=storage_options)
        results.append(action_result)
    return EmbeddingRunResult(modalities=tuple(results), action=action, schema_commit_version=schema_commit_version)


def _generic_fills(config: EmbeddingPipelineConfig) -> tuple[ModalityFill, ...]:
    """Build the enabled spec-driven modalities' fills, in execution order."""
    return tuple(build(config) for modality, build in _GENERIC_FILL_BUILDERS if modality in config.modalities)


def _enabled_groups(fills: Sequence[ModalityFill], config: EmbeddingPipelineConfig) -> list[EmbeddingColumnGroup]:
    """Return every enabled modality's column group, so one commit widens them all.

    Action's group is named directly rather than read off a fill: its fill does
    not exist yet (it needs a basis, and resolving one reads the group's own
    provenance columns), so the columns must be added first.
    """
    groups = [fill.group for fill in fills]
    if Modality.ACTION in config.modalities:
        groups.append(ACTION_COLUMN_GROUP)
    return groups


def _plan_modality(dataset: lance.LanceDataset, fill: ModalityFill, *, max_fragments: int | None) -> EmbeddingScanPlan:
    """Count what one modality owes at the version ``dataset`` is open on.

    Driver-side Lance only - one filtered count per fragment over the predicate's
    own columns, no weights, no Ray - which is what lets the run decide whether it
    needs either. That count is a read rather than a metadata lookup, so its
    duration is logged: on a run that owes nothing it is the whole run's cost.

    Args:
        dataset: The clips table, open at the planning version.
        fill: The modality whose owed rows are counted.
        max_fragments: Optional cap on owed fragments, applied after the current
            ones are dropped.

    Returns:
        The owed fragments. Empty means the modality is already current.

    """
    # Built from the fill rather than passed in: staging re-derives the same
    # predicate from the same fill, so the plan and the scan cannot disagree.
    row_filter = pending_filter(fill.group, fill.applicability_filter, fill.result_identity)
    started = time.monotonic()
    plan = plan_embedding_scan(dataset, row_filter=row_filter, max_fragments=max_fragments)
    elapsed_s = time.monotonic() - started
    logger.info(
        f"{fill.modality} owes {plan.num_rows} row(s) across {len(plan.fragments)} fragment(s) of "
        f"{dataset.uri} at v{int(dataset.version)} under result identity {fill.result_identity} "
        f"(planned in {elapsed_s:.1f}s of corpus-wide filtered counting)"
    )
    return plan


def _stage_weights(fills: Sequence[ModalityFill], config: EmbeddingPipelineConfig) -> None:
    """Download the checkpoints the modalities that will RUN need, before Ray exists.

    ``download_models`` forks a child that initialises and tears down its own Ray,
    so it must run from a process that has never initialised Ray. That is what
    makes per-modality just-in-time staging impossible rather than merely
    inconvenient: image's staging would fork after text's Ray Data execution had
    already initialised Ray in this process.

    Args:
        fills: The modalities that owe work. A modality that owes nothing is
            excluded by the caller, so a current table downloads nothing.
        config: Supplies the weights base ``download_models`` reads from.

    """
    names = [fill.weights_name for fill in fills if fill.weights_name is not None]
    if not names:
        return
    logger.info(f"staging weights for {names} from {config.model_weights_path}")
    download_models(names, config.model_weights_path)


def _will_execute(planned: Sequence[_PlannedModality], config: EmbeddingPipelineConfig) -> bool:
    """Return whether this run may reach a Ray Data execution at all.

    An unchanged re-run must start no Ray, which is the common case on a
    scheduled pipeline. Action counts as "may execute" whatever its rows look
    like, because its owed work is not knowable until a basis is bound and
    binding one can require the distributed candidate-extraction pass; a run with
    action enabled therefore pays one ``ray.init`` even when it turns out to owe
    nothing.
    """
    return any(entry.plan.fragments for entry in planned) or Modality.ACTION in config.modalities


def _run_planned_modality(
    entry: _PlannedModality,
    *,
    config: EmbeddingPipelineConfig,
    staging_root: str,
    storage_options: dict[str, str] | None,
) -> ModalityResult:
    """Stage and publish one spec-driven modality, or report that it owed nothing."""
    fill = entry.fill
    if not entry.plan.fragments:
        return ModalityResult.nothing_to_do(fill.modality, dict(fill.result_identity))
    # Re-opened rather than reusing the widening handle, so this modality reads -
    # and commits against - the version the previous one committed. Lance treats
    # two Update commits touching one fragment as conflicting even on disjoint
    # column groups, so a stale read version would be rejected. The plan stays
    # valid across that re-open: an Update rewrites a fragment's column files
    # without changing fragment ids or row counts. That holds for the FILTERED
    # count publication now measures against too, because this modality's owed
    # predicate reads only its own group's columns, which a sibling group's
    # Update does not touch.
    dataset = _open_clips(config, storage_options=storage_options)
    workspace = resolve_workspace(
        staging_root=staging_root,
        source_uri=dataset.uri,
        read_version=int(dataset.version),
        modality=fill.modality,
        # The full identity IS the recovery contract here: every input to it is
        # fixed by the spec before the run starts, so there is no circularity to
        # break (which is exactly what action cannot say).
        recovery_contract=canonical_digest(fill.result_identity),
        group=fill.group,
        storage_profile=config.storage_profile,
    )
    return _stage_and_publish(
        dataset,
        fill,
        plan=entry.plan,
        workspace=workspace,
        rows_per_file=config.parquet_rows_per_file,
        storage_options=storage_options,
    )


def _run_action_modality(
    config: EmbeddingPipelineConfig, *, staging_root: str, storage_options: dict[str, str] | None
) -> tuple[ModalityResult, ActionPca | None]:
    """Bind the run's one PCA basis, then stage, publish, and judge the action group.

    The workspace is resolved BEFORE the basis and keyed on the descriptor
    contract, because the staged rows of an interrupted attempt are where the
    basis it already committed to is read from. Keying on the final identity
    would need the fingerprint that read exists to find.

    The ``None`` basis is load-bearing rather than defensive: with no clip
    carrying action data there is nothing to load and nothing to fit, so the run
    reports a skip WITHOUT starting an actor pool.

    Returns:
        The action modality's result, and the basis for ``EmbeddingRunResult``
        (``None`` when no clip carries action data or the owed-work plan is
        empty).

    """
    dataset = _open_clips(config, storage_options=storage_options)
    workspace = resolve_workspace(
        staging_root=staging_root,
        source_uri=dataset.uri,
        read_version=int(dataset.version),
        modality=Modality.ACTION,
        recovery_contract=action_recovery_contract(),
        group=ACTION_COLUMN_GROUP,
        storage_profile=config.storage_profile,
    )
    pca = resolve_action_pca(
        dataset,
        config,
        root_uri=action_pca_root_uri(config.clips_lance_uri),
        staged_fingerprints=staged_action_fingerprints(workspace),
    )
    if pca is None:
        # No computation could be identified, which is why the identity is empty
        # rather than merely unreported: there is no basis, so no vector this run
        # could have produced has a name.
        return ModalityResult.nothing_to_do(Modality.ACTION, {}), None

    fill = build_action_fill(config, pca.artifact)
    plan = _plan_modality(dataset, fill, max_fragments=config.max_fragments)
    if not plan.fragments:
        return ModalityResult.nothing_to_do(Modality.ACTION, dict(fill.result_identity)), None
    result = _stage_and_publish(
        dataset,
        fill,
        plan=plan,
        workspace=workspace,
        rows_per_file=config.parquet_rows_per_file,
        storage_options=storage_options,
    )
    # Re-opened, so the "the group holds nothing at all" test sees this run's own
    # commit rather than the pre-publication version the plan was built against.
    check_action_outcome(result, _open_clips(config, storage_options=storage_options))
    return result, pca


def _stage_and_publish(  # noqa: PLR0913 - each argument is an independent input to one modality's run
    dataset: lance.LanceDataset,
    fill: ModalityFill,
    *,
    plan: EmbeddingScanPlan,
    workspace: EmbeddingWorkspace,
    rows_per_file: int,
    storage_options: dict[str, str] | None,
) -> ModalityResult:
    """Compute one modality's owed rows, publish them in one commit, then drop the workspace."""
    stage_embedding_group(
        dataset,
        fill,
        workspace=workspace,
        plan=plan,
        rows_per_file=rows_per_file,
        storage_options=storage_options,
    )
    result = publish_embedding_group(
        dataset,
        fill,
        workspace=workspace,
        plan=plan,
        storage_options=storage_options,
    )
    # Unguarded on purpose: publication either commits or raises, so reaching this
    # line already means the vectors are canonical and the staged copy is
    # redundant. That ordering is what the cleanup rests on - before the commit
    # the staged rows are the only copy of completed GPU work.
    _cleanup(workspace)
    return result


def _cleanup(workspace: EmbeddingWorkspace) -> None:
    """Remove the published namespace, never failing the run for it.

    Swallowed rather than raised because the vectors are already canonical: a
    stale workspace costs storage, whereas failing here would cost an
    investigation into a run that actually succeeded. It is also why the results
    and the checkpoint are deleted TOGETHER by one call - a checkpoint surviving
    deleted results would tell the next run its rows are done while nothing holds
    them.
    """
    try:
        cleanup_workspace(workspace)
    except Exception as e:  # noqa: BLE001 - the vectors are canonical; a stale workspace only costs storage
        logger.warning(
            f"Embeddings are canonical, but cleanup of {workspace.root_uri} failed ({e!r}); "
            f"nothing reads the workspace, so delete it at leisure."
        )


def _open_clips(config: EmbeddingPipelineConfig, *, storage_options: dict[str, str] | None) -> lance.LanceDataset:
    """Open the clips table at its latest version, raising if it is absent."""
    return open_dataset_or_raise(config.clips_lance_uri, storage_options=storage_options)
