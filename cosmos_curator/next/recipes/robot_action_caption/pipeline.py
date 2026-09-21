# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run one finite artifact-driven robot-action-caption attempt.

Source validation precedes schema widening, the read version is pinned after
widening, and planning precedes resource acquisition. One active writer owns
the caption column group; sibling recipes may still commit unrelated columns.

See ``docs/curator/design/curator-next-robot-action-caption.md`` for the
load-bearing order, no-work path, and writer-exclusivity contract.
"""

import attrs
import lance
from loguru import logger

from cosmos_curator.core.utils.storage.storage_utils import get_lance_storage_options
from cosmos_curator.models.registry import resolve_registered_model, validate_model_directory
from cosmos_curator.next.core.ray_runtime import (
    configure_ray_data_eager_actor_autoscaling,
    configure_ray_data_progress,
    configure_ray_data_stability,
    curator_io_resources,
    ensure_ray_initialized,
)
from cosmos_curator.next.recipes.robot_action_caption.config import ResolvedRobotActionCaptionConfig
from cosmos_curator.next.recipes.robot_action_caption.contracts import caption_contract_digest
from cosmos_curator.next.recipes.robot_action_caption.datasource import plan_caption_scan
from cosmos_curator.next.recipes.robot_action_caption.inference import run_inference_phase
from cosmos_curator.next.recipes.robot_action_caption.publication import publish_captions
from cosmos_curator.next.recipes.robot_action_caption.table import (
    CURATE_SELECTED_VALUE,
    CURATE_SELECTION_REASON,
    caption_work_filter,
    count_rows_in_fragments,
    ensure_caption_fields,
    require_selection_column,
    require_source_columns,
    resolve_context_fields,
)
from cosmos_curator.next.recipes.robot_action_caption.workspace import (
    CaptionWorkspace,
    cleanup_workspace,
    resolve_workspace,
)
from cosmos_curator.next.utils.arrow_utils import s3_compatibility_env_vars
from cosmos_curator.next.utils.lance_utils import report_table_shape

# The variant is a global registry key. Configuration validation catches an
# unknown key; the pipeline resolves the canonical model identity here for the
# digest and staged artifact path. Model/processor compatibility is deliberately
# tested by vLLM while it loads the registered artifact, not predicted by a
# recipe-local allow-list.
#
# Switching variant is a NEW CAPTION CONTRACT, not a retuning: the resolved model
# id reaches the digest, so rows captioned under one variant are stale under
# another and get recaptioned when they enter the configured input scope.


@attrs.frozen
class CaptionRunResult:
    """One attempt's canonical outcome, in values a caller can act on.

    Attributes:
        clips_lance_uri: The table read and widened.
        source_version: The version the attempt pinned and selected work from.
        ending_version: The version the table is canonical at when the attempt
            finished. Never null: an attempt that published returns the version
            its last wave landed at, and one that owed nothing returns the
            pinned version, which already includes the widening commit.
        published_rows: Caption rows published, of every terminal verdict.
        error_rows: How many of ``published_rows`` carry the ``error`` verdict, so
            a SUBSET of that count rather than an addition to it. A share
            approaching ``published_rows`` means a fault true of every clip was
            charged one row at a time; the attempt still succeeded, and a
            published error row is terminal under this contract, so this count is
            what tells that outcome apart from a corpus of captions.
        truncated_rows: How many carry the ``truncated`` verdict, a subset on the
            same terms and disjoint from ``error_rows``. A caption stopped at a
            generation bound is usable text, so the attempt succeeds either way
            and only this count separates a corpus of complete captions from one
            largely cut off. ``success`` needs no field: it is the remainder.
        contract_digest: The semantics that produced them.
        shard_count: Total shards the campaign divided the fragments across, or
            ``None`` when one job owned every fragment. Execution scope, not
            caption semantics, which is why it is absent from the digest and
            present here: the summary is the machine-readable record of what one
            job did, and several jobs of one campaign are otherwise
            distinguishable only by their row counts.
        shard_index: The zero-based shard this job owned, ``None`` on the same
            terms. Null together with ``shard_count`` or neither.

    """

    clips_lance_uri: str
    source_version: int
    ending_version: int
    published_rows: int
    error_rows: int
    truncated_rows: int
    contract_digest: str
    shard_count: int | None
    shard_index: int | None

    def as_summary(self) -> dict[str, object]:
        """Return the JSON-compatible run summary the CLI publishes."""
        return {
            "clips_lance_uri": self.clips_lance_uri,
            "source_version": self.source_version,
            "ending_version": self.ending_version,
            "published_rows": self.published_rows,
            "error_rows": self.error_rows,
            "truncated_rows": self.truncated_rows,
            "caption_contract_digest": self.contract_digest,
            "shard_count": self.shard_count,
            "shard_index": self.shard_index,
        }

    def as_message(self) -> str:
        """Return the one operator-facing line describing this outcome.

        Owned here rather than composed at each call site so the driver log and
        the CLI cannot describe one attempt in two ways.
        """
        # Branching on the row count, not on whether the version moved:
        # ``ending_version`` is never null, so it cannot tell a published attempt
        # apart from one that owed nothing.
        if not self.published_rows:
            return f"robot-action-caption: no caption owed at v{self.ending_version}"
        # Publication refuses neither verdict, so this line is where both become
        # visible: the error count separates a captioned corpus from one that
        # errored identically on every clip, and the truncated count separates it
        # from one whose captions were cut off. Both commit, so nothing else
        # reports them.
        return (
            f"robot-action-caption: {self.published_rows} caption row(s) canonical "
            f"at v{self.ending_version} ({self.error_rows} error, {self.truncated_rows} truncated)"
        )


def run_config(config: ResolvedRobotActionCaptionConfig) -> CaptionRunResult:
    """Run one finite caption attempt against one pinned snapshot.

    Validation and model/staging faults raise before inference when possible.
    Durable schema, staged rows, and checkpoints are not rolled back; a retry
    reuses the same namespace. See the design document's failure and resume
    contract.

    Args:
        config: The resolved run configuration.

    Returns:
        The attempt's outcome.

    Raises:
        ValueError: If Lance reports a missing clip table or validation fault,
            or on a data-contract fault, staged-row fault, or failed commit.
        FileNotFoundError: If required model files are absent.
        OSError: If Lance reports an access or storage failure.

    """
    uri = config.input.clips_lance_uri
    storage_profile = config.execution.storage_profile
    storage_options = get_lance_storage_options(uri, profile_name=storage_profile)
    model = resolve_registered_model(config.model.variant)

    # Preserve native open exceptions so access, backend, missing-table, and
    # validation failures retain the dependency's available taxonomy.
    dataset = lance.dataset(uri, storage_options=storage_options)
    require_source_columns(dataset)
    require_selection_column(dataset, selected_only=config.input.selected_only)
    # Rebound, not called for effect: the widening opens its own handle so it can
    # retry a lost schema race, and the version pinned below must be the one the
    # winning pass left behind rather than this now-stale handle's.
    dataset = ensure_caption_fields(uri, storage_options=storage_options)
    context_fields = resolve_context_fields(dataset)
    read_version = int(dataset.version)
    shard = config.input.shard
    if shard is None:
        scope_label = "unsharded"
    else:
        scope_label = f"shard {shard.label}"
        # Stated once, because the two numbers are easy to read the other way
        # round and a campaign launched one-based would silently leave the last
        # shard's fragments owned by nobody.
        logger.info(
            f"Caption work {scope_label} on {uri}: this job owns the fragments whose manifest id "
            f"mod {shard.count} equals {shard.index}. Indices are zero-based, so a complete "
            f"campaign launches indices 0 through {shard.count - 1}."
        )
    # The WHOLE table's geometry, whatever this job owns: it is the only line
    # that records what the campaign was run against, and it is not recoverable
    # from any later line once the run is over. This job's own geometry - the
    # ceiling on its fetch concurrency, and the largest fragment its final block
    # spends its time on - is the owed distribution logged after planning.
    report_table_shape(dataset, f"Caption source table {uri}")

    digest = caption_contract_digest(
        model,
        context_fields,
        generation=config.generation,
        video=config.video,
        media=config.media,
    )
    row_filter = caption_work_filter(
        digest,
        model.model_id,
        selected_only=config.input.selected_only,
    )
    # The predicate itself, not just the scope name below: it is the only place
    # the selection clause and the pending clause are visible exactly as Lance
    # evaluates them, and every owed count in this function is derived from this
    # one string. Logged BEFORE planning so a filter Lance refuses to parse is
    # still readable from the log that reports the failure.
    logger.info(f"Caption row filter on {uri} @v{read_version}: {row_filter}")
    plan = plan_caption_scan(dataset, row_filter, shard=shard)
    row_scope = "selected-only" if config.input.selected_only else "all-pending"
    # Every number here is relative to what this job owns, which is why the
    # owned total is reported beside the owed one: on a sharded run "12 owed
    # fragments" alone says nothing about whether this job has a small scope or
    # a nearly-complete one. The largest owed fragment sizes the tail, since one
    # Ray Data block reads one owed fragment.
    logger.info(
        f"Caption attempt on {uri} @v{read_version} ({scope_label}): {plan.owned_fragments} owned fragment(s), "
        f"{len(plan.fragments)} of them owed, {plan.num_rows} owed row(s), "
        f"largest owed fragment {max((fragment.num_rows for fragment in plan.fragments), default=0)} row(s), "
        f"under {row_scope} scope and contract {digest}."
    )
    if not plan.fragments:
        if config.input.selected_only:
            unscoped_filter = caption_work_filter(digest, model.model_id, selected_only=False)
            # Counted over THIS JOB's fragments, never the table: an unowned
            # backlog belongs to whichever job owns it, and blaming Curate for
            # it sends the operator to a remedy that would change nothing. It is
            # also the only unbounded driver cost on an otherwise cheap path.
            unselected_pending = count_rows_in_fragments(dataset, plan.owned_fragment_ids, unscoped_filter)
            if unselected_pending:
                logger.warning(
                    f"Caption attempt on {uri} @v{read_version} ({scope_label}) owes nothing under "
                    f"selected-only scope, but {unselected_pending} row(s) in the fragments this job owns are "
                    f"caption-pending: no such pending row carries {CURATE_SELECTION_REASON} = "
                    f"{CURATE_SELECTED_VALUE!r}. Run the curate leg first, or set input.selected_only=false to "
                    "caption every pending row."
                )
        # Returned before model validation, workspace, Ray, or initialization is
        # touched. Publication is skipped rather than called with an empty wave,
        # and it depends on that: with nothing staged there is nothing it could
        # validate, so it carries no empty-wave path and this is the only place
        # the case is answered.
        no_work = CaptionRunResult(
            clips_lance_uri=uri,
            source_version=read_version,
            ending_version=read_version,
            published_rows=0,
            error_rows=0,
            truncated_rows=0,
            contract_digest=digest,
            shard_count=None if shard is None else shard.count,
            shard_index=None if shard is None else shard.index,
        )
        logger.info(f"{no_work.as_message()}; {no_work.as_summary()}")
        return no_work

    validate_model_directory(model.runtime_model_dir, model_key=model.key)
    workspace = resolve_workspace(
        staging_root=config.staging_root,
        source_uri=uri,
        digest=digest,
        storage_profile=storage_profile,
        shard=shard,
    )

    ensure_ray_initialized(
        local_resources=curator_io_resources(),
        # Both locations, not just the staging root: the clip table is
        # configured independently of it and a worker scans that table's
        # fragments itself, so it can reach an endpoint the staging root does not.
        runtime_env_vars=s3_compatibility_env_vars(
            uri,
            workspace.root_uri,
            storage_profile=storage_profile,
        ),
    )
    configure_ray_data_progress(progress=config.execution.progress)
    configure_ray_data_stability(disable_high_memory_detector=True)
    configure_ray_data_eager_actor_autoscaling()
    run_inference_phase(
        dataset,
        plan,
        config,
        model,
        workspace,
        context_fields=context_fields,
        digest=digest,
        row_filter=row_filter,
        storage_options=storage_options,
    )

    publication = publish_captions(
        dataset,
        workspace,
        plan.fragment_ids,
        digest=digest,
        model_id=model.model_id,
        row_filter=row_filter,
        storage_options=storage_options,
    )
    # Unguarded because publication either commits or raises, so reaching this
    # line already means the captions are canonical. That ordering is what the
    # cleanup rests on: before a commit the staged rows are the sole copy of
    # completed GPU work, and generation is unseeded, so deleting them would not
    # cost a re-run - it would cost different captions.
    _cleanup(workspace)

    result = CaptionRunResult(
        clips_lance_uri=uri,
        source_version=read_version,
        ending_version=publication.ending_version,
        published_rows=publication.published_rows,
        error_rows=publication.error_rows,
        truncated_rows=publication.truncated_rows,
        contract_digest=digest,
        shard_count=None if shard is None else shard.count,
        shard_index=None if shard is None else shard.index,
    )
    logger.info(f"{result.as_message()}; {result.as_summary()}")
    return result


def _cleanup(workspace: CaptionWorkspace) -> None:
    """Remove the published namespace, never failing the run for it.

    Swallowed rather than raised because the captions are already canonical, so
    failing here would cost an investigation into a run that actually succeeded.
    A leftover workspace is not free - the namespace is version-independent, so a
    later attempt can resolve to it - but it cannot corrupt a published caption,
    only refuse an attempt that re-owns one of its fragments.
    """
    try:
        cleanup_workspace(workspace)
    except Exception as e:  # noqa: BLE001 - the captions are canonical; the leftover can only refuse a later attempt
        logger.warning(
            f"Captions are canonical, but cleanup of {workspace.root_uri} failed ({e!r}); "
            f"delete it by hand. A later attempt on this table, contract and scope resolves to "
            f"the same workspace, and if it owes a row in a fragment holding these leftover "
            f"staged rows, publication refuses before committing until the workspace is gone."
        )
