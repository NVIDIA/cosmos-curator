# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reconcile recovery workspaces before selecting an enrichment attempt.

One active writer owns a field group and its compatible workspace namespace.
An unfinished selection is never expanded by appends or a changed batch limit.
Completed workspaces can be removed only when their receipt and canonical files
still prove publication. Missing historical proof retains the workspace without
blocking unrelated work. The recipe retains ownership of its Ray execution.
"""

from enum import Enum, auto

import attrs
import lance
from loguru import logger

from cosmos_curator.next.enrichment import commit, contracts, inputs
from cosmos_curator.next.enrichment.transactions import find_transaction
from cosmos_curator.next.enrichment.workspace import RecoveryStateError, RecoveryWorkspace


@attrs.frozen
class EnrichmentAttempt:
    """One frozen selection of a validated input and the recovery workspace that belongs to it.

    ``pending`` holds every fragment of the pinned input whose output fields are
    still pending; ``selected`` is the subset this attempt processes and publishes.
    A workspace exists exactly when fragments are selected, and its manifest is
    already durable. Create attempts with ``reconcile_and_select_attempt``.
    """

    input: inputs.EnrichmentInput = attrs.field(eq=False, repr=False)
    pending: tuple[int, ...]
    selected: tuple[int, ...]
    workspace: RecoveryWorkspace | None
    cleanup_complete: bool

    def processing_complete(self) -> bool:
        """Report a valid processing marker, which lets a retry skip straight to publication."""
        return self.workspace is not None and self.workspace.processing_complete(
            storage_profile=self.input.bindings.execution.storage_profile
        )

    def counts(self) -> dict[str, int]:
        """Count rows and fragments of the pinned input, excluding later appends.

        Already-terminal rows count toward ``input_rows`` only. Selected rows
        include results recovered from checkpoints and declared terminal errors.
        """
        rows = {
            int(fragment.fragment_id): int(fragment.count_rows()) for fragment in self.input.dataset.get_fragments()
        }
        remaining = set(self.pending) - set(self.selected)
        return {
            "input_rows": sum(rows.values()),
            "selected_rows": sum(rows[fragment_id] for fragment_id in self.selected),
            "remaining_rows": sum(rows[fragment_id] for fragment_id in remaining),
            "selected_fragments": len(self.selected),
            "remaining_fragments": len(remaining),
        }


class _SelectionState(Enum):
    UNFINISHED = auto()
    VERIFIED = auto()
    UNPROVEN = auto()


def _recorded_selection(workspace: RecoveryWorkspace) -> tuple[int, ...]:
    fragments = workspace.manifest.get("input_fragments")
    if not isinstance(fragments, list) or any(
        not isinstance(fragment, dict)
        or type(fragment.get("fragment_id")) is not int
        or fragment["fragment_id"] < 0
        or not isinstance(fragment.get("file_layout"), str)
        for fragment in fragments
    ):
        msg = f"Invalid fragment selection in recovery workspace: {workspace.root}. Remove the workspace."
        raise RecoveryStateError(msg)
    selected = tuple(fragment["fragment_id"] for fragment in fragments)
    if selected != tuple(sorted(set(selected))):
        msg = f"Recovery workspace selection is not sorted and unique: {workspace.root}. Remove the workspace."
        raise RecoveryStateError(msg)
    return selected


def _verify_receipt(
    dataset: lance.LanceDataset,
    spec: contracts.EnrichmentSpec,
    workspace: RecoveryWorkspace,
    selected: tuple[int, ...],
) -> bool:
    """Verify the canonical owned files, returning false when no receipt exists.

    Pending selection already validated the schema, and workspace reconciliation
    the selected input file layout.
    """
    transaction = find_transaction(
        dataset, property_key=inputs.PUBLICATION_RECEIPT_KEY, property_value=workspace.manifest_digest
    )
    if transaction is None:
        return False
    owned = set(inputs.field_ids(dataset, spec.output_fields.names).values())
    prepared = inputs.receipt_fragments(transaction, owned)
    if prepared is None or set(prepared) != set(selected):
        msg = (
            "Enrichment recovery receipt does not update exactly the owned fields of its recorded selection: "
            f"{workspace.root}. Inspect the table's transaction history; the workspace is retained."
        )
        raise RecoveryStateError(msg)
    # These are the immutable files validated before the receipted publication.
    commit.verify_result_files(dataset, spec, tuple(prepared.values()))
    return True


def _selection_state(
    dataset: lance.LanceDataset,
    spec: contracts.EnrichmentSpec,
    workspace: RecoveryWorkspace,
    selected: tuple[int, ...],
    pending: set[int],
) -> _SelectionState:
    """Separate pending work, proven publication, and terminal work lacking proof."""
    remaining = pending.intersection(selected)
    if remaining:
        if remaining != set(selected):
            msg = (
                f"Recovery selection is partially published; overlapping writers are unsupported: {workspace.root}. "
                "Confirm no other writer owns this field group, then remove the workspace."
            )
            raise RecoveryStateError(msg)
        return _SelectionState.UNFINISHED
    if _verify_receipt(dataset, spec, workspace, selected):
        return _SelectionState.VERIFIED
    commit.verify_completed(original=dataset, current=dataset, spec=spec, selected=selected)
    return _SelectionState.UNPROVEN


@attrs.frozen
class _Reconciled:
    active: list[tuple[tuple[int, ...], RecoveryWorkspace]]
    completed: list[RecoveryWorkspace]
    unproven: list[RecoveryWorkspace]


def _reconcile(enrichment_input: inputs.EnrichmentInput, pending: tuple[int, ...]) -> _Reconciled:
    """Classify every discovered workspace before any cleanup changes durable state."""
    dataset, bindings = enrichment_input.dataset, enrichment_input.bindings
    reconciled = _Reconciled([], [], [])
    prefix = inputs.workspace_prefix(dataset, bindings)
    for workspace in RecoveryWorkspace.discover(prefix, storage_profile=bindings.execution.storage_profile):
        selected = _recorded_selection(workspace)
        if inputs.resolve_workspace(dataset, bindings, selected=selected) != workspace:
            msg = (
                f"Recovery workspace does not match the current input file layout: {workspace.root}. "
                "Its selected input changed, which shared recovery does not support; "
                "remove the workspace to discard its staged results."
            )
            raise RecoveryStateError(msg)
        state = _selection_state(dataset, bindings.spec, workspace, selected, set(pending))
        if state is _SelectionState.UNFINISHED:
            reconciled.active.append((selected, workspace))
        elif state is _SelectionState.VERIFIED:
            reconciled.completed.append(workspace)
        else:
            reconciled.unproven.append(workspace)
    if len(reconciled.active) > 1:
        roots = ", ".join(workspace.root for _, workspace in reconciled.active)
        msg = (
            f"Multiple unfinished recovery selections exist for this field group: {roots}. "
            "Each may belong to another writer; remove abandoned workspaces before rerunning."
        )
        raise RecoveryStateError(msg)
    return reconciled


def reconcile_and_select_attempt(
    enrichment_input: inputs.EnrichmentInput,
    *,
    max_fragments: int | None = None,
) -> EnrichmentAttempt:
    """Clean verified workspaces, then resume saved work or select a new attempt.

    Validate every discovered workspace before attempting to delete any whose
    publication is verified. Cleanup is best effort and reported in
    ``cleanup_complete``. The returned workspace's manifest is durable before
    processing starts.

    Pending fragments come from the pinned input's output values and retained
    receipts. The limit applies only to a new selection; recovery retains its
    recorded membership. Multiple unfinished selections are rejected because this
    helper neither assigns concurrent writers nor merges their checkpoint stores.
    Terminal selections with matching inputs but no retained receipt remain on disk
    and report incomplete cleanup, while unrelated work can proceed.
    """
    if max_fragments is not None and max_fragments < 1:
        msg = "The fragment selection limit must be positive"
        raise ValueError(msg)
    dataset, bindings = enrichment_input.dataset, enrichment_input.bindings
    pending = inputs.pending_fragment_ids(dataset, bindings)
    reconciled = _reconcile(enrichment_input, pending)
    for unproven in reconciled.unproven:
        logger.warning(
            "Completed recovery selection has no matching publication receipt; "
            "preserving workspace {} and continuing with other work",
            unproven.root,
        )
    cleanup_complete = not reconciled.unproven
    for completed in reconciled.completed:
        cleaned = completed.cleanup(storage_profile=bindings.execution.storage_profile)
        cleanup_complete = cleaned and cleanup_complete
    workspace: RecoveryWorkspace | None
    if reconciled.active:
        ((selected, workspace),) = reconciled.active
    else:
        selected = tuple(sorted(pending)[:max_fragments])
        workspace = inputs.resolve_workspace(dataset, bindings, selected=selected) if selected else None
    if workspace is not None:
        workspace.ensure(storage_profile=bindings.execution.storage_profile)
    return EnrichmentAttempt(enrichment_input, pending, selected, workspace, cleanup_complete)
