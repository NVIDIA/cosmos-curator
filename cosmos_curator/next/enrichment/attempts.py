# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resume or clear the single recovery workspace before selecting an enrichment attempt.

One active writer owns a field group and its contract's fixed workspace.
An unfinished selection is never expanded by appends or a changed batch limit.
Current values prove completion because each selection publishes its fragments
in one atomic update. History pruning does not affect recovery or cleanup.
The recipe retains ownership of its Ray execution.
"""

import attrs

from cosmos_curator.next.core.recovery_workspace import RecoveryStateError, RecoveryWorkspace
from cosmos_curator.next.enrichment import inputs


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


def _recorded_selection(workspace: RecoveryWorkspace) -> tuple[int, ...]:
    fragments = workspace.manifest.get("input_fragments")
    if (
        not isinstance(fragments, list)
        or not fragments
        or any(
            not isinstance(fragment, dict)
            or type(fragment.get("fragment_id")) is not int
            or fragment["fragment_id"] < 0
            or not isinstance(fragment.get("file_layout"), str)
            for fragment in fragments
        )
    ):
        msg = f"Invalid fragment selection in recovery workspace: {workspace.root}. Remove the workspace."
        raise RecoveryStateError(msg)
    selected = tuple(fragment["fragment_id"] for fragment in fragments)
    if selected != tuple(sorted(set(selected))):
        msg = f"Recovery workspace selection is not sorted and unique: {workspace.root}. Remove the workspace."
        raise RecoveryStateError(msg)
    return selected


def reconcile_and_select_attempt(enrichment_input: inputs.EnrichmentInput) -> EnrichmentAttempt:
    """Resume saved work, or clear its verified publication before selecting new fragments.

    One fixed workspace holds this field group and contract's selection. Its
    manifest is durable before processing starts. A saved selection resumes before
    new pending fragments; current values prove whether it has already published.
    Cleanup must succeed before another selection can reuse the workspace.

    Pending fragments come from the pinned input's output values and retained
    receipts. ``execution.max_fragments`` limits only a new selection; recovery
    retains its recorded membership. Partial publication is rejected because each
    selection publishes atomically.
    """
    dataset, bindings = enrichment_input.dataset, enrichment_input.bindings
    storage_profile = bindings.execution.storage_profile
    pending = inputs.pending_fragment_ids(dataset, bindings)
    root = inputs.workspace_root(dataset, bindings)
    saved_workspace = RecoveryWorkspace.load(root, storage_profile=storage_profile)
    if saved_workspace is not None:
        selected = _recorded_selection(saved_workspace)
        if inputs.resolve_workspace(dataset, bindings, selected=selected) != saved_workspace:
            msg = (
                f"Recovery workspace does not match the current input file layout: {root}. "
                "Its selected input changed, which shared recovery does not support; "
                "remove the workspace to discard its staged results."
            )
            raise RecoveryStateError(msg)
        published = set(selected).difference(pending)
        if saved_workspace.resume_or_clear(selected, published, storage_profile=storage_profile):
            return EnrichmentAttempt(enrichment_input, pending, selected, saved_workspace)

    selected = tuple(sorted(pending)[: bindings.execution.max_fragments])
    workspace = inputs.resolve_workspace(dataset, bindings, selected=selected) if selected else None
    if workspace is not None:
        workspace.start(storage_profile=storage_profile)
    return EnrichmentAttempt(enrichment_input, pending, selected, workspace)
