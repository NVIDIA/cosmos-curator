# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resume or clear the single recovery workspace before selecting an append attempt.

One producer appends to a table and owns its contract's fixed workspace.
An unfinished selection resumes before any newly discovered item is selected,
and is never expanded by new discoveries or a changed selection limit.
Current rows prove completion because each selection atomically appends at
least one row per item. History pruning does not affect recovery or cleanup.
The recipe retains ownership of its Ray execution.
"""

from collections.abc import Callable

import attrs
import lance
import pyarrow as pa
import pyarrow.compute as pc
from loguru import logger

from cosmos_curator.next.append import inputs
from cosmos_curator.next.append.contracts import AppendBindings, AppendSpec
from cosmos_curator.next.core.recovery_workspace import RecoveryStateError, RecoveryWorkspace


@attrs.frozen
class AppendAttempt:
    """One frozen selection of work items and the recovery workspace that belongs to it.

    ``dataset`` is the target table, pinned before recovery or discovery.
    ``selection`` holds the items this attempt processes and publishes, in ID
    order. A resumed selection keeps its saved items without discovery, leaving
    ``input_items`` and ``remaining_items`` unknown. A workspace exists exactly
    when items are selected, and its manifest is already durable. Create attempts
    with ``reconcile_and_select_attempt``.
    """

    dataset: lance.LanceDataset = attrs.field(eq=False, repr=False)
    bindings: AppendBindings = attrs.field(eq=False, repr=False)
    selection: pa.Table = attrs.field(eq=False, repr=False)
    workspace: RecoveryWorkspace | None
    input_items: int | None
    remaining_items: int | None

    @property
    def selected(self) -> tuple[str, ...]:
        """Return the selected item IDs in order."""
        return tuple(self.selection[self.bindings.spec.id_column].to_pylist())

    def processing_complete(self) -> bool:
        """Report a valid processing marker, which lets a retry skip straight to publication."""
        return self.workspace is not None and self.workspace.processing_complete(
            storage_profile=self.bindings.execution.storage_profile
        )

    def counts(self) -> dict[str, int | None]:
        """Count this invocation's discovered, selected, and remaining items.

        Discovery counts are unknown on resume. Selected items include results
        recovered from checkpoints, empty outcomes, and terminal errors.
        """
        return {
            "input_items": self.input_items,
            "selected_items": self.selection.num_rows,
            "remaining_items": self.remaining_items,
        }


def _recorded_selection(workspace: RecoveryWorkspace, spec: AppendSpec) -> pa.Table:
    """Decode saved items through the same validator used for discovery."""
    invalid = f"Invalid item selection in recovery workspace: {workspace.root}. Remove the workspace."
    items = workspace.manifest.get("selected_items")
    if (
        not isinstance(items, list)
        or not items
        or any(not isinstance(item, dict) or item.keys() != set(spec.input_fields.names) for item in items)
    ):
        raise RecoveryStateError(invalid)
    try:
        selection = pa.Table.from_pylist(items, schema=spec.input_fields)
        return inputs.validate_items(selection, spec)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RecoveryStateError(invalid) from exc


def reconcile_and_select_attempt(bindings: AppendBindings, discover: Callable[[], pa.Table]) -> AppendAttempt:
    """Resume saved work, or clear its verified publication before selecting new items.

    One fixed workspace holds this table and contract's selection. Its manifest
    is durable before processing starts. A saved selection resumes without
    discovery; current rows prove whether it has already published. Cleanup must
    succeed before another selection can reuse the workspace.

    Pending items are discovered items absent from current rows.
    ``execution.max_items`` limits only a new selection, in item-ID order.
    Partial publication is rejected because each selection publishes atomically.
    """
    dataset = inputs.open_target(bindings)
    spec = bindings.spec
    id_column = spec.id_column
    storage_profile = bindings.execution.storage_profile
    root = inputs.workspace_root(dataset, bindings)
    saved_workspace = RecoveryWorkspace.load(root, storage_profile=storage_profile)
    if saved_workspace is not None:
        selection = _recorded_selection(saved_workspace, spec)
        if inputs.resolve_workspace(dataset, bindings, selection=selection) != saved_workspace:
            msg = (
                f"Recovery workspace does not match its recorded selection and contract: {root}. "
                "Remove the workspace to discard its staged results."
            )
            raise RecoveryStateError(msg)
        selected = tuple(selection[id_column].to_pylist())
        published = inputs.published_item_ids(dataset, bindings, selected)
        if saved_workspace.resume_or_clear(selected, published, storage_profile=storage_profile):
            logger.info(
                "Resuming an unfinished selection of {} item(s) from {}; discovery waits for a later run",
                selection.num_rows,
                root,
            )
            return AppendAttempt(dataset, bindings, selection, saved_workspace, None, None)

    items = inputs.validate_items(discover(), spec)
    published = inputs.published_item_ids(dataset, bindings, items[id_column].to_pylist())
    known = pa.array(list(published), type=items.schema.field(id_column).type)
    pending_items = items.filter(pc.invert(pc.is_in(items[id_column], value_set=known)))
    max_items = bindings.execution.max_items
    selection = pending_items if max_items is None else pending_items.slice(0, max_items)
    workspace = inputs.resolve_workspace(dataset, bindings, selection=selection) if selection.num_rows else None
    if workspace is not None:
        workspace.start(storage_profile=storage_profile)
    return AppendAttempt(
        dataset, bindings, selection, workspace, items.num_rows, pending_items.num_rows - selection.num_rows
    )
