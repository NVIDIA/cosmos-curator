# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpointed append publication to a producer-owned Lance table, with cross-run recovery.

Recipes use this API, imported from its defining modules:

- ``contracts``: ``AppendSpec`` declares the work items, staged item results,
  appended rows, result contract, terminal validation, expansion, and the
  optional terminal error column; ``AppendBindings`` binds the spec to one
  invocation's table, staging root, and execution settings
- ``execution``: ``run_append``, whose ``discover`` callback runs only for a new
  selection and whose ``process`` callback receives a ``ProcessingScope``
  that reads the selected items and writes canonical rows in its ``result_schema``,
  and ``summary_message``
- ``config``: ``AppendExecutionConfig``, the base of a recipe's ``execution`` settings

Everything else is internal to ``run_append``, which runs these steps in order:
``attempts`` opens the target through ``inputs``, looks up published item IDs in
current rows, and resumes saved work or discovers and selects new items.
``execution`` processes inside a checkpoint scope and atomically publishes all
outcomes through ``commit``. Every item contributes at least one row, including
empty outcomes and terminal errors, so recovery and diagnostics need no append
receipts or external error reports. Recovery workspaces come from
``cosmos_curator.next.core.recovery_workspace``, which enrichment shares. Each
table and contract reuses one fixed workspace; verified publication and cleanup
must finish before another selection starts.

Adopting recipes subclass the conformance suite in
``tests/cosmos_curator/next/append/conformance.py`` and run the fresh-cluster scenario in
``tests/cosmos_curator/next/append/fresh_cluster.py``; their docstrings list what a
recipe supplies. Design: ``docs/curator/design/curator-next-checkpoint-recovery.md``.
"""
