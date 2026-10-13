# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpointed whole-fragment enrichment of a Lance table, with cross-run recovery.

Recipes use this API, imported from its defining modules:

- ``contracts``: ``owned_fields`` declares the output fields; ``EnrichmentSpec`` adds the ID,
  required input, result contract, input readiness, and terminal validation;
  ``EnrichmentBindings`` binds the spec to one invocation's table, staging root, and
  execution settings
- ``execution``: ``run_enrichment``, whose ``process`` callback receives a ``ProcessingScope``
  that reads the selection and writes rows in its ``staged_schema``, and ``summary_message``
- ``config``: ``EnrichmentExecutionConfig``, the base of a recipe's ``execution`` settings
- ``cosmos_curator.next.core.lance_datasource``: ``ROW_ADDRESS_FIELDS``, for recipe work schemas
  that carry row addresses

Everything else is internal to ``run_enrichment``, which runs these steps in order:
``inputs`` opens and validates the pinned input and selects pending work, ``attempts``
resumes or selects an attempt, ``execution`` processes it inside a checkpoint scope and
publishes it, and ``commit`` implements the Lance publication protocol. ``transactions``
and ``workspace`` hold publication evidence lookup and durable recovery state.

Adopting recipes subclass the conformance suite in
``tests/cosmos_curator/next/enrichment/conformance.py`` and run the fresh-cluster scenario in
``tests/cosmos_curator/next/enrichment/fresh_cluster.py``; their docstrings list what a
recipe supplies. Design and adoption example: ``docs/curator/design/curator-next-checkpoint-recovery.md``.
"""
