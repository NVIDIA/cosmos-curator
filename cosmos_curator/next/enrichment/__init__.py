# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpointed whole-fragment enrichment of a Lance table, with cross-run recovery.

Recipes use this API, imported from its defining modules:

- ``contracts``: ``owned_fields`` declares the output fields; ``EnrichmentSpec`` adds the ID,
  required input, input readiness, and terminal validation; ``EnrichmentBindings`` adds the
  table, staging root, result contract, and execution settings for one invocation
- ``execution``: ``run_enrichment``, whose ``process`` callback receives a ``ProcessingScope``
  that reads the selection and writes rows in its ``staged_schema``, and ``summary_message``
- ``config``: ``EnrichmentExecutionConfig``, the base of a recipe's ``execution`` settings
- ``cosmos_curator.next.core.lance_datasource``: ``ROW_ADDRESS_FIELDS``, for recipe work schemas
  that carry row addresses

``open_enrichment_input``, ``attempts``, ``checkpointed_processing``, and
``publish_enrichment`` are the steps ``run_enrichment`` sequences. ``commit``,
``transactions``, and ``workspace`` implement the publication protocol and durable
recovery state. Adopting recipes subclass the conformance suite in
``tests/cosmos_curator/next/enrichment/conformance.py``, whose docstring lists the
fixtures they supply.
Design and adoption example: ``docs/curator/design/curator-next-checkpoint-recovery.md``.
"""
