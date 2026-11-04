# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Two-phase caption recovery with per-fragment Lance publication.

Deprecated: new enrichment recipes use shared enrichment publication in
``cosmos_curator.next.enrichment``, which video-caption adopts. Multimodal-caption
and multimodal-embedding still use this package and migrate explicitly;
multimodal-embedding first needs the shared contract to publish several field sets
in one commit. See ``docs/curator/design/curator-next-checkpoint-recovery.md``.

Phase A (``inference``) reads pending fragments, runs video-caption's shared caption
plan, and stages terminal rows as checkpointed Parquet; Phase B (``publication``)
validates them and commits one fragment at a time. ``lance_state`` classifies
fragments and registers result fields, and ``workspace`` owns the recovery scope.
"""
