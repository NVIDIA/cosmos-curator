# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpointed fixed-stride and TransNetV2 cutting with atomic Lance appends.

Explicit URIs or a recursively listed S3 root select immutable MP4 objects. A
source ID hashes its normalized URI; a span ID hashes that source, the first
video stream, and its zero-relative half-open nanosecond interval. Cutting
settings have a separate versioned hash, which is the result contract. Default
Lance/workspace locations are ``<media_root>/lance`` and
``<media_root>/staging/video-cut``.

Video-cut is the reference recipe for shared append publication
(``cosmos_curator.next.append``), which owns selection, checkpoints, atomic
publication of all outcomes, and recovery from current rows. Each source is one work item whose result
is its complete span plan or a terminal error. Spans remain flat rows; empty and
error outcomes each retain one row with a null span ID.
Input storage is stable, so missing or archived objects and recognized bad
media are terminal; failures of the environment or transient conditions
propagate.

The cutting methods are peer modules that each export ``plan(sources, config)``
and build results with ``contracts``. ``fixed_stride`` probes durations remotely
on task workers, falling back to a temporary local download when needed; source
bytes never enter Ray. ``transnetv2`` admits downloads through actors holding
one ``curator_io`` slot each, then passes the bytes to frame decoding and scene
inference on GPU actors, with pre-staged weights matching the pinned registry
revision; the opt-in NVDEC backend decodes on the inference actor.

See ``docs/curator/design/curator-next-video-cut.md`` for configuration, media
access, recovery tests, and the limits of current deployment qualification.
"""
