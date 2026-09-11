# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Robot-action caption recipe: caption egocentric manipulation clips in ``clips.lance``.

Adds five nullable caption columns to the ``clips.lance`` table produced by
``robot-action-split``, using a Qwen2.5-VL model served by the recipe's private
vLLM executor. Egocentric task context already stored on each row is rendered
into the prompt; the model text itself is semantic only and never a source of
geometry.

The recipe is two-phase so that a crash cannot lose completed GPU work. Phase A
stages terminal rows to Parquet under a snapshot-scoped workspace and never
touches Lance; Phase B publishes every touched fragment in ONE atomic
``LanceOperation.Update``. Ray Data's job-level row checkpoint filters clips that
already completed, so a resume pays only for the clips that did not.

::

  clips.lance @ pinned v
        |
        v
   scan (prompt + row-address id) --> media (base64 url) --> vLLM --> canonicalize
        |                                                                  |
        |                                              Phase A: write_parquet (staged)
        |                                                                  |
        v                                                                  v
   owed rows                                           Phase B: prepare fragments
   (selected scope AND pending contract)                       + ONE commit
                                                                           |
                                                                           v
                                                              clips.lance @ v+1

A row is pending when no run has claimed it, when the run that did used
different semantics or model identity, or when the caption sibling group is not
a valid terminal row. An attempt owes the intersection of that set and its input
scope: Curate-selected rows by default, or all pending rows when explicitly
configured. Latest contract wins within that scope, and a re-captioned value
overwrites the published one in the same atomic commit.

Every failure lands on one of exactly two sides, and which side it lands on is
the invariant the whole recipe rests on. A fault that would recur identically for
every clip must not become millions of published error rows, and one bad clip
must not fail a run that has already paid for thousands of captions::

    missing structural column -> systemic, refuse on driver  (table.py)
    incomplete model files   -> systemic, refuse on driver   (models/registry.py)
    unusable read credential  -> systemic, raise from stage  (media.py)
    absent context column     -> narrows the run's contract   (table.py)
    NULL context value        -> omits one prompt line        (prompt.py)
    unreadable clip           -> terminal error row           (media.py)
    oversized clip            -> terminal error row           (media.py)
    empty generation          -> terminal error row           (request.py)

The driver-side refusals are on the driver deliberately: inside a Ray task the
exception type is erased and the message reaches the operator wrapped in a
``RayError``, which is the one shape they cannot act on.

The invariant classifies a fault; it cannot bound how far one spreads. An absent
clip object is row-local because no retry of THAT object would help, yet media
deleted, lifecycle-expired, or a ``clips.lance`` restored without its media makes
that true of every owed clip, and the wave commits. Publication surfaces that rather
than refusing it: ``CaptionRunResult.error_rows`` is what separates it from a
corpus of captions, and a changed contract digest recaptions those rows only
when they enter the configured input scope.

Architecture, executor contract, and resume runbook:
``docs/curator/design/curator-next-robot-action-caption.md``.
"""
