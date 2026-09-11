# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpointed source video cutting with atomic Lance span snapshots.

Explicit URIs or a recursively listed S3 root select immutable MP4 objects. A
source ID hashes its normalized URI; a span ID hashes that source, the first
video stream, and its zero-relative half-open nanosecond interval. Cutting
settings have a separate versioned hash. Default Lance/workspace locations are
``<media_root>/lance`` and ``<media_root>/staging/video-cut``.

Native Ray checkpoints retain one v1 Parquet outcome per source, including empty
plans and terminal media errors. Durability advances in bounded source groups;
in-flight groups can replay. Operational failures propagate instead of becoming
terminal outcomes. No media payload is published.

One logical driver owns an output bound to a frozen source set and cutting
contract. Changing either requires a new output location. Workers expand
and pack spans across source shards into 100,000-row Lance 2.2 fragments, then
the driver atomically publishes the complete snapshot. Completion and outcome
counts commit with the table. Incomplete publication rebuilds from Parquet
without repeating media work. After a confirmed commit the invocation attempts
to delete the workspace it used or resumed. Completed reruns recognize Lance
alone without starting Ray or touching staging. Cleanup is best effort; orphaned
staging requires explicit cleanup or a storage expiration policy. Terminal error
details survive in an output report. Use a fresh output to retry terminal errors;
archived-object failures remain retryable in the same run. Execution tuning
remains reusable.

See ``docs/curator/design/curator-next-video-cut.md`` for configuration, metadata
probing, recovery tests, and the limits of current deployment qualification.
"""
