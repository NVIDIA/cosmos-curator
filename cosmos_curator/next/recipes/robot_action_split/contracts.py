# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Recipe-owned durable contract version constants for ``robot-action-split``."""

# Versions the generated MP4 byte contract and its matching clip identity.
# Bump when non-configurable cut semantics change such that the same span_group_id
# would produce different MP4 bytes (e.g. smart-cut -> full-reencode switch).
# Metadata-only Lance or sidecar field changes do not move this version.
MEDIA_CONTRACT_VERSION = 1

# Bump when the action binary format or field layout changes in a way that would
# produce different bytes for the same span/dataset (e.g. ACT2 header version bump,
# field reordering, or dtype change).
ACTION_CONTRACT_VERSION = 1

# Bump when CLIP_SCHEMA's field set or meaning changes. v4 adds the nullable
# episode-level task_description field without changing media identity. v3
# renamed the former OUTCOME_SCHEMA to CLIP_SCHEMA, matching video_split's successes-only shape:
# dropped status/error_stage/error_message (failures now go to a separate
# errors.json report, never to Lance), tightened clip_uri/action_data_uri to
# non-nullable, and added record_schema_version/media_contract_version columns
# for cross-run reconciliation integrity checks. The embed leg only reads
# clip_id/task_name/subtask_name/clip_uri/action_data_uri/source_dataset by
# name (see docs/curator/design/curator-next-embeddings.md), all unaffected.
CLIP_RECORD_SCHEMA_VERSION = 4
RECEIPT_SCHEMA_VERSION = 1
MANIFEST_SCHEMA_VERSION = 1
