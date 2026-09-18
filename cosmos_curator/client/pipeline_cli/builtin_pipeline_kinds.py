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

"""Explicit composition root for built-in config-backed pipeline kinds.

Pipeline-kind modules are lightweight adapters. They defer importing config
models and runtime implementations until the selected operation needs them, so
the registry can own concrete objects without eagerly loading every recipe.
"""

from cosmos_curator.client.pipeline_cli.legacy_kinds import CAPTION_JUDGE_KIND, VIDEO_SPLIT_LEGACY_KIND
from cosmos_curator.next.core.pipeline_kind import PipelineKindRegistry
from cosmos_curator.next.recipes.curation.pipeline_kind import CURATE_KIND
from cosmos_curator.next.recipes.data_integrity.pipeline_kind import DATA_INTEGRITY_KIND
from cosmos_curator.next.recipes.embeddings.pipeline_kind import EMBEDDINGS_KIND
from cosmos_curator.next.recipes.multimodal_caption.pipeline_kind import MULTIMODAL_CAPTION_KIND
from cosmos_curator.next.recipes.multimodal_split.pipeline_kind import MULTIMODAL_SPLIT_KIND
from cosmos_curator.next.recipes.robot_action_caption.pipeline_kind import ROBOT_ACTION_CAPTION_KIND
from cosmos_curator.next.recipes.robot_action_split.pipeline_kind import ROBOT_ACTION_SPLIT_KIND
from cosmos_curator.next.recipes.video_caption.pipeline_kind import VIDEO_CAPTION_KIND
from cosmos_curator.next.recipes.video_cut.pipeline_kind import VIDEO_CUT_KIND
from cosmos_curator.next.recipes.video_transcode.pipeline_kind import VIDEO_TRANSCODE_KIND

# Registering a kind also changes two restatements of the sorted name list in the
# test tree: the expected tuple in test_builtin_pipeline_kinds.py, and the
# "Valid pipeline kinds: ..." string inside the unknown-kind assertion in
# test_pipeline_app.py. The second is easy to miss because the list is embedded in
# a sentence; `rg "Valid pipeline kinds"` finds it.
BUILTIN_PIPELINE_KINDS = PipelineKindRegistry(
    (
        DATA_INTEGRITY_KIND,
        MULTIMODAL_CAPTION_KIND,
        MULTIMODAL_SPLIT_KIND,
        VIDEO_CAPTION_KIND,
        VIDEO_CUT_KIND,
        VIDEO_TRANSCODE_KIND,
        ROBOT_ACTION_SPLIT_KIND,
        ROBOT_ACTION_CAPTION_KIND,
        CURATE_KIND,
        EMBEDDINGS_KIND,
        # Deprecated alongside cosmos_curator.pipelines.ray_data; delete these
        # two entries and legacy_kinds.py together with that tree.
        VIDEO_SPLIT_LEGACY_KIND,
        CAPTION_JUDGE_KIND,
    )
)
