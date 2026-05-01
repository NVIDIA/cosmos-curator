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

"""Stage builders for SAM3-based object tracking."""

import attrs

from cosmos_curator.core.interfaces.stage_interface import CuratorStage, CuratorStageSpec
from cosmos_curator.pipelines.video.tracking.sam3_bbox_stage import SAM3BBoxStage, SAM3QualityConfig


@attrs.define(frozen=True)
class SAM3TrackingConfig:
    """Configuration for the SAM3 bounding-box tracking block."""

    prompts: list[str]
    target_fps: float = 10.0
    max_clip_duration_s: float = 30.0
    session_reset_s: float = 10.0
    quality: SAM3QualityConfig = attrs.Factory(SAM3QualityConfig)
    write_annotated_video: bool = False
    draw_trails: bool = False
    gpus_per_worker: float = 1.0
    num_workers_per_node: int = 0
    verbose: bool = False


def build_sam3_tracking_stages(config: SAM3TrackingConfig) -> list[CuratorStage | CuratorStageSpec]:
    """Build the SAM3 tracking stage block.

    Currently a single stage (``SAM3BBoxStage``) that performs inference and
    optionally produces the annotated mp4 in one pass. Returned as a list to
    match the ``build_*_stages`` convention used elsewhere in the pipeline.
    """
    stage = SAM3BBoxStage(
        prompts=list(config.prompts),
        target_fps=config.target_fps,
        max_clip_duration_s=config.max_clip_duration_s,
        session_reset_s=config.session_reset_s,
        quality_config=config.quality,
        write_annotated_video=config.write_annotated_video,
        draw_trails=config.draw_trails,
        gpus_per_worker=config.gpus_per_worker,
        verbose=config.verbose,
    )
    if config.num_workers_per_node > 0:
        return [CuratorStageSpec(stage, num_workers_per_node=config.num_workers_per_node)]
    return [stage]
