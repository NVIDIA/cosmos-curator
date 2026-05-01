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

"""Stage builders for SeedVR2 video super-resolution."""

import attrs

from cosmos_curator.core.interfaces.stage_interface import CuratorStage, CuratorStageSpec


@attrs.define(frozen=True)
class SuperResolutionConfig:
    """Configuration for the SeedVR2 super-resolution stage.

    All parameters mirror the inference args from the upstream SeedVR2 windowed
    inference script (inference_seedvr2_window.py).
    """

    # Model variant: seedvr2_3b, seedvr2_7b, seedvr2_7b_sharp
    variant: str = "seedvr2_7b"

    # Target output resolution
    target_height: int = 720
    target_width: int = 1280

    # Windowed inference (OOM prevention for long clips)
    window_frames: int = 128
    overlap_frames: int = 64
    blend_overlap: bool = True

    # Diffusion parameters
    seed: int = 666
    cfg_scale: float = 1.0
    cfg_rescale: float = 0.0
    sample_steps: int = 1

    # Sequence parallelism
    sp_size: int = 1

    # Output
    out_fps: float | None = None
    tmp_dir: str | None = None

    # Pipeline standard flags
    verbose: bool = False
    perf_profile: bool = False


def build_super_resolution_stages(config: SuperResolutionConfig) -> list[CuratorStage | CuratorStageSpec]:
    """Construct and return the super-resolution stage.

    Args:
        config: Super-resolution configuration.

    Returns:
        A list containing the configured super-resolution stage.

    """
    from cosmos_curator.pipelines.video.super_resolution.super_resolution_stage import (  # noqa: PLC0415
        SuperResolutionStage,
    )

    return [
        CuratorStageSpec(
            SuperResolutionStage(config=config),
            num_workers_per_node=1,
        ),
    ]
