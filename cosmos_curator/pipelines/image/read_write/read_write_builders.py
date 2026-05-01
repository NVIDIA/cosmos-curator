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

"""Builder functions for image ingest and output stages."""

import attrs

from cosmos_curator.core.interfaces.stage_interface import CuratorStage, CuratorStageSpec
from cosmos_curator.pipelines.image.read_write.image_load_stage import ImageLoadStage
from cosmos_curator.pipelines.image.read_write.image_writer_stage import ImageWriterStage


@attrs.define(frozen=True)
class ImageIngestConfig:
    """Configuration for the image ingest stage."""

    input_path: str
    input_s3_profile_name: str = "default"
    num_workers_per_node: int = 4
    num_run_attempts: int = 5
    verbose: bool = False
    perf_profile: bool = False


@attrs.define(frozen=True)
class ImageOutputConfig:
    """Configuration for the image writer stage."""

    output_path: str
    output_s3_profile_name: str = "default"
    num_workers_per_node: int = 8
    num_run_attempts: int = 5
    verbose: bool = False
    perf_profile: bool = False


def build_image_ingest_stages(config: ImageIngestConfig) -> list[CuratorStage | CuratorStageSpec]:
    """Build the image load stages."""
    return [
        CuratorStageSpec(
            ImageLoadStage(
                input_path=config.input_path,
                input_s3_profile_name=config.input_s3_profile_name,
                verbose=config.verbose,
                log_stats=config.perf_profile,
            ),
            num_workers_per_node=config.num_workers_per_node,
            num_run_attempts_python=config.num_run_attempts,
        ),
    ]


def build_image_output_stages(config: ImageOutputConfig) -> list[CuratorStage | CuratorStageSpec]:
    """Build the image writer stages."""
    return [
        CuratorStageSpec(
            ImageWriterStage(
                output_path=config.output_path,
                output_s3_profile_name=config.output_s3_profile_name,
                verbose=config.verbose,
                log_stats=config.perf_profile,
            ),
            num_workers_per_node=config.num_workers_per_node,
            num_run_attempts_python=config.num_run_attempts,
        ),
    ]
