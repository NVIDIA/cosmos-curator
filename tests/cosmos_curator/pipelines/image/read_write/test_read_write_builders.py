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

"""Tests for image read/write stage builders."""

from cosmos_curator.core.interfaces.stage_interface import CuratorStageSpec
from cosmos_curator.pipelines.image.read_write.image_load_stage import ImageLoadStage
from cosmos_curator.pipelines.image.read_write.image_writer_stage import ImageWriterStage
from cosmos_curator.pipelines.image.read_write.read_write_builders import (
    ImageIngestConfig,
    ImageOutputConfig,
    build_image_ingest_stages,
    build_image_output_stages,
)


def test_build_image_ingest_stages_returns_load_stage_spec() -> None:
    """The ingest builder should return one stage spec wrapping ``ImageLoadStage``."""
    stages = build_image_ingest_stages(
        ImageIngestConfig(
            input_path="/fake/input",
            input_s3_profile_name="default",
            num_workers_per_node=3,
            num_run_attempts=7,
        )
    )

    assert len(stages) == 1
    assert isinstance(stages[0], CuratorStageSpec)
    assert isinstance(stages[0].stage, ImageLoadStage)
    assert stages[0].num_workers_per_node == 3
    assert stages[0].num_run_attempts_python == 7


def test_build_image_output_stages_returns_writer_stage_spec() -> None:
    """The output builder should return one stage spec wrapping ``ImageWriterStage``."""
    stages = build_image_output_stages(
        ImageOutputConfig(
            output_path="/fake/output",
            output_s3_profile_name="default",
            num_workers_per_node=5,
            num_run_attempts=2,
        )
    )

    assert len(stages) == 1
    assert isinstance(stages[0], CuratorStageSpec)
    assert isinstance(stages[0].stage, ImageWriterStage)
    assert stages[0].num_workers_per_node == 5
    assert stages[0].num_run_attempts_python == 2
