# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Tests for ingest stage builder topology."""

from cosmos_curator.core.interfaces.stage_interface import CuratorStageSpec
from cosmos_curator.pipelines.video.read_write.metadata_writer_stage import ClipWriterStage
from cosmos_curator.pipelines.video.read_write.read_write_builders import (
    OutputConfig,
    build_output_stages,
)


def test_output_stage_forwards_caption_quality_flag() -> None:
    """ClipWriterStage should receive caption quality config."""
    config = OutputConfig(
        output_path="/fake/output",
        input_path="/fake/input",
        caption_quality_stats_enabled=True,
        caption_quality_flags_enabled=False,
    )

    stages = build_output_stages(config)

    assert len(stages) == 1
    stage_spec = stages[0]
    assert isinstance(stage_spec, CuratorStageSpec)
    assert isinstance(stage_spec.stage, ClipWriterStage)
    assert stage_spec.stage._caption_quality_stats_enabled is True
    assert stage_spec.stage._caption_quality_flags_enabled is False
