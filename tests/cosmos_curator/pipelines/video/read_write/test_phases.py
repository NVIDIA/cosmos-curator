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
from cosmos_curator.pipelines.video.read_write.read_write_builders import IngestConfig, build_ingest_stages
from cosmos_curator.pipelines.video.read_write.remux_stages import RemuxStage


def test_remux_stage_absent_from_ingest_stages() -> None:
    """RemuxStage must not appear in build_ingest_stages().

    RemuxStage was folded into VideoDownloader; leaving it in the stage list
    would run remux twice and waste a dedicated worker pool.
    """
    config = IngestConfig(input_path="/fake/path")
    stages = build_ingest_stages(config)

    assert len(stages) == 1, "build_ingest_stages should return exactly one stage (VideoDownloader)"
    # Unwrap CuratorStageSpec to catch RemuxStage whether bare or wrapped
    inner_stages = [s.stage if isinstance(s, CuratorStageSpec) else s for s in stages]
    assert not any(isinstance(s, RemuxStage) for s in inner_stages), "RemuxStage must not be in ingest stages"
