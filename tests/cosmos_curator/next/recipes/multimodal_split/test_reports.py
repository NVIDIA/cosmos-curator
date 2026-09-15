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

"""Tests for the artifacts a run leaves behind about itself.

Exercised directly rather than through planning or splitting: each writer here
takes only the output path (and, for the manifest, the output rate) it actually
needs, not the whole resolved config, so a session's receipts can be pinned
without building a rig to produce them.
"""

import json
from pathlib import Path

from cosmos_curator.next.media.spans import Span
from cosmos_curator.next.recipes.multimodal_split.reports import (
    failure,
    plan_error_uri,
    sensor_error_uri,
    write_clip_manifest,
    write_plan_errors,
    write_sensor_errors,
)


def test_write_plan_errors_writes_nothing_for_a_clean_session(tmp_path: Path) -> None:
    """A session with no refused spans leaves no report behind to be listed."""
    write_plan_errors("session-000", [], output_path=str(tmp_path), storage_profile="default")

    assert not Path(plan_error_uri(str(tmp_path), "session-000")).exists()


def test_write_plan_errors_names_the_stage_and_carries_every_failure(tmp_path: Path) -> None:
    """A refused session's report names the stage that refused it and keeps every entry."""
    failures = [failure("session-000", None, None, "plan_spans", ValueError("no overlap"))]

    write_plan_errors("session-000", failures, output_path=str(tmp_path), storage_profile="default")

    written = json.loads(Path(plan_error_uri(str(tmp_path), "session-000")).read_text(encoding="utf-8"))
    assert written["stage"] == "span_planner"
    assert written["failures"] == failures


def test_write_sensor_errors_writes_nothing_for_a_healthy_sensor(tmp_path: Path) -> None:
    """A sensor with no encode failures leaves no report behind to be listed."""
    write_sensor_errors("session-000", "front", [], output_path=str(tmp_path), storage_profile="default")

    assert not Path(sensor_error_uri(str(tmp_path), "session-000", "front")).exists()


def test_write_sensor_errors_names_the_sensor_that_failed(tmp_path: Path) -> None:
    """A sensor's report is written under its own name and carries its own failures."""
    failures = [failure("session-000", "left", None, "sample_and_encode", RuntimeError("encoder gave up"))]

    write_sensor_errors("session-000", "left", failures, output_path=str(tmp_path), storage_profile="default")

    written = json.loads(Path(sensor_error_uri(str(tmp_path), "session-000", "left")).read_text(encoding="utf-8"))
    assert written["sensor_key"] == "left"
    assert written["failures"] == failures


def test_write_clip_manifest_records_where_and_how_fast_a_clip_was_cut(tmp_path: Path) -> None:
    """A clip's manifest says which session it belongs to, its span, and its rate."""
    span = Span(start_ns=0, end_ns=1_000_000_000)

    write_clip_manifest(
        "session-000",
        "clip-abc",
        span,
        ["front", "left"],
        ["ego"],
        output_path=str(tmp_path),
        sampling_fps=30,
        storage_profile="default",
    )

    manifest_path = tmp_path / "sessions" / "session-000" / "clip-abc" / "clip.json"
    written = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert written["parent_id"] == "session-000"
    assert written["sampling_fps"] == 30
    assert written["planned_cameras"] == ["front", "left"]
    # Listed apart from the cameras, because the two are read differently: a
    # camera's entry names an MP4 on the episode's grid, a sensor's a file at
    # its own rate.
    assert written["planned_sensors"] == ["ego"]
