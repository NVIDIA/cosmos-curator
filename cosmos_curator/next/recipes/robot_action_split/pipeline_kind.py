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

"""Config-backed pipeline-kind surface for the ``robot-action-split`` recipe."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from cosmos_curator.next.core.pipeline_kind import PipelineKind, PipelinePreset, PipelineRunOutput, PreparedPipelineRun

_TEMPLATE_PREAMBLE = """\
# Optional output overrides are derived from `media_root` when omitted:
#   lance_uri: <media_root>/lance/clips.lance
#   staging_root_uri: <media_root>/staging/robot-action-split
# The clip table is bound to the split settings, source_dataset, action_format, and video_bitrate;
# changing any of them needs a new lance_uri.
# Set input.storage_profile when the source datasets live in a different store than the outputs,
# which use execution.storage_profile.
# Each run publishes at most execution.max_items chunks. Rerun while remaining_items is nonzero or null.
"""

_TEMPLATE_BODY = """\
schema_version: 1
kind: robot-action-split

input:
  uris: [s3://example-bucket/robot_data/lerobot_v30/libero_10/]
  # Registered in cosmos_curator/next/media/action_binary.py; set output.action_format: pickle otherwise.
  source_dataset: libero_10

output:
  media_root: s3://example-bucket/robot_clips/
"""


def _template_yaml() -> str:
    return _TEMPLATE_PREAMBLE + _TEMPLATE_BODY


def _template_payload() -> dict[str, Any]:
    import yaml  # noqa: PLC0415

    return {
        "kind": "robot-action-split",
        "description": "Extract subtask clips from a LeRobot/Mecka dataset with chunk-level Ray checkpoints.",
        "required_fields": [
            {"path": "schema_version", "example": 1},
            {"path": "kind", "example": "robot-action-split"},
            {
                "path": "input.uris",
                "example": ["s3://example-bucket/robot_data/lerobot_v30/libero_10/"],
            },
            {"path": "input.source_dataset", "example": "libero_10"},
            {"path": "output.media_root", "example": "s3://example-bucket/robot_clips/"},
        ],
        "config": yaml.safe_load(_TEMPLATE_BODY),
    }


def _validate(config: Path, overrides: Sequence[str]) -> dict[str, object]:
    from cosmos_curator.next.recipes.robot_action_split.config import resolve_config  # noqa: PLC0415

    resolve_config(config, overrides=list(overrides))
    return {"ok": True}


def _render(config: Path, overrides: Sequence[str]) -> str:
    import json  # noqa: PLC0415

    from cosmos_curator.next.recipes.robot_action_split.config import resolve_config  # noqa: PLC0415

    cfg = resolve_config(config, overrides=list(overrides))
    return json.dumps(cfg.model_dump(mode="json"), indent=2) + "\n"


def _schema_json() -> str:
    import json  # noqa: PLC0415

    from cosmos_curator.next.recipes.robot_action_split.config import (  # noqa: PLC0415
        ResolvedRobotActionSplitConfig,
    )

    return json.dumps(ResolvedRobotActionSplitConfig.model_json_schema(), indent=2) + "\n"


def _list_presets() -> list[PipelinePreset]:
    return []


def _prepare_run(
    config: Path,
    *,
    set_overrides: list[str],
) -> PreparedPipelineRun:
    from cosmos_curator.next.recipes.robot_action_split.config import resolve_config  # noqa: PLC0415

    resolved_config = resolve_config(config, overrides=set_overrides)

    def run() -> PipelineRunOutput:
        from cosmos_curator.next.append.execution import summary_message  # noqa: PLC0415
        from cosmos_curator.next.recipes.robot_action_split.pipeline import run_config  # noqa: PLC0415

        summary = run_config(resolved_config)
        message = f"{summary_message('robot-action-split', summary)}; {summary['failed_clips']} clip(s) failed"
        return PipelineRunOutput(json_payload=summary, message=message)

    return run


ROBOT_ACTION_SPLIT_KIND = PipelineKind(
    name="robot-action-split",
    template_yaml=_template_yaml,
    template_payload=_template_payload,
    validate=_validate,
    render=_render,
    schema_json=_schema_json,
    list_presets=_list_presets,
    prepare_run=_prepare_run,
)
