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

"""Pipeline-kind surfaces for the deprecated ``cosmos_curator.pipelines.ray_data`` recipes.

These adapt ``video_split`` and ``caption_judge`` to the same ``PipelineKind``
contract the Curator Next recipes implement, so the CLI and runtime share one
dispatch path while both generations coexist. Keeping the adapters together
also makes their eventual removal explicit: delete this module and its two
registry entries along with the ``ray_data`` tree.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from cosmos_curator.next.core.pipeline_kind import PipelineKind, PipelinePreset, PipelineRunOutput, PreparedPipelineRun


def _video_split_template_yaml() -> str:
    from cosmos_curator.pipelines.ray_data.video_split.config import (  # noqa: PLC0415
        user_config_to_yaml,
        video_split_config_template,
    )

    return user_config_to_yaml(video_split_config_template())


def _video_split_template_payload() -> dict[str, Any]:
    from cosmos_curator.pipelines.ray_data.video_split.config import video_split_template_payload  # noqa: PLC0415

    return video_split_template_payload()


def _video_split_validate(config: Path, overrides: Sequence[str]) -> dict[str, object]:
    from cosmos_curator.pipelines.ray_data.video_split.config import resolve_video_split_config  # noqa: PLC0415

    resolution = resolve_video_split_config(config, overrides=overrides)
    return {"ok": True, "selected_presets": resolution.selected_presets}


def _video_split_render(config: Path, overrides: Sequence[str]) -> str:
    from cosmos_curator.pipelines.ray_data.video_split.config import (  # noqa: PLC0415
        resolve_video_split_config,
        resolved_config_to_json,
    )

    resolution = resolve_video_split_config(config, overrides=overrides)
    return resolved_config_to_json(resolution.config)


def _video_split_schema_json() -> str:
    from cosmos_curator.pipelines.ray_data.video_split.config import user_video_split_schema_json  # noqa: PLC0415

    return user_video_split_schema_json()


def _video_split_list_presets() -> list[PipelinePreset]:
    from cosmos_curator.pipelines.ray_data.video_split.config import list_video_split_presets  # noqa: PLC0415

    return [
        {
            "section": preset["section"],
            "name": preset["name"],
            "qualified_name": preset["qualified_name"],
            "fragment": preset["fragment"],
        }
        for preset in list_video_split_presets()
    ]


def _video_split_prepare_run(
    config: Path,
    *,
    set_overrides: list[str],
) -> PreparedPipelineRun:
    from cosmos_curator.pipelines.ray_data.video_split.config import resolve_video_split_config  # noqa: PLC0415

    resolution = resolve_video_split_config(config, overrides=set_overrides)

    def run() -> PipelineRunOutput:
        from cosmos_curator.pipelines.ray_data.video_split.pipeline import run_config  # noqa: PLC0415

        clips_written = run_config(resolution.config)
        return PipelineRunOutput(
            json_payload={"clips_written": clips_written},
            message=f"Wrote {clips_written} clip(s)",
        )

    return run


def _caption_judge_template_yaml() -> str:
    from cosmos_curator.pipelines.ray_data.caption_judge.config import (  # noqa: PLC0415
        caption_judge_config_template,
        caption_judge_config_to_yaml,
    )

    return caption_judge_config_to_yaml(caption_judge_config_template())


def _caption_judge_template_payload() -> dict[str, Any]:
    from cosmos_curator.pipelines.ray_data.caption_judge.config import caption_judge_template_payload  # noqa: PLC0415

    return caption_judge_template_payload()


def _caption_judge_validate(config: Path, overrides: Sequence[str]) -> dict[str, object]:
    from cosmos_curator.pipelines.ray_data.caption_judge.config import resolve_caption_judge_config  # noqa: PLC0415

    resolve_caption_judge_config(config, overrides=overrides)
    return {"ok": True}


def _caption_judge_render(config: Path, overrides: Sequence[str]) -> str:
    from cosmos_curator.pipelines.ray_data.caption_judge.config import (  # noqa: PLC0415
        caption_judge_config_to_json,
        resolve_caption_judge_config,
    )

    resolved_config = resolve_caption_judge_config(config, overrides=overrides)
    return caption_judge_config_to_json(resolved_config)


def _caption_judge_schema_json() -> str:
    from cosmos_curator.pipelines.ray_data.caption_judge.config import user_caption_judge_schema_json  # noqa: PLC0415

    return user_caption_judge_schema_json()


def _caption_judge_list_presets() -> list[PipelinePreset]:
    return []


def _caption_judge_prepare_run(
    config: Path,
    *,
    set_overrides: list[str],
) -> PreparedPipelineRun:
    from cosmos_curator.pipelines.ray_data.caption_judge.config import resolve_caption_judge_config  # noqa: PLC0415

    resolved_config = resolve_caption_judge_config(config, overrides=set_overrides)

    def run() -> PipelineRunOutput:
        from cosmos_curator.pipelines.ray_data.caption_judge.driver import run_caption_judge_pipeline  # noqa: PLC0415
        from cosmos_curator.pipelines.ray_data.caption_judge.report_io import write_report  # noqa: PLC0415

        report = run_caption_judge_pipeline(config=resolved_config)
        report_path = write_report(
            report,
            resolved_config.output.report_path,
            report_format=resolved_config.output.report_format,
        )
        status = "PASSED" if report.passed else "FAILED"
        return PipelineRunOutput(
            json_payload={
                "passed": report.passed,
                "issues": report.issues.num_rows,
                "windows_judged": report.stats.windows_judged,
                "report_path": report_path,
            },
            message=(
                f"{status} caption judge: {report.issues.num_rows} issues, "
                f"{report.stats.windows_judged} judged windows, report: {report_path}"
            ),
        )

    return run


VIDEO_SPLIT_LEGACY_KIND = PipelineKind(
    name="video_split",
    template_yaml=_video_split_template_yaml,
    template_payload=_video_split_template_payload,
    validate=_video_split_validate,
    render=_video_split_render,
    schema_json=_video_split_schema_json,
    list_presets=_video_split_list_presets,
    prepare_run=_video_split_prepare_run,
)

CAPTION_JUDGE_KIND = PipelineKind(
    name="caption_judge",
    template_yaml=_caption_judge_template_yaml,
    template_payload=_caption_judge_template_payload,
    validate=_caption_judge_validate,
    render=_caption_judge_render,
    schema_json=_caption_judge_schema_json,
    list_presets=_caption_judge_list_presets,
    prepare_run=_caption_judge_prepare_run,
)
