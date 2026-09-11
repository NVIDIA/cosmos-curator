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

"""Config-backed pipeline-kind surface for the ``multimodal-split`` recipe.

A run discovers candidate sessions and splits each one into per-camera clips,
one MP4 per configured camera, then publishes one canonical Lance row for each
complete camera set. Non-camera modalities remain separate work.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from cosmos_curator.next.core.pipeline_kind import PipelineKind, PipelinePreset, PipelineRunOutput, PreparedPipelineRun


def _template_yaml() -> str:
    from cosmos_curator.next.recipes.multimodal_split.config import config_template_yaml  # noqa: PLC0415

    return config_template_yaml()


def _template_payload() -> dict[str, Any]:
    import yaml  # noqa: PLC0415

    return {
        "kind": "multimodal-split",
        "description": "Split AV recording sessions beneath a storage prefix into per-camera clips.",
        "required_fields": [
            {"path": "schema_version", "example": 1},
            {"path": "kind", "example": "multimodal-split"},
            {"path": "input.input_path_prefix", "example": "s3://example-bucket/recordings"},
            {
                "path": "sensors",
                "example": {"front": {"sensor_type": "camera", "uri_pattern": "*/camera_front.mp4"}},
            },
            {"path": "output.path", "example": "s3://example-bucket/media"},
        ],
        "config": yaml.safe_load(_template_yaml()),
    }


def _validate(config: Path, overrides: Sequence[str]) -> dict[str, object]:
    from cosmos_curator.next.recipes.multimodal_split.config import resolve_config  # noqa: PLC0415

    resolve_config(config, overrides=list(overrides))
    return {"ok": True}


def _render(config: Path, overrides: Sequence[str]) -> str:
    import json  # noqa: PLC0415

    from cosmos_curator.next.recipes.multimodal_split.config import resolve_config  # noqa: PLC0415

    resolved = resolve_config(config, overrides=list(overrides))
    return json.dumps(resolved.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n"


def _schema_json() -> str:
    import json  # noqa: PLC0415

    from cosmos_curator.next.recipes.multimodal_split.config import (  # noqa: PLC0415
        ResolvedMultimodalSplitConfig,
    )

    return json.dumps(ResolvedMultimodalSplitConfig.model_json_schema(), indent=2, ensure_ascii=False) + "\n"


def _list_presets() -> list[PipelinePreset]:
    return []


def _prepare_run(
    config: Path,
    *,
    set_overrides: list[str],
) -> PreparedPipelineRun:
    from cosmos_curator.next.recipes.multimodal_split.config import resolve_config  # noqa: PLC0415

    resolved = resolve_config(config, overrides=set_overrides)

    def run() -> PipelineRunOutput:
        from cosmos_curator.next.recipes.multimodal_split.pipeline import run_config  # noqa: PLC0415

        summary = run_config(resolved)
        return PipelineRunOutput(
            json_payload=summary,
            message=(
                f"multimodal-split: published {summary['episodes_newly_published']} new episode(s) and wrote "
                f"{summary['camera_clips_written']} camera clip(s) from {summary['sessions']} session(s) to "
                f"{summary['episodes_lance_uri']} at Lance v{summary['episodes_lance_version']}, with "
                f"{summary['failures']} failure(s)."
            ),
        )

    return run


MULTIMODAL_SPLIT_KIND = PipelineKind(
    name="multimodal-split",
    template_yaml=_template_yaml,
    template_payload=_template_payload,
    validate=_validate,
    render=_render,
    schema_json=_schema_json,
    list_presets=_list_presets,
    prepare_run=_prepare_run,
)
