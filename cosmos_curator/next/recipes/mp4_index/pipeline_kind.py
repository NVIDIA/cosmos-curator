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

"""Lazy CLI adapters for MP4 indexing recipes."""

from collections.abc import Sequence
from importlib import import_module
from pathlib import Path
from typing import cast

from cosmos_curator.next.core.pipeline_kind import PipelineKind, PipelinePreset, PipelineRunOutput, PreparedPipelineRun


def make_mp4_pipeline_kind(*, name: str, config_module: str, pipeline_module: str) -> PipelineKind:
    """Bind MP4 config callbacks and a deferred runtime without importing either at registration."""

    def template_yaml() -> str:
        return cast("str", import_module(config_module).config_template_yaml())

    def template_payload() -> dict[str, object]:
        return cast("dict[str, object]", import_module(config_module).config_template_payload())

    def validate(config: Path, overrides: Sequence[str]) -> dict[str, object]:
        import_module(config_module).resolve_config(config, overrides=overrides)
        return {"ok": True}

    def render(config: Path, overrides: Sequence[str]) -> str:
        module = import_module(config_module)
        return cast("str", module.resolved_config_to_json(module.resolve_config(config, overrides=overrides)))

    def schema_json() -> str:
        return cast("str", import_module(config_module).config_schema_json())

    def list_presets() -> list[PipelinePreset]:
        return []

    def prepare_run(config: Path, *, set_overrides: list[str]) -> PreparedPipelineRun:
        resolved = import_module(config_module).resolve_config(config, overrides=set_overrides)

        def run() -> PipelineRunOutput:
            summary = cast("dict[str, object]", import_module(pipeline_module).run_config(resolved))
            return PipelineRunOutput(
                json_payload=summary,
                message=f"{name}: {summary['videos_extracted']} video(s) indexed, {summary['videos_failed']} error(s)",
            )

        return run

    return PipelineKind(
        name=name,
        template_yaml=template_yaml,
        template_payload=template_payload,
        validate=validate,
        render=render,
        schema_json=schema_json,
        list_presets=list_presets,
        prepare_run=prepare_run,
    )


MP4_INDEX_KIND = make_mp4_pipeline_kind(
    name="mp4-index",
    config_module="cosmos_curator.next.recipes.mp4_index.config",
    pipeline_module="cosmos_curator.next.recipes.mp4_index.pipeline",
)
