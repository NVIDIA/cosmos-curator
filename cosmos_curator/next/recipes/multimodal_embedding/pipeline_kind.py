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

"""Config-backed pipeline kind for ``multimodal-embedding``."""

from collections.abc import Sequence
from pathlib import Path

from cosmos_curator.next.core.pipeline_kind import PipelineKind, PipelinePreset, PipelineRunOutput, PreparedPipelineRun


def _template_yaml() -> str:
    from cosmos_curator.next.recipes.multimodal_embedding.config import config_template_yaml  # noqa: PLC0415

    return config_template_yaml()


def _template_payload() -> dict[str, object]:
    from cosmos_curator.next.recipes.multimodal_embedding.config import config_template_payload  # noqa: PLC0415

    return config_template_payload()


def _validate(config: Path, overrides: Sequence[str]) -> dict[str, object]:
    from cosmos_curator.next.recipes.multimodal_embedding.config import resolve_config  # noqa: PLC0415

    resolve_config(config, overrides=overrides)
    return {"ok": True}


def _render(config: Path, overrides: Sequence[str]) -> str:
    from cosmos_curator.next.recipes.multimodal_embedding.config import (  # noqa: PLC0415
        resolve_config,
        resolved_config_to_json,
    )

    return resolved_config_to_json(resolve_config(config, overrides=overrides))


def _schema_json() -> str:
    from cosmos_curator.next.recipes.multimodal_embedding.config import config_schema_json  # noqa: PLC0415

    return config_schema_json()


def _list_presets() -> list[PipelinePreset]:
    return []


def _prepare_run(config: Path, *, set_overrides: list[str]) -> PreparedPipelineRun:
    from cosmos_curator.next.recipes.multimodal_embedding.config import resolve_config  # noqa: PLC0415

    resolved = resolve_config(config, overrides=set_overrides)

    def run() -> PipelineRunOutput:
        from cosmos_curator.next.recipes.multimodal_embedding.pipeline import run_config  # noqa: PLC0415

        summary = run_config(resolved)
        return PipelineRunOutput(
            json_payload=summary,
            message=(
                f"multimodal-embedding: {summary['published_fragments']} fragment(s) published, "
                f"{summary['skipped_complete_fragments']} complete fragment(s) skipped"
            ),
        )

    return run


MULTIMODAL_EMBEDDING_KIND = PipelineKind(
    name="multimodal-embedding",
    template_yaml=_template_yaml,
    template_payload=_template_payload,
    validate=_validate,
    render=_render,
    schema_json=_schema_json,
    list_presets=_list_presets,
    prepare_run=_prepare_run,
)
