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

"""Config-backed pipeline-kind surface for Curator Next ``data-integrity``.

Every import is deferred into the callable that needs it, so registering this kind
costs the CLI nothing: neither Ray nor ``lance`` loads until an operation asks for
them.
"""

from collections.abc import Sequence
from pathlib import Path

from cosmos_curator.next.core.pipeline_kind import PipelineKind, PipelinePreset, PipelineRunOutput, PreparedPipelineRun


def _template_yaml() -> str:
    from cosmos_curator.next.recipes.data_integrity.config import config_template_yaml  # noqa: PLC0415

    return config_template_yaml()


def _template_payload() -> dict[str, object]:
    from cosmos_curator.next.recipes.data_integrity.config import config_template_payload  # noqa: PLC0415

    return config_template_payload()


def _validate(config: Path, overrides: Sequence[str]) -> dict[str, object]:
    from cosmos_curator.next.recipes.data_integrity.config import (  # noqa: PLC0415
        require_store_root,
        resolve_config,
    )

    # Checked here as well as in _prepare_run so `pipeline validate` cannot pass a config
    # that `run-pipeline` then refuses.
    require_store_root(resolve_config(config, overrides=overrides))
    return {"ok": True}


def _render(config: Path, overrides: Sequence[str]) -> str:
    from cosmos_curator.next.recipes.data_integrity.config import (  # noqa: PLC0415
        resolve_config,
        resolved_config_to_json,
    )

    return resolved_config_to_json(resolve_config(config, overrides=overrides))


def _schema_json() -> str:
    from cosmos_curator.next.recipes.data_integrity.config import config_schema_json  # noqa: PLC0415

    return config_schema_json()


def _list_presets() -> list[PipelinePreset]:
    return []


def _prepare_run(
    config: Path,
    *,
    set_overrides: list[str],
) -> PreparedPipelineRun:
    from cosmos_curator.next.recipes.data_integrity.config import (  # noqa: PLC0415
        require_store_root,
        resolve_config,
    )

    # Resolved before the closure is returned, so a bad config fails here rather than
    # after Ray has started.
    resolved = resolve_config(config, overrides=set_overrides)
    require_store_root(resolved)

    def run() -> PipelineRunOutput:
        from cosmos_curator.next.recipes.data_integrity.pipeline import run_config  # noqa: PLC0415

        summary = run_config(resolved)
        # Findings are reported, never encoded in the exit status: a completed run
        # exits 0 however many streams failed their thresholds.
        message = (
            f"data-integrity: run {summary['run_id']} committed {summary['streams']} stream(s) "
            f"from {summary['sessions']} session(s); {summary['unreadable']} unreadable, "
            f"{summary['failed_metrics']} failed metric(s)"
        )
        # Only mentioned when there are any, unlike the counts above, which an operator
        # reads on every run. The unreachable and unlisted counts have no such branch:
        # a run that reaches this line has none, or it would have raised.
        if summary["empty_sessions"]:
            message += f"; {summary['empty_sessions']} session(s) held no streams"
        return PipelineRunOutput(json_payload=summary, message=message)

    return run


DATA_INTEGRITY_KIND = PipelineKind(
    name="data-integrity",
    template_yaml=_template_yaml,
    template_payload=_template_payload,
    validate=_validate,
    render=_render,
    schema_json=_schema_json,
    list_presets=_list_presets,
    prepare_run=_prepare_run,
)
