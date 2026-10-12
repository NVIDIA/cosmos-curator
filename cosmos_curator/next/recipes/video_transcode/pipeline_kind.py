# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lazy generic CLI adapter for checkpointed span transcoding."""

from collections.abc import Sequence
from pathlib import Path

from cosmos_curator.next.core.pipeline_kind import PipelineKind, PipelinePreset, PipelineRunOutput, PreparedPipelineRun


def _template_yaml() -> str:
    from cosmos_curator.next.recipes.video_transcode.config import config_template_yaml  # noqa: PLC0415

    return config_template_yaml()


def _template_payload() -> dict[str, object]:
    from cosmos_curator.next.recipes.video_transcode.config import config_template_payload  # noqa: PLC0415

    return config_template_payload()


def _validate(config: Path, overrides: Sequence[str]) -> dict[str, object]:
    from cosmos_curator.next.recipes.video_transcode.config import resolve_config  # noqa: PLC0415

    resolve_config(config, overrides=overrides)
    return {"ok": True}


def _render(config: Path, overrides: Sequence[str]) -> str:
    from cosmos_curator.next.recipes.video_transcode.config import (  # noqa: PLC0415
        resolve_config,
        resolved_config_to_json,
    )

    return resolved_config_to_json(resolve_config(config, overrides=overrides))


def _schema_json() -> str:
    from cosmos_curator.next.recipes.video_transcode.config import config_schema_json  # noqa: PLC0415

    return config_schema_json()


def _list_presets() -> list[PipelinePreset]:
    return []


def _prepare_run(config: Path, *, set_overrides: list[str]) -> PreparedPipelineRun:
    from cosmos_curator.next.recipes.video_transcode.config import resolve_config  # noqa: PLC0415

    resolved = resolve_config(config, overrides=set_overrides)

    def run() -> PipelineRunOutput:
        from cosmos_curator.next.enrichment.execution import summary_message  # noqa: PLC0415
        from cosmos_curator.next.recipes.video_transcode.pipeline import run_config  # noqa: PLC0415

        summary = run_config(resolved)
        return PipelineRunOutput(json_payload=summary, message=summary_message("video-transcode", summary))

    return run


VIDEO_TRANSCODE_KIND = PipelineKind(
    name="video-transcode",
    template_yaml=_template_yaml,
    template_payload=_template_payload,
    validate=_validate,
    render=_render,
    schema_json=_schema_json,
    list_presets=_list_presets,
    prepare_run=_prepare_run,
)
