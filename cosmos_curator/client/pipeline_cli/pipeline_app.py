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

"""Config-driven pipeline commands."""

import json
import sys
from pathlib import Path
from typing import Annotated, NoReturn, cast

import typer
from typer import Argument, Option

from cosmos_curator.next.core.pipeline_kind import PipelinePreset
from cosmos_curator.pipelines.builtin_pipeline_kinds import BUILTIN_PIPELINE_KINDS
from cosmos_curator.pipelines.pipeline_config import load_pipeline_kind_name


def _complete_pipeline_kind(incomplete: str) -> list[str]:
    return [name for name in BUILTIN_PIPELINE_KINDS.names() if name.startswith(incomplete)]


_PIPELINE_KIND_HELP = f"Pipeline kind ({', '.join(BUILTIN_PIPELINE_KINDS.names())})."
_KIND_ARGUMENT = Argument(
    help=_PIPELINE_KIND_HELP,
    metavar="KIND",
    autocompletion=_complete_pipeline_kind,
)

pipeline_app = typer.Typer(
    help="Pipeline config tooling.",
    no_args_is_help=True,
)
presets_app = typer.Typer(
    help="Inspect packaged pipeline presets.",
    no_args_is_help=True,
)
pipeline_app.add_typer(presets_app, name="presets")


@pipeline_app.command(no_args_is_help=True)
def template(
    *,
    kind: Annotated[str, _KIND_ARGUMENT],
    json_output: Annotated[bool, Option("--json", help="Emit machine-readable JSON output.")] = False,
) -> None:
    """Print an editable config template for a supported pipeline kind."""
    try:
        pipeline_kind = BUILTIN_PIPELINE_KINDS.get(kind)
    except ValueError as exc:
        _fail("unknown_kind", exc, json_output=json_output)
    if json_output:
        typer.echo(json.dumps(pipeline_kind.template_payload(), indent=2))
    else:
        sys.stdout.write(pipeline_kind.template_yaml())


@pipeline_app.command(no_args_is_help=True)
def validate(
    *,
    config: Annotated[Path, Argument(help="Path to a JSON/YAML pipeline config.")],
    set_overrides: Annotated[
        list[str] | None,
        Option("--set", help="Small resolved-config override in dotted PATH=VALUE form."),
    ] = None,
    json_output: Annotated[bool, Option("--json", help="Emit machine-readable JSON output.")] = False,
) -> None:
    """Validate a config file after defaults, presets, and overrides resolve."""
    from pydantic import ValidationError  # noqa: PLC0415

    try:
        pipeline_kind = BUILTIN_PIPELINE_KINDS.get(load_pipeline_kind_name(config))
        payload = pipeline_kind.validate(config, set_overrides or [])
    except (OSError, TypeError, ValueError, ValidationError) as exc:
        _fail("invalid", exc, json_output=json_output)

    if json_output:
        typer.echo(json.dumps(payload, indent=2))
    else:
        typer.echo("valid")


@pipeline_app.command(no_args_is_help=True)
def render(
    *,
    config: Annotated[Path, Argument(help="Path to a JSON/YAML pipeline config.")],
    set_overrides: Annotated[
        list[str] | None,
        Option("--set", help="Small resolved-config override in dotted PATH=VALUE form."),
    ] = None,
    json_output: Annotated[bool, Option("--json", help="Emit machine-readable JSON output.")] = False,
) -> None:
    """Render the canonical resolved config used for execution."""
    from pydantic import ValidationError  # noqa: PLC0415

    try:
        pipeline_kind = BUILTIN_PIPELINE_KINDS.get(load_pipeline_kind_name(config))
        rendered = pipeline_kind.render(config, set_overrides or [])
    except (OSError, TypeError, ValueError, ValidationError) as exc:
        _fail("render_failed", exc, json_output=json_output)
    sys.stdout.write(rendered)


@pipeline_app.command(no_args_is_help=True)
def schema(
    *,
    kind: Annotated[str, _KIND_ARGUMENT],
    json_output: Annotated[bool, Option("--json", help="Emit machine-readable JSON output.")] = False,
) -> None:
    """Print JSON Schema for a supported pipeline config."""
    try:
        pipeline_kind = BUILTIN_PIPELINE_KINDS.get(kind)
    except ValueError as exc:
        _fail("unknown_kind", exc, json_output=json_output)
    sys.stdout.write(pipeline_kind.schema_json())


class _RegisteredPipelinePreset(PipelinePreset):
    kind: str


class _InvalidPresetRegistrationError(ValueError):
    """A pipeline kind returned malformed preset metadata."""


def _validate_registered_preset(kind: str, index: int, raw_preset: object) -> _RegisteredPipelinePreset:
    location = f"Pipeline kind {kind!r} preset at index {index}"
    if not isinstance(raw_preset, dict):
        msg = f"{location} must be a dictionary"
        raise _InvalidPresetRegistrationError(msg)

    for field in ("name", "qualified_name"):
        value = raw_preset.get(field)
        if not isinstance(value, str) or not value:
            msg = f"{location} has invalid {field!r}; expected a non-empty string"
            raise _InvalidPresetRegistrationError(msg)

    if not isinstance(raw_preset.get("fragment"), dict):
        msg = f"{location} has invalid 'fragment'; expected a dictionary"
        raise _InvalidPresetRegistrationError(msg)

    if "section" in raw_preset and not isinstance(raw_preset["section"], str):
        msg = f"{location} has invalid 'section'; expected a string"
        raise _InvalidPresetRegistrationError(msg)

    preset = cast("_RegisteredPipelinePreset", dict(raw_preset))
    preset["kind"] = kind
    return preset


def _registered_presets() -> list[_RegisteredPipelinePreset]:
    presets: list[_RegisteredPipelinePreset] = []
    for pipeline_kind in BUILTIN_PIPELINE_KINDS:
        for index, preset in enumerate(pipeline_kind.list_presets()):
            presets.append(_validate_registered_preset(pipeline_kind.name, index, preset))
    return presets


def _find_preset(name: str) -> _RegisteredPipelinePreset:
    matches = [
        preset
        for preset in _registered_presets()
        if preset["qualified_name"] == name or ("." not in name and preset["name"] == name)
    ]
    if not matches:
        msg = f"Unknown pipeline preset: {name}"
        raise ValueError(msg)
    if len(matches) > 1:
        choices = ", ".join(f"{preset['kind']}:{preset['qualified_name']}" for preset in matches)
        msg = f"Ambiguous preset {name!r}; matches: {choices}"
        raise ValueError(msg)
    return matches[0]


@presets_app.command("list")
def list_presets(
    *,
    json_output: Annotated[bool, Option("--json", help="Emit machine-readable JSON output.")] = False,
) -> None:
    """List presets exposed by registered pipeline kinds."""
    try:
        presets = _registered_presets()
    except _InvalidPresetRegistrationError as exc:
        _fail("invalid_preset", exc, json_output=json_output)
    if json_output:
        typer.echo(json.dumps({"presets": presets}, indent=2))
        return

    for preset in presets:
        typer.echo(f"{preset['qualified_name']}")


@presets_app.command("show", no_args_is_help=True)
def show_preset(
    *,
    name: Annotated[str, Argument(help="Preset name, e.g. caption.balanced or balanced.")],
    json_output: Annotated[bool, Option("--json", help="Emit machine-readable JSON output.")] = False,
) -> None:
    """Show one registered preset by qualified or unique short name."""
    try:
        preset = _find_preset(name)
    except _InvalidPresetRegistrationError as exc:
        _fail("invalid_preset", exc, json_output=json_output)
    except ValueError as exc:
        _fail("unknown_preset", exc, json_output=json_output)

    if json_output:
        typer.echo(json.dumps(preset, indent=2))
    else:
        typer.echo(json.dumps(preset["fragment"], indent=2))


def _fail(code: str, exc: Exception, *, json_output: bool) -> NoReturn:
    if json_output:
        typer.echo(json.dumps({"ok": False, "error": code, "message": str(exc)}, indent=2), err=True)
    else:
        typer.echo(str(exc), err=True)
    raise typer.Exit(2)


if __name__ == "__main__":
    pipeline_app()
