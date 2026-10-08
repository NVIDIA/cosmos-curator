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

"""Run-only entrypoint for config-backed pipelines inside runtime environments.

This module owns the process exit status and derives it entirely from whether
preparation or the prepared run raised::

    0  the run returned; its payload (--json) or message is on stdout
    2  a config fault in either mode, and a run fault under --json. Reported on
       stderr - as {"ok": false, "error": "invalid"|"runtime", "message": ...}
       under --json, as one plain line without it
    1  a run fault without --json, where the exception re-raises as an ordinary
       traceback

A kind that published its work but still owes more therefore reports it by
raising, which costs the summary: under --json the error object REPLACES the
payload rather than joining it, so stdout stays empty. The two cannot both be had
without a third outcome here, which no caller has yet needed.

When ``COSMOS_CURATOR_OTLP_METRICS_PUSH`` is truthy the run is wrapped in
``metrics_push.otlp_push_scope``; a broken observability stack warns and
degrades to a plain run, and a plain run never imports the pusher.
"""

import json
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from typing import Annotated, NoReturn

import typer
from loguru import logger
from pydantic import ValidationError
from typer import Argument, Option

from cosmos_curator.core.utils.environment import OTLP_METRICS_PUSH_ENV_VAR, env_bool
from cosmos_curator.pipelines.builtin_pipeline_kinds import BUILTIN_PIPELINE_KINDS
from cosmos_curator.pipelines.pipeline_config import load_pipeline_kind_name


def main(
    config: Annotated[Path, Argument(help="Path to a JSON/YAML pipeline config.")],
    set_overrides: Annotated[
        list[str] | None,
        Option("--set", help="Small resolved-config override in dotted PATH=VALUE form."),
    ] = None,
    *,
    json_output: Annotated[bool, Option("--json", help="Emit machine-readable JSON output.")] = False,
) -> None:
    """Run a pipeline from a JSON/YAML config."""
    try:
        pipeline_kind = BUILTIN_PIPELINE_KINDS.get(load_pipeline_kind_name(config))
        run_pipeline = pipeline_kind.prepare_run(config, set_overrides=set_overrides or [])
    except (OSError, TypeError, ValueError, ValidationError) as exc:
        _fail("invalid", exc, json_output=json_output)

    try:
        with _metrics_push_scope():
            output = run_pipeline()
    except Exception as exc:
        if json_output:
            _fail("runtime", exc, json_output=True)
        raise

    if json_output:
        typer.echo(json.dumps(output.json_payload, indent=2))
    else:
        typer.echo(output.message)


def _metrics_push_scope() -> "AbstractContextManager[None]":
    """Return the OTLP push scope when enabled by env, else a no-op context."""
    if env_bool(OTLP_METRICS_PUSH_ENV_VAR) is not True:
        return nullcontext()
    try:
        # Deferred: a plain run must not pay for (or break on) the OTel imports.
        from cosmos_curator.core.utils.infra.metrics_push import otlp_push_scope  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"[otel-metrics] Failed to import metrics push hooks: {exc}")
        return nullcontext()
    return otlp_push_scope()


def _fail(code: str, exc: Exception, *, json_output: bool) -> NoReturn:
    if json_output:
        typer.echo(json.dumps({"ok": False, "error": code, "message": str(exc)}, indent=2), err=True)
    else:
        typer.echo(str(exc), err=True)
    raise typer.Exit(2)


if __name__ == "__main__":
    typer.run(main)
