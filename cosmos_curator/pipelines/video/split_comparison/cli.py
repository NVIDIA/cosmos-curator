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
"""Command-line entry point for split-comparison.

Two surfaces, both config-driven:

  python -m cosmos_curator.pipelines.video.split_comparison.cli --config audit.json     # load + run
  python -m cosmos_curator.pipelines.video.split_comparison.cli --print-default-config  # emit default JSON

Every tuning knob, both comparison targets, and the report destination live
inside :class:`SplitComparisonConfig`. The CLI is intentionally minimal --
add flags back only when ad-hoc override genuinely beats editing the config.
"""

import argparse
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import ray.data
from loguru import logger
from pydantic import ValidationError

from cosmos_curator.pipelines.video.split_comparison.config import (
    SplitComparisonConfig,
    example_default_config,
)
from cosmos_curator.pipelines.video.split_comparison.driver import compare_split_outputs
from cosmos_curator.pipelines.video.split_comparison.report_io import write_report
from cosmos_curator.pipelines.video.split_comparison.result_model import Report

_MAX_STDOUT_ISSUES = 5


def main(argv: Sequence[str] | None = None) -> int:
    """Parse args, run the comparison (or print default config), return an exit code."""
    args = _build_parser().parse_args(argv)

    if args.print_default_config:
        sys.stdout.write(example_default_config().model_dump_json(indent=2) + "\n")
        return 0

    try:
        config = _load_config(args.config)
    except (OSError, json.JSONDecodeError, ValidationError) as err:
        sys.stderr.write(f"Failed to load config from {args.config}: {err}\n")
        return 2

    _enable_ray_data_progress_ui()

    started = time.perf_counter()
    report = compare_split_outputs(config=config)
    path = write_report(report, config.report_path, report_format=config.report_format)
    elapsed = time.perf_counter() - started

    sys.stdout.write(_format_stdout_summary(report, path, elapsed_sec=elapsed))
    return 0 if report.passed else 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cosmos-curator split-compare",
        description="Compare two split-pipeline output trees and write a structured report.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--config",
        type=Path,
        metavar="PATH",
        help="Path to a JSON file conforming to SplitComparisonConfig. The audit spec lives here.",
    )
    group.add_argument(
        "--print-default-config",
        action="store_true",
        help=(
            "Emit the default config (with placeholder output paths) as indented JSON to stdout. "
            "Pipe to a file, replace REPLACE_WITH_OUTPUT_A_PATH / REPLACE_WITH_OUTPUT_B_PATH, "
            "edit knobs, then run with --config."
        ),
    )
    return parser


def _load_config(path: Path) -> SplitComparisonConfig:
    """Load + validate a config from a JSON file. Raises on read or validation errors."""
    return SplitComparisonConfig.model_validate_json(path.read_text())


def _enable_ray_data_progress_ui() -> None:
    """Enable Ray Data's tqdm/rich progress UI (CLI default)."""
    ray.data.DataContext.get_current().enable_rich_progress_bars = True
    ray.data.DataContext.get_current().use_ray_tqdm = False


def _format_stdout_summary(report: Report, path: str, *, elapsed_sec: float) -> str:
    verdict = "PASSED" if report.passed else "FAILED"
    lines: list[str] = [
        f"{verdict} split output comparison",
        f"stages run: {sorted(report.stages_run)}",
        f"issues: {report.issues.num_rows}",
    ]
    if not report.passed and report.issues.num_rows > 0:
        head = report.issues.slice(0, _MAX_STDOUT_ISSUES).to_pylist()
        lines.extend(f"- {_format_issue_line(row)}" for row in head)
        remaining = report.issues.num_rows - len(head)
        if remaining > 0:
            lines.append(f"- ... {remaining} more issues omitted from stdout")
    lines.append(f"comparison runtime: {elapsed_sec:.2f}s")
    lines.append(f"report: {path}")
    return "\n".join(lines) + "\n"


def _format_issue_line(row: dict[str, object]) -> str:
    code = row.get("code", "?")
    parts = [f"{code}: {row.get('message', '')}"]
    suffix: list[str] = []
    for key in ("feature", "video", "clip", "field", "output"):
        value = row.get(key)
        if value:
            suffix.append(f"{key}={value}")
    details = row.get("details")
    if isinstance(details, str):
        try:
            parsed = json.loads(details)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict) and parsed.get("error_type"):
            suffix.append(f"error_type={parsed['error_type']}")
    if suffix:
        parts.append(f"({', '.join(suffix)})")
    return " ".join(parts)


if __name__ == "__main__":
    # Mute any loguru records emitted under a "ray.*" logger name. Ray itself
    # uses stdlib ``logging`` and prints to stdout/stderr directly; that output
    # is not affected by this call -- silencing it would need stdlib-logging
    # config or stream redirection.
    logger.disable("ray")
    sys.exit(main())
