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
"""CLI for comparing split pipeline output summaries."""

import argparse
import sys
import time
from collections.abc import Sequence

from cosmos_curator.core.utils.storage.storage_utils import StorageWriter
from cosmos_curator.pipelines.video.output_comparison.comparison import compare_split_outputs
from cosmos_curator.pipelines.video.output_comparison.report import ComparisonReport, Issue, report_to_json

MAX_STDOUT_ISSUES = 5
_CAPTION_METADATA_LIMITATION = (
    "Caption artifact comparison currently supports only per-clip JSON metadata at "
    "metas/v0/<clip_uuid>.json. Outputs written with --upload-clip-info-in-chunks or "
    "--upload-clip-info-in-lance are not loaded for caption window comparison yet."
)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the split output comparison CLI.

    Args:
        argv: Optional argument sequence. When ``None``, arguments are read from ``sys.argv``.

    Returns:
        Process exit code. Returns ``0`` when comparison passes and ``1`` otherwise.

    """
    args = _build_parser().parse_args(argv)
    comparison_started_at = time.perf_counter()
    report = compare_split_outputs(
        args.output_a,
        args.output_b,
        profile_name=args.profile_name,
        token_count_abs_tolerance=args.token_count_abs_tolerance,
        token_count_rel_tolerance=args.token_count_rel_tolerance,
        video_limit=args.limit,
        selected_video_key=args.selected_video_key,
    )
    comparison_runtime_seconds = time.perf_counter() - comparison_started_at
    StorageWriter(args.report_path, profile_name=args.profile_name).write_str(f"{report_to_json(report)}\n")
    sys.stdout.write(
        _format_stdout_summary(
            report,
            args.report_path,
            comparison_runtime_seconds=comparison_runtime_seconds,
        )
    )
    return 0 if report.passed else 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare split pipeline output summary accounting.",
        epilog=_CAPTION_METADATA_LIMITATION,
    )
    parser.add_argument("output_a", help="First split pipeline output root.")
    parser.add_argument("output_b", help="Second split pipeline output root.")
    parser.add_argument("--report-path", required=True, help="Path to write the structured JSON report.")
    parser.add_argument("--profile-name", default="default", help="Storage profile name for remote paths.")
    selector_group = parser.add_mutually_exclusive_group()
    selector_group.add_argument(
        "--limit",
        type=_non_negative_int,
        default=None,
        help="Limit video-level feature comparison to the first N video keys from output A.",
    )
    selector_group.add_argument(
        "--video-key",
        dest="selected_video_key",
        default=None,
        help="Limit video-level feature comparison to one exact summary video key.",
    )
    parser.add_argument(
        "--token-count-abs-tolerance",
        type=float,
        default=0,
        help="Absolute tolerance for token total differences.",
    )
    parser.add_argument(
        "--token-count-rel-tolerance",
        type=float,
        default=0.01,
        help="Relative tolerance for token total differences.",
    )
    return parser


def _non_negative_int(value: str) -> int:
    limit = int(value)
    if limit < 0:
        msg = "--limit must be greater than or equal to 0"
        raise argparse.ArgumentTypeError(msg)
    return limit


def _format_stdout_summary(
    report: ComparisonReport,
    report_path: str,
    *,
    comparison_runtime_seconds: float | None = None,
) -> str:
    status = "PASSED" if report.passed else "FAILED"
    summary = report.summary_comparison
    issues = report.issues
    lines = [
        f"{status} split output comparison",
        (
            f"videos in both: {summary.videos_in_both}, "
            f"only in A: {len(summary.videos_only_in_a)}, "
            f"only in B: {len(summary.videos_only_in_b)}, "
            f"issues: {len(issues)}"
        ),
    ]
    if issues:
        lines.append("first issues:")
        lines.extend(_format_issue(issue) for issue in issues[:MAX_STDOUT_ISSUES])
        remaining_issue_count = len(issues) - MAX_STDOUT_ISSUES
        if remaining_issue_count > 0:
            lines.append(f"- {remaining_issue_count} more issues omitted from stdout")
    if comparison_runtime_seconds is not None:
        lines.append(f"comparison runtime: {comparison_runtime_seconds:.2f}s")
    lines.append(f"report: {report_path}")
    return "\n".join(lines) + "\n"


def _format_issue(issue: Issue) -> str:
    suffix_parts = []
    if issue.feature is not None:
        suffix_parts.append(f"feature={issue.feature}")
    if issue.video is not None:
        suffix_parts.append(f"video={issue.video}")
    if issue.clip is not None:
        suffix_parts.append(f"clip={issue.clip}")
    if issue.field is not None:
        suffix_parts.append(f"field={issue.field}")
    if issue.details is not None:
        error_type = issue.details.get("error_type")
        if isinstance(error_type, str):
            suffix_parts.append(f"error_type={error_type}")
    suffix = f" ({', '.join(suffix_parts)})" if suffix_parts else ""
    return f"- {issue.code}: {issue.message}{suffix}"


if __name__ == "__main__":
    raise SystemExit(main())
