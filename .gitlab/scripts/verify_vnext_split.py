#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fail unless ``video-cut`` published valid spans and no source errors.

Used by the ``vnext_fixed_stride_caption`` k8s_gpu_extended scenario. A zero exit code from the recipe does not
show that anything was cut: source failures are recorded as rows in the table, not raised.
"""
# ruff: noqa: T201

import argparse
import sys

import lance

from cosmos_curator.core.utils.storage.storage_utils import get_lance_storage_options


def main() -> int:
    """Validate every row of the published span table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spans_lance_uri", help="Span table written by video-cut.")
    parser.add_argument("--storage-profile", default="default")
    args = parser.parse_args()

    storage_options = get_lance_storage_options(args.spans_lance_uri, profile_name=args.storage_profile)
    table = lance.dataset(args.spans_lance_uri, storage_options=storage_options).to_table(
        columns=["source_uri", "span_id", "start_ns", "end_ns", "error_stage", "error_message"],
    )
    rows = table.to_pylist()

    failures = [
        f"{row['source_uri']}: {row['error_stage']}: {row['error_message']}" for row in rows if row["error_stage"]
    ]
    spans = [row for row in rows if row["span_id"] is not None]
    failures.extend(
        f"{row['span_id']}: end_ns {row['end_ns']} is not after start_ns {row['start_ns']}"
        for row in spans
        if not row["end_ns"] > row["start_ns"]
    )
    if not spans:
        failures.append("the table has no spans, so nothing was cut")
    if len({row["span_id"] for row in spans}) != len(spans):
        failures.append("span_id values are not unique")

    if failures:
        print(f"FAIL: {len(failures)} problem(s) across {len(rows)} row(s):", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1
    print(f"OK: {len(spans)} span(s) from {len(table.column('source_uri').unique())} source(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
