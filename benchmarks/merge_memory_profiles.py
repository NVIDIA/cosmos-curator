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

"""Summarize per-stage memray memory profiles into a combined report.

Standalone CLI tool -- not part of the main pipeline codebase.  Run it
after a pipeline execution to aggregate all ``.bin`` files produced by
``--profile-memory`` into a sorted summary table.

Unlike CPU profiles (pyinstrument), memray captures cannot be merged
into a single combined session.  This script instead:

1. Discovers all ``.bin`` files in the given directory.
2. Computes statistics for each capture (peak memory, total allocated,
   allocation count).
3. Prints a sorted summary table so the heaviest stages are immediately
   visible.
4. Optionally generates HTML flamegraphs for ``.bin`` files that are
   missing a corresponding ``.html``.

Usage examples::

    # Default: summary table sorted by peak memory
    python -m benchmarks.merge_memory_profiles /path/to/profiles/memory/

    # Sort by total allocated
    python -m benchmarks.merge_memory_profiles /path/to/profiles/memory/ --sort-by total

    # JSON output for programmatic consumption
    python -m benchmarks.merge_memory_profiles /path/to/profiles/memory/ --format json -o summary.json

    # Also generate HTML flamegraphs for .bin files missing them
    python -m benchmarks.merge_memory_profiles /path/to/profiles/memory/ --generate-flamegraphs

File naming convention (parsed from filenames)::

    <StageName>_<label>_<call_count>_<hostname>_<pid>.bin

Examples:
        RemuxStage_setup_on_node_1_node03_5819.bin
        ClipTranscodingStage_process_data_3_node03_10552.bin
        _root_main_1_node03_118.bin

"""

import argparse
import json
import pathlib
import re
import sys

import pandas as pd
from loguru import logger

from cosmos_curator.core.utils.misc import summarize

# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------

#                            +-- stage name (greedy)
#                            |             +-- label_count (e.g. setup_on_node_1)
#                            |             |              +-- hostname
#                            |             |              |       +-- pid
_FILENAME_RE = re.compile(r"^(.+?)_(\w+_\d+)_([^_]+)_(\d+)\.bin$")


def _parse_filename(name: str) -> dict[str, str]:
    """Extract structured fields from a profiling artifact filename.

    Returns a dict with keys: stage, label, hostname, pid.
    If the filename does not match the convention, returns a dict
    with only 'stage' set to the raw stem.

    Args:
        name: The filename (without directory) to parse.

    Returns:
        Dict with extracted fields.

    """
    m = _FILENAME_RE.match(name)
    if m:
        return {
            "stage": m.group(1),
            "label": m.group(2),
            "hostname": m.group(3),
            "pid": m.group(4),
        }
    # Fallback: use the stem as stage name.
    return {"stage": pathlib.Path(name).stem, "label": "", "hostname": "", "pid": ""}


# ---------------------------------------------------------------------------
# Human-readable size formatting (standalone, no memray dependency)
# ---------------------------------------------------------------------------

# Binary kibibyte divisor used for B -> KiB -> MiB -> GiB -> TiB conversion.
_KIB = 1024.0


def _size_fmt(num_bytes: int) -> str:
    """Format a byte count as a human-readable string.

    Uses binary prefixes (KiB, MiB, GiB, TiB).

    Args:
        num_bytes: Number of bytes.

    Returns:
        Formatted string like ``"1.23 GiB"`` or ``"456 B"``.

    """
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(num_bytes) < _KIB or unit == "TiB":
            return f"{num_bytes:.2f} {unit}" if unit != "B" else f"{num_bytes} {unit}"
        num_bytes /= _KIB  # type: ignore[assignment]
    return f"{num_bytes:.2f} TiB"


# ---------------------------------------------------------------------------
# Discovery and stats computation
# ---------------------------------------------------------------------------


def _discover_bins(directory: pathlib.Path) -> list[pathlib.Path]:
    """Glob ``*.bin`` files in *directory*, sorted by name.

    Excludes memray fork-fragment files (e.g. ``*.bin.187``) which are
    partial captures from forked child processes.

    Args:
        directory: Path to the directory containing ``.bin`` files.

    Returns:
        Sorted list of discovered capture file paths.

    Raises:
        FileNotFoundError: If the directory does not exist.

    """
    if not directory.is_dir():
        msg = f"Directory does not exist: {directory}"
        raise FileNotFoundError(msg)

    # Only match *.bin, not *.bin.NNN (fork fragments).
    return sorted(p for p in directory.glob("*.bin") if p.suffix == ".bin")


def _compute_stats(path: pathlib.Path, *, num_largest: int = 5) -> dict[str, object]:
    """Compute memory statistics for a single ``.bin`` capture.

    Args:
        path: Path to the memray ``.bin`` file.
        num_largest: Number of top allocation sites to include.

    Returns:
        Dict with keys: path, parsed (filename fields),
        peak_memory, total_allocated, total_allocations,
        top_by_size, top_by_count, duration_s, error.

    """
    parsed = _parse_filename(path.name)
    result: dict[str, object] = {"path": str(path), "parsed": parsed}

    try:
        from memray._memray import compute_statistics  # noqa: PLC0415

        stats = compute_statistics(
            str(path),
            report_progress=False,
            num_largest=num_largest,
        )

        result["peak_memory"] = stats.metadata.peak_memory
        result["total_allocated"] = stats.total_memory_allocated
        result["total_allocations"] = stats.total_num_allocations

        # Duration from metadata timestamps.
        dt = stats.metadata.end_time - stats.metadata.start_time
        result["duration_s"] = dt.total_seconds()

        # Top allocation sites formatted as readable strings.
        result["top_by_size"] = [
            {"location": f"{fn}:{f}:{ln}", "bytes": sz} for (fn, f, ln), sz in stats.top_locations_by_size
        ]
        result["top_by_count"] = [
            {"location": f"{fn}:{f}:{ln}", "count": ct} for (fn, f, ln), ct in stats.top_locations_by_count
        ]
        result["error"] = None

    except Exception as e:  # noqa: BLE001
        logger.warning(f"Failed to compute stats for {path.name}: {e}")
        result["peak_memory"] = 0
        result["total_allocated"] = 0
        result["total_allocations"] = 0
        result["duration_s"] = 0.0
        result["top_by_size"] = []
        result["top_by_count"] = []
        result["error"] = str(e)

    return result


# ---------------------------------------------------------------------------
# Flamegraph generation
# ---------------------------------------------------------------------------


def _generate_flamegraph(bin_path: pathlib.Path) -> pathlib.Path | None:
    """Generate an HTML flamegraph for a ``.bin`` capture.

    Skips generation if a corresponding ``.html`` file already exists.

    Args:
        bin_path: Path to the memray ``.bin`` file.

    Returns:
        Path to the generated ``.html`` file, or ``None`` if skipped
        or generation failed.

    """
    html_path = bin_path.with_suffix(".html")
    if html_path.exists():
        return None

    try:
        from memray import FileReader  # noqa: PLC0415
        from memray.reporters.flamegraph import FlameGraphReporter  # noqa: PLC0415

        reader = FileReader(str(bin_path))
        snapshot = reader.get_high_watermark_allocation_records(merge_threads=True)
        reporter = FlameGraphReporter.from_snapshot(
            snapshot,
            memory_records=tuple(reader.get_memory_snapshots()),
            native_traces=reader.metadata.has_native_traces,
        )
        with html_path.open("w") as f:
            reporter.render(
                outfile=f,
                metadata=reader.metadata,
                show_memory_leaks=False,
                merge_threads=True,
                inverted=False,
            )
        logger.info(f"  Generated flamegraph: {html_path.name}")
    except Exception as e:  # noqa: BLE001
        logger.warning(f"  Failed to generate flamegraph for {bin_path.name}: {e}")
        return None
    else:
        return html_path


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

# Sort key mapping: field name -> lambda extracting the sort value.
_SORT_KEYS: dict[str, str] = {
    "peak": "peak_memory",
    "total": "total_allocated",
    "allocations": "total_allocations",
    "duration": "duration_s",
}


def _build_dataframe(
    all_stats: list[dict[str, object]],
    sort_by: str,
) -> pd.DataFrame:
    """Build a pandas DataFrame from per-capture stats.

    Args:
        all_stats: List of per-capture stats dicts.
        sort_by: Sort key name (peak, total, allocations, duration).

    Returns:
        DataFrame with one row per capture, sorted descending by
        *sort_by*.

    """
    sort_field = _SORT_KEYS.get(sort_by, "peak_memory")
    rows: list[dict[str, object]] = []
    for entry in all_stats:
        parsed: dict[str, str] = entry.get("parsed", {})  # type: ignore[assignment]
        rows.append(
            {
                "Stage": parsed.get("stage", "?"),
                "Label": parsed.get("label", "?"),
                "Peak Memory": _size_fmt(entry.get("peak_memory", 0)),  # type: ignore[arg-type]
                "Total Allocated": _size_fmt(entry.get("total_allocated", 0)),  # type: ignore[arg-type]
                "Allocations": entry.get("total_allocations", 0),
                "Duration (s)": round(entry.get("duration_s", 0.0), 1),  # type: ignore[call-overload]
                "Host:PID": f"{parsed.get('hostname', '?')}:{parsed.get('pid', '?')}",
                # Hidden sort column (raw bytes for correct numeric ordering).
                "_sort": entry.get(sort_field, 0),
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("_sort", ascending=False).drop(columns=["_sort"]).reset_index(drop=True)
    return df


def _render_text(
    all_stats: list[dict[str, object]],
    *,
    sort_by: str,
    top_n: int,
    outfile: pathlib.Path | None,
) -> None:
    """Render a text summary table to stdout or a file.

    Uses ``pd.DataFrame`` for clean tabular output, consistent with
    ``dump_and_write_perf_stats`` in ``performance_utils.py``.

    Args:
        all_stats: List of per-capture stats dicts.
        sort_by: Sort key name (peak, total, allocations, duration).
        top_n: Number of top allocation sites to show per capture.
        outfile: Output file path, or ``None`` for stdout.

    """
    df = _build_dataframe(all_stats, sort_by)

    lines: list[str] = []
    lines.append("Memory Profile Summary")
    lines.append("=" * 80)
    lines.append("")

    # ---- Summary table via pandas ----
    with summarize.turn_off_pandas_display_limits():
        lines.append(df.to_string(index=False))

    # ---- Top allocation sites across all captures ----
    if top_n > 0:
        site_totals: dict[str, int] = {}
        for entry in all_stats:
            for site in entry.get("top_by_size", []):  # type: ignore[attr-defined]
                loc = site["location"]
                site_totals[loc] = max(site_totals.get(loc, 0), site["bytes"])

        ranked = sorted(site_totals.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
        if ranked:
            lines.append("")
            lines.append(f"Top {top_n} allocation sites (largest single allocation across all captures):")
            for i, (loc, sz) in enumerate(ranked, 1):
                lines.append(f"  {i:>3d}. {loc} -> {_size_fmt(sz)}")

    # ---- Error notes ----
    errors = [e for e in all_stats if e.get("error")]
    if errors:
        lines.append("")
        lines.append("* Captures with errors (stats may be incomplete):")
        for entry in errors:
            parsed: dict[str, str] = entry.get("parsed", {})  # type: ignore[assignment]
            lines.append(f"  - {parsed.get('stage', '?')}_{parsed.get('label', '?')}: {entry['error']}")

    lines.append("")
    output = "\n".join(lines)

    if outfile is not None:
        outfile.write_text(output)
        logger.info(f"Text report written to: {outfile}")
    else:
        sys.stdout.write(output)


def _render_json(
    all_stats: list[dict[str, object]],
    *,
    sort_by: str,
    outfile: pathlib.Path | None,
) -> None:
    """Render stats as JSON to stdout or a file.

    Args:
        all_stats: List of per-capture stats dicts.
        sort_by: Sort key name.
        outfile: Output file path, or ``None`` for stdout.

    """
    sort_field = _SORT_KEYS.get(sort_by, "peak_memory")
    sorted_stats = sorted(all_stats, key=lambda s: s.get(sort_field, 0), reverse=True)  # type: ignore[arg-type, return-value]

    output = json.dumps(sorted_stats, indent=2, default=str)

    if outfile is not None:
        outfile.write_text(output)
        logger.info(f"JSON report written to: {outfile}")
    else:
        sys.stdout.write(output)
        sys.stdout.write("\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    """Entry point for the memory profile summary CLI.

    Args:
        argv: Command-line arguments (defaults to ``sys.argv[1:]``).

    """
    parser = argparse.ArgumentParser(
        description="Summarize per-stage memray .bin memory profiles.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m benchmarks.merge_memory_profiles ./profiles/memory/\n"
            "  python -m benchmarks.merge_memory_profiles ./profiles/memory/ --sort-by total\n"
            "  python -m benchmarks.merge_memory_profiles ./profiles/memory/ --format json -o summary.json\n"
            "  python -m benchmarks.merge_memory_profiles ./profiles/memory/ --generate-flamegraphs\n"
        ),
    )
    parser.add_argument(
        "directory",
        type=pathlib.Path,
        help="Directory containing .bin files to summarize.",
    )
    parser.add_argument(
        "-o",
        "--outfile",
        type=pathlib.Path,
        default=None,
        help="Output file path. Defaults to stdout.",
    )
    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        dest="fmt",
        help="Output format (default: text).",
    )
    parser.add_argument(
        "--sort-by",
        choices=list(_SORT_KEYS.keys()),
        default="peak",
        help="Sort captures by this metric (default: peak).",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=5,
        metavar="N",
        help="Number of top allocation sites to show (default: 5).",
    )
    parser.add_argument(
        "--generate-flamegraphs",
        action="store_true",
        default=False,
        help="Generate HTML flamegraphs for .bin files missing them.",
    )

    args = parser.parse_args(argv)

    # ----------------------------------------------------------------
    # 1. Discover .bin files
    # ----------------------------------------------------------------
    logger.info(f"Scanning: {args.directory}")
    paths = _discover_bins(args.directory)

    if not paths:
        logger.error(f"No .bin files found in {args.directory}")
        sys.exit(1)

    logger.info(f"Found {len(paths)} capture(s)")

    # ----------------------------------------------------------------
    # 2. Compute stats for each capture
    # ----------------------------------------------------------------
    all_stats: list[dict[str, object]] = []
    for p in paths:
        logger.info(f"  Processing: {p.name}")
        all_stats.append(_compute_stats(p, num_largest=args.top))

    # ----------------------------------------------------------------
    # 3. Optionally generate missing flamegraphs
    # ----------------------------------------------------------------
    if args.generate_flamegraphs:
        generated = 0
        for p in paths:
            result = _generate_flamegraph(p)
            if result is not None:
                generated += 1
        if generated:
            logger.info(f"Generated {generated} flamegraph(s)")
        else:
            logger.info("All flamegraphs already exist")

    # ----------------------------------------------------------------
    # 4. Render output
    # ----------------------------------------------------------------
    if args.fmt == "json":
        _render_json(all_stats, sort_by=args.sort_by, outfile=args.outfile)
    else:
        _render_text(all_stats, sort_by=args.sort_by, top_n=args.top, outfile=args.outfile)


if __name__ == "__main__":
    main()
