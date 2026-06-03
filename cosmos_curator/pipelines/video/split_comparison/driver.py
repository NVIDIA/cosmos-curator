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
"""Top-level driver: ``compare_split_outputs`` + the per-stage Ray Data runners.

``compare_split_outputs`` runs summary comparison then fans out to
``run_metadata_stage`` (Stage 1) and ``run_video_index_stage`` (Stage 2),
both also defined here. The stage modules own their actor classes and
per-row comparators; this file owns how those are wired into Ray Data (block
oversubscription, actor pool sizing, runtime env).

Every public entry point takes :class:`SplitComparisonConfig` as its only
argument -- comparison targets and tuning knobs both live on the config.
See ``docs/curator/design/split-comparison.md``.
"""

import math
import os
import time

import pyarrow as pa
import ray
from loguru import logger
from pydantic import ValidationError
from ray.data import ActorPoolStrategy

from cosmos_curator.core.utils.pixi_runtime_envs import PixiRuntimeEnv
from cosmos_curator.core.utils.storage import storage_utils
from cosmos_curator.pipelines.video.split_comparison.clip_discovery import discover_clips
from cosmos_curator.pipelines.video.split_comparison.config import SplitComparisonConfig
from cosmos_curator.pipelines.video.split_comparison.metadata_stage import MetadataStage
from cosmos_curator.pipelines.video.split_comparison.result_model import (
    ISSUE_SCHEMA,
    VIDEOS_SCHEMA,
    Issue,
    Report,
    empty_issues,
    empty_videos,
    make_issue,
)
from cosmos_curator.pipelines.video.split_comparison.summary_compare import compare_summaries
from cosmos_curator.pipelines.video.split_comparison.summary_loader import load_summary
from cosmos_curator.pipelines.video.split_comparison.summary_schema import OutputSummary
from cosmos_curator.pipelines.video.split_comparison.video_index_stage import VideoIndexStage


def compare_split_outputs(*, config: SplitComparisonConfig) -> Report:
    """Compare two split-pipeline outputs and return a :class:`Report`.

    See the design doc's "Top-level flow" section for the fan-in topology.
    Summary comparison runs first on the driver; clips are then discovered
    from the intersection of processed videos; Stage 1 (metadata) and Stage 2
    (video index) run as two independent Ray Data pipelines over the same
    clip table. Stage 2 is gated on ``config.compare_video_index``.
    """
    started = time.perf_counter()
    summary_a, issue_a = _load_summary(config.output_a, profile_name=config.profile_name, output_label="a")
    summary_b, issue_b = _load_summary(config.output_b, profile_name=config.profile_name, output_label="b")
    if summary_a is None or summary_b is None:
        # Clip discovery needs both summaries, so a load failure can't be recovered
        # mid-run. Short-circuit to a failing Report carrying the structured
        # summary_load_failed issue(s) rather than letting the exception escape the
        # CLI as a stack trace -- write_report still runs on the returned Report.
        # Videos table is empty here: without both summaries we can't enumerate
        # the videos union, and the viewer's graceful-fallback contract covers it.
        load_issues = [issue for issue in (issue_a, issue_b) if issue is not None]
        return Report(
            issues=pa.Table.from_pylist(load_issues, schema=ISSUE_SCHEMA),
            videos=empty_videos(),
            passed=False,
            stages_run=frozenset({"summary"}),
            output_a=config.output_a,
            output_b=config.output_b,
            runtime_sec=time.perf_counter() - started,
            config=config,
        )
    summary_issues = compare_summaries(summary_a, summary_b, config.summary)
    videos, source_a, source_b, layout_issues = _derive_videos_and_sources(summary_a, summary_b)

    logger.info("Discovering clips")
    clips = discover_clips(summary_a, summary_b)
    logger.info(f"Discovered {clips.num_rows} clips")

    if config.video_key is not None:
        # Filter before head-slice so clip_limit applies to the filtered set.
        mask = [vk == config.video_key for vk in clips["video_key"].to_pylist()]
        clips = clips.filter(pa.array(mask))
    if config.clip_limit is not None and clips.num_rows > config.clip_limit:
        clips = clips.slice(0, config.clip_limit)

    logger.info("Running metadata stage")
    metadata_issues = run_metadata_stage(clips, config=config) if clips.num_rows else empty_issues()

    if config.compare_video_index and clips.num_rows > 0:
        logger.info("Running video index stage")
        video_index_issues = run_video_index_stage(clips, config=config)
    else:
        video_index_issues = empty_issues()

    issues = pa.concat_tables([summary_issues, layout_issues, metadata_issues, video_index_issues])
    in_a, in_b, in_both = _clip_output_counts(clips)
    return Report(
        issues=issues,
        videos=videos,
        passed=issues.num_rows == 0,
        stages_run=_stages_run(config, ran_clip_work=clips.num_rows > 0),
        clip_count=clips.num_rows,
        clips_in_a=in_a,
        clips_in_b=in_b,
        clips_in_both=in_both,
        output_a=config.output_a,
        output_b=config.output_b,
        source_a=source_a,
        source_b=source_b,
        runtime_sec=time.perf_counter() - started,
        config=config,
    )


def _derive_videos_and_sources(
    summary_a: OutputSummary,
    summary_b: OutputSummary,
) -> tuple[pa.Table, str, str, pa.Table]:
    """Materialize the union videos table and derive each side's source root.

    Returns ``(videos, source_a, source_b, layout_issues)``:

    * ``videos`` -- one row per ``video_key`` in ``summary_a.videos |
      summary_b.videos``, sorted, with boolean ``in_a`` / ``in_b`` columns.
    * ``source_a`` / ``source_b`` -- the derived source-video roots per side.
      Computed by stripping ``video_key`` off the first entry's
      ``source_video`` and asserting every other entry on that side follows
      the same layout. ``""`` when the side has no videos or the assertion
      fails.
    * ``layout_issues`` -- ``ISSUE_SCHEMA`` table of
      ``summary_source_layout_inconsistent`` rows (one per side that failed
      the assertion); empty when both sides are uniform.
    """
    keys = sorted(set(summary_a.videos) | set(summary_b.videos))
    rows = [{"video_key": key, "in_a": key in summary_a.videos, "in_b": key in summary_b.videos} for key in keys]
    videos = pa.Table.from_pylist(rows, schema=VIDEOS_SCHEMA)

    source_a, layout_issue_a = _derive_source_root(summary_a, output_label="a")
    source_b, layout_issue_b = _derive_source_root(summary_b, output_label="b")
    issue_rows = [issue for issue in (layout_issue_a, layout_issue_b) if issue is not None]
    layout_issues = pa.Table.from_pylist(issue_rows, schema=ISSUE_SCHEMA)
    return videos, source_a, source_b, layout_issues


def _derive_source_root(summary: OutputSummary, *, output_label: str) -> tuple[str, Issue | None]:
    """Derive ``summary``'s source-video root and verify the layout is uniform.

    Picks the first ``(video_key, source_video)`` pair, strips ``video_key``
    off the end of ``source_video`` to recover the root, then asserts every
    other entry in ``summary.videos`` reconstructs the same way. Returns
    ``(root, None)`` on success or ``("", Issue)`` on failure -- the issue
    is a ``summary_source_layout_inconsistent`` row that flows into the
    Report's issues table so consumers see why source paths are missing.
    """
    if not summary.videos:
        return "", None
    first_key, first_video = next(iter(sorted(summary.videos.items())))
    first_source = first_video.source_video
    if not first_source.endswith(first_key):
        return "", make_issue(
            "summary_source_layout_inconsistent",
            f"source_video for first entry in output {output_label.upper()} does not end with its video_key",
            output=output_label,
            video=first_key,
            details={"video_key": first_key, "source_video": first_source},
        )
    root = first_source.removesuffix(first_key)
    for key, video in summary.videos.items():
        expected = root + key
        if video.source_video != expected:
            return "", make_issue(
                "summary_source_layout_inconsistent",
                f"source_video for {key!r} in output {output_label.upper()} does not match derived root + video_key",
                output=output_label,
                video=key,
                details={
                    "derived_root": root,
                    "video_key": key,
                    "expected": expected,
                    "actual": video.source_video,
                },
            )
    return root, None


def _load_summary(
    output_root: str,
    *,
    profile_name: str,
    output_label: str,
) -> tuple[OutputSummary | None, Issue | None]:
    """Load one output's ``summary.json``, or return a structured load-failure issue.

    Returns ``(summary, None)`` on success and ``(None, issue)`` on any failure
    (missing/unreadable file, malformed JSON, schema validation). The caller
    short-circuits when either side fails -- there is nothing to compare without
    both summaries.
    """
    summary_path = storage_utils.get_full_path(output_root, "summary.json")
    try:
        return load_summary(output_root, profile_name=profile_name), None
    except Exception as exc:  # noqa: BLE001 -- any load failure becomes a structured issue
        issue = make_issue(
            "summary_load_failed",
            f"Failed to load output {output_label.upper()} summary at {summary_path}: {exc}",
            output=output_label,
            field=_summary_error_field(exc),
            details={"path": str(summary_path), "error_type": exc.__class__.__name__, "error": str(exc)},
        )
        return None, issue


def _summary_error_field(exc: Exception) -> str | None:
    """Surface the offending field name when the failure is a pydantic schema error.

    Returns the first location component of the first error -- pydantic ``loc``
    paths read like ``("videos", "video.mp4", "num_clips_passed")``; the leading
    field name is what shows up in the issue's ``field`` column.
    """
    if not isinstance(exc, ValidationError):
        return None
    errors = exc.errors()
    if not errors:
        return None
    loc = errors[0].get("loc") or ()
    return str(loc[0]) if loc else None


def _stage_sizing(num_rows: int, workers: int, target_batch_size: int) -> tuple[int, int, int]:
    """Resolve ``(pool_size, num_blocks, batch_size)`` from the stage's tuning knobs.

    Block count = ``ceil(num_rows / target_batch_size)``, floored at ``pool_size``
    so every actor gets at least one block, and capped at ``num_rows`` so we
    never overshoot. The realized batch size is then ``ceil(num_rows / num_blocks)``,
    which can be smaller than the target when the floor or cap kicks in -- callers
    log it for visibility.
    """
    pool_size = min(workers, num_rows)
    num_blocks = max(pool_size, math.ceil(num_rows / target_batch_size))
    num_blocks = min(num_rows, num_blocks)
    batch_size = math.ceil(num_rows / num_blocks)
    return pool_size, num_blocks, batch_size


def run_metadata_stage(clips: pa.Table, *, config: SplitComparisonConfig) -> pa.Table:
    """Run Stage 1 as one Ray Data pipeline. Returns the issue table."""
    # Skip filtered clips -- only passed clips have per-clip metadata JSON to compare.
    passed_mask = [kind == "clip" for kind in clips["artifact_kind"].to_pylist()]
    passed_clips = clips.filter(pa.array(passed_mask))
    if passed_clips.num_rows == 0:
        return empty_issues()
    pool_size, num_blocks, batch_size = _stage_sizing(
        passed_clips.num_rows,
        config.metadata_workers,
        config.metadata_batch_size,
    )
    logger.info(
        "metadata stage: clips={} workers={} blocks={} batch_size={} (target={})",
        passed_clips.num_rows,
        pool_size,
        num_blocks,
        batch_size,
        config.metadata_batch_size,
    )
    ds = ray.data.from_arrow(passed_clips, override_num_blocks=num_blocks)
    issues_ds = ds.map_batches(
        MetadataStage,
        batch_format="pyarrow",
        batch_size=batch_size,
        compute=ActorPoolStrategy(size=pool_size),
        num_cpus=config.metadata_cpus_per_worker,
        runtime_env=PixiRuntimeEnv(MetadataStage.conda_env_name),
        fn_constructor_kwargs={
            "output_a": config.output_a,
            "output_b": config.output_b,
            "profile_name": config.profile_name,
            "config": config,
        },
    )
    rows = list(issues_ds.iter_rows())
    return pa.Table.from_pylist(rows, schema=ISSUE_SCHEMA)


def run_video_index_stage(clips: pa.Table, *, config: SplitComparisonConfig) -> pa.Table:
    """Run Stage 2 as one Ray Data pipeline. Returns the issue table."""
    if clips.num_rows == 0:
        return empty_issues()
    # Stage 2 has no per-actor memory ceiling (actors only hold smart_open params),
    # so worker count derives directly from the host -- one knob, no asymmetry.
    workers = max(1, int((os.cpu_count() or 8) / config.video_index_cpus_per_worker))
    pool_size, num_blocks, batch_size = _stage_sizing(
        clips.num_rows,
        workers,
        config.video_index_batch_size,
    )
    logger.info(
        "video_index stage: clips={} workers={} blocks={} batch_size={} (target={})",
        clips.num_rows,
        pool_size,
        num_blocks,
        batch_size,
        config.video_index_batch_size,
    )
    ds = ray.data.from_arrow(clips, override_num_blocks=num_blocks)
    issues_ds = ds.map_batches(
        VideoIndexStage,
        batch_format="pyarrow",
        batch_size=batch_size,
        compute=ActorPoolStrategy(size=pool_size),
        num_cpus=config.video_index_cpus_per_worker,
        runtime_env=PixiRuntimeEnv(VideoIndexStage.conda_env_name),
        fn_constructor_kwargs={
            "output_a": config.output_a,
            "output_b": config.output_b,
            "profile_name": config.profile_name,
            "policy": config.video_index,
        },
    )
    rows = list(issues_ds.iter_rows())
    return pa.Table.from_pylist(rows, schema=ISSUE_SCHEMA)


def _clip_output_counts(clips: pa.Table) -> tuple[int, int, int]:
    """Return ``(in_a_count, in_b_count, in_both_count)`` over the discovered clip table.

    Each row already represents a unique clip (union semantics); these counts answer
    how many of those rows live on which output(s).
    """
    if clips.num_rows == 0:
        return 0, 0, 0
    in_a = clips["in_a"].to_pylist()
    in_b = clips["in_b"].to_pylist()
    return (
        sum(1 for value in in_a if value),
        sum(1 for value in in_b if value),
        sum(1 for a, b in zip(in_a, in_b, strict=True) if a and b),
    )


def _stages_run(config: SplitComparisonConfig, *, ran_clip_work: bool) -> frozenset[str]:
    stages = {"summary"}
    if ran_clip_work:
        stages.add("metadata")
        if config.compare_video_index:
            stages.add("video_index")
    return frozenset(stages)
