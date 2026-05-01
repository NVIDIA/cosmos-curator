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

"""Performance metrics and per-task stats for pipeline stages.

This module is the home for lightweight, always-on performance metrics:

Metrics data model
    ``StagePerfStats`` captures per-task numbers (wall-clock time,
    idle time, RSS before/after, input data size, wall-clock
    timestamps).  ``dump_and_write_perf_stats()`` aggregates these
    across tasks and persists a ``performance_stats.json`` summary.

Per-task stats collection (opt-in, inside each stage)
    ``StageTimer`` is a lightweight helper that stages use inside
    their ``process_data()`` loop via ``reinit()`` / ``log_stats()``
    to collect ``StagePerfStats`` without any external dependency.

For automatic stage instrumentation with profiling backends
(pyinstrument, memray, torch.profiler), see
``cosmos_curator.core.utils.infra.profiling``.
"""

import contextlib
import statistics
import time
from collections.abc import Generator
from typing import Self

import attrs
import pandas as pd
import psutil
from loguru import logger

from cosmos_curator.core.interfaces.stage_interface import CuratorStage
from cosmos_curator.core.utils.infra.tracing import TracedSpan, traced_span
from cosmos_curator.core.utils.misc import summarize
from cosmos_curator.core.utils.storage import storage_utils, writer_utils

# Cache the psutil Process object for the current process to avoid
# repeated /proc lookups on each RSS snapshot.
_CURRENT_PROCESS = psutil.Process()


def _get_rss_mb() -> float:
    """Get the current process RSS (Resident Set Size) in megabytes.

    Uses a cached psutil.Process object to avoid repeated /proc lookups.

    Returns:
        RSS in megabytes.

    """
    return _CURRENT_PROCESS.memory_info().rss / (1024 * 1024)


@attrs.define
class StagePerfStats:
    """Statistics for tracking stage performance metrics.

    Attributes:
        process_time: Total processing time in seconds.
        actor_idle_time: Time the actor spent idle in seconds.
        input_data_size_mb: Size of input data in megabytes.
        rss_before_mb: Process RSS (MB) before process_data() call.
        rss_after_mb: Process RSS (MB) after process_data() call.
        rss_delta_mb: Change in RSS (MB) during process_data() (can be negative).
        wall_start: Absolute wall-clock time (time.time()) when process_data() began.
            Used for Gantt charts and performance analysis across stages.
        wall_end: Absolute wall-clock time (time.time()) when process_data() ended.

    """

    process_time: float = 0.0
    actor_idle_time: float = 0.0
    input_data_size_mb: float = 0.0
    rss_before_mb: float = 0.0
    rss_after_mb: float = 0.0
    rss_delta_mb: float = 0.0
    wall_start: float = 0.0
    wall_end: float = 0.0

    def __add__(self, other: Self) -> Self:
        """Add two StagePerfStats.

        For timing/size fields, values are summed.
        For RSS fields, max is used (peak memory is more informative than sum).
        For wall-clock timestamps, min(start) and max(end) give the overall span.
        """
        return self.__class__(
            process_time=self.process_time + other.process_time,
            actor_idle_time=self.actor_idle_time + other.actor_idle_time,
            input_data_size_mb=self.input_data_size_mb + other.input_data_size_mb,
            rss_before_mb=max(self.rss_before_mb, other.rss_before_mb),
            rss_after_mb=max(self.rss_after_mb, other.rss_after_mb),
            rss_delta_mb=max(self.rss_delta_mb, other.rss_delta_mb),
            # Earliest start and latest end across aggregated tasks.
            # When one side has wall_start=0 (no data), use the other side's value.
            wall_start=(
                min(self.wall_start, other.wall_start)
                if self.wall_start and other.wall_start
                else self.wall_start or other.wall_start
            ),
            wall_end=max(self.wall_end, other.wall_end),
        )

    def __radd__(self, other: int | Self) -> Self:
        """Add two StagePerfStats together, if right is 0, returns itself."""
        if other == 0:
            return self
        assert isinstance(other, StagePerfStats)
        return self.__add__(other)

    def reset(self) -> None:
        """Reset the stats."""
        self.process_time = 0.0
        self.actor_idle_time = 0.0
        self.input_data_size_mb = 0.0
        self.rss_before_mb = 0.0
        self.rss_after_mb = 0.0
        self.rss_delta_mb = 0.0
        self.wall_start = 0.0
        self.wall_end = 0.0

    def to_dict(self) -> dict[str, float]:
        """Convert the stats to a dictionary."""
        return attrs.asdict(self)


def _summarize_perf_stats(
    task_stats: list[dict[str, StagePerfStats]],
) -> dict[str, dict[str, float]]:
    """Aggregate per-task stage stats into per-stage summaries."""
    data = {}
    # Collect stage names from ALL tasks (not just the first) so stages
    # that only ran for a subset of tasks are still included in the summary.
    all_stages: set[str] = set()
    for ts in task_stats:
        all_stages.update(ts)
    for stage in sorted(all_stages):
        # Use .get() to default to zero-valued stats for tasks where
        # the stage has no recorded data (e.g. stage failed before
        # logging perf stats).
        data[stage] = sum(
            (x.get(stage, StagePerfStats()) for x in task_stats),
            StagePerfStats(),
        ).to_dict()
    return data


def dump_and_write_perf_stats(
    task_stats: list[dict[str, StagePerfStats]],
    output_path: str | None,
    output_s3_profile_name: str,
) -> None:
    """Dump and write performance stats.

    Args:
        task_stats: List of task stats.
        output_path: Output path.
        output_s3_profile_name: Output S3 profile name.

    """
    if len(task_stats) == 0:
        return
    data = _summarize_perf_stats(task_stats)
    with summarize.turn_off_pandas_display_limits():
        pdf = pd.DataFrame.from_dict(data, orient="index")
        logger.info(f"Per-stage performance stats:\n{pdf!s}")
    if output_path is not None:
        client = storage_utils.get_storage_client(target_path=output_path, profile_name=output_s3_profile_name)
        dest = storage_utils.get_next_file("performance_stats", "json", output_path, client)
        writer_utils.write_json(
            data,
            dest,
            "performance stats",
            "pipeline",
            verbose=True,
            client=client,
        )


class StageTimer:
    """Per-task stats tracker for pipeline stages.

    Tracks things at a per "process_data" call level.
    """

    def __init__(self, stage: CuratorStage) -> None:
        """Initialize the stage timer.

        Args:
            stage: The stage to track.

        """
        self._stage_name = str(stage.__class__.__name__)
        self._reset()
        self._last_active_time = time.time()
        self._initialized = False

    def _reset(self) -> None:
        self._num_gpus = 0.0
        self._num_cpus = 0.0
        self._num_samples = 0
        self._durations_s: list[float] = []
        self._source_video_duration_s = 0.0
        self._input_data_size_b = 0
        self._start = 0.0
        self._idle_time_s = 0.0
        self._startup_time_s = 0.0
        self._rss_before_mb = 0.0

    def reinit(self, stage: CuratorStage, stage_input_size: int = 1) -> None:
        """Reinitialize the stage timer.

        Captures an RSS snapshot before processing begins so we can
        compute the memory delta in ``log_stats()``.

        Also emits an OTel ``batch_start`` event on the current span
        (if tracing is active) with the input size and initial RSS.

        Args:
            stage: The stage to reinitialize the timer for.
            stage_input_size: The size of the stage input.

        """
        self._reset()
        self._num_gpus = stage.resources.gpus
        self._num_cpus = stage.resources.cpus
        self._input_data_size_b = stage_input_size
        # Snapshot RSS before process_data() work begins.
        self._rss_before_mb = _get_rss_mb()
        self._start = time.time()
        if self._initialized:
            self._idle_time_s = self._start - self._last_active_time
        else:
            self._startup_time_s = self._start - self._last_active_time
        self._initialized = True

        # OTel: mark the start of a new batch on the current span.
        TracedSpan.current().add_event(
            "batch_start",
            attributes={
                "stage.name": self._stage_name,
                "stage.input_data_size_b": stage_input_size,
                "stage.rss_before_mb": round(self._rss_before_mb, 1),
                "stage.idle_time_s": round(self._idle_time_s, 3),
            },
        )

    @contextlib.contextmanager
    def time_process(self, num_samples: int = 1, source_video_duration_s: float = 0) -> Generator[None, None, None]:
        """Time the process of the stage.

        When OTel tracing is active, creates a child span
        ``"{stage_name}.sample"`` that captures per-sample processing
        duration and attributes.

        Args:
            num_samples: The number of samples to process.
            source_video_duration_s: The duration of the source video.

        """
        start_time = time.time()
        with traced_span(
            f"{self._stage_name}.sample",
            attributes={
                "stage.name": self._stage_name,
                "stage.num_samples": num_samples,
                "stage.source_video_duration_s": source_video_duration_s,
            },
        ):
            yield
        end_time = time.time()
        duration = end_time - start_time
        self._num_samples += num_samples
        self._source_video_duration_s += source_video_duration_s
        for _ in range(num_samples):
            self._durations_s.append(duration / num_samples)

    def log_stats(self, *, verbose: bool = False) -> tuple[str, StagePerfStats]:
        """Log the stats of the stage.

        Captures an RSS snapshot after processing so we can report
        rss_before_mb, rss_after_mb, and rss_delta_mb.

        Args:
            verbose: Whether to log the stats.

        Returns:
            A tuple of the stage name and the stage performance stats.

        """
        num_gpus = self._num_gpus
        num_cpus = self._num_cpus
        end = time.time()
        process_data_dur_s = end - self._start
        num_samples = self._num_samples
        avg_dur_s = statistics.mean(self._durations_s) if self._durations_s else 0
        source_video_len_s = self._source_video_duration_s
        input_data_size_mb = self._input_data_size_b / 1024 / 1024
        start_time_s = self._startup_time_s
        idle_time_s = self._idle_time_s

        # Snapshot RSS after process_data() work completes.
        rss_before_mb = self._rss_before_mb
        rss_after_mb = _get_rss_mb()
        rss_delta_mb = rss_after_mb - rss_before_mb

        if verbose:
            logger.info(
                f"Stats: {process_data_dur_s=:.3f} - {num_samples=} - {avg_dur_s=:.3f} - "
                f"{num_gpus=} - {num_cpus=} - {start_time_s=:.3f} - {idle_time_s=:.3f} - "
                f"{source_video_len_s=:.1f} - {input_data_size_mb=:.3f} - "
                f"{rss_before_mb=:.1f} - {rss_after_mb=:.1f} - {rss_delta_mb=:+.1f}",
            )

        # OTel: annotate the current span (e.g. process_data lifecycle
        # span from _ProfiledStage) with aggregate per-call stats.
        TracedSpan.current().set_attributes(
            {
                "stage.name": self._stage_name,
                "stage.process_time_s": round(process_data_dur_s, 3),
                "stage.idle_time_s": round(idle_time_s, 3),
                "stage.num_samples": num_samples,
                "stage.avg_sample_duration_s": round(avg_dur_s, 3),
                "stage.input_data_size_mb": round(input_data_size_mb, 3),
                "stage.source_video_duration_s": round(source_video_len_s, 1),
                "stage.rss_before_mb": round(rss_before_mb, 1),
                "stage.rss_after_mb": round(rss_after_mb, 1),
                "stage.rss_delta_mb": round(rss_delta_mb, 1),
                "stage.num_gpus": num_gpus,
                "stage.num_cpus": num_cpus,
            }
        )

        self._last_active_time = time.time()

        stage_perf_stats = StagePerfStats(
            process_time=process_data_dur_s,
            actor_idle_time=idle_time_s,
            input_data_size_mb=input_data_size_mb,
            rss_before_mb=rss_before_mb,
            rss_after_mb=rss_after_mb,
            rss_delta_mb=rss_delta_mb,
            # Absolute wall-clock timestamps for distributed trace / Gantt chart.
            wall_start=self._start,
            wall_end=end,
        )
        return self._stage_name, stage_perf_stats
