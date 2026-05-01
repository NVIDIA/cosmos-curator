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

"""Tests for performance_utils: StagePerfStats and StageTimer.

Covers the new fields added in Phase 1.1 (rss_before_mb, rss_after_mb,
rss_delta_mb) and Phase 1.2 (wall_start, wall_end), as well as the
original aggregation, reset, and serialization behavior.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import attrs
import pytest

from cosmos_curator.core.utils.infra.performance_utils import (
    StagePerfStats,
    StageTimer,
    _get_rss_mb,
    _summarize_perf_stats,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_stage() -> MagicMock:
    """Create a mock CuratorStage with minimal resource attributes.

    Returns a MagicMock that satisfies StageTimer's __init__ and reinit()
    requirements (needs __class__.__name__, resources.gpus, resources.cpus).
    """
    stage = MagicMock()
    stage.__class__ = type("FakeStage", (), {})
    stage.resources.gpus = 0.5
    stage.resources.cpus = 2.0
    return stage


# ---------------------------------------------------------------------------
# StagePerfStats -- defaults and backward compatibility
# ---------------------------------------------------------------------------


class TestStagePerfStatsDefaults:
    """Verify all fields default to 0.0 for backward compatibility."""

    def test_all_fields_default_to_zero(self) -> None:
        """Verify default-constructed StagePerfStats has all fields at 0.0."""
        stats = StagePerfStats()
        assert stats.process_time == 0.0
        assert stats.actor_idle_time == 0.0
        assert stats.input_data_size_mb == 0.0
        assert stats.rss_before_mb == 0.0
        assert stats.rss_after_mb == 0.0
        assert stats.rss_delta_mb == 0.0
        assert stats.wall_start == 0.0
        assert stats.wall_end == 0.0

    def test_old_style_construction_ignores_new_fields(self) -> None:
        """Existing callers that only pass original fields still work."""
        stats = StagePerfStats(process_time=1.5, actor_idle_time=0.3, input_data_size_mb=4.0)
        assert stats.process_time == 1.5
        assert stats.rss_before_mb == 0.0
        assert stats.wall_start == 0.0


# ---------------------------------------------------------------------------
# StagePerfStats.__add__ -- aggregation behavior
# ---------------------------------------------------------------------------


class TestStagePerfStatsAdd:
    """Verify __add__ sums timing fields, takes max for RSS, min/max for wall timestamps."""

    def test_timing_fields_are_summed(self) -> None:
        """Verify process_time, actor_idle_time, input_data_size_mb are summed."""
        s1 = StagePerfStats(process_time=1.0, actor_idle_time=0.5, input_data_size_mb=2.0)
        s2 = StagePerfStats(process_time=3.0, actor_idle_time=1.0, input_data_size_mb=5.0)
        result = s1 + s2
        assert result.process_time == pytest.approx(4.0)
        assert result.actor_idle_time == pytest.approx(1.5)
        assert result.input_data_size_mb == pytest.approx(7.0)

    def test_rss_fields_use_max(self) -> None:
        """Verify rss_before_mb, rss_after_mb, rss_delta_mb use max aggregation."""
        s1 = StagePerfStats(rss_before_mb=100.0, rss_after_mb=150.0, rss_delta_mb=50.0)
        s2 = StagePerfStats(rss_before_mb=120.0, rss_after_mb=200.0, rss_delta_mb=80.0)
        result = s1 + s2
        assert result.rss_before_mb == pytest.approx(120.0)
        assert result.rss_after_mb == pytest.approx(200.0)
        assert result.rss_delta_mb == pytest.approx(80.0)

    def test_wall_timestamps_min_start_max_end(self) -> None:
        """Verify wall_start uses min and wall_end uses max across operands."""
        now = time.time()
        s1 = StagePerfStats(wall_start=now, wall_end=now + 1.0)
        s2 = StagePerfStats(wall_start=now - 0.5, wall_end=now + 2.0)
        result = s1 + s2
        assert result.wall_start == pytest.approx(now - 0.5)
        assert result.wall_end == pytest.approx(now + 2.0)

    def test_wall_start_skips_zero(self) -> None:
        """When one side has wall_start=0 (no data), use the other's value."""
        now = time.time()
        s1 = StagePerfStats(wall_start=now, wall_end=now + 1.0)
        s_empty = StagePerfStats()  # wall_start=0
        result = s1 + s_empty
        assert result.wall_start == pytest.approx(now)
        assert result.wall_end == pytest.approx(now + 1.0)

    def test_wall_start_both_zero_stays_zero(self) -> None:
        """When both sides have wall_start=0, result is 0."""
        result = StagePerfStats() + StagePerfStats()
        assert result.wall_start == 0.0
        assert result.wall_end == 0.0


# ---------------------------------------------------------------------------
# StagePerfStats.__radd__ -- sum() compatibility
# ---------------------------------------------------------------------------


class TestStagePerfStatsRadd:
    """Verify __radd__ enables sum([...], StagePerfStats()) pattern."""

    def test_radd_with_zero_returns_self(self) -> None:
        """Verify 0 + StagePerfStats returns the original instance."""
        s = StagePerfStats(process_time=5.0, rss_after_mb=100.0)
        result = 0 + s
        assert result is s

    def test_sum_aggregates_correctly(self) -> None:
        """Verify sum([s1, s2, s3], StagePerfStats()) aggregates all fields correctly."""
        now = time.time()
        stats = [
            StagePerfStats(process_time=1.0, rss_after_mb=100.0, wall_start=now, wall_end=now + 1.0),
            StagePerfStats(process_time=2.0, rss_after_mb=200.0, wall_start=now - 1.0, wall_end=now + 3.0),
            StagePerfStats(process_time=0.5, rss_after_mb=150.0, wall_start=now + 0.5, wall_end=now + 2.0),
        ]
        result = sum(stats, StagePerfStats())
        assert result.process_time == pytest.approx(3.5)
        assert result.rss_after_mb == pytest.approx(200.0)
        assert result.wall_start == pytest.approx(now - 1.0)
        assert result.wall_end == pytest.approx(now + 3.0)


# ---------------------------------------------------------------------------
# StagePerfStats.reset
# ---------------------------------------------------------------------------


class TestStagePerfStatsReset:
    """Verify reset() zeros all fields."""

    def test_reset_clears_all_fields(self) -> None:
        """Verify reset() sets every field back to 0.0."""
        stats = StagePerfStats(
            process_time=5.0,
            actor_idle_time=1.0,
            input_data_size_mb=10.0,
            rss_before_mb=100.0,
            rss_after_mb=200.0,
            rss_delta_mb=100.0,
            wall_start=1000.0,
            wall_end=2000.0,
        )
        stats.reset()
        for field in attrs.fields(StagePerfStats):
            assert getattr(stats, field.name) == 0.0, f"{field.name} was not reset to 0.0"


# ---------------------------------------------------------------------------
# StagePerfStats.to_dict
# ---------------------------------------------------------------------------


class TestStagePerfStatsToDict:
    """Verify to_dict() returns all fields as a JSON-serializable dict."""

    def test_contains_all_field_names(self) -> None:
        """Verify to_dict() returns all 8 expected field names."""
        d = StagePerfStats().to_dict()
        expected_keys = {
            "process_time",
            "actor_idle_time",
            "input_data_size_mb",
            "rss_before_mb",
            "rss_after_mb",
            "rss_delta_mb",
            "wall_start",
            "wall_end",
        }
        assert set(d.keys()) == expected_keys

    def test_values_are_floats(self) -> None:
        """Verify all dict values are float for JSON serialization."""
        d = StagePerfStats(process_time=1.5, rss_after_mb=42.0, wall_start=1000.0).to_dict()
        for key, value in d.items():
            assert isinstance(value, float), f"{key} is {type(value)}, expected float"

    def test_round_trip_preserves_values(self) -> None:
        """Verify to_dict() preserves field values for round-trip checks."""
        original = StagePerfStats(
            process_time=3.14,
            rss_delta_mb=-5.0,
            wall_start=1700000000.0,
            wall_end=1700000001.5,
        )
        d = original.to_dict()
        assert d["process_time"] == pytest.approx(3.14)
        assert d["rss_delta_mb"] == pytest.approx(-5.0)
        assert d["wall_start"] == pytest.approx(1700000000.0)
        assert d["wall_end"] == pytest.approx(1700000001.5)


# ---------------------------------------------------------------------------
# _get_rss_mb
# ---------------------------------------------------------------------------


class TestGetRssMb:
    """Verify _get_rss_mb returns a sensible positive value."""

    def test_returns_positive_float(self) -> None:
        """Verify _get_rss_mb returns a positive float for a live process."""
        rss = _get_rss_mb()
        assert isinstance(rss, float)
        assert rss > 0.0, "RSS should be positive for a running Python process"


# ---------------------------------------------------------------------------
# StageTimer -- RSS and wall timestamp population
# ---------------------------------------------------------------------------


class TestStageTimerLogStats:
    """Verify StageTimer.log_stats() populates RSS and wall-clock fields."""

    def test_log_stats_populates_rss_fields(self, mock_stage: MagicMock) -> None:
        """After reinit + log_stats, RSS fields should be non-zero."""
        timer = StageTimer(mock_stage)
        timer.reinit(mock_stage, stage_input_size=1024)

        # Simulate some processing via time_process context manager.
        with timer.time_process(num_samples=1):
            pass

        _name, stats = timer.log_stats()
        assert stats.rss_before_mb > 0.0
        assert stats.rss_after_mb > 0.0
        # Delta can be zero or small, but the fields must be populated.
        assert isinstance(stats.rss_delta_mb, float)

    def test_log_stats_populates_wall_timestamps(self, mock_stage: MagicMock) -> None:
        """After reinit + log_stats, wall_start < wall_end."""
        timer = StageTimer(mock_stage)
        timer.reinit(mock_stage, stage_input_size=512)

        with timer.time_process(num_samples=1):
            pass

        _name, stats = timer.log_stats()
        assert stats.wall_start > 0.0
        assert stats.wall_end > 0.0
        assert stats.wall_start <= stats.wall_end

    def test_log_stats_returns_stage_name(self, mock_stage: MagicMock) -> None:
        """Stage name should match the mock stage's class name."""
        timer = StageTimer(mock_stage)
        timer.reinit(mock_stage)

        with timer.time_process():
            pass

        name, _stats = timer.log_stats()
        assert name == "FakeStage"

    def test_log_stats_process_time_positive(self, mock_stage: MagicMock) -> None:
        """process_time should reflect actual elapsed time."""
        timer = StageTimer(mock_stage)
        timer.reinit(mock_stage)

        with timer.time_process():
            time.sleep(0.01)  # 10ms

        _name, stats = timer.log_stats()
        assert stats.process_time > 0.0

    def test_log_stats_input_data_size(self, mock_stage: MagicMock) -> None:
        """input_data_size_mb should reflect the stage_input_size passed to reinit."""
        timer = StageTimer(mock_stage)
        # 10 MB = 10 * 1024 * 1024 bytes
        timer.reinit(mock_stage, stage_input_size=10 * 1024 * 1024)

        with timer.time_process():
            pass

        _name, stats = timer.log_stats()
        assert stats.input_data_size_mb == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# StageTimer.reinit -- state reset
# ---------------------------------------------------------------------------


class TestStageTimerReinit:
    """Verify reinit() resets internal counters between process_data calls."""

    def test_reinit_resets_durations(self, mock_stage: MagicMock) -> None:
        """After reinit, previous timing data should be cleared."""
        timer = StageTimer(mock_stage)

        # First invocation
        timer.reinit(mock_stage, stage_input_size=100)
        with timer.time_process(num_samples=5):
            pass
        _name, stats1 = timer.log_stats()
        assert stats1.process_time > 0

        # Second invocation -- should not accumulate from first
        timer.reinit(mock_stage, stage_input_size=200)
        with timer.time_process(num_samples=1):
            pass
        _name, stats2 = timer.log_stats()

        # input_data_size_mb should reflect the SECOND reinit, not the first
        assert stats2.input_data_size_mb == pytest.approx(200.0 / (1024 * 1024))

    def test_second_reinit_records_idle_time(self, mock_stage: MagicMock) -> None:
        """After the first log_stats, a second reinit should record actor_idle_time."""
        timer = StageTimer(mock_stage)

        # First call
        timer.reinit(mock_stage)
        with timer.time_process():
            pass
        timer.log_stats()

        # Small delay between calls to create measurable idle time
        time.sleep(0.02)

        # Second call
        timer.reinit(mock_stage)
        with timer.time_process():
            pass
        _name, stats = timer.log_stats()

        # idle_time should be > 0 because we slept between log_stats and reinit
        assert stats.actor_idle_time > 0.0


class TestSummarizePerfStatsMissingStages:
    """Verify _summarize_perf_stats handles tasks with different sets of stage keys."""

    def test_missing_stage_in_later_task(self) -> None:
        """Task 0 has stages A and B, task 1 only has A. Should not crash."""
        task_stats = [
            {
                "StageA": StagePerfStats(process_time=1.0),
                "StageB": StagePerfStats(process_time=2.0),
            },
            {
                "StageA": StagePerfStats(process_time=3.0),
                # StageB is missing -- e.g. the stage failed before recording stats.
            },
        ]
        result = _summarize_perf_stats(task_stats)

        assert "StageA" in result
        assert "StageB" in result
        assert result["StageA"]["process_time"] == pytest.approx(4.0)  # both tasks
        assert result["StageB"]["process_time"] == pytest.approx(2.0)  # task 0 only

    def test_stage_only_in_later_task(self) -> None:
        """Task 0 has stage A, task 1 has stages A and B. B must appear in summary."""
        task_stats = [
            {
                "StageA": StagePerfStats(process_time=1.0),
            },
            {
                "StageA": StagePerfStats(process_time=2.0),
                "StageB": StagePerfStats(process_time=5.0),
            },
        ]
        result = _summarize_perf_stats(task_stats)

        assert "StageA" in result
        assert "StageB" in result
        assert result["StageA"]["process_time"] == pytest.approx(3.0)  # both tasks
        assert result["StageB"]["process_time"] == pytest.approx(5.0)  # task 1 only

    def test_empty_task_stats(self) -> None:
        """Empty list should return empty dict without crashing."""
        result = _summarize_perf_stats([])
        assert result == {}

    def test_all_tasks_have_all_stages(self) -> None:
        """When all tasks have all stages, behavior is unchanged from the original."""
        task_stats = [
            {"X": StagePerfStats(process_time=1.0), "Y": StagePerfStats(process_time=2.0)},
            {"X": StagePerfStats(process_time=3.0), "Y": StagePerfStats(process_time=4.0)},
        ]
        result = _summarize_perf_stats(task_stats)

        assert result["X"]["process_time"] == pytest.approx(4.0)
        assert result["Y"]["process_time"] == pytest.approx(6.0)
