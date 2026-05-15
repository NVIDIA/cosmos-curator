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

"""Tests for the GPU startup helper.

These tests pin the contract that the actor pool's setup-retry loop relies on:
``gpu_stage_startup`` must raise ``GpuNotCleanError`` (a ``RuntimeError`` subclass
so existing broad excepts keep working) when the assigned GPU is still holding
residual memory after the bounded retry budget, and the per-stage cleanliness
bar can be tightened via ``expected_free_fraction`` for callers like vLLM that
demand more headroom than the default 0.9 heuristic.

Everything here is exercised against a mocked ``pynvml`` so the tests run on
CPU-only CI runners.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from cosmos_curator.core.utils.infra import gpu_start_helper
from cosmos_curator.core.utils.infra.gpu_start_helper import (
    GpuNotCleanError,
    _required_free_fraction,
    gpu_stage_startup,
)

# ---------------------------------------------------------------------------
# GpuNotCleanError shape
# ---------------------------------------------------------------------------


def test_gpu_not_clean_error_is_runtime_error_subclass() -> None:
    """``GpuNotCleanError`` must be catchable by code that excepts ``RuntimeError``.

    The actor pool's setup-retry loop catches a broad set of exceptions, and
    several callers further down the stack handle ``RuntimeError`` explicitly.
    Keeping ``GpuNotCleanError`` in that hierarchy preserves existing behavior
    while still letting targeted code branch on the specific type.
    """
    assert issubclass(GpuNotCleanError, RuntimeError)


# ---------------------------------------------------------------------------
# _required_free_fraction
# ---------------------------------------------------------------------------


class TestRequiredFreeFraction:
    """Unit tests for the per-stage cleanliness threshold computation."""

    def test_default_full_gpu_uses_90_percent_heuristic(self) -> None:
        """With no explicit override, a 1-GPU stage needs 0.9 of the device free."""
        assert _required_free_fraction(num_gpus=1.0, expected_free_fraction=None) == pytest.approx(0.9)

    def test_default_scales_with_fractional_gpu_share(self) -> None:
        """A 0.25-GPU stage tolerates 75% of the device being used by neighbors.

        The historical heuristic was ``min(1.0, num_gpus) * 0.9`` so a stage that
        only owns a 0.25 fraction of a GPU should not flag the GPU as dirty just
        because another co-tenant is using its share. Preserving this scaling
        keeps non-vLLM stages from triggering false positives.
        """
        assert _required_free_fraction(num_gpus=0.25, expected_free_fraction=None) == pytest.approx(0.225)

    def test_default_caps_at_one_for_multi_gpu_stages(self) -> None:
        """Multi-GPU stages still cap the heuristic at the 90% single-device bar."""
        assert _required_free_fraction(num_gpus=4.0, expected_free_fraction=None) == pytest.approx(0.9)

    def test_explicit_override_takes_precedence(self) -> None:
        """Explicit overrides (e.g. vLLM's tightened ``gpu_memory_utilization``) win.

        Callers pass this so we fail-fast on residuals the downstream loader
        won't tolerate.
        """
        assert _required_free_fraction(num_gpus=1.0, expected_free_fraction=0.98) == pytest.approx(0.98)

    def test_explicit_override_caps_at_one(self) -> None:
        """A caller passing ``utilization + headroom`` that sums >1.0 must not exceed 1.0.

        Otherwise the cleanliness check could never pass because no GPU is more
        than 100% free.
        """
        assert _required_free_fraction(num_gpus=1.0, expected_free_fraction=1.5) == pytest.approx(1.0)

    def test_explicit_override_ignores_num_gpus(self) -> None:
        """When the caller is explicit, ``num_gpus`` does not silently downscale the bar.

        A 0.25-GPU stage that explicitly asks for 0.98 free must get 0.98, not
        ``0.25 * 0.98``.
        """
        assert _required_free_fraction(num_gpus=0.25, expected_free_fraction=0.98) == pytest.approx(0.98)


# ---------------------------------------------------------------------------
# _dump_gpu_info retry loop (exercised via gpu_stage_startup)
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_pynvml(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Replace pynvml with a MagicMock that defaults to a clean GPU.

    Returns the mock so each test can override ``nvmlDeviceGetMemoryInfo`` to
    simulate residual memory.
    """
    mock = MagicMock()
    handle = object()
    mock.nvmlInit.return_value = None
    mock.nvmlShutdown.return_value = None
    mock.nvmlDeviceGetHandleByIndex.return_value = handle
    mock.nvmlDeviceGetComputeRunningProcesses.return_value = []
    # Default: 184 GiB total, ~0 used -> clean.
    mock.nvmlDeviceGetMemoryInfo.return_value = SimpleNamespace(total=184 * 1024**3, used=0)
    monkeypatch.setattr(gpu_start_helper, "pynvml", mock)
    return mock


@pytest.fixture
def fast_sleep(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Skip the inter-retry sleep so the retry loop completes instantly."""
    sleeper = MagicMock()
    monkeypatch.setattr(gpu_start_helper.time, "sleep", sleeper)
    return sleeper


@pytest.fixture
def visible_devices(monkeypatch: pytest.MonkeyPatch) -> None:
    """Expose a single visible GPU (index 0) so the helper actually probes it."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")


@pytest.mark.usefixtures("visible_devices")
class TestGpuStageStartupRetryLoop:
    """Behavior of the ``check_mem=True`` (i.e. ``pre_setup=True``) path."""

    def test_clean_gpu_passes_on_first_attempt(
        self,
        fake_pynvml: MagicMock,
        fast_sleep: MagicMock,
    ) -> None:
        """A GPU with no residual memory should not trigger any retries or raise."""
        gpu_stage_startup("MyStage", num_gpus=1.0, pre_setup=True)

        assert fake_pynvml.nvmlDeviceGetMemoryInfo.call_count == 1
        assert fast_sleep.call_count == 0

    def test_raises_after_exhausting_retries_when_unclean(
        self,
        fake_pynvml: MagicMock,
        fast_sleep: MagicMock,
    ) -> None:
        """Persistent residual memory must surface as ``GpuNotCleanError``.

        Also pins the retry budget: ``_START_UP_RETRIES + 1`` total probes with
        ``_START_UP_RETRIES`` sleeps between them. This keeps the wall-clock
        budget visible to anyone reviewing the constants — if a future bump
        makes the loop noisier, this assertion catches it.
        """
        # 50% free -> below both the default 0.9 heuristic and any reasonable
        # explicit override.
        total = 184 * 1024**3
        fake_pynvml.nvmlDeviceGetMemoryInfo.return_value = SimpleNamespace(total=total, used=total // 2)

        with pytest.raises(GpuNotCleanError, match="NOT clean to start"):
            gpu_stage_startup("MyStage", num_gpus=1.0, pre_setup=True)

        expected_probes = gpu_start_helper._START_UP_RETRIES + 1
        assert fake_pynvml.nvmlDeviceGetMemoryInfo.call_count == expected_probes
        assert fast_sleep.call_count == gpu_start_helper._START_UP_RETRIES
        # Each sleep should use the configured interval; pinning this stops a
        # silent regression to the legacy ``120s`` value.
        for call in fast_sleep.call_args_list:
            assert call.args == (gpu_start_helper._START_UP_RETRY_INTERVAL_S,)

    def test_eventually_clean_gpu_does_not_raise(
        self,
        fake_pynvml: MagicMock,
        fast_sleep: MagicMock,
    ) -> None:
        """A GPU that comes clean before the budget is exhausted must succeed.

        This is the common case for late-pipeline races where the CUDA driver
        is still reclaiming a freshly-exited process's context.
        """
        total = 184 * 1024**3
        dirty = SimpleNamespace(total=total, used=total // 2)
        clean = SimpleNamespace(total=total, used=0)
        # First two probes show residual memory, then the driver finishes
        # reclaim and subsequent probes are clean.
        fake_pynvml.nvmlDeviceGetMemoryInfo.side_effect = [dirty, dirty, clean]

        gpu_stage_startup("MyStage", num_gpus=1.0, pre_setup=True)

        assert fake_pynvml.nvmlDeviceGetMemoryInfo.call_count == 3
        # Two retries because two probes returned dirty; the third probe was
        # clean so the loop broke before sleeping again.
        assert fast_sleep.call_count == 2

    def test_tight_expected_free_fraction_flags_small_residual(
        self,
        fake_pynvml: MagicMock,
        fast_sleep: MagicMock,
    ) -> None:
        """A ~5% residual is benign at the default bar but trips a 0.98 caller.

        This is the diagnostic the vLLM caption stage relies on: any non-trivial
        residual (>=~3.7 GiB on a 184 GiB device) should force a re-spawn rather
        than letting vLLM's ``gpu_memory_utilization`` check fail with a cryptic
        message a few seconds later.
        """
        total = 184 * 1024**3
        # ~5% used -> 95% free, which clears 0.9 but fails 0.98.
        fake_pynvml.nvmlDeviceGetMemoryInfo.return_value = SimpleNamespace(total=total, used=total * 5 // 100)

        # Default bar tolerates 5% residual.
        gpu_stage_startup("MyStage", num_gpus=1.0, pre_setup=True)

        # Reset between calls so the second invocation's call counts are
        # independent of the first.
        fake_pynvml.reset_mock()
        fast_sleep.reset_mock()

        # 0.98 bar rejects 5% residual.
        with pytest.raises(GpuNotCleanError):
            gpu_stage_startup("MyStage", num_gpus=1.0, pre_setup=True, expected_free_fraction=0.98)

    def test_post_setup_skips_cleanliness_check(
        self,
        fake_pynvml: MagicMock,
        fast_sleep: MagicMock,
    ) -> None:
        """``pre_setup=False`` is a passive dump; it must never raise.

        The helper is reused in ``gpu_stage_cleanup`` to log post-stage GPU
        state. If a residual on shutdown started raising here we would mask
        the original setup or process_data exception with a teardown error.
        """
        total = 184 * 1024**3
        fake_pynvml.nvmlDeviceGetMemoryInfo.return_value = SimpleNamespace(total=total, used=total // 2)

        gpu_stage_startup("MyStage", num_gpus=1.0, pre_setup=False)

        assert fake_pynvml.nvmlDeviceGetMemoryInfo.call_count == 1
        assert fast_sleep.call_count == 0

    def test_missing_cuda_visible_devices_is_a_warn_no_raise(
        self,
        fake_pynvml: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """If ``CUDA_VISIBLE_DEVICES`` is unset, the helper must not crash setup.

        Same rationale as the post-setup path: the cleanliness check is a
        guard, not a load-bearing precondition. Tests on non-GPU runners
        exercise this branch frequently in practice.
        """
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

        gpu_stage_startup("MyStage", num_gpus=1.0, pre_setup=True)

        assert fake_pynvml.nvmlDeviceGetMemoryInfo.call_count == 0
