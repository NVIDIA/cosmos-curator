# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests that Curator Next owns its Ray Data runtime configuration."""

import pathlib
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from cosmos_curator.core.utils import environment
from cosmos_curator.next.core import ray_runtime


def test_stability_configuration_preserves_high_memory_detector_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Shared stability settings do not suppress memory warnings for every recipe."""
    high_memory_detector_config = SimpleNamespace(detection_time_interval_s=30)
    context = SimpleNamespace(
        default_map_logical_memory_enabled=False,
        retried_map_errors=[],
        max_map_retries=0,
        issue_detectors_config=SimpleNamespace(high_memory_detector_config=high_memory_detector_config),
    )
    monkeypatch.setattr(ray_runtime.ray.data.DataContext, "get_current", lambda: context)

    ray_runtime.configure_ray_data_stability()

    assert high_memory_detector_config.detection_time_interval_s == 30


def test_stability_configuration_can_disable_expected_video_memory_warnings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pipelines with intentional large payloads can suppress reservation warnings."""
    high_memory_detector_config = SimpleNamespace(detection_time_interval_s=30)
    context = SimpleNamespace(
        default_map_logical_memory_enabled=False,
        retried_map_errors=[],
        max_map_retries=0,
        issue_detectors_config=SimpleNamespace(high_memory_detector_config=high_memory_detector_config),
    )
    monkeypatch.setattr(ray_runtime.ray.data.DataContext, "get_current", lambda: context)

    ray_runtime.configure_ray_data_stability(disable_high_memory_detector=True)

    assert high_memory_detector_config.detection_time_interval_s == -1


def test_stability_configuration_keeps_rays_no_progress_window_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A recipe with short tasks keeps deadlock detection at Ray's own window."""
    context = SimpleNamespace(
        default_map_logical_memory_enabled=False,
        retried_map_errors=[],
        max_map_retries=0,
        execution_no_progress_timeout_s=1800,
        issue_detectors_config=SimpleNamespace(high_memory_detector_config=None),
    )
    monkeypatch.setattr(ray_runtime.ray.data.DataContext, "get_current", lambda: context)

    ray_runtime.configure_ray_data_stability()

    assert context.execution_no_progress_timeout_s == 1800


def test_stability_configuration_can_widen_the_no_progress_window(monkeypatch: pytest.MonkeyPatch) -> None:
    """A legitimately slow task must not be mistaken for a wedged one."""
    context = SimpleNamespace(
        default_map_logical_memory_enabled=False,
        retried_map_errors=[],
        max_map_retries=0,
        execution_no_progress_timeout_s=1800,
        issue_detectors_config=SimpleNamespace(high_memory_detector_config=None),
    )
    monkeypatch.setattr(ray_runtime.ray.data.DataContext, "get_current", lambda: context)

    ray_runtime.configure_ray_data_stability(no_progress_timeout_s=7200)

    assert context.execution_no_progress_timeout_s == 7200


def test_stability_configuration_tolerates_a_ray_without_the_no_progress_knob(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Widening the window on a Ray build that lacks it is a no-op, not an error."""
    context = SimpleNamespace(
        default_map_logical_memory_enabled=False,
        retried_map_errors=[],
        max_map_retries=0,
        issue_detectors_config=SimpleNamespace(high_memory_detector_config=None),
    )
    monkeypatch.setattr(ray_runtime.ray.data.DataContext, "get_current", lambda: context)

    ray_runtime.configure_ray_data_stability(no_progress_timeout_s=7200)

    assert not hasattr(context, "execution_no_progress_timeout_s")


def test_eager_actor_autoscaling_removes_conservative_growth_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    """Model-loading actor pools can grow geometrically while resources and their own ceiling still cap them."""
    autoscaling_config = SimpleNamespace(
        actor_pool_util_upscaling_threshold=1.75,
        actor_pool_max_upscaling_delta=1,
    )
    context = SimpleNamespace(autoscaling_config=autoscaling_config)
    monkeypatch.setattr(ray_runtime.ray.data.DataContext, "get_current", lambda: context)

    ray_runtime.configure_ray_data_eager_actor_autoscaling()

    assert autoscaling_config.actor_pool_util_upscaling_threshold == 1.0
    assert autoscaling_config.actor_pool_max_upscaling_delta is None


def test_recipes_do_not_reach_into_the_deprecated_ray_data_package(repo_root: pathlib.Path) -> None:
    """``next`` must stay importable once ``pipelines.ray_data`` is deleted."""
    probe = (
        "import sys;"
        "import cosmos_curator.next.recipes.video_split.pipeline;"
        "print([m for m in sys.modules if m.startswith('cosmos_curator.pipelines')])"
    )
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", probe],
        check=True,
        capture_output=True,
        text=True,
        cwd=repo_root,
    )

    assert result.stdout.strip() == "[]"


def test_local_ray_initialization_advertises_required_resources(monkeypatch: pytest.MonkeyPatch) -> None:
    """A recipe-owned local node receives the same logical IO capacity as cluster workers."""
    monkeypatch.delenv("RAY_ADDRESS", raising=False)
    monkeypatch.delenv(environment.SLURM_RAY_ENV_VAR_NAME, raising=False)
    monkeypatch.setattr(ray_runtime.ray, "is_initialized", lambda: False)
    ray_init = Mock()
    monkeypatch.setattr(ray_runtime.ray, "init", ray_init)
    monkeypatch.setattr(ray_runtime.ray, "cluster_resources", lambda: {"curator_io": 7.0})

    ray_runtime.ensure_ray_initialized(local_resources={"curator_io": 7.0})

    ray_init.assert_called_once_with(ignore_reinit_error=True, resources={"curator_io": 7.0})


def test_attached_cluster_must_already_advertise_required_resources(monkeypatch: pytest.MonkeyPatch) -> None:
    """Submitting custom-resource tasks cannot silently hang on an incompatible external cluster."""
    monkeypatch.setattr(ray_runtime.ray, "is_initialized", lambda: True)
    monkeypatch.setattr(ray_runtime.ray, "cluster_resources", lambda: {"CPU": 64.0})

    with pytest.raises(RuntimeError, match="curator_io"):
        ray_runtime.ensure_ray_initialized(local_resources={"curator_io": 16.0})


def test_slurm_driver_connects_without_trying_to_mutate_node_resources(monkeypatch: pytest.MonkeyPatch) -> None:
    """Slurm launchers own node resources before the recipe connects to their cluster."""
    monkeypatch.setenv(environment.SLURM_RAY_ENV_VAR_NAME, "True")
    monkeypatch.setattr(ray_runtime.ray, "is_initialized", lambda: False)
    ray_init = Mock()
    monkeypatch.setattr(ray_runtime.ray, "init", ray_init)
    monkeypatch.setattr(ray_runtime.ray, "cluster_resources", lambda: {"curator_io": 16.0})

    ray_runtime.ensure_ray_initialized(local_resources={"curator_io": 16.0})

    ray_init.assert_called_once_with(ignore_reinit_error=True)


@pytest.mark.parametrize("value", ["0", "-1", "many"])
def test_invalid_local_io_slot_capacity_is_rejected(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    """Bad environment overrides fail before Ray starts with an unusable resource."""
    monkeypatch.setenv(environment.CURATOR_IO_SLOTS_PER_NODE_ENV_VAR, value)

    with pytest.raises(ValueError, match=environment.CURATOR_IO_SLOTS_PER_NODE_ENV_VAR):
        ray_runtime.curator_io_slots_per_node()
