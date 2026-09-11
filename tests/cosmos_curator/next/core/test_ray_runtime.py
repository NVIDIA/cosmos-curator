# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests that Curator Next owns its Ray Data runtime configuration."""

import pathlib
import subprocess
import sys
import textwrap
import threading
from collections.abc import Iterator
from contextlib import suppress
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pyarrow.fs as pafs
import pytest
import ray
from ray.data.checkpoint import CheckpointConfig

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
    assert "ProbeError" in context.retried_map_errors
    assert context.max_map_retries == 3


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


@pytest.fixture
def _one_live_node(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give the startup snapshot a node list, which a faked Ray has not got.

    ``ensure_ray_initialized`` logs what the cluster advertises before it
    verifies anything, so a test that fakes ``ray.init`` has to fake the node
    list too; otherwise the snapshot queries a GCS this process never started.
    """
    monkeypatch.setattr(
        ray_runtime.ray,
        "nodes",
        lambda: [{"Alive": True, "Resources": {"CPU": 8.0, environment.CURATOR_IO_RESOURCE_NAME: 16.0}}],
    )


@pytest.mark.usefixtures("_one_live_node")
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


@pytest.mark.usefixtures("_one_live_node")
def test_ray_initialization_propagates_runtime_environment_variables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Job environment variables reach workers through Ray's runtime environment."""
    monkeypatch.delenv("RAY_ADDRESS", raising=False)
    monkeypatch.delenv(environment.SLURM_RAY_ENV_VAR_NAME, raising=False)
    monkeypatch.setattr(ray_runtime.ray, "is_initialized", lambda: False)
    ray_init = Mock()
    monkeypatch.setattr(ray_runtime.ray, "init", ray_init)

    ray_runtime.ensure_ray_initialized(
        runtime_env_vars={"AWS_RESPONSE_CHECKSUM_VALIDATION": "when_required"},
    )

    ray_init.assert_called_once_with(
        ignore_reinit_error=True,
        runtime_env={"env_vars": {"AWS_RESPONSE_CHECKSUM_VALIDATION": "when_required"}},
    )


@pytest.mark.usefixtures("_one_live_node")
def test_ray_initialization_omits_an_empty_runtime_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty mapping preserves the existing minimal Ray initialization call."""
    monkeypatch.delenv("RAY_ADDRESS", raising=False)
    monkeypatch.delenv(environment.SLURM_RAY_ENV_VAR_NAME, raising=False)
    monkeypatch.setattr(ray_runtime.ray, "is_initialized", lambda: False)
    ray_init = Mock()
    monkeypatch.setattr(ray_runtime.ray, "init", ray_init)

    ray_runtime.ensure_ray_initialized(runtime_env_vars={})

    ray_init.assert_called_once_with(ignore_reinit_error=True)


@pytest.mark.usefixtures("_one_live_node")
def test_local_ray_initialization_combines_resources_and_runtime_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A locally owned Ray node receives both independent initialization settings."""
    monkeypatch.delenv("RAY_ADDRESS", raising=False)
    monkeypatch.delenv(environment.SLURM_RAY_ENV_VAR_NAME, raising=False)
    monkeypatch.setattr(ray_runtime.ray, "is_initialized", lambda: False)
    ray_init = Mock()
    monkeypatch.setattr(ray_runtime.ray, "init", ray_init)
    monkeypatch.setattr(ray_runtime.ray, "cluster_resources", lambda: {"curator_io": 11.0})

    ray_runtime.ensure_ray_initialized(
        local_resources={"curator_io": 11.0},
        runtime_env_vars={"AWS_RESPONSE_CHECKSUM_VALIDATION": "when_required"},
    )

    ray_init.assert_called_once_with(
        ignore_reinit_error=True,
        resources={"curator_io": 11.0},
        runtime_env={"env_vars": {"AWS_RESPONSE_CHECKSUM_VALIDATION": "when_required"}},
    )


@pytest.mark.usefixtures("_one_live_node")
def test_attached_cluster_must_already_advertise_required_resources(monkeypatch: pytest.MonkeyPatch) -> None:
    """Submitting custom-resource tasks cannot silently hang on an incompatible external cluster."""
    monkeypatch.setattr(ray_runtime.ray, "is_initialized", lambda: True)
    monkeypatch.setattr(ray_runtime.ray, "cluster_resources", lambda: {"CPU": 64.0})

    with pytest.raises(RuntimeError, match="curator_io"):
        ray_runtime.ensure_ray_initialized(local_resources={"curator_io": 16.0})


@pytest.mark.usefixtures("_one_live_node")
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


def test_an_environment_set_io_budget_reports_the_variable_as_its_source(
    monkeypatch: pytest.MonkeyPatch,
    loguru_records: list[dict[str, Any]],
) -> None:
    """A resolved slot count must say whether anyone chose it.

    The variable is set to exactly the built-in default, so the resolved NUMBER
    is identical to the sibling test's. A line reporting only the value would
    satisfy both, which is the ambiguity that prolonged the incident this log
    exists for.
    """
    monkeypatch.setenv(
        environment.CURATOR_IO_SLOTS_PER_NODE_ENV_VAR,
        str(environment.DEFAULT_CURATOR_IO_SLOTS_PER_NODE),
    )

    ray_runtime.curator_io_slots_per_node()

    (message,) = [record["message"] for record in loguru_records]
    assert environment.CURATOR_IO_SLOTS_PER_NODE_ENV_VAR in message
    assert "built-in default" not in message


def test_an_absent_environment_variable_reports_the_built_in_default_as_its_source(
    monkeypatch: pytest.MonkeyPatch,
    loguru_records: list[dict[str, Any]],
) -> None:
    """The other half of the same distinction: nothing was passed, so nothing was chosen."""
    monkeypatch.delenv(environment.CURATOR_IO_SLOTS_PER_NODE_ENV_VAR, raising=False)

    ray_runtime.curator_io_slots_per_node()

    (message,) = [record["message"] for record in loguru_records]
    assert "built-in default" in message


# Two live nodes advertising DIFFERENT counts, so no total below can be recovered
# from a single node, and one dead node advertising more than both, so counting it
# would be visible in every figure.
_HETEROGENEOUS_NODES = [
    {
        "Alive": True,
        "NodeManagerAddress": "10.0.0.1",
        "Resources": {"CPU": 96.0, "GPU": 8.0, environment.CURATOR_IO_RESOURCE_NAME: 16.0},
    },
    {
        "Alive": True,
        "NodeManagerAddress": "10.0.0.2",
        "Resources": {"CPU": 32.0, "GPU": 2.0, environment.CURATOR_IO_RESOURCE_NAME: 4.0},
    },
    {
        "Alive": False,
        "NodeManagerAddress": "10.0.0.3",
        "Resources": {"CPU": 512.0, "GPU": 64.0, environment.CURATOR_IO_RESOURCE_NAME: 99.0},
    },
]


def test_the_startup_snapshot_totals_every_node_and_breaks_them_down(
    monkeypatch: pytest.MonkeyPatch,
    loguru_records: list[dict[str, Any]],
) -> None:
    """A heterogeneous cluster's totals are the reason the per-node breakdown is printed."""
    monkeypatch.setattr(ray_runtime.ray, "is_initialized", lambda: True)
    monkeypatch.setattr(ray_runtime.ray, "nodes", lambda: _HETEROGENEOUS_NODES)

    ray_runtime.ensure_ray_initialized()

    (message,) = [record["message"] for record in loguru_records]
    assert f"2 live node(s) advertising 128 CPU, 10 GPU, 20 {environment.CURATOR_IO_RESOURCE_NAME}" in message
    assert f"node 10.0.0.1: 96 CPU, 8 GPU, 16 {environment.CURATOR_IO_RESOURCE_NAME}" in message
    assert f"node 10.0.0.2: 32 CPU, 2 GPU, 4 {environment.CURATOR_IO_RESOURCE_NAME}" in message


def test_the_startup_snapshot_counts_no_resource_from_a_dead_node(
    monkeypatch: pytest.MonkeyPatch,
    loguru_records: list[dict[str, Any]],
) -> None:
    """A node Ray has lost still appears in ``ray.nodes()``, and its GPUs are gone."""
    monkeypatch.setattr(ray_runtime.ray, "is_initialized", lambda: True)
    monkeypatch.setattr(ray_runtime.ray, "nodes", lambda: _HETEROGENEOUS_NODES)

    ray_runtime.ensure_ray_initialized()

    (message,) = [record["message"] for record in loguru_records]
    assert "10.0.0.3" not in message
    assert "512" not in message


@pytest.fixture
def _no_installed_checkpoint() -> Iterator[None]:
    """Leave the process-global checkpoint config clear on both sides of a test."""
    ray.data.DataContext.get_current().checkpoint_config = None
    yield
    ray.data.DataContext.get_current().checkpoint_config = None


@pytest.mark.usefixtures("_no_installed_checkpoint")
def test_the_installed_checkpoint_keeps_its_progress_record_after_a_successful_execution(
    tmp_path: pathlib.Path,
) -> None:
    """Ray's default deletes the record when the DATASET succeeds, which is before publication."""
    with ray_runtime.installed_checkpoint(id_column="row_id", checkpoint_path=str(tmp_path)):
        active = ray.data.DataContext.get_current().checkpoint_config

        assert active is not None
        assert active.delete_checkpoint_on_success is False
        assert active.id_column == "row_id"
        assert active.checkpoint_path == str(tmp_path)


@pytest.mark.usefixtures("_no_installed_checkpoint")
def test_the_checkpoint_uses_the_configured_filesystem(tmp_path: pathlib.Path) -> None:
    """A run under a non-ambient credential profile must checkpoint through that profile.

    Without this the sink and the checkpoint resolve credentials independently, so
    a run can write results it cannot read back - surfacing as an empty resume
    rather than as an access error.
    """
    filesystem = pafs.LocalFileSystem()

    with ray_runtime.installed_checkpoint(
        id_column="row_id",
        checkpoint_path=str(tmp_path),
        filesystem=filesystem,
    ):
        active = ray.data.DataContext.get_current().checkpoint_config

        assert active is not None
        # Ray takes this as ``override_filesystem`` and stores it as ``filesystem``;
        # the rename is why this is asserted rather than assumed to pass through.
        assert active.filesystem is filesystem


@pytest.mark.usefixtures("_no_installed_checkpoint")
def test_an_already_installed_checkpoint_is_never_replaced(tmp_path: pathlib.Path) -> None:
    """Silently replacing another plan's config would checkpoint it under the wrong id column."""
    ray.data.DataContext.get_current().checkpoint_config = CheckpointConfig(
        id_column="someone_elses_id",
        checkpoint_path=str(tmp_path / "theirs"),
    )

    with (
        pytest.raises(RuntimeError, match="already installed"),
        ray_runtime.installed_checkpoint(id_column="row_id", checkpoint_path=str(tmp_path / "ours")),
    ):
        pytest.fail("the body must not run when a configuration is already installed")

    # The refusal must not clear the config it declined to replace.
    active = ray.data.DataContext.get_current().checkpoint_config
    assert active is not None
    assert active.id_column == "someone_elses_id"


def test_concurrent_checkpoint_installation_has_exactly_one_owner(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    """Concurrent plans cannot both pass the process-global installation guard."""

    class _RacingContext:
        """Expose the former check-then-set window deterministically."""

        def __init__(self) -> None:
            self._checkpoint_config: CheckpointConfig | None = None
            self.null_reads = threading.Barrier(2)
            self.first_null_read = threading.Event()

        @property
        def checkpoint_config(self) -> CheckpointConfig | None:
            if self._checkpoint_config is None:
                self.first_null_read.set()
                with suppress(threading.BrokenBarrierError):
                    self.null_reads.wait(timeout=0.2)
            return self._checkpoint_config

        @checkpoint_config.setter
        def checkpoint_config(self, value: CheckpointConfig | None) -> None:
            self._checkpoint_config = value

    context = _RacingContext()
    monkeypatch.setattr(ray_runtime.ray.data.DataContext, "get_current", lambda: context)
    first_entered = threading.Event()
    contender_done = threading.Event()
    outcomes: list[str] = []

    def install(name: str) -> None:
        try:
            with ray_runtime.installed_checkpoint(
                id_column=name,
                checkpoint_path=str(tmp_path / name),
            ):
                outcomes.append(f"{name}:entered")
                if name == "first":
                    first_entered.set()
                    assert contender_done.wait(timeout=2)
        except RuntimeError:
            outcomes.append(f"{name}:refused")
        finally:
            if name == "second":
                contender_done.set()

    first = threading.Thread(target=install, args=("first",))
    second = threading.Thread(target=install, args=("second",))
    first.start()
    assert context.first_null_read.wait(timeout=2)
    second.start()
    assert first_entered.wait(timeout=2)
    first.join(timeout=2)
    second.join(timeout=2)

    assert not first.is_alive()
    assert not second.is_alive()
    assert sorted(outcomes) == ["first:entered", "second:refused"]
    assert context.checkpoint_config is None


@pytest.mark.usefixtures("_no_installed_checkpoint")
def test_a_checkpoint_that_does_not_become_active_refuses_to_run_the_phase(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    """Installing onto a context the execution does not use is a silent no-op.

    Asserted rather than assumed because the cost of discovering it later is a
    full re-run of a GPU phase that recorded no progress.
    """
    context = ray.data.DataContext.get_current()

    class _SwallowsTheAssignment:
        """A context that reports no config and silently discards the one written to it.

        This is what installing onto a context the execution does not use looks
        like from inside the helper: the guard sees nothing installed, the write
        appears to succeed, and the read back afterwards still finds nothing.
        """

        checkpoint_config: CheckpointConfig | None = None

        def __setattr__(self, name: str, value: object) -> None:
            """Discard every assignment, including ``checkpoint_config``."""

    monkeypatch.setattr(ray_runtime.ray.data.DataContext, "get_current", _SwallowsTheAssignment)

    with (
        pytest.raises(RuntimeError, match="not the one just installed"),
        ray_runtime.installed_checkpoint(id_column="row_id", checkpoint_path=str(tmp_path / "ours")),
    ):
        pytest.fail("the body must not run when the installed configuration is not the active one")

    monkeypatch.undo()
    assert context.checkpoint_config is None


@pytest.mark.usefixtures("_no_installed_checkpoint")
def test_a_checkpoint_with_a_different_filesystem_refuses_to_run_the_phase(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    """Matching path text cannot make a differently credentialed filesystem safe."""
    expected_filesystem = pafs.LocalFileSystem()
    unexpected_filesystem = pafs.LocalFileSystem()

    class _RebindsTheFilesystem:
        """Replace an installed config with matching fields on another filesystem."""

        checkpoint_config: CheckpointConfig | None = None

        def __setattr__(self, name: str, value: object) -> None:
            """Substitute a differently bound config while preserving its path fields."""
            if name == "checkpoint_config" and value is not None:
                assert isinstance(value, CheckpointConfig)
                value = CheckpointConfig(
                    id_column=value.id_column,
                    checkpoint_path=value.checkpoint_path,
                    delete_checkpoint_on_success=value.delete_checkpoint_on_success,
                    override_filesystem=unexpected_filesystem,
                )
            object.__setattr__(self, name, value)

    context = _RebindsTheFilesystem()
    monkeypatch.setattr(ray_runtime.ray.data.DataContext, "get_current", lambda: context)

    with (
        pytest.raises(RuntimeError, match="not the one just installed"),
        ray_runtime.installed_checkpoint(
            id_column="row_id",
            checkpoint_path=str(tmp_path),
            filesystem=expected_filesystem,
        ),
    ):
        pytest.fail("the body must not run when the active filesystem differs")

    assert context.checkpoint_config is None


@pytest.mark.usefixtures("_no_installed_checkpoint")
def test_the_checkpoint_is_cleared_when_the_enclosed_plan_raises(tmp_path: pathlib.Path) -> None:
    """A leaked config would checkpoint the next plan in this process under this id column."""

    def install_and_fail() -> None:
        with ray_runtime.installed_checkpoint(id_column="row_id", checkpoint_path=str(tmp_path)):
            msg = "phase failed"
            raise ValueError(msg)

    with pytest.raises(ValueError, match="phase failed"):
        install_and_fail()

    assert ray.data.DataContext.get_current().checkpoint_config is None


def test_ray_runtime_imports_without_recipe_or_engine_dependencies(repo_root: pathlib.Path) -> None:
    """The module must be usable by any leg, including one with no vLLM or table format.

    ``pyarrow.fs`` is allowed because ``installed_checkpoint`` takes a filesystem
    in its signature; everything else here would make a generic runtime helper
    cost a recipe's dependency tree to import.
    """
    probe = textwrap.dedent(
        """
        import sys

        import cosmos_curator.next.core.ray_runtime  # noqa: F401

        forbidden = sorted(
            name
            for name in sys.modules
            if name.split(".")[0] in {"vllm", "lance", "transformers"}
            or name.startswith("cosmos_curator.next.recipes")
        )
        print(",".join(forbidden))
        """
    )
    completed = subprocess.run(  # noqa: S603 -- fixed probe run under the test interpreter
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=True,
        cwd=repo_root,
    )

    assert completed.stdout.strip() == ""
