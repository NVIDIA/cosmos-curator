# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint ownership and context isolation without executing Ray."""

import pathlib
import subprocess
import sys
import textwrap
import threading
from collections.abc import Iterator
from contextlib import suppress

import pyarrow.fs as pafs
import pytest
import ray
from ray.data.checkpoint import CheckpointConfig

from cosmos_curator.next.core import ray_checkpoints


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
    with ray_checkpoints.installed_checkpoint(id_column="row_id", checkpoint_path=str(tmp_path)):
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

    with ray_checkpoints.installed_checkpoint(
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
        ray_checkpoints.installed_checkpoint(id_column="row_id", checkpoint_path=str(tmp_path / "ours")),
    ):
        pytest.fail("the body must not run when a configuration is already installed")

    # The refusal must not clear the config it declined to replace.
    active = ray.data.DataContext.get_current().checkpoint_config
    assert active is not None
    assert active.id_column == "someone_elses_id"


@pytest.mark.usefixtures("_no_installed_checkpoint")
def test_nested_execution_cannot_replace_checkpoint_owner(tmp_path: pathlib.Path) -> None:
    """Reject nested ownership without clearing the enclosing execution's configuration."""
    context = ray.data.DataContext.get_current()
    with ray_checkpoints.installed_checkpoint(id_column="item_id", checkpoint_path=str(tmp_path)):
        checkpoint = context.checkpoint_config
        with (
            pytest.raises(RuntimeError, match="already installed"),
            ray_checkpoints.installed_checkpoint(id_column="other_id", checkpoint_path=str(tmp_path)),
        ):
            pytest.fail("Nested checkpoint scope must not run")
        assert context.checkpoint_config is checkpoint
    assert context.checkpoint_config is None


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
    monkeypatch.setattr(ray_checkpoints.ray.data.DataContext, "get_current", lambda: context)
    first_entered = threading.Event()
    contender_done = threading.Event()
    outcomes: list[str] = []

    def install(name: str) -> None:
        try:
            with ray_checkpoints.installed_checkpoint(
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

    monkeypatch.setattr(ray_checkpoints.ray.data.DataContext, "get_current", _SwallowsTheAssignment)

    with (
        pytest.raises(RuntimeError, match="not the one just installed"),
        ray_checkpoints.installed_checkpoint(id_column="row_id", checkpoint_path=str(tmp_path / "ours")),
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
    monkeypatch.setattr(ray_checkpoints.ray.data.DataContext, "get_current", lambda: context)

    with (
        pytest.raises(RuntimeError, match="not the one just installed"),
        ray_checkpoints.installed_checkpoint(
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
        with ray_checkpoints.installed_checkpoint(id_column="row_id", checkpoint_path=str(tmp_path)):
            msg = "phase failed"
            raise ValueError(msg)

    with pytest.raises(ValueError, match="phase failed"):
        install_and_fail()

    assert ray.data.DataContext.get_current().checkpoint_config is None


def test_ray_checkpoints_imports_without_recipe_or_engine_dependencies(repo_root: pathlib.Path) -> None:
    """Checkpoint installation works without recipe, workspace, or engine dependencies."""
    probe = textwrap.dedent(
        """
        import sys

        import cosmos_curator.next.core.ray_checkpoints  # noqa: F401

        forbidden = sorted(
            name
            for name in sys.modules
            if name.split(".")[0] in {"vllm", "lance", "transformers"}
            or name.startswith(("cosmos_curator.next.recipes", "cosmos_curator.next.enrichment"))
            or name == "cosmos_curator.next.utils.arrow_utils"
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
