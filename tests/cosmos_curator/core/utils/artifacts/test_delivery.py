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

"""Tests for the artifact delivery orchestrator (``delivery.py``).

Exercises the driver-side ``ArtifactDelivery`` lifecycle: staging
env-var setup, idempotent collection guards, and upload destination
routing.

::

    What we test                          How we test it
    +----------------------------------+  +--------------------------------------+
    | create() env-var setup           |  | Call create(), assert env var set    |
    |   -> real temp dir on disk       |  |   and points to existing directory   |
    | create() idempotency             |  | Two create() calls -> same env var  |
    | create() hook registration       |  | Mock register_pre_shutdown_hook     |
    |   -> callable registered         |  |   -> assert_called_once             |
    | collect() idempotency            |  | Set _collected=True -> returns 0    |
    | collect() empty output_dir       |  | output_dir="" -> returns 0          |
    | Upload dest routing              |  | output_dir + upload_subdir stored   |
    |   -> correct path composition    |  |   correctly for collect() to use    |
    | Remote partial success (lenient) |  | Mock 2 nodes, 1 fails -> partial    |
    |   -> count from ok nodes         |  |   count returned, _collected=True   |
    | Remote failure (strict)          |  | Mock node failure -> raises         |
    |   -> ArtifactDeliveryError       |  |   ArtifactDeliveryError             |
    | Remote timeout (lenient/strict)  |  | ray.wait returns empty -> 0/raise   |
    | Local partial failure (lenient)  |  | Transport returns nodes_failed ->   |
    |   -> total_files preserved       |  |   total_files returned              |
    | Local node failure (strict)      |  | Transport returns nodes_failed ->   |
    |   -> ArtifactDeliveryError       |  |   raises ArtifactDeliveryError      |
    +----------------------------------+  +--------------------------------------+

Test setup:
    ``ArtifactDelivery.create()`` calls
    ``ray_cluster_utils.register_pre_shutdown_hook()`` which requires
    a live Ray cluster.  All tests mock this single function via
    ``unittest.mock.patch`` on the ``ray_cluster_utils`` module.

    The ``_clean_staging_env`` autouse fixture clears the
    ``COSMOS_CURATOR_ARTIFACTS_STAGING_DIR`` env var before each test
    and removes any staging directories after the test completes.

    All paths use pytest's ``tmp_path`` fixture.

    No Ray cluster, no network, no GPU required.
"""

import os
import pathlib
import shutil
from collections.abc import Generator
from unittest.mock import MagicMock, patch

import pytest

from cosmos_curator.core.utils.artifacts.collector import CollectResult
from cosmos_curator.core.utils.artifacts.delivery import (
    ArtifactDelivery,
    ArtifactDeliveryError,
    _NodeUploadResult,
)

# delivery.py accesses the hook via the ray_cluster_utils module object,
# so the mock target is the function on that module (not on delivery).
_MOCK_HOOK_PATH = "cosmos_curator.core.utils.infra.ray_cluster_utils.register_pre_shutdown_hook"


@pytest.fixture(autouse=True)
def _clean_staging_env(monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    """Ensure COSMOS_CURATOR_ARTIFACTS_STAGING_DIR is clean before each test.

    After the test, clean up any staging directories that were created.
    """
    monkeypatch.delenv("COSMOS_CURATOR_ARTIFACTS_STAGING_DIR", raising=False)
    yield
    staging = os.environ.get("COSMOS_CURATOR_ARTIFACTS_STAGING_DIR", "")
    if staging and pathlib.Path(staging).exists():
        shutil.rmtree(staging, ignore_errors=True)


class TestArtifactDeliveryCreate:
    """Verify ArtifactDelivery.create() sets up the staging environment."""

    def test_sets_staging_env_var(self, tmp_path: pathlib.Path) -> None:
        """After create(), COSMOS_CURATOR_ARTIFACTS_STAGING_DIR is set to a real temp dir."""
        with patch(_MOCK_HOOK_PATH):
            ArtifactDelivery.create(
                kind="profiling",
                output_dir=str(tmp_path / "output"),
            )

        staging_dir = os.environ.get("COSMOS_CURATOR_ARTIFACTS_STAGING_DIR")
        assert staging_dir is not None
        assert pathlib.Path(staging_dir).is_dir()

    def test_second_create_reuses_existing_staging_dir(self, tmp_path: pathlib.Path) -> None:
        """Two create() calls reuse the same staging base directory."""
        output_dir = str(tmp_path / "output")
        with patch(_MOCK_HOOK_PATH):
            ArtifactDelivery.create(
                kind="profiling",
                output_dir=output_dir,
            )
            first_staging = os.environ.get("COSMOS_CURATOR_ARTIFACTS_STAGING_DIR")

            ArtifactDelivery.create(
                kind="traces",
                output_dir=output_dir,
            )
            second_staging = os.environ.get("COSMOS_CURATOR_ARTIFACTS_STAGING_DIR")

        # The env var should be the same both times (first create sets it,
        # second create finds it already set and reuses it).
        assert first_staging == second_staging
        assert first_staging is not None

    def test_second_create_does_not_leak_orphan_temp_dirs(self, tmp_path: pathlib.Path) -> None:
        """Calling create() twice must not leave orphan temp directories.

        Previously, ``tempfile.mkdtemp()`` was evaluated eagerly as
        the default argument to ``os.environ.get()``, creating a new
        physical directory on every call even when the env var was
        already set.  The orphan was never used or cleaned up.
        """
        output_dir = str(tmp_path / "output")
        with patch(_MOCK_HOOK_PATH):
            ArtifactDelivery.create(kind="profiling", output_dir=output_dir)
            active_staging = os.environ.get("COSMOS_CURATOR_ARTIFACTS_STAGING_DIR")
            assert active_staging is not None

            # Snapshot all cosmos_curator_staging_ dirs BEFORE the second call.
            staging_parent = pathlib.Path(active_staging).parent
            dirs_before = set(staging_parent.glob("cosmos_curator_staging_*"))

            ArtifactDelivery.create(kind="traces", output_dir=output_dir)

            # No new staging directories should have appeared.
            dirs_after = set(staging_parent.glob("cosmos_curator_staging_*"))
            orphans = dirs_after - dirs_before
            assert not orphans, f"Orphan temp directories created: {orphans}"

    def test_registers_pre_shutdown_hook_when_requested(self, tmp_path: pathlib.Path) -> None:
        """When collect_on_shutdown=True, register_pre_shutdown_hook is called."""
        mock_register = MagicMock()
        with patch(_MOCK_HOOK_PATH, mock_register):
            ArtifactDelivery.create(
                kind="profiling",
                output_dir=str(tmp_path / "output"),
                collect_on_shutdown=True,
            )

        mock_register.assert_called_once()
        # The registered hook should be a callable (bound method).
        registered_callable = mock_register.call_args[0][0]
        assert callable(registered_callable)


class TestCollectContract:
    """Verify collect() idempotency and guard clauses."""

    def test_second_collect_returns_zero(self, tmp_path: pathlib.Path) -> None:
        """After _collected is set, collect() returns 0 without side effects."""
        with patch(_MOCK_HOOK_PATH):
            delivery = ArtifactDelivery.create(
                kind="profiling",
                output_dir=str(tmp_path / "output"),
            )

        # Mark as already collected.
        delivery._collected = True

        result = delivery.collect()
        assert result == 0

    def test_empty_output_dir_skips_collection(self) -> None:
        """When output_dir is empty, collect() returns 0 immediately."""
        with patch(_MOCK_HOOK_PATH):
            delivery = ArtifactDelivery.create(
                kind="profiling",
                output_dir="",  # Empty output dir.
            )

        result = delivery.collect()
        assert result == 0


class TestKindStagingCleanup:
    """Verify _cleanup_kind_staging removes only its own subdirectory."""

    def test_sibling_staging_survives_cleanup(self, tmp_path: pathlib.Path) -> None:
        """When one delivery cleans up, sibling staging subdirectories are preserved."""
        output_dir = str(tmp_path / "output")
        with patch(_MOCK_HOOK_PATH):
            profiling_delivery = ArtifactDelivery.create(
                kind="profiling",
                output_dir=output_dir,
            )
            traces_delivery = ArtifactDelivery.create(
                kind="traces",
                output_dir=output_dir,
                upload_subdir="traces",
            )

        # Simulate workers writing artifacts to both staging subdirectories.
        profiling_dir = pathlib.Path(profiling_delivery._staging_dir)
        traces_dir = pathlib.Path(traces_delivery._staging_dir)
        profiling_dir.mkdir(parents=True, exist_ok=True)
        traces_dir.mkdir(parents=True, exist_ok=True)
        (profiling_dir / "cpu_report.html").write_text("cpu data")
        (traces_dir / "spans.jsonl").write_text("trace data")

        # Traces delivery cleans up its staging subdirectory.
        traces_delivery._cleanup_kind_staging("[test]")

        # Traces subdirectory should be gone.
        assert not traces_dir.exists()

        # Profiling subdirectory and its files must survive.
        assert profiling_dir.exists()
        assert (profiling_dir / "cpu_report.html").read_text() == "cpu data"

        # The shared base staging env var must still be set.
        assert os.environ.get("COSMOS_CURATOR_ARTIFACTS_STAGING_DIR") is not None


class TestUploadDestRouting:
    """Verify upload destination includes the upload_subdir."""

    def test_upload_dest_includes_subdir(self, tmp_path: pathlib.Path) -> None:
        """Upload destination combines output_dir and upload_subdir.

        The actual upload_dest is built inside collect(), so we verify
        that the stored _output_dir and _upload_subdir are correct and
        will produce the expected path:  "<output_dir>/<upload_subdir>".
        """
        output_dir = str(tmp_path / "output")
        with patch(_MOCK_HOOK_PATH):
            delivery = ArtifactDelivery.create(
                kind="traces",
                output_dir=output_dir,
                upload_subdir="traces",
            )

        # Verify the internal state that collect() will use
        # to build "upload_dest = f'{output_dir}/{upload_subdir}'".
        assert delivery._output_dir == output_dir
        assert delivery._upload_subdir == "traces"
        # Reconstruct the same logic collect() uses.
        expected_upload_dest = f"{delivery._output_dir}/{delivery._upload_subdir}"
        assert expected_upload_dest == f"{output_dir}/traces"


def _make_delivery(
    tmp_path: pathlib.Path,
    *,
    output_dir: str = "",
    strict: bool = False,
) -> ArtifactDelivery:
    """Create an ArtifactDelivery instance with mocked shutdown hook."""
    effective_output_dir = output_dir or str(tmp_path / "output")
    with patch(_MOCK_HOOK_PATH):
        return ArtifactDelivery.create(
            kind="profiling",
            output_dir=effective_output_dir,
            strict=strict,
        )


def _noop_span() -> MagicMock:
    """Return a MagicMock that satisfies TracedSpan.current() usage.

    MagicMock auto-creates child mocks for any attribute access, so
    explicit assignments are unnecessary.
    """
    return MagicMock()


_TWO_NODES = [
    {"NodeID": "id-a", "NodeName": "node-a"},
    {"NodeID": "id-b", "NodeName": "node-b"},
]

_MOCK_NODES_PATH = "cosmos_curator.core.utils.artifacts.delivery.ray_cluster_utils.get_live_nodes"
_MOCK_TRACED_SPAN_PATH = "cosmos_curator.core.utils.artifacts.delivery.TracedSpan"
_MOCK_RAY_PATH = "cosmos_curator.core.utils.artifacts.delivery.ray"
_MOCK_NODE_UPLOADER_PATH = "cosmos_curator.core.utils.artifacts.delivery._NodeUploader"
_MOCK_TRANSPORT_PATH = "cosmos_curator.core.utils.artifacts.delivery.RayFileTransport"
_MOCK_STALL_TIMEOUT_PATH = "cosmos_curator.core.utils.artifacts.delivery._REMOTE_COLLECT_TIMEOUT_S"


class TestCollectRemotePartialSuccess:
    """Verify _collect_remote preserves partial results when a node fails."""

    def test_lenient_returns_partial_count(self, tmp_path: pathlib.Path) -> None:
        """When one node fails in lenient mode, files from the successful node are counted."""
        delivery = _make_delivery(tmp_path, strict=False)

        ref_a, ref_b = MagicMock(name="ref_a"), MagicMock(name="ref_b")

        mock_actor = MagicMock()
        mock_actor.upload.remote.side_effect = [ref_a, ref_b]

        mock_uploader = MagicMock()
        mock_uploader.options.return_value.remote.return_value = mock_actor

        def fake_ray_get(ref: object) -> _NodeUploadResult:
            if ref is ref_a:
                return _NodeUploadResult(node_name="node-a", files_uploaded=5, errors=())
            msg = "actor crashed"
            raise RuntimeError(msg)

        mock_ray = MagicMock()
        mock_ray.wait.side_effect = [
            ([ref_a], [ref_b]),
            ([ref_b], []),
        ]
        mock_ray.get.side_effect = fake_ray_get
        mock_ray.kill = MagicMock()

        with (
            patch(_MOCK_NODES_PATH, return_value=_TWO_NODES),
            patch(_MOCK_NODE_UPLOADER_PATH, mock_uploader),
            patch(_MOCK_RAY_PATH, mock_ray),
            patch(_MOCK_TRACED_SPAN_PATH) as mock_ts,
        ):
            mock_ts.current.return_value = _noop_span()
            result = delivery._collect_remote("[test]", "s3://bucket/dest")

        assert result == 5
        assert delivery._collected is True

    def test_strict_raises_on_first_failure(self, tmp_path: pathlib.Path) -> None:
        """When one node fails in strict mode, ArtifactDeliveryError is raised."""
        delivery = _make_delivery(tmp_path, strict=True)

        ref_a, ref_b = MagicMock(name="ref_a"), MagicMock(name="ref_b")

        mock_actor = MagicMock()
        mock_actor.upload.remote.side_effect = [ref_a, ref_b]

        mock_uploader = MagicMock()
        mock_uploader.options.return_value.remote.return_value = mock_actor

        mock_ray = MagicMock()
        # First ready is the failing node.
        mock_ray.wait.return_value = ([ref_b], [ref_a])
        mock_ray.get.side_effect = RuntimeError("actor crashed")
        mock_ray.kill = MagicMock()

        with (
            patch(_MOCK_NODES_PATH, return_value=_TWO_NODES),
            patch(_MOCK_NODE_UPLOADER_PATH, mock_uploader),
            patch(_MOCK_RAY_PATH, mock_ray),
            patch(_MOCK_TRACED_SPAN_PATH) as mock_ts,
        ):
            mock_ts.current.return_value = _noop_span()
            with pytest.raises(ArtifactDeliveryError, match="Remote upload failed"):
                delivery._collect_remote("[test]", "s3://bucket/dest")


class TestCollectRemoteTimeout:
    """Verify _collect_remote timeout behavior."""

    def test_timeout_lenient_returns_zero(self, tmp_path: pathlib.Path) -> None:
        """When ray.wait times out in lenient mode, returns 0.

        The adaptive batching loop accumulates stall time in 100 ms
        increments.  We patch _REMOTE_COLLECT_TIMEOUT_S to a tiny
        value so the stall guard triggers after one empty ray.wait
        return.
        """
        delivery = _make_delivery(tmp_path, strict=False)

        ref_a = MagicMock(name="ref_a")
        mock_actor = MagicMock()
        mock_actor.upload.remote.return_value = ref_a

        mock_uploader = MagicMock()
        mock_uploader.options.return_value.remote.return_value = mock_actor

        mock_ray = MagicMock()
        # ray.wait returns empty ready list = stall.
        mock_ray.wait.return_value = ([], [ref_a])
        mock_ray.kill = MagicMock()

        with (
            patch(_MOCK_NODES_PATH, return_value=[_TWO_NODES[0]]),
            patch(_MOCK_NODE_UPLOADER_PATH, mock_uploader),
            patch(_MOCK_RAY_PATH, mock_ray),
            patch(_MOCK_TRACED_SPAN_PATH) as mock_ts,
            patch(_MOCK_STALL_TIMEOUT_PATH, 0.1),
        ):
            mock_ts.current.return_value = _noop_span()
            result = delivery._collect_remote("[test]", "s3://bucket/dest")

        assert result == 0

    def test_timeout_strict_raises(self, tmp_path: pathlib.Path) -> None:
        """When ray.wait times out in strict mode, ArtifactDeliveryError is raised.

        The adaptive batching loop accumulates stall time in 100 ms
        increments.  We patch _REMOTE_COLLECT_TIMEOUT_S to a tiny
        value so the stall guard triggers after one empty ray.wait
        return.
        """
        delivery = _make_delivery(tmp_path, strict=True)

        ref_a = MagicMock(name="ref_a")
        mock_actor = MagicMock()
        mock_actor.upload.remote.return_value = ref_a

        mock_uploader = MagicMock()
        mock_uploader.options.return_value.remote.return_value = mock_actor

        mock_ray = MagicMock()
        mock_ray.wait.return_value = ([], [ref_a])
        mock_ray.kill = MagicMock()

        with (
            patch(_MOCK_NODES_PATH, return_value=[_TWO_NODES[0]]),
            patch(_MOCK_NODE_UPLOADER_PATH, mock_uploader),
            patch(_MOCK_RAY_PATH, mock_ray),
            patch(_MOCK_TRACED_SPAN_PATH) as mock_ts,
            patch(_MOCK_STALL_TIMEOUT_PATH, 0.1),
        ):
            mock_ts.current.return_value = _noop_span()
            with pytest.raises(ArtifactDeliveryError, match="Timeout"):
                delivery._collect_remote("[test]", "s3://bucket/dest")


class TestCollectLocalPartialFailure:
    """Verify _collect_local strict/lenient behavior on node failures."""

    def test_lenient_returns_total_files_despite_node_failure(self, tmp_path: pathlib.Path) -> None:
        """When some nodes fail in lenient mode, total_files from successful nodes is returned."""
        delivery = _make_delivery(tmp_path, strict=False)

        mock_transport = MagicMock()
        mock_transport.collect.return_value = CollectResult(
            total_files=7,
            nodes_ok=("node-a",),
            nodes_failed=(("node-b", "connection lost"),),
        )

        writer_mock = MagicMock()
        writer_mock.base_path = str(tmp_path / "output")

        with (
            patch(_MOCK_TRANSPORT_PATH, return_value=mock_transport),
            patch(_MOCK_TRACED_SPAN_PATH) as mock_ts,
        ):
            mock_ts.current.return_value = _noop_span()
            result = delivery._collect_local("[test]", writer_mock)

        assert result == 7
        assert delivery._collected is True

    def test_strict_raises_on_node_failure(self, tmp_path: pathlib.Path) -> None:
        """When any node fails in strict mode, ArtifactDeliveryError is raised."""
        delivery = _make_delivery(tmp_path, strict=True)

        mock_transport = MagicMock()
        mock_transport.collect.return_value = CollectResult(
            total_files=7,
            nodes_ok=("node-a",),
            nodes_failed=(("node-b", "timeout"),),
        )

        writer_mock = MagicMock()
        writer_mock.base_path = str(tmp_path / "output")

        with (
            patch(_MOCK_TRANSPORT_PATH, return_value=mock_transport),
            patch(_MOCK_TRACED_SPAN_PATH) as mock_ts,
        ):
            mock_ts.current.return_value = _noop_span()
            with pytest.raises(ArtifactDeliveryError, match="Lost files from node"):
                delivery._collect_local("[test]", writer_mock)
