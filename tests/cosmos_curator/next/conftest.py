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

"""Shared fixtures and helpers for the Curator Next test tree.

Holds the single local Ray cluster, the mecka action-artifact builders, and the
row-address builder, each of which is needed by more than one package below this
one and so cannot live under any of them.
"""

import pathlib
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
import ray
from loguru import logger
from scipy.spatial.transform import Rotation  # type: ignore[import-untyped]

from cosmos_curator.next.media.action_binary import ACTION_BINARY_SPEC_BY_DATASET, encode_action_bin
from cosmos_curator.next.utils.lance_utils import ROW_ADDRESS_FRAGMENT_SHIFT

# A wrist track needs enough surviving frames that the validity gates are not
# testing the fixture's own size. Sixteen leaves room for a test to mask several
# frames and still describe a path.
DEFAULT_ACTION_FRAMES = 16

# The spec whose name the ACT2 header records. Tests register their own *dataset*
# against this spec rather than reusing a real dataset name.
MECKA_SPEC = "mecka"
SYNTHETIC_MECKA_DATASET = "ds_under_test"

# A non-dexterous spec, for tests that need an artifact the wrist descriptor cannot
# consume. Named for the spec it points at, not for any dataset that uses it.
NON_DEXTEROUS_SPEC = "libero"
SYNTHETIC_NON_DEXTEROUS_DATASET = "ds_under_test_plain"

# The custom resource production IO stages reserve, and the slot count the shared
# cluster advertises. An IO-slot count, not compute: it leaves the CPU ceiling
# below exactly as it is.
CURATOR_IO_RESOURCE = "curator_io"
CURATOR_IO_SLOTS = 16

# The CPU ceiling the shared cluster publishes. Named rather than left as a
# literal inside the starter because ``curator_io_cluster`` compares against it:
# a starter and a guard holding separate copies would disagree the moment either
# moved, and the guard's answer to that disagreement is to restart the cluster on
# every test that asks for it.
SHARED_CLUSTER_CPUS = 4


def row_address(fragment_id: int, offset: int) -> int:
    """Return the Lance address of the row at ``offset`` within ``fragment_id``.

    Inverts the shared format fact so a hand-built row carries a checkpoint id
    consistent with the fragment id beside it. Building from that constant is
    safe HERE because every caller treats the id as opaque and asserts only that
    it survives; the tests that assert the DERIVATION spell the shift themselves,
    so that they cannot be satisfied by the constant they are checking.
    """
    return (fragment_id << ROW_ADDRESS_FRAGMENT_SHIFT) | offset


@pytest.fixture
def loguru_records() -> Iterator[list[dict[str, Any]]]:
    """Capture loguru records for the test body, so level / text / traceback can be asserted.

    Loguru does not route through the stdlib ``logging`` tree, so pytest's
    ``caplog`` sees nothing from ``loguru.logger``. A dedicated sink is the
    supported way to inspect emitted records; ``record["exception"]`` is non-None
    only when the call used ``logger.opt(exception=True)`` (a bare
    ``exc_info=True`` is silently ignored by loguru).
    """
    records: list[dict[str, Any]] = []
    sink_id = logger.add(lambda message: records.append(message.record), level="DEBUG")
    try:
        yield records
    finally:
        logger.remove(sink_id)


@pytest.fixture(scope="session")
def ray_local() -> Iterator[None]:
    """Start one local Ray cluster for the whole session.

    This is the only place in the Curator Next test tree that starts a cluster,
    and it must stay that way. Session scope means pytest caches the fixture
    after the first test that requests it and never re-runs this body, so a
    module that shuts the cluster down to run one of its own leaves every later
    consumer holding a fixture that no longer matches a live cluster. Nothing
    reports that: Ray Data auto-initialises a whole-machine cluster when none
    exists rather than failing, so the later tests keep passing while silently
    losing the CPU ceiling below. A module needing a shape this one does not
    provide is a design question, not a local fix.

    Session scope is also what the suite's wall clock wants, since cluster
    startup dominates it and each test builds its own datasets under its own
    ``tmp_path``, so nothing leaks between tests through the cluster.

    ``num_gpus=0`` is deliberate: it keeps the cluster shape fixed for the whole
    run, so a test needing GPU resources must patch ``ray.cluster_resources``
    rather than re-initialise.

    Ray Data does not reject a task that asks for more resource than the cluster
    holds - it backpressures it under ``ResourceBudget`` forever, with no error
    and no progress - and no cluster size removes that hazard. A test driving a
    Ray Data stage must therefore pin the resource knobs it passes rather than
    inherit a production default sized for a real cluster.

    ``curator_io`` is advertised because production IO stages request slots of it,
    and a cluster without it does not report the gap usefully: the guarded entry
    points raise a bare resource error while the unguarded ones hand Ray Data a
    task that can never be scheduled. Sixteen is an IO-slot count, not compute -
    it leaves the CPU budget above exactly as it is.
    """
    _start_shared_cluster()
    try:
        yield
    finally:
        ray.shutdown()


@pytest.fixture
def curator_io_cluster(ray_local: None) -> None:
    """Guarantee the shared cluster's shape for a test whose stages request ``curator_io``.

    ``ray_local`` starts the cluster once and is then cached, so it cannot notice
    that the cluster it started is gone. It does go: a production path elsewhere in
    the tree shuts Ray down mid-session, and the next Ray Data execution silently
    auto-initialises a whole-machine cluster advertising no custom resources. Every
    later test then runs against a cluster nobody configured - harmless until a
    stage asks for a slot of ``curator_io``, at which point the guarded entry points
    fail with a resource error that names the resource and not the cause.

    Ask for this fixture rather than ``ray_local`` when a stage under test reserves
    ``curator_io``. It re-creates the documented shape when, and only when, the
    ambient cluster is not it.

    The check is on the advertised SLOT COUNT and is exact, not a floor, because
    any other count is a cluster nobody configured. Presence alone would accept a
    smaller count and then cap IO concurrency below what the test asked for, which
    reads as a slow stage; a larger count lifts the ceiling a test observing IO
    backpressure is relying on. Both failures are silent, so neither is accepted.

    The CPU count is checked on the same terms, because an IO stage's actor count
    is the lower of what the slots and the CPUs allow. A cluster carrying the
    right slot count on a whole machine's CPUs therefore satisfies a slots-only
    check while still moving that count, which moves the figure a test asserts
    against without touching the resource the fixture is named for.
    """
    del ray_local
    resources = ray.cluster_resources() if ray.is_initialized() else {}
    advertised = resources.get(CURATOR_IO_RESOURCE, 0.0)
    advertised_cpus = resources.get("CPU", 0.0)
    if advertised == float(CURATOR_IO_SLOTS) and advertised_cpus == float(SHARED_CLUSTER_CPUS):
        return
    logger.warning(
        f"The ambient Ray cluster advertises {advertised} {CURATOR_IO_RESOURCE} slots and "
        f"{advertised_cpus} CPU, not the {CURATOR_IO_SLOTS} slots and {SHARED_CLUSTER_CPUS} CPU the "
        f"shared cluster publishes; a previous module replaced it, so it is being restored."
    )
    ray.shutdown()
    _start_shared_cluster()


def _start_shared_cluster() -> None:
    """Start the one cluster shape the Curator Next tests share."""
    ray.init(
        # Deliberate headroom so co-scheduled stages under test are not starved.
        # Do not lower this to force a stage to pin its own CPU request: an
        # over-asking task backpressures at any cluster size (see ray_local).
        num_cpus=SHARED_CLUSTER_CPUS,
        num_gpus=0,
        resources={CURATOR_IO_RESOURCE: CURATOR_IO_SLOTS},
        include_dashboard=False,
        ignore_reinit_error=True,
        log_to_driver=False,
    )


@pytest.fixture
def dexterous_dataset(monkeypatch: pytest.MonkeyPatch) -> str:
    """Register a test-owned dataset name against the real mecka spec.

    Points a synthetic dataset at ``MECKA_SPEC`` rather than cloning the spec
    under a new name, so the test depends only on a mecka spec existing, not on
    which dataset names are registered today.

    Ray-safe despite the monkeypatch being driver-local: the registry is consulted
    only while ENCODING a bin (in the driver), and the resulting artifact is
    self-describing, so a worker decodes it without resolving the dataset name.
    """
    monkeypatch.setitem(ACTION_BINARY_SPEC_BY_DATASET, SYNTHETIC_MECKA_DATASET, MECKA_SPEC)
    return SYNTHETIC_MECKA_DATASET


@pytest.fixture
def non_dexterous_dataset(monkeypatch: pytest.MonkeyPatch) -> str:
    """Register a test-owned dataset name against a real non-dexterous spec.

    Lets a test say "a dataset whose artifacts carry no hand arrays" without naming
    a real dataset, which would couple it to today's registry contents.
    """
    monkeypatch.setitem(ACTION_BINARY_SPEC_BY_DATASET, SYNTHETIC_NON_DEXTEROUS_DATASET, NON_DEXTEROUS_SPEC)
    return SYNTHETIC_NON_DEXTEROUS_DATASET


@pytest.fixture
def mecka_payload() -> Callable[..., dict[str, npt.NDArray[np.float32]]]:
    """Return a builder for a mecka action payload with a moving wrist and static camera."""

    def build(n_frames: int = DEFAULT_ACTION_FRAMES, *, seed: int = 0) -> dict[str, npt.NDArray[np.float32]]:
        rng = np.random.default_rng(seed)
        t = np.linspace(0.0, 1.0, n_frames)
        wrist_pos = np.stack([t, 0.5 * np.sin(3 * t), 0.2 * t], axis=1)
        wrist_quat = Rotation.from_rotvec(np.stack([0.3 * t, 0.1 * t, 0.2 * t], axis=1)).as_quat()

        def arm() -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
            pos = np.zeros((n_frames, 63), dtype=np.float32)
            pos[:, 0:3] = wrist_pos + 0.001 * rng.standard_normal((n_frames, 3))
            rot = np.zeros((n_frames, 84), dtype=np.float32)
            rot[:, 0:4] = wrist_quat
            return pos, rot

        left_pos, left_rot = arm()
        right_pos, right_rot = arm()
        return {
            "hand_left_cam": left_pos,
            "hand_right_cam": right_pos,
            "hand_left_cam_rotation": left_rot,
            "hand_right_cam_rotation": right_rot,
            "camera_position": np.zeros((n_frames, 3), dtype=np.float32),
            "camera_rotation": np.tile([0.0, 0.0, 0.0, 1.0], (n_frames, 1)).astype(np.float32),
            "intrinsics": np.zeros(8, dtype=np.float32),
        }

    return build


@pytest.fixture
def make_mecka_bin(
    mecka_payload: Callable[..., dict[str, npt.NDArray[np.float32]]],
    dexterous_dataset: str,
) -> Callable[..., str]:
    """Return a builder that writes a valid mecka ACT2 file and yields its URI.

    Depends on ``dexterous_dataset`` so the default dataset name is registered
    before ``encode_action_bin`` resolves it.
    """

    def build(path: pathlib.Path, *, seed: int = 0, n_frames: int = DEFAULT_ACTION_FRAMES) -> str:
        payload = mecka_payload(n_frames, seed=seed)
        path.write_bytes(encode_action_bin(payload, dexterous_dataset))
        return str(path)

    return build


@pytest.fixture
def make_non_dexterous_bin(non_dexterous_dataset: str) -> Callable[..., str]:
    """Return a builder that writes a hand-array-free ACT2 file and yields its URI.

    The artifact decodes cleanly but carries nothing mecka wrist alignment can
    consume, so it is rejected by the geometry checks rather than by the reader.
    """

    def build(path: pathlib.Path, *, n_frames: int = DEFAULT_ACTION_FRAMES) -> str:
        payload = {
            "action": np.zeros((n_frames, 7), dtype=np.float32),
            "state": np.zeros((n_frames, 8), dtype=np.float32),
        }
        path.write_bytes(encode_action_bin(payload, non_dexterous_dataset))
        return str(path)

    return build
