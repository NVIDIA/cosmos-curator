# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#
# Portions of this file were derived from:
# https://github.com/rapidsai/ray-rapids/blob/main/rapids-experiments/raft_actor.py

"""RAFT Ray actor and pool initialization utilities.

Provides `RAFTActor`, a base Ray actor that pins to one GPU and initializes NCCL
and RAFT collective communications via `raft_dask`, plus
`initialize_raft_actor_pool` to build and wire up a group of actors for
multi-GPU workloads.

RAFT is a RAPIDS library that provides reusable GPU-accelerated algorithms and
systems primitives. In distributed settings it exposes a communication layer
that uses NCCL for multi-GPU collectives and integrates with Dask/Ray via
`raft_dask`.
"""

import os
from typing import Any, TypeVar

import ray
from loguru import logger

from cosmos_curator.core.utils.model import conda_utils

if conda_utils.is_running_in_env("cuml"):
    from pylibraft.common import DeviceResources  # type: ignore[import-not-found]
    from raft_dask.common import nccl  # type: ignore[import-not-found]
    from raft_dask.common.comms_utils import inject_comms_on_handle_coll_only  # type: ignore[import-not-found]


class RAFTActor:
    """A RAFT Ray actor.

    Attributes:
        _index (int): The index of the actor.
        _pool_size (int): The size of the pool.
        _unique_id (bytes): The NCCL unique ID for the actor group.
        display_name (str): Human-readable label for the actor.

    """

    def __init__(
        self,
        index: int,
        pool_size: int,
        *,
        verbose: bool = False,
    ) -> None:
        """Initialize the RAFT actor.

        Args:
            index (int): The index of the actor.
            pool_size (int): The size of the pool (i.e., number of actors).
            verbose (bool, optional): Whether to enable verbose debug logging within the actor. Defaults to False.

        Returns:
            None

        """
        gpu_ids = ray.get_gpu_ids()
        assert len(gpu_ids) == 1, "This actor must be run on a GPU."
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[0])
        self._index = index
        self._pool_size = pool_size
        self._verbose = verbose
        self.display_name = f"{type(self).__name__}-{index}/{pool_size}"

        # Create a NCCL unique ID for this actor session
        self._unique_id: bytes | None = nccl.unique_id() if self._index == 0 else None

        # CPUs assigned to this actor by Ray
        cpu_assigned = float(ray.get_runtime_context().get_assigned_resources().get("CPU", 0.0))  # type: ignore[no-untyped-call]
        self._assigned_cpus = max(1, round(cpu_assigned))
        if self._verbose:
            logger.debug(f"{self.display_name}: assigned CPUs = {self._assigned_cpus}")

    def broadcast_unique_id_from_root(self, actor_handles: list[ray.actor.ActorHandle[Any]]) -> None:
        """Broadcast the NCCL unique ID to the provided actor handles.

        Args:
            actor_handles (list[ray.actor.ActorHandle[Any]]): Actor handles to receive the unique ID.

        Returns:
            None

        Raises:
            RuntimeError: If called by a non-root actor.

        """
        if self._index != 0:
            error_message = "This method should only be called by the root"
            raise RuntimeError(error_message)
        futures = [actor.set_unique_id.remote(self._unique_id) for actor in actor_handles]
        ray.get(futures)

    def _setup_nccl(self) -> None:
        """Set up NCCL communicator.

        Returns:
            None

        """
        self._nccl = nccl.nccl()
        self._nccl.init(self._pool_size, self._unique_id, self._index)

    def _setup_raft(self) -> None:
        """Set up RAFT.

        Returns:
            None

        """
        self._raft_handle = DeviceResources()

        inject_comms_on_handle_coll_only(
            self._raft_handle, self._nccl, self._pool_size, self._index, verbose=self._verbose
        )

    def _setup_post(self) -> None:
        """Post-setup hook.

        Called after setting up NCCL and RAFT. Subclasses may override to perform
        additional setup steps.

        Returns:
            None

        """

    def setup(self) -> None:
        """Set up the actor.

        This method should be called after the root unique ID has been broadcast.

        Returns:
            None

        Raises:
            RuntimeError: If the NCCL unique ID is not set.

        """
        if self._unique_id is None:
            error_message = (
                "The NCCL unique ID is not set. Make sure `broadcast_unique_id_from_root` runs on the root "
                "before calling this method."
            )
            raise RuntimeError(error_message)

        try:
            if self._verbose:
                logger.debug("     Setting up NCCL...")
            self._setup_nccl()
            if self._verbose:
                logger.debug("     Setting up RAFT...")
            self._setup_raft()
            self._setup_post()
            if self._verbose:
                logger.debug("     Setup complete!")
        except Exception as e:
            if self._verbose:
                logger.debug(f"An error occurred while setting up: {e}.")
            raise

    def set_unique_id(self, unique_id: bytes) -> None:
        """Set the NCCL unique ID.

        Args:
            unique_id (bytes): The NCCL unique ID for the actor group.

        Returns:
            None

        Raises:
            RuntimeError: If the NCCL unique ID has already been set.

        """
        if self._verbose:
            logger.debug(f"{self.display_name}: set_unique_id")
        if self._unique_id is not None:
            error_message = "The NCCL unique ID has already been set on this actor; refusing to overwrite."
            raise RuntimeError(error_message)
        self._unique_id = unique_id

    def get_assigned_cpu_count(self) -> int:
        """Return the number of CPU cores effectively available to this actor process.

        Prefers Linux CPU affinity when available; falls back to 1 if unknown.
        """
        n = self._assigned_cpus if isinstance(self._assigned_cpus, int) else 0
        if n <= 0:
            n = int(os.cpu_count() or 1)
        return max(1, int(n))


T = TypeVar("T", bound=RAFTActor)


def initialize_raft_actor_pool[T: RAFTActor](pool_size: int, actor_class: type[T], *, verbose: bool = False) -> list[T]:
    """Initialize a pool of RAFT actors with the specified size and configuration.

    This function initializes a pool of RAFT actors with the specified size and
    configuration. It divides the cluster CPU cores evenly across actors and
    assigns that CPU share to each actor to avoid oversubscription. It starts the
    actors, sets up their communication, and returns a list of the initialized
    actors.

    Args:
        pool_size (int): The number of actors to initialize in the pool.
        actor_class (type[T]): The class of the RAFT actor to initialize, generally a subclass of `RAFTActor`.
        verbose (bool): Whether to enable verbose debug logging in the actors.

    Returns:
        list[T]: A list of the initialized RAFT actors in the pool.

    """
    # Compute even CPU share across actors (fractional CPUs allowed by Ray)
    total_cpus = float(ray.cluster_resources().get("CPU", 0.0))  # type: ignore[no-untyped-call]
    if total_cpus <= 0.0:
        error_message = "Ray reports 0 total CPUs; please start Ray with CPU resources."
        raise RuntimeError(error_message)
    per_actor_cpus = round(total_cpus / float(pool_size), 3)
    logger.info(f"Distributing {total_cpus} CPUs across {pool_size} actors → {per_actor_cpus} CPUs per actor.")

    # Start Actors (unnamed; we keep handles in `pool`)
    pool = [
        actor_class.options(num_cpus=per_actor_cpus).remote(i, pool_size, verbose=verbose)  # type: ignore[attr-defined]
        for i in range(pool_size)
    ]

    # Broadcast NCCL unique ID from root to peers using explicit handles
    ray.get(pool[0].broadcast_unique_id_from_root.remote(pool[1:]))

    # Setup Comms (NCCL/Sub-communicator)
    ray.get([pool[i].setup.remote() for i in range(pool_size)])

    return pool
