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

"""Concurrent custom-model imports through the Ray worker runtime, without weights or GPUs."""

import asyncio
import os
import sys
from pathlib import Path

import pytest
import ray

from cosmos_curator.core.utils.pixi_runtime_envs import PixiRuntimeEnv
from tests.cosmos_curator.next.conftest import ray_local  # noqa: F401 -- shared CPU cluster


class _CopyBarrier:
    """Hold both workers inside a cache-file copy before either can import the result."""

    def __init__(self) -> None:
        """Initialize a barrier for two concurrent module copies."""
        self._arrivals = 0
        self._ready = asyncio.Event()

    async def arrive(self) -> None:
        """Wait until both workers have opened their destination files."""
        self._arrivals += 1
        if self._arrivals == 2:
            self._ready.set()
        await asyncio.wait_for(self._ready.wait(), timeout=30)


class _ImportWorker:
    """Load custom model code only after the runtime hook has isolated this worker."""

    def __init__(self) -> None:
        """Verify cache isolation was configured before importing Transformers."""
        assert "transformers" not in sys.modules
        from transformers.utils import HF_MODULES_CACHE  # noqa: PLC0415

        self._cache = HF_MODULES_CACHE
        assert self._cache == os.environ["HF_MODULES_CACHE"]
        assert Path(self._cache).is_dir()

    def load(self, model_dir: str, barrier: ray.actor.ActorHandle) -> dict[str, str | int]:
        """Pause non-atomic copies to reproduce the dangerous overlap deterministically."""
        from transformers import dynamic_module_utils  # noqa: PLC0415

        original_copy = dynamic_module_utils.shutil.copy

        def copy_module(source: str | Path, destination: str | Path) -> str:
            """Hold the configuration file empty until both workers reach the copy."""
            if Path(source).name != "configuration_test.py":
                return original_copy(source, destination)
            data = Path(source).read_bytes()
            with Path(destination).open("wb") as handle:
                ray.get(barrier.arrive.remote(), timeout=40)
                handle.write(data)
            return str(destination)

        dynamic_module_utils.shutil.copy = copy_module
        try:
            processor = dynamic_module_utils.get_class_from_dynamic_module(
                "preprocessing_test.Processor", model_dir, local_files_only=True
            )
        finally:
            dynamic_module_utils.shutil.copy = original_copy
        # A later import in the same worker must reuse its existing cache.
        again = dynamic_module_utils.get_class_from_dynamic_module(
            "preprocessing_test.Processor", model_dir, local_files_only=True
        )
        assert processor is again
        return {"cache": self._cache, "value": processor.value, "pid": os.getpid(), "home": os.environ["HF_HOME"]}


@pytest.mark.integration
@pytest.mark.usefixtures("ray_local")
def test_ray_workers_import_custom_code_with_private_caches(tmp_path: Path) -> None:
    """Concurrent workers load shared staged code without touching shared caches or downloading models."""
    model_dir = tmp_path / "SharedModel"
    model_dir.mkdir()
    configuration = "class Config:\n    value = 42\n"
    preprocessing = "from .configuration_test import Config\nclass Processor:\n    value = Config.value\n"
    (model_dir / "configuration_test.py").write_text(configuration)
    (model_dir / "preprocessing_test.py").write_text(preprocessing)
    shared_cache = tmp_path / "shared-hf-cache"
    shared_cache.mkdir()
    sentinel = shared_cache / "weights.bin"
    sentinel.write_bytes(b"shared weights")
    env = PixiRuntimeEnv(
        "",
        env_vars={
            "HF_HOME": str(shared_cache),
            "HF_MODULES_CACHE": str(shared_cache),
            "HF_HUB_OFFLINE": "1",
            "TMPDIR": str(shared_cache),
        },
    )
    barrier = ray.remote(num_cpus=0)(_CopyBarrier).remote()
    workers = [ray.remote(num_cpus=1, runtime_env=env)(_ImportWorker).remote() for _ in range(2)]
    try:
        results = ray.get([worker.load.remote(str(model_dir), barrier) for worker in workers], timeout=90)
        assert len({row["cache"] for row in results}) == len({row["pid"] for row in results}) == 2
        assert all(Path(str(row["cache"])).parent == Path("/tmp") for row in results)  # noqa: S108
        assert all(row["value"] == 42 and row["home"] == str(shared_cache) for row in results)
        assert list(shared_cache.iterdir()) == [sentinel]
        assert (model_dir / "configuration_test.py").read_text() == configuration
        assert (model_dir / "preprocessing_test.py").read_text() == preprocessing
    finally:
        for worker in workers:
            ray.kill(worker)
        ray.kill(barrier)
