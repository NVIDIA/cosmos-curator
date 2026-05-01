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

"""Example Hello World using Ray Data.

Demonstrates two Ray Data patterns:
- Column expressions for lightweight transforms (lowercasing)
- ``map_batches`` with a stateful class for GPU inference (GPT-2)
"""

import pyarrow as pa  # type: ignore[import-untyped]
import ray
from ray.data import ActorPoolStrategy
from ray.data.expressions import col

from cosmos_curator.core.utils.arrow_utils import with_column
from cosmos_curator.core.utils.pixi_runtime_envs import PixiRuntimeEnv
from cosmos_curator.models.gpt2 import GPT2

EXAMPLE_PROMPTS = ["The KEY TO A CREATING GOOD art is", "Once upon a time"]


class _GPT2Predictor:
    """Stateful GPT-2 predictor for Ray Data actor pool."""

    def __init__(self) -> None:
        self._model = GPT2()
        self._model.setup()

    def __call__(self, batch: pa.Table) -> pa.Table:
        outputs = [self._model.generate(p) for p in batch["prompt"].to_pylist()]
        return with_column(batch, "output", pa.array(outputs))


def main() -> None:
    """Run the hello world pipeline on Ray Data."""
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    ds: ray.data.Dataset = ray.data.from_items([{"prompt": p} for p in EXAMPLE_PROMPTS])
    ds = ds.with_column("prompt", col("prompt").str.lower())
    ds = ds.map_batches(
        _GPT2Predictor,
        batch_size=1,
        batch_format="pyarrow",
        num_gpus=0.8,
        compute=ActorPoolStrategy(size=1),
        runtime_env=PixiRuntimeEnv("transformers"),
    )
    ds.show()


if __name__ == "__main__":
    main()
