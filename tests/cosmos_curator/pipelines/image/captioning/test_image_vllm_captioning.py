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
"""GPU test for image vLLM captioning (load → prep → caption)."""

import pathlib

import pytest

from cosmos_curator.core.interfaces.pipeline_interface import run_pipeline
from cosmos_curator.core.interfaces.runner_interface import RunnerInterface
from cosmos_curator.pipelines.image.captioning.image_vllm_stages import (
    ImageVllmCaptionStage,
    ImageVllmPrepStage,
)
from cosmos_curator.pipelines.image.read_write.image_load_stage import ImageLoadStage
from cosmos_curator.pipelines.image.utils.data_model import ImagePipeTask
from cosmos_curator.pipelines.video.utils.data_model import (
    VllmConfig,
    VllmSamplingConfig,
)


@pytest.mark.env("unified")
def test_image_vllm_caption_generation(
    sample_image_task: ImagePipeTask,
    sequential_runner: RunnerInterface,
    image_data_dir: pathlib.Path,
) -> None:
    """Run image pipeline (load → prep → caption) and assert non-empty caption."""
    model_variant = "qwen"
    vllm_config = VllmConfig(
        model_variant=model_variant,
        use_image_input=True,
        prompt_variant="image",
        sampling_config=VllmSamplingConfig(temperature=0.0, max_tokens=1024),
    )
    stages = [
        ImageLoadStage(
            input_path=str(image_data_dir),
            input_s3_profile_name="default",
            verbose=False,
            log_stats=False,
        ),
        ImageVllmPrepStage(vllm_config=vllm_config),
        ImageVllmCaptionStage(vllm_config=vllm_config),
    ]
    tasks = run_pipeline([sample_image_task], stages, runner=sequential_runner)

    assert tasks is not None
    assert len(tasks) == 1
    assert "caption_prep" not in tasks[0].image.errors, f"Caption prep should succeed: {tasks[0].image.errors}"
    assert tasks[0].image.caption.strip(), (
        f"Expected non-empty caption for {model_variant}, got: {tasks[0].image.caption!r}"
    )
    assert tasks[0].image.captions.get(model_variant) == tasks[0].image.caption
    assert tasks[0].image.caption_status in {"success", "truncated"}
    assert model_variant in tasks[0].image.token_counts
