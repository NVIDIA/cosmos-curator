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

"""SAM3 (Segment Anything 3) video segmentation model.

SAM3 requires transformers>=5.0.0, which conflicts with vLLM's hard cap of
transformers<5.  This model therefore runs in the dedicated `sam3` pixi
environment (isolated from vLLM) so both can coexist on the same node via
separate Ray workers.
"""

from typing import TYPE_CHECKING, Any

import torch
from loguru import logger

from cosmos_curator.core.interfaces.model_interface import ModelInterface
from cosmos_curator.core.utils.model import conda_utils, model_utils

if TYPE_CHECKING:
    from transformers import Sam3VideoConfig, Sam3VideoModel, Sam3VideoProcessor  # type: ignore[attr-defined]

if conda_utils.is_running_in_env("sam3"):
    from transformers import Sam3VideoConfig, Sam3VideoModel, Sam3VideoProcessor  # type: ignore[attr-defined]

_SAM3_MODEL_ID = "facebook/sam3"


class SAM3Model(ModelInterface):
    """SAM3 video segmentation model wrapper."""

    @property
    def conda_env_name(self) -> str:
        """Get the conda environment name.

        Returns:
            The conda environment name.

        """
        return "sam3"

    @property
    def model_id_names(self) -> list[str]:
        """Get the model ID names.

        Returns:
            A list of model ID names.

        """
        return [_SAM3_MODEL_ID]

    def setup(self, config_overrides: dict[str, Any] | None = None) -> None:
        """Set up the SAM3 model.

        Loads the processor and model from local weights. Logs GPU memory before and
        after loading.

        Args:
            config_overrides: optional dict of Sam3VideoConfig attribute overrides
                (e.g. ``{"recondition_every_nth_frame": 8, "score_threshold_detection": 0.6}``).

        """
        if not torch.cuda.is_available():
            msg = "SAM3 requires a CUDA-capable GPU but none was found"
            raise RuntimeError(msg)

        logger.info("Setting up SAM3 model")
        model_dir = model_utils.get_local_dir_for_weights_name(_SAM3_MODEL_ID)
        if not model_dir.exists():
            msg = f"SAM3 weights not found at {model_dir}. Download via: cosmos-curator model download --model sam3"
            raise FileNotFoundError(msg)

        mem_before = torch.cuda.memory_allocated()

        self.processor: Sam3VideoProcessor = Sam3VideoProcessor.from_pretrained(model_dir, local_files_only=True)

        config: Sam3VideoConfig | None = None
        if config_overrides:
            config = Sam3VideoConfig.from_pretrained(model_dir, local_files_only=True)
            for key, value in config_overrides.items():
                if not hasattr(config, key):
                    msg = f"Sam3VideoConfig has no attribute '{key}'"
                    raise ValueError(msg)
                setattr(config, key, value)
            logger.info(f"SAM3 config overrides: {config_overrides}")

        self.model: Sam3VideoModel = Sam3VideoModel.from_pretrained(
            model_dir,
            config=config,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        ).cuda()
        self.model.eval()

        mem_after = torch.cuda.memory_allocated()
        footprint_gb = (mem_after - mem_before) / 1024**3
        logger.info(f"SAM3 load footprint: {footprint_gb:.3f} GB ({mem_after - mem_before:,} bytes)")
