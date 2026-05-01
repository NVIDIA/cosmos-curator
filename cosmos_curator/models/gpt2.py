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

"""Model GPT2."""

from loguru import logger

from cosmos_curator.core.interfaces.model_interface import ModelInterface
from cosmos_curator.core.utils.model import conda_utils, model_utils

# pyright: reportMissingImports=false
# pyright: reportUnboundVariable=false
if conda_utils.is_running_in_env("transformers"):
    from transformers import GPT2LMHeadModel, GPT2Tokenizer


class GPT2(ModelInterface):
    """Interface for GPT-2 text generation model."""

    def __init__(self, max_output_len: int = 64) -> None:
        """Initialize the GPT-2 model.

        Args:
            max_output_len: The maximum length of the output text.

        """
        # constructor runs in default conda env, so cannot initialize the model here
        super().__init__()
        self.max_output_len = max_output_len

    # need override conda_env_name to tell underlying logic which conda env to use
    @property
    def conda_env_name(self) -> str:
        """Get the conda environment name.

        Returns:
            The conda environment name.

        """
        return "transformers"

    # need override model_id_names to tell underlying logic which model IDs to use
    @property
    def model_id_names(self) -> list[str]:
        """Get the model ID names.

        Returns:
            A list of model ID names.

        """
        return ["openai-community/gpt2"]

    # need override setup which is called automatically when creating stage actors
    def setup(self) -> None:
        """Set up the GPT-2 model.

        This method initializes the model and its configuration for text generation.
        It also sets up the tokenizer and model.

        """
        # this runs in the specified conda environment on the actor
        logger.info(f"Setting up GPT2 model in env={conda_utils.get_conda_env_name()}")
        model_dir = model_utils.get_local_dir_for_weights_name(self.model_id_names[0])
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.model = GPT2LMHeadModel.from_pretrained(model_dir)
        self.model.to("cuda")

    # actual data processing function to be called by the stage using this model
    def generate(self, prompt: str) -> str:
        """Generate text using the GPT-2 model.

        Args:
            prompt: The input prompt for text generation.

        Returns:
            The generated text.

        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        output_ids = self.model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            pad_token_id=self.tokenizer.eos_token_id,
            max_length=self.max_output_len,
            num_return_sequences=1,
            repetition_penalty=1.2,
        )
        return str(self.tokenizer.decode(output_ids[0], skip_special_tokens=True))
