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

"""T5 Encoder Model."""

import enum
import pathlib
from collections.abc import Iterable
from typing import Any, Final

import attrs
import numpy as np
import numpy.typing as npt
import torch

from cosmos_curator.core.interfaces.model_interface import ModelInterface
from cosmos_curator.core.utils.environment import CONTAINER_PATHS_CODE_DIR
from cosmos_curator.core.utils.misc import grouping
from cosmos_curator.core.utils.model import model_utils, pixi_utils

# pyright: reportMissingImports=false
# pyright: reportUnboundVariable=false
if pixi_utils.is_running_in_env("unified"):
    from transformers import T5Config, T5EncoderModel, T5TokenizerFast
    from transformers import logging as transformers_logging

    # Suppresses a lot of unhelpful warnings from transformers.
    transformers_logging.set_verbosity_error()  # type: ignore[no-untyped-call]


_CONFIG_PATH = CONTAINER_PATHS_CODE_DIR / pathlib.Path("cosmos_curator/models/configs/t5_encoder.json")
_T5_MODEL_ID: Final = "google-t5/t5-11b"
_T5_MODEL_WEIGHTS_REDUCED: Final = "pytorch_model.bin.reduced"


class ModelVariant(enum.Enum):
    """Enumeration of T5 model variants with different precision settings."""

    T5_XXL = 0
    T5_XXL_16_BIT = 1


@attrs.define
class EncodedSample:
    """Container for encoded text samples with associated metadata."""

    encoded_text: npt.NDArray[Any]
    length: int
    attn_mask: npt.NDArray[Any]
    offset_mappings: npt.NDArray[Any] | None = None

    def truncate(self, variant: ModelVariant) -> None:
        """Truncate the encoded text to the specified variant.

        Args:
            variant: The model variant to truncate to.

        """
        if variant == ModelVariant.T5_XXL_16_BIT:
            self.encoded_text = self.encoded_text[0 : self.length]
        else:
            self.encoded_text = self.encoded_text[0 : self.length].astype(np.float16)
        self.attn_mask = self.attn_mask[0 : self.length].astype(np.int32)
        if self.offset_mappings is not None:
            self.offset_mappings = self.offset_mappings[0 : self.length].astype(np.int32)


class T5Encoder(ModelInterface):
    """Interface for T5 text encoder model with support for different precision variants."""

    def __init__(
        self,
        variant: ModelVariant = ModelVariant.T5_XXL,
        device: str = "cuda",
        max_length: int | None = None,
    ) -> None:
        """Initialize the T5 encoder model.

        Args:
            variant: The model variant to use.
            device: The device to run the model on.
            max_length: The maximum length of the encoded text.

        """
        super().__init__()
        if max_length is None:
            max_length = 512

        self._variant = variant
        self._device = device
        self._max_length = int(max_length)
        self._output_mapping = self._variant in [
            ModelVariant.T5_XXL,
            ModelVariant.T5_XXL_16_BIT,
        ]

    @property
    def conda_env_name(self) -> str:
        """Get the conda environment name.

        Returns:
            The conda environment name.

        """
        return "unified"

    @property
    def model_id_names(self) -> list[str]:
        """Get the model ID names.

        Returns:
            A list of model ID names.

        """
        return [_T5_MODEL_ID]

    def _load_for_t5(self) -> None:
        """Load the T5 model."""
        tokenizer_local_path = model_utils.get_local_dir_for_weights_name(_T5_MODEL_ID)
        unquant_model_path = model_utils.get_local_dir_for_weights_name(_T5_MODEL_ID) / _T5_MODEL_WEIGHTS_REDUCED

        self._tokenizer = T5TokenizerFast.from_pretrained(tokenizer_local_path, model_max_length=self._max_length)
        self._model = T5EncoderModel.from_pretrained(
            unquant_model_path,
            config=T5Config.from_json_file(_CONFIG_PATH),
            low_cpu_mem_usage=True,
        )
        if self._variant == ModelVariant.T5_XXL_16_BIT:
            self._model = self._model.half()  # cast T5 encoder's weights to fp16

    def setup(self) -> None:
        """Set up the T5 encoder model."""
        self._load_for_t5()
        self._model.to(self._device)
        self._model.eval()
        self._model.requires_grad_(requires_grad=False)

    @torch.inference_mode()
    def _encode_for_batch(
        self,
        prompts: list[str],
        *,
        truncate: bool = True,
    ) -> list[EncodedSample]:
        batch_encoding = self._tokenizer.batch_encode_plus(
            prompts,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self._max_length,
            return_length=True,
            return_offsets_mapping=self._output_mapping,
        )

        # We expect all the processing is done in GPU.
        input_ids = batch_encoding.input_ids.to(self._device)
        attn_mask = batch_encoding.attention_mask.to(self._device)
        if self._output_mapping:
            offsets_mapping = batch_encoding["offset_mapping"]
            offsets_mapping = offsets_mapping.cpu().numpy()
        else:
            offsets_mapping = None

        outputs = self._model(input_ids=input_ids, attention_mask=attn_mask)
        encoded_text = outputs.last_hidden_state

        lengths = attn_mask.sum(dim=1).cpu()  # batch_encoding["lengths"] is not valid for T5TokenizerFast
        for batch_id in range(encoded_text.shape[0]):
            encoded_text[batch_id][lengths[batch_id] :] = 0

        encoded_text = encoded_text.cpu().numpy()
        attn_mask = attn_mask.cpu().numpy()

        encoded_text = encoded_text[:, : self._max_length]
        attn_mask = attn_mask[:, : self._max_length]

        out = []  # type: list[EncodedSample]
        for idx in range(encoded_text.shape[0]):
            offsets = offsets_mapping[idx] if self._output_mapping else None

            out.append(
                EncodedSample(encoded_text[idx], lengths[idx], attn_mask[idx], offsets),
            )

        if truncate:
            for x in out:
                x.truncate(self._variant)
        return out

    def encode(
        self,
        prompts: Iterable[str],
        *,
        truncate: bool = True,
        batch_size: int = 8,
    ) -> list[EncodedSample]:
        """Encode text prompts using the T5 model.

        Args:
            prompts: Iterable of text prompts to encode.
            truncate: Whether to truncate the encoded output.
            batch_size: Size of batches for processing.

        Returns:
            List of encoded samples containing the encoded text and metadata.

        """
        prompts = [x.strip() for x in prompts]
        out = []
        for batch in grouping.split_by_chunk_size(prompts, batch_size):
            out.extend(self._encode_for_batch(batch, truncate=truncate))
        return out
