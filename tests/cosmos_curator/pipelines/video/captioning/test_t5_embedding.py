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
"""Test the t5 encoding."""

import pickle
from pathlib import Path

import numpy as np
import pytest
from loguru import logger

from cosmos_curator.models import t5_encoder  # type: ignore[import-untyped]

_MAX_ABS_DIF = 0.01


def get_captions() -> list[str]:
    """Return the test caption for t5."""
    return [
        "The video opens with a scene of a soccer match in progress. Players are seen wearing two different kits, one predominantly white with red accents and the other predominantly red with white accents. The players are actively moving around the field, with some running towards the ball and others positioning themselves strategically. The field is marked with white lines, and there is a goalpost with a net at one end. The background shows a crowd of spectators and various advertisements on banners around the field. The lighting suggests it is daytime, and the grass on the field appears well-maintained."  # noqa: E501
    ]


def get_ref_embeddings() -> list[np.ndarray]:  # type:  ignore[type-arg]
    """Return the reference embedding."""
    filepath = Path(__file__).parent
    embeddings_file = filepath / "example_t5_embeddings.pickle"
    with Path.open(embeddings_file, "rb") as f:
        return pickle.load(f)  # type: ignore[no-any-return] # noqa: S301


@pytest.mark.env("unified")
def test_t5_encoding() -> None:
    """Tests the t5 encoding."""
    # Set up the t5 model
    model = t5_encoder.T5Encoder(t5_encoder.ModelVariant.T5_XXL)
    model.setup()

    # obtain the caption to pass to the model
    captions = get_captions()
    results = model.encode(captions)

    # get model embeddings
    embeddings = [result.encoded_text for result in results]
    logger.info(f"length of embeddings: {len(embeddings)}")

    ref_embeddings = get_ref_embeddings()
    assert len(embeddings) == len(ref_embeddings)

    # assert that embeddings match reference embeddings
    max_abs_diff = 0.0
    for i, embedding in enumerate(embeddings):
        logger.info(f"embedding {i} shape: {embedding.shape}")
        assert embedding.shape == ref_embeddings[i].shape
        abs_diff = abs(embedding - ref_embeddings[i])
        max_abs_diff = max(np.max(abs_diff), max_abs_diff)

    if max_abs_diff > _MAX_ABS_DIF:
        logger.error(f"T5-encoder test failed with {max_abs_diff=}.")
    else:
        logger.info(f"T5-encoder test passed with {max_abs_diff=}.")
