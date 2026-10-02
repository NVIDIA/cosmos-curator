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

"""GPU CI tests for Cosmos-Embed1 using staged model weights."""

import numpy as np
import pytest

from cosmos_curator.core.utils.model import pixi_utils
from cosmos_curator.next.recipes.multimodal_embedding.contracts import resolve_model
from cosmos_curator.next.recipes.multimodal_embedding.model_adapter import (
    CosmosEmbed1VideoEncoder,
    CosmosEmbed1VideoPreprocessor,
    validate_projection,
)

pytestmark = pytest.mark.env("legacy-transformers")


def test_real_model_encodes_eight_rgb_frames() -> None:
    """The pinned processor and checkpoint return a normalized 768D video vector."""
    if not pixi_utils.is_running_in_env("legacy-transformers"):
        pytest.skip("Cosmos-Embed1 requires the legacy-transformers Pixi environment")
    model = resolve_model("cosmos_embed1_336p")
    preprocessor = CosmosEmbed1VideoPreprocessor(model)
    encoder = CosmosEmbed1VideoEncoder(model)
    frames = tuple(np.full((336, 336, 3), index * 16, dtype=np.uint8) for index in range(8))
    vector = validate_projection(encoder.encode_batch(preprocessor.preprocess(frames)[None, ...]), dimension=768)
    assert vector.shape == (model.vector_dimension,)
    assert vector.dtype == np.float32
    assert np.isfinite(vector).all()
    assert abs(float(np.linalg.norm(vector.astype(np.float64))) - 1.0) <= 0.02
