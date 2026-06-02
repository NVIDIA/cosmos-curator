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
"""Shared utilities for OpenAI-compatible embedding API calls."""

import base64
import io
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import tenacity
from PIL import Image

from cosmos_curator.core.utils.model import pixi_utils

if TYPE_CHECKING:
    import openai
    from openai.types.create_embedding_response import CreateEmbeddingResponse

if pixi_utils.is_running_in_env("unified"):
    import openai
    from openai.types.create_embedding_response import CreateEmbeddingResponse


def frame_to_base64_jpeg(frame: npt.NDArray[np.uint8]) -> str:
    """Encode a single image frame (H, W, C uint8 array) as a base64 JPEG data URI."""
    img = Image.fromarray(frame)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def call_openai_embedding_api(
    client: "openai.OpenAI",
    model_name: str,
    content_parts: list[dict[str, Any]],
    max_retries: int = 3,
    retry_delay_seconds: float = 1.0,
) -> npt.NDArray[np.float32]:
    """Call an OpenAI-compatible embedding endpoint with retry logic.

    Args:
        client: Initialized OpenAI client.
        model_name: Model name to pass in the API request.
        content_parts: List of content parts (image_url entries) to embed.
        max_retries: Number of retries before giving up.
        retry_delay_seconds: Fixed delay between retries.

    Returns:
        Embedding as a float32 numpy array.

    Raises:
        RuntimeError: If the API returns no data.
        openai.AuthenticationError: On authentication failure (not retried).
        openai.NotFoundError: On 404 (not retried).
        openai.BadRequestError: On bad request (not retried).

    """

    @tenacity.retry(
        stop=tenacity.stop_after_attempt(max_retries),
        wait=tenacity.wait_fixed(retry_delay_seconds),
        retry=tenacity.retry_if_not_exception_type(
            (openai.AuthenticationError, openai.NotFoundError, openai.BadRequestError),
        ),
        reraise=True,
    )
    def _call() -> npt.NDArray[np.float32]:
        response: CreateEmbeddingResponse = client.post(
            "/embeddings",
            cast_to=CreateEmbeddingResponse,
            body={
                "model": model_name,
                "messages": [{"role": "user", "content": content_parts}],
                "encoding_format": "float",
            },
        )
        if not response.data:
            msg = f"OpenAI-compatible embedding API returned no data. model={model_name!r}"
            raise RuntimeError(msg)
        return np.array(response.data[0].embedding, dtype=np.float32)

    return _call()
