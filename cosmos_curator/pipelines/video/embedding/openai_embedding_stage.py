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
"""OpenAI-compatible API embedding stage for remote video embedding inference (e.g. vLLM serving)."""

import concurrent.futures
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import nvtx  # type: ignore[import-untyped]
from loguru import logger

from cosmos_curator.core.interfaces.stage_interface import CuratorStage, CuratorStageResource
from cosmos_curator.core.utils.config.config import maybe_load_config, resolve_model_name_auto
from cosmos_curator.core.utils.infra.performance_utils import StageTimer
from cosmos_curator.core.utils.model import pixi_utils
from cosmos_curator.pipelines.common.openai_embedding_utils import call_openai_embedding_api, frame_to_base64_jpeg
from cosmos_curator.pipelines.video.utils.data_model import (
    Clip,
    SplitPipeTask,
)
from cosmos_curator.pipelines.video.utils.decoder_utils import (
    FrameExtractionPolicy,
    FrameExtractionSignature,
)

if TYPE_CHECKING:
    import openai


if pixi_utils.is_running_in_env("unified"):
    import openai


class OpenAIEmbeddingStage(CuratorStage):
    """Generate video clip embeddings using an OpenAI-compatible embedding API.

    Reads pre-extracted frames from ``clip.extracted_frames``, encodes each
    frame as a base64 JPEG image, and sends them as multiple ``image_url``
    entries to a remote OpenAI-compatible endpoint (e.g. vLLM serving an
    embedding model at ``/v1/embeddings``).
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        model_name: str,
        target_fps: float = 2.0,
        max_retries: int = 3,
        retry_delay_seconds: float = 1.0,
        max_concurrent_requests: int = 8,
        verbose: bool = False,
        log_stats: bool = False,
    ) -> None:
        """Initialize the OpenAI-compatible API embedding stage.

        Args:
            model_name: Model name to pass in the API request.
            target_fps: Target FPS for frame extraction signature lookup.
            max_retries: Number of retries per clip before giving up.
            retry_delay_seconds: Delay between retries.
            max_concurrent_requests: Maximum number of concurrent API requests
                to the embedding endpoint.  Higher values let vLLM's
                continuous-batching scheduler work more efficiently.
            verbose: Emit verbose logging.
            log_stats: Whether to record stage performance statistics.

        """
        super().__init__()
        self._timer = StageTimer(self)
        self._model_name = model_name
        self._max_retries = max_retries
        self._retry_delay_seconds = retry_delay_seconds
        self._max_concurrent_requests = max_concurrent_requests
        self._verbose = verbose
        self._log_stats = log_stats
        self._client: openai.OpenAI | None = None
        self._frame_extraction_signature = FrameExtractionSignature(
            extraction_policy=FrameExtractionPolicy.sequence,
            target_fps=target_fps,
        ).to_str()

    @property
    def resources(self) -> CuratorStageResource:
        """Get the resource requirements for this stage."""
        return CuratorStageResource(cpus=1.0)

    @property
    def conda_env_name(self) -> str:
        """Use the unified environment (openai package lives there)."""
        return "unified"

    def stage_setup(self) -> None:
        """Create the OpenAI API client using credentials from the config file."""
        config = maybe_load_config()
        endpoint = config.openai.embedding if config is not None and config.openai is not None else None
        if endpoint is None or not endpoint.api_key:
            error_msg = (
                "OpenAI embedding configuration not found. "
                "Provide openai.embedding.api_key in ~/.config/cosmos_curator/config.yaml"
            )
            raise RuntimeError(error_msg)

        client_kwargs: dict[str, Any] = {"api_key": endpoint.api_key}
        if endpoint.base_url:
            client_kwargs["base_url"] = endpoint.base_url
        self._client = openai.OpenAI(**client_kwargs)
        self._model_name = resolve_model_name_auto(self._client, self._model_name, endpoint_label="OpenAI embedding")

    def _generate_embedding(self, frames: npt.NDArray[np.uint8]) -> npt.NDArray[np.float32]:
        """Generate an embedding from extracted video frames with retry logic.

        Encodes each frame as a base64 JPEG and sends them as multiple
        ``image_url`` content parts using vLLM's chat-embedding extension
        of the ``/v1/embeddings`` endpoint.
        """
        client = self._client
        if client is None:
            msg = "OpenAI client not initialized; call stage_setup before generating embeddings."
            raise RuntimeError(msg)

        content_parts: list[dict[str, Any]] = [
            {"type": "image_url", "image_url": {"url": frame_to_base64_jpeg(frame)}} for frame in frames
        ]
        return call_openai_embedding_api(
            client,
            self._model_name,
            content_parts,
            max_retries=self._max_retries,
            retry_delay_seconds=self._retry_delay_seconds,
        )

    def _embed_clip(self, clip: Clip) -> None:
        """Embed a single clip — designed to run inside a thread pool."""
        ef = clip.extracted_frames.resolve()
        if ef is None or self._frame_extraction_signature not in ef:
            clip.errors["openai_embedding"] = "extracted frames missing"
            logger.error(f"Clip {clip.uuid} has no extracted frames for {self._frame_extraction_signature}")
            return
        frames = ef[self._frame_extraction_signature]
        try:
            embedding = self._generate_embedding(frames)
        except Exception as exc:  # noqa: BLE001
            clip.errors["openai_embedding"] = str(exc)
            if self._verbose:
                logger.exception(f"OpenAI API embedding failed for clip {clip.uuid}")
            else:
                logger.warning(f"OpenAI API embedding failed for clip {clip.uuid}: {exc}")
            return
        clip.openai_embedding = embedding
        if self._verbose:
            logger.info(f"OpenAI API embedding clip {clip.uuid}: sent {len(frames)} frames, shape={embedding.shape}")
        clip.extracted_frames.drop()

    @nvtx.annotate("OpenAIEmbeddingStage")  # type: ignore[untyped-decorator]
    def process_data(self, tasks: list[SplitPipeTask]) -> list[SplitPipeTask]:
        """Generate embeddings for each clip using pre-extracted frames sent to the API."""
        for task in tasks:
            self._timer.reinit(self, task.get_major_size())
            video = task.video
            with (
                self._timer.time_process(len(video.clips)),
                concurrent.futures.ThreadPoolExecutor(
                    max_workers=self._max_concurrent_requests,
                ) as pool,
            ):
                list(pool.map(self._embed_clip, video.clips))

            if self._log_stats:
                stage_name, stage_perf_stats = self._timer.log_stats()
                task.stage_perf[stage_name] = stage_perf_stats

        return tasks
