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

"""Data model for image pipelines."""

import pathlib
from typing import Any

import attrs
import numpy as np
import numpy.typing as npt

from cosmos_curator.core.interfaces.stage_interface import PipelineTask
from cosmos_curator.core.sensors.data.image_data import ImageData
from cosmos_curator.core.utils.data.lazy_data import LazyData
from cosmos_curator.core.utils.infra.performance_utils import StagePerfStats
from cosmos_curator.core.utils.storage import storage_client
from cosmos_curator.pipelines.video.utils.data_model import CaptionFailureReason, TokenCounts


@attrs.define
class Image:
    """Container for image content and processing results.

    Stores source path, optional encoded bytes (for Ray transport),
    and errors. Image-native; not a 1-frame video.
    """

    input_image: storage_client.StoragePrefix | pathlib.Path
    """Source path (local or S3) for the image."""

    relative_path: str = ""
    """Path relative to input root; used to preserve directory structure in output."""

    encoded_data: LazyData[npt.NDArray[np.uint8]] = attrs.field(
        factory=LazyData,
        converter=LazyData.coerce,  # type: ignore[misc]
    )
    """Raw image bytes as numpy array; LazyData for zero-copy Ray transport."""

    image_data: ImageData | None = None
    """Decoded image samples aligned to the sensor sampling grid."""

    errors: dict[str, str] = attrs.Factory(dict)
    """Per-stage error messages (e.g. 'download', 'write')."""

    model_input: dict[str, Any] = attrs.Factory(dict)
    """Model inputs keyed by variant (e.g. 'qwen'); populated by caption prep stage."""

    caption: str = ""
    """Primary caption text from the most recent caption stage; kept for compatibility."""

    captions: dict[str, str] = attrs.Factory(dict)
    """Normalized captions keyed by model variant, mirroring the video pipeline."""

    filter_captions: dict[str, str] = attrs.Factory(dict)
    """Filter/classifier captions keyed by model variant for CPU postprocessing stages."""

    token_counts: dict[str, TokenCounts] = attrs.Factory(dict)
    """Per-model token usage from caption generation."""

    caption_status: str | None = None
    """Normalized caption outcome (success/truncated/blocked/error/skipped)."""

    caption_failure_reason: CaptionFailureReason | None = None
    """Failure reason populated when caption_status == 'error'."""

    filter_caption_status: dict[str, str] = attrs.Factory(dict)
    """Per-model filter/classifier caption outcome (success/truncated/blocked/error/skipped)."""

    filter_caption_failure_reason: dict[str, CaptionFailureReason | None] = attrs.Factory(dict)
    """Per-model failure reason populated when a filter/classifier caption errors."""

    qwen_type_classification: list[str] | None = None
    """Classifier labels inferred from local filter/classifier caption postprocessing."""

    qwen_rejection_stage: str | None = None
    """Which stage rejected the image: 'semantic' or 'classifier'."""

    qwen_rejection_reasons: dict[str, str] | None = None
    """Accumulated rejection reasons from semantic filter and/or classifier postprocessing."""

    is_filtered: bool = False
    """Whether the image was filtered out by semantic or classifier postprocessing."""

    embeddings: dict[str, npt.NDArray[np.float32]] = attrs.Factory(dict)
    """Embeddings keyed by model key (e.g. 'clip', 'cosmos_embed1_336p', 'internvideo2', 'openai')."""

    intern_video_2_text_match: tuple[str, float] | None = None
    """Best text-match result for InternVideo2 embedding verification, when requested by tests."""

    cosmos_embed1_text_match: tuple[str, float] | None = None
    """Best text-match result for Cosmos-Embed1 embedding verification, when requested by tests."""

    width: int | None = None
    """Image width (e.g. after resize in caption prep); None if not set."""

    height: int | None = None
    """Image height (e.g. after resize in caption prep); None if not set."""

    def get_major_size(self) -> int:
        """Return size in bytes of encoded_data for performance accounting."""
        return self.encoded_data.nbytes

    def has_caption(self) -> bool:
        """Whether this image has a usable normalized caption result."""
        return self.caption_status in {"success", "truncated"}


@attrs.define
class ImagePipeTask(PipelineTask):
    """Pipeline task for a single image (load → write, later filter/caption/embed).

    One task = one image. session_id identifies the image for logging and summary.
    """

    session_id: str
    """Identifier for this task (e.g. str(image.input_image) for logging)."""

    image: Image
    """The image payload."""

    stage_perf: dict[str, StagePerfStats] = attrs.Factory(dict)
    """Per-stage performance stats (e.g. from StageTimer.log_stats)."""

    def get_major_size(self) -> int:
        """Return total size in bytes of the task for performance accounting."""
        return self.image.get_major_size()
