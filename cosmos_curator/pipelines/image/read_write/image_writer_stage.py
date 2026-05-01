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

"""Image writer stage: write image bytes and minimal metadata to output path."""

import hashlib
import io
import json

import numpy as np
import nvtx  # type: ignore[import-untyped]
from loguru import logger

from cosmos_curator.core.interfaces.stage_interface import CuratorStage, CuratorStageResource
from cosmos_curator.core.utils.infra.performance_utils import StageTimer
from cosmos_curator.core.utils.storage.storage_utils import StorageWriter
from cosmos_curator.pipelines.image.utils.data_model import ImagePipeTask


def get_image_output_id(session_id: str) -> str:
    """Stable id for output filenames (e.g. images/{id}.jpg, metas/{id}.json)."""
    return hashlib.sha256(session_id.encode()).hexdigest()[:16]


def _output_extension(relative_path: str) -> str:
    """File extension for the written image (preserve original or default .jpg)."""
    if not relative_path:
        return ".jpg"
    # take last suffix (e.g. .tar.gz -> .gz; .jpg -> .jpg)
    idx = relative_path.rfind(".")
    return relative_path[idx:] if idx >= 0 else ".jpg"


def _first_align_timestamp_ns(task: ImagePipeTask) -> int | None:
    """Return the first sampled align timestamp from ``image_data`` when available."""
    image_data = task.image.image_data
    if image_data is None or len(image_data.align_timestamps_ns) == 0:
        return None
    return int(image_data.align_timestamps_ns[0])


def _first_sensor_timestamp_ns(task: ImagePipeTask) -> int | None:
    """Return the first sampled sensor timestamp from ``image_data`` when available."""
    image_data = task.image.image_data
    if image_data is None or len(image_data.sensor_timestamps_ns) == 0:
        return None
    return int(image_data.sensor_timestamps_ns[0])


class ImageWriterStage(CuratorStage):
    """Stage that writes each task's image and minimal metadata to output_path."""

    def __init__(
        self,
        output_path: str,
        output_s3_profile_name: str,
        *,
        verbose: bool = False,
        log_stats: bool = False,
    ) -> None:
        """Initialize the image writer stage.

        Args:
            output_path: Base path for output (images/, metas/ under this).
            output_s3_profile_name: S3 profile for remote output.
            verbose: Whether to log per-image write.
            log_stats: Whether to record stage performance in task.stage_perf.

        """
        self._timer = StageTimer(self)
        self._output_path = output_path
        self._output_s3_profile_name = output_s3_profile_name
        self._verbose = verbose
        self._log_stats = log_stats

    @property
    def resources(self) -> CuratorStageResource:
        """Resource configuration for this stage."""
        return CuratorStageResource(cpus=0.25)

    @staticmethod
    def _build_metadata(task: ImagePipeTask) -> dict[str, object]:
        """Build metadata JSON payload for an image task."""
        image = task.image
        meta: dict[str, object] = {
            "source_path": str(image.input_image),
            "relative_path": image.relative_path,
            "width": image.width,
            "height": image.height,
            "has_caption": image.has_caption(),
            "is_filtered": image.is_filtered,
            "align_timestamp_ns": _first_align_timestamp_ns(task),
            "sensor_timestamp_ns": _first_sensor_timestamp_ns(task),
            "caption_status": image.caption_status,
            "caption_failure_reason": image.caption_failure_reason,
            "token_counts": {
                model: {
                    "prompt_tokens": counts.prompt_tokens,
                    "output_tokens": counts.output_tokens,
                }
                for model, counts in image.token_counts.items()
            },
        }
        if image.caption.strip():
            meta["caption"] = image.caption
        if image.filter_caption_status:
            meta["filter_caption_status"] = image.filter_caption_status
        filter_failures = {
            model: reason for model, reason in image.filter_caption_failure_reason.items() if reason is not None
        }
        if filter_failures:
            meta["filter_caption_failure_reason"] = filter_failures
        if image.qwen_type_classification is not None:
            meta["qwen_type_classification"] = image.qwen_type_classification
        if image.qwen_rejection_stage is not None:
            meta["qwen_rejection_stage"] = image.qwen_rejection_stage
        if image.qwen_rejection_reasons:
            meta["qwen_rejection_reasons"] = image.qwen_rejection_reasons
        if image.embeddings:
            meta["embedding_keys"] = list(image.embeddings.keys())
        if image.errors:
            meta["errors"] = dict(image.errors)
        return meta

    @nvtx.annotate("ImageWriterStage")  # type: ignore[untyped-decorator]
    def process_data(self, tasks: list[ImagePipeTask]) -> list[ImagePipeTask] | None:
        """Write each task's image to output_path/images/{id}{ext} and metadata to metas/{id}.json."""
        writer = StorageWriter(
            self._output_path,
            profile_name=self._output_s3_profile_name,
        )
        for task in tasks:
            self._timer.reinit(self, task.get_major_size())
            image = task.image
            data = image.encoded_data.resolve()
            if data is None or data.nbytes == 0:
                image.errors["write"] = "no data"
                logger.warning(f"Skip write for {task.session_id}: no encoded_data")
                continue
            with self._timer.time_process():
                out_id = get_image_output_id(task.session_id)
                ext = _output_extension(image.relative_path)
                image_dir = "filtered_images" if image.is_filtered else "images"
                image_sub = f"{image_dir}/{out_id}{ext}"
                meta_sub = f"metas/{out_id}.json"
                writer.write_bytes_to(image_sub, data.tobytes())
                meta = self._build_metadata(task)
                writer.write_str_to(meta_sub, json.dumps(meta, indent=2))
                for model_key, embedding in image.embeddings.items():
                    npy_buf = io.BytesIO()
                    np.save(npy_buf, embedding)
                    writer.write_bytes_to(f"embeddings/{model_key}/{out_id}.npy", npy_buf.getvalue())
                if self._verbose:
                    logger.info(f"Wrote image {task.session_id} -> {image_sub}, {meta_sub}")
            if self._log_stats:
                stage_name, stage_perf_stats = self._timer.log_stats()
                task.stage_perf[stage_name] = stage_perf_stats
        return tasks
