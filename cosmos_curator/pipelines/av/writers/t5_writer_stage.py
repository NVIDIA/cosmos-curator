# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
"""Module for writing T5 embeddings to storage.

This module provides functionality for writing T5 embeddings to storage (local or S3)
in pickle format, with support for different prompt variants and environments.
"""

import io
import pathlib
import pickle
import uuid

from loguru import logger

from cosmos_curator.core.interfaces.stage_interface import CuratorStage
from cosmos_curator.core.utils.infra.performance_utils import StageTimer
from cosmos_curator.core.utils.storage import s3_client
from cosmos_curator.core.utils.storage.s3_client import is_s3path
from cosmos_curator.core.utils.storage.writer_utils import write_bytes
from cosmos_curator.pipelines.av.captioning.captioning_stages import (
    PROMPT_TYPES_FOR_T5_EMBEDDING,
)
from cosmos_curator.pipelines.av.utils.av_data_model import (
    AvClipAnnotationTask,
    ClipForAnnotation,
)


def _get_t5_embedding_url(  # noqa: PLR0913
    clip_uuid: uuid.UUID,
    prefix: str,
    env: str | None,
    clip_prefix: str,
    prompt_variant: str,
    version: str,
) -> s3_client.S3Prefix | pathlib.Path:
    """Generate a URL or path for storing T5 embedding data.

    Args:
        clip_uuid: UUID of the clip
        prefix: Base path for output
        env: Environment name (e.g., "prod", "dev", "test", "local") or None
        clip_prefix: Prefix for clip files
        prompt_variant: Prompt variant used for embeddings
        version: Version identifier

    Returns:
        An S3Prefix object if the prefix is an S3 path, otherwise a
        pathlib.Path object

    """
    # Only support one prompt type for now
    if env is None:
        full_path = f"{prefix}/{clip_prefix}/{prompt_variant}/{clip_uuid}.bin"
    else:
        full_path = f"{prefix}/{env}/{clip_prefix}/{prompt_variant}/{version}/{clip_uuid}.bin"
    if is_s3path(prefix):
        return s3_client.S3Prefix(full_path)
    return pathlib.Path(full_path)


def _write_t5_embeddings(  # noqa: PLR0913
    s3_client: s3_client.S3Client | None,
    clips: list[ClipForAnnotation],
    output_prefix: str,
    clip_prefix: str,
    env: str | None,
    version: str,
    prompt_variant: str,
    verbose: bool,  # noqa: FBT001
) -> int:
    """Write T5 embeddings for clips to storage.

    Args:
        s3_client: Client for S3 operations, or None for local storage
        clips: List of clips to write embeddings for
        output_prefix: Base path for output
        clip_prefix: Prefix for clip files
        env: Environment name (e.g., "prod", "dev", "test", "local") or None
        version: Version identifier
        prompt_variant: Prompt variant used for embeddings
        verbose: Whether to log verbose output

    Returns:
        Number of clips successfully processed

    """
    num_uploaded_clips = 0
    for clip in clips:
        embeddings = [window.t5_xxl_embeddings.get(prompt_variant, None) for window in clip.caption_windows]
        if any(x is None for x in embeddings):
            logger.error(f"Clip {clip.uuid} has no t5 embedding")
            continue
        dest = _get_t5_embedding_url(clip.uuid, output_prefix, env, clip_prefix, prompt_variant, version)
        clip.t5_xxl_embedding_urls[prompt_variant] = str(dest)
        buffer = io.BytesIO()
        pickle.dump(embeddings, buffer)
        write_bytes(
            buffer.getvalue(),
            dest,
            f"clip-{clip.uuid}",
            "unknown",
            verbose=verbose,
            client=s3_client,
            overwrite=True,
        )
        num_uploaded_clips += 1
    return num_uploaded_clips


class T5WriterStage(CuratorStage):
    """Stage for writing T5 embeddings to storage.

    This stage handles writing T5 embeddings to storage (local or S3) in pickle format.
    It supports multiple prompt variants and different environments.
    """

    def __init__(  # noqa: PLR0913
        self,
        env: str | None,
        output_prefix: str,
        run_id: uuid.UUID,
        version: str,
        prompt_variants: list[str] | None = None,
        verbose: bool = False,  # noqa: FBT001, FBT002
        log_stats: bool = False,  # noqa: FBT001, FBT002
    ) -> None:
        """Initialize a stage that writes T5 embeddings to storage.

        Args:
            env: Environment name (e.g., "prod", "dev", "test", "local") or None
            output_prefix: Base path for output files
            run_id: Unique identifier for this run
            version: Version string
            prompt_variants: List of prompt variants to process
            verbose: Whether to enable verbose logging
            log_stats: Whether to log performance statistics

        """
        self._timer = StageTimer(self)
        self._env = env
        self._output_prefix = output_prefix.rstrip("/")
        self._run_id = run_id
        self._version = version
        self._prompt_variants = ["default"] if prompt_variants is None else prompt_variants
        self._verbose = verbose
        self._log_stats = log_stats
        self._clip_prefix = "t5_embeddings"

    def stage_setup(self) -> None:
        """Set up S3 client for writing embeddings."""
        super().stage_setup()
        self._s3_client = s3_client.create_s3_client(
            target_path=self._output_prefix,
            can_overwrite=True,
        )

    def process_data(  # type: ignore[override]
        self, tasks: list[AvClipAnnotationTask]
    ) -> list[AvClipAnnotationTask] | None:
        """Process and write T5 embeddings with performance tracking.

        This method writes T5 embeddings for each supported prompt variant to storage.
        After processing, the embeddings are cleared from memory.

        Args:
            tasks: Tasks containing clips with T5 embeddings to process

        Returns:
            List containing the input task if successful

        Raises:
            Exception: If storage write fails

        """
        return [self._process_data(task) for task in tasks]

    def _process_data(self, task: AvClipAnnotationTask) -> AvClipAnnotationTask:
        self._timer.reinit(self, task.get_major_size())
        with self._timer.time_process():
            for prompt_variant in self._prompt_variants:
                if prompt_variant not in PROMPT_TYPES_FOR_T5_EMBEDDING:
                    continue
                try:
                    num_t5 = _write_t5_embeddings(
                        self._s3_client,
                        task.clips,
                        self._output_prefix,
                        self._clip_prefix,
                        self._env,
                        self._version,
                        prompt_variant,
                        self._verbose,
                    )
                    logger.info(f"Uploaded {num_t5} sets of T5 embeddings")
                except Exception as e:  # noqa: BLE001
                    logger.error(f"Failed to write T5 embeddings {e!s}")

            # Clear embeddings from memory
            for clip in task.clips:
                for window in clip.caption_windows:
                    window.t5_xxl_embeddings.clear()

        if self._log_stats:
            stage_name, stage_perf_stats = self._timer.log_stats()
            task.stage_perf[stage_name] = stage_perf_stats
        return task
