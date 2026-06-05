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
"""Write metadata for clips to DB."""

import base64
import io
import json
import pathlib
import pickle
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import lance
import numpy as np
import numpy.typing as npt
import pandas as pd
import pyarrow as pa  # type: ignore[import-untyped]
from azure.core.exceptions import AzureError, ResourceNotFoundError
from botocore.exceptions import ClientError
from loguru import logger

from cosmos_curator.core.interfaces.stage_interface import CuratorStage, CuratorStageResource
from cosmos_curator.core.utils.data.ref_resolver import batch_resolve
from cosmos_curator.core.utils.infra.performance_utils import StageTimer
from cosmos_curator.core.utils.storage import storage_client, storage_utils
from cosmos_curator.core.utils.storage.storage_utils import (
    get_files_relative,
    get_full_path,
    read_json_file,
)
from cosmos_curator.core.utils.storage.writer_utils import (
    write_bytes,
    write_json,
    write_jsonl,
    write_lance_fragments,
    write_parquet,
    write_text,
)
from cosmos_curator.models.vllm_sentinels import VLLM_UNKNOWN_CAPTION
from cosmos_curator.pipelines.video.tracking.serialization import (
    sam3_events_envelope,
    sam3_instances_envelope,
    sam3_objects_envelope,
)
from cosmos_curator.pipelines.video.utils.data_model import (
    CAPTION_OK_STATUSES,
    CAPTION_QUALITY_FLAG_KEYS,
    CaptionOutcome,
    CaptionQualityStats,
    Clip,
    ClipStats,
    SplitPipeTask,
    Video,
    VideoMetadata,
    Window,
)


class ClipWriterStage(CuratorStage):
    """Stage that writes clips and metadata for clip transcoding.

    This class processes video clips through a series of steps including embedding generation,
    metadata extraction, and writing to storage.
    """

    def __init__(  # noqa: PLR0913
        self,
        output_path: str,
        input_path: str,
        output_s3_profile_name: str,
        *,
        upload_clips: bool,
        upload_clip_info_in_chunks: bool,
        upload_clip_info_in_lance: bool,
        upload_cds_parquet: bool,
        dry_run: bool,
        generate_embeddings: bool,
        embedding_algorithm: str,
        embedding_model_version: str,
        generate_previews: bool,
        caption_models: list[str] | None = None,
        enhanced_caption_models: list[str] | None = None,
        caption_quality_stats_enabled: bool = False,
        caption_quality_flags_enabled: bool = True,
        generate_cosmos_predict_dataset: bool = False,
        verbose: bool = False,
        log_stats: bool = False,
    ) -> None:
        """Construct stage that writes clips and metadata for clip transcoding."""
        if caption_models is None:
            caption_models = ["qwen"]
        if enhanced_caption_models is None:
            enhanced_caption_models = []
        self._timer = StageTimer(self)
        self._set_input_path(input_path)
        self._set_output_path(output_path)
        self._output_s3_profile_name = output_s3_profile_name
        self._upload_clips = upload_clips
        self._upload_clip_info_in_chunks = upload_clip_info_in_chunks
        self._upload_clip_info_in_lance = upload_clip_info_in_lance
        self._emit_per_clip_metadata = not (self._upload_clip_info_in_chunks or self._upload_clip_info_in_lance)
        self._emit_jsonl_metadata = self._upload_clip_info_in_chunks
        self._emit_lance_metadata = self._upload_clip_info_in_lance
        self._upload_cds_parquet = upload_cds_parquet
        self._dry_run = dry_run
        self._generate_embeddings = generate_embeddings
        self._embedding_algorithm = embedding_algorithm
        self._embedding_model_version = embedding_model_version
        self._generate_previews = generate_previews
        self._caption_models = caption_models
        self._enhanced_caption_models = enhanced_caption_models
        self._caption_quality_stats_enabled = caption_quality_stats_enabled
        self._caption_quality_flags_enabled = caption_quality_flags_enabled
        self._generate_cosmos_predict_dataset = generate_cosmos_predict_dataset
        self._verbose = verbose
        self._log_stats = log_stats
        self._embedding_buffer: list[dict[str, Any]] = []
        self._metadata_buffer: list[dict[str, Any]] = []
        self._cds_data_buffer: list[dict[str, Any]] = []
        self._max_workers = 4
        self._lance_storage_options: dict[str, str] | None = None

    def stage_setup(self) -> None:
        """Initialize stage resources and configuration."""
        self._storage_client = storage_utils.get_storage_client(
            self._output_path,
            profile_name=self._output_s3_profile_name,
            can_overwrite=True,
        )
        self._lance_storage_options = storage_utils.get_lance_storage_options(
            self.get_output_path_meta_lance(self._output_path, "v0"),
            profile_name=self._output_s3_profile_name,
        )

    def _set_input_path(self, input_path: str) -> None:
        """Set the input path for the stage.

        Args:
            input_path: Path to input data.

        """
        self._input_path = input_path.rstrip("/") + "/"

    def _set_output_path(self, output_path: str) -> None:
        """Set the output path for the stage.

        Args:
            output_path: Path to write output data.

        """
        self._output_path = output_path.rstrip("/") + "/"

    @property
    def resources(self) -> CuratorStageResource:
        """Get the resource requirements for this stage.

        Returns:
            Resource configuration for the stage.

        """
        return CuratorStageResource(cpus=0.25)

    @staticmethod
    def _get_output_path(output_path: str, extra: str) -> str:
        return output_path.rstrip("/") + "/" + extra.strip("/")

    @staticmethod
    def get_output_path_processed_videos(output_path: str) -> str:
        """Get path to store processed videos."""
        return ClipWriterStage._get_output_path(output_path, "processed_videos")

    @staticmethod
    def get_output_path_processed_clip_chunks(
        output_path: str,
    ) -> str:
        """Get path to store processed clip chunks."""
        return ClipWriterStage._get_output_path(output_path, "processed_clip_chunks")

    @staticmethod
    def get_output_path_video_errors(
        output_path: str,
    ) -> str:
        """Get path to store video errors."""
        return ClipWriterStage._get_output_path(output_path, "video_errors")

    @staticmethod
    def get_output_path_clips(output_path: str, *, filtered: bool = False) -> str:
        """Get path to store generated clips."""
        directory = "filtered_clips" if filtered else "clips"
        return ClipWriterStage._get_output_path(output_path, directory)

    @staticmethod
    def get_output_path_previews(output_path: str) -> str:
        """Get path to store generated clips."""
        return ClipWriterStage._get_output_path(output_path, "previews")

    @staticmethod
    def get_output_path_metas(output_path: str, version: str) -> str:
        """Get path to store clip metadatas."""
        return ClipWriterStage._get_output_path(output_path, f"metas/{version}")

    @staticmethod
    def get_output_path_meta_jsonls(output_path: str, version: str) -> str:
        """Get path to store clip metadata in a jsonl file."""
        return ClipWriterStage._get_output_path(output_path, f"metas_jsonl/{version}")

    @staticmethod
    def get_output_path_meta_lance(output_path: str, version: str) -> str:
        """Get path to store clip metadata in a Lance dataset."""
        return ClipWriterStage._get_output_path(output_path, f"lance/{version}")

    @staticmethod
    def get_output_path_meta_lance_fragments(output_path: str, version: str) -> str:
        """Get path to store Lance fragment sidecars."""
        return ClipWriterStage._get_output_path(output_path, f"lance_fragments/{version}")

    @staticmethod
    def get_output_path_meta_lance_fragments_processed(output_path: str, version: str) -> str:
        """Get path to store processed Lance fragment sidecars."""
        return ClipWriterStage._get_output_path(output_path, f"processed_lance_fragments/{version}")

    @staticmethod
    def _embd_stem(embedding_algorithm: str) -> str:
        if embedding_algorithm == "internvideo2":
            return "iv2_embd"
        if embedding_algorithm.startswith("cosmos-embed1-"):
            variant = embedding_algorithm.removeprefix("cosmos-embed1-")
            return f"ce1_embd_{variant}"
        if embedding_algorithm == "openai":
            return "openai_embd"
        logger.error(f"Unknown embedding algorithm: {embedding_algorithm}")
        return f"{embedding_algorithm}_embd"

    @staticmethod
    def get_output_path_embds(output_path: str, embedding_algorithm: str) -> str:
        """Get path to store generated clips."""
        return ClipWriterStage._get_output_path(output_path, ClipWriterStage._embd_stem(embedding_algorithm))

    @staticmethod
    def get_output_path_embd_parquets(output_path: str, embedding_algorithm: str) -> str:
        """Get path to store generated clip embeddings in a parquet file."""
        return ClipWriterStage._get_output_path(
            output_path, f"{ClipWriterStage._embd_stem(embedding_algorithm)}_parquet"
        )

    @staticmethod
    def get_output_path_embd_lance(output_path: str, embedding_algorithm: str) -> str:
        """Get path to store generated clip embeddings in a Lance dataset."""
        return ClipWriterStage._get_output_path(output_path, f"{ClipWriterStage._embd_stem(embedding_algorithm)}_lance")

    @staticmethod
    def get_output_path_embd_lance_fragments(output_path: str, embedding_algorithm: str) -> str:
        """Get path to store staged Lance fragments for embeddings."""
        return ClipWriterStage._get_output_path(
            output_path, f"{ClipWriterStage._embd_stem(embedding_algorithm)}_lance_fragments"
        )

    @staticmethod
    def get_output_path_embd_lance_fragments_processed(output_path: str, embedding_algorithm: str) -> str:
        """Get path to store processed Lance fragment sidecars for embeddings."""
        return ClipWriterStage._get_output_path(
            output_path, f"{ClipWriterStage._embd_stem(embedding_algorithm)}_lance_fragments_processed"
        )

    @staticmethod
    def get_output_path_cds_parquets(output_path: str) -> str:
        """Get path to store generated clips in a parquet file."""
        return ClipWriterStage._get_output_path(output_path, "cds_parquet")

    @staticmethod
    def get_output_path_sam3_instances(output_path: str) -> str:
        """Get path to store per-clip SAM3 ``instances.json`` files."""
        return ClipWriterStage._get_output_path(output_path, "sam3_instances")

    @staticmethod
    def get_output_path_sam3_objects(output_path: str) -> str:
        """Get path to store per-clip SAM3 ``objects.json`` files."""
        return ClipWriterStage._get_output_path(output_path, "sam3_objects")

    @staticmethod
    def get_output_path_sam3_events(output_path: str) -> str:
        """Get path to store per-clip SAM3 ``events.json`` files."""
        return ClipWriterStage._get_output_path(output_path, "sam3_events")

    @staticmethod
    def get_output_path_sam3_tracked(output_path: str) -> str:
        """Get path to store per-clip SAM3 annotated ``tracked.mp4`` files."""
        return ClipWriterStage._get_output_path(output_path, "sam3_tracked")

    @staticmethod
    def get_video_uuid(input_video_path: str) -> uuid.UUID:
        """Get a UUID for the video based on its input path."""
        return uuid.uuid5(uuid.NAMESPACE_URL, f"{input_video_path}")

    @staticmethod
    def get_output_path_cosmos_predict_dataset(output_path: str) -> str:
        """Get path to store generated cosmos predict dataset."""
        return ClipWriterStage._get_output_path(output_path, "cosmos_predict2_video2world_dataset")

    @staticmethod
    def get_output_path_per_window_clips(output_path: str) -> str:
        """Get path to store per-window clips."""
        return ClipWriterStage._get_output_path(
            ClipWriterStage.get_output_path_cosmos_predict_dataset(output_path),
            "videos",
        )

    @staticmethod
    def get_output_path_per_window_metas(output_path: str) -> str:
        """Get path to store per-window clip metadatas."""
        return ClipWriterStage._get_output_path(
            ClipWriterStage.get_output_path_cosmos_predict_dataset(output_path),
            "metas",
        )

    @staticmethod
    def get_output_path_per_window_t5_embeds(output_path: str) -> str:
        """Get path to store per-window T5 embeddings."""
        return ClipWriterStage._get_output_path(
            ClipWriterStage.get_output_path_cosmos_predict_dataset(output_path),
            "t5_xxl",
        )

    @staticmethod
    def get_grouped_clips_uri(
        video_uuid: uuid.UUID,
        chunk_index: int,
        path_prefix: str,
        file_type: str,
    ) -> storage_client.StoragePrefix | pathlib.Path:
        """Get URI for grouped clips data (embeddings/metadata) for a chunk of clips from a video."""
        output_clip_chunk_file = f"{video_uuid}_{chunk_index}.{file_type}"
        return get_full_path(path_prefix, output_clip_chunk_file)

    def _write_data(
        self,
        buffer: bytes | npt.NDArray[np.uint8],
        dest: storage_client.StoragePrefix | pathlib.Path,
        desc: str,
        source_video: str,
    ) -> None:
        write_bytes(buffer, dest, desc, source_video, verbose=self._verbose, client=self._storage_client)

    def _write_json_data(
        self,
        data: dict,  # type: ignore[type-arg]
        dest: storage_client.StoragePrefix | pathlib.Path,
        desc: str,
        source_video: str,
    ) -> None:
        write_json(data, dest, desc, source_video, verbose=self._verbose, client=self._storage_client)

    def _write_text_data(
        self,
        text: str,
        dest: storage_client.StoragePrefix | pathlib.Path,
        desc: str,
        source_video: str,
    ) -> None:
        write_text(text, dest, desc, source_video, verbose=self._verbose, client=self._storage_client)

    def _process_video(self, video: Video, *, is_primary: bool = True) -> None:  # noqa: C901, PLR0912
        """Process a video and write clips/metadata.

        For multi-cam tasks, per-clip metadata and embedding paths use shared UUIDs
        across cameras. Only the primary camera (index 0) should write these to
        avoid secondary cameras overwriting primary metadata. MP4 writes use
        relative_path and remain per-camera.
        """
        batch_resolve(
            [clip.encoded_data for clip in video.clips] + [clip.encoded_data for clip in video.filtered_clips]
        )
        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            with self._timer.time_process(len(video.clips)):
                # collect all embeddings/metadatas for a chunk of clips from a video
                # (primary-only: metadata/embeddings use shared UUIDs; secondary would overwrite)
                for clip in video.clips:
                    if is_primary:
                        self._add_clip_embedding_to_buffer(clip)
                        self._add_clip_metadata_to_buffer(clip, video.metadata)
                        self._add_cds_data_to_buffer(clip)

                # schedule all clip-level writes for a chunk of clips from a single video and wait
                futures_clips = []
                futures_clips += [
                    executor.submit(self._write_clip_mp4, clip, video.relative_path) for clip in video.clips
                ]
                if is_primary:
                    futures_clips += [executor.submit(self._write_clip_window_webp, clip) for clip in video.clips]
                    futures_clips += [executor.submit(self._write_clip_embedding, clip) for clip in video.clips]
                    # SAM3BBoxStage sets ``sam3_instances`` on every clip it processes (even when the list
                    # is empty), so ``is not None`` is the canonical "did SAM3 run?" signal.
                    if any(clip.sam3_instances is not None for clip in video.clips):
                        futures_clips += [executor.submit(self._write_clip_sam3, clip) for clip in video.clips]
                    futures_clips += [
                        executor.submit(self._write_clip_metadata, clip, video.metadata) for clip in video.clips
                    ]

                # filtered clips
                futures_clips += [
                    executor.submit(self._write_clip_mp4, clip, video.relative_path, filtered=True)
                    for clip in video.filtered_clips
                ]
                if is_primary:
                    futures_clips += [
                        executor.submit(self._write_clip_metadata, clip, video.metadata, filtered=True)
                        for clip in video.filtered_clips
                    ]

                # wait for all clip-level tasks to finish and gather stats
                for future_c in futures_clips:
                    result = future_c.result()
                    if result is not None:
                        video.clip_stats.combine(result)

                # for cosmos-predictX (primary-only: uses shared UUID paths)
                futures_no_rt = []
                if is_primary:
                    futures_no_rt = [executor.submit(self._write_per_window_data, clip) for clip in video.clips]
                # write video-level metadata after all clip-level tasks are done
                futures_no_rt += [executor.submit(self._write_video_metadata, video)]
                metadata_rows = list(self._metadata_buffer)
                self._metadata_buffer.clear()
                # write buffered embeddings and metadata (primary-only: shared UUIDs)
                if is_primary:
                    futures_no_rt += [executor.submit(self._write_grouped_embeddings_to_parquet, video)]
                    futures_no_rt += [executor.submit(self._write_grouped_metadata, video, metadata_rows)]
                    futures_no_rt += [executor.submit(self._write_grouped_cds_data_to_parquet, video)]

                for future_n in futures_no_rt:
                    future_n.result()

            # clean up intermediate data
            for clip in video.clips + video.filtered_clips:
                clip.encoded_data.drop()
                clip.intern_video_2_embedding = None
                clip.cosmos_embed1_embedding = None
                clip.openai_embedding = None
                for window in clip.windows:
                    window.mp4_bytes.drop()
                    for model_variant in window.model_input:
                        del window.model_input[model_variant]
                    window.caption.clear()
                    window.token_counts.clear()
                    window.enhanced_caption.clear()
                    window.caption_status = None
                    window.caption_failure_reason = None
                    window.flag_length_outlier = None
                    window.flag_repetition = None
                    window.flag_near_duplicate = None
                    window.webp_bytes.drop()

    def process_data(self, tasks: list[SplitPipeTask]) -> list[SplitPipeTask] | None:  # type: ignore[override]
        """Save clip bytes and metadata."""
        for task in tasks:
            self._timer.reinit(self, task.get_major_size())
            for video_index, video in enumerate(task.videos):
                is_primary = video_index == 0
                self._process_video(video, is_primary=is_primary)

            if self._log_stats:
                stage_name, stage_perf_stats = self._timer.log_stats()
                task.stage_perf[stage_name] = stage_perf_stats

        return tasks

    def _write_grouped_embeddings_to_parquet(self, video: Video) -> None:
        if self._embedding_buffer and not self._dry_run:
            path = self.get_grouped_clips_uri(
                self.get_video_uuid(video.input_path),
                video.clip_chunk_index,
                self.get_output_path_embd_parquets(self._output_path, self._embedding_algorithm),
                "parquet",
            )
            pdf = pd.DataFrame(self._embedding_buffer)
            write_parquet(
                pdf,
                path,
                "embedding",
                video.input_path,
                verbose=self._verbose,
                client=self._storage_client,
                overwrite=True,
            )
        self._embedding_buffer.clear()

    def _write_grouped_metadata_to_jsonl(self, video: Video, metadata_rows: list[dict[str, Any]]) -> None:
        if metadata_rows and not self._dry_run and self._emit_jsonl_metadata:
            jsonl_rows = []
            for row in metadata_rows:
                row_copy = {k: v for k, v in row.items() if k != "embedding"}
                jsonl_rows.append(row_copy)
            path = self.get_grouped_clips_uri(
                self.get_video_uuid(video.input_path),
                video.clip_chunk_index,
                self.get_output_path_meta_jsonls(self._output_path, "v0"),
                "jsonl",
            )
            write_jsonl(
                jsonl_rows,
                path,
                "metadata",
                video.input_path,
                verbose=self._verbose,
                client=self._storage_client,
                overwrite=True,
            )

    def _write_grouped_metadata_to_lance(self, video: Video, metadata_rows: list[dict[str, Any]]) -> None:
        if not self._emit_lance_metadata or not metadata_rows or self._dry_run:
            return

        video_uuid = str(self.get_video_uuid(video.input_path))
        enriched_rows = [
            {
                **row,
                "video_uuid": video_uuid,
                "clip_chunk_index": video.clip_chunk_index,
            }
            for row in metadata_rows
        ]
        table = pa.Table.from_pylist(enriched_rows)

        fragment_staging = self.get_output_path_meta_lance(self._output_path, "v0")
        fragments_json, schema_b64 = write_lance_fragments(
            table,
            fragment_staging,
            schema=table.schema,
            storage_options=self._lance_storage_options,
            verbose=self._verbose,
        )
        sidecar_path = get_full_path(
            self.get_output_path_meta_lance_fragments(self._output_path, "v0"),
            f"{video_uuid}_{video.clip_chunk_index}.json",
        )
        sidecar_payload = {
            "fragments": fragments_json,
            "schema_b64": schema_b64,
            "video_uuid": video_uuid,
            "clip_chunk_index": video.clip_chunk_index,
        }
        write_json(
            sidecar_payload,
            sidecar_path,
            "lance fragments metadata",
            video.input_path,
            verbose=self._verbose,
            client=self._storage_client,
            overwrite=True,
        )

    def _write_grouped_metadata(self, video: Video, metadata_rows: list[dict[str, Any]]) -> None:
        if not metadata_rows:
            return
        self._write_grouped_metadata_to_jsonl(video, metadata_rows)
        if self._upload_clip_info_in_lance:
            self._write_grouped_metadata_to_lance(video, metadata_rows)

    def _write_grouped_cds_data_to_parquet(self, video: Video) -> None:
        if self._cds_data_buffer and not self._dry_run:
            path = self.get_grouped_clips_uri(
                self.get_video_uuid(video.input_path),
                video.clip_chunk_index,
                self.get_output_path_cds_parquets(self._output_path),
                "parquet",
            )
            pdf = pd.DataFrame(self._cds_data_buffer)
            write_parquet(
                pdf,
                path,
                "cds",
                video.input_path,
                verbose=self._verbose,
                client=self._storage_client,
                overwrite=True,
            )
        self._cds_data_buffer.clear()

    def _get_window_uri(
        self,
        video_span_uuid: uuid.UUID,
        window: tuple[int, int],
        path_prefix: str,
        file_type: str,
    ) -> storage_client.StoragePrefix | pathlib.Path:
        output_window_file = f"{window[0]}_{window[1]}.{file_type}"
        return get_full_path(path_prefix, str(video_span_uuid), output_window_file)

    def _get_cosmos_predict_uri(
        self,
        video_span_uuid: uuid.UUID,
        window: tuple[int, int],
        path_prefix: str,
        file_type: str,
    ) -> storage_client.StoragePrefix | pathlib.Path:
        output_file = f"{video_span_uuid}_{window[0]}_{window[1]}.{file_type}"
        return get_full_path(path_prefix, output_file)

    def _get_clip_uri(
        self,
        video_span_uuid: uuid.UUID,
        path_prefix: str,
        file_type: str,
        relative_path: str | None = None,
        filename_suffix: str | None = None,
    ) -> storage_client.StoragePrefix | pathlib.Path:
        suffix = filename_suffix or ""
        if relative_path:
            output_clip_file = f"{video_span_uuid}/{relative_path}{suffix}.{file_type}"
        else:
            output_clip_file = f"{video_span_uuid}{suffix}.{file_type}"
        return get_full_path(path_prefix, output_clip_file)

    def _get_video_uri(self, input_video_path: str) -> storage_client.StoragePrefix | pathlib.Path:
        assert input_video_path.startswith(self._input_path)
        video_metadata_path = input_video_path[len(self._input_path) :] + ".json"
        output_path_videos = self.get_output_path_processed_videos(self._output_path)
        return get_full_path(output_path_videos, video_metadata_path)

    def _get_clip_chunk_uri(self, input_video_path: str, idx: int) -> storage_client.StoragePrefix | pathlib.Path:
        assert input_video_path.startswith(self._input_path)
        clip_chunk_path = input_video_path[len(self._input_path) :] + f"_{idx}.json"
        output_path_videos = self.get_output_path_processed_clip_chunks(self._output_path)
        return get_full_path(output_path_videos, clip_chunk_path)

    def _get_video_error_uri(self, input_video_path: str, idx: int) -> storage_client.StoragePrefix | pathlib.Path:
        assert input_video_path.startswith(self._input_path)
        error_chunk_path = input_video_path[len(self._input_path) :] + f"_{idx}.json"
        output_path_videos = self.get_output_path_video_errors(self._output_path)
        return get_full_path(output_path_videos, error_chunk_path)

    def _write_clip_window_webp(self, clip: Clip) -> ClipStats:
        clip_stats = ClipStats()
        has_webp = False
        for window in clip.windows:
            webp_data = window.webp_bytes.resolve()
            if webp_data is not None:
                dest = self._get_window_uri(
                    clip.uuid,
                    (window.start_frame, window.end_frame),
                    self.get_output_path_previews(self._output_path),
                    "webp",
                )
                if not self._dry_run:
                    self._write_data(
                        webp_data,
                        dest,
                        f"webp {clip.uuid} {window.start_frame}_{window.end_frame}",
                        clip.source_video,
                    )
                has_webp = True
            elif self._generate_previews:
                logger.error(
                    f"Clip {clip.uuid} window [{window.start_frame}, {window.end_frame}] "
                    f"from {clip.source_video} has no webp, skip uploading to s3",
                )
        clip_stats.num_with_webp += 1 if has_webp else 0
        return clip_stats

    def _write_clip_mp4(
        self,
        clip: Clip,
        relative_path: str,
        *,
        filtered: bool = False,
    ) -> ClipStats:
        clip_stats = ClipStats()
        data = clip.encoded_data.resolve()
        if data is not None:
            dest = self._get_clip_uri(
                clip.uuid,
                self.get_output_path_clips(self._output_path, filtered=filtered),
                "mp4",
                relative_path,
            )
            if self._upload_clips and not self._dry_run:
                self._write_data(data, dest, f"clip {clip.uuid}", clip.source_video)
            clip_stats.num_transcoded += 1
        else:
            logger.warning(f"Clip {clip.uuid} from {clip.source_video} has no buffer, skip uploading to s3")
        if not filtered:
            clip_stats.num_passed += 1
        return clip_stats

    def _write_clip_sam3(self, clip: Clip) -> ClipStats:
        """Write SAM3 per-clip outputs (instances/objects/events JSON + annotated mp4).

        Silently no-ops when the clip has no SAM3 data (i.e. ``SAM3BBoxStage``
        was not run or produced no detections).
        """
        clip_stats = ClipStats()
        if self._dry_run:
            return clip_stats

        source_video = clip.source_video

        if clip.sam3_instances is not None:
            dest = self._get_clip_uri(
                clip.uuid,
                self.get_output_path_sam3_instances(self._output_path),
                "json",
                filename_suffix="_instances",
            )
            self._write_json_data(
                sam3_instances_envelope(clip.sam3_instances),
                dest,
                f"sam3 instances {clip.uuid}",
                source_video,
            )

        if clip.sam3_objects_by_frame is not None:
            dest = self._get_clip_uri(
                clip.uuid,
                self.get_output_path_sam3_objects(self._output_path),
                "json",
                filename_suffix="_objects",
            )
            self._write_json_data(
                sam3_objects_envelope(clip.sam3_objects_by_frame),
                dest,
                f"sam3 objects {clip.uuid}",
                source_video,
            )

        if clip.sam3_events is not None:
            dest = self._get_clip_uri(
                clip.uuid,
                self.get_output_path_sam3_events(self._output_path),
                "json",
                filename_suffix="_events",
            )
            self._write_json_data(
                sam3_events_envelope(clip.sam3_events),
                dest,
                f"sam3 events {clip.uuid}",
                source_video,
            )

        annotated = clip.sam3_annotated_video.resolve()
        if annotated is not None and self._upload_clips:
            dest = self._get_clip_uri(
                clip.uuid,
                self.get_output_path_sam3_tracked(self._output_path),
                "mp4",
            )
            self._write_data(annotated, dest, f"sam3 tracked {clip.uuid}", source_video)

        return clip_stats

    def _get_clip_embedding(self, clip: Clip) -> npt.NDArray[np.float32] | None:
        if self._embedding_algorithm == "internvideo2":
            return clip.intern_video_2_embedding
        if self._embedding_algorithm.startswith("cosmos-embed1-"):
            return clip.cosmos_embed1_embedding
        if self._embedding_algorithm == "openai":
            return clip.openai_embedding
        return None

    def _add_clip_embedding_to_buffer(self, clip: Clip) -> None:
        embedding_to_write = self._get_clip_embedding(clip)
        if embedding_to_write is not None:
            self._embedding_buffer.append(
                {
                    "id": str(clip.uuid),
                    "embedding": embedding_to_write.reshape(-1).tolist(),
                },
            )
        elif self._generate_embeddings:
            logger.error(
                f"Clip {clip.uuid} from {clip.source_video} has no {self._embedding_algorithm} embedding, "
                "skip adding to buffer"
            )

    def _write_clip_embedding(self, clip: Clip) -> ClipStats:
        clip_stats = ClipStats()
        embedding = self._get_clip_embedding(clip)
        if embedding is not None:
            buffer = io.BytesIO()
            pickle.dump(embedding, buffer)
            dest = self._get_clip_uri(
                clip.uuid,
                self.get_output_path_embds(self._output_path, self._embedding_algorithm),
                "pickle",
            )
            if not self._dry_run and self._emit_per_clip_metadata:
                self._write_data(buffer.getvalue(), dest, f"embedding {clip.uuid}", clip.source_video)
            clip_stats.num_with_embeddings += 1
        elif self._generate_embeddings:
            logger.error(
                f"Clip {clip.uuid} from {clip.source_video} has no {self._embedding_algorithm} embedding, "
                "skip uploading"
            )

        return clip_stats

    def _make_clip_metadata(  # noqa: C901, PLR0912, PLR0915
        self, clip: Clip, video_metadata: VideoMetadata, *, filtered: bool = False
    ) -> dict[str, Any]:
        data: dict[str, Any] = {
            "span_uuid": str(clip.uuid),
            "source_video": str(clip.source_video),
            "duration_span": list(clip.span),
            "width_source": video_metadata.width,
            "height_source": video_metadata.height,
            "framerate_source": video_metadata.framerate,
            "clip_location": str(
                self._get_clip_uri(
                    clip.uuid,
                    self.get_output_path_clips(self._output_path, filtered=filtered),
                    "mp4",
                )
            ),
        }

        clip_metadata = None
        try:
            clip_metadata = clip.extract_metadata()
        except Exception as e:  # noqa: BLE001
            logger.exception(f"Failed to extract metadata for {clip.source_video=} {clip.uuid=}, {clip.span=}")
            clip.errors["extract_metadata"] = str(e)

        if clip_metadata:
            data.update(clip_metadata)
        if clip.motion_score_global_mean is not None:
            data["motion_score"] = {
                "global_mean": clip.motion_score_global_mean,
                "per_patch_min_256": clip.motion_score_per_patch_min_256,
            }
        if clip.aesthetic_score is not None:
            data["aesthetic_score"] = clip.aesthetic_score
        if clip.qwen_type_classification is not None:
            data["qwen_type_classification"] = clip.qwen_type_classification
        if clip.qwen_rejection_stage is not None:
            data["qwen_rejection_stage"] = clip.qwen_rejection_stage
        if clip.has_artificial_text is not None:
            data["post_production_text"] = bool(clip.has_artificial_text)
        if clip.sam3_instances is not None:
            data["sam3_num_instances"] = len(clip.sam3_instances)
        if clip.sam3_events is not None:
            data["sam3_num_events"] = len(clip.sam3_events)
        if len(clip.errors) > 0:
            data["errors"] = list(clip.errors)
        has_caption = False
        data["windows"] = []
        data["filtered_windows"] = []
        for window in clip.filter_windows:
            curr_filter_window: dict[str, Any] = {
                "start_frame": window.start_frame,
                "end_frame": window.end_frame,
            }
            if "qwen_rejection_reasons" in window.caption:
                curr_filter_window["qwen_rejection_reasons"] = window.caption["qwen_rejection_reasons"]
            if window.errors:
                curr_filter_window["errors"] = dict(window.errors)
            data["filtered_windows"].append(curr_filter_window)
        total_prompt_tokens = 0
        total_output_tokens = 0
        num_caption_windows = 0
        for window in clip.windows:
            curr_window: dict[str, Any] = {
                "start_frame": window.start_frame,
                "end_frame": window.end_frame,
                "caption_status": window.caption_status,
                "caption_failure_reason": window.caption_failure_reason,
            }
            # Enabled metadata emits flag keys for every window. None means no evaluated flag value,
            # including caption paths that do not populate them.
            if self._caption_quality_flags_enabled:
                curr_window["flag_length_outlier"] = window.flag_length_outlier
                curr_window["flag_repetition"] = window.flag_repetition
                curr_window["flag_near_duplicate"] = window.flag_near_duplicate
            window_has_caption = False
            for model in self._caption_models:
                if model in window.caption:
                    curr_window[f"{model}_caption"] = window.caption[model]
                    if window.caption_status in CAPTION_OK_STATUSES:
                        window_has_caption = True
                if model in window.token_counts:
                    counts = window.token_counts[model]
                    curr_window[f"{model}_prompt_tokens"] = counts.prompt_tokens
                    curr_window[f"{model}_output_tokens"] = counts.output_tokens
                    total_prompt_tokens += counts.prompt_tokens
                    total_output_tokens += counts.output_tokens
            for model in self._enhanced_caption_models:
                if model in window.enhanced_caption:
                    curr_window[f"{model}_enhanced_caption"] = window.enhanced_caption[model]
            if window_has_caption:
                has_caption = True
                num_caption_windows += 1
            data["windows"].append(curr_window)
        data["valid"] = bool(clip.encoded_data and len(clip.windows) > 0)
        data["has_caption"] = has_caption
        # Per-clip marker only; _write_video_metadata() intentionally does not emit it.
        data["caption_quality_flags_enabled"] = self._caption_quality_flags_enabled
        data["total_prompt_tokens"] = total_prompt_tokens
        data["total_output_tokens"] = total_output_tokens
        data["num_caption_windows"] = num_caption_windows
        embedding = self._get_clip_embedding(clip)
        if embedding is not None:
            data["embedding"] = embedding.reshape(-1).tolist()
            data["embedding_model_name"] = self._embedding_algorithm
            data["embedding_model_version"] = self._embedding_model_version

        return data

    def _add_clip_metadata_to_buffer(
        self, clip: Clip, video_metadata: VideoMetadata, *, filtered: bool = False
    ) -> None:
        if self._upload_clip_info_in_chunks or self._upload_clip_info_in_lance:
            data = self._make_clip_metadata(clip, video_metadata, filtered=filtered)
            if data:
                self._metadata_buffer.append(data)

    def _write_clip_metadata(self, clip: Clip, video_metadata: VideoMetadata, *, filtered: bool = False) -> ClipStats:
        clip_stats = ClipStats()
        data = self._make_clip_metadata(clip, video_metadata, filtered=filtered)
        dest = self._get_clip_uri(clip.uuid, self.get_output_path_metas(self._output_path, "v0"), "json")
        if not self._dry_run and self._emit_per_clip_metadata:
            data_to_write = {
                k: v
                for k, v in data.items()
                if k not in ("embedding", "embedding_model_name", "embedding_model_version")
            }
            self._write_json_data(data_to_write, dest, f"metadata {clip.uuid}", clip.source_video)
        clip_stats.num_with_caption += 1 if data.get("has_caption", False) else 0
        clip_stats.num_caption_windows += data.get("num_caption_windows", 0)
        clip_stats.total_prompt_tokens += data.get("total_prompt_tokens", 0)
        clip_stats.total_output_tokens += data.get("total_output_tokens", 0)
        clip_duration = clip.span[1] - clip.span[0]
        clip_stats.total_clip_duration += clip_duration
        clip_stats.max_clip_duration = max(clip_stats.max_clip_duration, clip_duration)
        if not filtered and self._caption_quality_stats_enabled:
            # Piggyback on the existing per-clip metadata path; do not rescan persisted metas/v0.
            clip_stats.caption_quality_stats = self._make_clip_caption_quality_stats(clip)
        return clip_stats

    def _make_clip_caption_quality_stats(self, clip: Clip) -> CaptionQualityStats:
        stats = CaptionQualityStats()
        # Final single-model eligibility is enforced by the summary writer; chunks record the active field locally.
        active_caption_model = self._caption_models[0] if self._caption_models else ""

        for window in clip.windows:
            stats.caption_windows_checked += 1
            status = window.caption_status if window.caption_status is not None else CaptionOutcome.SKIPPED.value
            # Preserve unexpected runtime keys in chunk stats; summary parsing rejects non-v1 keys and omits the
            # root artifact instead of crashing this stage.
            stats.caption_status_counts[status] = stats.caption_status_counts.get(status, 0) + 1

            if window.caption_failure_reason is not None:
                stats.caption_failure_reason_counts[window.caption_failure_reason] = (
                    stats.caption_failure_reason_counts.get(window.caption_failure_reason, 0) + 1
                )
            self._count_caption_quality_flags(stats, window)

            if status not in CAPTION_OK_STATUSES:
                continue

            caption_text = window.caption.get(active_caption_model)
            stripped = caption_text.strip() if caption_text is not None else ""
            if stripped == "":
                stats.empty_caption_count += 1
            elif stripped == VLLM_UNKNOWN_CAPTION:
                stats.sentinel_caption_count += 1
        return stats

    def _count_caption_quality_flags(self, stats: CaptionQualityStats, window: Window) -> None:
        if not self._caption_quality_flags_enabled:
            return

        flag_values = {key: getattr(window, key) for key in CAPTION_QUALITY_FLAG_KEYS}
        if all(value is not None for value in flag_values.values()):
            stats.caption_quality_flags_evaluated_count += 1
            for key, value in flag_values.items():
                if value is True:
                    stats.caption_quality_flag_counts[key] += 1

    def _add_caption_quality_stats_to_video_data(self, data: dict[str, Any], video: Video) -> None:
        if not self._caption_quality_stats_enabled:
            return

        # None means this chunk had no non-filtered clips; persist zeros so summary aggregation can prove completeness.
        caption_quality_stats = video.clip_stats.caption_quality_stats or CaptionQualityStats()
        data["caption_quality_stats"] = caption_quality_stats.to_dict(include_unknown_keys=True)

    def _add_cds_data_to_buffer(self, clip: Clip) -> None:
        if self._upload_cds_parquet:
            embedding_to_write = self._get_clip_embedding(clip)
            if embedding_to_write is not None:
                data = {
                    "id": str(clip.uuid),
                    "embedding": embedding_to_write.reshape(-1).tolist(),
                    "$meta": json.dumps(
                        {
                            "clip_location": str(
                                self._get_clip_uri(
                                    clip.uuid,
                                    self.get_output_path_clips(self._output_path),
                                    "mp4",
                                )
                            ),
                            "model_name": str(self._embedding_algorithm),
                            "model_version": str(self._embedding_model_version),
                            "source_video_location": str(clip.source_video),
                            # Raw observability join — may omit failed captions when no caption text was written.
                            # Gating on caption_status is a downstream policy decision.
                            "caption": " | ".join(clip.get_all_captions()),
                        }
                    ),
                }
                self._cds_data_buffer.append(data)

    def _write_video_metadata(self, video: Video) -> None:
        if isinstance(video.input_video, storage_client.StoragePrefix):
            input_video_path = video.input_video.path
        else:
            input_video_path = video.input_video.as_posix()

        if video.errors:
            self._write_video_errors(video, input_video_path)
            return

        data: dict[str, Any] = {}
        # write video-level metadata from the first clip chunk
        if video.clip_chunk_index == 0:
            data = {
                "video": input_video_path,
                "height": video.metadata.height,
                "width": video.metadata.width,
                "framerate": video.metadata.framerate,
                "num_frames": video.metadata.num_frames,
                "duration": video.metadata.duration,
                "video_codec": video.metadata.video_codec,
                "pixel_format": video.metadata.pixel_format,
                "audio_format": video.metadata.audio_codec,
                "num_total_clips": video.num_total_clips,
                "num_clip_chunks": video.num_clip_chunks,
                "video_uuid": self.get_video_uuid(input_video_path),
            }
            dest = self._get_video_uri(input_video_path)
            self._write_json_data(data, dest, "video metadata", input_video_path)
        # each clip chunk writes its own clip stats
        data = {
            "video": input_video_path,
            "clip_chunk_index": video.clip_chunk_index,
            "num_clips_filtered_by_motion": video.clip_stats.num_filtered_by_motion,
            "num_clips_filtered_by_aesthetic": video.clip_stats.num_filtered_by_aesthetic,
            "num_clips_filtered_by_qwen_classifier": video.clip_stats.num_filtered_by_qwen_classifier,
            "num_clips_filtered_by_qwen_semantic": video.clip_stats.num_filtered_by_qwen_semantic,
            "num_clips_filtered_by_artificial_text": video.clip_stats.num_filtered_by_artificial_text,
            "num_clips_passed": video.clip_stats.num_passed,
            "num_clips_transcoded": video.clip_stats.num_transcoded,
            "num_clips_with_embeddings": video.clip_stats.num_with_embeddings,
            "num_clips_with_caption": video.clip_stats.num_with_caption,
            "num_caption_windows": video.clip_stats.num_caption_windows,
            "num_clips_with_webp": video.clip_stats.num_with_webp,
            "total_clip_duration": video.clip_stats.total_clip_duration,
            "max_clip_duration": video.clip_stats.max_clip_duration,
            "total_prompt_tokens": video.clip_stats.total_prompt_tokens,
            "total_output_tokens": video.clip_stats.total_output_tokens,
            "clips": [str(clip.uuid) for clip in video.clips],
            "filtered_clips": [str(clip.uuid) for clip in video.filtered_clips],
            "all_windows": {},
            "all_windows_enhanced_caption": {},
        }
        self._add_caption_quality_stats_to_video_data(data, video)
        for clip in video.clips:
            clip_uuid = str(clip.uuid)
            data["all_windows"][clip_uuid] = {}
            data["all_windows_enhanced_caption"][clip_uuid] = {}
            for window in clip.windows:
                window_key = f"{window.start_frame}_{window.end_frame}"
                # Raw observability dump — writes caption text regardless of caption_status;
                # failed captions may be absent when no caption text was written. Gating on status is a downstream
                # policy decision.
                for model in self._caption_models:
                    if model in window.caption:
                        data["all_windows"][clip_uuid][window_key] = window.caption[model]
                        break
                # Try each enhanced caption model in order, using the first one found.
                for model in self._enhanced_caption_models:
                    if model in window.enhanced_caption:
                        data["all_windows_enhanced_caption"][clip_uuid][window_key] = window.enhanced_caption[model]
                        break
        dest = self._get_clip_chunk_uri(input_video_path, video.clip_chunk_index)
        self._write_json_data(data, dest, "clip chunk metadata", input_video_path)

    def _write_video_errors(self, video: Video, input_video_path: str) -> None:
        error_data = {
            "video": input_video_path,
            "clip_chunk_index": video.clip_chunk_index,
            "errors": video.errors,
        }
        error_dest = self._get_video_error_uri(input_video_path, video.clip_chunk_index)
        self._write_json_data(error_data, error_dest, "video errors", input_video_path)

    def _write_per_window_data(self, clip: Clip) -> None:
        if not self._generate_cosmos_predict_dataset:
            return
        for window in clip.windows:
            mp4_data = window.mp4_bytes.resolve()
            if mp4_data is None:
                logger.error(
                    f"Clip {clip.uuid} window [{window.start_frame}, {window.end_frame}] "
                    f"from {clip.source_video} has no mp4 bytes, skip uploading to dataset",
                )
                continue
            caption_ok = window.caption_status in CAPTION_OK_STATUSES
            if not caption_ok:
                logger.error(
                    f"Clip {clip.uuid} window [{window.start_frame}, {window.end_frame}] "
                    f"from {clip.source_video} has no caption, skip uploading to dataset",
                )
                continue
            if len(window.t5_xxl_embedding) == 0:
                logger.error(
                    f"Clip {clip.uuid} window [{window.start_frame}, {window.end_frame}] "
                    f"from {clip.source_video} has no T5 XXL embedding, skip uploading to dataset",
                )
                continue
            # upload mp4 bytes
            dest_video = self._get_cosmos_predict_uri(
                clip.uuid,
                (window.start_frame, window.end_frame),
                self.get_output_path_per_window_clips(self._output_path),
                "mp4",
            )
            self._write_data(
                mp4_data,
                dest_video,
                "dataset mp4 {clip.uuid} {window.start_frame}_{window.end_frame}",
                clip.source_video,
            )
            # upload metadata
            dest_meta = self._get_cosmos_predict_uri(
                clip.uuid,
                (window.start_frame, window.end_frame),
                self.get_output_path_per_window_metas(self._output_path),
                "txt",
            )
            caption_text = list(window.caption.values())
            self._write_text_data(
                caption_text[0],
                dest_meta,
                "dataset caption {clip.uuid} {window.start_frame}_{window.end_frame}",
                clip.source_video,
            )
            # upload T5 XXL embedding
            dest_t5 = self._get_cosmos_predict_uri(
                clip.uuid,
                (window.start_frame, window.end_frame),
                self.get_output_path_per_window_t5_embeds(self._output_path),
                "pickle",
            )
            buffer = io.BytesIO()
            t5_embeddings = list(window.t5_xxl_embedding.values())
            pickle.dump([t5_embeddings[0]], buffer)
            self._write_data(
                buffer.getvalue(),
                dest_t5,
                "dataset t5_xxl {clip.uuid} {window.start_frame}_{window.end_frame}",
                clip.source_video,
            )


def consolidate_lance_fragments(output_path: str, output_s3_profile_name: str) -> None:
    """Consolidate staged Lance fragments into the final metadata dataset."""
    _consolidate_one(
        staging_root=ClipWriterStage.get_output_path_meta_lance_fragments(output_path, "v0"),
        processed_root=ClipWriterStage.get_output_path_meta_lance_fragments_processed(output_path, "v0"),
        final_uri=ClipWriterStage.get_output_path_meta_lance(output_path, "v0"),
        output_s3_profile_name=output_s3_profile_name,
    )


def _consolidate_one(staging_root: str, processed_root: str, final_uri: str, output_s3_profile_name: str) -> None:
    client = storage_utils.get_storage_client(
        staging_root, profile_name=output_s3_profile_name, can_overwrite=True, can_delete=True
    )
    storage_options = storage_utils.get_lance_storage_options(final_uri, profile_name=output_s3_profile_name)
    sidecars = [fname for fname in get_files_relative(staging_root, client) if fname.endswith(".json")]
    if not sidecars:
        return

    processed_client = storage_utils.get_storage_client(
        processed_root,
        profile_name=output_s3_profile_name,
        can_overwrite=True,
    )
    processed_seen: set[str] = set()
    if storage_utils.path_exists(processed_root, processed_client):
        processed_seen.update(
            fname for fname in get_files_relative(processed_root, processed_client) if fname.endswith(".json")
        )
    sidecars = [fname for fname in sidecars if fname not in processed_seen]
    if not sidecars:
        return

    fragments: list[lance.FragmentMetadata] = []
    schema_b64: str | None = None
    for rel_path in sidecars:
        payload = read_json_file(get_full_path(staging_root, rel_path), client)
        frag_payload = payload.get("fragments", [])
        fragments.extend(lance.FragmentMetadata.from_json(json.dumps(frag)) for frag in frag_payload)
        if schema_b64 is None:
            schema_b64 = payload.get("schema_b64")

    if schema_b64 is None:
        logger.warning(f"No schema found for Lance consolidation under {staging_root}")
        return

    schema_buf = base64.b64decode(schema_b64)
    schema = pa.ipc.read_schema(pa.py_buffer(schema_buf))
    try:
        dataset = lance.dataset(final_uri, storage_options=storage_options)
        read_version = dataset.version
        op: lance.LanceOperation.Append | lance.LanceOperation.Overwrite = lance.LanceOperation.Append(fragments)
    except (FileNotFoundError, ValueError):
        read_version = 0
        op = lance.LanceOperation.Overwrite(schema, fragments)

    lance.LanceDataset.commit(
        final_uri,
        op,
        read_version=read_version,
        storage_options=storage_options,
    )
    _archive_processed_sidecars(sidecars, staging_root, processed_root, client, processed_client)
    logger.info(f"Consolidated {len(fragments)} Lance fragments into {final_uri}")


def _archive_processed_sidecars(
    sidecars: list[str],
    staging_root: str,
    processed_root: str,
    staging_client: storage_client.StorageClient | None,
    processed_client: storage_client.StorageClient | None,
) -> None:
    """Archive processed sidecars and avoid reprocessing on subsequent runs."""
    for rel_path in sidecars:
        payload = read_json_file(get_full_path(staging_root, rel_path), staging_client)
        dest = get_full_path(processed_root, rel_path)
        write_json(
            payload,
            dest,
            "processed lance fragment metadata",
            rel_path,
            verbose=False,
            client=processed_client,
            overwrite=True,
        )
        staged_path = get_full_path(staging_root, rel_path)
        if isinstance(staged_path, pathlib.Path):
            staged_path.unlink(missing_ok=True)
        elif staging_client is not None:
            try:
                staging_client.delete_object(staged_path)
            except ValueError as exc:
                logger.warning(f"Skipping deletion for {staged_path}: {exc}")
            except ClientError as exc:
                err_code = exc.response.get("Error", {}).get("Code")
                if err_code in {"NoSuchKey", "404"}:
                    logger.warning(f"Remote sidecar already missing {staged_path}: {exc}")
                else:
                    error_msg = f"Failed to delete remote sidecar {staged_path}"
                    raise RuntimeError(error_msg) from exc
            except ResourceNotFoundError as exc:
                logger.warning(f"Remote sidecar already missing {staged_path}: {exc}")
            except AzureError as exc:
                error_msg = f"Failed to delete remote sidecar {staged_path}"
                raise RuntimeError(error_msg) from exc
            except Exception as exc:
                error_msg = f"Failed to delete remote sidecar {staged_path}"
                raise RuntimeError(error_msg) from exc
