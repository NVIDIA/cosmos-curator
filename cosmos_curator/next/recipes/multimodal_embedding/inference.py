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

"""Separate download, CPU preparation, GPU embedding, and checkpointed result staging."""

from collections.abc import Iterator
from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Final

import lance
import numpy as np
import numpy.typing as npt
import pyarrow as pa
import ray
from loguru import logger
from ray.data import ActorPoolStrategy
from ray.data.block import BlockMetadata
from ray.data.context import DataContext
from ray.data.datasource import Datasource, ReadTask

from cosmos_curator.core.utils import environment
from cosmos_curator.core.utils.pixi_runtime_envs import PixiRuntimeEnv, ray_data_gpu_runtime_env
from cosmos_curator.next.core.ray_autoscaling import configure_download_actor_autoscaling
from cosmos_curator.next.core.ray_checkpoints import installed_checkpoint
from cosmos_curator.next.recipes.multimodal_embedding.config import ResolvedMultimodalEmbeddingConfig
from cosmos_curator.next.recipes.multimodal_embedding.contracts import (
    EmbeddingTerminalResult,
    VideoEmbeddingModelSpec,
    build_terminal_result_row,
    logical_result_fields,
    resolve_model,
    result_field_names,
)
from cosmos_curator.next.recipes.multimodal_embedding.episode_input import (
    CAMERA_MAPPED_INPUT_SCHEMA,
    CAMERA_WORK_ID,
    EPISODE_SOURCE_COLUMNS,
    project_camera_work,
)
from cosmos_curator.next.recipes.multimodal_embedding.model_adapter import (
    VideoEncoder,
    VideoPreprocessor,
    load_video_encoder,
    load_video_preprocessor,
    validate_projection,
)
from cosmos_curator.next.recipes.multimodal_embedding.video_media import (
    VideoMediaError,
    check_episode_duration,
    decode_selected_frames,
    inspect_video,
)
from cosmos_curator.next.recipes.video_caption.lance_state import CaptionAttempt
from cosmos_curator.next.recipes.video_caption.workspace import CaptionWorkspace, record_phase_a_completion
from cosmos_curator.next.utils.media_reader import MediaObjectError, MediaObjectReader

_READY: Final = "ready"
_ERROR: Final = "error"
_UNSUPPORTED: Final = "unsupported_duration"
_INVALID_VECTOR: Final = "invalid_vector"
_PREPARED_NDIM: Final = 4
_RGB_CHANNELS: Final = 3
_FLOAT32_BYTES: Final = np.dtype("<f4").itemsize
_DOWNLOADED_SCHEMA = pa.schema(
    [
        *CAMERA_MAPPED_INPUT_SCHEMA,
        pa.field("download_status", pa.string(), nullable=False),
        pa.field("media_bytes", pa.large_binary()),
        pa.field("download_error_code", pa.string()),
        pa.field("download_error_message", pa.large_string()),
    ]
)
_PREPARED_SCHEMA = pa.schema(
    [
        *CAMERA_MAPPED_INPUT_SCHEMA,
        pa.field("preparation_status", pa.string(), nullable=False),
        pa.field("prepared_video", pa.binary()),
        pa.field("prepared_shape", pa.list_(pa.int32())),
        pa.field("media_duration_ns", pa.int64()),
        pa.field("preparation_error_code", pa.string()),
        pa.field("preparation_error_message", pa.large_string()),
    ]
)


def camera_staged_schema(model: VideoEmbeddingModelSpec) -> pa.Schema:
    """Checkpoint one camera result per row before assembling episode publication rows."""
    metadata_type = logical_result_fields("default", model_key=model.key).field(1).type
    return pa.schema(
        [
            *CAMERA_MAPPED_INPUT_SCHEMA,
            pa.field("embedding", pa.list_(pa.float32())),
            pa.field("embedding_metadata", metadata_type, nullable=False),
        ]
    )


def _prepared_row(  # noqa: PLR0913 -- stage rows carry independent outcome and tensor fields
    source: dict[str, Any],
    *,
    status: str,
    prepared_video: bytes | None = None,
    prepared_shape: list[int] | None = None,
    media_duration_ns: int | None = None,
    error_code: str | None = None,
    error_message: str | None = None,
) -> dict[str, Any]:
    return {
        **source,
        "preparation_status": status,
        "prepared_video": prepared_video,
        "prepared_shape": prepared_shape,
        "media_duration_ns": media_duration_ns,
        "preparation_error_code": error_code,
        "preparation_error_message": error_message,
    }


def _validate_prepared_tensor(tensor: npt.NDArray[np.float32], *, frame_count: int, media_duration_ns: int) -> None:
    if (
        tensor.ndim != _PREPARED_NDIM
        or tensor.shape[0] != frame_count
        or tensor.shape[1] != _RGB_CHANNELS
        or not np.isfinite(tensor).all()
    ):
        msg = "Video processor returned an invalid tensor"
        raise VideoMediaError(_INVALID_VECTOR, msg, media_duration_ns=media_duration_ns)


class VideoDownloadActor:
    """Fetch and verify one episode at a time while owning one IO slot."""

    def __init__(
        self,
        *,
        camera_keys: tuple[str, ...],
        max_episode_duration_s: float,
        storage_profile: str,
        media_attempts: int,
    ) -> None:
        """Save input binding, eligibility policy, and transport settings."""
        self._camera_keys = frozenset(camera_keys)
        self._input_schema = CAMERA_MAPPED_INPUT_SCHEMA
        self._output_schema = _DOWNLOADED_SCHEMA
        self._max_duration_s = max_episode_duration_s
        self._reader = MediaObjectReader(storage_profile=storage_profile, max_attempts=media_attempts)

    def __call__(self, batch: pa.Table) -> pa.Table:
        """Return verified MP4 bytes or a terminal non-model outcome per episode."""
        if batch.schema.names != self._input_schema.names:
            msg = "Media download expects the projected episode input schema"
            raise ValueError(msg)
        rows: list[dict[str, Any]] = []
        for source in batch.to_pylist():
            duration_ns = int(source["exclusive_end_ns"]) - int(source["start_ns"])
            if duration_ns <= 0 or source["camera_key"] not in self._camera_keys:
                msg = "Projected embedding row has an invalid duration or camera binding"
                raise ValueError(msg)
            row = {
                **source,
                "download_status": _READY,
                "media_bytes": None,
                "download_error_code": None,
                "download_error_message": None,
            }
            if duration_ns > self._max_duration_s * 1_000_000_000:
                row.update(download_status=_UNSUPPORTED, download_error_code=_UNSUPPORTED)
            else:
                try:
                    payload = self._reader.read(
                        str(source["media_location"]), max_bytes=int(source["media_size_bytes"])
                    )
                except MediaObjectError as exc:
                    row.update(
                        download_status=_ERROR,
                        download_error_code="media_object_error",
                        download_error_message=str(exc),
                    )
                else:
                    if len(payload) != source["media_size_bytes"]:
                        row.update(
                            download_status=_ERROR,
                            download_error_code="size_mismatch",
                            download_error_message=(
                                f"Media has {len(payload)} bytes; expected {source['media_size_bytes']}"
                            ),
                        )
                    elif sha256(payload).hexdigest() != source["media_sha256"]:
                        row.update(
                            download_status=_ERROR,
                            download_error_code="sha256_mismatch",
                            download_error_message="Media SHA-256 differs from the episode descriptor",
                        )
                    else:
                        row["media_bytes"] = payload
            rows.append(row)
        return pa.Table.from_pylist(rows, schema=self._output_schema)


class VideoPreparationActor:
    """Decode and preprocess downloaded MP4 bytes using CPU resources only."""

    def __init__(self, *, model: VideoEmbeddingModelSpec, camera_keys: tuple[str, ...], decode_cpus: int = 1) -> None:
        """Save the model binding and lazily load the CPU preprocessor."""
        self._model = model
        self._camera_keys = frozenset(camera_keys)
        self._source_schema = CAMERA_MAPPED_INPUT_SCHEMA
        self._input_schema = _DOWNLOADED_SCHEMA
        self._output_schema = _PREPARED_SCHEMA
        self._decode_cpus = decode_cpus
        self._preprocessor: VideoPreprocessor | None = None

    def __call__(self, batch: pa.Table) -> pa.Table:
        """Return compact tensors and carry download errors or unsupported rows through."""
        if batch.schema.names != self._input_schema.names:
            msg = "Media preparation expects downloaded episode rows"
            raise ValueError(msg)
        rows: list[dict[str, Any]] = []
        for downloaded in batch.to_pylist():
            source = {name: downloaded[name] for name in self._source_schema.names}
            duration_ns = int(source["exclusive_end_ns"]) - int(source["start_ns"])
            if duration_ns <= 0 or source["camera_key"] not in self._camera_keys:
                msg = "Downloaded embedding row has an invalid duration or camera binding"
                raise ValueError(msg)
            status = downloaded["download_status"]
            if status in {_ERROR, _UNSUPPORTED}:
                rows.append(
                    _prepared_row(
                        source,
                        status=status,
                        error_code=downloaded["download_error_code"],
                        error_message=downloaded["download_error_message"],
                    )
                )
                continue
            payload = downloaded["media_bytes"]
            if status != _READY or not isinstance(payload, bytes) or len(payload) != source["media_size_bytes"]:
                msg = "Downloaded episode has neither verified media nor a terminal item outcome"
                raise ValueError(msg)
            media_duration_ns: int | None = None
            try:
                timeline = inspect_video(payload, frame_count=self._model.video_frame_count, threads=self._decode_cpus)
                media_duration_ns = timeline.duration_ns
                check_episode_duration(timeline, episode_duration_ns=duration_ns)
                frames = decode_selected_frames(payload, timeline, threads=self._decode_cpus).frames
                if self._preprocessor is None:
                    self._preprocessor = load_video_preprocessor(self._model, cpus=self._decode_cpus)
                tensor = np.ascontiguousarray(self._preprocessor.preprocess(frames), dtype="<f4")
                _validate_prepared_tensor(
                    tensor, frame_count=self._model.video_frame_count, media_duration_ns=media_duration_ns
                )
                rows.append(
                    _prepared_row(
                        source,
                        status=_READY,
                        prepared_video=tensor.tobytes(),
                        prepared_shape=list(tensor.shape),
                        media_duration_ns=media_duration_ns,
                    )
                )
            except VideoMediaError as exc:
                error_duration_ns = exc.media_duration_ns if exc.media_duration_ns is not None else media_duration_ns
                rows.append(
                    _prepared_row(
                        source,
                        status=_ERROR,
                        error_code=exc.code,
                        error_message=str(exc),
                        media_duration_ns=error_duration_ns,
                    )
                )
        return pa.Table.from_pylist(rows, schema=self._output_schema)


class VideoEmbeddingBatchActor:
    """Batch prepared videos on one GPU and carry non-model rows through."""

    def __init__(
        self,
        *,
        model: VideoEmbeddingModelSpec,
        camera_keys: tuple[str, ...],
        max_episode_duration_s: float,
        camera_results: dict[str, tuple[str, str]],
    ) -> None:
        """Save result bindings and lazily load the GPU model."""
        self._model = model
        self._camera_keys = frozenset(camera_keys)
        self._camera_results = camera_results
        self._source_schema = CAMERA_MAPPED_INPUT_SCHEMA
        self._input_schema = _PREPARED_SCHEMA
        self._max_duration_s = max_episode_duration_s
        self._encoder: VideoEncoder | None = None
        self._schema = camera_staged_schema(model)

    def _decode_ready(self, row: dict[str, Any]) -> npt.NDArray[np.float32]:
        payload = row["prepared_video"]
        shape = row["prepared_shape"]
        if (
            not isinstance(payload, bytes)
            or not isinstance(shape, list)
            or len(shape) != _PREPARED_NDIM
            or shape[0] != self._model.video_frame_count
            or shape[1] != _RGB_CHANNELS
            or any(not isinstance(size, int) or size <= 0 for size in shape)
            or len(payload) != int(np.prod(shape)) * _FLOAT32_BYTES
        ):
            msg = "Prepared video payload does not match its tensor shape"
            raise ValueError(msg)
        return np.frombuffer(payload, dtype="<f4").reshape(shape)

    def _encode_ready(
        self, rows: list[dict[str, Any]], ready: list[int], videos: list[npt.NDArray[np.float32]]
    ) -> dict[int, EmbeddingTerminalResult]:
        if self._encoder is None:
            self._encoder = load_video_encoder(self._model)
        # CUDA/OOM/model exceptions abort the batch. Only an invalid individual
        # projection becomes a terminal item error.
        projections = self._encoder.encode_batch(np.stack(videos))
        if projections.shape != (len(ready), self._model.vector_dimension):
            msg = "Model returned an incompatible batch of video projections"
            raise RuntimeError(msg)
        outcomes: dict[int, EmbeddingTerminalResult] = {}
        for position, index in enumerate(ready):
            try:
                vector = validate_projection(
                    projections[position : position + 1], dimension=self._model.vector_dimension
                )
                outcomes[index] = EmbeddingTerminalResult(
                    status="success",
                    vector=vector,
                    max_episode_duration_s=self._max_duration_s,
                    error_code=None,
                    error_message=None,
                    media_duration_ns=rows[index]["media_duration_ns"],
                )
            except VideoMediaError as exc:
                outcomes[index] = EmbeddingTerminalResult(
                    status=_ERROR,
                    vector=None,
                    max_episode_duration_s=self._max_duration_s,
                    error_code=exc.code,
                    error_message=str(exc),
                    media_duration_ns=rows[index]["media_duration_ns"],
                )
        return outcomes

    def __call__(self, batch: pa.Table) -> pa.Table:
        """Encode ready tensors in one call and emit one terminal row per input."""
        if batch.schema.names != self._input_schema.names:
            msg = "Embedding actor expects prepared episode rows"
            raise ValueError(msg)
        rows = batch.to_pylist()
        ready: list[int] = []
        videos: list[npt.NDArray[np.float32]] = []
        for index, row in enumerate(rows):
            if row["camera_key"] not in self._camera_keys:
                msg = "Prepared embedding row has an incompatible camera binding"
                raise ValueError(msg)
            if row["preparation_status"] != _READY:
                continue
            ready.append(index)
            videos.append(self._decode_ready(row))
        outcomes = self._encode_ready(rows, ready, videos) if ready else {}
        terminal: list[dict[str, Any]] = []
        for index, row in enumerate(rows):
            if index not in outcomes:
                if row["preparation_status"] not in {_ERROR, _UNSUPPORTED}:
                    msg = "Prepared episode has neither a tensor nor a terminal item outcome"
                    raise ValueError(msg)
                outcomes[index] = EmbeddingTerminalResult(
                    status=row["preparation_status"],
                    vector=None,
                    max_episode_duration_s=self._max_duration_s,
                    error_code=row["preparation_error_code"],
                    error_message=row["preparation_error_message"],
                    media_duration_ns=row["media_duration_ns"],
                )
            source = {name: row[name] for name in self._source_schema.names}
            result_key, digest = self._camera_results[row["camera_key"]]
            result = build_terminal_result_row(
                source, model=self._model, result_key=result_key, digest=digest, result=outcomes[index]
            )
            vector_field, metadata_field = result_field_names(result_key, model_key=self._model.key)
            result = {"embedding": result[vector_field], "embedding_metadata": result[metadata_field]}
            terminal.append({**source, **result})
        return pa.Table.from_pylist(terminal, schema=self._schema)


def map_video_embeddings(
    source: ray.data.Dataset,
    config: ResolvedMultimodalEmbeddingConfig,
    *,
    num_input_rows: int,
    camera_results: dict[str, tuple[str, str]],
) -> ray.data.Dataset:
    """Download under IO limits, prepare under CPU/memory limits, then batch on GPUs."""
    if num_input_rows < 1:
        msg = f"Video embedding inference requires at least one input row, got {num_input_rows}"
        raise ValueError(msg)
    # Scope tuning to this plan, retaining Ray's conservative scale-up step.
    configure_download_actor_autoscaling(source.context)
    source.context.autoscaling_config.actor_pool_max_upscaling_delta = 1
    model = resolve_model(config.model.variant)
    camera_keys = tuple(camera_results)
    downloaded = source.map_batches(
        VideoDownloadActor,
        fn_constructor_kwargs={
            "camera_keys": camera_keys,
            "max_episode_duration_s": config.selection.max_episode_duration_s,
            "storage_profile": config.execution.storage_profile,
            "media_attempts": config.execution.media_attempts,
        },
        batch_format="pyarrow",
        batch_size=1,
        compute=ActorPoolStrategy(
            min_size=1,
            max_size=None if config.execution.media_concurrency == "auto" else config.execution.media_concurrency,
            max_tasks_in_flight_per_actor=1,
        ),
        num_cpus=0.25,
        max_concurrency=1,
        resources={environment.CURATOR_IO_RESOURCE_NAME: 1.0},
        # Bound worker-crash recovery; Ray Data otherwise defaults both limits to unlimited.
        max_restarts=5,
        max_task_retries=5,
    )
    prepared = downloaded.map_batches(
        VideoPreparationActor,
        fn_constructor_kwargs={
            "model": model,
            "camera_keys": camera_keys,
            "decode_cpus": config.execution.decode_cpus,
        },
        batch_format="pyarrow",
        batch_size=1,
        compute=ActorPoolStrategy(
            min_size=1,
            max_size=None if config.execution.decode_concurrency == "auto" else config.execution.decode_concurrency,
            max_tasks_in_flight_per_actor=1,
        ),
        runtime_env=PixiRuntimeEnv(model.runtime_environment),
        num_cpus=config.execution.decode_cpus,
        memory=int(config.execution.decode_memory_gib * 1024**3),
        max_concurrency=1,
        max_restarts=5,
        max_task_retries=5,
    )
    # Pending work bounds the pool while allowing workers that join later to contribute.
    configured = config.execution.inference_concurrency
    actor_ceiling = num_input_rows if configured == "auto" else min(configured, num_input_rows)
    return prepared.map_batches(
        VideoEmbeddingBatchActor,
        fn_constructor_kwargs={
            "model": model,
            "camera_keys": camera_keys,
            "max_episode_duration_s": config.selection.max_episode_duration_s,
            "camera_results": camera_results,
        },
        batch_format="pyarrow",
        batch_size=config.execution.inference_batch_size,
        compute=ActorPoolStrategy(min_size=1, max_size=actor_ceiling, max_tasks_in_flight_per_actor=1),
        runtime_env=ray_data_gpu_runtime_env(model.runtime_environment),
        num_cpus=config.execution.inference_cpus,
        num_gpus=config.execution.inference_gpus,
        max_concurrency=1,
        max_restarts=5,
        max_task_retries=5,
    )


@dataclass(frozen=True)
class _CameraRead:
    """Serializable fragment read that exposes camera identities to Ray's checkpoint filter."""

    uri: str
    version: int
    fragment_id: int
    camera_keys: tuple[str, ...]
    batch_size: int
    storage_options: dict[str, str] | None

    def read(self) -> Iterator[pa.Table]:
        dataset = lance.dataset(self.uri, version=self.version, storage_options=self.storage_options)
        fragment = dataset.get_fragment(self.fragment_id)
        if fragment is None:
            msg = f"Pinned camera fragment {self.fragment_id} is absent at version {self.version}"
            raise ValueError(msg)
        for batch in fragment.scanner(
            columns=list(EPISODE_SOURCE_COLUMNS), with_row_address=True, batch_size=self.batch_size
        ).to_batches():
            yield project_camera_work(pa.Table.from_batches([batch]), camera_keys=self.camera_keys)


class _CameraDatasource(Datasource):
    """Read pending episode/camera pairs without downloading media or collecting rows on the driver."""

    def __init__(
        self,
        dataset: lance.LanceDataset,
        config: ResolvedMultimodalEmbeddingConfig,
        pending_cameras: dict[int, tuple[str, ...]],
        storage_options: dict[str, str] | None,
    ) -> None:
        super().__init__()  # type: ignore[no-untyped-call]
        self._reads: list[tuple[_CameraRead, int]] = []
        for fragment in dataset.get_fragments():
            keys = pending_cameras.get(int(fragment.fragment_id), ())
            if keys:
                read = _CameraRead(
                    config.input.episodes_lance_uri,
                    int(dataset.version),
                    int(fragment.fragment_id),
                    keys,
                    config.execution.lance_read_batch_size,
                    storage_options,
                )
                self._reads.append((read, int(fragment.count_rows()) * len(keys)))
        self.num_rows = sum(rows for _read, rows in self._reads)

    def estimate_inmemory_data_size(self) -> int | None:
        return None

    def get_read_tasks(
        self, parallelism: int, per_task_row_limit: int | None = None, data_context: DataContext | None = None
    ) -> list[ReadTask]:
        del parallelism, data_context
        return [
            ReadTask(
                read.read,
                BlockMetadata(num_rows=rows, size_bytes=None, input_files=None, exec_stats=None),
                schema=CAMERA_MAPPED_INPUT_SCHEMA,
                per_task_row_limit=per_task_row_limit,
            )
            for read, rows in self._reads
        ]


def run_inference_phase(  # noqa: PLR0913 -- explicit finite work, output, and recovery bindings
    dataset: lance.LanceDataset,
    attempt: CaptionAttempt,
    config: ResolvedMultimodalEmbeddingConfig,
    workspace: CaptionWorkspace,
    *,
    camera_results: dict[str, tuple[str, str]],
    pending_cameras: dict[int, tuple[str, ...]],
    digest: str,
    storage_options: dict[str, str] | None,
) -> None:
    """Share actors across cameras and checkpoint each pair before atomic episode assembly."""
    source = _CameraDatasource(dataset, config, pending_cameras, storage_options)
    with installed_checkpoint(
        id_column=CAMERA_WORK_ID, checkpoint_path=workspace.checkpoints_uri, filesystem=workspace.filesystem
    ):
        rows = ray.data.read_datasource(source, override_num_blocks=len(pending_cameras))
        rows = rows.repartition(target_num_rows_per_block=config.execution.lance_read_batch_size, strict=True)
        embedded = map_video_embeddings(rows, config, num_input_rows=source.num_rows, camera_results=camera_results)
        groups = embedded.repartition(target_num_rows_per_block=config.execution.parquet_rows_per_file, strict=True)
        groups.write_parquet(
            workspace.results_uri,
            filesystem=workspace.filesystem,
            max_rows_per_file=config.execution.parquet_rows_per_file,
        )
    try:
        record_phase_a_completion(workspace, attempt, digest)
    except Exception as exc:  # noqa: BLE001 -- checkpointed rows remain authoritative
        logger.warning("Camera results are durable, but completion marker write failed: {!r}", exc)
