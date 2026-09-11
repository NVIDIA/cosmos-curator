# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Phase A: checkpoint-filtered media fetch, Ray Data LLM, and durable Parquet."""

import base64
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

import lance
import pyarrow as pa
import pyarrow.fs as pafs
import ray
from botocore.exceptions import BotoCoreError, ClientError
from loguru import logger
from ray.data.block import BlockMetadata
from ray.data.checkpoint import CheckpointConfig
from ray.data.context import DataContext
from ray.data.datasource import Datasource, ReadTask

from cosmos_curator.core.utils import environment
from cosmos_curator.core.utils.pixi_runtime_envs import ray_data_gpu_runtime_env
from cosmos_curator.next.recipes.video_caption.config import ResolvedVideoCaptionConfig
from cosmos_curator.next.recipes.video_caption.contracts import (
    DEFAULT_PROMPT,
    MAX_OUTPUT_TOKENS,
    CaptionModelSpec,
    terminal_metadata,
)
from cosmos_curator.next.recipes.video_caption.lance_state import CaptionAttempt
from cosmos_curator.next.recipes.video_caption.workspace import (
    CaptionWorkspace,
    record_phase_a_completion,
)
from cosmos_curator.next.recipes.video_split.storage import download_bytes
from cosmos_curator.next.utils.arrow_utils import filesystem_path

_ROW_ADDRESS_FRAGMENT_SHIFT = 32
_ROW_ADDRESS_OFFSET_MASK = (1 << _ROW_ADDRESS_FRAGMENT_SHIFT) - 1
_ITEM_S3_ERROR_CODES = frozenset({"404", "NoSuchKey", "NoSuchObject", "NotFound", "InvalidObjectState"})
_RETRYABLE_S3_ERROR_CODES = frozenset(
    {
        "500",
        "502",
        "503",
        "504",
        "InternalError",
        "RequestTimeout",
        "RequestTimeoutException",
        "ServiceUnavailable",
        "SlowDown",
        "Throttling",
        "ThrottlingException",
    }
)
_RETRYABLE_S3_HTTP_STATUSES = frozenset({429, 500, 502, 503, 504})
_CAPTION_INPUT_COLUMNS = ("clip_id", "clip_uri", "clip_size_bytes")
_VLLM_ENGINE_STAGE_NAME = "vLLMEngineStage"
_BYTES_PER_GIB = 1024**3
_PREPARE_MULTIMODAL_MEMORY_BYTES = 24 * _BYTES_PER_GIB
_CHAT_TEMPLATE_MEMORY_BYTES = 16 * _BYTES_PER_GIB
_MEDIA_FETCH_RESOURCES = {environment.CURATOR_IO_RESOURCE_NAME: 1.0}

MAPPED_INPUT_SCHEMA = pa.schema(
    [
        pa.field("fragment_id", pa.int64(), nullable=False),
        pa.field("row_offset", pa.int64(), nullable=False),
        pa.field("clip_id", pa.string(), nullable=False),
        pa.field("clip_uri", pa.large_string(), nullable=False),
        pa.field("clip_size_bytes", pa.int64(), nullable=False),
    ]
)

FETCHED_INPUT_SCHEMA = pa.schema(
    [
        *MAPPED_INPUT_SCHEMA,
        pa.field("video_data_url", pa.large_string()),
        pa.field("download_error_type", pa.string()),
        pa.field("download_error_message", pa.large_string()),
    ]
)


@dataclass(frozen=True)
class _CaptionFragmentRead:
    """Picklable driver-side metadata for one pinned Lance fragment."""

    fragment_id: int
    num_rows: int
    input_files: tuple[str, ...]
    schema: pa.Schema


class _CaptionLanceDatasource(Datasource):
    """Read selected fragments without putting live ``LanceFragment`` objects in Ray tasks.

    ``LanceFragment.__reduce__`` reopens its dataset without storage options. Ray's
    built-in Lance datasource captures explicitly supplied fragments in each read
    task, so pickling an S3 fragment can lose its configured credentials before the
    task even starts. This datasource plans from inert metadata and has each worker
    reopen the pinned dataset with the original storage options and fragment IDs.
    """

    def __init__(
        self,
        *,
        uri: str,
        version: int,
        fragments: tuple[_CaptionFragmentRead, ...],
        storage_options: dict[str, str] | None,
        batch_size: int,
    ) -> None:
        super().__init__()  # type: ignore[no-untyped-call]  # Ray's public Datasource constructor lacks typing.
        self._uri = uri
        self._version = version
        self._fragments = fragments
        self._storage_options = dict(storage_options) if storage_options is not None else None
        self._batch_size = batch_size

    @property
    def num_rows(self) -> int:
        """Return the pre-checkpoint-filter work ceiling without scanning rows."""
        return sum(fragment.num_rows for fragment in self._fragments)

    def estimate_inmemory_data_size(self) -> int | None:
        """Let Ray choose memory targets because Lance file size is not an Arrow size estimate."""
        return None

    def get_read_tasks(
        self,
        parallelism: int,
        per_task_row_limit: int | None = None,
        data_context: DataContext | None = None,
    ) -> list[ReadTask]:
        """Build tasks containing only fragment IDs and serializable connection data."""
        del data_context
        if parallelism < 1:
            msg = f"Caption Lance read parallelism must be positive, got {parallelism}"
            raise ValueError(msg)
        task_count = min(parallelism, len(self._fragments))
        tasks = []
        for task_index in range(task_count):
            start = task_index * len(self._fragments) // task_count
            stop = (task_index + 1) * len(self._fragments) // task_count
            fragments = self._fragments[start:stop]
            fragment_ids = tuple(fragment.fragment_id for fragment in fragments)
            metadata = BlockMetadata(
                num_rows=sum(fragment.num_rows for fragment in fragments),
                size_bytes=None,
                input_files=tuple(path for fragment in fragments for path in fragment.input_files),
                exec_stats=None,
            )
            tasks.append(
                ReadTask(
                    partial(
                        _read_caption_fragments,
                        uri=self._uri,
                        version=self._version,
                        fragment_ids=fragment_ids,
                        storage_options=self._storage_options,
                        batch_size=self._batch_size,
                    ),
                    metadata,
                    schema=fragments[0].schema,
                    per_task_row_limit=per_task_row_limit,
                )
            )
        return tasks


class _ExactVLLMBatchProcessor:
    """Run a public Ray LLM processor with exact row blocks at its GPU boundary.

    Ray Data's ordinary ``map_batches(batch_size=N)`` rebundler stops after it
    accumulates *at least* ``N`` rows. It does not split the final input block,
    so a 33-row bundle reaches the stage UDF as calls of 32 and 1. Multimodal
    rows are byte-shaped into variable-size blocks, which makes those remainders
    common rather than a single end-of-stream tail.

    Ray's public Processor does not expose an operator insertion hook. Replay its
    public stage descriptions and insert a strict streaming repartition directly
    before the vLLM stage so every non-final GPU task contains an exact batch.
    """

    def __init__(self, processor: Any, *, batch_size: int) -> None:  # noqa: ANN401 -- public Ray Processor facade
        if batch_size < 1:
            msg = f"Exact vLLM batch size must be positive, got {batch_size}"
            raise ValueError(msg)
        stage_names = processor.list_stage_names()
        if stage_names.count(_VLLM_ENGINE_STAGE_NAME) != 1:
            msg = f"Expected exactly one {_VLLM_ENGINE_STAGE_NAME}, found {stage_names}"
            raise ValueError(msg)
        if processor.config.batch_size != batch_size:
            msg = (
                f"Exact vLLM batch size {batch_size} does not match processor batch size {processor.config.batch_size}"
            )
            raise ValueError(msg)
        self._processor = processor
        self._batch_size = batch_size

    def __call__(self, dataset: ray.data.Dataset) -> ray.data.Dataset:
        """Build the processor DAG with exact batches immediately before vLLM."""
        processor = self._processor
        if processor.preprocess is not None:
            dataset = dataset.map(processor.preprocess, **processor.preprocess_map_kwargs)

        for stage_name in processor.list_stage_names():
            if stage_name == _VLLM_ENGINE_STAGE_NAME:
                logger.info("Enforcing exact {}-row blocks at the caption vLLM boundary", self._batch_size)
                dataset = dataset.repartition(target_num_rows_per_block=self._batch_size, strict=True)
            stage = processor.get_stage_by_name(stage_name)
            stage_kwargs = stage.get_dataset_map_batches_kwargs(
                batch_size=processor.config.batch_size,
                data_column=processor.DATA_COLUMN,
            )
            dataset = dataset.map_batches(stage.fn, **stage_kwargs)

        if processor.postprocess is not None:
            dataset = dataset.map(processor.postprocess, **processor.postprocess_map_kwargs)
        return dataset


class SharedMediaError(RuntimeError):
    """A credentials or storage-availability failure that must stop Phase A."""


class _ItemMediaError(RuntimeError):
    """A deterministic failure scoped to one clip."""


def result_schema(spec: CaptionModelSpec) -> pa.Schema:
    """Return the explicit Parquet recovery schema."""
    return pa.schema(
        [
            pa.field("fragment_id", pa.int64(), nullable=False),
            pa.field("row_offset", pa.int64(), nullable=False),
            pa.field("clip_id", pa.string(), nullable=False),
            spec.caption_field,
            spec.metadata_field,
        ]
    )


def decompose_row_addresses(batch: pa.Table) -> pa.Table:
    """Turn Lance's global row address into physical fragment and local offset."""
    required = {"_rowaddr", "clip_id", "clip_uri", "clip_size_bytes"}
    missing = sorted(required - set(batch.schema.names))
    if missing:
        msg = f"Lance caption read is missing required column(s): {', '.join(missing)}"
        raise ValueError(msg)
    rows = []
    for row in batch.select(["_rowaddr", "clip_id", "clip_uri", "clip_size_bytes"]).to_pylist():
        row_address = int(row["_rowaddr"])
        rows.append(
            {
                "fragment_id": row_address >> _ROW_ADDRESS_FRAGMENT_SHIFT,
                "row_offset": row_address & _ROW_ADDRESS_OFFSET_MASK,
                "clip_id": row["clip_id"],
                "clip_uri": row["clip_uri"],
                "clip_size_bytes": row["clip_size_bytes"],
            }
        )
    return pa.Table.from_pylist(rows, schema=MAPPED_INPUT_SCHEMA)


def _repartition_caption_inputs_for_fetch(
    dataset: ray.data.Dataset,
    *,
    rows_per_block: int,
) -> ray.data.Dataset:
    """Expose enough lightweight input blocks for media fetches to scale cluster-wide."""
    logger.info("Repartitioning caption inputs into strict {}-row blocks before media fetch", rows_per_block)
    return dataset.repartition(target_num_rows_per_block=rows_per_block, strict=True)


def fetch_media_batch(
    batch: pa.Table,
    *,
    storage_profile: str,
    attempts: int,
) -> pa.Table:
    """Fetch exact MP4 bytes, validate length, and emit cross-node data URLs."""
    rows = []
    for row in batch.cast(MAPPED_INPUT_SCHEMA).to_pylist():
        try:
            payload = _read_clip_with_retries(str(row["clip_uri"]), storage_profile=storage_profile, attempts=attempts)
            expected_size = int(row["clip_size_bytes"])
            _validate_clip_size(payload, expected_size=expected_size)
        except _ItemMediaError as exc:
            rows.append(
                row
                | {
                    "video_data_url": None,
                    "download_error_type": type(exc).__name__.removeprefix("_"),
                    "download_error_message": str(exc),
                }
            )
            continue
        encoded = base64.b64encode(payload).decode("ascii")
        rows.append(
            row
            | {
                "video_data_url": f"data:video/mp4;base64,{encoded}",
                "download_error_type": None,
                "download_error_message": None,
            }
        )
    return pa.Table.from_pylist(rows, schema=FETCHED_INPUT_SCHEMA)


def caption_llm_preprocess(row: dict[str, Any]) -> dict[str, Any]:
    """Create the exact Qwen user turn, or mark a download failure to bypass LLM stages."""
    # Ray's processor keeps every top-level input value through its internal
    # stages. Move the large data URL into the message instead of retaining a
    # second reference after multimodal preparation has replaced the message.
    video_data_url = row.pop("video_data_url", None)
    if row.get("download_error_type") is not None:
        return {
            "__inference_error__": f"{row['download_error_type']}: {row['download_error_message']}",
        }
    if not isinstance(video_data_url, str):
        msg = "Successful media fetch must provide a video_data_url string"
        raise TypeError(msg)
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "video_url", "video_url": {"url": video_data_url}},
                    {"type": "text", "text": DEFAULT_PROMPT},
                ],
            }
        ],
        "sampling_params": {
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "min_p": 0.0,
            "presence_penalty": 1.5,
            "repetition_penalty": 1.0,
            "max_tokens": MAX_OUTPUT_TOKENS,
        },
        # Ray's multimodal stage has already decoded and sampled the video. Qwen
        # requires this flag to avoid treating those prepared frames as raw video.
        "mm_processor_kwargs": {"do_sample_frames": False},
    }


def caption_llm_postprocess(row: dict[str, Any]) -> dict[str, Any]:
    """Keep only small inference values and physical identity after success."""
    return {
        "fragment_id": row["fragment_id"],
        "row_offset": row["row_offset"],
        "clip_id": row["clip_id"],
        "caption_text": row["generated_text"],
        "prompt_token_count": row.get("num_input_tokens"),
        "generated_token_count": row.get("num_generated_tokens"),
    }


def canonicalize_llm_batch(
    batch: pa.Table,
    *,
    spec: CaptionModelSpec,
    digest: str,
) -> pa.Table:
    """Convert Ray LLM success/error shapes into the exact terminal schema."""
    rows = []
    for raw in batch.to_pylist():
        error_message = str(raw.get("__inference_error__") or "")
        if error_message:
            error_type = raw.get("download_error_type")
            if not error_type:
                prefix, separator, _ = error_message.partition(":")
                error_type = prefix if separator and prefix else "InferenceError"
            rows.append(
                {
                    "fragment_id": int(raw["fragment_id"]),
                    "row_offset": int(raw["row_offset"]),
                    "clip_id": str(raw["clip_id"]),
                    spec.caption_field_name: None,
                    spec.metadata_field_name: terminal_metadata(
                        spec,
                        digest,
                        status="error",
                        prompt_token_count=None,
                        generated_token_count=None,
                        error_type=str(error_type),
                        error_message=error_message,
                    ),
                }
            )
            continue
        caption_text = raw.get("caption_text")
        if not isinstance(caption_text, str):
            msg = "Successful Ray LLM output must contain string caption_text"
            raise TypeError(msg)
        prompt_tokens = _optional_int(raw.get("prompt_token_count"))
        generated_tokens = _optional_int(raw.get("generated_token_count"))
        status = "truncated" if generated_tokens is not None and generated_tokens >= MAX_OUTPUT_TOKENS else "success"
        rows.append(
            {
                "fragment_id": int(raw["fragment_id"]),
                "row_offset": int(raw["row_offset"]),
                "clip_id": str(raw["clip_id"]),
                spec.caption_field_name: caption_text,
                spec.metadata_field_name: terminal_metadata(
                    spec,
                    digest,
                    status=status,
                    prompt_token_count=prompt_tokens,
                    generated_token_count=generated_tokens,
                    error_type=None,
                    error_message=None,
                ),
            }
        )
    return pa.Table.from_pylist(rows, schema=result_schema(spec))


def _caption_lance_datasource(
    dataset: lance.LanceDataset,
    attempt: CaptionAttempt,
    *,
    storage_options: dict[str, str] | None,
    batch_size: int,
) -> _CaptionLanceDatasource:
    """Snapshot selected-fragment metadata while keeping live handles on the driver."""
    fragments = []
    for fragment_id in attempt.pending_fragment_ids:
        fragment = dataset.get_fragment(fragment_id)
        if fragment is None:
            msg = f"Pending fragment {fragment_id} disappeared from attempt version {attempt.version}"
            raise ValueError(msg)
        fragments.append(
            _CaptionFragmentRead(
                fragment_id=fragment_id,
                num_rows=int(fragment.count_rows()),
                input_files=tuple(
                    str(data_file.path)
                    for data_file in fragment.data_files()  # type: ignore[no-untyped-call]  # Missing in Lance stubs.
                ),
                schema=fragment.schema,
            )
        )
    return _CaptionLanceDatasource(
        uri=dataset.uri,
        version=attempt.version,
        fragments=tuple(fragments),
        storage_options=storage_options,
        batch_size=batch_size,
    )


def _read_caption_fragments(
    *,
    uri: str,
    version: int,
    fragment_ids: tuple[int, ...],
    storage_options: dict[str, str] | None,
    batch_size: int,
) -> Iterator[pa.Table]:
    """Reopen a pinned Lance snapshot with credentials and stream selected fragments."""
    dataset = lance.dataset(uri, version=version, storage_options=storage_options)
    fragments = []
    for fragment_id in fragment_ids:
        fragment = dataset.get_fragment(fragment_id)
        if fragment is None:
            msg = f"Pending fragment {fragment_id} disappeared from attempt version {version}"
            raise ValueError(msg)
        fragments.append(fragment)
    scanner = dataset.scanner(
        columns=list(_CAPTION_INPUT_COLUMNS),
        fragments=fragments,
        with_row_address=True,
        batch_size=batch_size,
    )
    for batch in scanner.to_reader():
        yield pa.Table.from_batches([batch])


def run_inference_phase(  # noqa: PLR0913 -- explicit phase boundary inputs are independently meaningful
    dataset: lance.LanceDataset,
    attempt: CaptionAttempt,
    config: ResolvedVideoCaptionConfig,
    spec: CaptionModelSpec,
    digest: str,
    workspace: CaptionWorkspace,
    *,
    storage_options: dict[str, str] | None,
) -> None:
    """Execute one checkpointed read-to-Parquet plan for the attempt's pending fragments."""
    if not attempt.pending_fragment_ids:
        return
    caption_source = _caption_lance_datasource(
        dataset,
        attempt,
        storage_options=storage_options,
        batch_size=config.execution.lance_read_batch_size,
    )

    checkpoint = CheckpointConfig(
        id_column="clip_id",
        checkpoint_path=workspace.checkpoints_uri,
        delete_checkpoint_on_success=False,
        override_filesystem=workspace.filesystem,
    )
    with installed_checkpoint_config(checkpoint):
        source = ray.data.read_datasource(
            caption_source,
            override_num_blocks=len(attempt.pending_fragment_ids),
        )
        mapped = source.map_batches(
            decompose_row_addresses,
            batch_format="pyarrow",
            batch_size=None,
            zero_copy_batch=True,
        )
        fetch_inputs = _repartition_caption_inputs_for_fetch(
            mapped,
            rows_per_block=config.execution.lance_read_batch_size,
        )
        # Ray accepts fn_kwargs for a keyword-only batch UDF; its public stub does not model that callable shape.
        media_compute = (
            None
            if config.execution.media_concurrency == "auto"
            else ray.data.TaskPoolStrategy(size=config.execution.media_concurrency)
        )
        fetched = fetch_inputs.map_batches(
            fetch_media_batch,  # type: ignore[arg-type]
            batch_format="pyarrow",
            batch_size=config.execution.media_batch_size,
            fn_kwargs={
                "storage_profile": config.execution.storage_profile,
                "attempts": config.execution.media_attempts,
            },
            num_cpus=config.execution.media_cpus,
            resources=_MEDIA_FETCH_RESOURCES,
            compute=media_compute,
        )
        inference_worker_ceiling = _resolve_inference_worker_ceiling(config, caption_source.num_rows)
        gpus_per_replica = config.execution.tensor_parallel_size * config.execution.pipeline_parallel_size
        available_gpus = float(ray.available_resources().get("GPU", 0.0))  # type: ignore[no-untyped-call]
        initial_inference_workers = _resolve_initial_inference_workers(
            available_gpus=available_gpus,
            gpus_per_replica=gpus_per_replica,
            worker_ceiling=inference_worker_ceiling,
        )
        logger.info(
            "Caption inference actor pool will eagerly request {} replica(s) and can scale to {}, after observing {} "
            "currently available GPU(s); each replica requests {} GPU(s) (TP={}, PP={})",
            initial_inference_workers,
            inference_worker_ceiling,
            available_gpus,
            gpus_per_replica,
            config.execution.tensor_parallel_size,
            config.execution.pipeline_parallel_size,
        )
        processor = _build_llm_processor(
            config,
            spec,
            initial_inference_workers=initial_inference_workers,
            inference_worker_ceiling=inference_worker_ceiling,
        )
        inferred = processor(fetched)
        terminal = inferred.map_batches(
            canonicalize_llm_batch,
            batch_format="pyarrow",
            batch_size=config.execution.inference_batch_size,
            fn_kwargs={"spec": spec, "digest": digest},
            zero_copy_batch=True,
        )
        checkpoint_groups = terminal.repartition(
            target_num_rows_per_block=config.execution.parquet_rows_per_file,
            strict=True,
        )
        checkpoint_groups.write_parquet(
            workspace.results_uri,
            filesystem=workspace.filesystem,
            max_rows_per_file=config.execution.parquet_rows_per_file,
        )
    try:
        record_phase_a_completion(workspace, attempt, digest)
    except Exception as exc:  # noqa: BLE001 -- this marker is only a restart optimization
        logger.warning(
            "Phase A results are durable, but writing completion marker {} failed; a restart will let Ray reload "
            "checkpoints instead: {!r}",
            workspace.phase_a_completion_uri,
            exc,
        )
    else:
        logger.info(
            "Recorded Phase A completion for {} pending fragment(s) at {}",
            len(attempt.pending_fragment_ids),
            workspace.phase_a_completion_uri,
        )


@contextmanager
def installed_checkpoint_config(checkpoint: CheckpointConfig) -> Iterator[None]:
    """Install the recipe-owned job checkpoint and always clear it before Phase B."""
    context = ray.data.DataContext.get_current()
    if context.checkpoint_config is not None:
        msg = "video-caption cannot replace an already installed Ray Data checkpoint configuration"
        raise RuntimeError(msg)
    context.checkpoint_config = checkpoint
    try:
        yield
    finally:
        context.checkpoint_config = None


def staged_result_files(workspace: CaptionWorkspace) -> tuple[str, ...]:
    """Return all durable Parquet result paths in stable order."""
    return tuple(
        sorted(
            path
            for path in _files(workspace.filesystem, filesystem_path(workspace.results_uri), recursive=True)
            if path.endswith(".parquet")
        )
    )


def _resolve_inference_worker_ceiling(config: ResolvedVideoCaptionConfig, num_input_rows: int) -> int:
    """Bound the actor pool by pending work, never by a snapshot of cluster GPUs."""
    if num_input_rows < 1:
        msg = f"Caption inference requires at least one input row, got {num_input_rows}"
        raise ValueError(msg)
    configured = config.execution.inference_concurrency
    return num_input_rows if configured == "auto" else min(configured, num_input_rows)


def _resolve_initial_inference_workers(
    *,
    available_gpus: float,
    gpus_per_replica: int,
    worker_ceiling: int,
) -> int:
    """Seed all currently schedulable replicas without turning a resource snapshot into the pool ceiling."""
    if gpus_per_replica < 1:
        msg = f"gpus_per_replica must be positive, got {gpus_per_replica}"
        raise ValueError(msg)
    if worker_ceiling < 1:
        msg = f"worker_ceiling must be positive, got {worker_ceiling}"
        raise ValueError(msg)
    schedulable_replicas = int(max(0.0, available_gpus) // gpus_per_replica)
    return min(worker_ceiling, max(1, schedulable_replicas))


def _build_llm_processor(
    config: ResolvedVideoCaptionConfig,
    spec: CaptionModelSpec,
    *,
    initial_inference_workers: int,
    inference_worker_ceiling: int,
) -> Any:  # noqa: ANN401
    """Build Ray 2.58's vLLM processor against the pre-staged local directory."""
    from ray.data.llm import (  # noqa: PLC0415 -- keep CLI/config import layers light
        ChatTemplateStageConfig,
        DetokenizeStageConfig,
        PrepareMultimodalStageConfig,
        TokenizerStageConfig,
        build_processor,
        vLLMEngineProcessorConfig,
    )

    if not 1 <= initial_inference_workers <= inference_worker_ceiling:
        msg = (
            "initial_inference_workers must be between one and inference_worker_ceiling, got "
            f"{initial_inference_workers} and {inference_worker_ceiling}"
        )
        raise ValueError(msg)
    cpu_stage_concurrency = (initial_inference_workers, inference_worker_ceiling)
    processor_config = vLLMEngineProcessorConfig(
        model_source=str(spec.runtime_model_dir),
        batch_size=config.execution.inference_batch_size,
        concurrency=(initial_inference_workers, inference_worker_ceiling),
        max_concurrent_batches=config.execution.max_concurrent_batches,
        should_continue_on_error=False,
        runtime_env=ray_data_gpu_runtime_env("default"),
        prepare_multimodal_stage=PrepareMultimodalStageConfig(
            enabled=True,
            concurrency=cpu_stage_concurrency,
            memory=_PREPARE_MULTIMODAL_MEMORY_BYTES,
            model_config_kwargs={"media_io_kwargs": {"video": {"num_frames": -1}}},
        ),
        chat_template_stage=ChatTemplateStageConfig(
            enabled=True,
            concurrency=cpu_stage_concurrency,
            memory=_CHAT_TEMPLATE_MEMORY_BYTES,
            chat_template_kwargs={"enable_thinking": False},
        ),
        tokenize_stage=TokenizerStageConfig(enabled=True, concurrency=cpu_stage_concurrency),
        detokenize_stage=DetokenizeStageConfig(enabled=True, concurrency=cpu_stage_concurrency),
        engine_kwargs={
            "tensor_parallel_size": config.execution.tensor_parallel_size,
            "pipeline_parallel_size": config.execution.pipeline_parallel_size,
            "media_io_kwargs": {"video": {"num_frames": -1}},
        },
    )
    processor = build_processor(
        processor_config,
        preprocess=caption_llm_preprocess,
        postprocess=caption_llm_postprocess,
    )
    return _ExactVLLMBatchProcessor(processor, batch_size=config.execution.inference_batch_size)


def _read_clip_with_retries(uri: str, *, storage_profile: str, attempts: int) -> bytes:
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return _read_clip(uri, storage_profile=storage_profile)
        except ClientError as exc:
            code = str(exc.response.get("Error", {}).get("Code", ""))
            if code in _ITEM_S3_ERROR_CODES:
                msg = f"Clip object is unavailable ({code})"
                raise _ItemMediaError(msg) from exc
            status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
            if code not in _RETRYABLE_S3_ERROR_CODES and status not in _RETRYABLE_S3_HTTP_STATUSES:
                msg = f"Shared S3 request failed ({code or type(exc).__name__})"
                raise SharedMediaError(msg) from exc
            last_error = exc
            if attempt < attempts:
                time.sleep(min(2 ** (attempt - 1), 8))
        except (BotoCoreError, OSError) as exc:
            last_error = exc
            if attempt < attempts:
                time.sleep(min(2 ** (attempt - 1), 8))
    msg = f"Shared media storage remained unavailable after {attempts} attempt(s)"
    raise SharedMediaError(msg) from last_error


def _read_clip(uri: str, *, storage_profile: str) -> bytes:
    if uri.lower().startswith("s3://"):
        return download_bytes(uri, storage_profile=storage_profile)
    parsed = urlsplit(uri)
    if parsed.scheme.lower() == "file" and parsed.netloc.lower() in {"", "localhost"}:
        path = Path(unquote(parsed.path))
    elif not parsed.scheme:
        path = Path(uri)
    else:
        msg = f"Unsupported clip URI scheme: {parsed.scheme or '<none>'}"
        raise _ItemMediaError(msg)
    try:
        return path.read_bytes()
    except (FileNotFoundError, IsADirectoryError) as exc:
        msg = "Clip object is unavailable"
        raise _ItemMediaError(msg) from exc


def _validate_clip_size(payload: bytes, *, expected_size: int) -> None:
    if len(payload) != expected_size:
        msg = f"Expected {expected_size} MP4 bytes but downloaded {len(payload)}"
        raise _ItemMediaError(msg)


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        msg = f"Ray LLM token counts must be integers or null, got {type(value).__name__}"
        raise TypeError(msg)
    return value


def _files(filesystem: pafs.FileSystem, root: str, *, recursive: bool) -> list[str]:
    infos = filesystem.get_file_info(pafs.FileSelector(root, recursive=recursive, allow_not_found=True))
    return [info.path for info in infos if info.type == pafs.FileType.File]
