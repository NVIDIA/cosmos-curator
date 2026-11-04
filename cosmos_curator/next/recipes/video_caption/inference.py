# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Caption complete MP4s with Ray Data LLM: exact media fetch, vLLM, and terminal rows.

``caption_rows`` is the processing plan between a recipe's read and its durable
write. Video-caption runs it inside shared enrichment's checkpoint scope;
multimodal-caption's two-phase runner reuses it. A ``CaptionRequest`` names each
row's media, the identity carried into terminal rows, and how to build them.
"""

import base64
import copy
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from functools import partial
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal, Protocol
from urllib.parse import unquote, urlsplit

import pyarrow as pa
import ray
from botocore.exceptions import BotoCoreError, ClientError
from loguru import logger

from cosmos_curator.core.utils import environment
from cosmos_curator.core.utils.pixi_runtime_envs import ray_data_gpu_runtime_env
from cosmos_curator.next.recipes.video_caption.contracts import (
    DEFAULT_PROMPT,
    VIDEO_NUM_FRAMES,
    CaptionInferenceBindings,
    resolved_max_output_tokens,
    resolved_sampling_params,
)
from cosmos_curator.next.utils.storage import download_bytes

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
_VLLM_ENGINE_STAGE_NAME = "vLLMEngineStage"
_BYTES_PER_GIB = 1024**3
_PREPARE_MULTIMODAL_MEMORY_BYTES = 24 * _BYTES_PER_GIB
_CHAT_TEMPLATE_MEMORY_BYTES = 16 * _BYTES_PER_GIB
_MEDIA_FETCH_RESOURCES = {environment.CURATOR_IO_RESOURCE_NAME: 1.0}


TerminalRowBuilder = Callable[..., dict[str, Any]]


class ContextStage(Protocol):
    """Post-fetch context enrichment used by the shared caption plan."""

    def __call__(
        self,
        batch: pa.Table,
        *,
        storage_profile: str,
        attempts: int,
    ) -> pa.Table:
        """Append transient prompt context and declared small result values."""
        ...


class CaptionExecution(Protocol):
    """Media and inference settings the caption plan reads, excluded from caption identity.

    Recipes pass their resolved execution config; ``max_model_len`` is the one
    setting that, when set, also enters the caption contract.
    """

    @property
    def storage_profile(self) -> str:
        """Name the storage credentials for media reads."""
        ...

    @property
    def inference_concurrency(self) -> int | Literal["auto"]:
        """Cap vLLM replicas, or let Ray scale them with work and live GPUs."""
        ...

    @property
    def inference_batch_size(self) -> int:
        """Rows per exact vLLM batch."""
        ...

    @property
    def max_concurrent_batches(self) -> int:
        """Concurrent batch calls per vLLM replica."""
        ...

    @property
    def tensor_parallel_size(self) -> int:
        """GPUs per replica across tensor parallelism."""
        ...

    @property
    def pipeline_parallel_size(self) -> int:
        """GPUs per replica across pipeline parallelism."""
        ...

    @property
    def max_model_len(self) -> int | None:
        """Longest request the engine accepts, or vLLM's default."""
        ...

    @property
    def lance_read_batch_size(self) -> int:
        """Rows per media-fetch task, which recipes also use as their Lance scanner batch size."""
        ...

    @property
    def media_concurrency(self) -> int | Literal["auto"]:
        """Cap concurrent media-fetch tasks, or let Ray scale them within IO slots."""
        ...

    @property
    def media_batch_size(self) -> int:
        """Rows per media-fetch call."""
        ...

    @property
    def media_cpus(self) -> float:
        """CPUs reserved by each media-fetch task."""
        ...

    @property
    def media_attempts(self) -> int:
        """Attempts for transient media storage failures."""
        ...


@dataclass(frozen=True)
class MediaColumns:
    """Input columns naming each row's complete media object and the bytes it must have."""

    uri: str
    size: str
    sha256: str | None = None


@dataclass(frozen=True)
class CaptionRequest:
    """One recipe's caption media, request, and terminal rows, independent of its reads and writes.

    Input rows carry ``fragment_id``, ``row_offset``, the ``identity_columns``
    that reach ``build_terminal_row``, and the ``media`` columns. An optional
    ``context_stage`` renders a per-row prompt and may add the declared
    ``carried_result_columns``. Terminal rows use ``result_schema``.
    """

    media: MediaColumns
    identity_columns: tuple[str, ...]
    prompt_template: str
    build_terminal_row: TerminalRowBuilder
    result_schema: pa.Schema
    inference: CaptionInferenceBindings = field(default_factory=CaptionInferenceBindings)
    context_stage: ContextStage | None = None
    carried_result_columns: tuple[str, ...] = ()


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
    """A credentials or storage-availability failure that must stop processing."""


class ItemMediaError(RuntimeError):
    """A deterministic failure scoped to one media item."""


def fetch_media_batch(
    batch: pa.Table,
    *,
    media: MediaColumns,
    storage_profile: str,
    attempts: int,
) -> pa.Table:
    """Fetch exact MP4 bytes, validate configured evidence, and emit data URLs."""
    rows = []
    fetched_schema = pa.schema(
        [
            *batch.schema,
            pa.field("video_data_url", pa.large_string()),
            pa.field("download_error_type", pa.string()),
            pa.field("download_error_message", pa.large_string()),
        ]
    )
    for row in batch.to_pylist():
        try:
            expected_sha256 = None if media.sha256 is None else str(row[media.sha256])
            payload = read_and_validate_media(
                str(row[media.uri]),
                storage_profile=storage_profile,
                attempts=attempts,
                expected_size=int(row[media.size]),
                expected_sha256=expected_sha256,
            )
        except ItemMediaError as exc:
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
    return pa.Table.from_pylist(rows, schema=fetched_schema)


def caption_llm_preprocess(
    row: dict[str, Any],
    *,
    assistant_prefill: str | None = None,
    prompt_template: str = DEFAULT_PROMPT,
    request_mm_processor_kwargs: Mapping[str, Any],
    sampling_params: Mapping[str, float | int],
) -> dict[str, Any]:
    """Create one model's chat turn, or mark a download failure to bypass LLM stages."""
    # Ray's processor keeps every top-level input value through its internal
    # stages. Move the large data URL into the message instead of retaining a
    # second reference after multimodal preparation has replaced the message.
    video_data_url = row.pop("video_data_url", None)
    rendered_prompt = row.pop("rendered_prompt", None)
    if row.get("download_error_type") is not None:
        return {
            "__inference_error__": f"{row['download_error_type']}: {row['download_error_message']}",
        }
    if not isinstance(video_data_url, str):
        msg = "Successful media fetch must provide a video_data_url string"
        raise TypeError(msg)
    prompt = prompt_template if rendered_prompt is None else rendered_prompt
    if not isinstance(prompt, str):
        msg = "Rendered caption prompt must be a string or null"
        raise TypeError(msg)
    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": [
                {"type": "video_url", "video_url": {"url": video_data_url}},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    if assistant_prefill is not None:
        messages.append({"role": "assistant", "content": assistant_prefill})
    request: dict[str, Any] = {
        "messages": messages,
        "sampling_params": copy.deepcopy(dict(sampling_params)),
    }
    # Ray's multimodal stage has already decoded and sampled the video. Qwen
    # requires its bound flag to avoid treating those prepared frames as raw video.
    if request_mm_processor_kwargs:
        request["mm_processor_kwargs"] = copy.deepcopy(dict(request_mm_processor_kwargs))
    return request


def caption_llm_postprocess(
    row: dict[str, Any],
    *,
    staged_binding_columns: tuple[str, ...],
    carried_result_columns: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Keep only small inference values and physical identity after success."""
    return {
        "fragment_id": row["fragment_id"],
        "row_offset": row["row_offset"],
        **{name: row[name] for name in staged_binding_columns},
        **{name: row[name] for name in carried_result_columns},
        "caption_text": row["generated_text"],
        "prompt_token_count": row.get("num_input_tokens"),
        "generated_token_count": row.get("num_generated_tokens"),
    }


def canonicalize_llm_batch(
    batch: pa.Table,
    *,
    request: CaptionRequest,
) -> pa.Table:
    """Convert Ray LLM success/error shapes into the request's exact terminal schema."""
    max_output_tokens = resolved_max_output_tokens(request.inference)
    rows = []
    for raw in batch.to_pylist():
        error_message = str(raw.get("__inference_error__") or "")
        if error_message:
            error_type = raw.get("download_error_type")
            if not error_type:
                prefix, separator, _ = error_message.partition(":")
                error_type = prefix if separator and prefix else "InferenceError"
            rows.append(
                request.build_terminal_row(
                    raw,
                    caption=None,
                    status="error",
                    prompt_token_count=None,
                    generated_token_count=None,
                    error_type=str(error_type),
                    error_message=error_message,
                )
            )
            continue
        caption_text = raw.get("caption_text")
        if not isinstance(caption_text, str):
            msg = "Successful Ray LLM output must contain string caption_text"
            raise TypeError(msg)
        prompt_tokens = _optional_int(raw.get("prompt_token_count"))
        generated_tokens = _optional_int(raw.get("generated_token_count"))
        status = "truncated" if generated_tokens is not None and generated_tokens >= max_output_tokens else "success"
        rows.append(
            request.build_terminal_row(
                raw,
                caption=caption_text,
                status=status,
                prompt_token_count=prompt_tokens,
                generated_token_count=generated_tokens,
                error_type=None,
                error_message=None,
            )
        )
    return pa.Table.from_pylist(rows, schema=request.result_schema)


def caption_rows(
    rows: ray.data.Dataset,
    request: CaptionRequest,
    execution: CaptionExecution,
    model_dir: Path,
    *,
    max_rows: int,
) -> ray.data.Dataset:
    """Fetch each row's media, caption it with vLLM, and return terminal rows in ``request.result_schema``.

    Each media-fetch task takes ``execution.lance_read_batch_size`` rows.
    ``max_rows`` bounds the replica pool by the work rather than by a snapshot of
    cluster GPUs. Deterministic per-item media failures become terminal error
    rows; shared storage, multimodal preparation, and inference failures raise.
    """
    media_compute = (
        None if execution.media_concurrency == "auto" else ray.data.TaskPoolStrategy(size=execution.media_concurrency)
    )
    # Ray merges each read task's scanner batches into one block and sizes read tasks from the
    # CPUs present when the plan starts. Fixed blocks keep fetch tasks the same size regardless,
    # including the rows a resume leaves after checkpoint filtering.
    rows = rows.repartition(target_num_rows_per_block=execution.lance_read_batch_size, strict=True)
    # Ray accepts fn_kwargs for keyword-only batch UDFs; its public stub does not model that callable shape.
    fetched = rows.map_batches(
        fetch_media_batch,  # type: ignore[arg-type]
        batch_format="pyarrow",
        batch_size=execution.media_batch_size,
        fn_kwargs={
            "media": request.media,
            "storage_profile": execution.storage_profile,
            "attempts": execution.media_attempts,
        },
        num_cpus=execution.media_cpus,
        resources=_MEDIA_FETCH_RESOURCES,
        compute=media_compute,
    )
    if request.context_stage is not None:
        fetched = fetched.map_batches(
            request.context_stage,  # type: ignore[arg-type]
            batch_format="pyarrow",
            batch_size=execution.media_batch_size,
            fn_kwargs={"storage_profile": execution.storage_profile, "attempts": execution.media_attempts},
            num_cpus=execution.media_cpus,
            resources=_MEDIA_FETCH_RESOURCES,
            compute=media_compute,
        )
    inference_worker_ceiling = _resolve_inference_worker_ceiling(execution, max_rows)
    gpus_per_replica = execution.tensor_parallel_size * execution.pipeline_parallel_size
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
        execution.tensor_parallel_size,
        execution.pipeline_parallel_size,
    )
    processor = _build_llm_processor(
        execution,
        model_dir,
        request.identity_columns,
        inference_bindings=request.inference,
        prompt_template=request.prompt_template,
        carried_result_columns=request.carried_result_columns,
        initial_inference_workers=initial_inference_workers,
        inference_worker_ceiling=inference_worker_ceiling,
    )
    return processor(fetched).map_batches(
        canonicalize_llm_batch,  # type: ignore[arg-type]
        batch_format="pyarrow",
        batch_size=execution.inference_batch_size,
        fn_kwargs={"request": request},
        zero_copy_batch=True,
    )


def _resolve_inference_worker_ceiling(config: CaptionExecution, num_input_rows: int) -> int:
    """Bound the actor pool by pending work, never by a snapshot of cluster GPUs."""
    if num_input_rows < 1:
        msg = f"Caption inference requires at least one input row, got {num_input_rows}"
        raise ValueError(msg)
    configured = config.inference_concurrency
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


def _build_llm_processor(  # noqa: PLR0913 -- construction inputs map to independent Ray settings
    config: CaptionExecution,
    model_dir: Path,
    staged_binding_columns: tuple[str, ...],
    *,
    inference_bindings: CaptionInferenceBindings | None = None,
    prompt_template: str = DEFAULT_PROMPT,
    carried_result_columns: tuple[str, ...] = (),
    initial_inference_workers: int,
    inference_worker_ceiling: int,
) -> Callable[[ray.data.Dataset], ray.data.Dataset]:
    """Build Ray 2.58's vLLM processor against the pre-staged local directory."""
    from ray.data.llm import (  # noqa: PLC0415 -- keep CLI/config import layers light
        ChatTemplateStageConfig,
        DetokenizeStageConfig,
        PrepareMultimodalStageConfig,
        TokenizerStageConfig,
        build_processor,
        vLLMEngineProcessorConfig,
    )

    from cosmos_curator.next.core.ray_llm_compat import install_prepare_multimodal_shim  # noqa: PLC0415

    if not 1 <= initial_inference_workers <= inference_worker_ceiling:
        msg = (
            "initial_inference_workers must be between one and inference_worker_ceiling, got "
            f"{initial_inference_workers} and {inference_worker_ceiling}"
        )
        raise ValueError(msg)
    bindings = inference_bindings or CaptionInferenceBindings()
    sampling_params = resolved_sampling_params(bindings)
    reserved_engine_keys = {"tensor_parallel_size", "pipeline_parallel_size", "media_io_kwargs"}
    conflicting_engine_keys = sorted(reserved_engine_keys & set(bindings.engine_kwargs))
    if conflicting_engine_keys:
        msg = f"Model inference bindings cannot override shared engine setting(s): {', '.join(conflicting_engine_keys)}"
        raise ValueError(msg)
    if "media_io_kwargs" in bindings.prepare_model_config_kwargs:
        msg = "Model inference bindings cannot override shared preparation setting: media_io_kwargs"
        raise ValueError(msg)
    cpu_stage_concurrency = (initial_inference_workers, inference_worker_ceiling)
    video_media_io_kwargs = {"video": {"num_frames": VIDEO_NUM_FRAMES, "fps": bindings.video_fps}}
    engine_kwargs = {
        "tensor_parallel_size": config.tensor_parallel_size,
        "pipeline_parallel_size": config.pipeline_parallel_size,
        "media_io_kwargs": copy.deepcopy(video_media_io_kwargs),
        **copy.deepcopy(dict(bindings.engine_kwargs)),
    }
    if config.max_model_len is not None:
        # An explicit execution setting wins over the model binding's own.
        engine_kwargs["max_model_len"] = config.max_model_len
    processor_config = vLLMEngineProcessorConfig(
        model_source=str(model_dir),
        batch_size=config.inference_batch_size,
        concurrency=(initial_inference_workers, inference_worker_ceiling),
        max_concurrent_batches=config.max_concurrent_batches,
        should_continue_on_error=False,
        runtime_env=ray_data_gpu_runtime_env("default"),
        prepare_multimodal_stage=PrepareMultimodalStageConfig(
            enabled=True,
            concurrency=cpu_stage_concurrency,
            memory=_PREPARE_MULTIMODAL_MEMORY_BYTES,
            model_config_kwargs={
                "media_io_kwargs": copy.deepcopy(video_media_io_kwargs),
                **copy.deepcopy(dict(bindings.prepare_model_config_kwargs)),
            },
        ),
        chat_template_stage=ChatTemplateStageConfig(
            enabled=True,
            concurrency=cpu_stage_concurrency,
            memory=_CHAT_TEMPLATE_MEMORY_BYTES,
            chat_template_kwargs=dict(bindings.chat_template_kwargs),
        ),
        tokenize_stage=TokenizerStageConfig(enabled=True, concurrency=cpu_stage_concurrency),
        detokenize_stage=DetokenizeStageConfig(enabled=True, concurrency=cpu_stage_concurrency),
        engine_kwargs=engine_kwargs,
    )
    preprocess = partial(
        caption_llm_preprocess,
        assistant_prefill=bindings.assistant_prefill,
        prompt_template=prompt_template,
        request_mm_processor_kwargs=bindings.request_mm_processor_kwargs,
        sampling_params=sampling_params,
    )
    processor = build_processor(
        processor_config,
        preprocess=preprocess,
        postprocess=partial(
            caption_llm_postprocess,
            staged_binding_columns=staged_binding_columns,
            carried_result_columns=carried_result_columns,
        ),
    )
    install_prepare_multimodal_shim(processor)
    return _ExactVLLMBatchProcessor(processor, batch_size=config.inference_batch_size)


def read_and_validate_media(  # noqa: PLR0913 -- explicit media contract inputs keep callers honest
    uri: str,
    *,
    storage_profile: str,
    attempts: int,
    expected_size: int,
    expected_sha256: str | None = None,
    media_label: str = "MP4",
) -> bytes:
    """Read media with shared retry semantics and validate its descriptor."""
    payload = _read_clip_with_retries(
        uri,
        storage_profile=storage_profile,
        attempts=attempts,
        media_label=media_label,
    )
    _validate_clip_size(payload, expected_size=expected_size, media_label=media_label)
    if expected_sha256 is not None:
        _validate_media_sha256(payload, expected_sha256=expected_sha256, media_label=media_label)
    return payload


def _read_clip_with_retries(uri: str, *, storage_profile: str, attempts: int, media_label: str = "Clip") -> bytes:
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return _read_clip(uri, storage_profile=storage_profile, media_label=media_label)
        except ClientError as exc:
            code = str(exc.response.get("Error", {}).get("Code", ""))
            if code in _ITEM_S3_ERROR_CODES:
                msg = f"{media_label} object is unavailable ({code})"
                raise ItemMediaError(msg) from exc
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


def _read_clip(uri: str, *, storage_profile: str, media_label: str = "Clip") -> bytes:
    if uri.lower().startswith("s3://"):
        return download_bytes(uri, storage_profile=storage_profile)
    parsed = urlsplit(uri)
    if parsed.scheme.lower() == "file" and parsed.netloc.lower() in {"", "localhost"}:
        path = Path(unquote(parsed.path))
    elif not parsed.scheme:
        path = Path(uri)
    else:
        msg = f"Unsupported {media_label} URI scheme: {parsed.scheme or '<none>'}"
        raise ItemMediaError(msg)
    try:
        return path.read_bytes()
    except (FileNotFoundError, IsADirectoryError) as exc:
        msg = f"{media_label} object is unavailable"
        raise ItemMediaError(msg) from exc


def _validate_clip_size(payload: bytes, *, expected_size: int, media_label: str = "MP4") -> None:
    if len(payload) != expected_size:
        msg = f"Expected {expected_size} {media_label} bytes but downloaded {len(payload)}"
        raise ItemMediaError(msg)


def _validate_media_sha256(payload: bytes, *, expected_sha256: str, media_label: str = "MP4") -> None:
    actual = sha256(payload).hexdigest()
    if actual != expected_sha256:
        msg = f"Expected {media_label} SHA-256 {expected_sha256} but downloaded {actual}"
        raise ItemMediaError(msg)


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        msg = f"Ray LLM token counts must be integers or null, got {type(value).__name__}"
        raise TypeError(msg)
    return value
