# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the shared caption plan: media fetch, vLLM processor construction, and terminal rows."""

import base64
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pyarrow as pa
import pytest
import ray.data.llm as ray_data_llm
from botocore.exceptions import ClientError
from ray.llm._internal.batch.stages import prepare_multimodal_stage

from cosmos_curator.next.core import ray_llm_compat
from cosmos_curator.next.core.lance_datasource import ROW_ADDRESS_FIELDS
from cosmos_curator.next.recipes.video_caption import inference
from cosmos_curator.next.recipes.video_caption.config import resolve_config_data
from cosmos_curator.next.recipes.video_caption.contracts import (
    DEFAULT_PROMPT,
    MAX_OUTPUT_TOKENS,
    CaptionInferenceBindings,
    CaptionModelSpec,
    resolved_sampling_params,
)
from cosmos_curator.next.recipes.video_caption.inference import (
    CaptionRequest,
    MediaColumns,
    SharedMediaError,
    canonicalize_llm_batch,
    caption_llm_postprocess,
    caption_llm_preprocess,
    fetch_media_batch,
)
from cosmos_curator.next.recipes.video_transcode.contracts import CLIP_SCHEMA

_CLIP_MEDIA = MediaColumns(uri="clip_uri", size="clip_size_bytes")
# Rows as the shared enrichment reader yields them for READ_COLUMNS.
_WORK_SCHEMA = pa.schema(
    [
        CLIP_SCHEMA.field("clip_id"),
        CLIP_SCHEMA.field("clip_uri"),
        CLIP_SCHEMA.field("clip_size_bytes"),
        *ROW_ADDRESS_FIELDS,
    ]
)


def _annotate_batch_size(batch: pa.Table) -> pa.Table:
    """Expose the batch size seen by a test stage on every output row."""
    return batch.append_column("observed_batch_size", pa.array([batch.num_rows] * batch.num_rows, type=pa.int64()))


def _s3_client_error(code: str, status: int) -> ClientError:
    return ClientError(
        {
            "Error": {"Code": code, "Message": "request failed"},
            "ResponseMetadata": {"HTTPStatusCode": status},
        },
        "GetObject",
    )


def test_media_fetch_uses_exact_bytes_and_cross_node_data_url(tmp_path: Path) -> None:
    """Media bytes are length-checked and embedded in an MP4 data URL."""
    payload = b"\x00\x00\x00\x18ftypmp42"
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(payload)
    source = pa.Table.from_pylist(
        [
            {
                "fragment_id": 0,
                "row_offset": 0,
                "clip_id": "clip",
                "clip_uri": str(clip),
                "clip_size_bytes": len(payload),
            }
        ],
        schema=_WORK_SCHEMA,
    )

    fetched = fetch_media_batch(
        source,
        media=_CLIP_MEDIA,
        storage_profile="default",
        attempts=2,
    ).to_pylist()[0]

    assert fetched["video_data_url"] == f"data:video/mp4;base64,{base64.b64encode(payload).decode('ascii')}"
    assert fetched["download_error_type"] is None
    assert fetched["download_error_message"] is None


@pytest.mark.parametrize(
    ("uri", "size", "message"),
    [
        ("/does/not/exist.mp4", 10, "unavailable"),
        ("existing", 99, "Expected 99 MP4 bytes"),
        ("https://example.com/clip.mp4", 10, "Unsupported MP4 URI scheme"),
    ],
)
def test_item_media_failures_become_rows(
    tmp_path: Path,
    uri: str,
    size: int,
    message: str,
) -> None:
    """Deterministic missing, malformed, and unsupported media become item errors."""
    if uri == "existing":
        path = tmp_path / "clip.mp4"
        path.write_bytes(b"short")
        uri = str(path)
    source = pa.Table.from_pylist(
        [{"fragment_id": 0, "row_offset": 0, "clip_id": "clip", "clip_uri": uri, "clip_size_bytes": size}],
        schema=_WORK_SCHEMA,
    )

    fetched = fetch_media_batch(
        source,
        media=_CLIP_MEDIA,
        storage_profile="default",
        attempts=1,
    ).to_pylist()[0]

    assert fetched["video_data_url"] is None
    assert fetched["download_error_type"] == "ItemMediaError"
    assert message in fetched["download_error_message"]


def test_shared_media_failure_stops_the_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Repeated shared filesystem failures stop processing instead of becoming rows."""
    source = pa.Table.from_pylist(
        [{"fragment_id": 0, "row_offset": 0, "clip_id": "clip", "clip_uri": "/clip", "clip_size_bytes": 1}],
        schema=_WORK_SCHEMA,
    )
    attempts = 0

    def unavailable(uri: str, *, storage_profile: str, media_label: str) -> bytes:
        del uri, storage_profile, media_label
        nonlocal attempts
        attempts += 1
        msg = "shared mount unavailable"
        raise OSError(msg)

    monkeypatch.setattr(inference, "_read_clip", unavailable)
    monkeypatch.setattr(inference.time, "sleep", lambda _: None)

    with pytest.raises(SharedMediaError, match="after 3 attempt"):
        fetch_media_batch(
            source,
            media=_CLIP_MEDIA,
            storage_profile="default",
            attempts=3,
        )
    assert attempts == 3


@pytest.mark.parametrize(
    ("code", "status"),
    [("SlowDown", 503), ("RequestTimeout", 400), ("503", 503)],
)
def test_transient_s3_client_errors_use_the_retry_budget(
    code: str,
    status: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Transient service responses retry and can recover within the configured budget."""
    calls = 0

    def flaky_read(uri: str, *, storage_profile: str, media_label: str) -> bytes:
        del uri, storage_profile, media_label
        nonlocal calls
        calls += 1
        if calls < 3:
            raise _s3_client_error(code, status)
        return b"clip"

    monkeypatch.setattr(inference, "_read_clip", flaky_read)
    monkeypatch.setattr(inference.time, "sleep", lambda _: None)

    payload = inference._read_clip_with_retries("s3://bucket/clip.mp4", storage_profile="default", attempts=3)

    assert payload == b"clip"
    assert calls == 3


def test_permanent_s3_client_error_fails_without_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Authentication and other permanent shared failures stop the phase immediately."""
    calls = 0

    def denied(uri: str, *, storage_profile: str, media_label: str) -> bytes:
        del uri, storage_profile, media_label
        nonlocal calls
        calls += 1
        code = "AccessDenied"
        raise _s3_client_error(code, 403)

    monkeypatch.setattr(inference, "_read_clip", denied)

    with pytest.raises(SharedMediaError, match="AccessDenied"):
        inference._read_clip_with_retries("s3://bucket/clip.mp4", storage_profile="default", attempts=3)
    assert calls == 1


def test_preprocess_matches_the_caption_contract() -> None:
    """Preprocessing emits the exact Qwen message and does not resample prepared frames."""
    video_data_url = "data:video/mp4;base64,AAAA"
    row = {"video_data_url": video_data_url, "download_error_type": None}

    output = caption_llm_preprocess(
        row,
        request_mm_processor_kwargs=CaptionInferenceBindings().request_mm_processor_kwargs,
        sampling_params=resolved_sampling_params(CaptionInferenceBindings()),
    )

    assert output["messages"] == [
        {
            "role": "user",
            "content": [
                {"type": "video_url", "video_url": {"url": video_data_url}},
                {"type": "text", "text": DEFAULT_PROMPT},
            ],
        }
    ]
    assert "video_data_url" not in row
    assert output["sampling_params"] == {
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 1.5,
        "repetition_penalty": 1.0,
        "max_tokens": MAX_OUTPUT_TOKENS,
    }
    assert output["mm_processor_kwargs"] == {"do_sample_frames": False}


def test_download_error_bypasses_llm() -> None:
    """A terminal download failure is marked for every LLM stage to bypass."""
    output = caption_llm_preprocess(
        {"download_error_type": "ItemMediaError", "download_error_message": "missing"},
        request_mm_processor_kwargs=CaptionInferenceBindings().request_mm_processor_kwargs,
        sampling_params=resolved_sampling_params(CaptionInferenceBindings()),
    )

    assert output == {"__inference_error__": "ItemMediaError: missing"}


def test_preprocess_prefers_exact_rendered_prompt_without_formatting() -> None:
    """A context stage can supply one exact prompt while unrelated braces remain literal."""
    rendered = "context {unrelated} bytes"
    output = caption_llm_preprocess(
        {
            "video_data_url": "data:video/mp4;base64,AAAA",
            "rendered_prompt": rendered,
            "download_error_type": None,
        },
        prompt_template="fallback must not be used",
        request_mm_processor_kwargs={},
        sampling_params={"max_tokens": 1},
    )

    assert output["messages"][0]["content"][1] == {"type": "text", "text": rendered}


def test_download_error_preserves_identity_through_terminal_adapter(
    caption_spec: CaptionModelSpec,
    caption_request: CaptionRequest,
) -> None:
    """Download failure normalization retains physical identity in the terminal row."""
    source = {
        "fragment_id": 2,
        "row_offset": 3,
        "clip_id": "clip",
        "video_data_url": None,
        "download_error_type": "ItemMediaError",
        "download_error_message": "missing",
    }
    preprocessed = source | caption_llm_preprocess(
        source,
        request_mm_processor_kwargs=CaptionInferenceBindings().request_mm_processor_kwargs,
        sampling_params=resolved_sampling_params(CaptionInferenceBindings()),
    )
    terminal = canonicalize_llm_batch(
        pa.Table.from_pylist([preprocessed]),
        request=caption_request,
    ).to_pylist()[0]

    assert (terminal["fragment_id"], terminal["row_offset"], terminal["clip_id"]) == (2, 3, "clip")
    assert terminal[caption_spec.metadata_field_name]["status"] == "error"


def test_llm_processor_uses_public_builtin_multimodal_stage(
    tmp_path: Path,
    caption_spec: CaptionModelSpec,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Processor construction leaves Ray's configured public multimodal stage intact."""
    config = resolve_config_data(
        {
            "schema_version": 1,
            "kind": "video-caption",
            "input": {
                "media_root": str(tmp_path / "media"),
                "clips_lance_uri": str(tmp_path / "clips.lance"),
            },
            "model": {"variant": caption_spec.variant},
            "output": {"staging_root_uri": str(tmp_path / "staging")},
        }
    )

    class _BuiltProcessor:
        def __init__(self, processor_config: object) -> None:
            self.config = processor_config

        def list_stage_names(self) -> list[str]:
            return ["vLLMEngineStage"]

    captured: dict[str, object] = {}

    def build_processor(
        processor_config: object,
        *,
        preprocess: object,
        postprocess: object,
    ) -> object:
        captured.update(config=processor_config, preprocess=preprocess, postprocess=postprocess)
        return _BuiltProcessor(processor_config)

    monkeypatch.setattr(ray_data_llm, "build_processor", build_processor)

    processor = inference._build_llm_processor(
        config.execution,
        caption_spec.runtime_model_dir,
        ("clip_id",),
        initial_inference_workers=3,
        inference_worker_ceiling=4,
    )

    assert isinstance(processor, inference._ExactVLLMBatchProcessor)
    preprocess = cast("Any", captured["preprocess"])
    assert preprocess.func is caption_llm_preprocess
    assert preprocess.keywords == {
        "assistant_prefill": None,
        "prompt_template": DEFAULT_PROMPT,
        "request_mm_processor_kwargs": {"do_sample_frames": False},
        "sampling_params": {
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "min_p": 0.0,
            "presence_penalty": 1.5,
            "repetition_penalty": 1.0,
            "max_tokens": 2_048,
        },
    }
    assert cast("Any", captured["postprocess"]).func is caption_llm_postprocess
    processor_config = captured["config"]
    assert isinstance(processor_config, ray_data_llm.vLLMEngineProcessorConfig)
    assert processor_config.batch_size == 32
    assert processor_config.concurrency == (3, 4)
    assert processor_config.max_concurrent_batches == 8
    assert processor_config.max_tasks_in_flight_per_actor is None
    assert processor_config.should_continue_on_error is False
    multimodal_config = processor_config.prepare_multimodal_stage
    assert isinstance(multimodal_config, ray_data_llm.PrepareMultimodalStageConfig)
    assert multimodal_config.enabled is True
    assert multimodal_config.concurrency == (3, 4)
    assert multimodal_config.memory == 24 * 1024**3
    expected_media_io_kwargs = {"media_io_kwargs": {"video": {"num_frames": -1, "fps": 2}}}
    assert multimodal_config.model_config_kwargs == expected_media_io_kwargs
    chat_template_config = processor_config.chat_template_stage
    assert isinstance(chat_template_config, ray_data_llm.ChatTemplateStageConfig)
    assert chat_template_config.concurrency == (3, 4)
    assert chat_template_config.memory == 16 * 1024**3
    assert processor_config.engine_kwargs["media_io_kwargs"] == expected_media_io_kwargs["media_io_kwargs"]


def test_llm_processor_runs_media_preparation_through_the_vllm_compat_udf(
    tmp_path: Path,
    caption_spec: CaptionModelSpec,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ray's own preparation UDF cannot import the pinned vLLM's chat parser."""
    config = resolve_config_data(
        {
            "schema_version": 1,
            "kind": "video-caption",
            "input": {
                "media_root": str(tmp_path / "media"),
                "clips_lance_uri": str(tmp_path / "clips.lance"),
            },
            "model": {"variant": caption_spec.variant},
            "output": {"staging_root_uri": str(tmp_path / "staging")},
        }
    )
    stage = prepare_multimodal_stage.PrepareMultimodalStage(fn_constructor_kwargs={}, map_batches_kwargs={})

    class _BuiltProcessor:
        def __init__(self, processor_config: object) -> None:
            self.config = processor_config

        def list_stage_names(self) -> list[str]:
            return [ray_llm_compat.PREPARE_MULTIMODAL_STAGE, "vLLMEngineStage"]

        def get_stage_by_name(self, name: str) -> object:
            assert name == ray_llm_compat.PREPARE_MULTIMODAL_STAGE
            return stage

    def build_processor(processor_config: object, **_: object) -> object:
        return _BuiltProcessor(processor_config)

    monkeypatch.setattr(ray_data_llm, "build_processor", build_processor)

    inference._build_llm_processor(
        config.execution,
        caption_spec.runtime_model_dir,
        ("clip_id",),
        initial_inference_workers=1,
        inference_worker_ceiling=1,
    )

    assert stage.fn is ray_llm_compat.PrepareMultimodalUDF


def test_llm_processor_applies_model_runtime_bindings(
    tmp_path: Path,
    caption_spec: CaptionModelSpec,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Static model bindings reach Ray's chat and vLLM stages without changing execution sizing."""
    config = resolve_config_data(
        {
            "schema_version": 1,
            "kind": "video-caption",
            "input": {
                "media_root": str(tmp_path / "media"),
                "clips_lance_uri": str(tmp_path / "clips.lance"),
            },
            "model": {"variant": caption_spec.variant},
            "output": {"staging_root_uri": str(tmp_path / "staging")},
        }
    )

    class _BuiltProcessor:
        def __init__(self, processor_config: object) -> None:
            self.config = processor_config

        def list_stage_names(self) -> list[str]:
            return ["vLLMEngineStage"]

    captured: dict[str, object] = {}

    def build_processor(
        processor_config: object,
        *,
        preprocess: object,
        postprocess: object,
    ) -> object:
        captured.update(config=processor_config, preprocess=preprocess, postprocess=postprocess)
        return _BuiltProcessor(processor_config)

    monkeypatch.setattr(ray_data_llm, "build_processor", build_processor)
    bindings = CaptionInferenceBindings(
        assistant_prefill="<think></think>",
        chat_template_kwargs={},
        video_fps=4,
        prepare_model_config_kwargs={"hf_overrides": {"architectures": ["Cosmos3ForConditionalGeneration"]}},
        request_mm_processor_kwargs={},
        engine_kwargs={"mm_encoder_tp_mode": "data"},
    )

    inference._build_llm_processor(
        config.execution,
        caption_spec.runtime_model_dir,
        ("clip_id",),
        inference_bindings=bindings,
        initial_inference_workers=1,
        inference_worker_ceiling=1,
    )

    preprocess = cast("Any", captured["preprocess"])
    assert preprocess.func is caption_llm_preprocess
    assert preprocess.keywords == {
        "assistant_prefill": "<think></think>",
        "prompt_template": DEFAULT_PROMPT,
        "request_mm_processor_kwargs": {},
        "sampling_params": {
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "min_p": 0.0,
            "presence_penalty": 1.5,
            "repetition_penalty": 1.0,
            "max_tokens": 2_048,
        },
    }
    processor_config = captured["config"]
    assert isinstance(processor_config, ray_data_llm.vLLMEngineProcessorConfig)
    assert processor_config.chat_template_stage.chat_template_kwargs == {}
    assert processor_config.prepare_multimodal_stage.model_config_kwargs == {
        "media_io_kwargs": {"video": {"num_frames": -1, "fps": 4}},
        "hf_overrides": {"architectures": ["Cosmos3ForConditionalGeneration"]},
    }
    assert processor_config.engine_kwargs["media_io_kwargs"] == {"video": {"num_frames": -1, "fps": 4}}
    assert processor_config.engine_kwargs["mm_encoder_tp_mode"] == "data"


@pytest.mark.parametrize(
    ("execution", "binding_engine_kwargs", "expected"),
    [
        pytest.param({}, {}, None, id="unset-leaves-vllm-default"),
        pytest.param({"max_model_len": 32_768}, {}, 32_768, id="execution-setting-reaches-engine"),
        pytest.param({"max_model_len": 32_768}, {"max_model_len": 8_192}, 32_768, id="execution-wins-over-binding"),
        pytest.param({}, {"max_model_len": 8_192}, 8_192, id="binding-kept-when-unset"),
    ],
)
def test_llm_processor_max_model_len_resolution(
    tmp_path: Path,
    caption_spec: CaptionModelSpec,
    monkeypatch: pytest.MonkeyPatch,
    execution: dict[str, object],
    binding_engine_kwargs: dict[str, object],
    expected: int | None,
) -> None:
    """An explicit execution limit overrides the model binding; otherwise the engine default is untouched."""
    config = resolve_config_data(
        {
            "schema_version": 1,
            "kind": "video-caption",
            "input": {
                "media_root": str(tmp_path / "media"),
                "clips_lance_uri": str(tmp_path / "clips.lance"),
            },
            "model": {"variant": caption_spec.variant},
            "output": {"staging_root_uri": str(tmp_path / "staging")},
            "execution": execution,
        }
    )
    captured_configs: list[object] = []

    class _BuiltProcessor:
        def __init__(self, processor_config: object) -> None:
            self.config = processor_config

        def list_stage_names(self) -> list[str]:
            return ["vLLMEngineStage"]

    def build_processor(processor_config: object, **_: object) -> object:
        captured_configs.append(processor_config)
        return _BuiltProcessor(processor_config)

    monkeypatch.setattr(ray_data_llm, "build_processor", build_processor)

    inference._build_llm_processor(
        config.execution,
        caption_spec.runtime_model_dir,
        ("clip_id",),
        inference_bindings=CaptionInferenceBindings(engine_kwargs=binding_engine_kwargs),
        initial_inference_workers=1,
        inference_worker_ceiling=1,
    )

    (processor_config,) = (cast("Any", captured) for captured in captured_configs)
    assert processor_config.engine_kwargs.get("max_model_len") == expected


def test_llm_processor_deep_copies_model_engine_kwargs(
    tmp_path: Path,
    caption_spec: CaptionModelSpec,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each processor build receives fresh nested model engine settings."""
    config = resolve_config_data(
        {
            "schema_version": 1,
            "kind": "video-caption",
            "input": {
                "media_root": str(tmp_path / "media"),
                "clips_lance_uri": str(tmp_path / "clips.lance"),
            },
            "model": {"variant": caption_spec.variant},
            "output": {"staging_root_uri": str(tmp_path / "staging")},
        }
    )
    captured_configs: list[object] = []

    class _BuiltProcessor:
        def __init__(self, processor_config: object) -> None:
            self.config = processor_config

        def list_stage_names(self) -> list[str]:
            return ["vLLMEngineStage"]

    def build_processor(processor_config: object, **_: object) -> object:
        captured_configs.append(processor_config)
        return _BuiltProcessor(processor_config)

    monkeypatch.setattr(ray_data_llm, "build_processor", build_processor)
    bindings = CaptionInferenceBindings(
        prepare_model_config_kwargs={"hf_overrides": {"architectures": ["Cosmos3ForConditionalGeneration"]}},
        engine_kwargs={"limit_mm_per_prompt": {"images": 0, "video": 1}},
    )

    for _ in range(2):
        inference._build_llm_processor(
            config.execution,
            caption_spec.runtime_model_dir,
            ("clip_id",),
            inference_bindings=bindings,
            initial_inference_workers=1,
            inference_worker_ceiling=1,
        )

    first, second = (cast("Any", processor_config) for processor_config in captured_configs)
    first.engine_kwargs["limit_mm_per_prompt"]["video"] = 9
    first.prepare_multimodal_stage.model_config_kwargs["hf_overrides"]["architectures"][0] = "other"

    assert second.engine_kwargs["limit_mm_per_prompt"] == {"images": 0, "video": 1}
    assert second.prepare_multimodal_stage.model_config_kwargs["hf_overrides"] == {
        "architectures": ["Cosmos3ForConditionalGeneration"]
    }
    assert bindings.prepare_model_config_kwargs["hf_overrides"] == {
        "architectures": ["Cosmos3ForConditionalGeneration"]
    }
    assert bindings.engine_kwargs["limit_mm_per_prompt"] == {"images": 0, "video": 1}


@pytest.mark.parametrize(
    "reserved_key",
    ["tensor_parallel_size", "pipeline_parallel_size", "media_io_kwargs"],
)
def test_llm_processor_rejects_model_overrides_of_shared_engine_settings(
    tmp_path: Path,
    caption_spec: CaptionModelSpec,
    reserved_key: str,
) -> None:
    """A model binding cannot retune parallelism or frame decode behind the shared builder."""
    config = resolve_config_data(
        {
            "schema_version": 1,
            "kind": "video-caption",
            "input": {
                "media_root": str(tmp_path / "media"),
                "clips_lance_uri": str(tmp_path / "clips.lance"),
            },
            "model": {"variant": caption_spec.variant},
            "output": {"staging_root_uri": str(tmp_path / "staging")},
        }
    )
    bindings = CaptionInferenceBindings(engine_kwargs={reserved_key: "ignored"})

    with pytest.raises(ValueError, match=rf"cannot override shared engine setting\(s\): {reserved_key}"):
        inference._build_llm_processor(
            config.execution,
            caption_spec.runtime_model_dir,
            ("clip_id",),
            inference_bindings=bindings,
            initial_inference_workers=1,
            inference_worker_ceiling=1,
        )


def test_llm_processor_rejects_model_override_of_shared_preparation_setting(
    tmp_path: Path,
    caption_spec: CaptionModelSpec,
) -> None:
    """A model binding cannot replace the shared prepared-video settings."""
    config = resolve_config_data(
        {
            "schema_version": 1,
            "kind": "video-caption",
            "input": {
                "media_root": str(tmp_path / "media"),
                "clips_lance_uri": str(tmp_path / "clips.lance"),
            },
            "model": {"variant": caption_spec.variant},
            "output": {"staging_root_uri": str(tmp_path / "staging")},
        }
    )

    with pytest.raises(ValueError, match="cannot override shared preparation setting: media_io_kwargs"):
        inference._build_llm_processor(
            config.execution,
            caption_spec.runtime_model_dir,
            ("clip_id",),
            inference_bindings=CaptionInferenceBindings(prepare_model_config_kwargs={"media_io_kwargs": {}}),
            initial_inference_workers=1,
            inference_worker_ceiling=1,
        )


def test_exact_vllm_batch_processor_inserts_strict_repartition() -> None:
    """The processor plan places an exact-row repartition immediately before vLLM."""

    class _Stage:
        fn = staticmethod(_annotate_batch_size)

        def get_dataset_map_batches_kwargs(self, *, batch_size: int, data_column: str) -> dict[str, object]:
            assert data_column == "__data"
            return {"batch_size": batch_size, "batch_format": "pyarrow", "zero_copy_batch": True}

    class _Processor:
        DATA_COLUMN = "__data"
        config = SimpleNamespace(batch_size=32)

        def __init__(self) -> None:
            self.preprocess = None
            self.postprocess = None
            self.preprocess_map_kwargs: dict[str, object] = {}
            self.postprocess_map_kwargs: dict[str, object] = {}
            self.stage = _Stage()

        def list_stage_names(self) -> list[str]:
            return ["vLLMEngineStage"]

        def get_stage_by_name(self, name: str) -> _Stage:
            assert name == "vLLMEngineStage"
            return self.stage

    class _DatasetPlan:
        def __init__(self) -> None:
            self.calls: list[tuple[str, object]] = []

        def repartition(self, *, target_num_rows_per_block: int, strict: bool) -> "_DatasetPlan":
            self.calls.append(
                (
                    "repartition",
                    {"target_num_rows_per_block": target_num_rows_per_block, "strict": strict},
                )
            )
            return self

        def map_batches(self, function: object, **kwargs: object) -> "_DatasetPlan":
            self.calls.append(("map_batches", (function, kwargs)))
            return self

    source = _DatasetPlan()
    result = inference._ExactVLLMBatchProcessor(_Processor(), batch_size=32)(source)  # type: ignore[arg-type]

    assert result is source
    assert source.calls[0] == ("repartition", {"target_num_rows_per_block": 32, "strict": True})
    assert source.calls[1][0] == "map_batches"
    function, kwargs = source.calls[1][1]
    assert function is _annotate_batch_size
    assert kwargs == {"batch_size": 32, "batch_format": "pyarrow", "zero_copy_batch": True}


@pytest.mark.parametrize(
    ("configured", "num_input_rows", "expected"),
    [("auto", 12, 12), (3, 12, 3), (20, 12, 12)],
)
def test_inference_worker_ceiling_is_work_bounded(
    tmp_path: Path,
    configured: str | int,
    num_input_rows: int,
    expected: int,
) -> None:
    """The actor ceiling follows work and an optional user cap, not current cluster size."""
    config = resolve_config_data(
        {
            "schema_version": 1,
            "kind": "video-caption",
            "input": {"media_root": str(tmp_path / "media")},
            "model": {"variant": "qwen3_8_27b_fp8"},
            "execution": {"inference_concurrency": configured},
        }
    )

    assert inference._resolve_inference_worker_ceiling(config.execution, num_input_rows) == expected


@pytest.mark.parametrize(
    ("available_gpus", "gpus_per_replica", "worker_ceiling", "expected"),
    [(0.0, 1, 12, 1), (4.0, 1, 12, 4), (8.0, 4, 12, 2), (16.0, 1, 3, 3)],
)
def test_initial_inference_workers_seed_live_gpus_without_capping_growth(
    available_gpus: float,
    gpus_per_replica: int,
    worker_ceiling: int,
    expected: int,
) -> None:
    """The eager floor follows current capacity while the independent work ceiling remains authoritative."""
    assert (
        inference._resolve_initial_inference_workers(
            available_gpus=available_gpus,
            gpus_per_replica=gpus_per_replica,
            worker_ceiling=worker_ceiling,
        )
        == expected
    )


def test_postprocess_keeps_only_small_identity_and_token_values() -> None:
    """Successful LLM output drops media and multimodal intermediates."""
    output = caption_llm_postprocess(
        {
            "fragment_id": 2,
            "row_offset": 3,
            "clip_id": "clip",
            "generated_text": "caption",
            "num_input_tokens": 4,
            "num_generated_tokens": 5,
            "video_data_url": "large",
        },
        staged_binding_columns=("clip_id",),
    )

    assert output == {
        "fragment_id": 2,
        "row_offset": 3,
        "clip_id": "clip",
        "caption_text": "caption",
        "prompt_token_count": 4,
        "generated_token_count": 5,
    }


def test_postprocess_carries_only_declared_small_context_values() -> None:
    """Context digest crosses the shared engine boundary without transient prompt or media bytes."""
    output = caption_llm_postprocess(
        {
            "fragment_id": 2,
            "row_offset": 3,
            "episode_id": "episode",
            "context_digest": "c" * 64,
            "rendered_prompt": "large prompt",
            "video_data_url": "large video",
            "generated_text": "caption",
        },
        staged_binding_columns=("episode_id",),
        carried_result_columns=("context_digest",),
    )

    assert output == {
        "fragment_id": 2,
        "row_offset": 3,
        "episode_id": "episode",
        "context_digest": "c" * 64,
        "caption_text": "caption",
        "prompt_token_count": None,
        "generated_token_count": None,
    }


def test_canonicalization_emits_success_truncated_and_error_rows(
    caption_spec: CaptionModelSpec,
    caption_request: CaptionRequest,
) -> None:
    """Ray LLM shapes normalize into all three canonical terminal states."""
    raw = pa.Table.from_pylist(
        [
            {
                "fragment_id": 0,
                "row_offset": 0,
                "clip_id": "success",
                "caption_text": "caption",
                "prompt_token_count": 7,
                "generated_token_count": 9,
                "download_error_type": None,
                "__inference_error__": "",
            },
            {
                "fragment_id": 0,
                "row_offset": 1,
                "clip_id": "truncated",
                "caption_text": "long caption",
                "prompt_token_count": 7,
                "generated_token_count": MAX_OUTPUT_TOKENS,
                "download_error_type": None,
                "__inference_error__": "",
            },
            {
                "fragment_id": 0,
                "row_offset": 2,
                "clip_id": "error",
                "download_error_type": "ItemMediaError",
                "__inference_error__": "ItemMediaError: missing",
            },
        ]
    )

    terminal = canonicalize_llm_batch(raw, request=caption_request)
    rows = terminal.to_pylist()

    assert terminal.schema.equals(caption_request.result_schema, check_metadata=True)
    assert [row[caption_spec.metadata_field_name]["status"] for row in rows] == ["success", "truncated", "error"]
    assert rows[2][caption_spec.caption_field_name] is None
    assert rows[2][caption_spec.metadata_field_name]["error_type"] == "ItemMediaError"


def test_canonicalization_uses_the_model_bound_output_limit(
    caption_spec: CaptionModelSpec,
    caption_request: CaptionRequest,
) -> None:
    """Terminal state follows the model binding's request limit, not a module default."""
    request = replace(caption_request, inference=CaptionInferenceBindings(sampling_params={"max_tokens": 5}))
    raw = pa.Table.from_pylist(
        [
            {
                "fragment_id": 0,
                "row_offset": 0,
                "clip_id": "model-bound-limit",
                "caption_text": "caption",
                "prompt_token_count": 7,
                "generated_token_count": 5,
                "download_error_type": None,
                "__inference_error__": "",
            }
        ]
    )

    terminal = canonicalize_llm_batch(raw, request=request).to_pylist()[0]

    assert terminal[caption_spec.metadata_field_name]["status"] == "truncated"
