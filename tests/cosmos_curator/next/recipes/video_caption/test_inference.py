# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for checkpoint, media, and terminal inference adapters."""

import base64
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import lance
import pyarrow as pa
import pyarrow.fs as pafs
import pyarrow.parquet as pq
import pytest
import ray
import ray.cloudpickle as ray_cloudpickle
import ray.data.llm as ray_data_llm
from botocore.exceptions import ClientError
from lance.fragment import LanceFragment
from ray.data.checkpoint import CheckpointConfig

from cosmos_curator.next.recipes.video_caption import inference
from cosmos_curator.next.recipes.video_caption.bindings import CaptionInferenceBindings, CaptionRunBindings
from cosmos_curator.next.recipes.video_caption.config import resolve_config_data
from cosmos_curator.next.recipes.video_caption.contracts import (
    DEFAULT_PROMPT,
    MAX_OUTPUT_TOKENS,
    CaptionModelSpec,
    resolved_sampling_params,
)
from cosmos_curator.next.recipes.video_caption.inference import (
    MAPPED_INPUT_SCHEMA,
    SharedMediaError,
    canonicalize_llm_batch,
    caption_llm_postprocess,
    caption_llm_preprocess,
    decompose_row_addresses,
    fetch_media_batch,
    installed_checkpoint_config,
    result_schema,
    run_inference_phase,
    staged_result_files,
)
from cosmos_curator.next.recipes.video_caption.lance_state import CaptionAttempt, capture_attempt, ensure_caption_fields
from cosmos_curator.next.recipes.video_caption.workspace import CaptionWorkspace, phase_a_completion_covers
from cosmos_curator.next.recipes.video_transcode.contracts import CLIP_SCHEMA
from tests.cosmos_curator.next.conftest import CURATOR_IO_RESOURCE, CURATOR_IO_SLOTS
from tests.cosmos_curator.next.recipes.video_caption.helpers import append_clip_fragment, write_clip_fragment


class _FakeCaptionProcessor:
    """Small CPU stand-in that preserves Ray LLM's postprocessed row shape."""

    def __init__(self, expected_ids: set[str]) -> None:
        self._expected_ids = expected_ids

    def __call__(self, dataset: ray.data.Dataset) -> ray.data.Dataset:
        return dataset.map(_fake_inference, fn_kwargs={"expected_ids": self._expected_ids})


def _fake_inference(row: dict[str, Any], *, expected_ids: set[str]) -> dict[str, object]:
    clip_id = str(row["clip_id"])
    if clip_id not in expected_ids or row.get("download_error_type") is not None:
        msg = f"Checkpoint filter leaked unexpected clip {clip_id!r} into media/inference"
        raise AssertionError(msg)
    return {
        "fragment_id": row["fragment_id"],
        "row_offset": row["row_offset"],
        "clip_id": clip_id,
        "caption_text": f"caption for {clip_id}",
        "prompt_token_count": 4,
        "generated_token_count": 8,
        "__inference_error__": "",
    }


def _annotate_batch_size(batch: pa.Table) -> pa.Table:
    """Expose the batch size seen by a test stage on every output row."""
    return batch.append_column("observed_batch_size", pa.array([batch.num_rows] * batch.num_rows, type=pa.int64()))


def _workspace(tmp_path: Path) -> CaptionWorkspace:
    root = tmp_path / "workspace"
    return CaptionWorkspace(
        root_uri=str(root),
        manifest_uri=str(root / "workspace.json"),
        results_uri=str(root / "results"),
        checkpoints_uri=str(root / "checkpoints"),
        filesystem=pafs.LocalFileSystem(),
    )


def _s3_client_error(code: str, status: int) -> ClientError:
    return ClientError(
        {
            "Error": {"Code": code, "Message": "request failed"},
            "ResponseMetadata": {"HTTPStatusCode": status},
        },
        "GetObject",
    )


def test_row_addresses_decompose_into_fragment_and_local_offset() -> None:
    """Lance row addresses preserve physical fragment and local row identity."""
    table = pa.table(
        {
            "_rowaddr": pa.array([(7 << 32) | 19], type=pa.uint64()),
            "clip_id": ["clip"],
            "clip_uri": ["relative/clip.mp4"],
            "clip_size_bytes": pa.array([4], type=pa.int64()),
        }
    )

    mapped = decompose_row_addresses(table)

    assert mapped.schema == MAPPED_INPUT_SCHEMA
    assert mapped.to_pylist() == [
        {
            "fragment_id": 7,
            "row_offset": 19,
            "clip_id": "clip",
            "clip_uri": "relative/clip.mp4",
            "clip_size_bytes": 4,
        }
    ]


def test_caption_lance_read_task_reopens_with_storage_options_without_pickling_fragments(
    clip_dataset_factory: Callable[..., tuple[str, lance.LanceDataset]],
    clip_row_factory: Callable[..., dict[str, object]],
    monkeypatch: pytest.MonkeyPatch,
    caption_bindings: CaptionRunBindings,
) -> None:
    """Ray tasks carry credentials and fragment IDs, never storage-blind fragment handles."""
    uri, dataset = clip_dataset_factory(rows=[clip_row_factory(0)])
    attempt = CaptionAttempt(version=int(dataset.version), pending_fragment_ids=(0,), complete_fragment_ids=())
    storage_options = {
        "aws_access_key_id": "id",
        "aws_secret_access_key": "secret",
        "aws_session_token": "token",
        "aws_endpoint": "https://s3.example.test",
    }
    datasource = inference._caption_lance_datasource(
        dataset,
        attempt,
        storage_options=storage_options,
        batch_size=17,
        source_columns=caption_bindings.input.source_columns,
        source_schema=caption_bindings.input.source_schema,
    )
    read_task = datasource.get_read_tasks(parallelism=1)[0]

    def reject_fragment_pickle(_fragment: LanceFragment) -> object:
        msg = "a live LanceFragment crossed the Ray serialization boundary"
        raise AssertionError(msg)

    monkeypatch.setattr(LanceFragment, "__reduce__", reject_fragment_pickle)
    calls: list[tuple[str, int, dict[str, str] | None]] = []
    fragment_token = object()
    table = pa.table(
        {
            "clip_id": ["clip-0"],
            "clip_uri": ["s3://bucket/clip-0.mp4"],
            "clip_size_bytes": pa.array([1], type=pa.int64()),
            "_rowaddr": pa.array([0], type=pa.uint64()),
        }
    )

    class FakeDataset:
        def get_fragment(self, fragment_id: int) -> object | None:
            assert fragment_id == 0
            return fragment_token

        def scanner(self, **kwargs: object) -> object:
            assert kwargs == {
                "columns": ["clip_id", "clip_uri", "clip_size_bytes"],
                "fragments": [fragment_token],
                "with_row_address": True,
                "batch_size": 17,
            }

            class FakeScanner:
                def to_reader(self) -> pa.RecordBatchReader:
                    return pa.RecordBatchReader.from_batches(table.schema, table.to_batches())

            return FakeScanner()

    def reopen_dataset(
        reopened_uri: str,
        *,
        version: int,
        storage_options: dict[str, str] | None,
    ) -> FakeDataset:
        calls.append((reopened_uri, version, storage_options))
        return FakeDataset()

    monkeypatch.setattr(inference.lance, "dataset", reopen_dataset)

    restored_task = ray_cloudpickle.loads(ray_cloudpickle.dumps(read_task))
    output = list(restored_task())

    assert calls == [(uri, attempt.version, storage_options)]
    assert len(output) == 1
    assert output[0].equals(table)


def test_media_fetch_uses_exact_bytes_and_cross_node_data_url(
    tmp_path: Path,
    caption_bindings: CaptionRunBindings,
) -> None:
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
        schema=MAPPED_INPUT_SCHEMA,
    )

    fetched = fetch_media_batch(
        source,
        input_bindings=caption_bindings.input,
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
        ("https://example.com/clip.mp4", 10, "Unsupported clip URI scheme"),
    ],
)
def test_item_media_failures_become_rows(
    tmp_path: Path,
    uri: str,
    size: int,
    message: str,
    caption_bindings: CaptionRunBindings,
) -> None:
    """Deterministic missing, malformed, and unsupported media become item errors."""
    if uri == "existing":
        path = tmp_path / "clip.mp4"
        path.write_bytes(b"short")
        uri = str(path)
    source = pa.Table.from_pylist(
        [{"fragment_id": 0, "row_offset": 0, "clip_id": "clip", "clip_uri": uri, "clip_size_bytes": size}],
        schema=MAPPED_INPUT_SCHEMA,
    )

    fetched = fetch_media_batch(
        source,
        input_bindings=caption_bindings.input,
        storage_profile="default",
        attempts=1,
    ).to_pylist()[0]

    assert fetched["video_data_url"] is None
    assert fetched["download_error_type"] == "ItemMediaError"
    assert message in fetched["download_error_message"]


def test_shared_media_failure_stops_the_batch(
    monkeypatch: pytest.MonkeyPatch,
    caption_bindings: CaptionRunBindings,
) -> None:
    """Repeated shared filesystem failures fail Phase A instead of becoming rows."""
    source = pa.Table.from_pylist(
        [{"fragment_id": 0, "row_offset": 0, "clip_id": "clip", "clip_uri": "/clip", "clip_size_bytes": 1}],
        schema=MAPPED_INPUT_SCHEMA,
    )
    attempts = 0

    def unavailable(uri: str, *, storage_profile: str) -> bytes:
        del uri, storage_profile
        nonlocal attempts
        attempts += 1
        msg = "shared mount unavailable"
        raise OSError(msg)

    monkeypatch.setattr(inference, "_read_clip", unavailable)
    monkeypatch.setattr(inference.time, "sleep", lambda _: None)

    with pytest.raises(SharedMediaError, match="after 3 attempt"):
        fetch_media_batch(
            source,
            input_bindings=caption_bindings.input,
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

    def flaky_read(uri: str, *, storage_profile: str) -> bytes:
        del uri, storage_profile
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

    def denied(uri: str, *, storage_profile: str) -> bytes:
        del uri, storage_profile
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


def test_download_error_preserves_identity_through_terminal_adapter(
    caption_spec: CaptionModelSpec,
    caption_bindings: CaptionRunBindings,
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
        bindings=caption_bindings,
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


def test_caption_input_repartition_fans_fragment_blocks_out_to_fetch_tasks() -> None:
    """The media boundary requests strict small blocks without executing a Ray job."""

    class _DatasetPlan:
        def __init__(self) -> None:
            self.call: tuple[int, bool] | None = None

        def repartition(self, *, target_num_rows_per_block: int, strict: bool) -> "_DatasetPlan":
            self.call = (target_num_rows_per_block, strict)
            return self

    source = _DatasetPlan()
    result = inference._repartition_caption_inputs_for_fetch(source, rows_per_block=128)  # type: ignore[arg-type]

    assert result is source
    assert source.call == (128, True)


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


def test_canonicalization_emits_success_truncated_and_error_rows(
    caption_spec: CaptionModelSpec,
    caption_bindings: CaptionRunBindings,
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

    terminal = canonicalize_llm_batch(raw, bindings=caption_bindings)
    rows = terminal.to_pylist()

    assert terminal.schema.equals(result_schema(caption_spec), check_metadata=True)
    assert [row[caption_spec.metadata_field_name]["status"] for row in rows] == ["success", "truncated", "error"]
    assert rows[2][caption_spec.caption_field_name] is None
    assert rows[2][caption_spec.metadata_field_name]["error_type"] == "ItemMediaError"


def test_canonicalization_uses_the_model_bound_output_limit(caption_bindings: CaptionRunBindings) -> None:
    """Terminal state follows the model binding's request limit, not a module default."""
    bindings = replace(caption_bindings, inference=CaptionInferenceBindings(sampling_params={"max_tokens": 5}))
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

    terminal = canonicalize_llm_batch(raw, bindings=bindings).to_pylist()[0]

    assert terminal[caption_bindings.result.metadata_field_name]["status"] == "truncated"


def test_checkpoint_config_is_always_cleared(tmp_path: Path) -> None:
    """The job checkpoint cannot leak into Phase B, even after an exception."""
    context = ray.data.DataContext.get_current()
    context.checkpoint_config = None
    checkpoint = CheckpointConfig(id_column="clip_id", checkpoint_path=str(tmp_path))

    def install_and_fail() -> None:
        with installed_checkpoint_config(checkpoint):
            assert context.checkpoint_config is checkpoint
            msg = "boom"
            raise RuntimeError(msg)

    with pytest.raises(RuntimeError, match="boom"):
        install_and_fail()

    assert context.checkpoint_config is None


@pytest.mark.integration
def test_checkpointed_phase_a_reuses_results_against_a_newer_input_superset(
    tmp_path: Path,
    clip_dataset_factory: Callable[..., tuple[str, lance.LanceDataset]],
    clip_row_factory: Callable[..., dict[str, object]],
    caption_spec: CaptionModelSpec,
    caption_digest: str,
    caption_bindings: CaptionRunBindings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Committed clip IDs skip media and inference after a post-attempt split append."""
    old_paths = [tmp_path / "clip-0.mp4", tmp_path / "clip-1.mp4"]
    rows = []
    for index, path in enumerate(old_paths):
        payload = f"video-{index}".encode()
        path.write_bytes(payload)
        row = clip_row_factory(index)
        row["clip_uri"] = str(path)
        row["clip_size_bytes"] = len(payload)
        rows.append(row)
    uri, _ = clip_dataset_factory(rows=rows, rows_per_fragment=1)
    dataset = ensure_caption_fields(uri, storage_options=None, bindings=caption_bindings, attempts=3)
    attempt = capture_attempt(dataset, bindings=caption_bindings)
    config = resolve_config_data(
        {
            "schema_version": 1,
            "kind": "video-caption",
            "input": {"media_root": str(tmp_path), "clips_lance_uri": uri},
            "model": {"variant": "qwen3_8_27b_fp8"},
            "output": {"staging_root_uri": str(tmp_path / "staging")},
            "execution": {
                "media_batch_size": 1,
                "parquet_rows_per_file": 2,
            },
        }
    )
    workspace = _workspace(tmp_path)

    ray.shutdown()
    # The shared cluster's own slot count, not a literal: ``curator_io_cluster``
    # restarts any cluster whose count differs, so a copy here would silently
    # trigger a mid-session restart the moment the fixture's constant moved.
    ray.init(num_cpus=4, resources={CURATOR_IO_RESOURCE: CURATOR_IO_SLOTS}, include_dashboard=False)
    try:
        monkeypatch.setattr(
            inference,
            "_build_llm_processor",
            lambda *_args, **_kwargs: _FakeCaptionProcessor({"clip-0", "clip-1"}),
        )
        run_inference_phase(
            dataset,
            attempt,
            config.execution,
            caption_bindings,
            caption_spec.runtime_model_dir,
            workspace,
            storage_options=None,
        )
        assert phase_a_completion_covers(workspace, attempt, caption_digest) is True
        assert len(staged_result_files(workspace)) == 1
        assert len(list(Path(workspace.checkpoints_uri).glob("*.parquet"))) == 1

        for path in old_paths:
            path.unlink()
        new_payload = b"new-video"
        new_path = tmp_path / "clip-99.mp4"
        new_path.write_bytes(new_payload)
        appended_row = clip_row_factory(99)
        appended_row["clip_uri"] = str(new_path)
        appended_row["clip_size_bytes"] = len(new_payload)
        candidate = write_clip_fragment(
            pa.Table.from_pylist([appended_row], schema=CLIP_SCHEMA),
            uri=uri,
            storage_profile="default",
        )
        assert candidate is not None
        append_clip_fragment(candidate, uri=uri, storage_profile="default", attempts=3)
        latest = lance.dataset(uri)
        superset_attempt = capture_attempt(latest, bindings=caption_bindings)
        assert phase_a_completion_covers(workspace, superset_attempt, caption_digest) is False
        monkeypatch.setattr(
            inference,
            "_build_llm_processor",
            lambda *_args, **_kwargs: _FakeCaptionProcessor({"clip-99"}),
        )

        run_inference_phase(
            latest,
            superset_attempt,
            config.execution,
            caption_bindings,
            caption_spec.runtime_model_dir,
            workspace,
            storage_options=None,
        )
        assert phase_a_completion_covers(workspace, superset_attempt, caption_digest) is True
    finally:
        ray.data.DataContext.get_current().checkpoint_config = None
        ray.shutdown()

    result_tables = [pq.read_table(path, filesystem=workspace.filesystem) for path in staged_result_files(workspace)]
    staged = pa.concat_tables(result_tables)
    assert sorted(staged["clip_id"].to_pylist()) == ["clip-0", "clip-1", "clip-99"]
