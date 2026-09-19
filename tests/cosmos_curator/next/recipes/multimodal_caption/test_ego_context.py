# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused synthetic contract tests for deterministic Ego caption context."""

import json
from functools import partial
from hashlib import sha256
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest
from foxglove_schemas_protobuf.PoseInFrame_pb2 import PoseInFrame
from google.protobuf import descriptor_pb2
from google.protobuf.descriptor import FileDescriptor

from cosmos_curator.core.sensors.data.egotrajectory_data import EgoTrajectory
from cosmos_curator.core.sensors.exceptions import AlignmentError, AlignmentFailureReason
from cosmos_curator.core.sensors.sensors.ego_trajectory_sensor import EgoTrajectorySensor
from cosmos_curator.next.recipes.multimodal_caption import ego_context, ego_context_adapter, pipeline
from cosmos_curator.next.recipes.multimodal_caption.contracts import (
    EGO_CONTEXT_MAX_NEAREST_POSE_DELTA_NS,
    build_terminal_result_row,
    caption_contract_digest,
    caption_inference_bindings,
    resolve_context_definition,
    resolve_ego_identity,
    resolve_model,
    result_field_names,
    result_fields,
    staged_result_schema,
)
from cosmos_curator.next.recipes.multimodal_caption.ego_context import (
    EgoContextBuilder,
    EgoTrajectoryContextPolicy,
    build_ego_trajectory_absolute_poses,
    build_ego_trajectory_summary,
)
from cosmos_curator.next.recipes.multimodal_caption.ego_context_adapter import append_ego_context_batch
from cosmos_curator.next.recipes.video_caption.bindings import (
    CaptionPublicationBindings,
    CaptionResultBindings,
    CaptionRunBindings,
)
from cosmos_curator.next.recipes.video_caption.inference import (
    ItemMediaError,
    SharedMediaError,
    canonicalize_llm_batch,
    caption_llm_postprocess,
    caption_llm_preprocess,
    fetch_media_batch,
)
from tests.cosmos_curator.core.sensors.test_utils import McapSample, write_protobuf_mcap

_SCHEMA_NAME = "foxglove.PoseInFrame"
_TOPIC = "/ego"
_FRAME = "world"
_MAPPING = {
    "fields": {
        "sensor_timestamp_ns": {"from": "timestamp", "type": "protobuf_timestamp"},
        "align_timestamp_ns": {"from": "$mcap.logtime", "type": "timestamp", "unit": "ns"},
        "position_m": {
            "group": ["pose.position.x", "pose.position.y", "pose.position.z"],
            "type": "float",
        },
        "orientation_xyzw": {
            "group": [
                "pose.orientation.x",
                "pose.orientation.y",
                "pose.orientation.z",
                "pose.orientation.w",
            ],
            "type": "float",
        },
        "pose_valid": {"default": True},
    }
}


def _descriptor_set() -> bytes:
    descriptor_set = descriptor_pb2.FileDescriptorSet()
    added: set[str] = set()

    def add(descriptor: FileDescriptor) -> None:
        if descriptor.name in added:
            return
        for dependency in descriptor.dependencies:
            add(dependency)
        descriptor.CopyToProto(descriptor_set.file.add())
        added.add(descriptor.name)

    add(PoseInFrame.DESCRIPTOR.file)
    return descriptor_set.SerializeToString()


def _payload(timestamp_ns: int, position_x: float) -> bytes:
    message = PoseInFrame(frame_id="ignored")
    message.timestamp.seconds = timestamp_ns // 1_000_000_000
    message.timestamp.nanos = timestamp_ns % 1_000_000_000
    message.pose.position.x = position_x
    message.pose.orientation.w = 1.0
    return bytes(message.SerializeToString())


def _mcap(tmp_path: Path, samples: list[tuple[int, int, float]]) -> bytes:
    path = tmp_path / f"trajectory-{len(list(tmp_path.iterdir()))}.mcap"
    write_protobuf_mcap(
        path,
        [
            McapSample(log_time_ns=log_time, data=_payload(timestamp, position))
            for log_time, timestamp, position in samples
        ],
        topic=_TOPIC,
        schema_name=_SCHEMA_NAME,
        schema_data=_descriptor_set(),
        library="synthetic multimodal caption test",
    )
    return path.read_bytes()


def test_context_bytes_pin_summary_grid_halo_and_digest(tmp_path: Path) -> None:
    """Production mapping with divergent clocks pins alignment, provenance, and payload bytes."""
    mapping = pipeline._load_ego_mapping()
    payload = _mcap(tmp_path, [(90, 900, 9.0), (140, 1_400, 14.0), (200, 2_000, 20.0), (260, 2_600, 26.0)])
    decoded = EgoTrajectorySensor(
        payload,
        topic=_TOPIC,
        schema_name=_SCHEMA_NAME,
        protobuf_mapping=mapping,
        frame=_FRAME,
    ).read_all()
    assert decoded.align_timestamps_ns.tolist() == [90, 140, 200, 260]
    assert decoded.sensor_timestamps_ns.tolist() == [900, 1_400, 2_000, 2_600]
    context_bytes, digest = build_ego_trajectory_absolute_poses(
        payload,
        episode_start_ns=100,
        exclusive_end_ns=250,
        topic=_TOPIC,
        schema_name=_SCHEMA_NAME,
        protobuf_mapping=mapping,
        frame=_FRAME,
        policy=EgoTrajectoryContextPolicy(representation_version=1, cadence_period_ns=50, max_delta_ns=60),
    )

    expected = (
        b'{"frame":"world","representation_version":1,"samples":['
        b'{"orientation_xyzw":[0.0,0.0,0.0,1.0],"position_m":[9.0,0.0,0.0],'
        b'"relative_time_ns":0,"requested_align_timestamp_ns":100},'
        b'{"orientation_xyzw":[0.0,0.0,0.0,1.0],"position_m":[14.0,0.0,0.0],'
        b'"relative_time_ns":50,"requested_align_timestamp_ns":150},'
        b'{"orientation_xyzw":[0.0,0.0,0.0,1.0],"position_m":[20.0,0.0,0.0],'
        b'"relative_time_ns":100,"requested_align_timestamp_ns":200}],'
        b'"summary":{"duration_ns":150,"net_displacement_m":6.0,"path_length_m":6.0,'
        b'"recorded_pose_count":2}}'
    )
    assert context_bytes == expected
    assert len(context_bytes) == 526
    assert (
        digest == sha256(expected).hexdigest() == ("fc77a63292dd2b6dadcd9197069c28107a8388aff0710d47001c25bdb0ed4385")
    )

    summary_bytes, summary_digest = build_ego_trajectory_summary(
        payload,
        episode_start_ns=100,
        exclusive_end_ns=250,
        topic=_TOPIC,
        schema_name=_SCHEMA_NAME,
        protobuf_mapping=mapping,
        frame=_FRAME,
        policy=EgoTrajectoryContextPolicy(representation_version=1, cadence_period_ns=50, max_delta_ns=60),
    )
    expected_summary = (
        b'{"frame":"world","representation_version":1,"summary":{"duration_ns":150,'
        b'"net_displacement_m":6.0,"path_length_m":6.0,"recorded_pose_count":2}}'
    )
    absolute_without_samples = json.loads(expected)
    del absolute_without_samples["samples"]
    assert summary_bytes == json.dumps(
        absolute_without_samples,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    assert summary_bytes == expected_summary
    assert len(summary_bytes) == 143
    assert summary_digest == "fd72ee8606ad1c536e45aab0db825ff1b8408488319a2b2de0ad2a1920ea1c19"


def test_context_rejects_invalid_pose_and_maximum_delta(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """V1 rejects incompatible validity and unsupported grid requests."""
    invalid = EgoTrajectory(
        align_timestamps_ns=np.array([90, 100], dtype=np.int64),
        sensor_timestamps_ns=np.array([90, 100], dtype=np.int64),
        positions_m=np.zeros((2, 3), dtype=np.float64),
        orientations_xyzw=np.array([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
        pose_valid=np.array([False, True], dtype=np.bool_),
        frame=_FRAME,
    )
    monkeypatch.setattr(ego_context.EgoTrajectorySensor, "read_all", lambda _self: invalid)
    with pytest.raises(ValueError, match="pose_valid=true"):
        build_ego_trajectory_absolute_poses(
            b"unused",
            episode_start_ns=100,
            exclusive_end_ns=200,
            topic=_TOPIC,
            schema_name=_SCHEMA_NAME,
            protobuf_mapping=_MAPPING,
            frame=_FRAME,
            policy=EgoTrajectoryContextPolicy(representation_version=1, cadence_period_ns=50, max_delta_ns=60),
        )

    monkeypatch.undo()
    payload = _mcap(tmp_path, [(100, 1_000, 1.0)])
    with pytest.raises(AlignmentError, match="max_delta_ns=10 exceeded"):
        build_ego_trajectory_absolute_poses(
            payload,
            episode_start_ns=100,
            exclusive_end_ns=200,
            topic=_TOPIC,
            schema_name=_SCHEMA_NAME,
            protobuf_mapping=_MAPPING,
            frame=_FRAME,
            policy=EgoTrajectoryContextPolicy(representation_version=1, cadence_period_ns=50, max_delta_ns=10),
        )
    with pytest.raises(AlignmentError, match="max_delta_ns=10 exceeded"):
        build_ego_trajectory_summary(
            payload,
            episode_start_ns=100,
            exclusive_end_ns=200,
            topic=_TOPIC,
            schema_name=_SCHEMA_NAME,
            protobuf_mapping=_MAPPING,
            frame=_FRAME,
            policy=EgoTrajectoryContextPolicy(representation_version=1, cadence_period_ns=50, max_delta_ns=10),
        )


@pytest.mark.parametrize(
    "build_context",
    [build_ego_trajectory_absolute_poses, build_ego_trajectory_summary],
)
def test_nearest_pose_delta_is_inclusive_at_twenty_milliseconds(
    tmp_path: Path,
    build_context: EgoContextBuilder,
) -> None:
    """Exactly 20 ms qualifies, while one nanosecond beyond it is an alignment item error."""
    passing_payload = _mcap(tmp_path, [(20_000_000, 1, 1.0)])
    build_context(
        passing_payload,
        episode_start_ns=0,
        exclusive_end_ns=20_000_001,
        topic=_TOPIC,
        schema_name=_SCHEMA_NAME,
        protobuf_mapping=_MAPPING,
        frame=_FRAME,
        policy=EgoTrajectoryContextPolicy(
            representation_version=1,
            cadence_period_ns=50_000_000,
            max_delta_ns=EGO_CONTEXT_MAX_NEAREST_POSE_DELTA_NS,
        ),
    )

    failing_payload = _mcap(tmp_path, [(20_000_001, 1, 1.0)])
    with pytest.raises(AlignmentError, match="max_delta_ns=20000000 exceeded"):
        build_context(
            failing_payload,
            episode_start_ns=0,
            exclusive_end_ns=20_000_002,
            topic=_TOPIC,
            schema_name=_SCHEMA_NAME,
            protobuf_mapping=_MAPPING,
            frame=_FRAME,
            policy=EgoTrajectoryContextPolicy(
                representation_version=1,
                cadence_period_ns=50_000_000,
                max_delta_ns=EGO_CONTEXT_MAX_NEAREST_POSE_DELTA_NS,
            ),
        )


def test_maximum_duration_has_twenty_samples_and_pinned_size(tmp_path: Path) -> None:
    """The fixed cadence alone bounds a ten-second episode to 20 samples."""
    samples = [(index * 500_000_000, index * 500_000_000 + 1, float(index)) for index in range(20)]
    context_bytes, _ = build_ego_trajectory_absolute_poses(
        _mcap(tmp_path, samples),
        episode_start_ns=0,
        exclusive_end_ns=10_000_000_000,
        topic=_TOPIC,
        schema_name=_SCHEMA_NAME,
        protobuf_mapping=_MAPPING,
        frame=_FRAME,
        policy=EgoTrajectoryContextPolicy(
            representation_version=1,
            cadence_period_ns=500_000_000,
            max_delta_ns=EGO_CONTEXT_MAX_NEAREST_POSE_DELTA_NS,
        ),
    )
    assert context_bytes.count(b"requested_align_timestamp_ns") == 20
    assert len(context_bytes) == 2_916


def _adapter_batch(
    *,
    start_ns: int = 0,
    exclusive_end_ns: int = 10,
    media_location: str = "/ego.mcap",
    media_size_bytes: int = 1,
    media_sha256: str = "a" * 64,
) -> pa.Table:
    return pa.Table.from_pylist(
        [
            {
                "episode_id": "episode",
                "start_ns": start_ns,
                "exclusive_end_ns": exclusive_end_ns,
                "ego_media_location": media_location,
                "ego_media_size_bytes": media_size_bytes,
                "ego_media_sha256": media_sha256,
                "video_data_url": "data:video/mp4;base64,AAAA",
                "download_error_type": None,
                "download_error_message": None,
            }
        ],
        schema=pa.schema(
            [
                pa.field("episode_id", pa.string(), nullable=False),
                pa.field("start_ns", pa.int64(), nullable=False),
                pa.field("exclusive_end_ns", pa.int64(), nullable=False),
                pa.field("ego_media_location", pa.large_string(), nullable=False),
                pa.field("ego_media_size_bytes", pa.int64(), nullable=False),
                pa.field("ego_media_sha256", pa.string(), nullable=False),
                pa.field("video_data_url", pa.large_string()),
                pa.field("download_error_type", pa.string()),
                pa.field("download_error_message", pa.large_string()),
            ]
        ),
    )


@pytest.mark.parametrize(
    "error",
    [
        ItemMediaError("missing"),
        AlignmentError(AlignmentFailureReason.EGO_TRAJECTORY_MAX_DELTA_EXCEEDED, "unsupported"),
        TypeError("decode type"),
    ],
)
def test_adapter_translates_deterministic_item_errors(
    error: Exception,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Expected item/decode/alignment failures reuse the inference-error columns."""
    build_context = build_ego_trajectory_absolute_poses
    if isinstance(error, ItemMediaError):
        monkeypatch.setattr(
            ego_context_adapter,
            "read_and_validate_media",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(error),
        )
    else:
        monkeypatch.setattr(ego_context_adapter, "read_and_validate_media", lambda *_args, **_kwargs: b"unused")

        def raise_error(*_args: object, **_kwargs: object) -> tuple[bytes, str]:
            raise error

        build_context = raise_error
    result = append_ego_context_batch(
        _adapter_batch(),
        storage_profile="default",
        attempts=1,
        schema_name=_SCHEMA_NAME,
        protobuf_mapping=_MAPPING,
        topic=_TOPIC,
        frame=_FRAME,
        policy=EgoTrajectoryContextPolicy(representation_version=1, cadence_period_ns=5, max_delta_ns=5),
        prompt_template="context={ego_trajectory_context}",
        build_context=build_context,
    ).to_pylist()[0]
    assert result["download_error_type"] == type(error).__name__
    assert result["rendered_prompt"] is None
    assert result["context_digest"] is None


def test_adapter_translates_duplicate_alignment_timestamps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real decoder's strict-order ValueError becomes one item terminal error."""
    payload = _mcap(tmp_path, [(100, 1_000, 1.0), (100, 2_000, 2.0)])
    batch = _adapter_batch()
    batch = batch.set_column(
        batch.schema.get_field_index("ego_media_size_bytes"),
        "ego_media_size_bytes",
        pa.array([len(payload)], type=pa.int64()),
    ).set_column(
        batch.schema.get_field_index("ego_media_sha256"),
        "ego_media_sha256",
        pa.array([sha256(payload).hexdigest()], type=pa.string()),
    )
    monkeypatch.setattr(ego_context_adapter, "read_and_validate_media", lambda *_args, **_kwargs: payload)

    result = append_ego_context_batch(
        batch,
        storage_profile="default",
        attempts=1,
        schema_name=_SCHEMA_NAME,
        protobuf_mapping=_MAPPING,
        topic=_TOPIC,
        frame=_FRAME,
        policy=EgoTrajectoryContextPolicy(representation_version=1, cadence_period_ns=5, max_delta_ns=5),
        prompt_template="context={ego_trajectory_context}",
        build_context=build_ego_trajectory_absolute_poses,
    ).to_pylist()[0]

    assert result["download_error_type"] == "ValueError"
    assert result["download_error_message"] == "Failed to decode Ego trajectory (ValueError)"
    assert result["rendered_prompt"] is None
    assert result["context_digest"] is None


def test_adapter_skips_prior_camera_failure_and_propagates_shared_storage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Camera item failures skip Ego reads, while shared storage still fails Phase A."""
    batch = _adapter_batch()
    failed = batch.set_column(
        batch.schema.get_field_index("download_error_type"),
        "download_error_type",
        pa.array(["ItemMediaError"]),
    )
    monkeypatch.setattr(
        ego_context_adapter,
        "read_and_validate_media",
        lambda *_args, **_kwargs: pytest.fail("a failed camera must not read Ego media"),
    )
    skipped = append_ego_context_batch(
        failed,
        storage_profile="default",
        attempts=1,
        schema_name=_SCHEMA_NAME,
        protobuf_mapping=_MAPPING,
        topic=_TOPIC,
        frame=_FRAME,
        policy=EgoTrajectoryContextPolicy(representation_version=1, cadence_period_ns=5, max_delta_ns=5),
        prompt_template="context={ego_trajectory_context}",
        build_context=build_ego_trajectory_absolute_poses,
    )
    assert skipped.select(["rendered_prompt", "context_digest"]).to_pylist() == [
        {"rendered_prompt": None, "context_digest": None}
    ]

    monkeypatch.setattr(
        ego_context_adapter,
        "read_and_validate_media",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(SharedMediaError("credentials")),
    )
    with pytest.raises(SharedMediaError, match="credentials"):
        append_ego_context_batch(
            batch,
            storage_profile="default",
            attempts=1,
            schema_name=_SCHEMA_NAME,
            protobuf_mapping=_MAPPING,
            topic=_TOPIC,
            frame=_FRAME,
            policy=EgoTrajectoryContextPolicy(representation_version=1, cadence_period_ns=5, max_delta_ns=5),
            prompt_template="context={ego_trajectory_context}",
            build_context=build_ego_trajectory_absolute_poses,
        )


def test_adapter_success_preserves_camera_buffer_and_exact_prompt(tmp_path: Path) -> None:
    """The real adapter path reads, validates, decodes, renders, and preserves camera Arrow storage."""
    payload = _mcap(tmp_path, [(90, 900, 9.0), (140, 1_400, 14.0), (200, 2_000, 20.0), (260, 2_600, 26.0)])
    media_path = tmp_path / "adapter-success.mcap"
    media_path.write_bytes(payload)
    policy = EgoTrajectoryContextPolicy(representation_version=1, cadence_period_ns=50, max_delta_ns=60)
    prompt_template = "before {unrelated}\n{ego_trajectory_context}\nafter"
    batch = _adapter_batch(
        start_ns=100,
        exclusive_end_ns=250,
        media_location=str(media_path),
        media_size_bytes=len(payload),
        media_sha256=sha256(payload).hexdigest(),
    )
    expected_context, expected_digest = build_ego_trajectory_absolute_poses(
        payload,
        episode_start_ns=100,
        exclusive_end_ns=250,
        topic=_TOPIC,
        schema_name=_SCHEMA_NAME,
        protobuf_mapping=_MAPPING,
        frame=_FRAME,
        policy=policy,
    )
    original_buffer = batch.column("video_data_url").chunk(0).buffers()[2]
    assert original_buffer is not None

    result = append_ego_context_batch(
        batch,
        storage_profile="default",
        attempts=1,
        schema_name=_SCHEMA_NAME,
        protobuf_mapping=_MAPPING,
        topic=_TOPIC,
        frame=_FRAME,
        policy=policy,
        prompt_template=prompt_template,
        build_context=build_ego_trajectory_absolute_poses,
    )
    row = result.to_pylist()[0]
    result_buffer = result.column("video_data_url").chunk(0).buffers()[2]

    assert row["download_error_type"] is None
    assert row["download_error_message"] is None
    assert row["rendered_prompt"] == prompt_template.replace(
        "{ego_trajectory_context}", expected_context.decode("utf-8"), 1
    )
    assert row["context_digest"] == expected_digest
    assert row["video_data_url"] == "data:video/mp4;base64,AAAA"
    assert result_buffer is not None
    assert (result_buffer.address, result_buffer.size) == (original_buffer.address, original_buffer.size)

    expected_summary, expected_summary_digest = build_ego_trajectory_summary(
        payload,
        episode_start_ns=100,
        exclusive_end_ns=250,
        topic=_TOPIC,
        schema_name=_SCHEMA_NAME,
        protobuf_mapping=_MAPPING,
        frame=_FRAME,
        policy=policy,
    )
    summary_result = append_ego_context_batch(
        batch,
        storage_profile="default",
        attempts=1,
        schema_name=_SCHEMA_NAME,
        protobuf_mapping=_MAPPING,
        topic=_TOPIC,
        frame=_FRAME,
        policy=policy,
        prompt_template=prompt_template,
        build_context=build_ego_trajectory_summary,
    ).to_pylist()[0]
    assert summary_result["rendered_prompt"] == prompt_template.replace(
        "{ego_trajectory_context}", expected_summary.decode("utf-8"), 1
    )
    assert summary_result["context_digest"] == expected_summary_digest

    empty = append_ego_context_batch(
        batch.slice(0, 0),
        storage_profile="default",
        attempts=1,
        schema_name=_SCHEMA_NAME,
        protobuf_mapping=_MAPPING,
        topic=_TOPIC,
        frame=_FRAME,
        policy=policy,
        prompt_template=prompt_template,
        build_context=build_ego_trajectory_absolute_poses,
    )
    assert empty.schema.field("download_error_type").type == pa.string()
    assert empty.schema.field("download_error_message").type == pa.large_string()
    assert empty.schema.field("rendered_prompt").type == pa.large_string()
    assert empty.schema.field("context_digest").type == pa.string()


def test_ego_context_composes_through_shared_engine_and_canonicalization(tmp_path: Path) -> None:
    """Real Ego bindings carry only source evidence and digest across the shared engine."""
    ego_payload = _mcap(tmp_path, [(90, 900, 9.0), (140, 1_400, 14.0), (200, 2_000, 20.0)])
    ego_path = tmp_path / "composed-ego.mcap"
    ego_path.write_bytes(ego_payload)
    video_payload = b"synthetic mp4 bytes"
    video_path = tmp_path / "composed-video.mp4"
    video_path.write_bytes(video_payload)

    definition = resolve_context_definition("ego_trajectory_absolute_poses")
    ego = resolve_ego_identity(definition, sensor_key="ego", topic=_TOPIC, frame=_FRAME)
    assert ego is not None
    prompt = "context={ego_trajectory_context}; keep {unrelated}"
    model = resolve_model("cosmos3_nano")
    digest = caption_contract_digest(model, "front_wide", definition, ego, prompt, "composed")
    caption_name, metadata_name = result_field_names(definition.variant, "composed", model.key)
    input_bindings, context_stage, carried = pipeline._resolve_input_runtime(
        definition,
        ego,
        camera_key="front_wide",
        prompt_template=prompt,
        mapping=pipeline._load_ego_mapping(),
    )
    assert context_stage is not None
    assert carried == ("context_digest",)
    result_bindings = CaptionResultBindings(
        caption_field_name=caption_name,
        metadata_field_name=metadata_name,
        fields=result_fields(definition.variant, "composed", model.key, ego=ego, digest=digest),
        staged_schema=staged_result_schema(definition.variant, "composed", model.key, ego=ego, digest=digest),
        digest=digest,
        build_terminal_row=partial(
            build_terminal_result_row,
            model=model,
            definition=definition,
            ego=ego,
            prompt_template=prompt,
            result_key="composed",
            digest=digest,
        ),
        validate_terminal_value=lambda *_args, **_kwargs: None,
        validate_staged_row=lambda *_args, **_kwargs: None,
        carried_result_columns=carried,
    )
    bindings = CaptionRunBindings(
        input=input_bindings,
        result=result_bindings,
        publication=CaptionPublicationBindings(
            validate_base_schema=lambda *_args, **_kwargs: None,
            protected_field_names=(),
            base_field_owner="multimodal-split",
            transaction_properties={},
        ),
        prompt_template=prompt,
        inference=caption_inference_bindings(model.key),
        context_stage=context_stage,
    )
    mapped = pa.Table.from_pylist(
        [
            {
                "fragment_id": 0,
                "row_offset": 0,
                "episode_id": "episode",
                "materialization_digest": "materialization",
                "timestamp_source": "sensor",
                "start_ns": 100,
                "exclusive_end_ns": 250,
                "camera_key": "front_wide",
                "media_location": str(video_path),
                "media_size_bytes": len(video_payload),
                "media_sha256": sha256(video_payload).hexdigest(),
                "ego_sensor_key": "ego",
                "ego_sensor_type": "egotrajectory",
                "ego_media_location": str(ego_path),
                "ego_media_size_bytes": len(ego_payload),
                "ego_media_sha256": sha256(ego_payload).hexdigest(),
            }
        ],
        schema=input_bindings.mapped_schema,
    )
    fetched = fetch_media_batch(mapped, input_bindings=input_bindings, storage_profile="default", attempts=1)
    contextual = context_stage(fetched, storage_profile="default", attempts=1)
    engine_row = contextual.to_pylist()[0]
    context_digest = engine_row["context_digest"]
    request = caption_llm_preprocess(
        engine_row,
        assistant_prefill=bindings.inference.assistant_prefill,
        prompt_template=bindings.prompt_template,
        request_mm_processor_kwargs=bindings.inference.request_mm_processor_kwargs,
        sampling_params={"max_tokens": 10},
    )
    rendered = request["messages"][0]["content"][1]["text"]
    assert rendered.startswith("context={")
    assert rendered.endswith("; keep {unrelated}")
    assert request["messages"][1] == {"role": "assistant", "content": "<think></think>"}

    postprocessed = caption_llm_postprocess(
        engine_row | {"generated_text": "caption", "num_input_tokens": 4, "num_generated_tokens": 8},
        staged_binding_columns=input_bindings.staged_binding_columns,
        carried_result_columns=carried,
    )
    assert postprocessed["context_digest"] == context_digest
    assert "rendered_prompt" not in postprocessed
    assert "video_data_url" not in postprocessed
    terminal = canonicalize_llm_batch(pa.Table.from_pylist([postprocessed]), bindings=bindings)
    assert terminal.schema.equals(result_bindings.staged_schema, check_metadata=True)
    terminal_row = terminal.to_pylist()[0]
    assert terminal_row["timestamp_source"] == "sensor"
    assert terminal_row[caption_name] == "caption"
    assert terminal_row[metadata_name]["context_digest"] == context_digest


def test_adapter_translates_malformed_mcap_without_parser_payload(tmp_path: Path) -> None:
    """Malformed containers become safe adapter item errors rather than failing Phase A."""
    payload = b"not an mcap"
    media_path = tmp_path / "malformed.mcap"
    media_path.write_bytes(payload)
    batch = _adapter_batch(
        media_location=str(media_path),
        media_size_bytes=len(payload),
        media_sha256=sha256(payload).hexdigest(),
    )

    row = append_ego_context_batch(
        batch,
        storage_profile="default",
        attempts=1,
        schema_name=_SCHEMA_NAME,
        protobuf_mapping=_MAPPING,
        topic=_TOPIC,
        frame=_FRAME,
        policy=EgoTrajectoryContextPolicy(representation_version=1, cadence_period_ns=5, max_delta_ns=5),
        prompt_template="context={ego_trajectory_context}",
        build_context=build_ego_trajectory_absolute_poses,
    ).to_pylist()[0]

    assert row["download_error_type"] == "ValueError"
    assert row["download_error_message"] == "Failed to decode Ego MCAP container (InvalidMagic)"
    assert "not an mcap" not in row["download_error_message"]
    assert row["rendered_prompt"] is None
    assert row["context_digest"] is None


def test_adapter_translates_empty_in_span_context(tmp_path: Path) -> None:
    """A valid MCAP with no pose in the episode span becomes adapter item-error output."""
    payload = _mcap(tmp_path, [(100, 1_000, 1.0), (200, 2_000, 2.0)])
    media_path = tmp_path / "empty-in-span.mcap"
    media_path.write_bytes(payload)
    batch = _adapter_batch(
        start_ns=0,
        exclusive_end_ns=10,
        media_location=str(media_path),
        media_size_bytes=len(payload),
        media_sha256=sha256(payload).hexdigest(),
    )

    row = append_ego_context_batch(
        batch,
        storage_profile="default",
        attempts=1,
        schema_name=_SCHEMA_NAME,
        protobuf_mapping=_MAPPING,
        topic=_TOPIC,
        frame=_FRAME,
        policy=EgoTrajectoryContextPolicy(representation_version=1, cadence_period_ns=5, max_delta_ns=5),
        prompt_template="context={ego_trajectory_context}",
        build_context=build_ego_trajectory_absolute_poses,
    ).to_pylist()[0]

    assert row["download_error_type"] == "ValueError"
    assert row["download_error_message"] == "ego-trajectory context requires at least one recorded in-span pose"
    assert row["rendered_prompt"] is None
    assert row["context_digest"] is None


def test_adapter_labels_ego_integrity_failure(tmp_path: Path) -> None:
    """Ego descriptor failures identify MCAP media without changing camera diagnostics."""
    media_path = tmp_path / "integrity.mcap"
    media_path.write_bytes(b"x")
    batch = _adapter_batch(media_location=str(media_path), media_size_bytes=2)

    row = append_ego_context_batch(
        batch,
        storage_profile="default",
        attempts=1,
        schema_name=_SCHEMA_NAME,
        protobuf_mapping=_MAPPING,
        topic=_TOPIC,
        frame=_FRAME,
        policy=EgoTrajectoryContextPolicy(representation_version=1, cadence_period_ns=5, max_delta_ns=5),
        prompt_template="context={ego_trajectory_context}",
        build_context=build_ego_trajectory_absolute_poses,
    ).to_pylist()[0]

    assert row["download_error_type"] == "ItemMediaError"
    assert row["download_error_message"] == "Expected 2 Ego MCAP bytes but downloaded 1"
    assert row["rendered_prompt"] is None
    assert row["context_digest"] is None


@pytest.mark.parametrize(
    ("media_location", "expected_message"),
    [
        ("/does/not/exist.mcap", "Ego MCAP object is unavailable"),
        ("https://example.com/ego.mcap", "Unsupported Ego MCAP URI scheme: https"),
    ],
)
def test_adapter_labels_ego_read_failures(media_location: str, expected_message: str) -> None:
    """Missing and unsupported Ego locations retain the MCAP media identity."""
    row = append_ego_context_batch(
        _adapter_batch(media_location=media_location),
        storage_profile="default",
        attempts=1,
        schema_name=_SCHEMA_NAME,
        protobuf_mapping=_MAPPING,
        topic=_TOPIC,
        frame=_FRAME,
        policy=EgoTrajectoryContextPolicy(representation_version=1, cadence_period_ns=5, max_delta_ns=5),
        prompt_template="context={ego_trajectory_context}",
        build_context=build_ego_trajectory_absolute_poses,
    ).to_pylist()[0]

    assert row["download_error_type"] == "ItemMediaError"
    assert row["download_error_message"] == expected_message
    assert row["rendered_prompt"] is None
    assert row["context_digest"] is None


def test_adapter_sanitizes_alignment_error_source_timestamps(monkeypatch: pytest.MonkeyPatch) -> None:
    """Durable alignment errors keep policy diagnostics but omit private clocks."""
    monkeypatch.setattr(ego_context_adapter, "read_and_validate_media", lambda *_args, **_kwargs: b"unused")

    def fail_alignment(*_args: object, **_kwargs: object) -> tuple[bytes, str]:
        raise AlignmentError(
            AlignmentFailureReason.EGO_TRAJECTORY_MAX_DELTA_EXCEEDED,
            "max delta was 11 ns for requested=123456789, source=987654321",
            align_timestamps_ns=np.array([123456789], dtype=np.int64),
            sensor_timestamps_ns=np.array([777777777], dtype=np.int64),
            known_source_timestamps_ns=np.array([987654321], dtype=np.int64),
            delta_ns=11,
            max_delta_ns=10,
        )

    row = append_ego_context_batch(
        _adapter_batch(),
        storage_profile="default",
        attempts=1,
        schema_name=_SCHEMA_NAME,
        protobuf_mapping=_MAPPING,
        topic=_TOPIC,
        frame=_FRAME,
        policy=EgoTrajectoryContextPolicy(representation_version=1, cadence_period_ns=5, max_delta_ns=5),
        prompt_template="context={ego_trajectory_context}",
        build_context=fail_alignment,
    ).to_pylist()[0]

    assert row["download_error_type"] == "AlignmentError"
    assert row["download_error_message"] == ("ego_trajectory_max_delta_exceeded: delta_ns=11, max_delta_ns=10")
    assert all(value not in row["download_error_message"] for value in ("123456789", "777777777", "987654321"))


def test_adapter_sanitizes_decoder_source_values(monkeypatch: pytest.MonkeyPatch) -> None:
    """Durable decoder errors retain a stable category but omit source values."""
    private_source_value = "123456789"

    class FailingEgoTrajectorySensor:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def read_all(self) -> EgoTrajectory:
            msg = f"invalid embedded timestamp {private_source_value}"
            raise ValueError(msg)

    monkeypatch.setattr(ego_context, "EgoTrajectorySensor", FailingEgoTrajectorySensor)
    monkeypatch.setattr(ego_context_adapter, "read_and_validate_media", lambda *_args, **_kwargs: b"unused")

    row = append_ego_context_batch(
        _adapter_batch(),
        storage_profile="default",
        attempts=1,
        schema_name=_SCHEMA_NAME,
        protobuf_mapping=_MAPPING,
        topic=_TOPIC,
        frame=_FRAME,
        policy=EgoTrajectoryContextPolicy(representation_version=1, cadence_period_ns=5, max_delta_ns=5),
        prompt_template="context={ego_trajectory_context}",
        build_context=build_ego_trajectory_absolute_poses,
    ).to_pylist()[0]

    assert row["download_error_type"] == "ValueError"
    assert row["download_error_message"] == "Failed to decode Ego trajectory (ValueError)"
    assert private_source_value not in row["download_error_message"]
