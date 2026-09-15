# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for canonical episode validation and selected-camera projection."""

from collections.abc import Callable
from functools import partial
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import lance
import pyarrow as pa
import pytest

from cosmos_curator.next.recipes.multimodal_caption import episode_input
from cosmos_curator.next.recipes.multimodal_caption.contracts import (
    build_terminal_result_row,
    caption_contract_digest,
    resolve_model,
    result_field_names,
    result_fields,
)
from cosmos_curator.next.recipes.multimodal_caption.episode_input import (
    EPISODE_MAPPED_INPUT_SCHEMA,
    EPISODE_SOURCE_COLUMNS,
    EPISODE_SOURCE_SCHEMA,
    EPISODE_STAGED_BINDING_COLUMNS,
    project_episode_camera,
    validate_complete_fragment,
    validate_episode_input,
)
from cosmos_curator.next.recipes.multimodal_split.identities import MEDIA_CONTRACT_VERSION
from cosmos_curator.next.recipes.multimodal_split.records import RECORD_SCHEMA_VERSION
from cosmos_curator.next.recipes.video_caption.bindings import CaptionInputBindings
from cosmos_curator.next.recipes.video_caption.inference import fetch_media_batch


def test_episode_input_accepts_canonical_unique_rows_and_resolves_exact_camera(
    episode_dataset_factory: Callable[..., tuple[str, lance.LanceDataset]],
) -> None:
    """Canonical unique episodes resolve one explicitly named camera."""
    uri, dataset = episode_dataset_factory()

    assert validate_episode_input(dataset, uri=uri, camera_key="front_wide") is None
    with pytest.raises(ValueError, match="0 camera descriptors"):
        validate_episode_input(dataset, uri=uri, camera_key="missing")


def test_episode_input_rejects_duplicate_ids_and_ambiguous_camera(
    episode_dataset_factory: Callable[..., tuple[str, lance.LanceDataset]],
    episode_factory: Callable[..., dict[str, object]],
) -> None:
    """Duplicate episode keys and ambiguous camera keys fail preflight."""
    duplicate = [episode_factory(0, episode_id="same"), episode_factory(1, episode_id="same")]
    uri, dataset = episode_dataset_factory(rows=duplicate)
    with pytest.raises(ValueError, match="not globally unique"):
        validate_episode_input(dataset, uri=uri, camera_key="front_wide")

    ambiguous = episode_factory(2)
    cameras = list(ambiguous["camera_media"])  # type: ignore[arg-type]
    cameras.append(dict(cameras[-1]))
    ambiguous["camera_media"] = cameras
    uri, dataset = episode_dataset_factory(rows=[ambiguous])
    with pytest.raises(ValueError, match="2 camera descriptors"):
        validate_episode_input(dataset, uri=uri, camera_key="front_wide")


def test_episode_input_reports_null_id_before_uniqueness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A null publication key is reported directly rather than as a duplicate."""
    batch = pa.RecordBatch.from_pylist([{"rows": 2, "episode_ids": 1, "null_episode_ids": 1}])
    query = SimpleNamespace(build=lambda: SimpleNamespace(to_batch_records=lambda: [batch]))
    dataset = SimpleNamespace(sql=lambda _query: query)
    monkeypatch.setattr(episode_input, "validate_episode_table", lambda *_args, **_kwargs: None)

    with pytest.raises(ValueError, match="must all be non-null"):
        validate_episode_input(dataset, uri="episodes.lance", camera_key="front_wide")  # type: ignore[arg-type]


def test_episode_input_rejects_incompatible_contract_versions(
    episode_dataset_factory: Callable[..., tuple[str, lance.LanceDataset]],
    episode_factory: Callable[..., dict[str, object]],
) -> None:
    """Canonical Arrow shape cannot disguise an incompatible episode contract."""
    row = episode_factory(0)
    row["media_contract_version"] = 999
    uri, dataset = episode_dataset_factory(rows=[row])

    expected_versions = (
        f"record_schema_version={RECORD_SCHEMA_VERSION} and media_contract_version={MEDIA_CONTRACT_VERSION}"
    )
    with pytest.raises(ValueError, match=expected_versions):
        validate_episode_input(dataset, uri=uri, camera_key="front_wide")


def test_projector_flattens_only_the_selected_camera(
    episode_factory: Callable[..., dict[str, object]],
) -> None:
    """Projection emits only the selected camera and physical identity."""
    episode = episode_factory(0)
    source = pa.Table.from_pylist([episode], schema=EPISODE_SOURCE_SCHEMA).append_column(
        "_rowaddr", pa.array([(7 << 32) | 3], type=pa.uint64())
    )

    projected = project_episode_camera(source, camera_key="front_wide")
    row = projected.to_pylist()[0]

    assert projected.schema == EPISODE_MAPPED_INPUT_SCHEMA
    assert (row["fragment_id"], row["row_offset"], row["episode_id"]) == (7, 3, "episode-0")
    assert row["camera_key"] == "front_wide"
    assert row["media_location"] == "/output/front_wide.mp4"
    assert row["media_sha256"] == "2" * 64


def test_media_fetch_checks_selected_size_and_sha256(tmp_path: Path) -> None:
    """Episode media fetch verifies both recorded byte count and hash."""
    payload = b"episode-mp4"
    media = tmp_path / "front-wide.mp4"
    media.write_bytes(payload)
    input_bindings = CaptionInputBindings(
        source_columns=EPISODE_SOURCE_COLUMNS,
        source_schema=EPISODE_SOURCE_SCHEMA,
        mapped_schema=EPISODE_MAPPED_INPUT_SCHEMA,
        project_batch=partial(project_episode_camera, camera_key="front_wide"),
        row_id_column="episode_id",
        media_uri_column="media_location",
        media_size_column="media_size_bytes",
        media_sha256_column="media_sha256",
        staged_binding_columns=EPISODE_STAGED_BINDING_COLUMNS,
    )
    row = {
        "fragment_id": 0,
        "row_offset": 0,
        "episode_id": "episode",
        "media_location": str(media),
        "media_size_bytes": len(payload),
        "media_sha256": sha256(payload).hexdigest(),
        "materialization_digest": "materialization",
        "camera_key": "front_wide",
    }
    source = pa.Table.from_pylist([row], schema=EPISODE_MAPPED_INPUT_SCHEMA)

    fetched = fetch_media_batch(
        source,
        input_bindings=input_bindings,
        storage_profile="default",
        attempts=1,
    ).to_pylist()[0]
    assert fetched["download_error_type"] is None

    row["media_sha256"] = "0" * 64
    mismatch = fetch_media_batch(
        pa.Table.from_pylist([row], schema=EPISODE_MAPPED_INPUT_SCHEMA),
        input_bindings=input_bindings,
        storage_profile="default",
        attempts=1,
    ).to_pylist()[0]
    assert mismatch["download_error_type"] == "ItemMediaError"
    assert "SHA-256" in mismatch["download_error_message"]


@pytest.mark.parametrize("binding_name", ["episode_id", "materialization_digest", "camera_key", "media_sha256"])
def test_complete_fragment_rejects_incompatible_episode_binding(
    binding_name: str,
    episode_factory: Callable[..., dict[str, object]],
) -> None:
    """Complete episode fragments remain bound to their selected camera source."""
    episode = episode_factory(0)
    model = resolve_model("qwen3_8_27b_fp8")
    digest = caption_contract_digest(model, "front_wide")
    caption_name, metadata_name = result_field_names(model.key)
    selected = next(camera for camera in episode["camera_media"] if camera["camera_key"] == "front_wide")  # type: ignore[union-attr]
    staged = build_terminal_result_row(
        {
            "fragment_id": 0,
            "row_offset": 0,
            "episode_id": episode["episode_id"],
            "materialization_digest": episode["materialization_digest"],
            "camera_key": selected["camera_key"],
            "media_location": selected["media_location"],
            "media_size_bytes": selected["media_size_bytes"],
            "media_sha256": selected["media_sha256"],
        },
        model=model,
        digest=digest,
        caption="caption",
        status="success",
        prompt_token_count=4,
        generated_token_count=8,
        error_type=None,
        error_message=None,
    )
    callback_camera_key = "front_wide"
    if binding_name == "episode_id":
        episode["episode_id"] = "different-episode"
    elif binding_name == "materialization_digest":
        episode["materialization_digest"] = "different-materialization"
    elif binding_name == "camera_key":
        callback_camera_key = "front_left"
    else:
        selected["media_sha256"] = "f" * 64
    metadata_field = result_fields(model.key).field(metadata_name)
    table = pa.Table.from_pylist(
        [{**episode, metadata_name: staged[metadata_name]}],
        schema=pa.schema([*EPISODE_SOURCE_SCHEMA, metadata_field]),
    )

    def scan(*, columns: list[str]) -> SimpleNamespace:
        assert columns == [*EPISODE_SOURCE_COLUMNS, metadata_name]
        return SimpleNamespace(to_batches=table.to_batches)

    fragment = SimpleNamespace(scanner=scan)
    with pytest.raises(ValueError, match=f"metadata {binding_name} does not match"):
        validate_complete_fragment(  # type: ignore[arg-type]
            fragment,
            camera_key=callback_camera_key,
            metadata_field_name=metadata_name,
        )
    assert staged[caption_name] == "caption"
