# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check NVDEC sampling, pinned CUDA processors, and visual encoders on a GPU."""

import gc
import io
from pathlib import Path

import av
import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pytest

from cosmos_curator.core.utils import environment
from cosmos_curator.next.recipes.video_embed.bindings import resolve_bindings
from cosmos_curator.next.recipes.video_embed.config import ResolvedVideoEmbedConfig, resolve_config_data
from cosmos_curator.next.recipes.video_embed.contracts import ModelSpec, resolve_model_spec, validate_model_directory
from cosmos_curator.next.recipes.video_embed.processing import VideoDecoder, VideoEmbedder, download_clip, sample_video
from tests.cosmos_curator.next.recipes.video_embed.conftest import write_clips
from tests.cosmos_curator.next.recipes.video_embed.test_processing import video_bytes

pytestmark = pytest.mark.env("legacy-transformers")


@pytest.mark.parametrize("frame_count", [1, 3, 8, 15])
def test_nvdec_sampling(frame_count: int) -> None:
    """Include repeated short-clip frames and delayed B frames in display order."""
    import torch  # noqa: PLC0415

    if not torch.cuda.is_available():
        pytest.skip("NVDEC requires a GPU")
    payload = video_bytes(frame_count)
    actual = sample_video(payload)
    assert actual.is_cuda
    assert actual.dtype == torch.uint8
    with av.open(io.BytesIO(payload)) as container:
        decoded = np.stack([frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)])
    assert len(decoded) == frame_count
    reference = decoded[np.linspace(0, frame_count - 1, 8, dtype=int)].transpose(0, 3, 1, 2)
    # NVDEC and FFmpeg RGB conversion round differently. Adjacent fixture frames
    # differ by about ten levels, so three levels still detects wrong frame selection.
    np.testing.assert_allclose(actual.cpu().numpy(), reference, rtol=0, atol=3)
    # The encoded frames increase in brightness. This also checks display order
    # independently of both decoder APIs and their B-frame reorder buffers.
    means = actual.float().mean(dim=(1, 2, 3)).cpu().numpy()
    assert np.all(np.diff(means) >= 0)
    assert means[-1] - means[0] >= (frame_count - 1) * 8


def test_nvdec_reuse_across_clips() -> None:
    """Reused sessions drain B frames and retain samples across independent streams."""
    import torch  # noqa: PLC0415

    if not torch.cuda.is_available():
        pytest.skip("NVDEC requires a GPU")
    decoder = VideoDecoder()
    retained = []
    for frame_count, width, height, codec in (
        (15, 64, 48, "mpeg4"),
        (3, 64, 48, "mpeg4"),
        (8, 96, 64, "mpeg4"),
        (8, 64, 48, "mpeg2video"),
        (1, 64, 48, "mpeg4"),
    ):
        payload = video_bytes(frame_count, width=width, height=height, codec=codec)
        expected = sample_video(payload)
        actual = sample_video(payload, decoder=decoder)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        retained.append((actual, expected))
    for actual, expected in retained:
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_nvdec_reuse_after_interrupted_decode(tmp_path: Path) -> None:
    """An abandoned stream must not leave delayed frames in a cached session."""
    import torch  # noqa: PLC0415

    if not torch.cuda.is_available():
        pytest.skip("NVDEC requires a GPU")
    decoder = VideoDecoder()
    path = tmp_path / "interrupted.mp4"
    path.write_bytes(video_bytes(15))
    frames = decoder.decode(str(path), 0)
    next(frames)
    frames.close()
    payload = video_bytes(3)
    torch.testing.assert_close(sample_video(payload, decoder=decoder), sample_video(payload), rtol=0, atol=0)


def test_nvdec_reuse_without_declared_frame_count() -> None:
    """Counting and sampling passes can reuse a drained session for an unindexed clip."""
    import torch  # noqa: PLC0415

    if not torch.cuda.is_available():
        pytest.skip("NVDEC requires a GPU")
    payload = video_bytes(15, codec="mpeg2video", container_format="mpegts")
    with av.open(io.BytesIO(payload)) as container:
        assert container.streams.video[0].frames == 0
        decoded = np.stack([frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)])
    decoder = VideoDecoder()
    for _ in range(2):
        actual = sample_video(payload, decoder=decoder)
        reference = decoded[np.linspace(0, len(decoded) - 1, 8, dtype=int)].transpose(0, 3, 1, 2)
        np.testing.assert_allclose(actual.cpu().numpy(), reference, rtol=0, atol=3)


@pytest.mark.parametrize("variant", ["224p", "336p", "448p"])
def test_reference_video_inference(variant: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Match pinned processor output and the model's normalized visual_proj on a GPU."""
    import torch  # noqa: PLC0415 -- GPU-only test
    from transformers import AutoProcessor  # noqa: PLC0415

    spec: ModelSpec = resolve_model_spec(variant)
    if not spec.model_dir.exists():
        monkeypatch.setattr(
            environment, "CONTAINER_PATHS_MODEL_WEIGHT_CACHE_DIR", environment.LOCAL_WORKSPACE_PATH / "models"
        )
    if not spec.model_dir.exists() or not torch.cuda.is_available():
        pytest.skip("Staged Cosmos-Embed1 weights and a GPU are required")
    directory = validate_model_directory(spec)
    payload = video_bytes(15)
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(payload)
    corrupt_clip = tmp_path / "corrupt.mp4"
    corrupt_clip.write_bytes(b"not a video")
    row = {
        "clip_id": "test-clip",
        "fragment_id": 0,
        "row_offset": 0,
        "clip_uri": str(clip),
        "clip_size_bytes": len(payload),
    }
    missing = row | {"clip_id": "missing", "clip_uri": str(tmp_path / "missing.mp4")}
    corrupt = row | {
        "clip_id": "corrupt",
        "clip_uri": str(corrupt_clip),
        "clip_size_bytes": corrupt_clip.stat().st_size,
    }
    downloaded = pa.Table.from_pylist([download_clip(source, max_attempts=1) for source in (row, missing, corrupt)])

    # Sampling is checked against a full CPU decode above. Independently apply
    # the complete pinned processor on those CUDA frames for the model reference.
    reference_batch = sample_video(payload).unsqueeze(0)
    processor = AutoProcessor.from_pretrained(str(directory), trust_remote_code=True, local_files_only=True)
    reference_inputs = processor(videos=reference_batch)
    assert reference_inputs["videos"].is_cuda

    config = resolve_config_data(
        {
            "schema_version": 1,
            "kind": "video-embed",
            "input": {"media_root": str(tmp_path)},
            "model": {"variant": variant},
        }
    )
    staged_schema = resolve_bindings(config).spec.staged_schema
    embedder = VideoEmbedder(spec=spec, model_dir=str(directory), staged_schema=staged_schema)
    model_inference = embedder._model.get_video_embeddings
    projections: list[npt.NDArray[np.float32]] = []

    def capture_projection(*, videos: torch.Tensor) -> object:
        torch.testing.assert_close(videos, reference_inputs["videos"].to(dtype=torch.bfloat16), rtol=0, atol=0)
        output = model_inference(videos=videos)
        projections.append(output.visual_proj.float().cpu().numpy().copy())
        return output

    with monkeypatch.context() as patch:
        patch.setattr(embedder._model, "get_video_embeddings", capture_projection)
        results = embedder(downloaded).to_pylist()
    result = results[0]
    assert [value[spec.metadata_field]["status"] for value in results] == ["success", "error", "error"]
    assert all(value[spec.vector_field] is None for value in results[1:])
    assert len(projections) == 1
    # Preserve the actual model projection exactly, catching extra normalization
    # or other postprocessing even when it fits the independent-call tolerance.
    np.testing.assert_array_equal(result[spec.vector_field], projections[0][0])
    with torch.inference_mode():
        expected = (
            embedder._model.get_video_embeddings(**reference_inputs.to("cuda", dtype=torch.bfloat16))
            .visual_proj.float()
            .cpu()
            .numpy()[0]
        )
    np.testing.assert_array_equal(result[spec.vector_field], expected)
    assert np.isclose(np.linalg.norm(expected), 1, atol=0.02)
    assert len(result[spec.vector_field]) == spec.dimension
    model_inference = None
    del embedder
    gc.collect()
    torch.cuda.empty_cache()


def test_full_recipe_on_gpu(config: ResolvedVideoEmbedConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run download tasks, NVDEC/model actors, checkpoints, and atomic Lance publication."""
    import lance  # noqa: PLC0415 -- runtime-only dependencies
    import ray  # noqa: PLC0415
    import torch  # noqa: PLC0415

    from cosmos_curator.next.recipes.video_embed.pipeline import run_config  # noqa: PLC0415

    spec = resolve_model_spec(config.model.variant)
    if not spec.model_dir.exists():
        monkeypatch.setattr(
            environment, "CONTAINER_PATHS_MODEL_WEIGHT_CACHE_DIR", environment.LOCAL_WORKSPACE_PATH / "models"
        )
    if not spec.model_dir.exists() or not torch.cuda.is_available():
        pytest.skip("Staged Cosmos-Embed1 weights and a GPU are required")
    input_table = write_clips(config, payload=video_bytes(8))
    last_uri = input_table.to_table(columns=["clip_uri"]).column(0)[-1].as_py()
    Path(last_uri).unlink()
    try:
        ray.init(
            address="local",
            num_cpus=6,
            num_gpus=1,
            resources={environment.CURATOR_IO_RESOURCE_NAME: 1},
            include_dashboard=False,
            log_to_driver=False,
            object_store_memory=256 * 1024**2,
        )
        summary = run_config(config)
        published = lance.dataset(config.input.clips_lance_uri)
        assert summary["input_rows"] == 4
        assert summary["cleanup_complete"] is True
        assert published.version == input_table.version + 2  # field registration, then one data commit
        rows = published.to_table(columns=spec.fields.names).to_pylist()
        assert [row[spec.metadata_field]["status"] for row in rows] == ["success", "success", "success", "error"]
        assert all(len(row[spec.vector_field]) == 768 for row in rows[:-1])
        assert rows[-1][spec.vector_field] is None
    finally:
        ray.shutdown()
