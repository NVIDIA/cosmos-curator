# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Download clips on CPU tasks; decode, preprocess, and embed them on the GPU."""

import io
from collections import OrderedDict
from functools import cache
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, Any

import av
import numpy as np
import numpy.typing as npt
import pyarrow as pa

from cosmos_curator.next.recipes.video_embed.contracts import ADDRESS_SCHEMA, NUM_FRAMES, ModelSpec, terminal_row
from cosmos_curator.next.utils.media_reader import MediaObjectError, MediaObjectReader

if TYPE_CHECKING:
    from collections.abc import Generator

    import torch

INPUT_SCHEMA = pa.schema([*ADDRESS_SCHEMA, ("clip_uri", pa.large_string()), ("clip_size_bytes", pa.int64())])
_DECODER_CACHE_SIZE = 4


@cache
def _clip_reader(storage_profile: str, max_attempts: int) -> MediaObjectReader:
    """Reuse storage connections in each task worker without a dedicated actor pool."""
    return MediaObjectReader(storage_profile=storage_profile, max_attempts=max_attempts)


def download_clip(row: dict[str, Any], *, storage_profile: str = "default", max_attempts: int = 3) -> dict[str, Any]:
    """Fetch encoded clip bytes, turning only deterministic object faults into terminal data."""
    try:
        payload = _clip_reader(storage_profile, max_attempts).read(row["clip_uri"], max_bytes=row["clip_size_bytes"])
        if len(payload) != row["clip_size_bytes"]:
            msg = "Clip byte length differs from video-transcode metadata"
            raise MediaObjectError(msg)  # noqa: TRY301 -- handle size and object faults identically
    except MediaObjectError as exc:
        return row | {"clip_bytes": None, "error_type": type(exc).__name__, "error_message": str(exc)}
    return row | {"clip_bytes": payload, "error_type": None, "error_message": None}


def sample_indices(frame_count: int) -> npt.NDArray[np.int64]:
    """Use the model card's endpoint-inclusive integer linspace, including repeats."""
    if frame_count < 1:
        msg = "Clip contains no video frames"
        raise MediaObjectError(msg)
    return np.linspace(0, frame_count - 1, NUM_FRAMES, dtype=np.int64)


def sample_video(payload: bytes, *, decoder: "VideoDecoder | None" = None) -> "torch.Tensor":
    """Keep eight presentation-ordered RGB frames in CUDA memory.

    MP4 headers normally provide the exact frame count. Otherwise count in an
    initial NVDEC pass. Always drain reordered frames and verify the count.
    PyNvVideoCodec 2.0's memory callback cannot seek, so use a temporary file to
    also support MP4s whose moov atom follows their media data.
    """
    try:
        with av.open(io.BytesIO(payload), mode="r") as container:
            if not container.streams.video:
                msg = "Clip contains no video stream"
                raise MediaObjectError(msg)
            stream = container.streams.video[0]
            frame_count, stream_index = stream.frames, stream.index
    except (av.InvalidDataError, av.EOFError) as exc:
        msg = f"Invalid video bitstream ({type(exc).__name__})"
        raise MediaObjectError(msg) from exc
    if decoder is None:
        decoder = VideoDecoder()
    with NamedTemporaryFile(suffix=".mp4") as clip:
        clip.write(payload)
        clip.flush()
        if not frame_count:
            frame_count = sum(1 for _ in decoder.decode(clip.name, stream_index))
        indices = sample_indices(frame_count)
        wanted = set(indices.tolist())
        selected = {}
        decoded_count = 0
        for index, frame in enumerate(decoder.decode(clip.name, stream_index)):
            if index in wanted:
                # NVDEC reuses surfaces; retain selected frames before decoding more.
                selected[index] = frame.clone()
            decoded_count += 1
    if decoded_count != frame_count:
        msg = f"Decoded {decoded_count} frames but MP4 declares {frame_count}"
        raise MediaObjectError(msg)
    shapes = {tuple(frame.shape) for frame in selected.values()}
    if len(shapes) != 1:
        msg = "Sampled frames have inconsistent dimensions"
        raise MediaObjectError(msg)
    import torch  # noqa: PLC0415 -- only successfully decoded clips require CUDA tensors

    return torch.stack([selected[int(index)] for index in indices]).permute(0, 3, 1, 2).contiguous()


class VideoDecoder:
    """Retain up to four compatible NVDEC sessions for serial use by one actor."""

    def __init__(self) -> None:
        """Create the actor-local cache without initializing CUDA on the driver."""
        self._decoders: OrderedDict[tuple[int, ...], Any] = OrderedDict()

    def decode(self, path: str, stream_index: int) -> "Generator[torch.Tensor]":
        """Yield borrowed RGB surfaces in display order, including delayed B frames."""
        import PyNvVideoCodec as nvc  # type: ignore[import-not-found]  # noqa: PLC0415, N813
        import torch  # noqa: PLC0415
        from torch.utils.dlpack import from_dlpack  # noqa: PLC0415

        demuxer = nvc.CreateDemuxer(path)
        if demuxer.GetVideoStreamId() != stream_index:
            msg = "NVDEC selected a different video stream"
            raise MediaObjectError(msg)
        gpu = torch.cuda.current_device()
        stream = torch.cuda.current_stream()
        key = tuple(
            int(value)
            for value in (
                gpu,
                stream.cuda_stream,
                demuxer.GetNvCodecId(),
                demuxer.Width(),
                demuxer.Height(),
                demuxer.BitDepth(),
                demuxer.ChromaFormat(),
                demuxer.ColorSpace(),
                demuxer.ColorRange(),
            )
        )
        decoder = self._decoders.pop(key, None)
        if decoder is None:
            if len(self._decoders) == _DECODER_CACHE_SIZE:
                self._decoders.popitem(last=False)
            decoder = nvc.CreateDecoder(
                gpuid=gpu,
                codec=demuxer.GetNvCodecId(),
                cudastream=stream.cuda_stream,
                usedevicememory=True,
                outputColorType=nvc.OutputColorType.RGB,
            )
        else:
            # Finish copies from the preceding clip before reusing its surfaces.
            stream.synchronize()
        # In pinned PyNvVideoCodec 2.0, the final empty packet drains the decoder.
        # Leave failed or interrupted sessions out of the cache; NVDEC/CUDA errors
        # still propagate because they are not necessarily bad media.
        for packet in demuxer:
            for frame in decoder.Decode(packet):
                yield from_dlpack(frame)
        self._decoders[key] = decoder


class VideoEmbedder:
    """Decode, preprocess, and run one BF16 Cosmos-Embed1 model on the same GPU."""

    def __init__(self, *, spec: ModelSpec, model_dir: str) -> None:
        """Initialize every new actor incarnation solely from staged weights."""
        import torch  # noqa: PLC0415
        from transformers import AutoModel, AutoProcessor  # noqa: PLC0415

        torch.set_num_threads(1)
        self._spec = spec
        self._decoder = VideoDecoder()
        self._processor = AutoProcessor.from_pretrained(  # type: ignore[no-untyped-call]
            model_dir, trust_remote_code=True, local_files_only=True
        )
        if self._processor.num_video_frames != NUM_FRAMES or self._processor.resolution != spec.resolution:
            msg = "Staged Cosmos-Embed1 processor differs from the selected model-card contract"
            raise ValueError(msg)
        self._model = AutoModel.from_pretrained(model_dir, trust_remote_code=True, local_files_only=True)
        self._model = self._model.to("cuda", dtype=torch.bfloat16).eval()
        if self._model.config.embed_dim != spec.dimension:
            msg = "Staged Cosmos-Embed1 embedding dimension differs from the selected contract"
            raise ValueError(msg)

    def __call__(self, batch: pa.Table) -> pa.Table:
        """Batch valid clips while preserving row order and terminal media errors."""
        import torch  # noqa: PLC0415

        rows = batch.to_pylist()
        positions = []
        inputs = []
        vectors: dict[int, list[float]] = {}
        with torch.inference_mode():
            for index, row in enumerate(rows):
                if row["error_type"] is not None:
                    continue
                try:
                    frames = sample_video(row.pop("clip_bytes"), decoder=self._decoder)
                except MediaObjectError as exc:
                    row.update(error_type=type(exc).__name__, error_message=str(exc))
                    continue
                # The pinned processor owns resize, antialiasing, and scaling.
                videos = self._processor(videos=frames.unsqueeze(0))["videos"]
                expected = (1, NUM_FRAMES, 3, self._spec.resolution, self._spec.resolution)
                if videos.shape != expected or videos.dtype != torch.float32 or not videos.is_cuda:
                    msg = "Pinned processor returned an unexpected CUDA video tensor"
                    raise ValueError(msg)
                inputs.append(videos.to(dtype=torch.bfloat16))
                positions.append(index)
                del frames
            if inputs:
                # The model already L2-normalizes visual_proj. Preserve its BF16
                # result, converting to float32 only for durable storage.
                output = self._model.get_video_embeddings(videos=torch.cat(inputs)).visual_proj.float().cpu().numpy()
                if output.shape != (len(positions), self._spec.dimension):
                    msg = "Cosmos-Embed1 returned an unexpected embedding batch shape"
                    raise ValueError(msg)
                vectors = dict(zip(positions, output.tolist(), strict=True))
        return pa.Table.from_pylist(
            [terminal_row(row, self._spec, vectors.get(index)) for index, row in enumerate(rows)],
            schema=self._spec.result_schema,
        )
