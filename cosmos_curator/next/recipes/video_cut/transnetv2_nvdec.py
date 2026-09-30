# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NVDEC preprocessing for the explicit TransNetV2 GPU cutting contract."""

from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import numpy.typing as npt
import torch
from torch.utils.dlpack import from_dlpack

from cosmos_curator.next.media.ffmpeg import InvalidMediaError, UnsupportedMediaError
from cosmos_curator.next.recipes.video_cut.transnetv2 import normalize_frame_timestamps

# One resize call per frame spent most GPU time on small launches, which also
# serialize badly when actors share a GPU. The byte bound limits 4K staging.
_RESIZE_BATCH_FRAMES = 32
_RESIZE_BATCH_BYTES = 256 * 1024**2


class _BatchedResize:
    """Stage decoded frames on the GPU and antialias-resize each batch to 48x27 RGB in one call."""

    def __init__(self, stream: torch.cuda.Stream) -> None:
        import cvcuda  # type: ignore[import-not-found]  # noqa: PLC0415

        self._cvcuda = cvcuda
        self._stream = cvcuda.as_stream(stream.cuda_stream)
        # Reuse the CV-CUDA output buffer. Retaining one pooled tensor per
        # frame makes allocation progressively slower on long sources.
        self._small = torch.empty((_RESIZE_BATCH_FRAMES, 27, 48, 3), dtype=torch.uint8, device=stream.device)
        self._staging: torch.Tensor | None = None
        self._staged = 0
        self.frames: list[torch.Tensor] = []

    def add(self, pixels: torch.Tensor) -> None:
        """Copy one decoded frame, which the decoder does not retain, into the current batch."""
        if self._staging is None or self._staging.shape[1:] != pixels.shape:
            self.flush()
            capacity = max(1, min(_RESIZE_BATCH_FRAMES, _RESIZE_BATCH_BYTES // pixels.nbytes))
            self._staging = torch.empty((capacity, *pixels.shape), dtype=torch.uint8, device=pixels.device)
        self._staging[self._staged].copy_(pixels)
        self._staged += 1
        if self._staged == len(self._staging):
            self.flush()

    def flush(self) -> None:
        """Resize staged frames, including a final partial batch or one before a resolution change."""
        count, self._staged = self._staged, 0
        if self._staging is None or not count:
            return
        self._cvcuda.hq_resize_into(
            self._cvcuda.as_tensor(self._small[:count], "NHWC"),
            self._cvcuda.as_tensor(self._staging[:count], "NHWC"),
            interpolation=self._cvcuda.Interp.LINEAR,
            antialias=True,
            stream=self._stream,
        )
        self.frames.append(self._small[:count].clone())


def decode_frames_nvdec(path: Path) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.int64]]:
    """Keep full-resolution frames on the GPU; return only tiny RGB frames and native PTS."""
    import PyNvVideoCodec as nvc  # type: ignore[import-not-found]  # noqa: PLC0415, N813

    # PyNvVideoCodec selects a video stream internally. Reject a different stream
    # rather than silently diverging from video-cut's first-video-stream contract.
    try:
        with av.open(str(path)) as container:
            if not container.streams.video:
                msg = "Source has no video stream"
                raise InvalidMediaError(msg)
            stream_index = container.streams.video[0].index
    except av.error.InvalidDataError as exc:
        msg = "Source contains invalid video data"
        raise InvalidMediaError(msg) from exc
    demuxer = nvc.CreateDemuxer(str(path))
    if demuxer.GetVideoStreamId() != stream_index:
        msg = "NVDEC selected a different video stream; use split.decoder=pyav for this source"
        raise UnsupportedMediaError(msg)
    time_base = Fraction(demuxer.GetTimebaseNum(), demuxer.GetTimebaseDen())
    stream = torch.cuda.current_stream()
    decoder = nvc.CreateDecoder(
        gpuid=torch.cuda.current_device(),
        codec=demuxer.GetNvCodecId(),
        cudastream=stream.cuda_stream,
        usedevicememory=True,
        outputColorType=nvc.OutputColorType.RGB,
    )
    resize = _BatchedResize(stream)
    timestamps = []
    # PyNvVideoCodec 2.0.4's demux iterator supplies the final empty packet to
    # drain reordered frames; the decoder has no separate Flush method.
    for packet in demuxer:
        for frame in decoder.Decode(packet):
            pts = frame.getPTS()
            if pts == -(1 << 63):
                msg = "Decoded video frame has no presentation timestamp"
                raise InvalidMediaError(msg)
            timestamps.append(round(pts * time_base * 1_000_000_000))
            resize.add(from_dlpack(frame))
    resize.flush()
    times = normalize_frame_timestamps(timestamps)
    # The common inference windowing code consumes NumPy. Only 3,888 bytes per
    # frame cross to the host, independent of the source's resolution.
    return torch.cat(resize.frames).cpu().numpy(), times
