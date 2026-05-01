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
"""Helper script to convert a video file to an MCAP file.

This is not meant to be a production-quality tool, but rather a tool to
create an mcap file for use with McapCameraSensor, which it only meant
to be used as an example of how to use MCAP as a data source for a sensor.

Output:
- One MCAP file. Single topic (default /camera/rgb); each message payload is
  raw uint8 RGBrow-major [H, W, 3], C-contiguous; width and height are stored
  on the channel metadata.
- log_time and publish_time are presentation time in nanoseconds
  (MCAP uint64): PTS x time_base when both are set, otherwise frame.time
  (seconds) scaled to nanoseconds.

Run:

python -m cosmos_curator.core.sensors.scripts.make_mcap_from_mp4 --source in.mp4 --dest out.mcap
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, cast

import av
import numpy as np
import numpy.typing as npt
from mcap.writer import CompressionType, Writer
from tqdm import tqdm

from cosmos_curator.core.sensors.types.types import VideoIndexCreationMethod
from cosmos_curator.core.sensors.utils.io import open_file
from cosmos_curator.core.sensors.utils.mcap import VIDEO_METADATA_RECORD_NAME
from cosmos_curator.core.sensors.utils.video import make_index_and_metadata, pts_to_ns

_RGB_CHANNELS = 3

_RGB_SCHEMA = {
    "type": "object",
    "title": "cosmos_curator.sensors.rgb8_frame",
    "description": (
        "Opaque message body: exactly width*height*3 bytes, RGB8 row-major (H rows x W pixels x 3). "
        "width and height are repeated as strings on the MCAP channel metadata."
    ),
}


def _decode_frame_rgb24(frame: av.VideoFrame) -> npt.NDArray[np.uint8]:
    """Decode *frame* to contiguous uint8 RGB (H, W, 3)."""
    arr: npt.NDArray[Any] = frame.to_ndarray(format="rgb24")
    if not arr.flags.c_contiguous:
        arr = np.ascontiguousarray(arr)
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8, copy=False)
    return cast("npt.NDArray[np.uint8]", arr)


def write_mcap(
    dest: Path,
    source: Path,
    *,
    topic: str,
    index_method: VideoIndexCreationMethod = VideoIndexCreationMethod.FROM_HEADER,
) -> tuple[int, int, int]:
    """Decode all video frames and write MCAP. Returns (frame_count, height, width)."""
    index, metadata = make_index_and_metadata(source, index_method=index_method)
    expected_frames = int(index.pts_ns.shape[0])

    dest.parent.mkdir(parents=True, exist_ok=True)

    with (
        open_file(dest, mode="wb") as out_file,
        av.open(str(source)) as container,
    ):
        writer: Writer | None = None
        try:
            writer = Writer(out_file, compression=CompressionType.ZSTD)
            writer.start(library="cosmos_curator make_mcap_from_mp4")

            schema_id = writer.register_schema(
                name=_RGB_SCHEMA["title"],
                encoding="jsonschema",
                data=json.dumps(_RGB_SCHEMA).encode("utf-8"),
            )

            if not container.streams.video:
                msg = f"No video stream in {source}"
                raise ValueError(msg)

            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            stream.thread_count = 0
            ctx = stream.codec_context
            width = int(ctx.width)
            height = int(ctx.height)
            if width <= 0 or height <= 0:
                msg = f"Invalid coded dimensions width={width} height={height}"
                raise ValueError(msg)

            frame_bytes = width * height * _RGB_CHANNELS

            channel_id = writer.register_channel(
                schema_id=schema_id,
                topic=topic,
                message_encoding="rgb8",
                metadata={
                    "width": str(width),
                    "height": str(height),
                },
            )

            writer.add_metadata(VIDEO_METADATA_RECORD_NAME, metadata.to_string_dict())

            written = 0
            for frame in tqdm(
                container.decode(stream),
                total=expected_frames,
                unit="frame",
                desc="Decode → MCAP",
            ):
                if frame.pts is None or frame.time_base is None:
                    msg = "Frame PTS or time_base is None"
                    raise ValueError(msg)
                t_ns = pts_to_ns(frame.pts, frame.time_base)
                arr = _decode_frame_rgb24(frame)
                raw = arr.tobytes()
                if len(raw) != frame_bytes:
                    msg = (
                        f"Decoded frame size {len(raw)} != expected {frame_bytes} "
                        f"(codec context {width}x{height}); "
                        "coded size may not match decoded RGB output."
                    )
                    raise ValueError(msg)
                seq = written + 1
                writer.add_message(
                    channel_id=channel_id,
                    log_time=t_ns,
                    data=raw,
                    publish_time=t_ns,
                    sequence=seq,
                )
                written += 1

            if written != expected_frames:
                msg = (
                    f"Decoded {written} frames but VideoIndex packet count is {expected_frames}; "
                    "decode count does not match index (see get_video_index in video utils)."
                )
                raise ValueError(msg)
        finally:
            if writer is not None:
                writer.finish()  # type: ignore[no-untyped-call]

    return written, height, width


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description=(
            "Decode all video frames with PyAV into an MCAP file with one channel of raw RGB8 "
            "(row-major HxWx3 uint8; timestamps in nanoseconds per MCAP message fields)."
        ),
    )
    parser.add_argument("--source", type=Path, required=True, help="Input video (e.g. MP4)")
    parser.add_argument("--dest", type=Path, required=True, help="Output .mcap path")
    parser.add_argument(
        "--topic",
        type=str,
        default="/camera/rgb",
        help="MCAP topic for RGB frames (default: /camera/rgb)",
    )
    parser.add_argument(
        "--full-demux-index",
        action="store_true",
        help=(
            "Build VideoIndex via full demux (slow); default uses container header index "
            "(same as VideoIndexCreationMethod.FROM_HEADER)."
        ),
    )
    args = parser.parse_args()

    if not args.source.is_file():
        sys.stderr.write(f"error: source is not a file: {args.source}\n")
        sys.exit(1)

    try:
        mode = VideoIndexCreationMethod.FULL_DEMUX if args.full_demux_index else VideoIndexCreationMethod.FROM_HEADER
        n, height, width = write_mcap(args.dest, args.source, topic=args.topic, index_method=mode)
    except ValueError as e:
        sys.stderr.write(f"error: {e}\n")
        sys.exit(1)

    frame_bytes = width * height * _RGB_CHANNELS
    sys.stdout.write(f"Wrote {n} frames ({height}x{width} RGB, {frame_bytes} B/frame) -> {args.dest}\n")


if __name__ == "__main__":
    main()
