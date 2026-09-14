# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Measure real FFprobe range traffic for front- and tail-indexed MP4 files."""

import json
import socket
import subprocess
import threading
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pytest

from cosmos_curator.next.media.ffmpeg import probe_video_source
from cosmos_curator.next.recipes.video_cut import processing
from cosmos_curator.next.recipes.video_cut.config import ResolvedVideoCutConfig
from cosmos_curator.next.recipes.video_cut.contracts import INPUT_SCHEMA, make_source_id


@pytest.mark.integration
@pytest.mark.parametrize("faststart", [True, False])
def test_remote_probe_reads_less_than_whole_mp4(
    config: ResolvedVideoCutConfig,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    record_property: Callable[[str, object], None],
    *,
    faststart: bool,
) -> None:
    """The recipe obtains real metadata without its whole-object download fallback."""
    path = tmp_path / "source.mp4"
    subprocess.run(  # noqa: S603
        [  # noqa: S607 - FFmpeg comes from the selected runtime environment.
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=640x360:rate=30",
            "-t",
            "20",
            "-c:v",
            "mpeg4",
            "-q:v",
            "2",
            "-threads",
            "1",
            *(["-movflags", "+faststart"] if faststart else []),
            str(path),
        ],
        capture_output=True,
        check=True,
        timeout=60,
    )
    metadata = probe_video_source(path)
    size = path.stat().st_size
    transfers: list[tuple[str, int]] = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_: object) -> None:
            pass

        def do_GET(self) -> None:
            requested = self.headers.get("Range", "")
            start, end = 0, size - 1
            if requested:
                first, last = requested.removeprefix("bytes=").split("-", maxsplit=1)
                if first:
                    start = int(first)
                    end = min(int(last), end) if last else end
                else:
                    start = max(size - int(last), 0)
            self.send_response(206 if requested else 200)
            self.send_header("Content-Length", str(end - start + 1))
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Accept-Ranges", "bytes")
            if requested:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            self.connection.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 16 * 1024)
            sent = 0
            try:
                with path.open("rb") as source:
                    source.seek(start)
                    while sent < end - start + 1:
                        chunk = source.read(min(64 * 1024, end - start + 1 - sent))
                        view = memoryview(chunk)
                        while view:
                            count = self.connection.send(view)
                            sent += count
                            view = view[count:]
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                transfers.append((requested, sent))

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        server.daemon_threads = False
        thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
        thread.start()
        signed = f"http://127.0.0.1:{server.server_port}/source.mp4?signature=secret"

        def unexpected_download(*_: object) -> None:
            pytest.fail("Successful range probing must not download the source")

        client = SimpleNamespace(
            s3=SimpleNamespace(
                head_object=lambda **_: {"ContentLength": size}, generate_presigned_url=lambda *_, **__: signed
            ),
            download_to_path=unexpected_download,
        )
        monkeypatch.setattr(processing, "s3_client", lambda _: client)
        uri = config.input.uris[0]
        batch = pa.Table.from_pylist([{"source_id": make_source_id(uri), "source_uri": uri}], schema=INPUT_SCHEMA)
        try:
            result = processing.plan_sources(batch, split=config.split, execution=config.execution).to_pylist()[0]
        finally:
            server.shutdown()
            thread.join(timeout=5)
    transferred = sum(count for _, count in transfers)
    record_property("source_bytes", size)
    record_property("probe_bytes_sent", transferred)
    assert result["status"] == "success"
    assert result["source_duration_ns"] == metadata.duration_ns == 20_000_000_000
    assert result["source_frame_count"] == metadata.frame_count == 600
    assert result["source_width"] == metadata.width
    assert result["source_height"] == metadata.height
    assert "secret" not in json.dumps(result)
    assert all(request.startswith("bytes=") for request, _ in transfers)
    assert 0 < transferred < size
