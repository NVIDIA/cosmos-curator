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

"""Fixtures for ``multimodal-split`` processing tests."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import av
import numpy as np
import pytest
from mcap.writer import Writer as McapWriter

from cosmos_curator.core.sensors.utils.timestamps import write_timestamps_ns
from cosmos_curator.next.recipes.multimodal_split.config import ResolvedMultimodalSplitConfig
from cosmos_curator.next.recipes.multimodal_split.plan import span_planner
from cosmos_curator.next.recipes.multimodal_split.split import split_session

_NS_PER_SECOND = 1_000_000_000
_WIDTH = 32
_HEIGHT = 16

# A recorder's clock when it started this session, taken from real footage: an
# uptime counter, nowhere near zero and nowhere near a wall-clock epoch. Fixtures
# default to it because a session that starts at zero cannot tell a capture
# timeline apart from a presentation one, and that is the whole distinction the
# recipe now turns on.
DEVICE_ORIGIN_NS = 3_517_693_838_000


@pytest.fixture
def make_session() -> Callable[..., Path]:
    """Return a builder for a session directory holding one camera video.

    Frame ``i`` is a flat image of brightness ``i * step``, which makes the clip
    monotonically brighten and gives a decoded frame an approximate index. It is
    only approximate: at the default step of 2, lossy encoding collides adjacent
    frames, so decoded means are ascending but not distinct. A test that needs to
    name *which* source frame was selected should read
    ``selected_source_timestamps_ns``, which is exact, or raise the step first.
    """

    def build(
        root: Path,
        session_id: str,
        *,
        frames: int = 90,
        fps: int = 30,
        uri_pattern: str = "camera/front.mp4",
        step: int = 2,
        origin_ns: int = DEVICE_ORIGIN_NS,
    ) -> Path:
        path = root / session_id / uri_pattern
        path.parent.mkdir(parents=True, exist_ok=True)
        with av.open(str(path), mode="w") as container:
            stream = container.add_stream("libopenh264", rate=fps)
            stream.width, stream.height, stream.pix_fmt = _WIDTH, _HEIGHT, "yuv420p"
            for index in range(frames):
                image = np.full((_HEIGHT, _WIDTH, 3), (index * step) % 256, dtype=np.uint8)
                for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")):
                    container.mux(packet)
            for packet in stream.encode(None):
                container.mux(packet)

        # The observation-timeline artifact the recorder's converter writes beside
        # every camera. Both paths take their bounds from this instead of from the
        # container, so a session without one is not processable. It carries the
        # recorder's clock, which is what makes a camera clip joinable with the
        # other sensors; the container's own timeline starts at zero.
        # round(k * period), not k * round(period): a period that is not a whole
        # number of nanoseconds drifts otherwise, and the drift costs a clip.
        period_ns = _NS_PER_SECOND / fps
        timestamps_ns = np.array([origin_ns + round(index * period_ns) for index in range(frames)], dtype=np.int64)
        with (path.parent / f"{path.name}.timestamps.npy").open("wb") as timeline:
            write_timestamps_ns(timestamps_ns, timeline)
        return root / session_id

    return build


EGO_TOPIC = "/egotrajectory"


@pytest.fixture
def make_ego_mcap() -> Callable[..., Path]:
    """Return a builder for a session's egotrajectory MCAP.

    Written with the upstream MCAP writer rather than the recipe's own, because
    this stands in for what a recorder produced: the recipe reads these files and
    does not write them, and the options that matter here -- chunking, a second
    topic -- are properties of the recorder, not of the cut.

    The payloads are opaque bytes rather than encoded poses. Nothing in this
    recipe decodes them: planning reads the topic's first and last ``log_time``
    from the index, and the cut copies records through unexamined. A fixture that
    encoded real poses would assert the recipe understands a schema it is
    deliberately built not to.
    """

    def build(
        session: Path,
        *,
        first_ns: int = DEVICE_ORIGIN_NS,
        # One past the camera's last frame, which is how a recorder behaves: the
        # trajectory is still logging when the camera stops. At exactly 300 the
        # trajectory would end 10ms early and cost the session its third clip,
        # which is the narrowing rule working, not a fixture quirk -- the test
        # that wants it asks for it.
        count: int = 301,
        rate_hz: int = 100,
        uri_pattern: str = "ego/egotrajectory.mcap",
        topic: str = EGO_TOPIC,
        gap: tuple[int, int] | None = None,
        chunked: bool = True,
        extra_topic: str | None = None,
    ) -> Path:
        period_ns = _NS_PER_SECOND // rate_hz
        # A dropout, as a half-open window of indices. A recorder that loses the
        # trajectory for a while still writes an MCAP whose bounds span the whole
        # session, which is why planning cannot see the hole.
        skipped = range(*gap) if gap is not None else range(0)
        path = session / uri_pattern
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            writer = McapWriter(handle, chunk_size=10_000, use_chunking=chunked)
            writer.start(library="test recorder")
            schema_id = writer.register_schema(name="test.Pose", encoding="test-schema", data=b"schema-bytes")
            channels = {
                name: writer.register_channel(schema_id=schema_id, topic=name, message_encoding="test-bytes")
                for name in ([topic] if extra_topic is None else [topic, extra_topic])
            }
            for index in range(count):
                if index in skipped:
                    continue
                log_time = first_ns + index * period_ns
                for name, channel_id in channels.items():
                    writer.add_message(
                        channel_id=channel_id,
                        log_time=log_time,
                        data=f"{name}-{index}".encode(),
                        publish_time=log_time,
                        sequence=index,
                    )
            writer.finish()
        return path

    return build


def split_all(
    item: dict[str, Any],
    *,
    config: ResolvedMultimodalSplitConfig,
    storage_profile: str = "default",
) -> list[dict[str, Any]]:
    """Plan one session and split every camera it planned, as the pipeline does.

    Splitting takes a planned camera now, not a session, so a test that wants a
    session's clips runs both stages -- which is also what keeps them honest
    about the row the planner actually hands over.
    """
    outcomes = [
        outcome
        for plan in span_planner(item, config=config, storage_profile=storage_profile)
        if plan["result_kind"] == "camera_plan"
        for outcome in split_session(plan, config=config, storage_profile=storage_profile)
    ]
    return [
        {
            **outcome,
            "clip_id": outcome["split_clip_id"],
            "sensor_key": outcome["sensor_key"],
            "clip_location": outcome["media_location"],
            "clip_bytes": outcome["media_size_bytes"],
        }
        for outcome in outcomes
        if outcome["status"] == "success"
    ]


@pytest.fixture
def make_config() -> Callable[..., ResolvedMultimodalSplitConfig]:
    """Return a builder for a resolved config pointed at a temporary tree."""

    def build(
        input_root: Path,
        media_root: Path,
        *,
        duration_s: float = 1.0,
        output_fps: int = 30,
        max_delta_s: float = 0.05,
        uri_pattern: str = "camera/front.mp4",
        camera_key: str = "front",
        cameras: dict[str, str] | None = None,
        optional_cameras: frozenset[str] = frozenset(),
        mcap_sensors: dict[str, str] | None = None,
        optional_sensors: frozenset[str] = frozenset(),
        topic: str | None = None,
        video_bitrate: str = "400K",
        gop_size: int = 5,
        progress: bool = False,
        storage_profile: str = "default",
        memory_gb: float | None = None,
        planner_num_cpus: float = 0.125,
        read_buffer_bytes: int | None = None,
    ) -> ResolvedMultimodalSplitConfig:
        camera_map = cameras or {camera_key: uri_pattern}
        return ResolvedMultimodalSplitConfig.model_validate(
            {
                "schema_version": 1,
                "kind": "multimodal-split",
                "input": {"input_path_prefix": str(input_root)},
                "sensors": {
                    **{
                        key: {
                            "sensor_type": "camera",
                            "uri_pattern": pattern,
                            "required": key not in optional_cameras,
                        }
                        for key, pattern in camera_map.items()
                    },
                    **{
                        key: {
                            "sensor_type": "egotrajectory",
                            "uri_pattern": pattern,
                            "required": key not in optional_sensors,
                            "topic": topic,
                        }
                        for key, pattern in (mcap_sensors or {}).items()
                    },
                },
                "output": {"path": str(media_root)},
                "clip": {"duration_s": duration_s, "sampling_fps": output_fps},
                "video": {
                    "max_delta_s": max_delta_s,
                    "encoder": {"video_bitrate": video_bitrate, "gop_size": gop_size},
                },
                "execution": {
                    "progress": progress,
                    "storage_profile": storage_profile,
                    "split_session": {"memory_gb": memory_gb},
                    "span_planner": {"num_cpus": planner_num_cpus},
                    "read_buffer_bytes": read_buffer_bytes,
                },
            }
        )

    return build
