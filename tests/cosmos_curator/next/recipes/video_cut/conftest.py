# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small local destinations with immutable S3 source identities."""

from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pytest

from cosmos_curator.next.recipes.video_cut.config import ResolvedVideoCutConfig, resolve_config_data
from cosmos_curator.next.utils.identity import canonical_digest


@pytest.fixture
def video_path(tmp_path: Path) -> Path:
    """Encode a tiny variable-cadence video with a nonzero presentation origin."""
    path = tmp_path / "source.mp4"
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=25)
        stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
        for offset in (0, 1, 3, 6):
            frame = av.VideoFrame.from_ndarray(np.full((48, 64, 3), offset * 30, dtype=np.uint8), format="rgb24")
            frame.pts = 100 + offset
            frame.time_base = Fraction(1, 25)
            container.mux(stream.encode(frame))
        container.mux(stream.encode())
    return path


@pytest.fixture
def config(tmp_path: Path) -> ResolvedVideoCutConfig:
    """Return a two-second cutter with deliberately small durable groups."""
    return resolve_config_data(
        {
            "schema_version": 1,
            "kind": "video-cut",
            "input": {"uris": ["s3://test-bucket/long.mp4", "s3://test-bucket/short.mp4"]},
            "output": {"media_root": str(tmp_path)},
            "split": {"duration_s": 2.0, "stride_s": 2.0, "min_duration_s": 1.0},
            "execution": {"sources_per_checkpoint": 2, "spans_per_fragment": 2},
        }
    )


@pytest.fixture
def snapshot_id(config: ResolvedVideoCutConfig) -> str:
    """Identify the fixture's selected sources independently of recipe configuration."""
    return canonical_digest(config.input.uris)
