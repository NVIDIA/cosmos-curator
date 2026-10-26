# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small local destinations with immutable S3 source identities and deterministic source plans."""

from fractions import Fraction
from pathlib import Path
from typing import Any

import av
import numpy as np
import pyarrow as pa
import pytest

from cosmos_curator.next.media.ffmpeg import VideoMetadata
from cosmos_curator.next.recipes.video_cut import pipeline
from cosmos_curator.next.recipes.video_cut.bindings import resolve_bindings
from cosmos_curator.next.recipes.video_cut.config import FixedStrideConfig, ResolvedVideoCutConfig, resolve_config_data
from cosmos_curator.next.recipes.video_cut.contracts import error_result, make_source_id, media_fields
from cosmos_curator.next.recipes.video_cut.discovery import discover_items
from cosmos_curator.next.recipes.video_cut.fixed_stride import planned_source
from tests.cosmos_curator.next.append.conformance import RecipeUnderTest

SOURCE_URIS = (
    "s3://test-bucket/long.mp4",
    "s3://test-bucket/long-2.mp4",
    "s3://test-bucket/short.mp4",
    "s3://test-bucket/error.mp4",
)


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
    """Return a two-second cutter with deliberately small checkpoint groups and fragments."""
    return resolve_config_data(
        {
            "schema_version": 1,
            "kind": "video-cut",
            "input": {"uris": list(SOURCE_URIS)},
            "output": {"media_root": str(tmp_path)},
            "split": {"duration_s": 2.0, "stride_s": 2.0, "min_duration_s": 1.0},
            "execution": {"parquet_rows_per_file": 2, "rows_per_fragment": 2},
        }
    )


def probed_source(uri: str, duration_ns: int) -> dict[str, Any]:
    """Describe a source as both methods' probes do, with synthetic media fields."""
    source = {"source_id": make_source_id(uri), "source_uri": uri}
    return source | media_fields(VideoMetadata(duration_ns, 64, 48, 30.0, None, "h264"), 50)


def plan_outcome(uri: str, config: ResolvedVideoCutConfig) -> dict[str, Any]:
    """Plan a source deterministically from its name: long sources cut into three two-second spans."""
    if uri.endswith("error.mp4"):
        source = {"source_id": make_source_id(uri), "source_uri": uri}
        return error_result(source, error_stage="source-read", error_message="synthetic missing source")
    assert isinstance(config.split, FixedStrideConfig)
    return planned_source(probed_source(uri, 100 if uri.endswith("short.mp4") else 5_000_000_000), config.split)


@pytest.fixture
def results(config: ResolvedVideoCutConfig) -> pa.Table:
    """Build canonical source outcomes: two with spans, an empty plan, and a missing source."""
    assert config.input.uris is not None
    return pa.Table.from_pylist(
        [plan_outcome(uri, config) for uri in config.input.uris], schema=resolve_bindings(config).spec.result_fields
    )


def _retune(config: ResolvedVideoCutConfig) -> ResolvedVideoCutConfig:
    overrides = [
        "execution.parquet_rows_per_file=3",
        "execution.rows_per_fragment=5",
        "execution.fixed_stride.probe_cpus=0.5",
    ]
    return resolve_config_data(config.model_dump(mode="json"), overrides=overrides)


def _recontract(config: ResolvedVideoCutConfig) -> ResolvedVideoCutConfig:
    return resolve_config_data(config.model_dump(mode="json"), overrides=["split.stride_s=1.0"])


# How the shared conformance suite and fresh-cluster scenario drive this recipe.
RECIPE = RecipeUnderTest(
    pipeline=pipeline,
    resolve_bindings=resolve_bindings,
    discover=discover_items,
    retune=_retune,
    recontract=_recontract,
)


@pytest.fixture
def recipe() -> RecipeUnderTest:
    """Drive the video-cut pipeline with its own bindings and discovery."""
    return RECIPE
