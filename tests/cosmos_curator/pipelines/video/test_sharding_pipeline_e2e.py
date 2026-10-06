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
"""CPU end-to-end coverage for the video shard pipeline's webdataset output.

Builds a synthetic split-pipeline output tree, runs the production ``shard`` CLI entry point with the T5 encoder
stubbed and ``run_pipeline`` routed through ``SequentialRunner``, then validates the written tar shard layout.
"""

import argparse
import functools
import json
import pathlib
import pickle
import sys
import tarfile
import uuid
import zlib
from typing import Any

import attrs
import numpy as np
import numpy.typing as npt
import pytest
from loguru import logger

from cosmos_curator.core.interfaces.pipeline_interface import run_pipeline
from cosmos_curator.core.interfaces.runner_interface import RunnerInterface
from cosmos_curator.models import t5_encoder
from cosmos_curator.pipelines.video import sharding_pipeline
from cosmos_curator.pipelines.video.captioning import captioning_stages

_T5_HIDDEN_DIM = 1024  # T5-XXL d_model, matching captioning/example_t5_embeddings.pickle
_FRAMERATE = 30.0
_TEN_SECONDS = 300
_THREE_SECONDS = 90  # below the 5s minimum of every standard duration bin
# Declared via metadata ``num_bytes``: exactly two clips fill one ``--target-tar-size-mb 1`` tar.
_DECLARED_CLIP_BYTES = 512 * 1024
_MAX_TARS_PER_PART = 2
_DATA_KEY_EXTENSIONS = {"video": "mp4", "metas": "json", "t5_xxl": "pickle"}
_BIN_1080 = "resolution_1080/aspect_ratio_16_9/duration_5_10"
_BIN_720 = "resolution_720/aspect_ratio_16_9/duration_5_10"


@attrs.frozen
class _ClipSpec:
    name: str
    width: int = 1920
    height: int = 1080
    num_frames: int = _TEN_SECONDS
    num_windows: int = 2
    valid: bool = True

    @property
    def uuid(self) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, self.name))


_CLIP_SPECS = (
    *(_ClipSpec(f"hd-{idx}") for idx in range(4)),
    _ClipSpec("hd-three-windows", num_windows=3),
    _ClipSpec("720p", width=1280, height=720, num_windows=1),
    _ClipSpec("invalid", valid=False),
    _ClipSpec("too-short", num_frames=_THREE_SECONDS),
)


def _fake_embedding(prompt: str) -> npt.NDArray[np.float16]:
    """Return a deterministic, prompt-specific embedding shaped like a truncated T5-XXL output."""
    num_tokens = len(prompt.split()) + 1  # + EOS
    rng = np.random.default_rng(zlib.crc32(prompt.encode()))
    return rng.standard_normal((num_tokens, _T5_HIDDEN_DIM)).astype(np.float16)


def _fake_encode(_self: t5_encoder.T5Encoder, prompts: list[str], **_kwargs: object) -> list[t5_encoder.EncodedSample]:
    samples = []
    for prompt in prompts:
        embedding = _fake_embedding(prompt)
        samples.append(t5_encoder.EncodedSample(embedding, len(embedding), np.ones(len(embedding), dtype=np.int32)))
    return samples


def _clip_metadata(spec: _ClipSpec, input_dir: pathlib.Path) -> dict[str, Any]:
    """Return per-clip metadata in the split pipeline's ``metas/v0/<uuid>.json`` format."""
    window_ns = int(spec.num_frames / _FRAMERATE * 1e9) // spec.num_windows
    return {
        "span_uuid": spec.uuid,
        "source_video": str(input_dir / "videos" / "source.mp4"),
        "clip_location": str(input_dir / "clips" / f"{spec.uuid}.mp4"),
        "width": spec.width,
        "height": spec.height,
        "framerate": _FRAMERATE,
        "num_frames": spec.num_frames,
        "num_bytes": _DECLARED_CLIP_BYTES,
        "valid": spec.valid,
        "has_caption": True,
        "windows": [
            {
                "start_ns": idx * window_ns,
                "end_ns": (idx + 1) * window_ns,
                "caption_status": "success",
                "qwen_caption": f"Clip {spec.name} window {idx}: a red car drives down a sunny street.",
            }
            for idx in range(spec.num_windows)
        ],
    }


def _clip_payload(spec: _ClipSpec) -> bytes:
    """Return distinct per-clip bytes; the writer packs clips opaquely, so decodability is irrelevant."""
    return spec.uuid.encode() * 64


def _write_split_output(input_dir: pathlib.Path) -> dict[str, dict[str, Any]]:
    """Write a minimal split-pipeline output tree and return clip metadata keyed by clip UUID."""
    metas_dir = input_dir / "metas" / "v0"
    clips_dir = input_dir / "clips"
    metas_dir.mkdir(parents=True)
    clips_dir.mkdir()
    metadata_by_uuid = {}
    for spec in _CLIP_SPECS:
        metadata = _clip_metadata(spec, input_dir)
        (metas_dir / f"{spec.uuid}.json").write_text(json.dumps(metadata))
        (clips_dir / f"{spec.uuid}.mp4").write_bytes(_clip_payload(spec))
        metadata_by_uuid[spec.uuid] = metadata
    summary = {"source.mp4": {"clips": [spec.uuid for spec in _CLIP_SPECS]}}
    (input_dir / "summary.json").write_text(json.dumps(summary))
    return metadata_by_uuid


def _read_tar(path: pathlib.Path) -> list[tuple[str, bytes]]:
    members = []
    with tarfile.open(path) as tar:
        for member in tar.getmembers():
            payload = tar.extractfile(member)
            assert payload is not None, f"{path}:{member.name} is not a regular file"
            members.append((member.name, payload.read()))
    return members


def _shard_args(input_dir: pathlib.Path, output_dir: pathlib.Path) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sharding_pipeline._setup_parser(parser)
    return parser.parse_args(
        [
            "--input-clip-path",
            str(input_dir),
            "--output-dataset-path",
            str(output_dir),
            "--target-tar-size-mb",
            "1",
            "--max-tars-per-part",
            str(_MAX_TARS_PER_PART),
        ]
    )


def test_shard_writes_webdataset_tar_schema(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
    sequential_runner: RunnerInterface,
    capfd: pytest.CaptureFixture[str],
) -> None:
    """Shard valid, binnable clips into parallel video/metas/t5_xxl tars with matching per-sample keys."""
    monkeypatch.setattr(t5_encoder.T5Encoder, "setup", lambda _self: None)
    monkeypatch.setattr(t5_encoder.T5Encoder, "encode", _fake_encode)
    # NVML startup probing can sleep for minutes waiting on a busy GPU host.
    monkeypatch.setattr(captioning_stages, "gpu_stage_startup", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        sharding_pipeline,
        "run_pipeline",
        functools.partial(run_pipeline, runner=sequential_runner),
    )

    input_dir = tmp_path / "split_output"
    output_dir = tmp_path / "dataset"
    metadata_by_uuid = _write_split_output(input_dir)
    payload_by_uuid = {spec.uuid: _clip_payload(spec) for spec in _CLIP_SPECS}
    specs_by_name = {spec.name: spec for spec in _CLIP_SPECS}

    # Other tests can remove Loguru's default sinks; bind the captured stderr explicitly.
    sink_id = logger.add(sys.stderr, format="{message}", level="WARNING")
    try:
        sharding_pipeline.shard(_shard_args(input_dir, output_dir))
    finally:
        logger.remove(sink_id)
    captured_err = capfd.readouterr().err

    # Input samples are ordered by metadata path (i.e. UUID), then packed two per tar and two tars per part.
    hd_uuids = sorted(spec.uuid for spec in _CLIP_SPECS if spec.name.startswith("hd-"))
    expected_tars = {
        (_BIN_1080, "part_000000/000000.tar"): hd_uuids[0:2],
        (_BIN_1080, "part_000000/000001.tar"): hd_uuids[2:4],
        (_BIN_1080, "part_000001/000000.tar"): hd_uuids[4:],
        (_BIN_720, "part_000000/000000.tar"): [specs_by_name["720p"].uuid],
    }
    dataset_root = output_dir / "v0"
    bins = sorted({lbin for lbin, _ in expected_tars})
    written_files = {str(path.relative_to(dataset_root)) for path in dataset_root.rglob("*") if path.is_file()}
    assert written_files == {
        "wdinfo_list.csv",
        *(f"{lbin}/wdinfo.json" for lbin in bins),
        *(f"{lbin}/{data_key}/{tar}" for lbin, tar in expected_tars for data_key in _DATA_KEY_EXTENSIONS),
    }

    for (lbin, tar), keys in expected_tars.items():
        members = {data_key: _read_tar(dataset_root / lbin / data_key / tar) for data_key in _DATA_KEY_EXTENSIONS}
        for data_key, extension in _DATA_KEY_EXTENSIONS.items():
            assert [name for name, _ in members[data_key]] == [f"{key}.{extension}" for key in keys]

        for key, (_, mp4), (_, metas), (_, t5_xxl) in zip(
            keys, members["video"], members["metas"], members["t5_xxl"], strict=True
        ):
            assert mp4 == payload_by_uuid[key]
            metadata = json.loads(metas)
            assert metadata == metadata_by_uuid[key]
            embeddings = pickle.loads(t5_xxl)  # noqa: S301
            for embedding, window in zip(embeddings, metadata["windows"], strict=True):
                assert embedding.dtype == np.float16
                np.testing.assert_array_equal(embedding, _fake_embedding(window["qwen_caption"]))

    for lbin in bins:
        bin_tars = sorted(tar for tar_bin, tar in expected_tars if tar_bin == lbin)
        assert json.loads((dataset_root / lbin / "wdinfo.json").read_text()) == {
            "data_keys": ["video", "t5_xxl", "metas"],
            "chunk_size": _MAX_TARS_PER_PART,
            "data_list": bin_tars,
            "root": str(dataset_root / lbin),
            "total_key_count": sum(len(expected_tars[lbin, tar]) for tar in bin_tars),
        }
    assert sorted((dataset_root / "wdinfo_list.csv").read_text().splitlines()) == [
        str(dataset_root / lbin / "wdinfo.json") for lbin in bins
    ]

    assert f"Clip {specs_by_name['invalid'].uuid} is invalid, skipping" in captured_err
    assert "Found 1 samples which do not correspond to a lbin" in captured_err
