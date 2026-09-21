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
"""Tests for config-driven split output validation."""

import json
from pathlib import Path
from unittest.mock import Mock

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cosmos_curator.core.utils.config.pipeline_config_loader import load_pipeline_config
from cosmos_curator.core.utils.storage import storage_utils
from cosmos_curator.pipelines.video.output_validation import main, validate_split_output

_REPO_ROOT = Path(__file__).resolve().parents[4]
_CI_CONFIGS = sorted((_REPO_ROOT / "examples/ci").glob("*.json"))
_CAPTION_STATS = {
    "schema_version": 1,
    "pipeline": "split_video_pipeline",
    "caption_windows_checked": 1,
    "caption_status_counts": {
        "success": 1,
        "truncated": 0,
        "blocked": 0,
        "error": 0,
        "skipped": 0,
    },
    "caption_failure_reason_counts": {"exception": 0, "timeout": 0},
    "caption_quality_flags_evaluated_count": 0,
    "caption_quality_flag_counts": {},
    "empty_caption_count": 0,
    "sentinel_caption_count": 0,
}


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _write_config(path: Path, **overrides: object) -> None:
    args: dict[str, object] = {
        "input_video_path": "/input",
        "output_clip_path": "/output",
        "output_s3_profile_name": "default",
        "embedding_algorithm": "cosmos-embed1-224p",
        "generate_embeddings": True,
        "generate_captions": True,
        "caption_quality_stats_enabled": True,
        "generate_previews": True,
        "upload_clips": True,
        "upload_cds_parquet": True,
        "generate_cosmos_predict_dataset": True,
    }
    args.update(overrides)
    _write_json(
        path,
        {
            "pipeline": "split",
            "expected_output": {
                "min_processed_videos": 1,
                "min_passed_clips": 1,
            },
            "args": args,
        },
    )


def _write_complete_output(root: Path) -> None:
    _write_json(
        root / "summary.json",
        {
            "num_processed_videos": 1,
            "total_num_clips_passed": 1,
            "total_num_clips_transcoded": 1,
            "total_num_clips_with_embeddings": 1,
            "total_num_clips_with_caption": 1,
            "total_num_caption_windows": 1,
            "total_num_clips_with_webp": 1,
        },
    )
    _write_json(root / "processed_videos/video.mp4.json", {"num_clip_chunks": 1})
    _write_json(
        root / "processed_clip_chunks/video.mp4_0.json",
        {"all_windows": {"clip-id": {"0_10": "A generated caption."}}},
    )
    _write_json(root / "caption_quality_stats.json", _CAPTION_STATS)
    embedding_path = root / "ce1_embd_224p_parquet/video-id_0.parquet"
    embedding_path.parent.mkdir(parents=True)
    pq.write_table(pa.table({"id": ["clip-id"], "embedding": [[1.0, 2.0]]}), embedding_path)
    cds_path = root / "cds_parquet/video-id_0.parquet"
    cds_path.parent.mkdir(parents=True)
    pq.write_table(
        pa.table({"id": ["clip-id"], "embedding": [[1.0, 2.0]], "$meta": ['{"caption": "text"}']}),
        cds_path,
    )
    for relative_path in (
        "clips/clip-id.mp4",
        "previews/clip-id/0_10.webp",
        "cosmos_predict2_video2world_dataset/videos/clip-id_0_10.mp4",
        "cosmos_predict2_video2world_dataset/metas/clip-id_0_10.txt",
        "cosmos_predict2_video2world_dataset/t5_xxl/clip-id_0_10.pickle",
    ):
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"output")


def test_validate_split_output_checks_enabled_artifacts(tmp_path: Path) -> None:
    """A complete output should satisfy summary, metadata, and enabled artifact checks."""
    config_path = tmp_path / "scenario.json"
    output_path = tmp_path / "output"
    _write_config(config_path)
    _write_complete_output(output_path)

    report = validate_split_output(config_path, output_path.as_posix())

    assert report.passed
    assert report.errors == ()
    assert any(check.startswith("caption quality") for check in report.checks)
    assert any(check.startswith("CDS parquet") for check in report.checks)
    assert any(check.startswith("Cosmos-Predict T5 embeddings") for check in report.checks)


def test_validate_split_output_reports_missing_enabled_artifacts(tmp_path: Path) -> None:
    """Missing configured outputs should be reported together."""
    config_path = tmp_path / "scenario.json"
    output_path = tmp_path / "output"
    _write_config(config_path)
    _write_json(
        output_path / "summary.json",
        {
            "num_processed_videos": 0,
            "total_num_clips_passed": 0,
            "total_num_clips_transcoded": 0,
            "total_num_clips_with_embeddings": 0,
            "total_num_clips_with_caption": 0,
            "total_num_caption_windows": 0,
            "total_num_clips_with_webp": 0,
        },
    )

    report = validate_split_output(config_path, output_path.as_posix())

    assert not report.passed
    assert any("num_processed_videos" in error for error in report.errors)
    assert any("embedding parquet" in error for error in report.errors)
    assert any("caption quality stats" in error for error in report.errors)
    assert any("Cosmos-Predict videos" in error for error in report.errors)


def test_validate_split_output_rejects_invalid_parquet(tmp_path: Path) -> None:
    """Every named parquet object must be readable and contain the output schema."""
    config_path = tmp_path / "scenario.json"
    output_path = tmp_path / "output"
    _write_config(
        config_path,
        generate_captions=False,
        generate_previews=False,
        upload_clips=False,
        upload_cds_parquet=False,
        generate_cosmos_predict_dataset=False,
    )
    _write_json(
        output_path / "summary.json",
        {
            "num_processed_videos": 1,
            "total_num_clips_passed": 1,
            "total_num_clips_transcoded": 1,
            "total_num_clips_with_embeddings": 1,
        },
    )
    _write_json(output_path / "processed_videos/video.mp4.json", {})
    _write_json(output_path / "processed_clip_chunks/video.mp4_0.json", {})
    parquet_root = output_path / "ce1_embd_224p_parquet"
    parquet_root.mkdir(parents=True)
    pq.write_table(pa.table({"id": ["clip-id"], "embedding": [[1.0]]}), parquet_root / "valid.parquet")
    (parquet_root / "nested").mkdir()
    (parquet_root / "nested/invalid.parquet").write_bytes(b"not parquet")

    report = validate_split_output(config_path, output_path.as_posix())

    assert not report.passed
    assert any("nested/invalid.parquet" in error for error in report.errors)


def test_validate_split_output_requires_aggregate_parquet_rows(tmp_path: Path) -> None:
    """Parquet rows across all shards must satisfy the expected clip minimum."""
    config_path = tmp_path / "scenario.json"
    output_path = tmp_path / "output"
    _write_config(
        config_path,
        generate_captions=False,
        generate_previews=False,
        upload_clips=False,
        upload_cds_parquet=False,
        generate_cosmos_predict_dataset=False,
    )
    config = json.loads(config_path.read_text())
    config["expected_output"]["min_passed_clips"] = 2
    _write_json(config_path, config)
    _write_json(
        output_path / "summary.json",
        {
            "num_processed_videos": 1,
            "total_num_clips_passed": 2,
            "total_num_clips_transcoded": 2,
            "total_num_clips_with_embeddings": 2,
        },
    )
    _write_json(output_path / "processed_videos/video.mp4.json", {})
    _write_json(output_path / "processed_clip_chunks/video.mp4_0.json", {})
    parquet_path = output_path / "ce1_embd_224p_parquet/only.parquet"
    parquet_path.parent.mkdir(parents=True)
    pq.write_table(pa.table({"id": ["clip-id"], "embedding": [[1.0]]}), parquet_path)

    report = validate_split_output(config_path, output_path.as_posix())

    assert not report.passed
    assert any("embedding parquet contains 1 total row(s), expected at least 2" in error for error in report.errors)


def test_validate_split_output_accepts_truncated_nonempty_caption(tmp_path: Path) -> None:
    """Truncated captions are accepted by the producer and should satisfy the minimum."""
    config_path = tmp_path / "scenario.json"
    output_path = tmp_path / "output"
    _write_config(config_path)
    _write_complete_output(output_path)
    stats = json.loads(json.dumps(_CAPTION_STATS))
    stats["caption_status_counts"]["success"] = 0
    stats["caption_status_counts"]["truncated"] = 1
    _write_json(output_path / "caption_quality_stats.json", stats)

    report = validate_split_output(config_path, output_path.as_posix())

    assert report.passed
    assert "caption quality: 1/1 window(s) accepted" in report.checks


@pytest.mark.parametrize(
    ("counter", "expected_error"),
    [
        ("empty_caption_count", "1 empty accepted caption"),
        ("sentinel_caption_count", "1 sentinel accepted caption"),
    ],
)
def test_validate_split_output_rejects_invalid_accepted_caption(
    tmp_path: Path,
    counter: str,
    expected_error: str,
) -> None:
    """Accepted captions must still contain usable generated text."""
    config_path = tmp_path / "scenario.json"
    output_path = tmp_path / "output"
    _write_config(config_path)
    _write_complete_output(output_path)
    stats = dict(_CAPTION_STATS)
    stats[counter] = 1
    _write_json(output_path / "caption_quality_stats.json", stats)

    report = validate_split_output(config_path, output_path.as_posix())

    assert not report.passed
    assert any(expected_error in error for error in report.errors)


def test_validate_split_output_rejects_caption_error_outcome(tmp_path: Path) -> None:
    """Caption errors should fail validation even when enough accepted captions exist."""
    config_path = tmp_path / "scenario.json"
    output_path = tmp_path / "output"
    _write_config(config_path)
    _write_complete_output(output_path)
    stats = json.loads(json.dumps(_CAPTION_STATS))
    stats["caption_windows_checked"] = 2
    stats["caption_status_counts"]["error"] = 1
    _write_json(output_path / "caption_quality_stats.json", stats)

    report = validate_split_output(config_path, output_path.as_posix())

    assert not report.passed
    assert any("1 error window" in error for error in report.errors)


def test_validate_split_output_does_not_require_caption_stats_for_multi_cam(tmp_path: Path) -> None:
    """Multi-camera runs do not emit the single-camera caption quality artifact."""
    config_path = tmp_path / "scenario.json"
    output_path = tmp_path / "output"
    _write_config(config_path, multi_cam=True)
    _write_complete_output(output_path)
    (output_path / "caption_quality_stats.json").unlink()

    report = validate_split_output(config_path, output_path.as_posix())

    assert report.passed


def test_validate_split_output_does_not_require_disabled_caption_stats(tmp_path: Path) -> None:
    """Captioning runs may explicitly disable the optional quality artifact."""
    config_path = tmp_path / "scenario.json"
    output_path = tmp_path / "output"
    _write_config(config_path, caption_quality_stats_enabled=False)
    _write_complete_output(output_path)
    (output_path / "caption_quality_stats.json").unlink()

    report = validate_split_output(config_path, output_path.as_posix())

    assert report.passed


def test_validate_split_output_ignores_disabled_optional_artifacts(tmp_path: Path) -> None:
    """A clip-only scenario should not require caption, embedding, or dataset outputs."""
    config_path = tmp_path / "scenario.json"
    output_path = tmp_path / "output"
    _write_config(
        config_path,
        generate_embeddings=False,
        generate_captions=False,
        generate_previews=False,
        upload_cds_parquet=False,
        generate_cosmos_predict_dataset=False,
    )
    config = json.loads(config_path.read_text())
    del config["args"]["embedding_algorithm"]
    _write_json(config_path, config)
    _write_json(
        output_path / "summary.json",
        {
            "num_processed_videos": 1,
            "total_num_clips_passed": 1,
            "total_num_clips_transcoded": 1,
        },
    )
    _write_json(output_path / "processed_videos/video.mp4.json", {})
    _write_json(output_path / "processed_clip_chunks/video.mp4_0.json", {})
    (output_path / "clips").mkdir(parents=True)
    (output_path / "clips/clip-id.mp4").write_bytes(b"clip")

    report = validate_split_output(config_path, output_path.as_posix())

    assert report.passed


def test_validate_split_output_uses_remote_profile_client_and_nested_keys(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Remote reads should preserve nested relative keys and reuse the selected client."""
    config_path = tmp_path / "scenario.json"
    _write_config(
        config_path,
        output_s3_profile_name="ci-profile",
        generate_captions=False,
        generate_previews=False,
        upload_clips=False,
        upload_cds_parquet=False,
        generate_cosmos_predict_dataset=False,
    )
    output_path = "s3://bucket/pipeline/jobs/123/scenario"
    client = Mock()
    get_client = Mock(return_value=client)
    listed_clients: list[object] = []

    def list_files(path: str, selected_client: object) -> list[str]:
        listed_clients.append(selected_client)
        if path.endswith("/processed_videos"):
            return ["nested/video.mp4.json"]
        if path.endswith("/processed_clip_chunks"):
            return ["nested/video.mp4_0.json"]
        if path.endswith("/video_errors"):
            return []
        if path.endswith("/ce1_embd_224p_parquet"):
            return ["nested/shard.parquet"]
        msg = f"Unexpected list path: {path}"
        raise AssertionError(msg)

    def read_json(path: object, selected_client: object) -> object:
        assert selected_client is client
        assert str(path) == f"{output_path}/summary.json"
        return {
            "num_processed_videos": 1,
            "total_num_clips_passed": 1,
            "total_num_clips_transcoded": 1,
            "total_num_clips_with_embeddings": 1,
        }

    sink = pa.BufferOutputStream()
    pq.write_table(pa.table({"id": ["clip-id"], "embedding": [[1.0]]}), sink)

    def read_bytes(path: object, selected_client: object) -> bytes:
        assert selected_client is client
        assert str(path) == f"{output_path}/ce1_embd_224p_parquet/nested/shard.parquet"
        return sink.getvalue().to_pybytes()

    monkeypatch.setattr(storage_utils, "get_storage_client", get_client)
    monkeypatch.setattr(storage_utils, "get_files_relative", list_files)
    monkeypatch.setattr(storage_utils, "read_json_file", read_json)
    monkeypatch.setattr(storage_utils, "read_bytes", read_bytes)

    report = validate_split_output(config_path, output_path)

    assert report.passed
    get_client.assert_called_once_with(output_path, profile_name="ci-profile")
    assert listed_clients
    assert all(selected_client is client for selected_client in listed_clients)


def test_validate_split_output_reports_remote_listing_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A remote listing failure should be reported without replacing the shared client."""
    config_path = tmp_path / "scenario.json"
    _write_config(
        config_path,
        generate_embeddings=False,
        generate_captions=False,
        generate_previews=False,
        upload_clips=False,
        upload_cds_parquet=False,
        generate_cosmos_predict_dataset=False,
    )
    output_path = "s3://bucket/pipeline/jobs/456/scenario"
    client = Mock()
    used_clients: list[object] = []

    def list_files(path: str, selected_client: object) -> list[str]:
        used_clients.append(selected_client)
        if path.endswith("/video_errors"):
            msg = "list denied"
            raise RuntimeError(msg)
        return ["nested/output.json"]

    monkeypatch.setattr(storage_utils, "get_storage_client", Mock(return_value=client))
    monkeypatch.setattr(storage_utils, "get_files_relative", list_files)
    monkeypatch.setattr(
        storage_utils,
        "read_json_file",
        Mock(
            return_value={
                "num_processed_videos": 1,
                "total_num_clips_passed": 1,
                "total_num_clips_transcoded": 1,
            }
        ),
    )

    report = validate_split_output(config_path, output_path)

    assert not report.passed
    assert any("Failed to list video error files" in error and "list denied" in error for error in report.errors)
    assert used_clients
    assert all(selected_client is client for selected_client in used_clients)


def test_k8s_gpu_outputs_are_attempt_unique() -> None:
    """Both validated k8s jobs must isolate retries with the GitLab job ID."""
    for script_name in ("k8s_gpu_tests.sh", "k8s_gpu_extended.sh"):
        script = (_REPO_ROOT / ".gitlab/scripts" / script_name).read_text()
        assert ': "${CI_JOB_ID:?CI_JOB_ID must be set}"' in script
        assert "${S3_OUTPUT_PATH}/jobs/${CI_JOB_ID}/" in script


@pytest.mark.parametrize("config_path", _CI_CONFIGS, ids=lambda path: path.name)
def test_ci_scenarios_define_positive_expected_output(config_path: Path) -> None:
    """Every k8s GPU scenario should declare a non-empty expected-output contract."""
    config = json.loads(config_path.read_text())

    expected = config["expected_output"]
    assert expected["min_processed_videos"] >= 1
    assert expected["min_passed_clips"] >= 1
    assert {
        "output_s3_profile_name",
        "generate_embeddings",
        "generate_captions",
        "caption_quality_stats_enabled",
        "generate_previews",
        "upload_clips",
        "upload_cds_parquet",
        "generate_cosmos_predict_dataset",
    } <= config["args"].keys()
    if config["args"]["generate_embeddings"]:
        assert config["args"]["embedding_algorithm"]
    if not config["args"]["generate_captions"]:
        assert not config["args"]["caption_quality_stats_enabled"]
    loaded_pipeline_args = load_pipeline_config(config_path)
    assert "expected_output" not in loaded_pipeline_args
    assert loaded_pipeline_args["_pipeline"] == "split"


def test_output_validation_cli_returns_failure(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The CLI should return nonzero and explain a missing summary."""
    config_path = tmp_path / "scenario.json"
    _write_config(config_path)

    exit_code = main([config_path.as_posix(), (tmp_path / "missing").as_posix()])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert "summary.json is missing or invalid" in captured.err
