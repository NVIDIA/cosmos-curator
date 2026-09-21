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
"""Validate split-pipeline outputs against a CI scenario config."""

import argparse
import io
import pathlib
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import yaml
from pyarrow import parquet as pq

from cosmos_curator.core.utils.storage import storage_utils
from cosmos_curator.core.utils.storage.storage_client import StorageClient, StoragePrefix
from cosmos_curator.pipelines.video.read_write.metadata_writer_stage import ClipWriterStage

_EXPECTED_OUTPUT_KEY = "expected_output"
_EXPECTED_OUTPUT_FIELDS = frozenset({"min_processed_videos", "min_passed_clips"})


@dataclass(frozen=True)
class _ScenarioOptions:
    output_s3_profile_name: str
    upload_clips: bool
    generate_embeddings: bool
    embedding_algorithm: str | None
    generate_captions: bool
    caption_quality_stats_enabled: bool
    multi_cam: bool
    generate_previews: bool
    upload_cds_parquet: bool
    generate_cosmos_predict_dataset: bool


@dataclass(frozen=True)
class ExpectedOutput:
    """Minimum successful work expected from one CI scenario."""

    min_processed_videos: int
    min_passed_clips: int


@dataclass(frozen=True)
class ValidationReport:
    """Output checks performed and any failures found."""

    checks: tuple[str, ...]
    errors: tuple[str, ...]

    @property
    def passed(self) -> bool:
        """Return whether every expected-output check passed."""
        return not self.errors


@dataclass
class _ValidationState:
    checks: list[str]
    errors: list[str]


@dataclass(frozen=True)
class _FileExpectation:
    path: str
    suffix: str
    description: str
    minimum: int


@dataclass(frozen=True)
class _ParquetExpectation:
    root: str
    files: tuple[str, ...]
    required_columns: frozenset[str]
    description: str
    minimum_rows: int


def validate_split_output(
    config_path: str | pathlib.Path,
    output_path: str,
    *,
    profile_name: str | None = None,
) -> ValidationReport:
    """Validate one split-pipeline output using its scenario config."""
    args, expected = _load_scenario(config_path)
    effective_profile = profile_name or args.output_s3_profile_name
    client = storage_utils.get_storage_client(output_path, profile_name=effective_profile)
    state = _ValidationState(checks=[], errors=[])

    summary = _read_summary(output_path, client, state)
    if summary is None:
        return ValidationReport(tuple(state.checks), tuple(state.errors))

    _check_minimum(summary, "num_processed_videos", expected.min_processed_videos, state)
    _check_minimum(summary, "total_num_clips_passed", expected.min_passed_clips, state)
    _check_minimum(summary, "total_num_clips_transcoded", expected.min_passed_clips, state)

    _check_files(
        _FileExpectation(
            path=ClipWriterStage.get_output_path_processed_videos(output_path),
            suffix=".json",
            description="processed-video metadata",
            minimum=expected.min_processed_videos,
        ),
        client,
        state,
    )
    _check_files(
        _FileExpectation(
            path=ClipWriterStage.get_output_path_processed_clip_chunks(output_path),
            suffix=".json",
            description="processed clip-chunk metadata",
            minimum=expected.min_processed_videos,
        ),
        client,
        state,
    )
    _check_no_video_errors(output_path, client, state)

    if args.upload_clips:
        _check_files(
            _FileExpectation(
                path=ClipWriterStage.get_output_path_clips(output_path),
                suffix=".mp4",
                description="uploaded clips",
                minimum=expected.min_passed_clips,
            ),
            client,
            state,
        )
    if args.generate_embeddings:
        assert args.embedding_algorithm is not None
        _check_minimum(summary, "total_num_clips_with_embeddings", expected.min_passed_clips, state)
        embedding_path = ClipWriterStage.get_output_path_embd_parquets(output_path, args.embedding_algorithm)
        embedding_files = _check_files(
            _FileExpectation(
                path=embedding_path,
                suffix=".parquet",
                description=f"{args.embedding_algorithm} embedding parquet",
                minimum=1,
            ),
            client,
            state,
        )
        _check_parquet(
            _ParquetExpectation(
                root=embedding_path,
                files=tuple(embedding_files),
                required_columns=frozenset({"id", "embedding"}),
                description="embedding parquet",
                minimum_rows=expected.min_passed_clips,
            ),
            client,
            state,
        )
    if args.generate_captions:
        _check_minimum(summary, "total_num_clips_with_caption", expected.min_passed_clips, state)
        _check_minimum(summary, "total_num_caption_windows", 1, state)
        if args.caption_quality_stats_enabled and not args.multi_cam:
            _check_caption_quality_stats(output_path, expected.min_passed_clips, client, state)
    if args.generate_previews:
        _check_minimum(summary, "total_num_clips_with_webp", expected.min_passed_clips, state)
        _check_files(
            _FileExpectation(
                path=ClipWriterStage.get_output_path_previews(output_path),
                suffix=".webp",
                description="preview images",
                minimum=expected.min_passed_clips,
            ),
            client,
            state,
        )
    if args.upload_cds_parquet:
        cds_path = ClipWriterStage.get_output_path_cds_parquets(output_path)
        cds_files = _check_files(
            _FileExpectation(
                path=cds_path,
                suffix=".parquet",
                description="CDS parquet",
                minimum=1,
            ),
            client,
            state,
        )
        _check_parquet(
            _ParquetExpectation(
                root=cds_path,
                files=tuple(cds_files),
                required_columns=frozenset({"id", "embedding", "$meta"}),
                description="CDS parquet",
                minimum_rows=expected.min_passed_clips,
            ),
            client,
            state,
        )
    if args.generate_cosmos_predict_dataset:
        _check_cosmos_predict_dataset(output_path, expected.min_passed_clips, client, state)

    return ValidationReport(tuple(state.checks), tuple(state.errors))


def _load_scenario(config_path: str | pathlib.Path) -> tuple[_ScenarioOptions, ExpectedOutput]:
    path = pathlib.Path(config_path)
    try:
        document = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError) as exc:
        msg = f"Failed to load pipeline config {path}: {exc}"
        raise ValueError(msg) from exc
    if not isinstance(document, Mapping):
        msg = f"Pipeline config must contain a mapping: {path}"
        raise TypeError(msg)
    if document.get("pipeline") != "split":
        msg = f"Output validation only supports split pipeline configs: {path}"
        raise ValueError(msg)

    raw_args = document.get("args")
    if not isinstance(raw_args, Mapping):
        msg = f"Pipeline config must contain an args mapping: {path}"
        raise TypeError(msg)
    generate_embeddings = _required_bool_arg(raw_args, "generate_embeddings", path)
    args = _ScenarioOptions(
        output_s3_profile_name=_required_string_arg(raw_args, "output_s3_profile_name", path),
        upload_clips=_required_bool_arg(raw_args, "upload_clips", path),
        generate_embeddings=generate_embeddings,
        embedding_algorithm=(
            _required_string_arg(raw_args, "embedding_algorithm", path) if generate_embeddings else None
        ),
        generate_captions=_required_bool_arg(raw_args, "generate_captions", path),
        caption_quality_stats_enabled=_required_bool_arg(raw_args, "caption_quality_stats_enabled", path),
        multi_cam=_bool_arg(raw_args, "multi_cam", path, default=False),
        generate_previews=_required_bool_arg(raw_args, "generate_previews", path),
        upload_cds_parquet=_required_bool_arg(raw_args, "upload_cds_parquet", path),
        generate_cosmos_predict_dataset=_required_bool_arg(raw_args, "generate_cosmos_predict_dataset", path),
    )
    if args.upload_cds_parquet and not args.generate_embeddings:
        msg = f"args.upload_cds_parquet requires args.generate_embeddings in {path}"
        raise ValueError(msg)

    raw_expected = document.get(_EXPECTED_OUTPUT_KEY)
    if not isinstance(raw_expected, Mapping):
        msg = f"Pipeline config must define an {_EXPECTED_OUTPUT_KEY} mapping: {path}"
        raise TypeError(msg)
    unknown = sorted(set(raw_expected) - _EXPECTED_OUTPUT_FIELDS)
    if unknown:
        msg = f"Unknown {_EXPECTED_OUTPUT_KEY} fields in {path}: {unknown}"
        raise ValueError(msg)

    return args, ExpectedOutput(
        min_processed_videos=_positive_int(raw_expected.get("min_processed_videos"), "min_processed_videos", path),
        min_passed_clips=_positive_int(raw_expected.get("min_passed_clips"), "min_passed_clips", path),
    )


def _required_bool_arg(args: Mapping[object, object], field_name: str, config_path: pathlib.Path) -> bool:
    value = args.get(field_name)
    if not isinstance(value, bool):
        msg = f"args.{field_name} must be a boolean in {config_path}"
        raise TypeError(msg)
    return value


def _bool_arg(
    args: Mapping[object, object],
    field_name: str,
    config_path: pathlib.Path,
    *,
    default: bool,
) -> bool:
    value = args.get(field_name, default)
    if not isinstance(value, bool):
        msg = f"args.{field_name} must be a boolean in {config_path}"
        raise TypeError(msg)
    return value


def _required_string_arg(args: Mapping[object, object], field_name: str, config_path: pathlib.Path) -> str:
    value = args.get(field_name)
    if not isinstance(value, str) or not value:
        msg = f"args.{field_name} must be a non-empty string in {config_path}"
        raise TypeError(msg)
    return value


def _positive_int(value: object, field_name: str, config_path: pathlib.Path) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        msg = f"{_EXPECTED_OUTPUT_KEY}.{field_name} must be a positive integer in {config_path}"
        raise ValueError(msg)
    return value


def _read_summary(
    output_path: str,
    client: StorageClient | None,
    state: _ValidationState,
) -> Mapping[str, Any] | None:
    summary_path = storage_utils.get_full_path(output_path, "summary.json")
    try:
        summary: object = storage_utils.read_json_file(summary_path, client)
    except Exception as exc:  # noqa: BLE001
        state.errors.append(f"summary.json is missing or invalid at {summary_path}: {exc}")
        return None
    if not isinstance(summary, Mapping):
        state.errors.append(f"summary.json must contain a mapping at {summary_path}")
        return None
    return summary


def _check_minimum(
    summary: Mapping[str, Any],
    field_name: str,
    minimum: int,
    state: _ValidationState,
) -> None:
    value = summary.get(field_name)
    if not isinstance(value, int) or isinstance(value, bool):
        state.errors.append(f"summary.json field {field_name!r} must be an integer, got {value!r}")
        return
    if value < minimum:
        state.errors.append(f"summary.json field {field_name!r} is {value}, expected at least {minimum}")
        return
    state.checks.append(f"{field_name}={value}")


def _check_files(
    expectation: _FileExpectation,
    client: StorageClient | None,
    state: _ValidationState,
) -> list[str]:
    try:
        files = [
            name
            for name in storage_utils.get_files_relative(expectation.path, client)
            if name.endswith(expectation.suffix)
        ]
    except Exception as exc:  # noqa: BLE001
        state.errors.append(f"Failed to list {expectation.description} at {expectation.path}: {exc}")
        return []
    if len(files) < expectation.minimum:
        state.errors.append(
            f"Found {len(files)} {expectation.description} file(s) at {expectation.path}, "
            f"expected at least {expectation.minimum} ({expectation.suffix})"
        )
    else:
        state.checks.append(f"{expectation.description}: {len(files)} file(s)")
    return files


def _check_parquet(
    expectation: _ParquetExpectation,
    client: StorageClient | None,
    state: _ValidationState,
) -> None:
    if not expectation.files:
        return
    total_rows = 0
    valid = True
    for relative_path in expectation.files:
        path = storage_utils.get_full_path(expectation.root, relative_path)
        try:
            metadata = pq.read_metadata(io.BytesIO(storage_utils.read_bytes(path, client)))
        except Exception as exc:  # noqa: BLE001
            state.errors.append(f"Failed to read {expectation.description} file {path}: {exc}")
            valid = False
            continue
        missing_columns = sorted(expectation.required_columns - set(metadata.schema.to_arrow_schema().names))
        if missing_columns:
            state.errors.append(f"{expectation.description} file {path} is missing columns: {missing_columns}")
            valid = False
        total_rows += metadata.num_rows

    if total_rows < expectation.minimum_rows:
        state.errors.append(
            f"{expectation.description} contains {total_rows} total row(s), expected at least "
            f"{expectation.minimum_rows}"
        )
        valid = False
    if valid:
        state.checks.append(f"{expectation.description}: {len(expectation.files)} file(s), {total_rows} total row(s)")


def _check_no_video_errors(
    output_path: str,
    client: StorageClient | None,
    state: _ValidationState,
) -> None:
    path = ClipWriterStage.get_output_path_video_errors(output_path)
    error_files: list[str]
    try:
        error_files = storage_utils.get_files_relative(path, client)
    except Exception as exc:  # noqa: BLE001
        state.errors.append(f"Failed to list video error files at {path}: {exc}")
        return
    if error_files:
        state.errors.append(f"Pipeline wrote {len(error_files)} video error file(s) at {path}")
    else:
        state.checks.append("no video error files")


def _check_caption_quality_stats(
    output_path: str,
    minimum_accepted: int,
    client: StorageClient | None,
    state: _ValidationState,
) -> None:
    errors_before = len(state.errors)
    path = storage_utils.get_full_path(output_path, "caption_quality_stats.json")
    counters = _read_caption_counters(path, client, state)
    if counters is None:
        return

    status_total = sum(counters[name] for name in ("success", "truncated", "blocked", "error", "skipped"))
    if status_total != counters["caption_windows_checked"]:
        state.errors.append(
            f"Caption status counts total {status_total}, but caption_windows_checked is "
            f"{counters['caption_windows_checked']} at {path}"
        )
    accepted = counters["success"] + counters["truncated"]
    if accepted < minimum_accepted:
        state.errors.append(
            f"Caption quality stats contain {accepted} accepted window(s), expected at least "
            f"{minimum_accepted} at {path}"
        )
    if counters["error"]:
        state.errors.append(f"Caption quality stats contain {counters['error']} error window(s) at {path}")
    if counters["empty_caption_count"]:
        state.errors.append(
            f"Caption quality stats contain {counters['empty_caption_count']} empty accepted caption(s) at {path}"
        )
    if counters["sentinel_caption_count"]:
        state.errors.append(
            f"Caption quality stats contain {counters['sentinel_caption_count']} sentinel accepted caption(s) at {path}"
        )
    if len(state.errors) == errors_before:
        state.checks.append(f"caption quality: {accepted}/{counters['caption_windows_checked']} window(s) accepted")


def _read_caption_counters(
    path: pathlib.Path | StoragePrefix,
    client: StorageClient | None,
    state: _ValidationState,
) -> dict[str, int] | None:
    try:
        stats: object = storage_utils.read_json_file(path, client)
    except Exception as exc:  # noqa: BLE001
        state.errors.append(f"Failed to read caption quality stats from {path}: {exc}")
        return None
    if not isinstance(stats, Mapping):
        state.errors.append(f"Caption quality stats must contain a mapping at {path}")
        return None
    if stats.get("schema_version") != 1 or stats.get("pipeline") != "split_video_pipeline":
        state.errors.append(f"Caption quality stats have an unsupported schema at {path}")
        return None

    counts = stats.get("caption_status_counts")
    if not isinstance(counts, Mapping):
        state.errors.append(f"caption_status_counts must contain a mapping at {path}")
        return None
    fields = {
        "caption_windows_checked": stats.get("caption_windows_checked"),
        "success": counts.get("success"),
        "truncated": counts.get("truncated"),
        "blocked": counts.get("blocked"),
        "error": counts.get("error"),
        "skipped": counts.get("skipped"),
        "empty_caption_count": stats.get("empty_caption_count"),
        "sentinel_caption_count": stats.get("sentinel_caption_count"),
    }
    invalid = {name: value for name, value in fields.items() if not _is_nonnegative_int(value)}
    if invalid:
        state.errors.append(f"Caption quality stats contain invalid counters at {path}: {invalid}")
        return None

    counters: dict[str, int] = {}
    for name, value in fields.items():
        assert isinstance(value, int)
        assert not isinstance(value, bool)
        counters[name] = value
    return counters


def _is_nonnegative_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _check_cosmos_predict_dataset(
    output_path: str,
    minimum_files: int,
    client: StorageClient | None,
    state: _ValidationState,
) -> None:
    expected_files = (
        (
            ClipWriterStage.get_output_path_per_window_clips(output_path),
            ".mp4",
            "Cosmos-Predict videos",
        ),
        (
            ClipWriterStage.get_output_path_per_window_metas(output_path),
            ".txt",
            "Cosmos-Predict captions",
        ),
        (
            ClipWriterStage.get_output_path_per_window_t5_embeds(output_path),
            ".pickle",
            "Cosmos-Predict T5 embeddings",
        ),
    )
    for path, suffix, description in expected_files:
        _check_files(
            _FileExpectation(path=path, suffix=suffix, description=description, minimum=minimum_files),
            client,
            state,
        )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config_path", help="Split-pipeline scenario JSON or YAML config.")
    parser.add_argument("output_path", help="Output root produced by the scenario.")
    parser.add_argument("--profile-name", default=None, help="Override the config's output storage profile.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run expected-output validation from the command line."""
    args = _parse_args(argv)
    try:
        report = validate_split_output(args.config_path, args.output_path, profile_name=args.profile_name)
    except (OSError, TypeError, ValueError) as exc:
        sys.stderr.write(f"ERROR: {exc}\n")
        return 2

    for check in report.checks:
        sys.stdout.write(f"PASS: {check}\n")
    for error in report.errors:
        sys.stderr.write(f"ERROR: {error}\n")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
