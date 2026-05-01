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
"""Tests for shard CLI ↔ settings alignment and attrs validation."""

import argparse

import attrs
import pytest

from cosmos_curator.pipelines.common_pipeline_settings import (
    CommonPipelineSettings,
    composite_from_namespace,
    composite_to_namespace,
    sync_common_from_namespace,
)
from cosmos_curator.pipelines.video.shard_pipeline_settings import ShardPipelineSettings
from cosmos_curator.pipelines.video.sharding_pipeline import _setup_parser


def _shard_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    _setup_parser(parser)
    return parser


def _action_for_dest(parser: argparse.ArgumentParser, dest: str) -> argparse.Action | None:
    """Return the user-registered :class:`~argparse.Action` for *dest* (first match)."""
    for action in parser._actions:
        if action.dest == dest:
            return action
    return None


def _dests_from_parser(parser: argparse.ArgumentParser) -> set[str]:
    """User-facing ``dest`` names (excludes built-in ``help``)."""
    skip = frozenset({"help"})
    out: set[str] = set()
    for action in parser._actions:
        dest = action.dest
        if dest is None or dest is argparse.SUPPRESS or dest in skip:
            continue
        out.add(dest)
    return out


def _expected_settings_dests() -> set[str]:
    """All attribute names that must exist on ``argparse.Namespace`` after shard parse."""
    common = {f.name for f in attrs.fields(CommonPipelineSettings)}
    shard_only = {f.name for f in attrs.fields(ShardPipelineSettings) if f.name != "common"}
    return common | shard_only


def _shard_settings_from_args(args: argparse.Namespace) -> ShardPipelineSettings:
    """Build :class:`ShardPipelineSettings` like :func:`shard` in ``sharding_pipeline``."""
    return composite_from_namespace(ShardPipelineSettings, args)


def test_shard_parser_add_argument_type_matches_field_scalar_type() -> None:
    """``add_argument(..., type=...)`` matches attrs field type for inferred scalar flags."""
    parser = _shard_parser()
    expectations: list[tuple[str, type]] = [
        # CommonPipelineSettings — inferred from field annotations (no explicit cli arg_type).
        ("input_s3_profile_name", str),
        ("output_s3_profile_name", str),
        ("execution_mode", str),
        ("limit", int),
        ("model_weights_path", str),
        ("profile_cpu_exclude", str),
        ("profile_memory_exclude", str),
        ("profile_gpu_exclude", str),
        # ShardPipelineSettings — same pattern.
        ("input_clip_path", str),
        ("output_dataset_path", str),
        ("captioning_algorithm", str),
        ("annotation_version", str),
        ("input_semantic_dedup_s3_profile_name", str),
        ("semantic_dedup_epsilon", float),
        ("max_tars_per_part", int),
        ("target_tar_size_mb", int),
        ("min_clips_per_tar", int),
        ("input_semantic_dedup_path", str),
    ]
    for dest, expected_type in expectations:
        action = _action_for_dest(parser, dest)
        assert action is not None, f"no action for dest={dest!r}"
        assert action.type is expected_type, f"{dest}: expected action.type {expected_type}, got {action.type!r}"


def test_shard_parser_boolean_flags_use_action_not_type() -> None:
    """Boolean flags use ``action=``; argparse must not apply a scalar ``type=`` converter."""
    parser = _shard_parser()
    for dest in (
        "verbose",
        "perf_profile",
        "profile_tracing",
        "profile_cpu",
        "profile_memory",
        "profile_gpu",
        "drop_small_shards",
    ):
        action = _action_for_dest(parser, dest)
        assert action is not None
        assert action.type is None, f"{dest}: expected no type= converter, got {action.type!r}"


def test_shard_parser_parse_coerces_inferred_numeric_flags() -> None:
    """End-to-end: inferred ``int`` / ``float`` types coerce argv values on parse."""
    parser = _shard_parser()
    args = parser.parse_args(
        [
            "--input-clip-path",
            "/in",
            "--output-dataset-path",
            "/out",
            "--limit",
            "9",
            "--semantic-dedup-epsilon",
            "0.5",
            "--max-tars-per-part",
            "3",
        ],
    )
    assert args.limit == 9
    assert type(args.limit) is int
    assert args.semantic_dedup_epsilon == 0.5
    assert type(args.semantic_dedup_epsilon) is float
    assert args.max_tars_per_part == 3
    assert type(args.max_tars_per_part) is int


def test_shard_parser_dests_match_settings_fields() -> None:
    """Every CLI dest maps to a settings field and vice versa (no drift)."""
    parser = _shard_parser()
    parser_dests = _dests_from_parser(parser)
    expected = _expected_settings_dests()
    assert parser_dests == expected, (
        f"Parser/settings mismatch.\n"
        f"Only on parser: {sorted(parser_dests - expected)}\n"
        f"Only on settings: {sorted(expected - parser_dests)}"
    )


def test_shard_settings_from_minimal_valid_parse() -> None:
    """Minimal argv produces valid :class:`ShardPipelineSettings` with expected paths."""
    parser = _shard_parser()
    args = parser.parse_args(
        [
            "--input-clip-path",
            "/data/clips",
            "--output-dataset-path",
            "/data/out",
        ],
    )
    settings = _shard_settings_from_args(args)
    assert settings.input_clip_path == "/data/clips"
    assert settings.output_dataset_path == "/data/out"
    assert settings.common.verbose is False
    assert settings.common.execution_mode == "AUTO"
    assert settings.captioning_algorithm == "qwen"
    assert settings.max_tars_per_part >= 1


def test_sync_common_from_namespace_updates_common_after_ns_mutation() -> None:
    """``sync_common_from_namespace`` copies common fields from the flat namespace into ``settings.common``."""
    parser = _shard_parser()
    args = parser.parse_args(
        [
            "--input-clip-path",
            "/in",
            "--output-dataset-path",
            "/out",
            "--no-perf-profile",
        ],
    )
    settings = composite_from_namespace(ShardPipelineSettings, args)
    assert settings.common.perf_profile is False
    ns = composite_to_namespace(settings)
    ns.perf_profile = True
    sync_common_from_namespace(settings, ns)
    assert settings.common.perf_profile is True


def test_composite_to_namespace_matches_common_and_shard_fields() -> None:
    """Flattened namespace exposes common dests and shard-only dests at top level."""
    parser = _shard_parser()
    args = parser.parse_args(
        [
            "--input-clip-path",
            "/data/clips",
            "--output-dataset-path",
            "/data/out",
            "--limit",
            "5",
        ],
    )
    settings = composite_from_namespace(ShardPipelineSettings, args)
    ns = composite_to_namespace(settings)
    assert ns.limit == settings.common.limit == 5
    assert ns.input_clip_path == "/data/clips"
    assert ns.perf_profile == settings.common.perf_profile


@pytest.mark.parametrize(
    ("extra_argv", "expected_exc", "match_substr"),
    [
        (
            ["--max-tars-per-part", "0"],
            ValueError,
            "max_tars_per_part",
        ),
        (
            ["--semantic-dedup-epsilon", "-0.01"],
            ValueError,
            "semantic_dedup_epsilon",
        ),
        (
            ["--limit", "-1"],
            ValueError,
            "limit",
        ),
    ],
)
def test_shard_settings_validation_rejects_invalid_values(
    extra_argv: list[str],
    expected_exc: type[BaseException],
    match_substr: str,
) -> None:
    """Values argparse accepts but attrs validators reject fail at settings construction."""
    parser = _shard_parser()
    base = [
        "--input-clip-path",
        "/in",
        "--output-dataset-path",
        "/out",
    ]
    args = parser.parse_args(base + extra_argv)
    with pytest.raises(expected_exc, match=match_substr):
        _shard_settings_from_args(args)


def test_common_settings_rejects_invalid_execution_mode() -> None:
    """``execution_mode`` must be AUTO, BATCH, or STREAMING."""
    ns = argparse.Namespace(
        input_s3_profile_name="default",
        output_s3_profile_name="default",
        execution_mode="INVALID",
        limit=0,
        verbose=False,
        model_weights_path="/weights",
        perf_profile=True,
        profile_tracing=False,
        profile_tracing_sampling=0.01,
        profile_tracing_otlp_endpoint="",
        profile_cpu=False,
        profile_memory=False,
        profile_gpu=False,
        profile_cpu_exclude="_root",
        profile_memory_exclude="_root",
        profile_gpu_exclude="_root",
    )
    with pytest.raises(ValueError, match="execution_mode"):
        CommonPipelineSettings.from_namespace(ns)


def test_shard_settings_rejects_empty_input_clip_path() -> None:
    """``min_len(1)`` rejects empty string if it reaches settings (manual namespace)."""
    ns = argparse.Namespace(
        input_s3_profile_name="default",
        output_s3_profile_name="default",
        execution_mode="BATCH",
        limit=0,
        verbose=False,
        model_weights_path="s3://w/",
        perf_profile=True,
        profile_tracing=False,
        profile_tracing_sampling=0.01,
        profile_tracing_otlp_endpoint="",
        profile_cpu=False,
        profile_memory=False,
        profile_gpu=False,
        profile_cpu_exclude="_root",
        profile_memory_exclude="_root",
        profile_gpu_exclude="_root",
        input_clip_path="",
        output_dataset_path="/out",
        captioning_algorithm="qwen",
        annotation_version="v0",
        input_semantic_dedup_s3_profile_name="default",
        semantic_dedup_epsilon=0.01,
        max_tars_per_part=10,
        target_tar_size_mb=500,
        min_clips_per_tar=1,
        drop_small_shards=False,
        input_semantic_dedup_path=None,
    )
    with pytest.raises(ValueError, match="input_clip_path"):
        _shard_settings_from_args(ns)
