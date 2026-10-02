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
"""Tests for mp4-index config loading and resolution."""

import json
import pathlib

import pydantic
import pytest
import yaml

from cosmos_curator.next.recipes.mp4_index import config

_MINIMAL_URIS_CONFIG: dict[str, object] = {
    "schema_version": 1,
    "kind": "mp4-index",
    "input": {"uris": ["s3://bucket/a.mp4"], "container_kind": "nonfragmented"},
    "output": {"path": "s3://bucket/mp4-index"},
}

_MINIMAL_PATH_PREFIX_CONFIG: dict[str, object] = {
    "schema_version": 1,
    "kind": "mp4-index",
    "input": {
        "path_prefix": "s3://bucket/raw",
        "globs": ["unit-*/*.mp4"],
        "container_kind": "fragmented",
    },
    "output": {"path": "s3://bucket/mp4-index"},
}


def _write(tmp_path: pathlib.Path, payload: dict[str, object], name: str = "config.yaml") -> pathlib.Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(payload))
    return path


def test_resolves_a_minimal_uris_config(tmp_path: pathlib.Path) -> None:
    """The smallest valid uris-mode config resolves, with execution fully defaulted."""
    path = _write(tmp_path, _MINIMAL_URIS_CONFIG)

    resolved = config.resolve_config(path)

    assert resolved.schema_version == 1
    assert resolved.kind == "mp4-index"
    assert resolved.input.uris == ("s3://bucket/a.mp4",)
    assert resolved.input.path_prefix is None
    assert resolved.input.container_kind == "nonfragmented"
    assert resolved.execution.session_id_path_depth == 2
    assert resolved.execution.storage_profile == "default"
    assert resolved.execution.min_actors == 2
    assert resolved.execution.max_actors is None
    assert resolved.execution.actor_memory_gb is None
    assert resolved.execution.skip_known_failures is True
    assert resolved.execution.videos_per_fragment == 1_000
    assert resolved.execution.progress is False
    assert resolved.execution.skip_existing_sidecars is True
    assert resolved.execution.chunk_bytes == 64 * 1024 * 1024
    assert resolved.execution.no_progress_timeout_s == 3600.0


def test_resolves_a_minimal_path_prefix_config(tmp_path: pathlib.Path) -> None:
    """The smallest valid path_prefix-mode config resolves."""
    path = _write(tmp_path, _MINIMAL_PATH_PREFIX_CONFIG)

    resolved = config.resolve_config(path)

    assert resolved.input.uris is None
    assert resolved.input.path_prefix == "s3://bucket/raw"
    assert resolved.input.globs == ("unit-*/*.mp4",)
    assert resolved.input.exclude_globs is None
    assert resolved.input.limit is None


def test_output_derives_frames_metadata_moov_moof_and_error_locations(tmp_path: pathlib.Path) -> None:
    """All fixed output locations derive from output.path -- not separately configured."""
    path = _write(tmp_path, _MINIMAL_URIS_CONFIG)

    resolved = config.resolve_config(path)

    assert resolved.output.manifest_lance_uri == "s3://bucket/mp4-index/manifest.lance"
    assert resolved.output.frame_index_root == "s3://bucket/mp4-index/frames"
    assert resolved.output.metadata_root == "s3://bucket/mp4-index/metadata"
    assert resolved.output.moov_sidecar_root == "s3://bucket/mp4-index/moov"
    assert resolved.output.moof_table_root == "s3://bucket/mp4-index/moof"
    assert resolved.output.errors_lance_uri == "s3://bucket/mp4-index/errors.lance"


def test_output_derivation_works_for_a_local_path(tmp_path: pathlib.Path) -> None:
    """A local output root doesn't have a URI scheme to preserve, and still joins correctly."""
    payload = {**_MINIMAL_URIS_CONFIG, "output": {"path": "/data/mp4-index"}}
    path = _write(tmp_path, payload)

    resolved = config.resolve_config(path)

    assert resolved.output.manifest_lance_uri == "/data/mp4-index/manifest.lance"
    assert resolved.output.frame_index_root == "/data/mp4-index/frames"
    assert resolved.output.metadata_root == "/data/mp4-index/metadata"
    assert resolved.output.moov_sidecar_root == "/data/mp4-index/moov"
    assert resolved.output.moof_table_root == "/data/mp4-index/moof"
    assert resolved.output.errors_lance_uri == "/data/mp4-index/errors.lance"


def test_limit_override_resolves(tmp_path: pathlib.Path) -> None:
    """input.limit, unset by default, survives resolution once set."""
    payload = {**_MINIMAL_PATH_PREFIX_CONFIG, "input": {**_MINIMAL_PATH_PREFIX_CONFIG["input"], "limit": 5}}  # type: ignore[dict-item]
    path = _write(tmp_path, payload)

    resolved = config.resolve_config(path)

    assert resolved.input.limit == 5


def test_rejects_both_uris_and_path_prefix(tmp_path: pathlib.Path) -> None:
    """Setting both input modes at once is a validation error, not a silent precedence rule."""
    payload = {
        **_MINIMAL_URIS_CONFIG,
        "input": {"uris": ["s3://bucket/a.mp4"], "path_prefix": "s3://bucket/raw", "globs": ["*.mp4"]},
    }
    path = _write(tmp_path, payload)

    with pytest.raises(pydantic.ValidationError):
        config.resolve_config(path)


def test_rejects_neither_uris_nor_path_prefix(tmp_path: pathlib.Path) -> None:
    """Setting neither input mode is a validation error, not an empty-selection default."""
    payload = {**_MINIMAL_URIS_CONFIG, "input": {}}
    path = _write(tmp_path, payload)

    with pytest.raises(pydantic.ValidationError):
        config.resolve_config(path)


def test_rejects_path_prefix_without_globs(tmp_path: pathlib.Path) -> None:
    """path_prefix without globs fails validation rather than matching nothing silently."""
    payload = {**_MINIMAL_URIS_CONFIG, "input": {"path_prefix": "s3://bucket/raw"}}
    path = _write(tmp_path, payload)

    with pytest.raises(pydantic.ValidationError):
        config.resolve_config(path)


def test_rejects_globs_alongside_uris(tmp_path: pathlib.Path) -> None:
    """globs/exclude_globs/limit are path_prefix-only options -- setting one with uris is refused."""
    payload = {
        **_MINIMAL_URIS_CONFIG,
        "input": {**_MINIMAL_URIS_CONFIG["input"], "globs": ["*.mp4"]},  # type: ignore[dict-item]
    }
    path = _write(tmp_path, payload)

    with pytest.raises(pydantic.ValidationError, match="only meaningful with path_prefix"):
        config.resolve_config(path)


def test_rejects_exclude_globs_alongside_uris(tmp_path: pathlib.Path) -> None:
    """Same rule for exclude_globs specifically, not just globs."""
    payload = {
        **_MINIMAL_URIS_CONFIG,
        "input": {**_MINIMAL_URIS_CONFIG["input"], "exclude_globs": ["*.tmp.mp4"]},  # type: ignore[dict-item]
    }
    path = _write(tmp_path, payload)

    with pytest.raises(pydantic.ValidationError, match="only meaningful with path_prefix"):
        config.resolve_config(path)


def test_rejects_limit_alongside_uris(tmp_path: pathlib.Path) -> None:
    """Same rule for limit specifically, not just globs."""
    payload = {**_MINIMAL_URIS_CONFIG, "input": {**_MINIMAL_URIS_CONFIG["input"], "limit": 5}}  # type: ignore[dict-item]
    path = _write(tmp_path, payload)

    with pytest.raises(pydantic.ValidationError, match="only meaningful with path_prefix"):
        config.resolve_config(path)


def test_rejects_a_limit_below_one(tmp_path: pathlib.Path) -> None:
    """Limit must be a positive cap, not zero or negative."""
    payload = {**_MINIMAL_PATH_PREFIX_CONFIG, "input": {**_MINIMAL_PATH_PREFIX_CONFIG["input"], "limit": 0}}  # type: ignore[dict-item]
    path = _write(tmp_path, payload)

    with pytest.raises(pydantic.ValidationError):
        config.resolve_config(path)


def test_rejects_a_missing_container_kind(tmp_path: pathlib.Path) -> None:
    """container_kind is required -- there is no default corpus shape to assume."""
    payload = {**_MINIMAL_URIS_CONFIG, "input": {"uris": ["s3://bucket/a.mp4"]}}
    path = _write(tmp_path, payload)

    with pytest.raises(pydantic.ValidationError):
        config.resolve_config(path)


def test_rejects_an_unknown_container_kind_value(tmp_path: pathlib.Path) -> None:
    """container_kind is one of exactly two literal values, not an arbitrary string."""
    payload = {
        **_MINIMAL_URIS_CONFIG,
        "input": {"uris": ["s3://bucket/a.mp4"], "container_kind": "maybe"},
    }
    path = _write(tmp_path, payload)

    with pytest.raises(pydantic.ValidationError):
        config.resolve_config(path)


def test_session_id_path_depth_override_resolves(tmp_path: pathlib.Path) -> None:
    """A unit layout one directory deeper overrides the depth=2 default."""
    payload = {**_MINIMAL_URIS_CONFIG, "execution": {"session_id_path_depth": 3}}
    path = _write(tmp_path, payload)

    resolved = config.resolve_config(path)

    assert resolved.execution.session_id_path_depth == 3


def test_skip_existing_sidecars_override_resolves(tmp_path: pathlib.Path) -> None:
    """The caching guard can be turned off to force a full re-extraction and re-index."""
    payload = {**_MINIMAL_URIS_CONFIG, "execution": {"skip_existing_sidecars": False}}
    path = _write(tmp_path, payload)

    resolved = config.resolve_config(path)

    assert resolved.execution.skip_existing_sidecars is False


def test_rejects_a_chunk_bytes_below_one(tmp_path: pathlib.Path) -> None:
    """chunk_bytes must be a positive size, not zero or negative."""
    payload = {**_MINIMAL_URIS_CONFIG, "execution": {"chunk_bytes": 0}}
    path = _write(tmp_path, payload)

    with pytest.raises(pydantic.ValidationError):
        config.resolve_config(path)


def test_rejects_a_no_progress_timeout_s_below_one(tmp_path: pathlib.Path) -> None:
    """no_progress_timeout_s must be a positive number of seconds, not zero or negative."""
    payload = {**_MINIMAL_URIS_CONFIG, "execution": {"no_progress_timeout_s": 0}}
    path = _write(tmp_path, payload)

    with pytest.raises(pydantic.ValidationError):
        config.resolve_config(path)


def test_rejects_an_unsupported_scheme(tmp_path: pathlib.Path) -> None:
    """A location scheme other than s3:// or file:// is refused, not silently passed through."""
    payload = {**_MINIMAL_URIS_CONFIG, "output": {"path": "gcs://bucket/prefix"}}
    path = _write(tmp_path, payload)

    with pytest.raises(pydantic.ValidationError):
        config.resolve_config(path)


def test_normalizes_a_file_uri_path_prefix_to_a_bare_local_path(tmp_path: pathlib.Path) -> None:
    """A file:// path_prefix resolves to a plain local path -- what discovery's own pathlib calls need.

    Without this, is_remote_path("file://...") is False (correctly "not S3/Azure"), but every
    "local" code path downstream hands the string straight to pathlib.Path(), which does not
    understand the file:// scheme at all and silently finds nothing.
    """
    local_dir = tmp_path / "input"
    local_dir.mkdir()
    payload = {
        **_MINIMAL_PATH_PREFIX_CONFIG,
        "input": {**_MINIMAL_PATH_PREFIX_CONFIG["input"], "path_prefix": f"file://{local_dir}"},
    }
    path = _write(tmp_path, payload)

    resolved = config.resolve_config(path)

    assert resolved.input.path_prefix == str(local_dir)


def test_normalizes_a_file_uri_in_uris_to_a_bare_local_path(tmp_path: pathlib.Path) -> None:
    """A file:// entry in uris resolves the same way path_prefix does -- uris had no validator at all."""
    media_path = tmp_path / "a.mp4"
    media_path.write_bytes(b"not-a-real-mp4")
    payload = {**_MINIMAL_URIS_CONFIG, "input": {**_MINIMAL_URIS_CONFIG["input"], "uris": [f"file://{media_path}"]}}
    path = _write(tmp_path, payload)

    resolved = config.resolve_config(path)

    assert resolved.input.uris == (str(media_path),)


def test_rejects_an_empty_output_path(tmp_path: pathlib.Path) -> None:
    """An empty output location is refused rather than accepted and failed later."""
    payload = {**_MINIMAL_URIS_CONFIG, "output": {"path": ""}}
    path = _write(tmp_path, payload)

    with pytest.raises(pydantic.ValidationError):
        config.resolve_config(path)


def test_rejects_an_output_path_with_leading_or_trailing_whitespace(tmp_path: pathlib.Path) -> None:
    """Whitespace-padded locations are refused, not silently stripped or matched as-is."""
    payload = {**_MINIMAL_URIS_CONFIG, "output": {"path": " s3://bucket/mp4-index"}}
    path = _write(tmp_path, payload)

    with pytest.raises(pydantic.ValidationError):
        config.resolve_config(path)


def test_rejects_unknown_top_level_fields(tmp_path: pathlib.Path) -> None:
    """The config is strict -- a typo'd field name fails loudly, not silently ignored."""
    payload = {**_MINIMAL_URIS_CONFIG, "not_a_real_field": True}
    path = _write(tmp_path, payload)

    with pytest.raises(pydantic.ValidationError):
        config.resolve_config(path)


def test_set_override_changes_a_field(tmp_path: pathlib.Path) -> None:
    """A dotted --set override is applied before validation."""
    path = _write(tmp_path, _MINIMAL_PATH_PREFIX_CONFIG)

    resolved = config.resolve_config(path, overrides=["input.limit=5"])

    assert resolved.input.limit == 5


def test_missing_config_file_raises_file_not_found(tmp_path: pathlib.Path) -> None:
    """A path that doesn't exist fails clearly rather than as a confusing parse error."""
    with pytest.raises(FileNotFoundError):
        config.resolve_config(tmp_path / "does-not-exist.yaml")


def test_json_config_is_also_accepted(tmp_path: pathlib.Path) -> None:
    """The loader isn't YAML-only -- a .json config resolves the same way."""
    path = tmp_path / "config.json"
    path.write_text(json.dumps(_MINIMAL_URIS_CONFIG))

    resolved = config.resolve_config(path)

    assert resolved.input.uris == ("s3://bucket/a.mp4",)


def test_config_template_yaml_round_trips() -> None:
    """The printed template itself parses and validates as a real config."""
    rendered = config.config_template_yaml()
    loaded = yaml.safe_load(rendered)

    resolved = config.ResolvedMp4IndexConfig.model_validate(loaded)

    assert resolved.kind == "mp4-index"


def test_config_template_payload_has_required_fields() -> None:
    """Agent-facing template metadata names every field a useful run actually requires."""
    payload = config.config_template_payload()

    assert payload["kind"] == "mp4-index"
    assert {field["path"] for field in payload["required_fields"]} == {
        "schema_version",
        "kind",
        "input.uris|input.path_prefix+input.globs",
        "input.container_kind",
        "output.path",
    }


def test_config_schema_json_is_valid_json() -> None:
    """The rendered JSON Schema parses and names the resolved config model."""
    schema = json.loads(config.config_schema_json())

    assert schema["title"] == "ResolvedMp4IndexConfig"


def test_resolved_config_to_json_round_trips(tmp_path: pathlib.Path) -> None:
    """The rendered resolved-config JSON carries the real resolved values."""
    path = _write(tmp_path, _MINIMAL_URIS_CONFIG)
    resolved = config.resolve_config(path)

    rendered = json.loads(config.resolved_config_to_json(resolved))

    assert rendered["kind"] == "mp4-index"
    assert rendered["input"]["uris"] == ["s3://bucket/a.mp4"]


def test_extraction_contract_id_is_stable_for_the_same_settings(tmp_path: pathlib.Path) -> None:
    """Two configs resolved separately, but with the same extraction-affecting fields, agree exactly."""
    first = config.resolve_config(_write(tmp_path, _MINIMAL_URIS_CONFIG, "a.yaml"))
    second = config.resolve_config(_write(tmp_path, _MINIMAL_URIS_CONFIG, "b.yaml"))

    assert first.extraction_contract_id == second.extraction_contract_id


def test_extraction_contract_id_changes_with_container_kind(tmp_path: pathlib.Path) -> None:
    """container_kind is result-defining -- changing it must change the contract id."""
    nonfragmented = config.resolve_config(_write(tmp_path, _MINIMAL_URIS_CONFIG, "a.yaml"))
    fragmented_payload = {
        **_MINIMAL_URIS_CONFIG,
        "input": {**_MINIMAL_URIS_CONFIG["input"], "container_kind": "fragmented"},
    }  # type: ignore[dict-item]
    fragmented = config.resolve_config(_write(tmp_path, fragmented_payload, "b.yaml"))

    assert nonfragmented.extraction_contract_id != fragmented.extraction_contract_id


def test_extraction_contract_id_changes_with_session_id_path_depth(tmp_path: pathlib.Path) -> None:
    """session_id_path_depth is result-defining (it changes session_camera_id) -- must change the contract id."""
    depth_2 = config.resolve_config(_write(tmp_path, _MINIMAL_URIS_CONFIG, "a.yaml"))
    depth_3_payload = {**_MINIMAL_URIS_CONFIG, "execution": {"session_id_path_depth": 3}}
    depth_3 = config.resolve_config(_write(tmp_path, depth_3_payload, "b.yaml"))

    assert depth_2.extraction_contract_id != depth_3.extraction_contract_id


def test_extraction_contract_id_is_unaffected_by_execution_tuning(tmp_path: pathlib.Path) -> None:
    """chunk_bytes/skip_existing_sidecars/actor sizing never change the output -- must not change the contract id."""
    default_tuning = config.resolve_config(_write(tmp_path, _MINIMAL_URIS_CONFIG, "a.yaml"))
    retuned_payload = {
        **_MINIMAL_URIS_CONFIG,
        "execution": {"chunk_bytes": 2048, "min_actors": 5, "skip_existing_sidecars": False, "progress": True},
    }
    retuned = config.resolve_config(_write(tmp_path, retuned_payload, "b.yaml"))

    assert default_tuning.extraction_contract_id == retuned.extraction_contract_id
