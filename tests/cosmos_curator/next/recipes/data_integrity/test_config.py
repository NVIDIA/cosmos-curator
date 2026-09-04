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

"""Unit tests for data-integrity config resolution."""

import json
import pathlib
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from cosmos_curator.core.sensors.data_integrity.instruments import DEFAULT_THRESHOLDS, MAX_PERCENT
from cosmos_curator.next.recipes.data_integrity import config


def _config_data(**overrides: Any) -> dict[str, Any]:  # noqa: ANN401 -- test builder, one block per key
    """Build the smallest config a test can vary one block of."""
    data: dict[str, Any] = {
        "schema_version": 1,
        "kind": "data-integrity",
        "input": {"sessions": ["/data/clips/one"]},
        "output": {"store_root": "/data/di-store"},
    }
    data.update(overrides)
    return data


def test_the_template_is_a_valid_config() -> None:
    """``pipeline template`` output must be runnable after editing, not just readable."""
    resolved = config.resolve_config_data(config.config_template())

    assert resolved.kind == config.KIND_NAME
    assert resolved.input.sessions


def test_the_template_shows_every_setting() -> None:
    """Nulls are rendered rather than dropped, so the template documents what can be set."""
    template = config.config_template()

    assert template["input"]["session_list_uri"] is None
    assert template["checks"]["expected_hz"] is None
    assert set(template["checks"]["thresholds"]) == {
        "max_strict_violations",
        "max_rate_deviation_percent",
        "max_gaps",
        "max_jitter_percent",
        "allow_frame_reordering",
        # Session-grain, and settable here for the same reason the rest are: a pipeline
        # run judges sessions too, so the policy it applies to them has to be stateable.
        "max_sensor_spread_ns",
        "max_non_overlap_percent",
    }
    assert template["execution"]["s3_profile_name"] is None
    assert yaml.safe_load(config.config_template_yaml()) == template


def test_thresholds_default_to_the_kernel_policy() -> None:
    """An omitted ``thresholds`` block must mean the same policy the CLIs apply."""
    resolved = config.resolve_config_data(_config_data())

    assert resolved.checks.thresholds.to_thresholds() == DEFAULT_THRESHOLDS


def test_a_non_overlap_share_above_a_hundred_percent_is_rejected() -> None:
    """The one threshold with a ceiling: a share of a session cannot exceed all of it.

    So a limit above 100 describes no session that exists, and asking for one is a
    typo rather than a laxer policy -- 100 already tolerates everything. The bound is
    the same one ``--max-non-overlap-percent`` enforces, both read off ``MAX_PERCENT``.
    """
    assert config.resolve_config_data(
        _config_data(checks={"thresholds": {"max_non_overlap_percent": 100.0}})
    ).checks.thresholds.max_non_overlap_percent == pytest.approx(MAX_PERCENT)

    with pytest.raises(ValidationError, match="less than or equal to 100"):
        config.resolve_config_data(_config_data(checks={"thresholds": {"max_non_overlap_percent": 101}}))


def test_a_negative_sensor_spread_limit_is_rejected() -> None:
    """The other session threshold has a floor rather than a ceiling: a spread is a distance.

    Zero is a real policy -- a rig that claims perfect synchronisation -- so the bound
    is ``ge``, not ``gt``.
    """
    resolved = config.resolve_config_data(_config_data(checks={"thresholds": {"max_sensor_spread_ns": 0}}))
    assert resolved.checks.thresholds.max_sensor_spread_ns == 0

    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        config.resolve_config_data(_config_data(checks={"thresholds": {"max_sensor_spread_ns": -1}}))


def test_dotted_overrides_apply_before_validation() -> None:
    """``--set`` values are validated like any other input rather than trusted."""
    resolved = config.resolve_config_data(
        _config_data(),
        overrides=["checks.expected_hz=30", "execution.session_concurrency=4"],
    )

    assert resolved.checks.expected_hz == 30.0
    assert resolved.execution.session_concurrency == 4

    with pytest.raises(ValidationError):
        config.resolve_config_data(_config_data(), overrides=["execution.session_concurrency=0"])


def test_an_empty_input_block_is_rejected() -> None:
    """No input form at all would run over nothing, which is a mistake, not a result."""
    with pytest.raises(ValidationError, match="at least one of"):
        config.resolve_config_data(_config_data(input={}))


@pytest.mark.parametrize(
    "input_block",
    [
        {"sessions": ["/data/clips/one"]},
        {"session_list_uri": "/data/sessions.txt"},
        {"session_roots": ["/data/clips"]},
        {"sessions": ["/data/clips/one"], "session_roots": ["/data/other"]},
    ],
)
def test_any_combination_of_input_forms_is_accepted(input_block: dict[str, Any]) -> None:
    """The three forms are combinable: a root plus a few extra sessions is normal."""
    assert config.resolve_config_data(_config_data(input=input_block)).input is not None


def test_an_azure_store_root_is_rejected() -> None:
    """``get_lance_storage_options`` cannot authenticate an az:// store, so refuse it here."""
    with pytest.raises(ValidationError, match="az://"):
        config.resolve_config_data(_config_data(output={"store_root": "az://container/di-store"}))


def test_an_unknown_store_scheme_is_rejected() -> None:
    """Anything else with a scheme would become a local directory named after the URI."""
    with pytest.raises(ValidationError, match="unsupported"):
        config.resolve_config_data(_config_data(output={"store_root": "gs://bucket/di-store"}))


def test_a_bucketless_s3_store_root_is_rejected() -> None:
    """``s3://`` alone fails far deeper, where the message belongs to Lance rather than us."""
    with pytest.raises(ValidationError, match="names no bucket"):
        config.resolve_config_data(_config_data(output={"store_root": "s3://"}))


def test_a_local_store_root_keeps_its_home_expansion() -> None:
    """``~`` is expanded once, here, so every writer downstream sees one path."""
    resolved = config.resolve_config_data(_config_data(output={"store_root": "~/di-store"}))

    assert resolved.output.store_root == str(pathlib.Path("~/di-store").expanduser())


def test_an_s3_store_root_is_left_verbatim() -> None:
    """An object key is opaque: normalizing it would rename the store."""
    resolved = config.resolve_config_data(_config_data(output={"store_root": "s3://bucket/di-store/"}))

    assert resolved.output.store_root == "s3://bucket/di-store/"


def test_a_deeper_session_depth_is_rejected() -> None:
    """Only immediate children are supported today, and silently ignoring depth would lie."""
    with pytest.raises(ValidationError):
        config.resolve_config_data(_config_data(input={"session_roots": ["/data/clips"], "session_depth": 2}))


def test_an_unknown_key_is_rejected() -> None:
    """``extra="forbid"`` turns a typo'd knob into an error instead of a silent default."""
    with pytest.raises(ValidationError):
        config.resolve_config_data(_config_data(execution={"stream_concurency": 4}))


def test_the_underscored_kind_is_not_this_recipe() -> None:
    """``data_integrity`` is not a spelling of ``data-integrity``; it is not a kind at all."""
    with pytest.raises(ValidationError):
        config.resolve_config_data(_config_data(kind="data_integrity"))


@pytest.mark.parametrize("suffix", [".yaml", ".json"])
def test_a_config_file_resolves_from_either_format(tmp_path: pathlib.Path, suffix: str) -> None:
    """Both spellings a caller might hand ``pipeline validate`` load the same way."""
    path = tmp_path / f"config{suffix}"
    data = _config_data()
    path.write_text(yaml.safe_dump(data) if suffix == ".yaml" else json.dumps(data))

    assert config.resolve_config(path).output.store_root == "/data/di-store"


def test_a_missing_config_file_is_reported_as_missing(tmp_path: pathlib.Path) -> None:
    """The runtime turns this into exit 2; a bare KeyError would not say why."""
    with pytest.raises(FileNotFoundError):
        config.resolve_config(tmp_path / "absent.yaml")


def test_the_schema_and_render_helpers_round_trip() -> None:
    """``pipeline schema`` and ``pipeline render`` both emit parseable JSON."""
    schema = json.loads(config.config_schema_json())
    rendered = json.loads(config.resolved_config_to_json(config.resolve_config_data(_config_data())))

    assert schema["title"] == "ResolvedDataIntegrityConfig"
    assert rendered["kind"] == config.KIND_NAME
    assert rendered["execution"]["azure_profile_name"] == "default"


def test_the_template_payload_names_the_input_alternatives() -> None:
    """An agent reading the payload should see that the input forms are alternatives."""
    payload = config.config_template_payload()
    paths = [field["path"] for field in payload["required_fields"]]

    assert payload["kind"] == config.KIND_NAME
    assert "input.sessions|input.session_list_uri|input.session_roots" in paths
