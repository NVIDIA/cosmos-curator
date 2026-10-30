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

"""Tests for the ``data-integrity`` pipeline-kind surface the CLI drives."""

import json
import pathlib
import subprocess
import sys

import pytest
import yaml
from pydantic import ValidationError

from cosmos_curator.next.recipes.data_integrity import pipeline
from cosmos_curator.next.recipes.data_integrity.pipeline_kind import DATA_INTEGRITY_KIND

#: What ``pipeline template`` / ``validate`` / ``schema`` must not have to load. Ray and
#: Lance dominate import time, and the CLI registers every kind on every invocation.
_HEAVY_MODULES = ("ray", "lance", "pyarrow")


def _write_config(tmp_path: pathlib.Path, store_root: str | None = "/data/di-store") -> pathlib.Path:
    """Write the smallest valid config to a file the kind's callables can take."""
    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "kind": "data-integrity",
                "input": {"sessions": ["/data/clips/one"]},
                "output": {"store_root": store_root},
            }
        )
    )
    return path


def test_registering_the_kind_loads_neither_ray_nor_lance(repo_root: pathlib.Path) -> None:
    """Every import is deferred into the callable that needs it, and this pins that."""
    probe = (
        "import sys;"
        "import cosmos_curator.next.recipes.data_integrity.pipeline_kind;"
        f"print([m for m in {_HEAVY_MODULES!r} if m in sys.modules])"
    )
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", probe],
        check=True,
        capture_output=True,
        text=True,
        cwd=repo_root,
    )

    assert result.stdout.strip() == "[]"


def test_the_kind_is_registered_under_its_hyphenated_name() -> None:
    """``data_integrity`` is a different name, not a spelling of this one."""
    assert DATA_INTEGRITY_KIND.name == "data-integrity"


def test_the_template_it_offers_is_the_config_it_accepts(tmp_path: pathlib.Path) -> None:
    """``pipeline template`` into a file, then ``pipeline validate`` on it, has to work."""
    path = tmp_path / "template.yaml"
    path.write_text(DATA_INTEGRITY_KIND.template_yaml())

    assert DATA_INTEGRITY_KIND.validate(path, []) == {"ok": True}


def test_rendering_resolves_the_config_it_was_given(tmp_path: pathlib.Path) -> None:
    """``pipeline render`` shows every default the run will actually use."""
    rendered = json.loads(DATA_INTEGRITY_KIND.render(_write_config(tmp_path), ["execution.session_concurrency=2"]))

    assert rendered["execution"]["session_concurrency"] == 2
    assert rendered["checks"]["thresholds"]["max_gaps"] == 0


def test_validation_reports_a_bad_config_rather_than_a_bad_run(tmp_path: pathlib.Path) -> None:
    """The store root is checked before any stream is read."""
    with pytest.raises(ValidationError, match="az://"):
        DATA_INTEGRITY_KIND.validate(_write_config(tmp_path, store_root="az://container/store"), [])


def test_validation_refuses_a_config_that_names_no_store(tmp_path: pathlib.Path) -> None:
    """``validate`` has to refuse exactly what ``run`` would, or it is not validation.

    A null ``store_root`` is legal on the model, because the Ray-free ``di`` tool may
    report without persisting and both tools share one ``kind``. A pipeline run has no
    report to fall back on, so the check lives on this side of the seam -- and it has to
    sit on both callables, since passing ``pipeline validate`` and then being rejected by
    ``run-pipeline`` is worse than a single clear refusal.
    """
    with pytest.raises(ValueError, match=r"output\.store_root is required"):
        DATA_INTEGRITY_KIND.validate(_write_config(tmp_path, store_root=None), [])


def test_the_schema_it_publishes_names_the_four_blocks() -> None:
    """``pipeline schema`` is what an agent reads to write a config."""
    schema = json.loads(DATA_INTEGRITY_KIND.schema_json())

    assert set(schema["properties"]) == {"schema_version", "kind", "input", "checks", "output", "execution"}


def test_the_kind_ships_no_presets() -> None:
    """Nothing about a dataset is guessable here: sessions and a store root are required."""
    assert DATA_INTEGRITY_KIND.list_presets() == []


def test_preparing_a_run_resolves_the_config_before_starting_ray(tmp_path: pathlib.Path) -> None:
    """A typo'd config must fail while the CLI is still the thing reporting errors."""
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump({"schema_version": 1, "kind": "data-integrity", "input": {}}))

    with pytest.raises(ValidationError):
        DATA_INTEGRITY_KIND.prepare_run(bad, set_overrides=[])


def test_a_prepared_run_reports_its_findings_without_failing(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Findings ride in the payload and the message; the exit status stays the runtime's."""
    summary = {
        "run_id": "abc",
        "sessions": 2,
        "streams": 7,
        "unreadable": 1,
        "unreachable": 0,
        "unlisted_sessions": 0,
        "empty_sessions": 0,
        "failed_metrics": 3,
        "store_root": "/data/di-store",
    }
    monkeypatch.setattr(pipeline, "run_config", lambda _config: summary)

    output = DATA_INTEGRITY_KIND.prepare_run(_write_config(tmp_path), set_overrides=[])()

    assert output.json_payload == summary
    assert "run abc committed 7 stream(s)" in output.message
    assert "1 unreadable, 3 failed metric(s)" in output.message
    assert "held no streams" not in output.message


def test_a_prepared_run_mentions_sessions_that_held_nothing(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The quiet finding: a session that lists clean and empty is worth a line, not a failure."""
    summary = {
        "run_id": "abc",
        "sessions": 2,
        "streams": 7,
        "unreadable": 0,
        "unreachable": 0,
        "unlisted_sessions": 0,
        "empty_sessions": 1,
        "failed_metrics": 0,
        "store_root": "/data/di-store",
    }
    monkeypatch.setattr(pipeline, "run_config", lambda _config: summary)

    output = DATA_INTEGRITY_KIND.prepare_run(_write_config(tmp_path), set_overrides=[])()

    assert "1 session(s) held no streams" in output.message
