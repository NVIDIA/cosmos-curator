# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The generic-CLI surface, exercised without a GPU, a model, or a clip table.

Every callback on the kind is reachable from ``cosmos-curator pipeline`` before
anything is staged, so all of them must work against a config file alone. That is
what these tests hold: the config operations touch no storage, and the run
operation is split so that resolution happens when the command is prepared while
execution waits for the prepared callable.

::

    validate / render / schema / template   config file only
    prepare_run(config)                     resolves, runs NOTHING
    prepared()                              -> run_config(resolved)

``run_config`` is replaced for the two dispatch tests: it opens Lance, starts Ray,
and reserves GPUs, none of which the adapter's own behaviour depends on.
"""

import json
from collections.abc import Callable
from pathlib import Path

import pytest
import yaml

from cosmos_curator.next.recipes.robot_action_caption import pipeline
from cosmos_curator.next.recipes.robot_action_caption.config import (
    KIND_NAME,
    ResolvedRobotActionCaptionConfig,
    resolve_config_data,
)
from cosmos_curator.next.recipes.robot_action_caption.pipeline import CaptionRunResult
from cosmos_curator.next.recipes.robot_action_caption.pipeline_kind import ROBOT_ACTION_CAPTION_KIND

ConfigFile = Callable[[], Path]


@pytest.fixture
def config_file(tmp_path: Path) -> ConfigFile:
    """Return a builder for the smallest config the resolver accepts.

    Local paths rather than the template's ``s3://`` example, so nothing in these
    tests could reach a network even if a callback grew a read.
    """

    def build() -> Path:
        path = tmp_path / "caption.yaml"
        path.write_text(
            yaml.safe_dump(
                {
                    "schema_version": 1,
                    "kind": KIND_NAME,
                    "input": {"clips_lance_uri": str(tmp_path / "clips.lance")},
                    "output": {"staging_root_uri": str(tmp_path / "staging")},
                }
            ),
            encoding="utf-8",
        )
        return path

    return build


def _result(*, published_rows: int, error_rows: int = 3, truncated_rows: int = 4) -> CaptionRunResult:
    """Return a run result whose every reported field is distinct.

    ``published_rows`` is the discriminator the outcome branches on, because
    ``ending_version`` is never null and so cannot distinguish a run that
    published from one that owed nothing.

    The verdict counts default non-zero and pairwise distinct on purpose. Zeros,
    or two fields sharing a value, let a line that renders the wrong field read
    correctly - and these are exactly the fields an operator uses to tell a
    captioned corpus from one that errored on every clip. They are overridable
    because both are SUBSETS of ``published_rows``, so a no-work result has to
    pass zeros to stay a shape the recipe could actually return.

    The shard pair is set to a real scope rather than ``None`` so the summary
    assertions below see the values a sharded campaign would emit, and distinct
    from every other integer here for the same reason the verdict counts are.
    """
    return CaptionRunResult(
        clips_lance_uri="/clips.lance",
        source_version=1,
        ending_version=2,
        published_rows=published_rows,
        error_rows=error_rows,
        truncated_rows=truncated_rows,
        contract_digest="d" * 64,
        shard_count=8,
        shard_index=5,
    )


def test_the_kind_is_registered_under_its_config_discriminator() -> None:
    """The CLI resolves a config's ``kind`` field straight to this adapter."""
    from cosmos_curator.client.pipeline_cli.builtin_pipeline_kinds import (  # noqa: PLC0415 - client is dev-only
        BUILTIN_PIPELINE_KINDS,
    )

    assert BUILTIN_PIPELINE_KINDS.get(KIND_NAME) is ROBOT_ACTION_CAPTION_KIND


def test_validate_accepts_a_minimal_config(config_file: ConfigFile) -> None:
    """Validation runs on the config alone, with no table and no weights present."""
    assert ROBOT_ACTION_CAPTION_KIND.validate(config_file(), ()) == {"ok": True}


def test_render_returns_the_resolved_config_as_json(config_file: ConfigFile) -> None:
    """The rendered form is the canonical config, which is what an operator archives."""
    rendered = json.loads(ROBOT_ACTION_CAPTION_KIND.render(config_file(), ()))

    assert rendered["kind"] == KIND_NAME


def test_render_applies_a_dotted_override(config_file: ConfigFile) -> None:
    """An override reaches the resolved config, so ``--set`` is not silently dropped."""
    rendered = json.loads(ROBOT_ACTION_CAPTION_KIND.render(config_file(), ("execution.inference_concurrency=8",)))

    assert rendered["execution"]["inference_concurrency"] == 8


def test_the_schema_describes_the_resolved_config_model() -> None:
    """The published schema is the one the resolver enforces, not a hand-written copy."""
    assert json.loads(ROBOT_ACTION_CAPTION_KIND.schema_json())["title"] == "ResolvedRobotActionCaptionConfig"


def test_the_template_is_itself_a_valid_config() -> None:
    """The template is offered as a starting point, so it has to resolve unedited."""
    template = ROBOT_ACTION_CAPTION_KIND.template_payload()["config"]

    assert resolve_config_data(template).kind == KIND_NAME


def test_the_template_yaml_parses_back_to_the_template_payload() -> None:
    """The YAML an operator is handed is the same config the payload advertises."""
    rendered = yaml.safe_load(ROBOT_ACTION_CAPTION_KIND.template_yaml())

    assert rendered == ROBOT_ACTION_CAPTION_KIND.template_payload()["config"]


def test_preparing_a_run_resolves_the_config_without_executing_it(
    config_file: ConfigFile,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preparation is the CLI's argument check, so it must claim no GPU and no table."""
    calls: list[ResolvedRobotActionCaptionConfig] = []
    monkeypatch.setattr(pipeline, "run_config", lambda config: calls.append(config) or _result(published_rows=7))

    ROBOT_ACTION_CAPTION_KIND.prepare_run(config_file(), set_overrides=[])

    assert calls == []


def test_invoking_a_prepared_run_passes_the_resolved_config(
    config_file: ConfigFile,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The resolved config reaches the recipe, rather than the raw file being re-read."""
    calls: list[ResolvedRobotActionCaptionConfig] = []
    monkeypatch.setattr(pipeline, "run_config", lambda config: calls.append(config) or _result(published_rows=7))

    ROBOT_ACTION_CAPTION_KIND.prepare_run(config_file(), set_overrides=[])()

    assert [config.kind for config in calls] == [KIND_NAME]


def test_a_published_run_reports_the_rows_the_version_and_the_verdict_counts(
    config_file: ConfigFile,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The operator-facing line names the version and both non-success verdicts.

    An attempt that charges a fault to every clip one row at a time still
    succeeds and still commits, so the row count alone cannot tell that outcome
    from a corpus of captions. The two subset counts are what separate them, and
    this line is the only place a plain-mode run sees either.
    """
    monkeypatch.setattr(pipeline, "run_config", lambda _config: _result(published_rows=7))

    output = ROBOT_ACTION_CAPTION_KIND.prepare_run(config_file(), set_overrides=[])()

    assert output.message == "robot-action-caption: 7 caption row(s) canonical at v2 (3 error, 4 truncated)"


def test_a_run_that_owed_no_work_says_so_while_still_naming_the_version(
    config_file: ConfigFile,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Owing nothing is a distinct outcome, but the table is still canonical somewhere.

    The version is reported either way because it is never null: a run that
    publishes reports its commit, and one that owes nothing reports the version it
    pinned, which already includes the caption-column widening.
    """
    monkeypatch.setattr(
        pipeline,
        "run_config",
        lambda _config: _result(published_rows=0, error_rows=0, truncated_rows=0),
    )

    output = ROBOT_ACTION_CAPTION_KIND.prepare_run(config_file(), set_overrides=[])()

    assert output.message == "robot-action-caption: no caption owed at v2"


def test_the_run_payload_is_the_published_fields(
    config_file: ConfigFile,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The machine result is a consumed format, so its exact mapping is pinned.

    A literal rather than a comparison against ``as_summary()`` itself: that
    re-derives both sides through one function, so a renamed or dropped key would
    move them together and agree. Pinning the mapping is what makes a change to
    the payload a deliberate one.

    The shard pair is part of that mapping because it is what tells one campaign
    job's summary from another's - without it eight jobs emit payloads
    distinguishable only by their row counts.
    """
    monkeypatch.setattr(pipeline, "run_config", lambda _config: _result(published_rows=7))

    output = ROBOT_ACTION_CAPTION_KIND.prepare_run(config_file(), set_overrides=[])()

    assert output.json_payload == {
        "clips_lance_uri": "/clips.lance",
        "source_version": 1,
        "ending_version": 2,
        "published_rows": 7,
        "error_rows": 3,
        "truncated_rows": 4,
        "caption_contract_digest": "d" * 64,
        "shard_count": 8,
        "shard_index": 5,
    }
