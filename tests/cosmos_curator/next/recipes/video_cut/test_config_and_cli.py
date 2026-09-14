# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public config/CLI contracts and stable work identities."""

import json
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError
from typer.testing import CliRunner

from cosmos_curator.next.recipes.video_cut import pipeline
from cosmos_curator.next.recipes.video_cut.config import (
    InputConfig,
    OutputConfig,
    ResolvedVideoCutConfig,
    config_template,
    resolve_config_data,
)
from cosmos_curator.next.recipes.video_cut.contracts import make_source_id, make_span_id


def test_normalization_and_contract_scope(config: ResolvedVideoCutConfig) -> None:
    """Tuning, destinations, and selection order leave compatible cutting work intact."""
    raw = config.model_dump(mode="json")
    raw["input"]["uris"] = list(reversed(raw["input"]["uris"])) * 2
    tuned = resolve_config_data(
        raw, overrides=["execution.sources_per_checkpoint=10", "execution.spans_per_fragment=99"]
    )
    assert tuned.input == config.input
    assert tuned.split.contract_id == config.split.contract_id
    changed = resolve_config_data(raw, overrides=["split.stride_s=1.0"])
    assert changed.split.contract_id != config.split.contract_id
    literal = "s3://test-bucket/a%2Fb//c.mp4"
    raw["input"]["uris"] = [literal]
    assert resolve_config_data(raw).input.uris == (literal,)
    assert make_source_id(literal) != make_source_id("s3://test-bucket/a/b//c.mp4")
    assert make_span_id(make_source_id(literal), 0, 10) != make_span_id(make_source_id(literal), 0, 11)


@pytest.mark.parametrize(
    "raw",
    [
        {},
        {"uris": []},
        {"uris": ["s3://test-bucket/a.mp4"], "root_uri": "s3://test-bucket/raw"},
        {"root_uri": "https://test-bucket/raw"},
        {"root_uri": "s3://test-bucket/raw?signature=secret"},
    ],
)
def test_input_requires_one_source_selection(raw: dict[str, object]) -> None:
    """A root and explicit objects are alternatives; invalid selectors fail offline."""
    with pytest.raises(ValidationError):
        InputConfig.model_validate(raw)


def test_media_root_derives_outputs_and_preserves_overrides(tmp_path: Path) -> None:
    """S3/local defaults share a root while existing explicit locations remain usable."""
    for root in ["s3://test-bucket/curated/video-cut/", str(tmp_path / "output") + "/"]:
        output = OutputConfig(media_root=root)
        normalized = root.rstrip("/")
        assert output.media_root == normalized
        assert output.spans_lance_uri == f"{normalized}/lance"
        assert output.staging_root_uri == f"{normalized}/staging/video-cut"
    output = OutputConfig(
        media_root="s3://test-bucket/new-root",
        spans_lance_uri=str(tmp_path / "existing.lance"),
        staging_root_uri="s3://test-bucket/existing-workspace",
    )
    assert output.spans_lance_uri == str(tmp_path / "existing.lance")
    assert output.staging_root_uri == "s3://test-bucket/existing-workspace"
    with pytest.raises(ValidationError, match="separate directories"):
        OutputConfig(media_root=str(tmp_path), staging_root_uri=str(tmp_path / "lance" / "staging"))


def test_root_config_renders_without_discovery(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """CLI validation/rendering needs no S3 access and resolves defaults after overrides."""
    from cosmos_curator.client.cli import cosmos_curator  # noqa: PLC0415 - deployment CLI dependencies are dev-only

    monkeypatch.setattr(pipeline, "resolve_input_uris", lambda *_args, **_kwargs: pytest.fail("unexpected discovery"))
    raw = {
        "schema_version": 1,
        "kind": "video-cut",
        "input": {"root_uri": "S3://test-bucket/raw/"},
        "output": {"media_root": "s3://test-bucket/output"},
    }
    path = tmp_path / "root.yaml"
    path.write_text(yaml.safe_dump(raw))
    runner = CliRunner()
    validation = runner.invoke(cosmos_curator, ["pipeline", "validate", str(path)])
    assert validation.exit_code == 0, validation.output
    rendered = runner.invoke(
        cosmos_curator, ["pipeline", "render", str(path), "--set", "output.media_root=s3://test-bucket/other"]
    )
    assert rendered.exit_code == 0, rendered.output
    resolved = json.loads(rendered.stdout)
    assert resolved["input"] == {"uris": None, "root_uri": "s3://test-bucket/raw"}
    assert resolved["output"]["spans_lance_uri"] == "s3://test-bucket/other/lance"
    assert resolved["output"]["staging_root_uri"] == "s3://test-bucket/other/staging/video-cut"
    explicit = resolve_config_data(raw | {"input": {"uris": ["s3://test-bucket/raw/a.mp4"]}})
    assert resolve_config_data(raw).split.contract_id == explicit.split.contract_id


@pytest.mark.parametrize(
    "override",
    [
        "input.uris=[]",
        "input.uris=[https://example.com/a.mp4]",
        "input.uris=[s3://test-bucket/a.mkv]",
        'input.uris=["s3://test-bucket/a.mp4?token=secret"]',
        'input.uris=["s3://test-bucket/a.mp4?"]',
        'input.uris=["s3://test-bucket/a.mp4/"]',
        'input.uris=["s3://test-bucket/a\\nb.mp4"]',
        "split.duration_s=0.0000000001",
        "split.min_duration_s=100.0",
        "execution.sources_per_checkpoint=0",
        "execution.typo=1",
        "kind=video_cut",
    ],
)
def test_invalid_configs_fail_before_work(override: str) -> None:
    """Bad source selections and settings never reach runtime."""
    with pytest.raises(ValidationError):
        resolve_config_data(config_template(), overrides=[override])


def test_cli_template_render_schema_and_dispatch(
    config: ResolvedVideoCutConfig,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Generic CLI and runtime select the new recipe, preserving lazy registration."""
    from cosmos_curator.client.cli import cosmos_curator  # noqa: PLC0415 - deployment CLI dependencies are dev-only
    from cosmos_curator.client.pipeline_cli import pipeline_runtime  # noqa: PLC0415 - client is dev-only

    runner = CliRunner()
    template = runner.invoke(cosmos_curator, ["pipeline", "template", "video-cut"])
    assert template.exit_code == 0, template.output
    assert yaml.safe_load(template.stdout) == config_template()
    assert "replace `uris` with `root_uri:" in template.stdout
    assert "derived from `media_root`" in template.stdout
    assert "root_uri" not in config_template()["input"]
    assert config_template()["execution"]["probe_cpus"] == 0.25
    assert config_template()["execution"]["sources_per_checkpoint"] == 1024
    assert config_template()["execution"]["spans_per_fragment"] == 100_000
    template_path = tmp_path / "template.yaml"
    template_path.write_text(template.stdout)
    rendered = runner.invoke(
        cosmos_curator,
        ["pipeline", "render", str(template_path), "--set", "output.media_root=s3://test-bucket/actual-output"],
    )
    assert rendered.exit_code == 0, rendered.output
    output = json.loads(rendered.stdout)["output"]
    assert output["spans_lance_uri"] == "s3://test-bucket/actual-output/lance"
    assert output["staging_root_uri"] == "s3://test-bucket/actual-output/staging/video-cut"
    schema = runner.invoke(cosmos_curator, ["pipeline", "schema", "video-cut", "--json"])
    assert schema.exit_code == 0, schema.output
    assert json.loads(schema.stdout)["properties"]["kind"]["const"] == "video-cut"
    path = tmp_path / "config.json"
    path.write_text(config.model_dump_json())
    kind = pipeline_runtime.BUILTIN_PIPELINE_KINDS.get("video-cut")
    assert kind.validate(path, ()) == {"ok": True}
    assert (
        json.loads(kind.render(path, ("execution.sources_per_checkpoint=3",)))["execution"]["sources_per_checkpoint"]
        == 3
    )
    seen = []

    def run(resolved: ResolvedVideoCutConfig) -> dict[str, int]:
        seen.append(resolved)
        return {"spans": 3, "errors": 0}

    monkeypatch.setattr(pipeline, "run_config", run)
    pipeline_runtime.main(path)
    captured = capsys.readouterr()
    assert "video-cut:" in captured.out
    assert captured.err == ""
    assert seen == [config]
