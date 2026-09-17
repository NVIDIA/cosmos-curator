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

"""Tests for the ``embeddings`` pipeline-kind surface: the CLI contract and its dispatch.

Three layers:

- Surface tests against the eight callbacks (template, validate, render, schema),
  which need no table and no Ray.
- Dispatch tests that stub ``run_embedding_pipeline``, pinning the payload and
  message the adapter assembles and proving the config is resolved exactly once.
- Two wiring smokes against a real local ``clips.lance``: one without Ray (the
  schema-widening commit and the no-action-data skip) and one ``ray_local`` fill
  through a stub embedder, which is the only place a config file drives real
  vectors into the table.

The Ray fill runs in the integration suite; the remaining tests run in the CPU job.
"""

import json
import pathlib
from collections.abc import Sequence

import lance
import numpy as np
import pyarrow as pa
import pytest
import yaml
from pydantic import ValidationError

from cosmos_curator.core.utils.pixi_runtime_envs import ray_data_gpu_runtime_env
from cosmos_curator.next.embeddings.action.pca import PcaArtifact
from cosmos_curator.next.embeddings.action.wrist_motion import DESCRIPTOR_DIM
from cosmos_curator.next.embeddings.schemas import ACTION_DIM, IMAGE_COLUMN_GROUP, IMAGE_DIM, image_columns_batch
from cosmos_curator.next.recipes.embeddings import config as config_module
from cosmos_curator.next.recipes.embeddings import embed
from cosmos_curator.next.recipes.embeddings.action_pca import ActionPca
from cosmos_curator.next.recipes.embeddings.config import EmbeddingPipelineConfig, Modality, resolve_config
from cosmos_curator.next.recipes.embeddings.embed import EmbeddingRunResult
from cosmos_curator.next.recipes.embeddings.modalities import (
    _IMAGE_APPLICABILITY_FILTER,
    ModalityFill,
    ModalityResult,
    WorkerResources,
)
from cosmos_curator.next.recipes.embeddings.pipeline_kind import EMBEDDINGS_KIND

from .conftest import DEFAULT_CLIPS_URI, ClipsTableFactory

_STUB_MODEL_ID = "stub-image-model"


def _write_config(root: pathlib.Path, **fields: object) -> pathlib.Path:
    """Write a minimal valid ``embeddings`` config, plus any extra top-level fields."""
    path = root / "embeddings.yaml"
    payload: dict[str, object] = {
        "schema_version": 1,
        "kind": "embeddings",
        "clips_lance_uri": DEFAULT_CLIPS_URI,
        **fields,
    }
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def _result(
    *modalities: ModalityResult,
    action: ActionPca | None = None,
    schema_commit_version: int | None = None,
) -> EmbeddingRunResult:
    """Return a synthetic run summary; the adapter only projects it, never inspects vectors.

    ``schema_commit_version`` defaults to "the columns were already there", which
    is what every run past a modality's first one sees.
    """
    return EmbeddingRunResult(modalities=modalities, action=action, schema_commit_version=schema_commit_version)


def _modality_result(
    modality: Modality,
    *,
    selected: int,
    filled: int,
    version: int | None,
    result_identity: dict[str, str] | None = None,
) -> ModalityResult:
    return ModalityResult(
        modality=modality,
        result_identity=result_identity if result_identity is not None else {},
        selected=selected,
        filled=filled,
        committed_version=version,
    )


def _synthetic_action_pca() -> ActionPca:
    """Return a loaded-basis wrapper over a zero basis; only its fingerprint is read here."""
    return ActionPca(
        artifact=PcaArtifact(
            mean=np.zeros(DESCRIPTOR_DIM, dtype=np.float64),
            components=np.zeros((ACTION_DIM, DESCRIPTOR_DIM), dtype=np.float64),
        ),
        samples_used=None,
    )


def _stub_run(monkeypatch: pytest.MonkeyPatch, result: EmbeddingRunResult) -> list[EmbeddingPipelineConfig]:
    """Replace the recipe driver with a recorder, returning the configs it was handed."""
    seen: list[EmbeddingPipelineConfig] = []

    def _record(config: EmbeddingPipelineConfig) -> EmbeddingRunResult:
        seen.append(config)
        return result

    monkeypatch.setattr(embed, "run_embedding_pipeline", _record)
    return seen


class _ConstantImageEmbedder:
    """Emit one always-valid image group row per input row, with no model and no media read."""

    def __call__(self, batch: pa.Table) -> pa.Table:
        """Return a constant vector for every row of ``batch``, in input order."""
        rows = batch.num_rows
        return image_columns_batch(
            rows,
            np.ones((rows, IMAGE_DIM), dtype=np.float32),
            _STUB_MODEL_ID,
            np.ones(rows, dtype=np.bool_),
        )


def _stub_image_fill(config: EmbeddingPipelineConfig) -> ModalityFill:
    """Build an image fill around the stub embedder, sized for the two-CPU local cluster."""
    return ModalityFill(
        modality=Modality.IMAGE,
        group=IMAGE_COLUMN_GROUP,
        source_columns=("clip_id", "clip_uri"),
        applicability_filter=_IMAGE_APPLICABILITY_FILTER,
        result_identity={"embedding_image_model_id": _STUB_MODEL_ID},
        embedder_cls=_ConstantImageEmbedder,
        embedder_kwargs={},
        resources=WorkerResources(scan_batch_size=config.image.batch_size, num_cpus=1),
        weights_name=None,
    )


def test_the_kind_is_registered_under_its_single_word_name() -> None:
    """The CLI resolves configs by their ``kind`` discriminator."""
    from cosmos_curator.client.pipeline_cli.builtin_pipeline_kinds import (  # noqa: PLC0415 - client is dev-only
        BUILTIN_PIPELINE_KINDS,
    )

    assert BUILTIN_PIPELINE_KINDS.get("embeddings") is EMBEDDINGS_KIND
    assert "embeddings" in BUILTIN_PIPELINE_KINDS.names()


def test_the_packaged_template_is_a_valid_config(tmp_path: pathlib.Path) -> None:
    """`pipeline template` output must be usable without hand-editing to make it parse."""
    path = tmp_path / "template.yaml"
    path.write_text(EMBEDDINGS_KIND.template_yaml(), encoding="utf-8")

    resolved = resolve_config(path)

    assert resolved.kind == "embeddings"
    assert resolved.modalities == (Modality.TEXT, Modality.IMAGE, Modality.ACTION)


def test_the_template_payload_agrees_with_the_template_yaml() -> None:
    """The JSON and YAML template surfaces must not drift apart."""
    payload = EMBEDDINGS_KIND.template_payload()

    assert payload["kind"] == "embeddings"
    assert payload["config"] == yaml.safe_load(EMBEDDINGS_KIND.template_yaml())


def test_validate_accepts_a_good_config(tmp_path: pathlib.Path) -> None:
    """`pipeline validate` is the operator's pre-flight check."""
    assert EMBEDDINGS_KIND.validate(_write_config(tmp_path), []) == {"ok": True}


def test_validate_rejects_an_unknown_field(tmp_path: pathlib.Path) -> None:
    """A misspelled or removed key fails the pre-flight rather than being ignored."""
    path = _write_config(tmp_path, source_lance_uri=DEFAULT_CLIPS_URI)

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        EMBEDDINGS_KIND.validate(path, [])


def test_validate_rejects_a_config_routed_to_this_kind_under_another_discriminator(
    tmp_path: pathlib.Path,
) -> None:
    """The adapter enforces its own discriminator, not just the registry's dispatch."""
    path = tmp_path / "embeddings.yaml"
    path.write_text(
        yaml.safe_dump({"schema_version": 1, "kind": "video-split", "clips_lance_uri": DEFAULT_CLIPS_URI}),
        encoding="utf-8",
    )

    # Matching only "kind" would pass even with the field deleted, since
    # extra="forbid" names the unknown key; the Literal message is what proves
    # the discriminator was enforced.
    with pytest.raises(ValidationError, match="Input should be 'embeddings'"):
        EMBEDDINGS_KIND.validate(path, [])


def test_validate_rejects_an_empty_modality_list(tmp_path: pathlib.Path) -> None:
    """A config that enables nothing is a config error, not a zero-work run."""
    path = _write_config(tmp_path, modalities=[])

    with pytest.raises(ValidationError, match="must not be empty"):
        EMBEDDINGS_KIND.validate(path, [])


def test_validate_applies_an_override_before_the_cross_field_rules(tmp_path: pathlib.Path) -> None:
    """An override reaches validation, so ``--set`` cannot smuggle past a cross-field rule.

    ``image.batch_size`` is left at its default and only the read width is pushed
    past it, so the cross-field rule is the sole constraint violated; match= is
    what proves that rule fired rather than the key becoming unknown.
    """
    path = _write_config(tmp_path)

    with pytest.raises(ValidationError, match="must not exceed batch_size"):
        EMBEDDINGS_KIND.validate(path, ["image.read_concurrency=1000"])


def test_render_emits_the_resolved_config(tmp_path: pathlib.Path) -> None:
    """Render shows the values that will actually execute, defaults included."""
    path = _write_config(tmp_path, max_fragments=2)

    rendered = json.loads(EMBEDDINGS_KIND.render(path, []))

    assert rendered == resolve_config(path).model_dump(mode="json")


def test_the_schema_documents_the_required_fields_and_the_modality_blocks() -> None:
    """`pipeline schema` is how an operator discovers what a config may hold."""
    schema = json.loads(EMBEDDINGS_KIND.schema_json())

    assert set(schema["required"]) == {"schema_version", "kind", "clips_lance_uri"}
    assert {"TextEmbeddingConfig", "ImageEmbeddingConfig", "ActionEmbeddingConfig"} <= set(schema["$defs"])


def test_the_kind_publishes_no_presets() -> None:
    """Every modality selection is one config line or one override, so there is nothing to preset."""
    assert EMBEDDINGS_KIND.list_presets() == []


def test_a_run_reports_the_recipe_summary_as_json_and_one_message(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The adapter projects the recipe's own counts; it adds no reporting of its own."""
    action = _synthetic_action_pca()
    seen = _stub_run(
        monkeypatch,
        _result(
            _modality_result(
                Modality.TEXT,
                selected=4,
                filled=4,
                version=6,
                result_identity={"embedding_text_model_id": "text-model"},
            ),
            _modality_result(
                Modality.IMAGE,
                selected=4,
                filled=3,
                version=7,
                result_identity={"embedding_image_model_id": "image-model"},
            ),
            action=action,
        ),
    )
    path = _write_config(tmp_path, modalities=["text", "image"])

    output = EMBEDDINGS_KIND.prepare_run(path, set_overrides=[])()

    assert [config.clips_lance_uri for config in seen] == [DEFAULT_CLIPS_URI]
    assert output.json_payload == {
        "clips_lance_uri": DEFAULT_CLIPS_URI,
        "ending_version": 7,
        "modalities": [
            {
                "modality": "text",
                "selected": 4,
                "filled": 4,
                "failed": 0,
                "committed_version": 6,
                "result_identity": {"embedding_text_model_id": "text-model"},
            },
            {
                "modality": "image",
                "selected": 4,
                "filled": 3,
                "failed": 1,
                "committed_version": 7,
                # Reported per modality rather than once per run: the two name
                # different computations, so a single run-level identity could not
                # say which group a model change took effect in.
                "result_identity": {"embedding_image_model_id": "image-model"},
            },
        ],
        "action_pca": {"fingerprint": action.fingerprint, "samples_used": None},
    }
    assert output.message == "embeddings: text 4/4, image 3/4; clips.lance at v7"


def test_a_run_that_committed_nothing_says_the_table_is_unchanged(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Report a run that owed nothing as unchanged rather than as a null version.

    A run over an already-complete table is the common case, so the message has to
    distinguish it from a commit rather than printing a null version.
    """
    _stub_run(monkeypatch, _result(_modality_result(Modality.ACTION, selected=0, filled=0, version=None)))
    path = _write_config(tmp_path, modalities=["action"])

    output = EMBEDDINGS_KIND.prepare_run(path, set_overrides=[])()

    assert output.json_payload["ending_version"] is None
    assert output.json_payload["action_pca"] is None
    assert output.message == "embeddings: action 0/0; clips.lance unchanged"


def test_a_run_whose_only_commit_added_columns_says_so_instead_of_unchanged(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A schema-only commit is reported as a version, with the fill's absence named.

    "unchanged" would be false about the table and would send a reader to a
    version that predates the columns; a bare version would suggest a fill landed.
    The counts alone cannot separate the two, since a run that filled nothing and
    a run that committed nothing both read ``0/0``.
    """
    _stub_run(
        monkeypatch,
        _result(_modality_result(Modality.ACTION, selected=0, filled=0, version=None), schema_commit_version=4),
    )
    path = _write_config(tmp_path, modalities=["action"])

    output = EMBEDDINGS_KIND.prepare_run(path, set_overrides=[])()

    # The payload keeps the two apart on its own: a run-level version with every
    # per-modality version null IS the schema-only commit.
    assert output.json_payload["ending_version"] == 4
    assert output.message == "embeddings: action 0/0; clips.lance at v4 (columns added, no fill committed)"


def test_a_widening_run_where_one_modality_filled_reports_the_fill_not_the_columns(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A first run of two modalities reports the fill's version, not the widening's.

    Widening and filling in the same run is the ordinary first run of a group, and
    both of its commits are real. Reporting the earlier one would name a version
    that predates the vectors, and appending "no fill committed" would deny a fill
    that a reader can see in the counts.
    """
    _stub_run(
        monkeypatch,
        _result(
            _modality_result(Modality.TEXT, selected=4, filled=4, version=5),
            _modality_result(Modality.ACTION, selected=0, filled=0, version=None),
            schema_commit_version=4,
        ),
    )
    path = _write_config(tmp_path, modalities=["text", "action"])

    output = EMBEDDINGS_KIND.prepare_run(path, set_overrides=[])()

    assert output.json_payload["ending_version"] == 5
    assert output.message == "embeddings: text 4/4, action 0/0; clips.lance at v5"


def test_a_failed_row_is_reported_rather_than_raised(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A row left NULL by a per-row failure is a returned count, not an exit status.

    The next run's own owed-work predicate re-selects that row, so the remedy is
    a re-run rather than a red exit -- and a red exit would make a corpus holding
    one permanently broken clip fail forever with nothing an operator could do.
    Structural failures are the opposite: every one of them raises before its own
    commit, so nothing partial reaches this adapter to classify.
    """
    _stub_run(monkeypatch, _result(_modality_result(Modality.IMAGE, selected=4, filled=3, version=7)))
    path = _write_config(tmp_path, modalities=["image"])

    output = EMBEDDINGS_KIND.prepare_run(path, set_overrides=[])()

    # Asserted so a fixture edit cannot make the test vacuous by removing the
    # failed row it exists to tolerate.
    assert output.json_payload["modalities"] == [
        {
            "modality": "image",
            "selected": 4,
            "filled": 3,
            "failed": 1,
            "committed_version": 7,
            "result_identity": {},
        }
    ]
    assert output.message == "embeddings: image 3/4; clips.lance at v7"


def test_overrides_reach_the_run_and_not_merely_validation(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--set`` must change what executes, so the resolved object carries the override."""
    seen = _stub_run(monkeypatch, _result(_modality_result(Modality.TEXT, selected=0, filled=0, version=None)))
    path = _write_config(tmp_path)

    EMBEDDINGS_KIND.prepare_run(path, set_overrides=["modalities=[text]", "max_fragments=2"])()

    assert [(config.modalities, config.max_fragments) for config in seen] == [((Modality.TEXT,), 2)]


def test_preparing_a_run_resolves_the_config_once_and_defers_the_work(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Resolution is eager so a bad config fails before Ray starts; the file is never reread.

    Calling the prepared closure twice must not reparse: an operator editing the
    config mid-run would otherwise get two different executions from one
    invocation.
    """
    resolutions: list[pathlib.Path] = []
    real_resolve = config_module.resolve_config

    def _counting_resolve(config_path: str | pathlib.Path, *, overrides: Sequence[str] = ()) -> EmbeddingPipelineConfig:
        resolutions.append(pathlib.Path(config_path))
        return real_resolve(config_path, overrides=overrides)

    monkeypatch.setattr(config_module, "resolve_config", _counting_resolve)
    runs = _stub_run(monkeypatch, _result(_modality_result(Modality.TEXT, selected=0, filled=0, version=None)))
    path = _write_config(tmp_path)

    prepared = EMBEDDINGS_KIND.prepare_run(path, set_overrides=[])
    assert resolutions == [path]
    assert runs == []

    prepared()
    prepared()

    assert resolutions == [path]
    assert len(runs) == 2


def test_a_bad_config_fails_at_prepare_time_rather_than_at_run_time(tmp_path: pathlib.Path) -> None:
    """Eager resolution is what lets the runtime report a config error as ``invalid``.

    A config failure discovered inside the returned closure would be reported as a
    runtime failure instead, so the split is a contract rather than an
    optimization.
    """
    path = _write_config(tmp_path, max_fragments=0)

    with pytest.raises(ValidationError, match="greater than or equal to 1"):
        EMBEDDINGS_KIND.prepare_run(path, set_overrides=[])


def test_a_mid_run_failure_escapes_the_closure_unwrapped(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The closure catches nothing, which is the other half of the invalid/runtime split.

    A closure that swallowed a mid-run failure into a degraded result would
    surface as a successful run, so the exception escaping is the contract - and
    it must escape with its own type, since a wrapper would erase what failed.
    """

    def _boom(_config: EmbeddingPipelineConfig) -> EmbeddingRunResult:
        msg = "fill failed on fragment 3"
        raise RuntimeError(msg)

    monkeypatch.setattr(embed, "run_embedding_pipeline", _boom)
    prepared = EMBEDDINGS_KIND.prepare_run(_write_config(tmp_path), set_overrides=[])

    with pytest.raises(RuntimeError, match="fragment 3"):
        prepared()


def test_a_config_driven_run_widens_the_table_it_names(
    tmp_path: pathlib.Path, make_clips_table: ClipsTableFactory
) -> None:
    """A run reaches real storage: the group's columns and the version it reports are the table's.

    No clip carries an action artifact, so the leg is skipped without an actor
    pool - which is what lets this prove the config-file-to-Lance-commit chain
    without Ray or a model.
    """
    clips_uri = make_clips_table(rows=2, action_uris=["", ""])
    path = _write_config(tmp_path, clips_lance_uri=clips_uri, modalities=["action"])

    output = EMBEDDINGS_KIND.prepare_run(path, set_overrides=[])()

    assert "embedding_action" in set(lance.dataset(clips_uri).schema.names)
    assert output.json_payload["clips_lance_uri"] == clips_uri
    assert output.json_payload["action_pca"] is None
    # Against a real table rather than a stub: the version reported is the one an
    # operator can open, which is the claim the stubbed message tests cannot make.
    # Paired with the null per-modality version, which is what makes the reported
    # one the widening's rather than a fill's.
    assert output.json_payload["modalities"][0]["committed_version"] is None
    assert output.json_payload["ending_version"] == lance.dataset(clips_uri).version


@pytest.mark.integration
@pytest.mark.usefixtures("ray_local")
def test_a_config_driven_run_fills_the_enabled_modality(
    tmp_path: pathlib.Path,
    make_clips_table: ClipsTableFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One config file drives real vectors into the table it names.

    The embedder is stubbed and the actor runs in this interpreter, so the test
    covers the wiring rather than a model: config file -> resolve -> prepare_run
    -> the recipe's widen, fill, and commit -> a non-NULL column group.
    """
    monkeypatch.setattr(
        "cosmos_curator.next.recipes.embeddings.fill.ray_data_gpu_runtime_env",
        lambda _env_name: ray_data_gpu_runtime_env(""),
    )
    monkeypatch.setattr(embed, "_GENERIC_FILL_BUILDERS", ((Modality.IMAGE, _stub_image_fill),))
    clips_uri = make_clips_table(rows=4, rows_per_file=2)
    path = _write_config(tmp_path, clips_lance_uri=clips_uri, modalities=["image"])

    output = EMBEDDINGS_KIND.prepare_run(path, set_overrides=[])()

    vectors = lance.dataset(clips_uri).to_table(columns=["embedding_image"]).column("embedding_image").to_pylist()
    assert all(vector is not None for vector in vectors)
    assert output.json_payload["modalities"] == [
        {
            "modality": "image",
            "selected": 4,
            "filled": 4,
            "failed": 0,
            "committed_version": lance.dataset(clips_uri).version,
            # The stub's own model id, reached through the whole chain: this is
            # the identity the published rows now carry.
            "result_identity": {"embedding_image_model_id": _STUB_MODEL_ID},
        }
    ]
