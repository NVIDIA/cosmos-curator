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

"""Config-backed pipeline-kind surface for the ``embeddings`` recipe.

One resolver serves every operation that reads a config file, so ``validate``,
``render``, and ``run-pipeline`` cannot disagree about what one means. ``schema``
takes no config and is published by the model itself::

    config.yaml --> resolve_config --> EmbeddingPipelineConfig
                                            |
              +------------+----------------+
              v            v                v
           validate      render         prepare_run
                                             |
                                             v
                                   run_embedding_pipeline

    (schema) ------------------> EmbeddingPipelineConfig.model_json_schema()

Every import that a mere ``--help`` does not need stays inside a callback. The
composition root imports this module at CLI startup, and the embedding runtime
reaches torch, transformers, sentence_transformers, lance, and ray, so a
module-level import here would put all of them on the startup path.

The adapter catches nothing and raises nothing of its own. That is what lets the
generic runtime tell a config fault from a run fault by where it arose: the
former surfaces from ``prepare_run``, the latter from the returned closure. It
also needs no partial-success channel, because every way a modality can fail to
publish raises before its own commit, so a returned result always describes work
that is canonical. How a raise is labelled and what it exits with belongs to
``pipeline_runtime`` - see that module rather than a paraphrase here.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from cosmos_curator.next.core.pipeline_kind import PipelineKind, PipelinePreset, PipelineRunOutput, PreparedPipelineRun

if TYPE_CHECKING:
    from cosmos_curator.next.recipes.embeddings.embed import EmbeddingRunResult


def _template_yaml() -> str:
    return """\
# Embeddings are written back into the SAME clips.lance this reads: a per-modality
# embedding_<modality>_* column group is added and filled in place. There are no
# side tables and no per-modality output URIs.
# All other settings and their defaults: cosmos-curator pipeline schema embeddings
schema_version: 1
kind: embeddings

# Read source AND write target. Base rows are appended upstream by robot-action-split.
clips_lance_uri: s3://example-bucket/robot_clips/lance/clips.lance

# Credential profile for the table, the clip media, and the action artifacts.
storage_profile: default

# Weights base for download_models. The packaged default is a placeholder that
# must be replaced before the text or image modality can load its checkpoint.
model_weights_path: s3://example-bucket/model_weights/

# Remove a modality to skip it. Action applies only to clips carrying an
# action_data_uri (Mecka ACT2); with none present it is skipped, not failed.
modalities: [text, image, action]

# Where a crashed run's staged rows and checkpoints live, so a rerun resumes
# instead of recomputing. Leave it out unless the staged rows belong on different
# storage from the table: the default is a sibling directory of clips_lance_uri
# suffixed __embedding_staging, which needs no configuration to work.
# staging_root_uri: s3://example-bucket/robot_clips/embedding_staging
"""


def _template_payload() -> dict[str, Any]:
    import yaml  # noqa: PLC0415

    return {
        "kind": "embeddings",
        "description": "Embed clips in place, adding one embedding column group per enabled modality.",
        "required_fields": [
            {"path": "schema_version", "example": 1},
            {"path": "kind", "example": "embeddings"},
            {"path": "clips_lance_uri", "example": "s3://example-bucket/robot_clips/lance/clips.lance"},
            # Defaulted in the model, but its default is a placeholder bucket, so a
            # config that leaves it alone cannot load a checkpoint.
            {"path": "model_weights_path", "example": "s3://example-bucket/model_weights/"},
        ],
        "config": yaml.safe_load(_template_yaml()),
    }


def _validate(config: Path, overrides: Sequence[str]) -> dict[str, object]:
    from cosmos_curator.next.recipes.embeddings.config import resolve_config  # noqa: PLC0415

    resolve_config(config, overrides=list(overrides))
    return {"ok": True}


def _render(config: Path, overrides: Sequence[str]) -> str:
    import json  # noqa: PLC0415

    from cosmos_curator.next.recipes.embeddings.config import resolve_config  # noqa: PLC0415

    resolved = resolve_config(config, overrides=list(overrides))
    return json.dumps(resolved.model_dump(mode="json"), indent=2) + "\n"


def _schema_json() -> str:
    import json  # noqa: PLC0415

    from cosmos_curator.next.recipes.embeddings.config import EmbeddingPipelineConfig  # noqa: PLC0415

    return json.dumps(EmbeddingPipelineConfig.model_json_schema(), indent=2) + "\n"


def _list_presets() -> list[PipelinePreset]:
    return []


def _run_payload(result: "EmbeddingRunResult", *, clips_lance_uri: str) -> dict[str, object]:
    """Project one run's summary onto the machine-readable ``--json`` payload.

    Assembled here rather than in the recipe so the reported shape is owned by
    the CLI surface that promises it, and the recipe keeps returning its own
    verification summary unchanged.

    ``result_identity`` is reported because it is the only fact here that answers
    "did my model change take effect": a rebuilt group and an untouched one
    differ in the identity their rows now carry, not in the counts.

    A non-null ``ending_version`` with every ``committed_version`` null is the
    schema widening: the columns were added and no fill followed them.
    """
    return {
        "clips_lance_uri": clips_lance_uri,
        "ending_version": result.ending_version,
        "modalities": [
            {
                "modality": modality.modality.value,
                "selected": modality.selected,
                "filled": modality.filled,
                "failed": modality.failed,
                "committed_version": modality.committed_version,
                "result_identity": modality.result_identity,
            }
            for modality in result.modalities
        ],
        "action_pca": (
            None
            if result.action is None
            else {"fingerprint": result.action.fingerprint, "samples_used": result.action.samples_used}
        ),
    }


def _version_phrase(result: "EmbeddingRunResult") -> str:
    """Describe what the run left on the table: nothing, a fill, or only the columns.

    "unchanged" is reserved for a run that committed nothing at all, widening
    included, so a version named here always describes the table the reader was
    given.
    """
    if result.ending_version is None:
        return "clips.lance unchanged"
    if result.committed_only_the_widening:
        return f"clips.lance at v{result.ending_version} (columns added, no fill committed)"
    return f"clips.lance at v{result.ending_version}"


def _run_message(result: "EmbeddingRunResult") -> str:
    """Summarize one run in a single line: filled/selected per modality, then the table version.

    The counts describe only what this run published, never how much of the group
    is complete: a top-up that embedded four new clips onto a million filled rows
    reads ``4/4``. Completeness is not claimed here because nothing claims it -
    the owed-work predicate is re-evaluated every run.
    """
    per_modality = ", ".join(
        f"{modality.modality.value} {modality.filled}/{modality.selected}" for modality in result.modalities
    )
    return f"embeddings: {per_modality}; {_version_phrase(result)}"


def _prepare_run(
    config: Path,
    *,
    set_overrides: list[str],
) -> PreparedPipelineRun:
    from cosmos_curator.next.recipes.embeddings.config import resolve_config  # noqa: PLC0415

    # Resolved eagerly, so a bad config fails as "invalid" before the runtime
    # starts Ray, and the closure never reparses the file.
    resolved = resolve_config(config, overrides=set_overrides)

    def run() -> PipelineRunOutput:
        from cosmos_curator.next.recipes.embeddings.embed import run_embedding_pipeline  # noqa: PLC0415

        result = run_embedding_pipeline(resolved)
        # Returned unconditionally: a modality either published its owed rows or
        # raised, so there is no partial outcome for this adapter to turn into an
        # exit status. A row counted in ``failed`` WAS attempted and left NULL,
        # which the next run's own predicate re-selects, so it is reported rather
        # than refused.
        return PipelineRunOutput(
            json_payload=_run_payload(result, clips_lance_uri=resolved.clips_lance_uri),
            message=_run_message(result),
        )

    return run


EMBEDDINGS_KIND = PipelineKind(
    name="embeddings",
    template_yaml=_template_yaml,
    template_payload=_template_payload,
    validate=_validate,
    render=_render,
    schema_json=_schema_json,
    list_presets=_list_presets,
    prepare_run=_prepare_run,
)
