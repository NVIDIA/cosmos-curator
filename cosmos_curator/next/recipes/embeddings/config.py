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

"""Typed config for the embedding recipe (pydantic; no business logic).

Embedding-specific knobs only: which modalities run, per-modality resources, the
shared ``clips.lance`` table (both the read source and the write target), the
recovery staging settings, and the dev fragment cap. There is no output-table or
side-table URI: the recipe updates ``clips.lance`` in place, and both the
action-PCA artifact directory and the recovery staging root default to locations
derived from the table. Validation is strict (unknown keys rejected, no silent
coercion) so a malformed config fails at assembly on the driver rather than deep
inside a Ray Data actor.

``resolve_config`` is the single entry point every config-driven surface goes
through - ``pipeline validate``, ``pipeline render``, and ``run-pipeline`` all
resolve here - so one file plus its ``--set`` overrides can never mean two
different things depending on which command read it.

Every modality's ``batch_size`` is its WORKER'S FRAGMENT-SCAN batch size - how
many rows of one Lance fragment reach the embedder per call. It is not a Ray Data
work-item size: a work item is always exactly one fragment.
"""

import enum
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Any, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator, model_validator

from cosmos_curator.core.utils.environment import MODEL_WEIGHTS_PREFIX
from cosmos_curator.next.core.config import apply_dotted_overrides
from cosmos_curator.next.embeddings.schemas import ACTION_DIM

_MODEL_CONFIG = ConfigDict(frozen=True, strict=True, extra="forbid")
_YAML_SUFFIXES = frozenset({".yaml", ".yml"})
_JSON_SUFFIXES = frozenset({".json"})

SchemaVersion = Literal[1]
EmbeddingsKind = Literal["embeddings"]


class Modality(enum.StrEnum):
    """One embedding modality."""

    TEXT = "text"
    IMAGE = "image"
    ACTION = "action"


# Lower bound is one more than the action embedding width: mean-centering costs a
# degree of freedom, so an ``ACTION_DIM``-row sample fits a rank-(ACTION_DIM - 1)
# basis whose last component is null-space noise. Derived from ``ACTION_DIM`` (not
# a hardcoded 98) so a change to the action width cannot leave the config
# accepting a now-invalid ``pca_sample_size`` that only fails deep inside the fit;
# the import-layer test proves ``schemas`` is pure (numpy + pyarrow), so importing
# it here pulls in no capability module.
_MIN_PCA_SAMPLE_SIZE = ACTION_DIM + 1


NonBlankStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
"""A string that is stored TRIMMED and may not be empty or whitespace-only.

Surrounding whitespace is stripped before the value is stored, so a padded
``" s3://bucket/x.lance "`` in the YAML is kept as ``"s3://bucket/x.lance"`` and a
stored URI can differ from the text it was configured with. Stripping first is
also what rejects a whitespace-only value: bare ``min_length=1`` accepts ``"   "``,
and such a URI would resolve against the process working directory instead of
failing.
"""


class TextEmbeddingConfig(BaseModel):
    """Resources for the text-embedding modality (small model; fractional GPU or CPU).

    Unlike the image modality, the text modality declares no ``memory_gb``: its
    per-actor heap is dominated by the model weights, not by a decoded-frame batch,
    so it relies on Ray Data's default logical footprint. A hand-tuned figure with
    no measurement behind it would be guesswork, so the asymmetry is intentional.
    """

    model_config = _MODEL_CONFIG

    num_gpus: float = Field(default=0.25, ge=0.0, description="GPUs per text embedder actor (0.0 = CPU).")
    batch_size: int = Field(default=256, ge=1, description="Rows per fragment-scan batch handed to the text embedder.")


class ImageEmbeddingConfig(BaseModel):
    """Resources for the image-embedding modality (concurrent media reads + a GPU model).

    None of these fields is independent of the others: the read width bounds the
    clips in flight that ``memory_gb`` must cover, and it deliberately does NOT
    bound ``num_cpus``, because the decode those reads feed is GIL-serialized.
    The argument behind that, and the measurements behind every default here, are
    in ``docs/curator/design/curator-next-embeddings.md`` section 4.2.
    """

    model_config = _MODEL_CONFIG

    num_gpus: float = Field(default=1.0, ge=0.0, description="GPUs per image embedder actor (0.0 = CPU).")
    batch_size: int = Field(default=64, ge=1, description="Rows per fragment-scan batch handed to the image embedder.")
    read_concurrency: int = Field(
        default=32,
        ge=1,
        description="Clips fetched and decoded concurrently inside ONE image actor; it widens the download only. "
        "May not exceed batch_size, which the reader draws its clips from. See "
        "docs/curator/design/curator-next-embeddings.md section 4.2.",
    )
    num_cpus: float = Field(
        default=2.0,
        gt=0.0,
        description="CPUs reserved per image actor; a fixed budget that does NOT scale with read_concurrency. No "
        "zero: an unreserved actor is the oversubscription this field prevents. See "
        "docs/curator/design/curator-next-embeddings.md section 4.2.",
    )
    memory_gb: float = Field(
        default=8.0,
        gt=0.0,
        description="Heap reserved per image actor (Ray memory=), covering the clips in flight, the decoded frames "
        "of one backbone call, and the computed vectors in flight to the column write. See "
        "docs/curator/design/curator-next-embeddings.md section 4.2.",
    )

    @model_validator(mode="after")
    def _check_read_width_fits_batch(self) -> Self:
        """Reject a read width the scan batch can never fill.

        A wider value is clamped at run time rather than honoured, so the run
        would read at a width the config does not report.

        Raises:
            ValueError: If ``read_concurrency`` exceeds ``batch_size``.

        """
        if self.read_concurrency > self.batch_size:
            msg = (
                f"read_concurrency ({self.read_concurrency}) must not exceed batch_size ({self.batch_size}): "
                f"the reader draws its clips from one scan batch, so the excess workers would never get work"
            )
            raise ValueError(msg)
        return self


class ActionEmbeddingConfig(BaseModel):
    """Resources for the CPU wrist-motion action-embedding modality plus PCA sampling.

    The read width and the CPU reservation are deliberately independent: the width
    overlaps the artifact fetch, which is network wait, while the geometry those
    fetches feed is GIL-serialized inside one actor. The reasoning and the
    measurements behind the defaults are in
    ``docs/curator/design/curator-next-embeddings.md`` section 5.6.
    """

    model_config = _MODEL_CONFIG

    num_cpus: float = Field(default=1.0, gt=0.0, description="CPUs per action embedder actor.")
    batch_size: int = Field(
        default=256, ge=1, description="Rows per fragment-scan batch handed to the action embedder."
    )
    read_concurrency: int = Field(
        default=32,
        ge=1,
        description="Action artifacts fetched concurrently inside ONE action actor; it widens the network wait "
        "only. The wrist geometry those fetches feed is GIL-serialized, so an actor's throughput ceiling does NOT "
        "move with this; dividing that ceiling takes more actors, and what bounds the actor count differs by "
        "consumer of this field: a fill's work item is a Lance fragment, so its pool is the table's fragment "
        "count, while the PCA-sampling pass builds work items from a URI list and is bounded instead by the CPUs "
        "allocated. May not exceed batch_size, which the reader draws its artifacts from. See "
        "docs/curator/design/curator-next-embeddings.md section 5.6.",
    )
    pca_sample_size: int = Field(
        default=50_000,
        ge=_MIN_PCA_SAMPLE_SIZE,
        description="Valid descriptors sampled to fit the action PCA basis (>= ACTION_DIM + 1). This bounds the "
        "configured CAP, not the achievable count: nothing here checks how many descriptors actually survive the "
        "validity gates, which is why a short run still fails later inside fit_action_pca.",
    )

    @model_validator(mode="after")
    def _check_read_width_fits_batch(self) -> Self:
        """Reject a read width the scan batch can never fill.

        A wider value is clamped at run time rather than honoured, so the run
        would read at a width the config does not report.

        Raises:
            ValueError: If ``read_concurrency`` exceeds ``batch_size``.

        """
        if self.read_concurrency > self.batch_size:
            msg = (
                f"read_concurrency ({self.read_concurrency}) must not exceed batch_size ({self.batch_size}): "
                f"the reader draws its artifacts from one scan batch, so the excess workers would never get work"
            )
            raise ValueError(msg)
        return self


class EmbeddingPipelineConfig(BaseModel):
    """Fully resolved config for one embedding run.

    Beyond the version gate and the discriminator, only ``clips_lance_uri`` is
    required; everything else defaults. Which modalities run is decided by
    ``modalities``, so a single modality can be embedded on its own. There is
    deliberately no output-table or side-table URI: the recipe updates
    ``clips.lance`` in place. The action-PCA artifact directory and the recovery
    staging root are both DERIVED from the table by default, so a run recovers
    from a crash without being configured to; ``staging_root_uri`` exists only to
    put the transient staged rows on different storage from the table.
    """

    model_config = _MODEL_CONFIG

    schema_version: SchemaVersion = Field(
        description="Config schema generation. Pinned rather than defaulted so a config written against a future "
        "generation is rejected instead of being read under this one's field meanings.",
    )
    kind: EmbeddingsKind = Field(
        description="Pipeline discriminator the generic CLI routes on; it must match the registered kind name.",
    )
    clips_lance_uri: NonBlankStr = Field(
        description="URI of the shared clips Lance table. It is BOTH read source and write target: base rows are "
        "appended upstream by robot_action_split and every modality's embedding columns are updated in place here.",
    )
    storage_profile: NonBlankStr = Field(
        default="default",
        description="Profile for the clips table, clip media, and action artifacts (one profile, as written).",
    )
    modalities: tuple[Modality, ...] = Field(
        default=(Modality.TEXT, Modality.IMAGE, Modality.ACTION),
        description="Modalities to embed; each runs independently. De-duplicated, first-seen order recorded. "
        "The recorded order does NOT drive execution: run_embedding_pipeline runs a fixed text->image->action "
        "cascade, so the tuple only records which modalities are enabled.",
    )
    text: TextEmbeddingConfig = Field(default_factory=TextEmbeddingConfig)
    image: ImageEmbeddingConfig = Field(default_factory=ImageEmbeddingConfig)
    action: ActionEmbeddingConfig = Field(default_factory=ActionEmbeddingConfig)
    model_weights_path: NonBlankStr = Field(
        default=MODEL_WEIGHTS_PREFIX,
        description="Weights base passed to download_models; the default is a placeholder bucket that must be "
        "overridden for a run against pre-staged weights.",
    )
    staging_root_uri: NonBlankStr | None = Field(
        default=None,
        description="Root for the transient checkpoint and staged-row directories a crashed run resumes from. "
        "Defaults to a sibling of clips_lance_uri suffixed __embedding_staging, so recovery needs no configuration; "
        "set it to put the staged rows on different storage from the table.",
    )
    parquet_rows_per_file: int = Field(
        default=4_096,
        ge=1,
        description="Rows per staged Parquet file, requested of Ray's sink as a hint rather than a guarantee. One "
        "file is one write bundle, so with the modality's batch_size it sets the recovery granularity: what a crash "
        "re-computes is bounded by the LARGER of the two, and raising a batch_size above this value makes this knob "
        "the smaller one and so no longer the bound.",
    )
    max_fragments: int | None = Field(
        default=None,
        ge=1,
        description="Cap the number of Lance fragments each modality visits, for a smoke test against an existing "
        "table. The cap is fragment-granular rather than row-granular because the unit of work IS a fragment: a row "
        "cap could not be honoured across independent workers without a shared counter. It bounds the fill only: the "
        "action leg's PCA candidate scan reads the whole table either way, so a capped run fits the same basis an "
        "uncapped one would.",
    )

    @field_validator("modalities", mode="before")
    @classmethod
    def _coerce_dedup_non_empty(cls, value: object) -> tuple[Modality, ...]:
        """Coerce strings to ``Modality``, de-duplicate (first-seen), reject empty.

        Runs ``mode="before"`` because the model is strict: a raw ``["text",
        "image"]`` from YAML/JSON would otherwise fail item validation against the
        ``Modality`` enum. Coercing here both accepts the string form and lets
        ``dict.fromkeys`` de-duplicate before the tuple is validated.

        Raises:
            ValueError: If ``value`` is not a list/tuple, names an unknown
                modality, or is empty.

        """
        if not isinstance(value, (list, tuple)):
            # ValueError (not TypeError) so pydantic wraps it into a ValidationError;
            # a raw TypeError would escape the model instead of failing validation.
            msg = f"modalities must be a list or tuple of modality names, got {type(value).__name__}"
            raise ValueError(msg)  # noqa: TRY004
        try:
            members = [Modality(item) for item in value]
        except ValueError as exc:
            valid = ", ".join(modality.value for modality in Modality)
            msg = f"unknown modality in {list(value)}; valid modalities are: {valid}"
            raise ValueError(msg) from exc
        deduped = tuple(dict.fromkeys(members))
        if not deduped:
            msg = "modalities must not be empty"
            raise ValueError(msg)
        return deduped


def _load_config_data(config_path: Path) -> dict[str, Any]:
    """Read one YAML or JSON config file into a mapping.

    Args:
        config_path: File to read; the suffix selects the parser.

    Returns:
        The file's top-level mapping, unvalidated.

    Raises:
        FileNotFoundError: If the path does not exist or is not a regular file
            (for example a directory).
        ValueError: If the suffix is neither a YAML nor a JSON extension.
        TypeError: If the file's top level is not a mapping.

    """
    if not config_path.exists():
        msg = f"Config file not found: {config_path}"
        raise FileNotFoundError(msg)
    if not config_path.is_file():
        msg = f"Config path is not a regular file: {config_path}"
        raise FileNotFoundError(msg)
    suffix = config_path.suffix.lower()
    if suffix not in _YAML_SUFFIXES and suffix not in _JSON_SUFFIXES:
        supported = ", ".join(sorted(_YAML_SUFFIXES | _JSON_SUFFIXES))
        msg = (
            f"Unsupported config extension {config_path.suffix!r}: {config_path}. Supported extensions are: {supported}"
        )
        raise ValueError(msg)
    with config_path.open(encoding="utf-8") as handle:
        loaded: object = yaml.safe_load(handle) if suffix in _YAML_SUFFIXES else json.load(handle)
    if not isinstance(loaded, dict):
        msg = f"Config file must contain a mapping at the top level, got {type(loaded).__name__}: {config_path}"
        raise TypeError(msg)
    return loaded


def resolve_config(
    config_path: str | Path,
    *,
    overrides: Sequence[str] = (),
) -> EmbeddingPipelineConfig:
    """Load a config file, apply ``--set`` overrides, and validate the result.

    Overrides are applied to the loaded mapping BEFORE validation, so an
    overridden value is checked by the same rules as a written one. Each value is
    parsed with ``yaml.safe_load``, which matters here because the model is
    strict: a quoted ``"2"`` would be rejected for ``max_fragments``.

    Args:
        config_path: YAML or JSON config file.
        overrides: ``PATH=VALUE`` assignments, e.g. ``["max_fragments=2"]``.

    Returns:
        The validated run configuration.

    Raises:
        FileNotFoundError: If the config file does not exist.
        TypeError: If the file's top level is not a mapping, or an override path
            passes through a non-mapping key.
        ValueError: If the config extension is unsupported, an override is
            malformed, or validation fails (``ValidationError`` is a
            ``ValueError``).

    """
    data = _load_config_data(Path(config_path))
    apply_dotted_overrides(data, overrides)
    return EmbeddingPipelineConfig.model_validate(data)
