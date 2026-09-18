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

"""Validation and file-resolution tests for the direct-``clips.lance`` embedding-recipe config."""

import re
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from cosmos_curator.next.embeddings.schemas import ACTION_DIM
from cosmos_curator.next.recipes.embeddings.config import _MIN_PCA_SAMPLE_SIZE, Modality, resolve_config

from .conftest import DEFAULT_CLIPS_URI, EmbeddingsConfigFactory


def _write_config(root: Path, **fields: object) -> Path:
    """Write a minimal valid config file, plus any extra top-level fields."""
    path = root / "config.yaml"
    payload: dict[str, object] = {
        "schema_version": 1,
        "kind": "embeddings",
        "clips_lance_uri": DEFAULT_CLIPS_URI,
        **fields,
    }
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def test_minimal_config_only_needs_clips_uri_and_defaults_all_modalities(
    make_embeddings_config: EmbeddingsConfigFactory,
) -> None:
    """Beyond the version gate and discriminator, only ``clips_lance_uri`` is required."""
    cfg = make_embeddings_config()
    assert cfg.modalities == (Modality.TEXT, Modality.IMAGE, Modality.ACTION)
    assert cfg.max_fragments is None


def test_modalities_are_deduplicated_order_preserved(make_embeddings_config: EmbeddingsConfigFactory) -> None:
    """Duplicate modalities collapse, keeping first-seen order."""
    cfg = make_embeddings_config(modalities=["image", "text", "image"])
    assert cfg.modalities == (Modality.IMAGE, Modality.TEXT)


def test_empty_modalities_rejected(make_embeddings_config: EmbeddingsConfigFactory) -> None:
    """An empty modality set is rejected - there would be nothing to run."""
    with pytest.raises(ValidationError, match="must not be empty"):
        make_embeddings_config(modalities=[])


def test_invalid_modality_rejected(make_embeddings_config: EmbeddingsConfigFactory) -> None:
    """A modality outside text/image/action is rejected by the StrEnum."""
    with pytest.raises(ValidationError, match="unknown modality"):
        make_embeddings_config(modalities=["audio"])


def test_max_fragments_is_accepted(make_embeddings_config: EmbeddingsConfigFactory) -> None:
    """``max_fragments`` (the smoke-test fragment cap) assembles cleanly."""
    cfg = make_embeddings_config(max_fragments=2)
    assert cfg.max_fragments == 2


def test_max_fragments_below_one_is_rejected(make_embeddings_config: EmbeddingsConfigFactory) -> None:
    """A cap of zero fragments would visit nothing, so it is rejected at assembly."""
    with pytest.raises(ValidationError, match="greater than or equal to 1"):
        make_embeddings_config(max_fragments=0)


def test_recovery_needs_no_configuration(make_embeddings_config: EmbeddingsConfigFactory) -> None:
    """A crashed run must be resumable by an operator who configured nothing for it.

    ``None`` is what selects the sibling-of-the-table default, so a value here
    would mean recovery only works where someone opted into it.
    """
    assert make_embeddings_config().staging_root_uri is None


def test_a_staging_root_may_put_staged_rows_on_different_storage(
    make_embeddings_config: EmbeddingsConfigFactory,
) -> None:
    """The staged rows are transient bulk, so they need not live beside the table."""
    cfg = make_embeddings_config(staging_root_uri="s3://scratch-bucket/embedding_staging")
    assert cfg.staging_root_uri == "s3://scratch-bucket/embedding_staging"


def test_blank_staging_root_uri_rejected(make_embeddings_config: EmbeddingsConfigFactory) -> None:
    """A whitespace-only root is rejected here rather than at resolution.

    ``resolve_workspace`` refuses one too, but that is a Ray-side failure partway
    into a run; this is the same fault caught at assembly.
    """
    with pytest.raises(ValidationError, match="at least 1 character"):
        make_embeddings_config(staging_root_uri="   ")


def test_parquet_rows_per_file_below_one_is_rejected(make_embeddings_config: EmbeddingsConfigFactory) -> None:
    """Zero rows per file would stage no recoverable unit, so a crash would re-compute everything."""
    with pytest.raises(ValidationError, match="greater than or equal to 1"):
        make_embeddings_config(parquet_rows_per_file=0)


def test_pca_sample_floor_is_derived_from_action_dim() -> None:
    """The sample floor tracks the action width so the two cannot drift."""
    assert _MIN_PCA_SAMPLE_SIZE == ACTION_DIM + 1


def test_pca_sample_size_below_floor_rejected(make_embeddings_config: EmbeddingsConfigFactory) -> None:
    """pca_sample_size below the derived floor is rejected at config assembly."""
    with pytest.raises(ValidationError, match="greater than or equal to"):
        make_embeddings_config(action={"pca_sample_size": _MIN_PCA_SAMPLE_SIZE - 1})


def test_padded_clips_uri_is_stored_trimmed(make_embeddings_config: EmbeddingsConfigFactory) -> None:
    """Surrounding whitespace is stripped, so the stored URI differs from the configured text.

    Every downstream comparison - the artifact root derived from the clips URI, a
    log line, an operator diffing two runs - sees the trimmed value, so the
    stripping is part of the config's contract rather than an input convenience.
    """
    cfg = make_embeddings_config(clips_lance_uri=f"  {DEFAULT_CLIPS_URI}\t")
    assert cfg.clips_lance_uri == DEFAULT_CLIPS_URI


def test_blank_clips_lance_uri_rejected(make_embeddings_config: EmbeddingsConfigFactory) -> None:
    """A whitespace-only clips URI is rejected (min_length allows it)."""
    with pytest.raises(ValidationError, match="at least 1 character"):
        make_embeddings_config(clips_lance_uri="   ")


def test_blank_storage_profile_rejected(make_embeddings_config: EmbeddingsConfigFactory) -> None:
    """A whitespace-only storage profile is rejected (min_length allows it)."""
    with pytest.raises(ValidationError, match="at least 1 character"):
        make_embeddings_config(storage_profile="  ")


def test_blank_model_weights_path_rejected(make_embeddings_config: EmbeddingsConfigFactory) -> None:
    """A whitespace-only weights path is rejected (min_length allows it)."""
    with pytest.raises(ValidationError, match="at least 1 character"):
        make_embeddings_config(model_weights_path="   ")


def test_unknown_key_rejected(make_embeddings_config: EmbeddingsConfigFactory) -> None:
    """The model forbids extra keys (extra='forbid'); a removed field name is now unknown."""
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        make_embeddings_config(source_lance_uri=DEFAULT_CLIPS_URI)


def test_per_modality_resources_are_overridable_and_bounded(
    make_embeddings_config: EmbeddingsConfigFactory,
) -> None:
    """Each modality's resource block accepts an override and rejects a negative value.

    Asserts the mechanism the config module owns rather than the numbers, which are
    tuning knobs: re-profiling a modality onto a different GPU fraction is a correct
    edit that should not redden this file.
    """
    cfg = make_embeddings_config(text={"num_gpus": 0.5})
    assert cfg.text.num_gpus == 0.5
    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        make_embeddings_config(text={"num_gpus": -1.0})


def test_image_read_concurrency_above_batch_size_rejected(
    make_embeddings_config: EmbeddingsConfigFactory,
) -> None:
    """A read width wider than the scan batch is rejected instead of silently clamped.

    The reader submits ``min(read_concurrency, len(pending))`` workers from one
    scan batch, so an over-wide value never takes effect. Explicit values are
    passed on both sides so re-tuning either default cannot redden this test.
    """
    # match= is load-bearing: with extra="forbid" a bare ValidationError also passes
    # when the field is simply gone, so the pattern is what proves the cross-field
    # rule fired rather than the key becoming unknown.
    with pytest.raises(ValidationError, match="must not exceed batch_size"):
        make_embeddings_config(image={"batch_size": 16, "read_concurrency": 17})


def test_image_read_concurrency_equal_to_batch_size_accepted(
    make_embeddings_config: EmbeddingsConfigFactory,
) -> None:
    """The bound is inclusive: a read width exactly filling the scan batch is valid."""
    cfg = make_embeddings_config(image={"batch_size": 16, "read_concurrency": 16})
    assert cfg.image.read_concurrency == cfg.image.batch_size


@pytest.mark.parametrize("width", [0, -1])
def test_action_read_concurrency_below_one_rejected(
    width: int, make_embeddings_config: EmbeddingsConfigFactory
) -> None:
    """A read width below one worker is rejected: the leg would fetch nothing.

    match= pins the bound that fired: with extra="forbid" a bare ValidationError
    also passes when the field is simply gone.
    """
    with pytest.raises(ValidationError, match="greater than or equal to 1"):
        make_embeddings_config(action={"read_concurrency": width})


def test_action_read_concurrency_above_batch_size_rejected(
    make_embeddings_config: EmbeddingsConfigFactory,
) -> None:
    """A read width wider than the scan batch is rejected instead of silently clamped.

    The extractor draws its artifacts from one scan batch, so an over-wide value
    never takes effect. Explicit values are passed on both sides so re-tuning
    either default cannot redden this test.
    """
    with pytest.raises(ValidationError, match="must not exceed batch_size"):
        make_embeddings_config(action={"batch_size": 8, "read_concurrency": 9})


def test_action_read_concurrency_equal_to_batch_size_accepted(
    make_embeddings_config: EmbeddingsConfigFactory,
) -> None:
    """The bound is inclusive: a read width exactly filling the scan batch is valid."""
    cfg = make_embeddings_config(action={"batch_size": 8, "read_concurrency": 8})
    assert cfg.action.read_concurrency == cfg.action.batch_size


def test_a_missing_config_file_is_reported_by_path(tmp_path: Path) -> None:
    """The operator gets the path they typed, not a parser error."""
    absent = tmp_path / "absent.yaml"
    with pytest.raises(FileNotFoundError, match=re.escape(f"Config file not found: {absent}")):
        resolve_config(absent)


def test_a_non_mapping_config_is_rejected(tmp_path: Path) -> None:
    """A YAML list or scalar at the top level fails before Pydantic sees it."""
    path = tmp_path / "config.yaml"
    path.write_text("- not-a-mapping\n", encoding="utf-8")

    with pytest.raises(TypeError, match="mapping at the top level"):
        resolve_config(path)


def test_the_wrong_kind_is_rejected(tmp_path: Path) -> None:
    """The discriminator must match, so a misrouted config fails loudly."""
    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump({"schema_version": 1, "kind": "video-cut", "clips_lance_uri": DEFAULT_CLIPS_URI}),
        encoding="utf-8",
    )

    # match= pins the Literal check specifically. A bare ValidationError, or one
    # matching only "kind", would also pass if the field were deleted outright:
    # extra="forbid" echoes the unknown key's name back in the error.
    with pytest.raises(ValidationError, match="Input should be 'embeddings'"):
        resolve_config(path)


def test_a_future_schema_version_is_rejected(tmp_path: Path) -> None:
    """The version gate refuses a config written against a generation this code cannot read."""
    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump({"schema_version": 2, "kind": "embeddings", "clips_lance_uri": DEFAULT_CLIPS_URI}),
        encoding="utf-8",
    )

    # match= pins the version gate itself, not the field name: matching only
    # "schema_version" would still pass if the field were deleted, because
    # extra="forbid" reports the now-unknown key under that same name.
    with pytest.raises(ValidationError, match="Input should be 1"):
        resolve_config(path)


def test_overrides_are_parsed_as_yaml_scalars_not_strings(tmp_path: Path) -> None:
    """The config is strict, so a --set value must arrive as an int, not "2"."""
    path = _write_config(tmp_path)

    resolved = resolve_config(path, overrides=["max_fragments=2"])

    assert resolved.max_fragments == 2


def test_an_override_may_target_a_nested_key_that_is_absent(tmp_path: Path) -> None:
    """Tuning one modality knob must not require its whole block to be written out."""
    path = _write_config(tmp_path)

    resolved = resolve_config(path, overrides=["image.read_concurrency=8"])

    assert resolved.image.read_concurrency == 8


def test_an_override_is_validated_by_the_same_cross_field_rule_as_a_written_value(tmp_path: Path) -> None:
    """Overrides are applied BEFORE validation, so --set cannot smuggle past a cross-field rule.

    ``image.batch_size`` stays at its default and only the read width is pushed
    past it, so the cross-field rule is the sole constraint violated and match=
    proves it is the rule that fired rather than the key becoming unknown.
    """
    path = _write_config(tmp_path)

    with pytest.raises(ValidationError, match="must not exceed batch_size"):
        resolve_config(path, overrides=["image.read_concurrency=1000"])


def test_an_integer_override_satisfies_a_float_field_under_strict_validation(tmp_path: Path) -> None:
    """``--set text.num_gpus=0`` is accepted, so operators need not spell floats as ``0.0``.

    The model is ``strict=True``, which rejects most cross-type coercion, but
    Pydantic exempts int-to-float widening. Pinned because it decides the spelling
    the docs recommend for every float knob reachable through ``--set``, and YAML
    parses an unsuffixed ``0`` as an int.
    """
    path = _write_config(tmp_path)

    resolved = resolve_config(path, overrides=["text.num_gpus=0"])

    assert resolved.text.num_gpus == 0.0


@pytest.mark.parametrize("override", ["nokeypath", "=value", "image..batch_size=1"])
def test_malformed_overrides_are_rejected(tmp_path: Path, override: str) -> None:
    """A mistyped --set is a config error rather than a silently ignored flag."""
    path = _write_config(tmp_path)

    with pytest.raises(ValueError, match="--set override"):
        resolve_config(path, overrides=[override])


def test_the_resolved_config_is_frozen(tmp_path: Path) -> None:
    """The executing contract cannot drift after resolution."""
    resolved = resolve_config(_write_config(tmp_path))

    with pytest.raises(ValidationError, match="frozen"):
        resolved.max_fragments = 3
