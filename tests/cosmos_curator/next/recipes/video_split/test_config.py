# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the strict video-split config contract."""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from cosmos_curator.core.utils.storage.s3_validation import MAX_S3_KEY_LENGTH_BYTES
from cosmos_curator.next.recipes.video_split.config import (
    FixedStrideConfig,
    TranscodeConfig,
    VideoSplitExecutionConfig,
    VideoSplitOutputConfig,
    config_template,
    config_template_yaml,
    resolve_config_data,
)


def _config(input_config: dict[str, object]) -> dict[str, object]:
    return {
        "schema_version": 1,
        "kind": "video-split",
        "input": input_config,
        "output": {"media_root": "s3://example-bucket/output/"},
    }


def test_explicit_uris_are_normalized_deduplicated_and_sorted() -> None:
    """The canonical config is independent of author URI order and duplicates."""
    config = resolve_config_data(
        _config(
            {
                "uris": [
                    "s3://example-bucket/z.MP4",
                    "S3://example-bucket/a.mp4",
                    "s3://example-bucket/z.MP4",
                ]
            }
        )
    )

    assert config.input.uris == (
        "s3://example-bucket/a.mp4",
        "s3://example-bucket/z.MP4",
    )
    assert config.output.media_root == "s3://example-bucket/output"
    assert config.output.clips_lance_uri == "s3://example-bucket/output/lance"
    assert config.output.errors_uri == "s3://example-bucket/output/errors.json"


def test_exact_input_uri_preserves_its_trailing_slash() -> None:
    """Normalization must not silently select a different S3 object key."""
    config = resolve_config_data(_config({"uris": ["s3://example-bucket/a.mp4/"]}))

    assert config.input.uris == ("s3://example-bucket/a.mp4/",)


def test_exact_input_uri_preserves_a_leading_slash_in_the_object_key() -> None:
    """The URI separator is not confused with a slash belonging to the key."""
    config = resolve_config_data(_config({"uris": ["s3://example-bucket/a.mp4", "s3://example-bucket//a.mp4"]}))

    assert config.input.uris == (
        "s3://example-bucket//a.mp4",
        "s3://example-bucket/a.mp4",
    )


@pytest.mark.parametrize(
    "input_config",
    [
        {},
        {
            "uris": ["s3://example-bucket/a.mp4"],
            "root_uri": "s3://example-bucket/raw/",
        },
    ],
)
def test_input_requires_exactly_one_selection_form(input_config: dict[str, object]) -> None:
    """Explicit objects and recursive root discovery cannot be mixed."""
    with pytest.raises(ValidationError, match="exactly one"):
        resolve_config_data(_config(input_config))


def test_config_rejects_non_s3_or_non_mp4_explicit_inputs() -> None:
    """V1 does not silently accept local files or other media suffixes."""
    with pytest.raises(ValidationError, match="s3://"):
        resolve_config_data(_config({"uris": ["local-a.mp4"]}))
    with pytest.raises(ValidationError, match="MP4"):
        resolve_config_data(_config({"uris": ["s3://example-bucket/a.mov"]}))


def test_config_accepts_s3_compatible_bucket_names() -> None:
    """S3-compatible stores may permit underscores that Amazon S3 rejects."""
    config = resolve_config_data(_config({"uris": ["s3://example_bucket/a.mp4"]}))

    assert config.input.uris == ("s3://example_bucket/a.mp4",)


def test_config_accepts_a_bucket_at_the_cross_backend_length_limit() -> None:
    """Ceph RGW and SwiftStack bucket names may be as long as 255 characters."""
    bucket = f"a{'_' * 253}z"

    config = resolve_config_data(_config({"uris": [f"s3://{bucket}/a.mp4"]}))

    assert config.input.uris == (f"s3://{bucket}/a.mp4",)


def test_config_rejects_a_bucket_over_the_cross_backend_length_limit() -> None:
    """Keep the dependency-light config validator aligned with S3Prefix's upper bound."""
    bucket = f"a{'b' * 254}z"

    with pytest.raises(ValidationError, match="Invalid S3 bucket name"):
        resolve_config_data(_config({"uris": [f"s3://{bucket}/a.mp4"]}))


def test_config_rejects_an_object_key_over_the_s3_length_limit() -> None:
    """Keep the dependency-light config validator aligned with S3Prefix's key limit."""
    key = f"{'a' * (MAX_S3_KEY_LENGTH_BYTES - 3)}.mp4"

    with pytest.raises(ValidationError, match="Invalid S3 object key"):
        resolve_config_data(_config({"uris": [f"s3://example-bucket/{key}"]}))


def test_lance_table_may_use_a_driver_local_path(tmp_path: Path) -> None:
    """A local Ray run can publish its Lance table on the driver's filesystem."""
    config = resolve_config_data(
        {
            **_config({"uris": ["s3://example-bucket/a.mp4"]}),
            "output": {
                "media_root": "s3://example-bucket/output",
                "clips_lance_uri": str(tmp_path / "lance"),
            },
        }
    )

    assert config.output.clips_lance_uri == str(tmp_path / "lance")


def test_overrides_apply_before_validation() -> None:
    """Dotted overrides participate in canonical validation."""
    config = resolve_config_data(
        _config({"root_uri": "s3://example-bucket/raw/"}),
        overrides=["split.duration_s=12", "split.min_duration_s=3", "execution.progress=true"],
    )

    assert config.split.duration_s == 12
    assert config.split.min_duration_s == 3
    assert config.execution.progress is True


def test_template_shows_every_supported_setting_and_resolves() -> None:
    """The human-facing template is a complete discoverability surface."""
    template = config_template()

    assert set(template["split"]) == set(FixedStrideConfig.model_fields)
    assert set(template["transcode"]) == set(TranscodeConfig.model_fields)
    assert set(template["output"]) == set(VideoSplitOutputConfig.model_fields)
    assert set(template["execution"]) == set(VideoSplitExecutionConfig.model_fields)
    assert template["execution"]["transcode_cpus"] == 5.0
    assert template["execution"]["ffmpeg_batch_size"] == 16
    assert template["execution"]["clips_per_publish_batch"] == 100_000

    rendered = config_template_yaml()
    assert "replace `uris` with `root_uri:" in rendered
    assert yaml.safe_load(rendered) == template
    assert resolve_config_data(template).model_dump(mode="json", exclude_none=True) == template
