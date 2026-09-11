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

"""Tests for the strict ``multimodal-split`` input and clip config sections."""

import pytest
from pydantic import ValidationError

from cosmos_curator.next.recipes.multimodal_split.config import (
    EncoderOpenH264Config,
    MultimodalSplitClipConfig,
    MultimodalSplitInputConfig,
    MultimodalSplitOutputConfig,
    MultimodalSplitSensorConfig,
    MultimodalSplitVideoConfig,
    ResolvedMultimodalSplitConfig,
)


def test_prefix_only_mode_defaults_to_no_list_and_no_limit() -> None:
    """The minimal config selects prefix-listing mode."""
    config = MultimodalSplitInputConfig(input_path_prefix="s3://example-bucket/recordings")

    assert config.input_path_prefix == "s3://example-bucket/recordings"
    assert config.session_id_list_path is None
    assert config.limit is None


def test_trailing_slashes_are_stripped_so_joins_stay_predictable() -> None:
    """Prefixes that differ only by trailing slashes canonicalize to one value."""
    bare = MultimodalSplitInputConfig(input_path_prefix="s3://example-bucket/recordings")
    slashed = MultimodalSplitInputConfig(input_path_prefix="s3://example-bucket/recordings///")

    assert slashed.input_path_prefix == bare.input_path_prefix


def test_session_id_list_path_keeps_its_exact_object_name() -> None:
    """A list path names an object, so its trailing characters are never stripped."""
    config = MultimodalSplitInputConfig(
        input_path_prefix="/data/recordings",
        session_id_list_path="/data/sessions.txt",
    )

    assert config.session_id_list_path == "/data/sessions.txt"


def test_unknown_fields_are_rejected() -> None:
    """Typos in a recipe config fail loudly instead of being silently ignored."""
    with pytest.raises(ValidationError, match="recurse"):
        MultimodalSplitInputConfig(input_path_prefix="/data/recordings", recurse=True)


def test_config_is_frozen() -> None:
    """Discovery inputs cannot drift after resolution."""
    config = MultimodalSplitInputConfig(input_path_prefix="/data/recordings")

    with pytest.raises(ValidationError, match="frozen"):
        config.limit = 5


def test_strict_mode_rejects_a_stringly_typed_limit() -> None:
    """A YAML-quoted limit is a config error rather than a silent coercion."""
    with pytest.raises(ValidationError, match="valid integer"):
        MultimodalSplitInputConfig(input_path_prefix="/data/recordings", limit="5")


@pytest.mark.parametrize("limit", [0, -1])
def test_limit_must_be_positive(limit: int) -> None:
    """A non-positive limit would silently discover nothing."""
    with pytest.raises(ValidationError, match="greater than or equal to 1"):
        MultimodalSplitInputConfig(input_path_prefix="/data/recordings", limit=limit)


@pytest.mark.parametrize("location", ["", "   ", " /data/recordings", "/data/recordings "])
def test_empty_or_padded_locations_are_rejected(location: str) -> None:
    """Whitespace in a location is a config authoring error, not something to trim silently."""
    with pytest.raises(ValidationError):
        MultimodalSplitInputConfig(input_path_prefix=location)


def test_unsupported_scheme_is_rejected() -> None:
    """Only local paths, file:// URIs, and s3:// URIs are discoverable."""
    with pytest.raises(ValidationError, match="Unsupported storage scheme"):
        MultimodalSplitInputConfig(input_path_prefix="gs://example-bucket/recordings")


def test_pathless_file_uri_is_rejected() -> None:
    """A file:// URI without a path would otherwise fall back to the working directory."""
    with pytest.raises(ValidationError, match="Unsupported local file URI"):
        MultimodalSplitInputConfig(input_path_prefix="file://")


def test_unsupported_scheme_is_rejected_for_the_session_id_list() -> None:
    """The list path gets the same scheme validation as the prefix."""
    with pytest.raises(ValidationError, match="Unsupported storage scheme"):
        MultimodalSplitInputConfig(
            input_path_prefix="/data/recordings",
            session_id_list_path="gs://example-bucket/sessions.txt",
        )


def test_a_local_prefix_may_be_paired_with_an_s3_session_id_list() -> None:
    """Mixed storage between the prefix and the list file is explicitly supported."""
    config = MultimodalSplitInputConfig(
        input_path_prefix="/data/recordings",
        session_id_list_path="s3://example-bucket/sessions.txt",
    )

    assert config.session_id_list_path == "s3://example-bucket/sessions.txt"


def test_a_scheme_without_a_location_is_rejected() -> None:
    """A templated prefix with empty variables must fail, not become a relative path.

    Stripping trailing slashes off the whole string would reduce 's3://' to 's3:',
    which is no longer recognized as remote and would silently resolve against the
    driver's working directory.
    """
    with pytest.raises(ValidationError, match="names a scheme but no location"):
        MultimodalSplitInputConfig(input_path_prefix="s3://")


def test_the_file_uri_root_survives_slash_stripping() -> None:
    """file:/// is the filesystem root and must not collapse to the bare scheme."""
    config = MultimodalSplitInputConfig(input_path_prefix="file:///")

    assert config.input_path_prefix == "file:///"


@pytest.mark.parametrize("prefix", ["/", "///"])
def test_the_filesystem_root_survives_slash_stripping(prefix: str) -> None:
    """A local root prefix reduces to a single separator rather than an empty string."""
    config = MultimodalSplitInputConfig(input_path_prefix=prefix)

    assert config.input_path_prefix == "/"


def test_trailing_slashes_are_stripped_from_the_key_not_the_scheme() -> None:
    """Only the location part of a URI loses its trailing slashes."""
    config = MultimodalSplitInputConfig(input_path_prefix="s3://example-bucket/recordings///")

    assert config.input_path_prefix == "s3://example-bucket/recordings"


@pytest.mark.parametrize(
    ("prefix", "expected"),
    [
        ("file:///", "file:///"),
        ("file://localhost/", "file://localhost/"),
        ("file:///data/recordings///", "file:///data/recordings"),
        ("s3://example-bucket/recordings///", "s3://example-bucket/recordings"),
        ("s3://example-bucket/", "s3://example-bucket"),
    ],
)
def test_only_the_path_component_loses_trailing_slashes(prefix: str, expected: str) -> None:
    """Stripping the whole string would drop the authority along with the slashes.

    ``file://localhost/`` would become ``file://localhost``, which has no path and
    is rejected by the very code meant to read it.
    """
    config = MultimodalSplitInputConfig(input_path_prefix=prefix)

    assert config.input_path_prefix == expected


@pytest.mark.parametrize(
    "prefix",
    [
        "file:///data/my#dir",
        "file:///data/my?dir",
        "file://localhost/data/my#dir",
        "s3://example-bucket/my#dir",
        "s3://example-bucket/my?dir",
    ],
)
def test_uris_with_an_unencoded_delimiter_are_rejected(prefix: str) -> None:
    """'#' and '?' are legal directory names but URI delimiters, so they truncate silently.

    ``file:///data/my#dir`` parses to the path ``/data/my``. Accepting it would
    curate a different directory than the one named, without any error.
    """
    with pytest.raises(ValidationError, match="unencoded"):
        MultimodalSplitInputConfig(input_path_prefix=prefix)


def test_an_unencoded_delimiter_is_rejected_in_the_session_id_list_path() -> None:
    """The list path truncates the same way and gets the same guard."""
    with pytest.raises(ValidationError, match="unencoded"):
        MultimodalSplitInputConfig(
            input_path_prefix="/data/recordings",
            session_id_list_path="file:///data/lists#2026/sessions.txt",
        )


def test_a_percent_encoded_delimiter_is_the_supported_spelling() -> None:
    """Percent-encoding is how a URI names a directory containing '#'."""
    config = MultimodalSplitInputConfig(input_path_prefix="file:///data/my%23dir")

    assert config.input_path_prefix == "file:///data/my%23dir"


def test_a_plain_local_path_may_contain_a_delimiter_character() -> None:
    """Only URIs reinterpret '#'; a bare path is passed through to the filesystem."""
    config = MultimodalSplitInputConfig(input_path_prefix="/data/my#dir")

    assert config.input_path_prefix == "/data/my#dir"


@pytest.mark.parametrize(
    "prefix",
    [
        "/data/recordings/../other",
        "../recordings",
        "file:///data/latest/..",
        "file:///data/%2E%2E/other",
        "s3://example-bucket/recordings/../other",
    ],
)
def test_parent_directory_segments_are_rejected(prefix: str) -> None:
    """'..' means two different directories depending on who resolves it.

    Listing goes through the kernel, which resolves a symlink before the '..';
    ``session_uri`` is built with ``os.path.abspath``, which cancels the segment
    lexically. A prefix like ``/data/latest/..`` would therefore emit URIs under
    a directory whose children were never listed. Rejecting it at config time is
    the pre-flight failure, before any listing happens.
    """
    with pytest.raises(ValidationError, match=r"'\.\.' segment"):
        MultimodalSplitInputConfig(input_path_prefix=prefix)


def test_parent_directory_segments_are_rejected_in_the_session_id_list_path() -> None:
    """The list path gets the same rule as the prefix."""
    with pytest.raises(ValidationError, match=r"'\.\.' segment"):
        MultimodalSplitInputConfig(
            input_path_prefix="/data/recordings",
            session_id_list_path="/data/../sessions.txt",
        )


@pytest.mark.parametrize("prefix", ["/data/..recordings", "/data/recordings..", "/data/.../recordings"])
def test_only_a_whole_parent_segment_is_rejected(prefix: str) -> None:
    """'..' inside a name is an ordinary directory, not a parent reference."""
    config = MultimodalSplitInputConfig(input_path_prefix=prefix)

    assert config.input_path_prefix == prefix


@pytest.mark.parametrize("prefix", ["s3://Example-Bucket/recordings", "s3://ab/recordings"])
def test_an_invalid_s3_bucket_name_is_rejected_at_config_time(prefix: str) -> None:
    """Discovery builds every S3 call from ``S3Prefix``, so its rules apply here too.

    Constructing one during validation moves an uppercase or too-short bucket
    name from the middle of a run to ``cosmos-curator pipeline validate``. The
    match is on ``S3Prefix``'s own message, so this fails if the check is dropped
    rather than passing on some other validator's rejection.
    """
    with pytest.raises(ValidationError, match="Invalid S3 bucket name"):
        MultimodalSplitInputConfig(input_path_prefix=prefix)


def test_an_invalid_s3_object_key_is_rejected_at_config_time() -> None:
    """A mistyped glob is caught here, which is the only place it is visible.

    ``S3Prefix`` itself accepts ``*`` because listings return keys that contain one.
    Config validation is stricter on purpose: a wildcard nobody expands makes the run
    match zero objects and report success.
    """
    with pytest.raises(ValidationError, match="Invalid S3 object key"):
        MultimodalSplitInputConfig(input_path_prefix="s3://example-bucket/recordings/*/raw")


def test_an_s3_session_id_list_path_is_validated_by_the_storage_layer_too() -> None:
    """A bucket-less s3 list path used to be accepted and fail only at read time.

    The prefix field caught this already through slash stripping, but the list
    path skips that step, so ``s3://`` reached ``read_text`` unchallenged.
    """
    with pytest.raises(ValidationError, match="Invalid S3 bucket name"):
        MultimodalSplitInputConfig(
            input_path_prefix="/data/recordings",
            session_id_list_path="s3://",
        )


@pytest.mark.parametrize("whitespace", ["\t", "\n", "\r"])
def test_uris_containing_stripped_whitespace_are_rejected(whitespace: str) -> None:
    """``urlsplit`` deletes tab, LF, and CR from a URI outright, per WHATWG.

    ``file:///data/rec<TAB>ord`` canonicalizes to ``file:///data/record``: not a
    truncation but a different directory name, with no error and nothing in the
    output to show a character went missing. All three are legal in POSIX
    filenames and S3 keys.
    """
    with pytest.raises(ValidationError, match="whitespace"):
        MultimodalSplitInputConfig(input_path_prefix=f"file:///data/rec{whitespace}ord")


@pytest.mark.parametrize("whitespace", ["\t", "\n", "\r"])
def test_stripped_whitespace_is_rejected_in_an_s3_uri(whitespace: str) -> None:
    """S3 keys carry the same characters and lose them the same way."""
    with pytest.raises(ValidationError, match="whitespace"):
        MultimodalSplitInputConfig(input_path_prefix=f"s3://example-bucket/rec{whitespace}ord")


def _resolved(**sections: object) -> ResolvedMultimodalSplitConfig:
    """Build a minimal valid config, overriding whichever sections a test is about.

    The whole-frame rule and the at-least-one-camera rule are facts about two
    sections together, so they can only be exercised through the resolved config
    rather than through either section alone.
    """
    payload: dict[str, object] = {
        "schema_version": 1,
        "kind": "multimodal-split",
        "input": {"input_path_prefix": "/data/recordings"},
        "sensors": {"front": {"sensor_type": "camera", "uri_pattern": "camera/front.mp4"}},
        "output": {"path": "/data/media"},
    }
    payload.update(sections)
    return ResolvedMultimodalSplitConfig.model_validate(payload)


def test_clip_defaults_describe_ten_second_clips_at_thirty_fps() -> None:
    """The defaults are the contract for a config that omits the clip section entirely."""
    clip = MultimodalSplitClipConfig()

    assert clip.duration_s == 10.0
    assert clip.sampling_fps == 30


def test_clip_config_is_frozen() -> None:
    """Clip geometry cannot drift after resolution any more than the input can."""
    clip = MultimodalSplitClipConfig()

    with pytest.raises(ValidationError, match="frozen"):
        clip.duration_s = 5.0


@pytest.mark.parametrize("value", [0, -1])
def test_sampling_fps_must_be_positive(value: int) -> None:
    """A rate of zero or less names no frames at all."""
    with pytest.raises(ValidationError, match="greater than or equal to 1"):
        MultimodalSplitClipConfig(sampling_fps=value)


@pytest.mark.parametrize("value", [30.0, 2.5, "30"])
def test_sampling_fps_must_be_an_integer(value: object) -> None:
    """Strict typing rejects a fractional or quoted rate instead of rounding it.

    The episode timeline is built from a positive integer FPS, so ``30.0`` is not
    silently narrowed and ``"30"`` is not silently parsed.
    """
    with pytest.raises(ValidationError, match="valid integer"):
        MultimodalSplitClipConfig(sampling_fps=value)


@pytest.mark.parametrize("duration_s", [0.0, -1.0])
def test_clip_duration_must_be_positive(duration_s: float) -> None:
    """A clip of zero or negative length has no frames to sample."""
    with pytest.raises(ValidationError, match="greater than 0"):
        MultimodalSplitClipConfig(duration_s=duration_s)


@pytest.mark.parametrize("duration_s", [float("inf"), float("-inf"), float("nan")])
def test_clip_duration_must_be_finite(duration_s: float) -> None:
    """A clip length has to become integer nanoseconds, which none of these can.

    ``inf`` is the case the ``> 0`` bound cannot see, since it satisfies it. The
    match is on the finiteness message for all three so that the bound alone
    cannot stand in for the check.
    """
    with pytest.raises(ValidationError, match="finite number"):
        MultimodalSplitClipConfig(duration_s=duration_s)


@pytest.mark.parametrize("field", ["stride_s", "min_duration_s"])
def test_stride_and_minimum_duration_are_not_settings_here(field: str) -> None:
    """Both are deliberately absent, and both have defined behaviour in their absence.

    No ``stride_s`` fixes the stride to the clip duration, giving contiguous
    non-overlapping clips. No ``min_duration_s`` means a trailing partial span is
    dropped rather than kept short. Rejecting them keeps a config copied from a
    design example or from ``video_split`` from implying behaviour that is not
    implemented.
    """
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        MultimodalSplitClipConfig(**{field: 2.0})


@pytest.mark.parametrize(("duration_s", "sampling_fps"), [(0.333, 30), (0.05, 30), (1.001, 30), (0.1, 24)])
def test_a_duration_that_is_not_a_whole_number_of_frames_is_rejected(duration_s: float, sampling_fps: int) -> None:
    """A clip is a whole number of output frames or it is not a valid clip.

    0.333 s at 30 fps is 9.99 frames. Refusing the config preserves the requested
    duration instead of silently dropping a partial frame during a run; the
    delivered grid determines row count only after validation and nanosecond
    normalization.
    """
    with pytest.raises(ValidationError, match="frames, which is not a whole number"):
        MultimodalSplitClipConfig(duration_s=duration_s, sampling_fps=sampling_fps)


@pytest.mark.parametrize(("duration_s", "sampling_fps"), [(10.0, 30), (0.5, 30), (2.5, 24), (0.04, 50)])
def test_a_fractional_duration_yielding_whole_frames_is_accepted(duration_s: float, sampling_fps: int) -> None:
    """The rule counts frames, not seconds, so a sub-second clip is fine when it lands on one.

    0.5 s at 30 fps is exactly 15 frames. Requiring a whole number of seconds
    instead would reject it for no reason.
    """
    clip = MultimodalSplitClipConfig(duration_s=duration_s, sampling_fps=sampling_fps)

    assert clip.duration_s == duration_s


def test_the_frame_count_is_computed_exactly_rather_than_in_binary_floating_point() -> None:
    """0.28 s at 25 fps is exactly 7 frames, but ``0.28 * 25`` is 7.000000000000001.

    Float and exact arithmetic disagree on which durations are whole frames for
    about one in a thousand millisecond-resolution durations, so a float remainder
    test would reject clips that are exact. The same hazard applies to the
    nanosecond rule below it.
    """
    config = _resolved(clip={"duration_s": 0.28, "sampling_fps": 25})

    assert config.clip.duration_s == 0.28


def test_the_encoder_defaults_to_libopenh264_and_its_bitrate() -> None:
    """The defaults match ``video_split``'s, so a clip encodes the same way in either recipe."""
    encoder = EncoderOpenH264Config()

    assert encoder.video_encoder == "libopenh264"
    assert encoder.video_bitrate == "4M"


@pytest.mark.parametrize(
    ("written", "canonical"),
    [("4M", "4M"), ("4.0M", "4M"), ("4m", "4M"), ("4.00m", "4M"), ("800k", "800K"), ("4.50M", "4.5M")],
)
def test_equivalent_bitrate_spellings_canonicalize_to_one_value(written: str, canonical: str) -> None:
    """One bitrate written three ways must resolve to one string.

    ``4M``, ``4.0M`` and ``4m`` name the same rate. Left alone they are three
    different values to compare, log, or hash into an output identity, so the
    resolved config holds only the canonical spelling.
    """
    assert EncoderOpenH264Config(video_bitrate=written).video_bitrate == canonical


@pytest.mark.parametrize("video_bitrate", ["4", "4G", "0.5M", "4MB", "four M", "0M"])
def test_a_malformed_bitrate_is_rejected(video_bitrate: str) -> None:
    """A bitrate needs a magnitude of at least one and a K or M suffix.

    A missing suffix, an unsupported one, or a leading zero would otherwise reach
    the encoder as an argument it cannot read, long after config load.
    """
    with pytest.raises(ValidationError, match="should match pattern"):
        EncoderOpenH264Config(video_bitrate=video_bitrate)


def test_an_unknown_encoder_is_rejected() -> None:
    """The encoder is a closed set, so a typo fails at config load rather than at spawn time."""
    with pytest.raises(ValidationError, match="Input should be 'libopenh264'"):
        EncoderOpenH264Config(video_encoder="libx264")


def test_audio_mode_is_not_a_setting_here() -> None:
    """Audio is a deferred capability for this pipeline, so there is no stream to copy.

    ``video_split`` carries an ``audio_mode``; a config copied across would
    otherwise imply this pipeline handles audio, which it does not.
    """
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        EncoderOpenH264Config(audio_mode="copy")


def test_a_camera_names_a_relative_artifact() -> None:
    """The one fact the split stage cannot infer: where in a session this camera's video is."""
    camera = MultimodalSplitSensorConfig(sensor_type="camera", uri_pattern="camera/front.mp4")

    assert camera.uri_pattern == "camera/front.mp4"


@pytest.mark.parametrize(
    ("max_delta_s", "expected_ns"),
    [(0.05, 50_000_000), (0.001, 1_000_000), (1.0, 1_000_000_000), (0.0335, 33_500_000)],
)
def test_a_tolerance_in_seconds_reaches_the_policy_as_whole_nanoseconds(max_delta_s: float, expected_ns: int) -> None:
    """The config is stated in the seconds every other duration here uses; the policy takes nanoseconds.

    Sub-millisecond values are the reason this rounds rather than truncates: the
    tolerance is a bound, and flooring it would quietly make it tighter than the
    operator asked for.
    """
    video = MultimodalSplitVideoConfig(max_delta_s=max_delta_s)

    assert video.max_delta_ns == expected_ns


@pytest.mark.parametrize(
    ("uri_pattern", "expected"),
    [
        ("/camera/front.mp4", "must be relative to a session"),
        ("s3://bucket/front.mp4", "must be relative to a session"),
        ("../front.mp4", "contains a '..' segment"),
        ("camera/../../front.mp4", "contains a '..' segment"),
        (" camera/front.mp4", "leading or trailing whitespace"),
        ("camera/front.mp4\t", "leading or trailing whitespace"),
        ("", "at least 1 character"),
    ],
)
def test_a_camera_artifact_that_escapes_its_session_is_rejected(uri_pattern: str, expected: str) -> None:
    """The pattern is joined onto a session URI, so anything absolute or upward leaves the session.

    Each case names the rule that must reject it. A bare "something raised"
    assertion would let one rule stand in for the others, so deleting any single
    check would still look covered.
    """
    with pytest.raises(ValidationError, match=expected):
        MultimodalSplitSensorConfig(sensor_type="camera", uri_pattern=uri_pattern)


@pytest.mark.parametrize("max_delta_s", [0.0, -1.0])
def test_a_non_positive_tolerance_is_rejected(max_delta_s: float) -> None:
    """A tolerance of zero demands an exact timestamp match, which no real recording provides."""
    with pytest.raises(ValidationError, match="greater than 0"):
        MultimodalSplitVideoConfig(max_delta_s=max_delta_s)


def test_exactly_one_camera_is_accepted() -> None:
    """One camera is the whole of the current scope, and its key names its output."""
    config = _resolved()

    assert list(config.cameras) == ["front"]


def test_no_cameras_is_rejected() -> None:
    """A run with no camera produces no video, which is never what the caller meant."""
    with pytest.raises(ValidationError, match="at least 1 item"):
        _resolved(sensors={})


def test_a_sensor_must_say_what_kind_it_is() -> None:
    """A suffix names a container, so it can say an artifact is an MCAP but not what is in it."""
    with pytest.raises(ValidationError, match="sensor_type"):
        _resolved(sensors={"front": {"uri_pattern": "camera/front.mp4"}})


def test_a_sensor_kind_the_recipe_cannot_process_is_refused_at_load() -> None:
    """Refused here rather than after a discovery pass has already listed every session."""
    with pytest.raises(ValidationError, match="camera"):
        _resolved(sensors={"imu": {"sensor_type": "imu", "uri_pattern": "*/imu.mcap"}})


def test_several_cameras_are_accepted() -> None:
    """A clip is a moment across every configured view, so the map carries more than one."""
    two = {
        "front": {"sensor_type": "camera", "uri_pattern": "camera/front.mp4"},
        "left": {"sensor_type": "camera", "uri_pattern": "camera/left.mp4"},
    }

    config = _resolved(sensors=two)

    assert sorted(config.cameras) == ["front", "left"]


@pytest.mark.parametrize("key", ["", "front/left", "..", ".", "front camera", "front\tleft"])
def test_a_camera_key_that_cannot_be_a_path_segment_is_rejected(key: str) -> None:
    """The key is written into the output layout, so it has to survive being a directory name.

    "." and ".." match the allowed characters but name a directory rather than
    sit in one, so both need rejecting separately from the character rule.
    """
    with pytest.raises(ValidationError, match="single path segment"):
        _resolved(sensors={key: {"sensor_type": "camera", "uri_pattern": "camera/front.mp4"}})


def test_the_output_path_is_canonicalized_like_the_input_prefix() -> None:
    """Output joins are as position-sensitive as input joins, so the same rules apply."""
    output = MultimodalSplitOutputConfig(path="s3://example-bucket/media///")

    assert output.path == "s3://example-bucket/media"


@pytest.mark.parametrize(
    ("root", "expected"),
    [
        ("/", "/episodes.lance"),
        ("/data/media", "/data/media/episodes.lance"),
        ("file:///", "file:///episodes.lance"),
        ("file:///data/media", "file:///data/media/episodes.lance"),
        ("s3://bucket/media", "s3://bucket/media/episodes.lance"),
    ],
)
def test_the_canonical_episode_table_has_one_derived_location(root: str, expected: str) -> None:
    """Version-one configs gain publication without a new output setting."""
    output = MultimodalSplitOutputConfig(path=root)

    assert output.episodes_lance_uri == expected


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("", "at least 1 character"),
        ("s3://bucket/media/../escape", "contains a '..' segment"),
        ("/data/media/../escape", "contains a '..' segment"),
        ("ftp://bucket/media", "scheme"),
        (" /data/media", "leading or trailing whitespace"),
    ],
)
def test_an_unusable_output_path_is_rejected(path: str, expected: str) -> None:
    """A bad output location should fail at load, not after a run has already decoded video."""
    with pytest.raises(ValidationError, match=expected):
        MultimodalSplitOutputConfig(path=path)


def test_a_file_uri_output_path_is_accepted() -> None:
    """``file://`` is a local path spelled out, and the recipe writes to local paths."""
    output = MultimodalSplitOutputConfig(path="file:///data/media/")

    assert output.path == "file:///data/media"


@pytest.mark.parametrize("missing", ["sensors", "output"])
def test_a_config_without_a_camera_or_an_output_location_is_rejected(missing: str) -> None:
    """Neither has a defensible default, so omitting one has to fail rather than be guessed at."""
    payload = {
        "schema_version": 1,
        "kind": "multimodal-split",
        "input": {"input_path_prefix": "/data/recordings"},
        "sensors": {"front": {"sensor_type": "camera", "uri_pattern": "camera/front.mp4"}},
        "output": {"path": "/data/media"},
    }
    del payload[missing]

    with pytest.raises(ValidationError, match=missing):
        ResolvedMultimodalSplitConfig.model_validate(payload)


@pytest.mark.parametrize("pattern", ["*/camera_front.mp4", "recorder-00/*/front.mp4", "*/*/front.mp4"])
def test_a_whole_path_segment_may_be_a_wildcard(pattern: str) -> None:
    """A rig's recorder layout is the rig's business, so a pattern need not name it."""
    camera = MultimodalSplitSensorConfig(sensor_type="camera", uri_pattern=pattern)

    assert camera.uri_pattern == pattern


@pytest.mark.parametrize("pattern", ["recorder-*/front.mp4", "camera_*.mp4", "*.mp4", "rec*/cam*.mp4"])
def test_a_wildcard_inside_a_segment_is_rejected(pattern: str) -> None:
    """Whole segments only: a partial match invites a pattern that resolves differently per session.

    "which recorder holds this camera" is the question worth answering. Matching
    part of a name answers a different one, and answers it differently as file
    names change.
    """
    with pytest.raises(ValidationError, match="whole path segment"):
        MultimodalSplitSensorConfig(sensor_type="camera", uri_pattern=pattern)
