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
"""Validate that a video's embedded header index matches a full packet scan."""

import argparse
import pathlib
import sys
from typing import Any

import boto3
import numpy as np
import numpy.typing as npt
from botocore.exceptions import BotoCoreError, NoCredentialsError, ProfileNotFound

from cosmos_curator.core.sensors.data.video import VideoIndex
from cosmos_curator.core.sensors.types.types import VideoIndexCreationMethod
from cosmos_curator.core.sensors.utils.video import _HeaderIndexUnavailableError, make_index_and_metadata

PASS_EXIT_CODE = 0
MISMATCH_EXIT_CODE = 1
ERROR_EXIT_CODE = 2

VIDEO_REQUIREMENTS_DOCS_URL = (
    "https://github.com/nvidia-cosmos/cosmos-curate/blob/main/docs/curator/design/"
    "SENSOR_LIBRARY_EFFICIENT_VIDEO_DECODE.md#from_header-vs-full_demux"
)
_S3_CREDENTIALS_HINT = (
    "Use --s3-profile-name to select an AWS profile, or configure standard AWS credentials "
    "with AWS_PROFILE, AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY, ~/.aws/credentials, or an IAM role."
)


class CliError(Exception):
    """Actionable user-facing CLI failure."""


def _is_s3_uri(source: str) -> bool:
    return source.startswith("s3://")


def _validate_source(source: str) -> None:
    if _is_s3_uri(source):
        return
    if "://" in source:
        msg = f"unsupported source URI {source!r}; use a local file path or an s3:// URI"
        raise CliError(msg)
    if not pathlib.Path(source).is_file():
        msg = f"source is not a file: {source}"
        raise CliError(msg)


def _make_client_params(source: str, s3_profile_name: str | None) -> dict[str, Any] | None:
    if not _is_s3_uri(source):
        return None

    try:
        session = boto3.Session(profile_name=s3_profile_name) if s3_profile_name else boto3.Session()
        credentials = session.get_credentials()
    except (BotoCoreError, ProfileNotFound) as e:
        msg = f"could not configure S3 access for {source!r}: {e}\n{_S3_CREDENTIALS_HINT}"
        raise CliError(msg) from e
    except Exception as e:
        msg = f"could not configure S3 access for {source!r}: {e}"
        raise CliError(msg) from e

    if credentials is None:
        msg = f"could not configure S3 access for {source!r}: {NoCredentialsError()}\n{_S3_CREDENTIALS_HINT}"
        raise CliError(msg)

    try:
        client = session.client("s3")
    except (BotoCoreError, ProfileNotFound) as e:
        msg = f"could not configure S3 access for {source!r}: {e}\n{_S3_CREDENTIALS_HINT}"
        raise CliError(msg) from e
    except Exception as e:
        msg = f"could not configure S3 access for {source!r}: {e}"
        raise CliError(msg) from e

    return {"transport_params": {"client": client}}


def _format_scalar(value: object) -> str:
    if isinstance(value, np.generic):
        return repr(value.item())
    return repr(value)


def _first_array_difference(
    field_name: str,
    header_values: npt.NDArray[Any],
    full_values: npt.NDArray[Any],
) -> str | None:
    shared_len = min(len(header_values), len(full_values))
    if shared_len > 0:
        diff_indices = np.flatnonzero(header_values[:shared_len] != full_values[:shared_len])
        if len(diff_indices) > 0:
            idx = int(diff_indices[0])
            return (
                f"{field_name} first differs at packet {idx}: "
                f"header={_format_scalar(header_values[idx])}, full_demux={_format_scalar(full_values[idx])}"
            )

    if len(header_values) != len(full_values):
        return f"{field_name} length differs: header={len(header_values)}, full_demux={len(full_values)}"

    return None


def make_mismatch_details(header_index: VideoIndex, full_index: VideoIndex) -> list[str]:
    """Build compact field-level details for an index mismatch."""
    details = [
        f"Header reports {len(header_index)} packets, full demux found {len(full_index)} packets.",
    ]

    if len(header_index.kf_pts_ns) != len(full_index.kf_pts_ns):
        details.append(
            f"Header reports {len(header_index.kf_pts_ns)} keyframes, "
            f"full demux found {len(full_index.kf_pts_ns)} keyframes."
        )

    if len(header_index.display_pts_ns) != len(full_index.display_pts_ns):
        details.append(
            f"Header reports {len(header_index.display_pts_ns)} displayable packets, "
            f"full demux found {len(full_index.display_pts_ns)} displayable packets."
        )

    if header_index.time_base != full_index.time_base:
        details.append(f"Time base differs: header={header_index.time_base}, full_demux={full_index.time_base}.")

    for field_name in ("offset", "size", "pts_ns", "pts_stream", "is_keyframe", "is_discard"):
        detail = _first_array_difference(field_name, getattr(header_index, field_name), getattr(full_index, field_name))
        if detail is not None:
            details.append(detail + ".")

    return details


def _format_mismatch_message(details: list[str]) -> str:
    detail_lines = "\n".join(f"  - {detail}" for detail in details)
    return f"""FAIL: Index mismatch detected.

This video file's table of contents (its "header") does not match what is
actually inside the file. This usually means the video was saved, copied,
or converted incorrectly and the header was not updated to match the real
contents.

What this means for you:
  - Tools that rely on the header for fast seeking may return incorrect or
    missing frames.
  - The video may still appear to play normally in a media player.

What to do:
  - For simple header/index issues, a stream-copy remux may be enough:
    `ffmpeg -i input.mp4 -c copy output.mp4`
  - Some videos, including DASH/fMP4 inputs or files with incorrect header
    tables, may need a full re-export or re-encode to a standard MP4.
  - For more detail, see: {VIDEO_REQUIREMENTS_DOCS_URL}

Mismatch detail (for advanced users):
{detail_lines}
"""


def _check_video_index(
    source: str,
    *,
    stream_idx: int,
    video_format: str | None,
    client_params: dict[str, Any] | None,
) -> tuple[bool, list[str]]:
    try:
        header_index, _ = make_index_and_metadata(
            source,
            stream_idx=stream_idx,
            video_format=video_format,
            index_method=VideoIndexCreationMethod.FROM_HEADER,
            client_params=client_params,
            allow_header_fallback=False,
        )
    except _HeaderIndexUnavailableError as e:
        full_index, _ = make_index_and_metadata(
            source,
            stream_idx=stream_idx,
            video_format=video_format,
            index_method=VideoIndexCreationMethod.FULL_DEMUX,
            client_params=client_params,
        )
        return False, [
            f"Header index could not be read from the file: {e}.",
            f"Full demux found {len(full_index)} packets.",
            f"Full demux found {len(full_index.kf_pts_ns)} keyframes.",
            f"Full demux found {len(full_index.display_pts_ns)} displayable packets.",
        ]

    full_index, _ = make_index_and_metadata(
        source,
        stream_idx=stream_idx,
        video_format=video_format,
        index_method=VideoIndexCreationMethod.FULL_DEMUX,
        client_params=client_params,
    )

    if header_index == full_index:
        return True, []
    return False, make_mismatch_details(header_index, full_index)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check whether an MP4 header index matches a full packet scan.",
        epilog="Exit codes: 0 = index consistent; 1 = mismatch detected; 2 = input, configuration, or runtime error.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source", required=True, help="Local path or s3:// URI to the MP4 file.")
    parser.add_argument("--stream-idx", type=int, default=0, help="Video stream index.")
    parser.add_argument(
        "--video-format", default=None, help="Optional container format hint passed to the video loader."
    )
    parser.add_argument(
        "--s3-profile-name",
        default=None,
        help="Optional AWS profile name used for s3:// sources. If omitted, boto3's default credential chain is used.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the video index check."""
    args = _parse_args(argv)

    try:
        _validate_source(args.source)
        client_params = _make_client_params(args.source, args.s3_profile_name)
        ok, details = _check_video_index(
            args.source,
            stream_idx=args.stream_idx,
            video_format=args.video_format,
            client_params=client_params,
        )
    except CliError as e:
        sys.stderr.write(f"error: {e}\n")
        return ERROR_EXIT_CODE
    except Exception as e:  # noqa: BLE001
        sys.stderr.write(f"error: could not check video index for {args.source!r}: {e}\n")
        return ERROR_EXIT_CODE

    if ok:
        sys.stdout.write("PASS: Video index is consistent.\n\n")
        sys.stdout.write("The video header matches the full packet scan for this file.\n")
        return PASS_EXIT_CODE

    sys.stdout.write(_format_mismatch_message(details))
    return MISMATCH_EXIT_CODE


if __name__ == "__main__":
    raise SystemExit(main())
