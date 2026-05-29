# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
"""Common arguments for different pipelines."""

import argparse

from cosmos_curator.pipelines.av.captioning.captioning_stages import VRI_PROMPTS
from cosmos_curator.pipelines.av.utils.av_data_info import (
    get_avail_camera_format_ids,
)
from cosmos_curator.pipelines.pipeline_args import add_profiling_args


def validate_choices(value: str, valid_choices: set[str], argument_name: str) -> list[str]:
    """Validate that each value in a comma-separated string is in the set of valid choices.

    Args:
        value: Comma-separated string of values to validate
        valid_choices: Set of valid choices to check against
        argument_name: Name of the argument being validated (for error messages)

    Returns:
        List of validated values

    Raises:
        argparse.ArgumentTypeError: If any value is not in valid_choices

    """
    values = [x.strip() for x in value.split(",")]
    invalid_values = [v for v in values if v not in valid_choices]
    if invalid_values:
        error = (
            f"Invalid values for {argument_name}: {', '.join(invalid_values)}. "
            f"Valid values are: {', '.join(sorted(valid_choices))}"
        )
        raise argparse.ArgumentTypeError(error)
    return values


def add_common_args(parser: argparse.ArgumentParser, pipeline_name: str) -> None:
    """Add common arguments to the parser.

    Args:
        parser: The parser to add the arguments to.
        pipeline_name: The name of the pipeline.

    """
    parser.add_argument(
        "--db-profile",
        type=str,
        choices=["local", "dev", "prod"],
        default=None,
        help="Database profile to use.",
    )

    if pipeline_name in ["ingest", "split"]:
        parser.add_argument(
            "--input-prefix",
            type=str,
            default=None,
            help="S3 or local path which has input raw videos",
        )
        parser.add_argument(
            "--source-version",
            type=str,
            default="v0",
            help="version of the source data.",
        )

    if pipeline_name in ["split", "caption", "trajectory", "shard"]:
        parser.add_argument(
            "--output-prefix",
            type=str,
            default="s3://curated-data/",
            help="Path prefix for output files, can be either an s3 path or local path.",
        )
        parser.add_argument(
            "--camera-format-id",
            type=str,
            choices=get_avail_camera_format_ids(),
            default="U",
            help="Camera format ID.",
        )
        parser.add_argument(
            "--clip-version",
            type=str,
            default="v5",
            help="version of the split-and-encode clips.",
        )
        parser.add_argument(
            "--encoder",
            type=str,
            default="libopenh264",
            choices=["libopenh264", "h264_nvenc"],
            help="Codec for transcoding clips",
        )
        parser.add_argument(
            "--output-format",
            type=str,
            default="default",
            choices=["default", "cosmos_predict2"],
            help="Output dataset format",
        )
        parser.add_argument(
            "--dataset-name",
            type=str,
            default=None,
            required=True,
            help="Name for the dataset (used in cosmos-predict2 directory structure)",
        )

    if pipeline_name in ["split", "caption"]:
        parser.add_argument(
            "--qwen-input-prepare-cpus-per-actor",
            type=float,
            default=4.0,
            help="Number of CPUs per actor for Qwen input preparation stage.",
        )
        parser.add_argument(
            "--qwen-batch-size",
            type=int,
            default=8,
            help="Batch size for Qwen model.",
        )
        parser.add_argument(
            "--target-clip-size",
            type=int,
            default=256,
            help="Size in frame count for each clip.",
        )
        parser.add_argument(
            "--front-window-size",
            type=int,
            default=57,
            help="Size in frame count for front window in each clip.",
        )
        parser.add_argument(
            "--caption-chunk-size",
            type=int,
            default=32,
            help="Number of clips to caption in one chunk.",
        )
        parser.add_argument(
            "--enhance-captions-lm-variant",
            type=str,
            default="qwen_lm",
            choices=["qwen_lm", "gpt_oss_20b", "openai"],
            help="Language model for enhance captioning stage.",
        )
        parser.add_argument(
            "--enhance-captions-openai-model",
            type=str,
            default="auto",
            help="OpenAI model for caption enhancement ('auto' queries /v1/models).",
        )
        parser.add_argument(
            "--qwen-lm-batch-size",
            type=int,
            default=128,
            help=(
                f"Only applies when --prompt-type is one of {','.join(VRI_PROMPTS)}. "
                f"Batch size for Qwen-LM enhance captioning stage."
            ),
        )
        parser.add_argument(
            "--qwen-lm-use-fp8-weights",
            action="store_true",
            default=False,
            help=(
                f"Only applies when --prompt-type is one of {','.join(VRI_PROMPTS)}. "
                f"Whether to use fp8 weights for Qwen-LM model or not."
            ),
        )
        parser.add_argument(
            "--captioning-max-output-tokens",
            type=int,
            default=512,
            help=(
                f"Only applies when --prompt-type is one of {','.join(VRI_PROMPTS)}. "
                f"Max number of output tokens requested from enhanced captioning model."
            ),
        )

    if pipeline_name in ["split", "caption", "shard"]:
        parser.add_argument(
            "--caption-version",
            type=str,
            default="v4",
            help="version of the captioning clip captions.",
        )
        parser.add_argument(
            "--prompt-types",
            type=lambda x: validate_choices(x, {"default", *VRI_PROMPTS}, "prompt-types"),
            default=["default"],
            help=(
                f"Comma-separated list of prompt types for captioning. Valid values: default, {', '.join(VRI_PROMPTS)}"
            ),
        )
        parser.add_argument(
            "--prompt-text",
            type=str,
            default=None,
            help="Custom prompt text to use instead of predefined prompt types",
        )

    if pipeline_name == "shard":
        parser.add_argument(
            "--prompt-type",
            type=str,
            default="default",
            help="Specific prompt type to be included in dataset generation",
        )

    if pipeline_name in ["trajectory", "shard"]:
        parser.add_argument(
            "--trajectory-version",
            type=str,
            default="v4",
            help="version of the trajectory for each clip.",
        )

    if pipeline_name in ["split", "caption", "trajectory", "shard"]:
        parser.add_argument(
            "--session-file",
            type=str,
            default=None,
            help="Full Pathname of file containing list of sessions",
        )
        parser.add_argument(
            "--limit",
            type=int,
            default=0,
            help="Limit number of input videos to process.",
        )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Run pipeline without uploading results or updating database.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Whether to print verbose logs.",
    )

    add_profiling_args(parser)
