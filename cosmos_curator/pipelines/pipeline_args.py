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
"""Common arguments for different pipelines."""

import argparse

from cosmos_curator.pipelines.common_pipeline_settings import (
    PROFILING_CLI_FIELDS,
    CommonPipelineSettings,
    add_settings_cli_arguments,
)


def add_profiling_args(parser: argparse.ArgumentParser) -> None:
    """Add profiling / instrumentation CLI flags to the parser.

    Args:
        parser: The argument parser to add profiling flags to.

    """
    add_settings_cli_arguments(parser, CommonPipelineSettings, only_fields=PROFILING_CLI_FIELDS)


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add common command line arguments to the parser.

    Includes S3, execution-mode, limit, verbose, model-weights, and
    profiling flags (same fields as :class:`~cosmos_curator.pipelines.common_pipeline_settings.CommonPipelineSettings`).

    Args:
        parser: The argument parser to add arguments to.

    """
    add_settings_cli_arguments(parser, CommonPipelineSettings)
