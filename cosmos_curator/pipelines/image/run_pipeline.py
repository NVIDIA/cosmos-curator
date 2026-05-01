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

"""Run image curation pipeline CLI."""

import argparse
import importlib

from cosmos_curator.core.utils.config.operation_context import check_if_running_in_pixi_env
from cosmos_curator.core.utils.infra.profiling import profiling_scope


def cli() -> None:
    """Run the image curation pipeline CLI.

    Parses command line arguments and executes the selected pipeline command.
    """
    annotate_module = importlib.import_module("cosmos_curator.pipelines.image.annotate_pipeline")

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Image curation pipelines",
    )
    subparsers = parser.add_subparsers(dest="command")
    annotate_module.add_annotate_command(subparsers)
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    with profiling_scope(args):
        args.func(args)


if __name__ == "__main__":
    check_if_running_in_pixi_env()
    cli()
