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

"""Run Pipeline CLI.

Two mutually exclusive invocation modes are supported::

    run_pipeline config.yaml              # config mode
    run_pipeline split --input-video-path /foo  # CLI mode

The first positional argument is treated as a config file when its
extension is ``.json``, ``.yaml``, or ``.yml``; otherwise it is
interpreted as a subcommand name and standard ``argparse`` processing
takes over.
"""

import argparse
import importlib
import sys

from cosmos_curator.core.utils.config.operation_context import check_if_running_in_pixi_env
from cosmos_curator.core.utils.config.pipeline_config_loader import load_pipeline_config

_CONFIG_EXTENSIONS = frozenset({".json", ".yaml", ".yml"})

_NVCF_ENTRY_POINTS: dict[str, str] = {
    "split": "cosmos_curator.pipelines.video.splitting_pipeline:nvcf_run_split",
    "dedup": "cosmos_curator.pipelines.video.dedup_pipeline:nvcf_run_semdedup",
    "shard": "cosmos_curator.pipelines.video.sharding_pipeline:nvcf_run_shard",
}


def _has_config_extension(arg: str) -> bool:
    """Return ``True`` when *arg* ends with a recognised config extension."""
    lower = arg.lower()
    return any(lower.endswith(ext) for ext in _CONFIG_EXTENSIONS)


def cli() -> None:
    """Run the video curation pipeline CLI.

    Supports two mutually exclusive modes:

    * **Config mode** -- pass a JSON/YAML config file as the sole
      positional argument.  The file must contain a ``pipeline`` key
      naming the subcommand.  Internally uses the same code-path as
      the NVCF invoke handler, which fills missing defaults
      automatically via ``fill_default_args``.
    * **CLI mode** -- pass a subcommand name followed by CLI flags.
    """
    if len(sys.argv) > 1 and _has_config_extension(sys.argv[1]):
        if len(sys.argv) > 2:  # noqa: PLR2004
            msg = "Config mode takes no extra arguments; put all values in the config file."
            raise SystemExit(msg)
        cfg = load_pipeline_config(sys.argv[1])
        command = cfg.pop("_pipeline", None)
        if not command or command not in _NVCF_ENTRY_POINTS:
            valid = ", ".join(sorted(_NVCF_ENTRY_POINTS))
            msg = f"Config must contain a valid 'pipeline' key (got: {command!r}). Valid pipelines: {valid}"
            raise SystemExit(msg)
        module_path, func_name = _NVCF_ENTRY_POINTS[command].rsplit(":", 1)
        module = importlib.import_module(module_path)
        entry_point = getattr(module, func_name)
        entry_point(argparse.Namespace(**cfg))
        return

    dedup_module = importlib.import_module("cosmos_curator.pipelines.video.dedup_pipeline")
    shard_module = importlib.import_module("cosmos_curator.pipelines.video.sharding_pipeline")
    split_module = importlib.import_module("cosmos_curator.pipelines.video.splitting_pipeline")

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Video curation pipelines",
    )
    subparsers = parser.add_subparsers(dest="command")
    shard_module.add_shard_command(subparsers)
    split_module.add_split_command(subparsers)
    dedup_module.add_dedup_command(subparsers)

    args = parser.parse_args()
    if not getattr(args, "command", None):
        parser.print_help()
        return

    args.func(args)


if __name__ == "__main__":
    check_if_running_in_pixi_env()
    cli()
