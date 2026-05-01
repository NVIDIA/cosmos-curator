# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Manage Prometheus service discovery."""

import argparse
import json
import logging
import pathlib

# Set up logging
logger = logging.getLogger(__name__)


def _setup_parser() -> argparse.ArgumentParser:
    """Set up the parser for the command line interface."""
    parser = argparse.ArgumentParser(description="Generate Slurm Prometheus service discovery file.")
    parser.add_argument(
        "--path",
        type=pathlib.Path,
        required=True,
        help="Path to the Slurm prometheus service discovery file.",
    )
    parser.add_argument("--job-user", type=str, required=True, help="User who submitted the job to Slurm.")
    parser.add_argument("--job-id", type=str, required=True, help="Job ID to include in the service discovery file.")
    parser.add_argument(
        "--job-name",
        type=str,
        required=True,
        help="Job name to include in the service discovery file.",
    )
    parser.add_argument(
        "--hostfile",
        type=pathlib.Path,
        required=True,
        help="Path to the hostfile containing node information.",
    )
    parser.add_argument("--port", type=int, required=True, help="Port on which the metrics are exposed.")
    return parser


def create_slurm_service_discovery(args: argparse.Namespace) -> None:
    """Create the Slurm Prometheus service discovery file."""
    targets = []
    with args.hostfile.open("rt") as fr:
        targets = [f"{line.strip()}:{args.port}" for line in fr if line.strip()]
    data = [
        {
            "labels": {
                "job": "cosmos-curator",
                "slurm_job_user": args.job_user,
                "slurm_job_id": args.job_id,
                "slurm_job_name": args.job_name,
            },
            "targets": targets,
        }
    ]
    with args.path.open("wt") as fw:
        json.dump(data, fw, indent=2)
    logger.info("Generated Slurm Prometheus service discovery file at %s", str(args.path))


if __name__ == "__main__":
    parser = _setup_parser()
    args = parser.parse_args()
    try:
        create_slurm_service_discovery(args)
    except Exception:
        logger.exception("Error creating Slurm service discovery file")
