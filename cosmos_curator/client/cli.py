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
"""cosmos-curator CLI top-level entrypoint."""

import typer

from cosmos_curator.client.image_cli import image_app
from cosmos_curator.client.local_cli import launch_local
from cosmos_curator.client.nvcf_cli import launch_nvcf
from cosmos_curator.client.slurm_cli import slurm
from cosmos_curator.client.view_cli import viewer_app

cosmos_curator = typer.Typer(
    context_settings={
        "max_content_width": 120,
    },
    pretty_exceptions_enable=False,
    no_args_is_help=True,
    add_completion=False,
    rich_markup_mode="rich",
)

cosmos_curator.add_typer(typer_instance=image_app.image_build, name="image", help="Image Build Functionalities")
cosmos_curator.add_typer(typer_instance=launch_local.cc_client_local, name="local", help="Local Functionalities")
cosmos_curator.add_typer(typer_instance=launch_nvcf.cc_client_nvcf, name="nvcf", help="NVCF Functionalities")
cosmos_curator.add_typer(typer_instance=slurm.slurm_cli, name="slurm", help="Slurm Functionalities")
cosmos_curator.add_typer(typer_instance=viewer_app.viewer_app, name="view", help="Viewer Functionalities")

if __name__ == "__main__":
    cosmos_curator()
