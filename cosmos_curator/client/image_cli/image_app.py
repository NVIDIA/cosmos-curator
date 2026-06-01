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

"""CLI to build image."""

import sys
from pathlib import Path
from typing import Annotated

import rich
import typer
from loguru import logger
from typer import Option

from cosmos_curator.client.environment import CONTAINER_PATHS_CODE_DIR
from cosmos_curator.client.utils import docker_utils
from cosmos_curator.client.utils.conda_envs import get_pixi_envs

image_build = typer.Typer(
    help="Commands for building container image.",
    no_args_is_help=True,
)


def get_image_label(image_name: str, image_tag: str) -> str:
    """Generate a Docker image label by combining name and tag.

    Args:
        image_name: Name of the Docker image.
        image_tag: Version tag for the image.

    Returns:
        Combined image label in format 'name:tag'.

    """
    return f"{image_name}:{image_tag}"


@image_build.command()
def build(  # noqa: PLR0913
    *,
    curator_path: Annotated[
        str,
        Option(
            help=("Path to the cosmos-curator repo directory"),
            rich_help_panel="docker",
        ),
    ] = str(Path(__file__).parent.parent.parent.parent),
    image_name: Annotated[
        str,
        Option(
            help=("The docker image name string to use."),
            rich_help_panel="docker",
        ),
    ] = "cosmos-curator",
    image_tag: Annotated[
        str,
        Option(
            help=("The docker image tag to use."),
            rich_help_panel="docker",
        ),
    ] = "1.0.0",
    envs: Annotated[
        str,
        Option(
            help=(
                "Comma-separated list of conda environments to build into the image. "
                "Supported values can be found under package/cosmos_curator/envs/."
            ),
            rich_help_panel="conda_envs",
        ),
    ] = "cuml,legacy-transformers,sam3,seedvr,unified",
    use_local_xenna_build: Annotated[
        bool,
        Option(
            help=(
                "When enabled, build cosmos-xenna package from local checkout. "
                "Otherwise, install cosmos-xenna from PyPI."
            ),
            rich_help_panel="common",
        ),
    ] = False,
    dockerfile_output_path: Annotated[
        str | None,
        Option(
            help=("Path to store the rendered Dockerfile. If not specified, will be written to a temp file"),
            rich_help_panel="common",
        ),
    ] = None,
    cache_from: Annotated[
        list[str] | None,
        Option(
            help=("Use an external cache source for a build. Useful for the CI/CD environment."),
            rich_help_panel="common",
        ),
    ] = None,
    cache_to: Annotated[
        str | None,
        Option(
            help=("Export build cache to an external cache destination. Useful for the CI/CD environment."),
            rich_help_panel="common",
        ),
    ] = None,
    push: Annotated[
        bool,
        Option(
            help=(
                "Push the image directly from BuildKit to the registry, "
                "skipping local daemon load. Faster in CI (single hop). "
                "Requires a buildx builder with the docker-container driver."
            ),
            rich_help_panel="common",
        ),
    ] = False,
    load: Annotated[
        bool,
        Option(
            help=(
                "Load the built image into the local Docker daemon. "
                "Enabled by default; use --no-load to skip (e.g. in CI)."
            ),
            rich_help_panel="common",
        ),
    ] = True,
    dry_run: Annotated[
        bool,
        Option(
            help="If True, only generate the Dockerfile and do not build the image.",
            rich_help_panel="common",
        ),
    ] = False,
    verbose: Annotated[
        bool,
        Option(
            help="If True, logs detailed information.",
            rich_help_panel="common",
        ),
    ] = False,
    nsight: Annotated[
        bool,
        Option(
            help="Install nsight-systems for CUDA profiling. Enable with --nsight for GPU kernel profiling.",
            rich_help_panel="common",
        ),
    ] = False,
    redistributable: Annotated[
        bool,
        Option(
            "--redistributable/--non-redistributable",
            help=(
                "Build the redistributable image variant. This excludes components we cannot redistribute "
                "and uses the narrower FFmpeg runtime policy."
            ),
            rich_help_panel="common",
        ),
    ] = False,
    extra_code_paths: Annotated[
        str | None,
        Option(
            help=(
                "Comma-separated list of paths to additional code paths to copy into the image. "
                f"These paths will be copied into the image under {CONTAINER_PATHS_CODE_DIR.as_posix()}."
            ),
            rich_help_panel="common",
        ),
    ] = None,
    slim: Annotated[
        bool,
        Option(
            help=(
                "Build a slim image containing only the lockfile and source code (no pixi install). "
                "Environments are installed at runtime via pixi run."
            ),
            rich_help_panel="docker",
        ),
    ] = False,
) -> None:
    """Build a docker image with the specified conda environments."""
    _curator_path = Path(curator_path)
    _dockerfile_output_path = Path(dockerfile_output_path) if dockerfile_output_path else None
    package_path = _curator_path / Path("package") / Path("cosmos_curator")

    env_names = _parse_envs(envs)
    # Validate that requested environments exist in Pixi
    pixi_envs = get_pixi_envs()
    invalid_envs = set(env_names) - set(pixi_envs)
    if invalid_envs:
        logger.error(f"Environments not available in Pixi: {sorted(invalid_envs)}")
        sys.exit(1)

    code_paths = []
    if extra_code_paths:
        code_paths.extend([path.rstrip("/") for path in extra_code_paths.split(",")])

    console = rich.console.Console()

    if slim:
        console.log("Generating slim docker image (no pixi install)")
    elif redistributable:
        console.log(f"Generating redistributable docker image with envs: {env_names}")
    else:
        console.log(f"Generating docker image with envs: {env_names}")
    dockerfile_template_path = package_path / Path("default.dockerfile.jinja2")

    dockerfile_path = docker_utils.generate_dockerfile(
        dockerfile_template_path=dockerfile_template_path,
        conda_env_names=env_names,
        use_local_xenna_build=use_local_xenna_build,
        code_paths=code_paths,
        dockerfile_output_path=_dockerfile_output_path,
        verbose=verbose,
        nsight=nsight,
        slim=slim,
        redistributable=redistributable,
    )

    if dry_run:
        logger.info("Dry-run mode enabled. Generating Dockerfile only.")
        return

    # Proceed with building the image if not in dry-run mode
    console.log("Starting full build process.")

    image_label = get_image_label(image_name, image_tag)
    docker_utils.build(
        curator_path=_curator_path,
        dockerfile_path=dockerfile_path,
        image=image_label,
        cache_from=cache_from,
        cache_to=cache_to,
        push=push,
        load=load,
        verbose=verbose,
    )
    logger.info(f"Built docker image: {image_label}")


def _parse_envs(env_string: str) -> list[str]:
    env_names = set()
    if env_string:
        env_names = set(env_string.split(","))
    env_names.add("model-download")
    env_names.add("default")
    return sorted(env_names)
