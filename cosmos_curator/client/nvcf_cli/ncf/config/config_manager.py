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

"""Manage and manipulate NVCF configuration settings."""

import json
import shutil
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.pretty import pprint
from typer import Context, Option

from cosmos_curator.client.nvcf_cli.ncf.common import (
    NvcfBase,
    base_callback,
    register_instance,
)
from cosmos_curator.client.utils.validations import validate_positive_integer


class ConfigManager(NvcfBase):
    """Manager for NVCF configuration settings.

    Provides methods to get and set configuration for the NVIDIA Cloud Function platform.
    """

    def __init__(self, url: str, nvcf_url: str, key: str, org: str, team: str, timeout: int) -> None:  # noqa: PLR0913
        """Initialize the ConfigManager.

        Args:
            url: Base NGC URL
            nvcf_url: Base NVCF URL
            key: NGC NVCF API Key
            org: Organization ID or name
            team: Team name within the organization
            timeout: Request timeout in seconds

        """
        super().__init__(url=url, nvcf_url=nvcf_url, key=key, org=org, team=team, timeout=timeout)

    def get_config(self) -> dict[str, Any]:
        """Get the current configuration.

        Returns:
            Dictionary containing the current configuration

        Raises:
            Exception: If there is an error getting the configuration

        """
        if self.config == {}:
            error_msg = f"No Configurations found, Please run '{self.exe} nvcf config set' to create configuration"
            self.logger.error(error_msg)
            return {}
        return self.config

    def set_config(self) -> None:
        """Save the current configuration to disk and copy example templates.

        This method:
        1. Creates the configuration directory
        2. Saves the config as JSON
        3. Creates template directories
        4. Warns if certain config values are missing
        """
        # Create config directory
        try:
            Path(self.cfgdir).mkdir(mode=0o700, parents=True, exist_ok=True)
        except (PermissionError, OSError):
            self.logger.exception("Failed to create config directory '%s'", self.cfgdir)
            return

        # Save config file
        config_path = Path(self.cfgdir) / self.CLIENT_NAME
        try:
            with config_path.open("w") as fc:
                json.dump(self.config, fc, indent=4)
            config_path.chmod(0o600)
            self.logger.info("Saved Config in %s", config_path)
        except (PermissionError, json.JSONDecodeError, OSError):
            self.logger.exception("Failed to save config file")
            return

        # Create template directories
        try:
            source_dir = Path(__file__).resolve().parent.parent.parent.parent.parent
            example_dir = source_dir / "examples"
            if not example_dir.is_dir():
                example_dir = source_dir.parent / "examples"  # happens when invoked from dev env
            template_dir = Path(self.cfgdir) / self.TEMPLATE_DIR
            template_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
            shutil.copytree(example_dir / "nvcf", template_dir, dirs_exist_ok=True)

            self.logger.info(
                "Templates needed for operations created in %s, "
                "please make copies before using and ensure they have correct entries",
                template_dir,
            )

            # Check for missing configuration
            if None in [self.config.get("backend"), self.config.get("gpu"), self.config.get("instance")]:
                self.logger.warning(
                    "This setting is good for 'invoke-function', however if you intend to "
                    "manage function life-cycle, please re-run with '--backend', "
                    "'--gpu' and '--instance' passed as command line argument"
                )
        except (PermissionError, FileNotFoundError, OSError):
            self.logger.exception("Failed to create templates in directory %s", self.cfgdir)
        except KeyError as e:
            error_msg = f"Failed to create templates due to missing config key: {e!s}"
            self.logger.error(error_msg)  # noqa: TRY400


nvcf_config = typer.Typer(
    context_settings={
        "max_content_width": 120,
    },
    pretty_exceptions_enable=False,
    no_args_is_help=True,
    callback=base_callback,
)
ins_name = "config"
ins_help = "NVCF Cluster Configuration setup"
register_instance(ins_name=ins_name, ins_help=ins_help, ins_type=ConfigManager, ins_app=nvcf_config)


@nvcf_config.command(name="get", help="Get the ACTIVE backend configuration")
def nvcf_config_get_config(
    ctx: Context,
) -> None:
    """Get the current active backend configuration.

    Args:
        ctx: Typer context containing the NVCF handler.

    """
    nvcf_hdl = ctx.obj["nvcfHdl"]
    resp = nvcf_hdl.get_config()
    if len(resp) != 0:
        if resp.get("backend") is None:
            resp.pop("backend")
        if resp.get("gpu") is None:
            resp.pop("gpu")
        if resp.get("instance") is None:
            resp.pop("instance")
        pprint(resp, expand_all=True)


@nvcf_config.command(
    name="set",
    help="""Set configurations including API key, organization ID or name, base NGC url, and base NVCF url.
            For backend, invoke 'nvcf function list-clusters' to find available options""",
    no_args_is_help=False,
)
def nvcf_config_set_config(  # noqa: C901, PLR0913, PLR0912
    ctx: Context,
    backend: Annotated[
        str | None,
        Option(
            help="The Name of Backend CSP",
            rich_help_panel="Config",
            envvar="NVCF_BACKEND",
        ),
    ] = None,
    gpu: Annotated[
        str | None,
        Option(
            help="The GPU Hardware Type",
            rich_help_panel="Config",
            envvar="NVCF_GPU_TYPE",
        ),
    ] = None,
    instance: Annotated[
        str | None,
        Option(
            help="The Hardware Instance Type",
            rich_help_panel="Config",
            envvar="NVCF_INSTANCE_TYPE",
        ),
    ] = None,
    key: Annotated[
        str | None,
        Option(
            help="NGC NVCF API Key",
            rich_help_panel="Common",
            envvar="NGC_NVCF_API_KEY",
        ),
    ] = None,
    org: Annotated[
        str | None,
        Option(
            help="Organization ID or name",
            rich_help_panel="Common",
            envvar="NGC_NVCF_ORG",
        ),
    ] = None,
    team: Annotated[
        str,
        Option(
            help="Team name within the org if applicable",
            rich_help_panel="Common",
            envvar="NGC_NVCF_TEAM",
        ),
    ] = "no-team",
    url: Annotated[
        str | None,
        Option(
            help="The Base NGC url",
            rich_help_panel="Common",
            envvar="NGC_BASE_URL",
        ),
    ] = None,
    nvcf_url: Annotated[
        str | None,
        Option(
            help="The Base NVCF url",
            rich_help_panel="Common",
            envvar="NVCF_BASE_URL",
        ),
    ] = None,
    timeout: Annotated[
        int | None,
        Option(
            help="Unused",
            rich_help_panel="Common",
            envvar=" NGC_NVCF_TIMEOUT",
        ),
    ] = None,
) -> None:
    """Set the backend configuration for NVCF operations.

    Args:
        ctx: Typer context containing the NVCF handler.
        backend: Name of the backend CSP.
        gpu: GPU hardware type.
        instance: Hardware instance type.
        key: set API Key.
        org: Organization ID or name.
        team: Team name within the organization.
        url: Base NGC url.
        nvcf_url: Base NVCF url
        timeout: Unused

    """
    if timeout is not None:
        validate_positive_integer(timeout)
    nvcf_hdl = ctx.obj["nvcfHdl"]
    if url is None:
        url = ctx.obj.get("config").get("url")
    if key is None:
        key = ctx.obj.get("config").get("key")
    if org is None:
        org = ctx.obj.get("config").get("org")
    assert team is not None
    if nvcf_url is None:
        nvcf_url = ctx.obj.get("config").get("nvcf_url")
    if timeout is None:
        timeout = ctx.obj.get("config").get("timeout")
    if None in [key, org]:
        nvcf_hdl.logger.error("Missing '--key' or '--org'")
        return

    """Manually set Common values"""
    if key is not None:
        ctx.obj["key"] = key
    if url is not None:
        ctx.obj["url"] = url
    if org is not None:
        ctx.obj["org"] = org
    if team is not None:
        ctx.obj["team"] = team
    if nvcf_url is not None:
        ctx.obj["nvcf_url"] = nvcf_url
    if timeout is not None:
        ctx.obj["timeout"] = timeout

    nvcf_hdl.save_config(
        url=ctx.obj["url"],
        key=ctx.obj["key"],
        org=ctx.obj["org"],
        team=ctx.obj["team"],
        nvcf_url=ctx.obj["nvcf_url"],
        timeout=ctx.obj["timeout"],
        backend=backend,
        instance=instance,
        gpu=gpu,
    )

    resp = nvcf_hdl.set_config()
    if resp is not None:
        pprint(resp, expand_all=True)


if __name__ == "__main__":
    nvcf_config()
