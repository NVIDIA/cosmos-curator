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

"""Define base classes and functionality for NVCF operations.

This module provides the base class and core functionality for NVIDIA Cloud Function operations,
including configuration management, instance registration, and common utilities.
"""

import json
import logging
import sys
from abc import ABC
from pathlib import Path
from typing import Annotated, Any

import typer
import yaml
from rich.console import Console
from rich.logging import RichHandler
from typer import Context, Option

from cosmos_curator.client.utils.validations import validate_positive_integer


class NvcfBase(ABC):  # noqa: B024
    """Base class for NVIDIA Cloud Function operations.

    This abstract base class provides common functionality for managing NVCF operations,
    including configuration handling, logging setup, and cluster management.
    """

    CURATOR_NAME = "config.yaml"
    CLIENT_NAME = "client.json"
    CONFIG_DIR = ".config/cosmos_curator"
    TEMPLATE_DIR = "templates"
    FUNCID_NAME = "funcid.json"

    def __init__(self, url: str, nvcf_url: str, key: str | None, org: str | None, team: str, timeout: int) -> None:  # noqa: PLR0913
        """Initialize the NVCF base class.

        Args:
            url: Base NGC URL
            nvcf_url: Base NVCF URL
            key: NGC NVCF API Key
            org: Organization ID or name
            team: Team name within the org
            timeout: Request timeout in seconds

        """
        self.url: str = url
        self.key: str | None = key
        self.org: str | None = org
        self.team: str = team
        self.nvcf_url: str = nvcf_url
        self.timeout: int = timeout
        self.cfg: dict[str, Any] = {}
        self.console = Console()

        rh = RichHandler(show_path=False, console=self.console)
        rh.setFormatter(logging.Formatter("%(message)s"))
        fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        logging.basicConfig(level=logging.INFO, format=fmt, handlers=[rh])
        self.logger = logging.getLogger(__name__)

        self.cfgdir = Path.home() / self.CONFIG_DIR
        self.idf = Path(self.cfgdir) / self.FUNCID_NAME
        self.exe = sys.argv[0]

        self.load_config()

        if self.key is None:
            self.key = self.cfg.get("key")

        if self.org is None:
            self.org = self.cfg.get("org")

    def get_hf_token_from_config(self) -> str | None:
        """Read Hugging Face API token from config.yaml.

        Returns:
            Optional[str]: API token if found, else None

        """
        fname = Path(self.cfgdir) / self.CURATOR_NAME
        token: str | None = None
        try:
            with fname.open() as fc:
                cfg = yaml.safe_load(fc)
                token = cfg.get("huggingface", {}).get("api_key")
        except (OSError, UnicodeDecodeError, PermissionError, FileNotFoundError, yaml.YAMLError):
            pass  # Caller deals with it

        return token

    def load_config(self) -> dict[str, Any] | None:
        """Load configuration from the config file.

        Returns:
            The loaded configuration dictionary, or None if no configuration exists.

        """
        fname = Path(self.cfgdir) / self.CLIENT_NAME
        try:
            with fname.open() as fc:
                self.cfg = json.load(fc)
        except (OSError, UnicodeDecodeError, PermissionError, FileNotFoundError, json.JSONDecodeError):
            pass  # Caller deals with it
        return None if len(self.cfg) == 0 else self.config

    def save_config(  # noqa: PLR0913
        self,
        url: str,
        nvcf_url: str,
        key: str,
        org: str,
        team: str,
        backend: str,
        instance: str,
        gpu: str,
        timeout: int,
    ) -> dict[str, Any]:
        """Save configuration to the config file.

        Args:
            url: Base NGC URL
            nvcf_url: Base NVCF URL
            key: NGC NVCF API Key
            org: Organization ID or name
            team: Team name within the organization
            backend: Backend configuration
            instance: Instance configuration
            gpu: GPU configuration
            timeout: Unused

        Returns:
            The updated configuration dictionary.

        """
        self.cfg["url"] = url
        self.cfg["key"] = key
        self.cfg["org"] = org
        self.cfg["team"] = team
        self.cfg["nvcf_url"] = nvcf_url
        self.cfg["backend"] = backend
        self.cfg["gpu"] = gpu
        self.cfg["instance"] = instance
        self.cfg["timeout"] = timeout
        return self.cfg

    def get_cluster(
        self,
        ctx: Context,
        backend: str | None,
        gpu: str | None,
        instance: str | None,
    ) -> tuple[bool, str | None, str | None, str | None]:
        """Get cluster configuration from context or config.

        Args:
            ctx: Typer context
            backend: Backend configuration
            gpu: GPU configuration
            instance: Instance configuration

        Returns:
            Tuple containing success flag and cluster configuration values.

        """
        config = ctx.obj.get("config")

        if backend is None and config is not None:
            backend = config.get("backend")

        if gpu is None and config is not None:
            gpu = config.get("gpu")

        if instance is None and config is not None:
            instance = config.get("instance")

        if backend is None or gpu is None or instance is None:
            return False, backend, gpu, instance

        return True, backend, gpu, instance

    @property
    def config(self) -> dict[str, Any]:
        """Get the current configuration.

        Returns:
            The current configuration dictionary.

        """
        return self.cfg


registered_instances: dict[str, dict[str, str | type[NvcfBase] | typer.Typer]] = {}


def register_instance(
    ins_name: str,
    ins_help: str,
    ins_type: type[NvcfBase],
    ins_app: typer.Typer,
) -> None:
    """Register a new NVCF instance.

    Args:
        ins_name: Name of the instance
        ins_help: Help text for the instance
        ins_type: Type of the instance
        ins_app: Typer application for the instance

    """
    if registered_instances.get(ins_name) is not None:
        typer.echo(
            typer.style(
                f"INFO: Instance {ins_name} is already registered",
                fg=typer.colors.YELLOW,
                bg=typer.colors.BLACK,
                bold=True,
            ),
            err=False,
        )
        return
    registered_instances[ins_name] = {"help": ins_help, "type": ins_type, "app": ins_app}


def cc_client_instances() -> dict[str, Any]:
    """Get all registered instances.

    Returns:
        Dictionary of registered instances.

    """
    return registered_instances


def base_callback(  # noqa: PLR0913
    ctx: Context,
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
        str,
        Option(
            help="The Base NGC url",
            rich_help_panel="Common",
            envvar="NGC_BASE_URL",
        ),
    ] = "https://api.ngc.nvidia.com",
    nvcf_url: Annotated[
        str,
        Option(
            help="The Base NVCF url",
            rich_help_panel="Common",
            envvar="NVCF_BASE_URL",
        ),
    ] = "https://api.nvcf.nvidia.com",
    timeout: Annotated[
        int | None,
        Option(
            help="Unused",
            rich_help_panel="Common",
            envvar="NGC_NVCF_TIMEOUT",
            callback=validate_positive_integer,
        ),
    ] = 15,
) -> None:
    """Handle base callback for NVCF operations.

    Args:
        ctx: Typer context
        key: NGC NVCF API Key
        org: Organization ID or name
        team: Team name within the organization
        url: Base NGC URL
        nvcf_url: Base NVCF URL
        timeout: Request timeout in seconds

    """
    ins_name = ctx.command.name
    ins_detail = cc_client_instances().get(str(ins_name), {})
    ins_type = None if not ins_detail else ins_detail.get("type")
    if ins_type is None:
        typer.echo(
            typer.style(
                f"FATAL: Instance {ins_name} not registered",
                fg=typer.colors.RED,
                bg=typer.colors.BLACK,
                bold=True,
            ),
            err=True,
        )
        raise typer.Exit(code=1)
    # Instantiate the functionality
    nvcf_hdl = ins_type(url=url, nvcf_url=nvcf_url, key=key, org=org, team=team, timeout=timeout)

    if nvcf_hdl.config is None and ins_name != "config":
        err_msg = f"No Configurations found, Please run '{nvcf_hdl.exe} nvcf config set' to create configuration"
        nvcf_hdl.logger.error(err_msg)
        raise typer.Exit(code=1)
    ctx.obj = {
        "url": url,
        "nvcf_url": nvcf_url,
        "key": key,
        "org": org,
        "team": team,
        "timeout": timeout,
        "config": nvcf_hdl.config,
        "nvcfHdl": nvcf_hdl,
    }
