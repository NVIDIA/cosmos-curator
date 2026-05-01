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

"""Manage and manipulate NVCF assets."""

import os
from pathlib import Path
from typing import Annotated, Any
from urllib.parse import quote as q

import requests
import typer
from rich.pretty import pprint
from rich.table import Table
from typer import Context, Option

from cosmos_curator.client.nvcf_cli.ncf.common import (
    NotFoundError,
    NvcfBase,
    NvcfClient,
    NVCFResponse,
    base_callback,
    register_instance,
)
from cosmos_curator.client.utils.validations import validate_positive_integer, validate_uuid

_EXCEPTION_MESSAGE = "unexpected empty reponse"


def _raise_runtime_err(msg: str) -> None:
    raise RuntimeError(msg)


class AssetManager(NvcfBase):
    """Manager for NVCF assets.

    Provides methods to upload, delete, and list assets in the NVIDIA Cloud Function platform.
    """

    def __init__(self, url: str, nvcf_url: str, key: str, org: str, team: str, timeout: int) -> None:  # noqa: PLR0913
        """Initialize the AssetManager.

        Args:
            url: Base NGC URL
            nvcf_url: Base NVCF URL
            key: NGC NVCF API Key
            org: Organization ID or name
            team: Team name within the organization
            timeout: Request timeout in seconds

        """
        super().__init__(url=url, nvcf_url=nvcf_url, key=key, org=org, team=team, timeout=timeout)
        self.ncg_api_hdl = NvcfClient(self.logger.getChild("ncgApiHdl"), self.url, self.key)
        self.nvcf_api_hdl = NvcfClient(self.logger.getChild("nvcfApiHdl"), self.nvcf_url, self.key)

    def do_upload(self, src_path: Path, desc: str, url: str) -> NVCFResponse | None:
        """Upload a file to the specified URL with a description.

        Args:
            src_path: Path to the file to upload
            desc: Description of the asset
            url: Upload URL

        Returns:
            NVCFResponse object or None if the upload fails

        """
        hdrs = {
            "Content-Type": "application/octet-stream",
            "x-amz-meta-nvcf-asset-description": desc,
        }
        with Path(src_path).open("rb") as fd:
            return self.nvcf_api_hdl.put_at(url=url, hdrs=hdrs, data=fd)

    def upload_asset(self, src_path: Path, desc: str, retries: int) -> dict[str, Any]:
        """Upload an asset to NVCF, retrying on failure.

        Args:
            src_path: Path to the file to upload
            desc: Description of the asset
            retries: Number of times to retry the upload

        Returns:
            Dictionary with upload result details

        """
        while retries > 0:
            try:
                data = {"contentType": "application/octet-stream", "description": desc}
                resp = self.nvcf_api_hdl.post("/v2/nvcf/assets", data=data)

                if resp is None:
                    raise RuntimeError(_EXCEPTION_MESSAGE)
                if resp.is_error:
                    _raise_runtime_err(resp.get_error(str(src_path)))

                asset_id = resp.get("assetId")
                upload_url = resp.get("uploadUrl")
                if asset_id is None or upload_url is None or not isinstance(upload_url, str):
                    _raise_runtime_err(f"unknown response received: {resp}")

                self.do_upload(src_path=src_path, desc=desc, url=str(upload_url))
                return {"Path": str(src_path), "AssetId": asset_id}

            except (OSError, requests.RequestException, ValueError) as e:
                retries -= 1
                if retries > 0:
                    self.logger.warning("Retrying upload for %s: %s. Attempts left: %d", src_path, e, retries)
                    continue

                raise RuntimeError(e) from e

        return {}

    def delete_asset(self, asset_id: str) -> None:
        """Delete an asset by its ID.

        Args:
            asset_id: The ID of the asset to delete

        Raises:
            NotFoundError: If the asset is not found
            Exception: For other errors

        """
        http_no_content: int = 204
        try:
            resp = self.nvcf_api_hdl.delete(f"/v2/nvcf/assets/{q(asset_id, safe='')}")

        except Exception as e:
            error_msg = f"failed to delete asset with id '{asset_id}'"
            raise RuntimeError(error_msg) from e

        if resp is None:  # take this out when .get can't return None anymore
            error_msg = "unexpected empty response"
            raise RuntimeError(error_msg)

        if resp.is_not_found:
            error_msg = "Asset"
            raise NotFoundError(error_msg, id=asset_id)

        if resp.is_error:
            error_msg = resp.get_error(f"asset with id '{asset_id}'")
            raise RuntimeError(error_msg)

        if resp.status != http_no_content:
            error_msg = f"unknown response received: {resp}"
            raise RuntimeError(error_msg)

    def list_all(self) -> Table:
        """List all assets in NVCF.

        Returns:
            A rich Table object containing asset IDs

        Raises:
            NotFoundError: If no assets are found
            Exception: For other errors

        """
        try:
            resp = self.nvcf_api_hdl.get("/v2/nvcf/assets")

        except Exception as e:
            error_msg = "failed to list assets"
            raise RuntimeError(error_msg) from e

        if resp is None:  # take this out when .get can't return None anymore
            error_msg = "unexpected empty response"
            raise RuntimeError(error_msg)

        if resp.is_error:
            error_msg = resp.get_error("assets")
            raise RuntimeError(error_msg)

        al: list[dict[str, Any]] = resp.get("assets", [])
        if len(al) == 0:
            error_msg = "No assets were found"
            raise NotFoundError(error_msg)

        out = Table(title="Assets")
        out.add_column(header="AssetId", overflow="fold")
        for a in al:
            out.add_row(a.get("assetId"))

        return out

    def list_detail(self, asset_id: str) -> dict[str, Any]:
        """Get details for a specific asset.

        Args:
            asset_id: The ID of the asset

        Returns:
            Dictionary with asset details

        Raises:
            NotFoundError: If the asset is not found
            Exception: For other errors

        """
        try:
            resp = self.nvcf_api_hdl.get(f"/v2/nvcf/assets/{q(asset_id, safe='')}")

        except Exception as e:
            error_msg = f"failed to list details for asset with id '{asset_id}'"
            raise RuntimeError(error_msg) from e

        if resp is None:  # take this out when .get can't return None anymore
            error_msg = "unexpected empty response"
            raise RuntimeError(error_msg)

        if resp.is_not_found:
            error_msg = "Asset"
            raise NotFoundError(error_msg, id=asset_id)

        if resp.is_error:
            error_msg = resp.get_error(f"asset with id '{asset_id}'")
            raise RuntimeError(error_msg)

        asset: dict[str, Any] = resp.get("asset", {})
        if not asset:
            error_msg = f"unknown response received: {resp}"
            raise RuntimeError(error_msg)

        return asset


nvcf_asset = typer.Typer(
    context_settings={
        "max_content_width": 120,
    },
    pretty_exceptions_enable=False,
    no_args_is_help=True,
    callback=base_callback,
)
ins_name = "asset"
ins_help = "NVCF Asset management"
register_instance(ins_name=ins_name, ins_help=ins_help, ins_type=AssetManager, ins_app=nvcf_asset)


@nvcf_asset.command(name="list-assets", help="List available assets")
def nvcf_asset_list_assets(
    ctx: Context,
) -> None:
    """List all available assets in NVCF.

    Args:
        ctx: Typer context containing the NVCF handler.

    Raises:
        typer.Exit: If no assets are found or if there's an error listing assets.

    """
    nvcf_hdl = ctx.obj["nvcfHdl"]
    try:
        table = nvcf_hdl.list_all()
        nvcf_hdl.console.print(table)

    except NotFoundError as e:
        nvcf_hdl.logger.warning(e)
        raise typer.Exit(code=1) from e

    except Exception as e:
        error_msg = f"Could not list assets: {e!s}"
        nvcf_hdl.logger.error(error_msg)  # noqa: TRY400
        raise typer.Exit(code=1) from e


@nvcf_asset.command(
    name="list-asset-detail",
    help="List details about a asset",
    no_args_is_help=True,
)
def nvcf_asset_list_asset_detail(
    ctx: Context,
    assetid: Annotated[
        str,
        Option(
            help="The AssetID",
            rich_help_panel="Detail",
            callback=validate_uuid,
        ),
    ],
) -> None:
    """Display detailed information about a specific asset.

    Args:
        ctx: Typer context containing the NVCF handler.
        assetid: The UUID of the asset to get details for.

    Raises:
        typer.Exit: If the asset is not found or if there's an error getting asset details.

    """
    nvcf_hdl = ctx.obj["nvcfHdl"]
    try:
        asset = nvcf_hdl.list_detail(asset_id=assetid)
        if asset is not None:
            pprint(asset, expand_all=True)

    except NotFoundError as e:
        nvcf_hdl.console.print(e)
        raise typer.Exit(code=1) from e

    except Exception as e:
        error_msg = f"Could not list asset details: {e!s}"
        nvcf_hdl.logger.error(error_msg)  # noqa: TRY400
        raise typer.Exit(code=1) from e


@nvcf_asset.command(name="upload-asset", help="Upload an asset", no_args_is_help=True)
def nvcf_asset_upload_asset(
    ctx: Context,
    src_path: Annotated[
        Path,
        Option(
            help="The full path name of the file to upload",
            rich_help_panel="Upload",
            exists=True,
            file_okay=True,
            dir_okay=False,
            writable=False,
            readable=True,
            resolve_path=True,
        ),
    ],
    description: Annotated[
        str,
        Option(
            help="A short description in quotes",
            rich_help_panel="Upload",
        ),
    ],
    retries: Annotated[
        int,
        Option(
            help="Number of times to retry the upload",
            rich_help_panel="Upload",
            callback=validate_positive_integer,
        ),
    ] = 7,
) -> None:
    """Upload a file as an asset to NVCF.

    Args:
        ctx: Typer context containing the NVCF handler.
        src_path: Path to the file to upload.
        description: Short description of the asset.
        retries: Number of times to retry the upload if it fails.

    Raises:
        typer.Exit: If the upload fails after all retries.

    """
    nvcf_hdl = ctx.obj["nvcfHdl"]

    try:
        if not Path(src_path).exists():
            error_msg = f"file {src_path} does not exist"
            _raise_runtime_err(error_msg)

        if not os.access(src_path, os.R_OK):
            error_msg = f"file {src_path} is not accessible"
            _raise_runtime_err(error_msg)

        resp = nvcf_hdl.upload_asset(src_path=src_path, desc=description, retries=retries)
        if resp is not None:
            pprint(resp, expand_all=True)

    except Exception as e:
        error_msg = f"Could not upload asset: {e!s}"
        nvcf_hdl.logger.error(error_msg)  # noqa: TRY400
        raise typer.Exit(code=1) from e


@nvcf_asset.command(name="delete-asset", help="Delete an asset", no_args_is_help=True)
def nvcf_asset_delete_asset(
    ctx: Context,
    assetid: Annotated[
        str,
        Option(
            help="The AssetID to delete",
            rich_help_panel="Delete",
            callback=validate_uuid,
        ),
    ],
) -> None:
    """Delete an asset from NVCF.

    Args:
        ctx: Typer context containing the NVCF handler.
        assetid: The UUID of the asset to delete.

    Raises:
        typer.Exit: If the asset is not found or if there's an error deleting the asset.

    """
    nvcf_hdl = ctx.obj["nvcfHdl"]
    try:
        nvcf_hdl.delete_asset(asset_id=assetid)

    except NotFoundError as e:
        nvcf_hdl.console.print(e)

    except Exception as e:
        error_msg = f"Could not delete asset: {e!s}"
        nvcf_hdl.logger.error(error_msg)  # noqa: TRY400
        raise typer.Exit(code=1) from e

    else:
        nvcf_hdl.logger.info("Asset with id '%s' was deleted", assetid)


if __name__ == "__main__":
    nvcf_asset()
