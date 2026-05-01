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

"""Manage and manipulate NVCF container images."""

import json
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, cast

import requests
import typer
from ngcbase.errors import (
    AccessDeniedException,
    ResourceAlreadyExistsException,
    ResourceNotFoundException,
)
from ngcsdk import Client
from rich.pretty import pprint
from rich.table import Table
from typer import Context, Option

from cosmos_curator.client.nvcf_cli.ncf.common import NotFoundError, NvcfBase, base_callback, register_instance

if TYPE_CHECKING:
    from registry.api.image import ImageAPI

_MODULE_NAME: str = "Image"


class ImageManager(NvcfBase):
    """Manager for NVCF container images.

    Provides methods to upload, download, delete, and list container images in the NVIDIA Cloud Function platform.
    """

    def __init__(self, url: str, nvcf_url: str, key: str, org: str, team: str, timeout: int) -> None:  # noqa: PLR0913
        """Initialize the ImageManager.

        Args:
            url: Base NGC URL
            nvcf_url: Base NVCF URL
            key: NGC NVCF API Key
            org: Organization ID or name
            team: Team name within the organization
            timeout: Request timeout in seconds

        """
        super().__init__(url=url, nvcf_url=nvcf_url, key=key, org=org, team=team, timeout=timeout)
        self.clnt: Client = Client()
        self.clnt.configure(api_key=self.key, org_name=self.org, team_name=self.team, ace_name="no-ace")
        self.image: ImageAPI = self.clnt.registry.image

    def upload_image(self, fname: str) -> dict[str, Any] | None:
        """Upload a container image using a JSON definition file.

        Args:
            fname: Path to the JSON definition file

        Returns:
            Dictionary with upload result details or None if upload fails

        """
        image = None
        ret = None
        try:
            with Path(fname).open() as fj:
                infod = json.load(fj)

            image = f"{infod.get('image')}"
            if not image.startswith("nvcr.io/"):
                image = f"nvcr.io/{self.org}/{image}"

            register = infod.get("definition", {})
            register["image"] = image

            ret = self.image.create(**register)
        except ResourceAlreadyExistsException:
            self.logger.info("%s already exist, updating tag", image)

        try:
            tag = infod.get("tag")
            upload = infod.get("definition", {})
            upload["image"] = f"{image}:{tag}"
            upload["default_yes"] = True
            upload["output"] = True

            self.image.push(**upload)

            self.logger.info("%s:%s Uploaded or tag updated", image, tag)
            if ret is not None:
                return cast("dict[str, Any]", ret.toDict())

        except (OSError, requests.RequestException, ValueError):
            self.logger.exception("Failed to upload %s: ", image)

        return None

    def download_image(self, iname: str) -> None:
        """Download a container image by name.

        Args:
            iname: Name of the image to download

        Raises:
            NotFoundError: If the image is not found
            Exception: For other errors

        """
        try:
            self.image.pull(image=f"{self.org}/{iname}")

        except ResourceNotFoundException as e:
            raise NotFoundError(_MODULE_NAME, name=iname) from e

        except AccessDeniedException as e:
            raise RuntimeError(e) from e

    def delete_image(self, iname: str) -> None:
        """Delete a container image by name.

        Args:
            iname: Name of the image to delete

        Raises:
            NotFoundError: If the image is not found
            Exception: For other errors

        """
        try:
            self.image.remove(pattern=f"{self.org}/{iname}", default_yes=True)

        except ResourceNotFoundException as e:
            raise NotFoundError(_MODULE_NAME, name=iname) from e

        except AccessDeniedException as e:
            raise RuntimeError(e) from e

    def list_all(self, *, all_accessible_orgs: bool) -> Table:
        """List all container images accessible to the organization.

        Args:
            all_accessible_orgs: Whether to list images from all accessible organizations

        Returns:
            A rich Table object containing image details

        Raises:
            NotFoundError: If no images are found
            Exception: For other errors

        """
        try:
            il = self.image.list()
        except Exception as e:
            raise RuntimeError(e) from e

        title_line = f"All Images {'that My Org Has Access to' if all_accessible_orgs else 'from My Org'}:"

        timages = Table(title=title_line)
        timages.add_column(header="Name", overflow="fold")
        timages.add_column(header="Tag", overflow="fold")
        timages.add_column(header="Size", overflow="fold")
        timages.add_column(header="Description", overflow="fold")
        timages.add_column(header="Publisher", width=15)
        timages.add_column(header="Org", overflow="fold")
        timages.add_column(header="Teams", overflow="fold")
        timages.add_column(header="LastUpdate", overflow="fold")

        for i in il:
            for ix in i:
                img_id = ix.toDict()

                name = img_id.get("name")
                tag = img_id.get("latestTag")
                size = str(img_id.get("latestImageSize"))
                desc = img_id.get("description")
                pub = img_id.get("publisher")
                org = ", ".join(img_id.get("sharedWithOrgs", []))
                team = ", ".join(img_id.get("sharedWithTeams", []))
                upddt = img_id.get("updatedDate")

                if not all_accessible_orgs and str(self.org) not in org and str(self.org) not in team:
                    continue

                timages.add_row(name, tag, size, desc, pub, org, team, upddt)

        return timages

    def list_detail(self, iname: str) -> dict[str, Any]:
        """Get details for a specific container image.

        Args:
            iname: Name of the image

        Returns:
            Dictionary with image details

        Raises:
            NotFoundError: If the image is not found
            Exception: For other errors

        """
        il = {}
        two_elem: int = 2
        four_elem: int = 4
        try:
            result = self.image.info(image=f"{self.org}/{iname}")
            if isinstance(result, tuple) and len(result) == two_elem:
                ils, _ = result
                il = ils.toDict()

            elif isinstance(result, tuple) and len(result) == four_elem:
                imgs, _manifest, _scan_list, arch_list = result
                images = imgs.toDict()
                arch_digest = [arch.get("digest") for arch in arch_list if arch.get("digest") is not None]

                image_list = images.get("images")
                ix = 0
                for img in image_list:
                    digest = img.get("digest")
                    if digest in arch_digest:
                        il[str(ix)] = img
                        ix += 1
            else:
                err_msg = f"unknown response received: {result}"
                raise RuntimeError(err_msg)

        except ResourceNotFoundException as e:
            raise NotFoundError(_MODULE_NAME, name=iname) from e

        except AccessDeniedException as e:
            raise RuntimeError(e) from e

        return il


nvcf_image = typer.Typer(
    context_settings={
        "max_content_width": 120,
    },
    pretty_exceptions_enable=False,
    no_args_is_help=True,
    callback=base_callback,
)
ins_name = "image"
ins_help = "NVCF Image management"
register_instance(ins_name=ins_name, ins_help=ins_help, ins_type=ImageManager, ins_app=nvcf_image)


@nvcf_image.command(name="list-images", help="List available images")
def nvcf_image_list_images(
    ctx: Context,
    *,
    all_images: Annotated[
        bool,
        Option(
            help="List all images from all organizations that I have access to",
            rich_help_panel="List",
        ),
    ] = False,
) -> None:
    """List all available container images.

    Args:
        ctx: Typer context containing the NVCF handler.
        all_images: Whether to list images from all accessible organizations.

    """
    nvcf_hdl = ctx.obj["nvcfHdl"]
    timages = nvcf_hdl.list_all(all_accessible_orgs=all_images)
    nvcf_hdl.console.print(timages)


@nvcf_image.command(
    name="list-image-detail",
    help="List details about an image",
    no_args_is_help=True,
)
def nvcf_image_list_image_detail(
    ctx: Context,
    iname: Annotated[
        str,
        Option(
            help="The Name of the Image 'image' and optional tag 'image:tag', with optional 'team/' prefix",
            rich_help_panel="Detail",
        ),
    ],
) -> None:
    """Display detailed information about a specific container image.

    Args:
        ctx: Typer context containing the NVCF handler.
        iname: Name of the image with optional tag and team prefix.

    """
    nvcf_hdl = ctx.obj["nvcfHdl"]
    il = nvcf_hdl.list_detail(iname=iname)
    pprint(il)


@nvcf_image.command(name="upload-image", help="Upload an image", no_args_is_help=True)
def nvcf_image_upload_image(
    ctx: Context,
    data_file: Annotated[
        Path,
        Option(
            help="The Name of the JSON Data file with all details",
            rich_help_panel="Upload",
            exists=True,
            file_okay=True,
            dir_okay=False,
            writable=False,
            readable=True,
            resolve_path=True,
        ),
    ],
) -> None:
    """Upload a container image using a JSON definition file.

    Args:
        ctx: Typer context containing the NVCF handler.
        data_file: Path to the JSON file containing image details.

    """
    nvcf_hdl = ctx.obj["nvcfHdl"]
    ret = nvcf_hdl.upload_image(fname=str(data_file))
    if ret is not None:
        pprint(ret)


@nvcf_image.command(name="download-image", help="Download an image", no_args_is_help=True)
def nvcf_image_download_image(
    ctx: Context,
    iname: Annotated[
        str,
        Option(
            help="The Name of the Image and version 'image:tag', with optional 'team/' prefix",
            rich_help_panel="Download",
        ),
    ],
) -> None:
    """Download a container image from the registry.

    Args:
        ctx: Typer context containing the NVCF handler.
        iname: Name of the image with version and optional team prefix.

    """
    nvcf_hdl = ctx.obj["nvcfHdl"]
    nvcf_hdl.download_image(iname=iname)


@nvcf_image.command(name="delete-image", help="Delete an image", no_args_is_help=True)
def nvcf_image_delete_image(
    ctx: Context,
    iname: Annotated[
        str,
        Option(
            help="The Name of the Image 'image' and optional version 'image:tag', with optional 'team/' prefix",
            rich_help_panel="Delete",
        ),
    ],
) -> None:
    """Delete a container image from the registry.

    Args:
        ctx: Typer context containing the NVCF handler.
        iname: Name of the image with optional version and team prefix.

    """
    nvcf_hdl = ctx.obj["nvcfHdl"]
    nvcf_hdl.delete_image(iname=iname)


if __name__ == "__main__":
    nvcf_image()
