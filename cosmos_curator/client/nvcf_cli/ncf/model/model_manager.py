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

"""Manage and manipulate NVCF model resources."""

from threading import RLock

from tqdm import tqdm

tqdm.set_lock(RLock())

# ruff: noqa: E402
import asyncio
import json
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, TypedDict, cast

import huggingface_hub.utils
import requests
import typer
from huggingface_hub import hf_hub_download, snapshot_download
from huggingface_hub.errors import (
    EntryNotFoundError,
    HfHubHTTPError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)
from ngcbase.errors import ResourceAlreadyExistsException, ResourceNotFoundException
from ngcsdk import Client
from rich.pretty import pprint
from rich.table import Table
from typer import Context, Option

from cosmos_curator.client.nvcf_cli.ncf.common import NotFoundError, NvcfBase, base_callback, register_instance

if TYPE_CHECKING:
    from registry.api.models import ModelAPI

_MODULE_NAME: str = "Model"


class ModelManager(NvcfBase):
    """Manager for NVCF model resources.

    Provides methods to upload, download, delete, and list models in the NVIDIA Cloud Function platform.
    """

    def progress(self) -> Callable[[int, int, int, int, int, int], None]:
        """Return a progress callback function for model uploads.

        Returns:
            A callback function that prints upload progress.

        """

        def __progress(  # noqa: PLR0913
            completed_b: int,
            failed_b: int,
            total_b: int,
            completed_c: int,
            failed_c: int,
            total_c: int,
        ) -> None:
            self.console.print(
                f"CompletedBytes: {completed_b} "
                f"FailedBytes: {failed_b} "
                f"TotalBytes: {total_b} "
                f"CompletedCount: {completed_c} "
                f"FailedCount: {failed_c} "
                f"TotalCount: {total_c}",
            )

        return __progress

    def __init__(self, url: str, nvcf_url: str, key: str, org: str, team: str, timeout: int) -> None:  # noqa: PLR0913
        """Initialize the ModelManager.

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
        self.model: ModelAPI = self.clnt.registry.model

    def upload_model(self, fname: str, src_path: str) -> dict[str, Any] | None:
        """Upload a model using a JSON definition file and source path.

        Args:
            fname: Path to the JSON definition file
            src_path: Path to the model source files

        Returns:
            Dictionary with upload result details or None if upload fails

        """
        with Path(fname).open() as fj:
            infod = json.load(fj)

        target = f"{self.org}/{infod.get('target')}"
        version = infod.get("version")

        register = infod.get("definition", {})
        register["target"] = target

        upload = infod.get("upload", {})
        upload["target"] = f"{target}:{version}"
        upload["source"] = src_path
        upload["progress_callback_func"] = self.progress()

        model = None
        try:
            model = self.model.create(**register)
        except ResourceAlreadyExistsException:
            self.logger.info("%s already exists, updating version", target)

        try:
            self.model.upload_version(**upload)
            self.logger.info("%s:%s uploaded successfully", target, version)
            if model is not None:
                return cast("dict[str, Any]", model.toDict())

        except (OSError, requests.RequestException, ValueError) as e:
            error_msg = f"Failed to upload model: {e!s}"
            raise RuntimeError(error_msg) from e

        return None

    def download_model(self, mname: str, dest: str) -> None:
        """Download a model by name to a destination directory.

        Args:
            mname: Name of the model to download
            dest: Destination directory path

        Raises:
            NotFoundError: If the model is not found
            Exception: For other errors

        """
        try:
            self.model.download_version(target=f"{self.org}/{mname}", destination=dest)

        except ResourceNotFoundException as e:
            raise NotFoundError(_MODULE_NAME, name=mname) from e

        except Exception:
            raise

    def delete_model(self, mname: str) -> None:
        """Delete a model by name.

        Args:
            mname: Name of the model to delete

        Raises:
            NotFoundError: If the model is not found
            Exception: For other errors

        """
        try:
            self.model.remove(target=f"{self.org}/{mname}")

        except ResourceNotFoundException as e:
            raise NotFoundError(_MODULE_NAME, name=mname) from e

        except Exception:
            raise

    def list_all(self, *, all_accessible_orgs: bool) -> Table:
        """List all models accessible to the organization.

        Args:
            all_accessible_orgs: Whether to list models from all accessible organizations

        Returns:
            A rich Table object containing model details

        Raises:
            NotFoundError: If no models are found
            Exception: For other errors

        """
        ml = self.model.list(org=self.org)

        title_line = "All Models"
        if all_accessible_orgs:
            title_line += " that My Org Has Access to"
        else:
            title_line += " from My Org"
        tmodels = Table(title=title_line)
        tmodels.add_column(header="Application", overflow="fold")
        tmodels.add_column(header="Name", overflow="fold")
        tmodels.add_column(header="Version", overflow="fold")
        tmodels.add_column(header="Size", overflow="fold")
        tmodels.add_column(header="Format", width=15)
        tmodels.add_column(header="Prec", width=6)
        tmodels.add_column(header="Publisher", width=15)
        tmodels.add_column(header="Org", overflow="fold")
        tmodels.add_column(header="Team", overflow="fold")
        tmodels.add_column(header="LastUpdate", overflow="fold")
        for m in ml:
            for mx in m:
                md = mx.toDict()
                app = md.get("application")
                name = md.get("name")
                version = md.get("latestVersionIdStr")
                size = str(md.get("latestVersionSizeInBytes"))
                format_val = md.get("modelFormat")
                precision = md.get("precision")
                pub = md.get("publisher")
                org = md.get("orgName")
                team = md.get("teamName")
                upddt = md.get("updatedDate")
                if not all_accessible_orgs and org != self.org:
                    continue
                tmodels.add_row(app, name, version, size, format_val, precision, pub, org, team, upddt)
        return tmodels

    def list_detail(self, mname: str) -> dict[str, Any]:
        """Get details for a specific model.

        Args:
            mname: Name of the model

        Returns:
            Dictionary with model details

        Raises:
            NotFoundError: If the model is not found
            Exception: For other errors

        """
        try:
            info = self.model.info(target=f"{self.org}/{mname}")
            return cast("dict[str, Any]", info.toDict())

        except ResourceNotFoundException as e:
            raise NotFoundError(_MODULE_NAME, name=mname) from e

        except Exception:
            raise

    class _ModelProps(TypedDict):
        """Model Properties loaded from the Model List JSON File."""

        model_id: str
        precision: str
        filelist: list[str] | None
        nvcf_model_id: str
        version: str

    async def _download_model_from_hf(
        self, model: _ModelProps, hf_token: str, download_dir: Path, cache_dir: Path | None = None
    ) -> None:
        """Asynchronously download an entire model or specific files from HF.

        Args:
            model: A model that will be downloaded and its properties
            hf_token: The HF Token
            download_dir: The directory under which the models will be downloaded
            cache_dir: HF Cache dir

        Raises:
            RuntimeError: On error

        """
        local_dir = download_dir / model["model_id"]
        local_dir.mkdir(parents=True, exist_ok=True)
        # Disable progress bars to avoid tqdm synchronization issues
        huggingface_hub.utils.disable_progress_bars()  # type: ignore[attr-defined]

        try:
            if model["filelist"] is None:
                await asyncio.to_thread(
                    snapshot_download,
                    repo_id=model["model_id"],
                    revision=model["version"],
                    local_dir=local_dir,
                    token=hf_token,
                    cache_dir=cache_dir,
                )
            else:
                tasks = [
                    asyncio.to_thread(
                        hf_hub_download,
                        repo_id=model["model_id"],
                        filename=file,
                        revision=model["version"],
                        local_dir=local_dir,
                        token=hf_token,
                        cache_dir=cache_dir,
                    )
                    for file in model["filelist"]
                ]
                await asyncio.gather(*tasks)
            self.logger.info("Model %s downloaded successfully", model["model_id"])
        except (RepositoryNotFoundError, RevisionNotFoundError, EntryNotFoundError, HfHubHTTPError, OSError) as e:
            mname = model["model_id"]
            msg = f"Failed to download {mname} : {e!s}"
            raise RuntimeError(msg) from e

    async def _download_all_models_from_hf(
        self, models: dict[str, _ModelProps], hf_token: str, download_dir: Path, cache_dir: Path | None
    ) -> None:
        """Start Asynchronous download tasks.

        Args:
            models: List of models and their properties
            hf_token: The HF Token
            download_dir: The directory under which the models will be downloaded
            cache_dir: HF Cache dir

        """
        tasks = [
            self._download_model_from_hf(model=model, hf_token=hf_token, download_dir=download_dir, cache_dir=cache_dir)
            for model in models.values()
        ]
        await asyncio.gather(*tasks)
        self.logger.info("All models downloaded successfully")

    def sync_models(
        self, models: dict[str, _ModelProps], hf_token: str, download_dir: Path, cache_dir: Path | None
    ) -> None:
        """Synchronize/updates NVCF Model Repository with HF Models.

        Args:
            models: List of models and their properties
            hf_token: The HF Token
            download_dir: The directory under which the models will be downloaded
            cache_dir: HF Cache dir

        """
        # First download all the models
        asyncio.run(
            self._download_all_models_from_hf(
                models=models,
                hf_token=hf_token,
                download_dir=download_dir,
                cache_dir=cache_dir,
            )
        )
        # Now upload, but only one model at a time
        for model in models.values():
            src_path = download_dir / model["model_id"]
            mdesc = model["model_id"].split("/", 1)
            app = mdesc[0]
            desc = mdesc[1] if len(mdesc) > 1 else app
            target = model["nvcf_model_id"] if self.team == "no-team" else f"{self.team}/{model['nvcf_model_id']}"
            descriptor = {
                "target": target,
                "version": model["version"],
                "definition": {
                    "application": app,
                    "framework": "PyTorch",
                    "model_format": "ckpt",
                    "precision": model["precision"],
                    "short_description": desc,
                    "overview_filename": None,
                    "bias_filename": None,
                    "explainability_filename": None,
                    "privacy_filename": None,
                    "safety_security_filename": None,
                    "display_name": desc,
                    "label": None,
                    "label_set": None,
                    "logo": None,
                    "public_dataset_name": None,
                    "public_dataset_link": None,
                    "public_dataset_license": None,
                    "built_by": None,
                    "publisher": None,
                },
                "upload": {
                    "gpu_model": None,
                    "memory_footprint": None,
                    "num_epochs": None,
                    "batch_size": None,
                    "accuracy_reached": None,
                    "description": None,
                    "link": None,
                    "link_type": None,
                    "dry_run": False,
                    "credential_files": None,
                    "metric_files": None,
                },
            }
            data_file: str | None = None
            with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as jf:
                json.dump(descriptor, jf)
                data_file = jf.name

            ret = self.upload_model(fname=data_file, src_path=str(src_path))
            if ret is not None:
                pprint(ret)


nvcf_model = typer.Typer(
    context_settings={
        "max_content_width": 120,
    },
    pretty_exceptions_enable=False,
    no_args_is_help=True,
    callback=base_callback,
)
ins_name = "model"
ins_help = "NVCF Model management"
register_instance(ins_name=ins_name, ins_help=ins_help, ins_type=ModelManager, ins_app=nvcf_model)


@nvcf_model.command(name="list-models", help="List available models")
def nvcf_model_list_models(
    ctx: Context,
    all_models: Annotated[bool, Option] = Option(
        default=False,
        help="List all models from all organizations that I have access to",
        rich_help_panel="List",
    ),
) -> None:
    """List all available models.

    Args:
        ctx: Typer context containing the NVCF handler.
        all_models: Whether to list models from all accessible organizations.

    """
    nvcf_hdl = ctx.obj["nvcfHdl"]
    tmodels = nvcf_hdl.list_all(all_accessible_orgs=all_models)
    nvcf_hdl.console.print(tmodels)


@nvcf_model.command(
    name="list-model-detail",
    help="List details about a model",
    no_args_is_help=True,
)
def nvcf_model_list_model_detail(
    ctx: Context,
    mname: Annotated[
        str,
        Option(
            help="The Name of the Model 'model' and optional version 'model:version', with optional 'team/' prefix",
            rich_help_panel="Detail",
        ),
    ],
) -> None:
    """Display detailed information about a specific model.

    Args:
        ctx: Typer context containing the NVCF handler.
        mname: Name of the model with optional version and team prefix.

    """
    nvcf_hdl = ctx.obj["nvcfHdl"]
    info = nvcf_hdl.list_detail(mname=mname)
    pprint(info)


@nvcf_model.command(name="upload-model", help="Upload a model to NVCF Model Registry", no_args_is_help=True)
def nvcf_model_upload_model(
    ctx: Context,
    data_file: Annotated[
        Path,
        typer.Option(
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
    src_path: Annotated[
        Path,
        Option(
            help="The upload source path",
            rich_help_panel="Upload",
            exists=True,
            file_okay=False,
            dir_okay=True,
            writable=False,
            readable=True,
            resolve_path=True,
        ),
    ],
) -> None:
    """Upload a model using a JSON definition file and source path.

    Args:
        ctx: Typer context containing the NVCF handler.
        data_file: Path to the JSON file containing model details.
        src_path: Path to the model source files.

    """
    nvcf_hdl = ctx.obj["nvcfHdl"]
    ret = nvcf_hdl.upload_model(fname=str(data_file), src_path=str(src_path))
    if ret is not None:
        pprint(ret)


@nvcf_model.command(name="download-model", help="Download a model from NVCF Model Registry", no_args_is_help=True)
def nvcf_model_download_model(
    ctx: Context,
    mname: Annotated[
        str,
        Option(
            help="The Name of the Model and version 'model:version', with optional 'team/' prefix",
            rich_help_panel="Download",
        ),
    ],
    save_path: Annotated[
        Path,
        Option(
            help="The download save path",
            rich_help_panel="Download",
            exists=True,
            file_okay=False,
            dir_okay=True,
            writable=True,
            readable=True,
            resolve_path=True,
        ),
    ],
) -> None:
    """Download a model from the registry.

    Args:
        ctx: Typer context containing the NVCF handler.
        mname: Name of the model with version and optional team prefix.
        save_path: Directory path where the model will be saved.

    """
    nvcf_hdl = ctx.obj["nvcfHdl"]
    nvcf_hdl.download_model(mname=mname, dest=str(save_path))


@nvcf_model.command(name="delete-model", help="Delete a model from NVCF Model Registry", no_args_is_help=True)
def nvcf_model_delete_model(
    ctx: Context,
    mname: Annotated[
        str,
        Option(
            help="The Name of the Model 'model' and optional version 'model:version', with optional 'team/' prefix",
            rich_help_panel="Delete",
        ),
    ],
) -> None:
    """Delete a model from the registry.

    Args:
        ctx: Typer context containing the NVCF handler.
        mname: Name of the model with optional version and team prefix.

    """
    nvcf_hdl = ctx.obj["nvcfHdl"]
    nvcf_hdl.delete_model(mname=mname)


@nvcf_model.command(
    name="sync-models", help="Synchronize all models from HF to NVCF Model Registry", no_args_is_help=True
)
def nvcf_model_sync_models(  # noqa: PLR0913
    ctx: Context,
    data_file: Annotated[
        Path,
        Option(
            help="The Name of the JSON Data file with details about all models",
            rich_help_panel="Sync",
            exists=True,
            file_okay=True,
            dir_okay=False,
            writable=False,
            readable=True,
            resolve_path=True,
        ),
    ],
    download_dir: Annotated[
        Path,
        Option(
            help="The Name of the directory where models will be downloaded from HF",
            rich_help_panel="Sync",
            exists=True,
            file_okay=False,
            dir_okay=True,
            writable=True,
            readable=True,
            resolve_path=True,
        ),
    ],
    cache_dir: Annotated[
        None | Path,
        Option(
            help="Optional HF Cache Directory",
            rich_help_panel="Sync",
            exists=True,
            file_okay=False,
            dir_okay=True,
            writable=True,
            readable=True,
            resolve_path=True,
        ),
    ] = None,
    token_file: Annotated[
        None | Path,
        Option(
            help="File with HF token on first line, instead of using from saved curator-configs, 2nd precedence",
            rich_help_panel="Sync",
            exists=True,
            file_okay=True,
            dir_okay=False,
            writable=False,
            readable=True,
            resolve_path=True,
        ),
    ] = None,
    token: Annotated[
        str | None,
        Option(
            help="HF token, used when present, instead of using from saved curator-configs, highest precedence",
            rich_help_panel="Sync",
        ),
    ] = None,
    mnames: Annotated[
        str | None,
        Option(
            help="Optional List of model names (first-level names from the data_file)",
            rich_help_panel="Sync",
        ),
    ] = None,
) -> None:
    """Synchronize all models from HF to NVCF Model registry.

    Args:
        ctx: Typer context containing the NVCF handler.
        data_file: The Name of the JSON Data file with details about all models
        download_dir: The Name of the directory where models will be downloaded from HF
        cache_dir: Optional HF Cache Directory
        token_file: Optional file name containing hf token on the first line, 2nd higest precedence
        token: Optional hf token, if present, other forms of tokens are ignored
        mnames: Optional comma separated model names to sync (first-level names from the data_file)

    """
    nvcf_hdl = ctx.obj["nvcfHdl"]
    hf_token = None
    if token is not None:
        hf_token = token
    elif token_file is not None:
        try:
            with token_file.open() as tf:
                hf_token = tf.readline().strip()
        except (OSError, UnicodeDecodeError) as e:
            typer.echo(
                typer.style(
                    f"ERROR: Failed to read {token_file}.",
                    fg=typer.colors.RED,
                    bg=typer.colors.BLACK,
                    bold=True,
                ),
            )
            raise typer.Exit(code=1) from e
    else:
        hf_token = nvcf_hdl.get_hf_token_from_config()
    if hf_token is None:
        typer.echo(
            typer.style(
                "ERROR: HF Token not available, cannot proceed.",
                fg=typer.colors.RED,
                bg=typer.colors.BLACK,
                bold=True,
            ),
        )
        raise typer.Exit(code=1)

    models: dict[str, ModelManager._ModelProps] = {}
    # Load the json file with all the models
    try:
        with data_file.open() as df:
            models = json.load(df)

    except (OSError, UnicodeDecodeError) as e:
        typer.echo(
            typer.style(
                f"ERROR: Failed to read {token_file}.",
                fg=typer.colors.RED,
                bg=typer.colors.BLACK,
                bold=True,
            ),
        )
        raise typer.Exit(code=1) from e
    mname_list: list[str] = []
    if mnames is not None:
        mname_list = mnames.split(",")
        missing = [mname for mname in mname_list if mname not in models]
        if len(missing) > 0:
            typer.echo(
                typer.style(
                    f"ERROR: Following names are not valid {missing}",
                    fg=typer.colors.RED,
                    bg=typer.colors.BLACK,
                    bold=True,
                ),
            )
            raise typer.Exit(code=1)
        # overwrite the models
        target_models = {mname: models[mname] for mname in mname_list}
        models = target_models
    try:
        nvcf_hdl.sync_models(
            models=models,
            hf_token=hf_token,
            download_dir=download_dir,
            cache_dir=cache_dir.expanduser() if cache_dir else None,
        )
    except RuntimeError as e:
        typer.echo(
            typer.style(
                f"ERROR: Failed to synchronize. models : {e!s}",
                fg=typer.colors.RED,
                bg=typer.colors.BLACK,
                bold=True,
            ),
        )
        raise typer.Exit(code=1) from e


if __name__ == "__main__":
    nvcf_model()
