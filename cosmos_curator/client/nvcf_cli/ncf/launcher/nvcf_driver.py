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

"""Drive and control NVCF instance operations."""

import base64
import tempfile
from pathlib import Path
from typing import Annotated

import typer
from rich.pretty import pprint
from typer import Context, Option

from cosmos_curator.client.nvcf_cli.ncf.common import (
    NotFoundError,
    base_callback,
    register_instance,
)
from cosmos_curator.client.utils.validations import validate_in, validate_positive_integer, validate_uuid

from .nvcf_helper import NvcfHelper

nvcf = typer.Typer(
    context_settings={
        "max_content_width": 120,
    },
    pretty_exceptions_enable=False,
    no_args_is_help=True,
    callback=base_callback,
)
ins_name = "function"
ins_help = "NVCF Function management"
register_instance(ins_name=ins_name, ins_help=ins_help, ins_type=NvcfHelper, ins_app=nvcf)

temp_dir = Path(tempfile.gettempdir())


def _raise_runtime_err(msg: str) -> None:
    """Raise a RuntimeError with the given message.

    Args:
        msg: The error message to include in the exception.

    Raises:
        RuntimeError: Always raises this exception with the provided message.

    """
    raise RuntimeError(msg)


def _get_s3_config_str(s3_config_file: Path | None) -> str | None:
    """Get the base64-encoded S3 config string from a file.

    Args:
        s3_config_file: Path to the S3 configuration file.

    Returns:
        S3 configuration string or None if not provided.

    """
    s3_config = None
    if s3_config_file is not None:
        try:
            with Path.open(s3_config_file, "rb") as af:
                s3_config = base64.b64encode(af.read()).decode()
        except Exception as e:  # noqa: BLE001
            _raise_runtime_err(f"Failed to process {s3_config_file} {e!s}")

    return s3_config


@nvcf.command(name="list-clusters", help="List available clusters")
def nvcf_list_clusters(
    ctx: Context,
) -> None:
    """List all available NVCF clusters.

    Args:
        ctx: The Typer context object.

    Returns:
        None.

    """
    nvcf_hdl = ctx.obj["nvcfHdl"]
    try:
        resp = nvcf_hdl.nvcf_helper_list_clusters()
        nvcf_hdl.console.print(resp)
    except Exception as e:
        error_msg = f"Could not list clusters: {e!s}"
        nvcf_hdl.logger.error(error_msg)  # noqa: TRY400
        raise typer.Exit(code=1) from e


@nvcf.command(name="list-functions", help="List available functions")
def nvcf_list_functions(
    ctx: Context,
) -> None:
    """List all available NVCF functions.

    Args:
        ctx: The Typer context object.

    Returns:
        None.

    """
    nvcf_hdl = ctx.obj["nvcfHdl"]
    try:
        resp = nvcf_hdl.nvcf_helper_list_functions()
        nvcf_hdl.console.print(resp)
    except Exception as e:
        error_msg = f"Could not list functions: {e!s}"
        nvcf_hdl.logger.error(error_msg)  # noqa: TRY400
        raise typer.Exit(code=1) from e


@nvcf.command(name="list-function-detail", help="List details about a function")
def nvcf_list_function_detail(
    ctx: Context,
    name: Annotated[
        str,
        Option(
            help="The Name of the function",
            rich_help_panel="Function Detail",
        ),
    ],
) -> None:
    """List detailed information about a specific NVCF function.

    Args:
        ctx: The Typer context object.
        name: The name of the function to get details for.

    Returns:
        None.

    """
    nvcf_hdl = ctx.obj["nvcfHdl"]
    try:
        resp = nvcf_hdl.nvcf_helper_list_function_detail(name)
        pprint(resp, expand_all=True)

    except NotFoundError as e:
        nvcf_hdl.console.print(f"Function with name '{name}' not found")
        raise typer.Exit(code=1) from e

    except Exception as e:
        error_msg = f"Could not list function details: {e!s}"
        nvcf_hdl.logger.error(error_msg)  # noqa: TRY400
        raise typer.Exit(code=1) from e


@nvcf.command(name="create-function", help="Create a function", no_args_is_help=True)
def nvcf_create_function(  # noqa: PLR0913
    ctx: Context,
    name: Annotated[
        str,
        Option(
            help="The Name of the function to create",
            rich_help_panel="Create",
        ),
    ],
    data_file: Annotated[
        Path,
        Option(
            help="JSON file with list of [tags, models, secrets, resources, envs] to be sent to the service",
            rich_help_panel="Create",
            exists=True,
            file_okay=True,
            dir_okay=False,
            writable=False,
            readable=True,
            resolve_path=True,
        ),
    ],
    image: Annotated[
        str,
        Option(
            help="The container image name, required for Container based functions",
            rich_help_panel="Create",
        ),
    ] = "",
    inference_ep: Annotated[
        str,
        Option(
            help="The endpoint to access the inference service",
            rich_help_panel="Create",
        ),
    ] = "/v1/run_pipeline",
    inference_port: Annotated[
        int,
        Option(
            help="The inference service port",
            rich_help_panel="Create",
            callback=validate_in(range(65536), excludes=(8080, 8010)),
        ),
    ] = 8000,
    health_ep: Annotated[
        str,
        Option(
            help="The endpoint to access the health of service",
            rich_help_panel="Create",
        ),
    ] = "/health",
    health_port: Annotated[
        int,
        Option(
            help="The port to access health endpoint, defaults to inference_port",
            rich_help_panel="Create",
            callback=validate_in(range(65536), excludes=(8080, 8010)),
        ),
    ] = 0,
    args: Annotated[
        str | None,
        Option(
            help="arguments be sent to the service",
            rich_help_panel="Create",
        ),
    ] = None,
    helm_chart: Annotated[
        str | None,
        Option(
            help="URL to chart",
            # Should this be just a name/version?
            # Generate based on helm.ngc.nvidia.com/<org>/charts/ then <n>-<version>.tgz
            # They can't actually specify an arbitrary URL here.
            rich_help_panel="Create",
        ),
    ] = None,
    helm_service_name: Annotated[
        str | None,
        Option(
            help="expected service name contained within the chart",
            rich_help_panel="Create",
        ),
    ] = "cosmos-curator",
) -> None:
    """Create a new NVCF function.

    Args:
        ctx: The Typer context object.
        name: The name of the function to create.
        data_file: JSON file containing function configuration.
        image: Container image name for the function.
        inference_ep: Endpoint for inference service.
        inference_port: Port for inference service.
        health_ep: Endpoint for health checks.
        health_port: Port for health checks.
        args: Additional arguments for the service.
        helm_chart: URL to Helm chart.
        helm_service_name: Expected service name in the Helm chart.

    Returns:
        None.

    """
    nvcf_hdl = ctx.obj["nvcfHdl"]
    if health_port <= 0:
        health_port = inference_port

    try:
        resp = nvcf_hdl.nvcf_helper_create_function(
            name,
            image,
            inference_ep,
            inference_port,
            health_ep,
            health_port,
            args,
            data_file,
            helm_chart,
            helm_service_name,
        )
        resp["name"] = name
        nvcf_hdl.store_ids(resp)

        nvcf_hdl.console.print(
            f"Function with name '{resp['name']}', id '{resp['id']}', and version '{resp['version']}' created"
        )

    except Exception as e:
        error_msg = f"Could not create function '{name}': {e!s}"
        nvcf_hdl.logger.error(error_msg)  # noqa: TRY400
        raise typer.Exit(code=1) from e


@nvcf.command(
    name="deploy-function",
    help="Deploy a function after creation",
    no_args_is_help=True,
)
def nvcf_deploy_function(  # noqa: PLR0913
    ctx: Context,
    data_file: Annotated[
        Path,
        Option(
            help="JSON file with additional data to be sent to the deployment",
            rich_help_panel="Deploy",
            exists=True,
            file_okay=True,
            dir_okay=False,
            writable=False,
            readable=True,
            resolve_path=True,
        ),
    ],
    backend: Annotated[
        str | None,
        Option(
            help="The Name of Backend CSP [required]",
            rich_help_panel="Deploy",
            envvar="NVCF_BACKEND",
        ),
    ] = None,
    gpu: Annotated[
        str | None,
        Option(
            help="The GPU Hardware Type [required]",
            rich_help_panel="Deploy",
            envvar="NVCF_GPU_TYPE",
        ),
    ] = None,
    instance: Annotated[
        str | None,
        Option(
            help="The Hardware Instance Type [required]",
            rich_help_panel="Deploy",
            envvar="NVCF_INSTANCE_TYPE",
        ),
    ] = None,
    funcid: Annotated[
        str | None,
        Option(
            help="The Id of the function to deploy [required]",
            rich_help_panel="Deploy",
            callback=validate_uuid,
        ),
    ] = None,
    version: Annotated[
        str | None,
        Option(
            help="The VersionId of the function [required]",
            rich_help_panel="Deploy",
            callback=validate_uuid,
        ),
    ] = None,
    min_instances: Annotated[
        int,
        Option(
            help="Minimum Number of Instances to deploy",
            rich_help_panel="Deploy",
            callback=validate_positive_integer,
        ),
    ] = 1,
    max_instances: Annotated[
        int,
        Option(
            help="Maximum Number of Instances to deploy",
            rich_help_panel="Deploy",
            callback=validate_positive_integer,
        ),
    ] = 1,
    max_concurrency: Annotated[
        int,
        Option(
            help="Maximum Number of concurrent requests processed by this deployment",
            rich_help_panel="Deploy",
            callback=validate_positive_integer,
        ),
    ] = 1,
    instance_count: Annotated[
        int,
        Option(
            help="Maximum Number of GPU worker nodes to be assigned to a function instance",
            rich_help_panel="Deploy",
            callback=validate_positive_integer,
        ),
    ] = 1,
) -> None:
    """Deploy an NVCF function after creation.

    Args:
        ctx: The Typer context object.
        data_file: JSON file with deployment configuration.
        backend: Name of the backend CSP.
        gpu: GPU hardware type.
        instance: Hardware instance type.
        funcid: Function ID to deploy.
        version: Function version ID.
        min_instances: Minimum number of instances.
        max_instances: Maximum number of instances.
        max_concurrency: Maximum concurrent requests.
        instance_count: Number of GPU worker nodes.

    Returns:
        None.

    """
    nvcf_hdl = ctx.obj["nvcfHdl"]
    try:
        ok, funcid, version = nvcf_hdl.id_version(funcid, version)
        if not ok:
            error_msg = "id and version are required"
            _raise_runtime_err(error_msg)

        _, backend, gpu, instance = nvcf_hdl.get_cluster(ctx, backend, gpu, instance)
        if backend is None:
            error_msg = "backend is required"
            _raise_runtime_err(error_msg)
        if gpu is None:
            error_msg = "gpu is required"
            _raise_runtime_err(error_msg)
        if instance is None:
            error_msg = "instance is required"
            _raise_runtime_err(error_msg)

        _ = nvcf_hdl.nvcf_helper_deploy_function(
            funcid,
            version,
            backend,
            gpu,
            instance,
            min_instances,
            max_instances,
            max_concurrency,
            data_file,
            instance_count,
        )
        nvcf_hdl.console.print(
            f"Function with id '{funcid}' and version '{version}' is being deployed; "
            "to check status: cosmos-curator nvcf function get-deployment-detail"
        )

    except Exception as e:
        error_msg = f"Could not deploy function: {e!s}"
        nvcf_hdl.logger.error(error_msg)  # noqa: TRY400
        raise typer.Exit(code=1) from e


@nvcf.command(name="invoke-batch", help="Invoke a batch of ACTIVE functions", no_args_is_help=True)
def nvcf_invoke_batch(  # noqa: PLR0913
    ctx: Context,
    data_file: Annotated[
        Path,
        Option(
            help="Template JSON file with arguments to be sent to the function",
            rich_help_panel="InvokeBatch",
            exists=True,
            file_okay=True,
            dir_okay=False,
            writable=False,
            readable=True,
            resolve_path=True,
        ),
    ],
    id_file: Annotated[
        Path,
        Option(
            help="JSON file with list of [{func:xx, vers:yy} pairs]",
            rich_help_panel="InvokeBatch",
            exists=True,
            file_okay=True,
            dir_okay=False,
            writable=False,
            readable=True,
            resolve_path=True,
        ),
    ],
    job_variant_file: Annotated[
        Path,
        Option(
            help="JSON file with list of variants to data-file eg: [{<input>:xx, <output>:yy} pairs]",
            rich_help_panel="InvokeBatch",
            exists=True,
            file_okay=True,
            dir_okay=False,
            writable=False,
            readable=True,
            resolve_path=True,
        ),
    ],
    s3_config_file: Annotated[
        Path,
        Option(
            help="Path to the S3 configuration file",
            rich_help_panel="InvokeBatch",
            exists=True,
            file_okay=True,
            dir_okay=False,
            writable=False,
            readable=True,
            resolve_path=True,
        ),
    ],
    legacy_cf: Annotated[bool, Option] = Option(
        default=False,
        help=("Pass this flag for legacy cloud functions."),
        rich_help_panel="InvokeBatch",
    ),
    retry_cnt: Annotated[
        int,
        Option(
            help="Number of times the invoke will be retried before failing",
            rich_help_panel="InvokeBatch",
            callback=validate_positive_integer,
        ),
    ] = 2,
    retry_delay: Annotated[
        int,
        Option(
            help="Wait time between each retry (in seconds)",
            rich_help_panel="InvokeBatch",
            callback=validate_positive_integer,
        ),
    ] = 300,
    out_dir: Annotated[
        Path,
        Option(
            help="Output dir in case content can be downloaded",
            rich_help_panel="InvokeBatch",
            exists=True,
            file_okay=False,
            dir_okay=True,
            writable=True,
            readable=True,
            resolve_path=True,
        ),
    ] = temp_dir,
) -> None:
    """Invoke a batch of active NVCF functions.

    Args:
        ctx: The Typer context object.
        data_file: Template JSON file with function arguments.
        id_file: JSON file with function and version IDs.
        job_variant_file: JSON file with job variants.
        s3_config_file: Path to S3 configuration file.
        retry_cnt: Number of retry attempts.
        retry_delay: Delay between retries in seconds.
        legacy_cf: Flag for legacy cloud functions.
        out_dir: Output dir in case content can be downloaded

    Returns:
        None.

    """
    nvcf_hdl = ctx.obj["nvcfHdl"]
    try:
        nvcf_hdl.nvcf_helper_invoke_batch(
            data_file=data_file,
            id_file=id_file,
            job_variant_file=job_variant_file,
            ddir=str(out_dir),
            s3_config=_get_s3_config_str(s3_config_file),
            legacy_cf=legacy_cf,
            retry_cnt=retry_cnt,
            retry_delay=retry_delay,
        )
    except Exception as e:
        error_msg = f"Could not invoke batch: {e!s}"
        nvcf_hdl.logger.error(error_msg)  # noqa: TRY400
        raise typer.Exit(code=1) from e


@nvcf.command(name="invoke-function", help="Invoke an ACTIVE function", no_args_is_help=True)
def nvcf_invoke_function(  # noqa: PLR0913
    ctx: Context,
    data_file: Annotated[
        Path,
        Option(
            help="JSON file with additional data to be sent to the deployment",
            rich_help_panel="Invoke",
            exists=True,
            file_okay=True,
            dir_okay=False,
            writable=False,
            readable=True,
            resolve_path=True,
        ),
    ],
    prompt_file: Annotated[
        Path | None,
        Option(
            help="Text file containing prompt to append to captioning_prompt_variant",
            rich_help_panel="Invoke",
            exists=True,
            file_okay=True,
            dir_okay=False,
            writable=False,
            readable=True,
            resolve_path=True,
        ),
    ] = None,
    wait: Annotated[bool, Option] = Option(
        default=True,
        help="Wait for request to complete by polling status",
        rich_help_panel="Invoke",
    ),
    legacy_cf: Annotated[bool, Option] = Option(
        default=False,
        help=(
            "Pass this flag for legacy cloud functions.\n\n"
            "[yellow]When assetid/assetfile is passed, this is forced to True[/yellow]"
        ),
        rich_help_panel="Invoke",
    ),
    funcid: Annotated[
        str | None,
        Option(
            help="The Id of the function to invoke",
            rich_help_panel="Invoke",
            callback=validate_uuid,
        ),
    ] = None,
    version: Annotated[
        str | None,
        Option(
            help="VersionId of the function",
            rich_help_panel="Invoke",
            callback=validate_uuid,
        ),
    ] = None,
    assetid: Annotated[
        str | None,
        Option(
            help="Optional comma delimited AssetID list to pass to the function",
            rich_help_panel="Invoke",
            callback=validate_uuid,
        ),
    ] = None,
    assetfile: Annotated[
        Path | None,
        Option(
            help="Optional file name containing list of AssetIDs to pass to the function, one per line",
            rich_help_panel="Invoke",
            exists=True,
            file_okay=True,
            dir_okay=False,
            writable=False,
            readable=True,
            resolve_path=True,
        ),
    ] = None,
    s3_config_file: Annotated[
        Path | None,
        Option(
            help="Path to the S3 configuration file",
            rich_help_panel="Invoke",
            exists=True,
            file_okay=True,
            dir_okay=False,
            writable=False,
            readable=True,
            resolve_path=True,
        ),
    ] = None,
    retry_cnt: Annotated[
        int,
        Option(
            help="Number of times the invoke will be retried before failing",
            rich_help_panel="Invoke",
            callback=validate_positive_integer,
        ),
    ] = 2,
    retry_delay: Annotated[
        int,
        Option(
            help="Wait time between each retry (in seconds)",
            rich_help_panel="Invoke",
            callback=validate_positive_integer,
        ),
    ] = 300,
    out_dir: Annotated[
        Path,
        Option(
            help="Output dir in case content can be downloaded",
            rich_help_panel="Invoke",
            exists=True,
            file_okay=False,
            dir_okay=True,
            writable=True,
            readable=True,
            resolve_path=True,
        ),
    ] = temp_dir,
) -> None:
    """Invoke an active NVCF function.

    Args:
        ctx: The Typer context object.
        data_file: JSON file with function data.
        prompt_file: Text file with prompt to append.
        funcid: Function ID to invoke.
        version: Function version ID.
        assetid: Comma-delimited list of asset IDs.
        assetfile: File containing asset IDs.
        s3_config_file: Path to S3 configuration.
        wait: Whether to wait for completion.
        retry_cnt: Number of retry attempts.
        retry_delay: Delay between retries in seconds.
        legacy_cf: Flag for legacy cloud functions.
        out_dir: Output dir in case content can be downloaded

    Returns:
        None.

    """
    nvcf_hdl = ctx.obj["nvcfHdl"]
    try:
        ok, funcid, version = nvcf_hdl.id_version(funcid, version)
        if not ok:
            error_msg = "id and version are required"
            _raise_runtime_err(error_msg)

        if assetid is not None and assetfile is not None:
            error_msg = "assetid and assetfile are mutually exclusive"
            _raise_runtime_err(error_msg)

        if assetid is not None:
            legacy_cf = True
            asset_id = assetid
        elif assetfile is not None:
            legacy_cf = True
            with Path(assetfile).open() as f:
                asset_id = f.read().strip()
        else:
            asset_id = None

        if wait:
            nvcf_hdl.nvcf_helper_invoke_wait_retry_function(
                funcid=funcid,
                ddir=str(out_dir),
                version=version,
                data_file=data_file,
                prompt_file=prompt_file,
                asset_id=asset_id,
                s3_config=_get_s3_config_str(s3_config_file),
                legacy_cf=legacy_cf,
                wait=wait,
                retry_cnt=retry_cnt,
                retry_delay=retry_delay,
            )
        else:
            resp = nvcf_hdl.nvcf_helper_invoke_function(
                funcid=funcid,
                ddir=str(out_dir),
                version=version,
                data_file=data_file,
                prompt_file=prompt_file,
                asset_id=asset_id,
                s3_config=_get_s3_config_str(s3_config_file),
            )
            nvcf_hdl.console.print(f"RequestId: {resp['reqid']}")

    except Exception as e:
        error_msg = f"Could not invoke function: {e!s}"
        nvcf_hdl.logger.error(error_msg)  # noqa: TRY400
        raise typer.Exit(code=1) from e


@nvcf.command(
    name="get-request-status",
    help="Get status of a function request",
    no_args_is_help=True,
)
def nvcf_get_request_status(  # noqa: PLR0913
    ctx: Context,
    reqid: Annotated[
        str,
        Option(
            help="The Request Id of the function",
            rich_help_panel="Request",
            callback=validate_uuid,
        ),
    ],
    funcid: Annotated[
        str | None,
        Option(
            help="The Id of the function that was used in invoke",
            rich_help_panel="Request",
            callback=validate_uuid,
        ),
    ] = None,
    version: Annotated[
        str | None,
        Option(
            help="VersionId of the function that was used in invoke",
            rich_help_panel="Request",
            callback=validate_uuid,
        ),
    ] = None,
    out_dir: Annotated[
        Path,
        Option(
            help="Output dir in case content can be downloaded",
            rich_help_panel="Request",
            exists=True,
            file_okay=False,
            dir_okay=True,
            writable=True,
            readable=True,
            resolve_path=True,
        ),
    ] = temp_dir,
    wait: Annotated[bool, Option] = Option(
        default=True,
        help="Wait for request to complete by polling status",
        rich_help_panel="Request",
    ),
    legacy_cf: Annotated[bool, Option] = Option(
        default=False,
        help="Pass this flag for legacy cloud functions.\n\n[red]This support will be removed in future[/red]",
        rich_help_panel="Request",
    ),
) -> None:
    """Get the status of an NVCF function request.

    Args:
        ctx: The Typer context object.
        reqid: Request ID to check.
        funcid: Function ID used in invoke.
        version: Function version ID used in invoke.
        wait: Whether to wait for completion.
        legacy_cf: Flag for legacy cloud functions.
        out_dir: Output dir in case content can be downloaded

    Returns:
        None.

    """
    nvcf_hdl = ctx.obj["nvcfHdl"]
    try:
        ok, funcid, version = nvcf_hdl.id_version(funcid, version)
        if not ok:
            error_msg = "id and version are required"
            _raise_runtime_err(error_msg)

        nvcf_hdl.nvcf_helper_get_request_status_with_wait(
            reqid=reqid,
            ddir=str(out_dir),
            funcid=funcid,
            version=version,
            legacy_cf=legacy_cf,
            wait=wait,
        )
    except Exception as e:
        error_msg = f"Could not get request status: {e!s}"
        nvcf_hdl.logger.error(error_msg)  # noqa: TRY400
        raise typer.Exit(code=1) from e


@nvcf.command(
    name="terminate-request",
    help="Terminate a function request",
    no_args_is_help=True,
)
def nvcf_terminate_request(
    ctx: Context,
    reqid: Annotated[
        str,
        Option(
            help="The Request Id of the function",
            rich_help_panel="Request",
            callback=validate_uuid,
        ),
    ],
    funcid: Annotated[
        str | None,
        Option(
            help="The Id of the function that was used in invoke",
            rich_help_panel="Request",
            callback=validate_uuid,
        ),
    ] = None,
    version: Annotated[
        str | None,
        Option(
            help="VersionId of the function that was used in invoke",
            rich_help_panel="Request",
            callback=validate_uuid,
        ),
    ] = None,
) -> None:
    """Terminate an NVCF function request.

    Args:
        ctx: The Typer context object.
        reqid: Request ID to terminate.
        funcid: Function ID used in invoke.
        version: Function version ID used in invoke.

    Returns:
        None.

    """
    nvcf_hdl = ctx.obj["nvcfHdl"]
    try:
        ok, funcid, version = nvcf_hdl.id_version(funcid, version)
        if not ok:
            error_msg = "id and version are required"
            _raise_runtime_err(error_msg)

        resp = nvcf_hdl.nvcf_helper_terminate_request(
            reqid=reqid,
            funcid=funcid,
            version=version,
        )
        if resp is not None:
            nvcf_hdl.console.print(f"{resp}")

    except Exception as e:
        error_msg = f"Could not terminate request: {e!s}"
        nvcf_hdl.logger.error(error_msg)  # noqa: TRY400
        raise typer.Exit(code=1) from e


@nvcf.command(
    name="get-deployment-detail",
    help="Get details about a deployment",
    no_args_is_help=False,
)
def nvcf_get_deployment_detail(
    ctx: Context,
    funcid: Annotated[
        str | None,
        Option(
            help="The id of the function",
            rich_help_panel="Deployment-Detail",
            callback=validate_uuid,
        ),
    ] = None,
    version: Annotated[
        str | None,
        Option(
            help="The version of the function",
            rich_help_panel="Deployment-Detail",
            callback=validate_uuid,
        ),
    ] = None,
) -> None:
    """Get details about an NVCF function deployment.

    Args:
        ctx: The Typer context object.
        funcid: Function ID to get details for.
        version: Function version ID.

    Returns:
        None.

    """
    nvcf_hdl = ctx.obj["nvcfHdl"]
    try:
        ok, funcid, version = nvcf_hdl.id_version(funcid, version)
        if not ok:
            error_msg = "id and version are required"
            _raise_runtime_err(error_msg)

        resp = nvcf_hdl.nvcf_helper_get_deployment_detail(
            funcid=funcid,
            version=version,
        )
        pprint(resp, expand_all=True)

    except Exception as e:
        error_msg = f"Could not get deployment detail: {e!s}"
        nvcf_hdl.logger.error(error_msg)  # noqa: TRY400
        raise typer.Exit(code=1) from e


@nvcf.command(name="delete-function", help="Delete a (current) function", no_args_is_help=False)
def nvcf_delete_function(
    ctx: Context,
    funcid: Annotated[
        str | None,
        Option(
            help="The id of the function to delete",
            rich_help_panel="Delete-id",
            callback=validate_uuid,
        ),
    ] = None,
    version: Annotated[
        str | None,
        Option(
            help="The version of the function to delete",
            rich_help_panel="Delete-id",
            callback=validate_uuid,
        ),
    ] = None,
) -> None:
    """Delete an NVCF function.

    Args:
        ctx: The Typer context object.
        funcid: Function ID to delete.
        version: Function version ID to delete.

    Returns:
        None.

    """
    nvcf_hdl = ctx.obj["nvcfHdl"]
    try:
        ok, funcid, version = nvcf_hdl.id_version(funcid, version)
        if not ok:
            error_msg = "id and version are required"
            _raise_runtime_err(error_msg)

        nvcf_hdl.nvcf_helper_delete_function(
            funcid=funcid,
            version=version,
        )
        nvcf_hdl.console.print(f"Function with id '{funcid}' and version '{version}' deleted")

    except Exception as e:
        error_msg = f"Could not delete function: {e!s}"
        nvcf_hdl.logger.error(error_msg)  # noqa: TRY400
        raise typer.Exit(code=1) from e


@nvcf.command(name="s3cred-function", help="Update S3 Credentials", no_args_is_help=True)
def nvcf_s3cred_function(
    ctx: Context,
    s3credfile: Annotated[
        Path,
        Option(
            help="File containing the S3 credentials, see template 's3cred.json' for format",
            rich_help_panel="Cred",
            exists=True,
            file_okay=True,
            dir_okay=False,
            writable=False,
            readable=True,
            resolve_path=True,
        ),
    ],
    funcid: Annotated[
        str | None,
        Option(
            help="The id of the function",
            rich_help_panel="Cred",
            callback=validate_uuid,
        ),
    ] = None,
    version: Annotated[
        str | None,
        Option(
            help="The version of the function",
            rich_help_panel="Cred",
            callback=validate_uuid,
        ),
    ] = None,
) -> None:
    """Update S3 credentials for an NVCF function.

    Args:
        ctx: The Typer context object.
        s3credfile: Path to S3 credentials file.
        funcid: Function ID to update.
        version: Function version ID.

    Returns:
        None.

    """
    nvcf_hdl = ctx.obj["nvcfHdl"]
    try:
        ok, funcid, version = nvcf_hdl.id_version(funcid, version)
        if not ok:
            error_msg = "id and version are required"
            _raise_runtime_err(error_msg)

        nvcf_hdl.nvcf_helper_s3cred_function(
            funcid=funcid,
            version=version,
            s3credfile=s3credfile,
        )
        nvcf_hdl.console.print(f"S3 credentials updated for function with id '{funcid}' and version '{version}'")

    except Exception as e:
        error_msg = f"Could not update S3 credentials: {e!s}"
        nvcf_hdl.logger.error(error_msg)  # noqa: TRY400
        raise typer.Exit(code=1) from e


@nvcf.command(
    name="undeploy-function",
    help="Undeploy before re-deploying a function",
    no_args_is_help=False,
)
def nvcf_undeploy_function(
    ctx: Context,
    funcid: Annotated[
        str | None,
        Option(
            help="The id of the function to undeploy",
            rich_help_panel="Undeploy-id",
            callback=validate_uuid,
        ),
    ] = None,
    version: Annotated[
        str | None,
        Option(
            help="The version of the function to undeploy",
            rich_help_panel="Undeploy-id",
            callback=validate_uuid,
        ),
    ] = None,
    graceful: Annotated[bool, Option] = Option(
        default=True,
        help="When set to False, does not wait for running jobs to finish",
        rich_help_panel="Undeploy-id",
    ),
) -> None:
    """Undeploy an NVCF function.

    Args:
        ctx: The Typer context object.
        funcid: Function ID to undeploy.
        version: Function version ID.
        graceful: Whether to wait for running jobs.

    Returns:
        None.

    """
    nvcf_hdl = ctx.obj["nvcfHdl"]
    try:
        ok, funcid, version = nvcf_hdl.id_version(funcid, version)
        if not ok:
            error_msg = "id and version are required"
            _raise_runtime_err(error_msg)

        name, status = nvcf_hdl.nvcf_helper_undeploy_function(
            funcid=funcid,
            version=version,
            graceful=graceful,
        )
        nvcf_hdl.console.print(f"Function '{name}' undeployed with status '{status}'")

    except Exception as e:
        error_msg = f"Could not undeploy function: {e!s}"
        nvcf_hdl.logger.error(error_msg)  # noqa: TRY400
        raise typer.Exit(code=1) from e


@nvcf.command(name="import-function", help="Import a function", no_args_is_help=True)
def nvcf_import_function(
    ctx: Context,
    funcid: Annotated[
        str,
        Option(
            help="The id of the function to import",
            rich_help_panel="Import",
            callback=validate_uuid,
        ),
    ],
    version: Annotated[
        str,
        Option(
            help="The version of the function to import",
            rich_help_panel="Import",
            callback=validate_uuid,
        ),
    ],
    name: Annotated[
        str,
        Option(
            help="Name of the imported function",
            rich_help_panel="Import",
        ),
    ],
) -> None:
    """Import an NVCF function.

    Args:
        ctx: The Typer context object.
        funcid: Function ID to import.
        version: Function version ID.
        name: Name for the imported function.

    Returns:
        None.

    """
    nvcf_hdl = ctx.obj["nvcfHdl"]
    ids = {"id": funcid, "version": version, "name": name}
    try:
        nvcf_hdl.store_ids(ids)
    except Exception as e:
        error_msg = f"Could not import function: {e!s}"
        nvcf_hdl.logger.error(error_msg)  # noqa: TRY400
        raise typer.Exit(code=1) from e
    else:
        nvcf_hdl.console.print(f"Function with id '{funcid}' and version '{version}' imported as '{name}'")


if __name__ == "__main__":
    nvcf()
