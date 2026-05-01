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

"""NVIDIA Cloud Functions (NVCF) Helper Module.

This module provides a comprehensive suite of utilities to interact with NVIDIA Cloud Functions.
It handles function management tasks including creation, deployment, invocation, status checking,
and undeployment of NVCF functions. The module supports both synchronous and asynchronous operations,
batch processing, and error handling with retry mechanisms.

Key capabilities include:
- Function lifecycle management (create, deploy, undeploy, delete)
- Function invocation with request tracking
- Status monitoring and log collection
- S3 credential management
- Batch job processing with concurrent execution
- Helm chart configuration for NVCF observability

"""

import concurrent.futures
import contextlib
import copy
import json
import tempfile
import time
from pathlib import Path
from typing import Annotated, Any
from urllib.parse import quote as q

from rich.table import Table

from cosmos_curator.client.nvcf_cli.ncf.common import NotFoundError, NvcfBase, NvcfClient

_HUNDRED_PCT: str = "100.0"
_MODULE_NAME: str = "Function"
_EXCEPTION_MESSAGE: str = "unexpected empty response"


def _raise_runtime_err(msg: str | dict[str, Any]) -> None:
    """Raise a RuntimeError with the given message.

    Args:
        msg: The error message to include in the exception.

    Raises:
        RuntimeError: Always raises this exception with the provided message.

    """
    if isinstance(msg, str):
        raise RuntimeError(msg)  # noqa: TRY004
    raise RuntimeError(json.dumps(msg))


def _raise_timeout_err(msg: str | dict[str, Any]) -> None:
    """Raise a TimeoutError with the given message.

    Args:
        msg: The error message to include in the exception.

    Raises:
        TimeoutError: Always raises this exception with the provided message.

    """
    if isinstance(msg, str):
        raise TimeoutError(msg)
    raise TimeoutError(json.dumps(msg))


def _extract_nvcf_error_details(resp: dict[str, Any], context: str, funcid: str | None = None) -> str:
    """Extract error details from NVCF API response with informative context.

    Args:
        resp: The NVCF API response dictionary
        context: Context string for the error (e.g., "function with name 'X'")
        funcid: Optional function ID to check if already present in detail message

    Returns:
        Formatted error message with key details

    """
    # Check prioritized list of detail keys for all error types
    detail_keys = [
        ("requestStatus", "statusDescription"),  # body.requestStatus.statusDescription
        ("detail",),  # body.detail
        ("issue", "detail"),  # body.issue.detail
        ("cause",),  # body.cause
    ]

    # Initialize with fallback value
    base_error = resp.get_error(context) if hasattr(resp, "get_error") else f"Error with {context}"
    result = base_error

    for key_path in detail_keys:
        value: Any = resp
        for key in key_path:
            value = value.get(key) if isinstance(value, dict) else None
            if value is None:
                break

        if value:
            # If the detail already contains the specific function ID, just return the detail
            # Otherwise, prepend with the base error for context
            result = str(value) if funcid and funcid in str(value) else f"{base_error}\nDetails: {value}"
            break

    return result


class CloudError(Exception):
    """Custom Error to distinguis a failed get-request-status vs cloud err."""


class NvcfHelper(NvcfBase):
    """A helper class for managing NVIDIA Cloud Functions (NVCF).

    This class provides a comprehensive set of utilities for interacting with NVCF services,
    enabling the full lifecycle management of cloud functions from creation to deletion.
    It handles API communication, error handling, and provides retry mechanisms for resilient
    function execution.

    Key capabilities:
    - Function creation, deployment, and undeployment
    - Function invocation and monitoring
    - Status tracking and log collection
    - Batch processing with concurrent execution
    - Credential management for S3 integration
    - Request termination and error handling

    The class extends NvcfBase and implements specific NVCF operations through
    the NCG and NVCF APIs.

    """

    def __init__(self, url: str, nvcf_url: str, key: str, org: str, team: str, timeout: int) -> None:  # noqa: PLR0913
        """Initialize the NvcfHelper with API endpoints and authentication.

        Args:
            url (str): Base URL for the NCG API.
            nvcf_url (str): Base URL for the NVCF API.
            key (str): API key for authentication.
            org (str): Organization identifier.
            team (str): Team name within the organization.
            timeout (int): Request timeout in seconds.

        """
        super().__init__(url=url, nvcf_url=nvcf_url, key=key, org=org, team=team, timeout=timeout)

        self.ncg_api_hdl = NvcfClient(self.logger.getChild("ncgApiHdl"), self.url, self.key)
        self.nvcf_api_hdl = NvcfClient(self.logger.getChild("nvcfApiHdl"), self.nvcf_url, self.key)

    def load_ids(self) -> dict[str, Any]:
        """Load function identifiers from the stored file.

        Returns:
            dict[str, Any]: Dictionary containing the function name, funcid, and version.
                            Returns empty values if the file doesn't exist or can't be read.

        """
        ids = {"name": None, "id": None, "version": None}
        with contextlib.suppress(Exception), Path(self.idf).open() as f:
            ids = json.load(f)
        return ids

    def store_ids(self, ids: dict[str, Any]) -> None:
        """Store function identifiers to a file.

        Args:
            ids (dict[str, Any]): Dictionary containing function identifiers to be stored.

        Raises:
            RuntimeError: If an error occurs while writing to the file.

        """
        with Path(self.idf).open("w") as f:
            json.dump(ids, f, indent=4)

    def cleanup_ids(self) -> None:
        """Remove the function identifiers file.

        This function attempts to delete the identifiers file and silently
        ignores any errors that occur during the deletion.

        """
        with contextlib.suppress(Exception):
            Path(self.idf).unlink()

    def id_version(self, funcid: str | None, version: str | None) -> tuple[bool, str | None, str | None]:
        """Retrieve and validate function ID and version.

        Args:
            funcid (str | None): Function ID to use. If None, load from stored file.
            version (str | None): Function version to use. If None, load from stored file.

        Returns:
            tuple[bool, str | None, str | None]: Tuple containing:
                - Boolean indicating if both funcid and version are valid
                - Function ID (may be None)
                - Function version (may be None)

        """
        obj = self.load_ids()
        if funcid is None:
            funcid = obj.get("id")
        if version is None:
            version = obj.get("version")

        if funcid is None or version is None:
            return False, funcid, version
        return True, funcid, version

    def nvcf_helper_list_clusters(self) -> Table:
        """List all available NVCF cluster groups.

        Returns:
            Table: A rich table containing information about all cluster groups,
                  including backend names, GPU types, instance types, and clusters.

        Raises:
            RuntimeError: If the API request fails or returns an error.

        """
        resp = self.ncg_api_hdl.get("/v2/nvcf/clusterGroups")

        if resp is None:
            raise RuntimeError(_EXCEPTION_MESSAGE)
        if resp.is_error:
            _raise_runtime_err(resp.get_error("clusters"))

        groups_data = resp.get("clusterGroups", [])
        groups = Table(title="Cluster Groups")
        groups.add_column("Backend Name")
        groups.add_column("GPU-Types Inst-Types")
        groups.add_column("Clusters")
        for ele in groups_data:
            name = ele.get("name")
            gl = ele.get("gpus")
            gpus = Table(show_header=False, box=None, show_edge=False)

            gpus.add_column()
            for gle in gl:
                gpu_name = gle.get("name")
                instance_types = gle.get("instanceTypes")
                instances = Table(show_header=False, show_edge=False)
                for ile in instance_types:
                    ins_name = ile.get("name")
                    instances.add_row(ins_name)
                gpus.add_row(gpu_name, instances, end_section=True)

            cl = ele.get("clusters")
            clusters = Table(show_header=False, show_edge=False)
            clusters.add_column()
            for cle in cl:
                cluster_name = cle.get("name")
                clusters.add_row(cluster_name)

            groups.add_row(name, gpus, clusters, end_section=True)

        return groups

    def nvcf_helper_list_functions(self) -> Table:
        """List all available NVCF functions.

        Returns:
            Table: A rich table containing information about all functions,
                  including names, statuses, IDs, versions, images, and endpoints.

        Raises:
            RuntimeError: If the API request fails or returns an error.

        """
        resp = self.ncg_api_hdl.get("/v2/nvcf/functions")

        if resp is None:
            raise RuntimeError(_EXCEPTION_MESSAGE)
        if resp.is_error:
            _raise_runtime_err(resp.get_error("functions"))

        funcs = resp.get("functions", [])
        functions = Table(title="Functions")
        functions.add_column("Name")
        functions.add_column("Status")
        functions.add_column("Id")
        functions.add_column("Version")
        functions.add_column("Image")
        functions.add_column("Endpoint")

        for ele in funcs:
            funcid = ele.get("id")
            version = ele.get("versionId")
            name = ele.get("name")
            status = ele.get("status")
            image = ele.get("containerImage")
            port = ele.get("inferencePort")
            endpoint = f"{port}:{ele.get('inferenceUrl')}"
            functions.add_row(name, status, funcid, version, image, endpoint)

        return functions

    def nvcf_helper_list_function_detail(self, name: str) -> list[dict[str, Any]]:
        """Get detailed information about a specific function by name.

        Args:
            name (str): The name of the function to get details for.

        Returns:
            list[dict[str, Any]]: A list of dictionaries containing details about
                                 the function, including ID, version, status, image,
                                 and endpoint information.

        Raises:
            NotFoundError: If the function with the specified name is not found.
            RuntimeError: If the API request fails or returns an error.

        """
        try:
            resp = self.ncg_api_hdl.get("/v2/nvcf/functions")
        except Exception as e:  # noqa: BLE001
            _raise_runtime_err(f"failed to list details for function with name '{name}': {e!s}")

        if resp is None:
            raise RuntimeError(_EXCEPTION_MESSAGE)
        if resp.is_not_found:
            raise NotFoundError(_MODULE_NAME, name=name)

        if resp.is_error:
            _raise_runtime_err(resp.get_error(f"function with name '{name}'"))

        data = []
        funcs = resp.get("functions", [])
        for ele in funcs:
            nm = ele.get("name")
            f = {}
            if nm == name:
                f["Id"] = ele.get("id")
                f["Version"] = ele.get("versionId")
                f["Status"] = ele.get("status")
                f["Image"] = ele.get("containerImage")
                port = ele.get("inferencePort")
                f["Endpoint"] = f"{port}:{ele.get('inferenceUrl')}"
                data.append(f)

        if len(data) == 0:
            raise NotFoundError(_MODULE_NAME, name=name)

        return data

    def nvcf_helper_create_function(  # noqa: C901, PLR0912, PLR0913, PLR0915
        self,
        name: str,
        image: str,
        inference_ep: str,
        inference_port: int,
        health_ep: str,
        health_port: int,
        args: str,
        data_file: str | None = None,
        helm_chart: str | None = None,
        helm_service_name: str | None = None,
    ) -> dict[str, Any]:
        """Create a new NVCF function with the specified parameters.

        Args:
            name (str): Name of the function to create.
            image (str): Container image to use for the function.
            inference_ep (str): Inference endpoint URL.
            inference_port (int): Inference port number.
            health_ep (str): Health check endpoint URL.
            health_port (int): Health check port number.
            args (str): Container arguments.
            data_file (str): Path to a JSON file containing additional function configuration.
            helm_chart (str | None): Path to Helm chart for deployment, if applicable.
            helm_service_name (str | None): Name of the Helm service, if applicable.

        Returns:
            dict[str, Any]: Dictionary containing the function ID and version.

        Raises:
            RuntimeError: If the API request fails, returns an error, or if the data file
                      contains invalid configuration.

        """
        create_data = {
            "name": name,
            "inferenceUrl": inference_ep,
            "inferencePort": inference_port,
            "health": {
                "protocol": "HTTP",
                "uri": health_ep,
                "port": health_port,
                "timeout": "PT10S",
                "expectedStatusCode": 200,
            },
            "functionType": "DEFAULT",
            "description": "Video Curation Service",
            "apiBodyFormat": "PREDICT_V2",
            "helmChart": helm_chart,
            "helmChartServiceName": helm_service_name,
            "containerImage": image,
            "containerArgs": args,
        }
        data = None
        if data_file is not None:
            try:
                with Path(data_file).open() as file:
                    data = json.load(file)
            except (FileNotFoundError, PermissionError, json.JSONDecodeError, OSError) as e:
                _raise_runtime_err(f"Could not read data: {e!s}")

        if data is not None:
            models = data.get("models")
            tags = data.get("tags")
            resources = data.get("resources")
            secrets = data.get("secrets")
            envs = data.get("envs")
            telemetries = data.get("telemetries")
            if secrets is not None:
                for s in secrets:
                    k = s.get("key")
                    v = s.get("value")
                    if v is None or v in {"FILL-IN", "FILL_IN"}:
                        _raise_runtime_err(
                            f"The data file has incorrect value for '{k}'. Please fix it before re-running"
                        )

            if models is not None:
                create_data["models"] = models
            if tags is not None:
                create_data["tags"] = tags
            if resources is not None:
                create_data["resources"] = resources
            if secrets is not None:
                create_data["secrets"] = secrets
            if envs is not None:
                create_data["containerEnvironment"] = envs
            if telemetries is not None:
                create_data["telemetries"] = telemetries
        if helm_chart is not None and image == "":
            # Have to unset unsupported parameters - we could create different template jsons, but coalesce seems
            # simpler while we convert to exclusive helm (or until NVCF supports parity)
            del create_data["containerImage"]
            del create_data["models"]
            # Also - have additional secrets to set to allow model retrieval
            existing_secrets = create_data["secrets"] or []
            create_data["secrets"] = [  # type: ignore[misc]
                *existing_secrets,
                {"name": "NGC_NVCF_ORG", "value": self.org},
                {"name": "NGC_NVCF_TEAM", "value": self.team},
                {"name": "NGC_NVCF_API_KEY", "value": self.key},
            ]
        else:
            create_data["helmChart"] = None
            create_data["helmChartServiceName"] = None

        resp = self.ncg_api_hdl.post("/v2/nvcf/functions", data=create_data)

        if resp is None:
            raise RuntimeError(_EXCEPTION_MESSAGE)
        if resp.is_error:
            error_msg = _extract_nvcf_error_details(resp, f"function with name '{name}'")
            _raise_runtime_err(error_msg)

        func = resp.get("function", {})
        if not func:
            _raise_runtime_err(f"unexpected response: {resp}")

        funcid = func.get("id")
        version = func.get("versionId")

        return {"id": funcid, "version": version}

    def nvcf_helper_deploy_function(  # noqa: C901, PLR0912, PLR0913
        self,
        funcid: str,
        version: str,
        backend: str,
        gpu: str,
        instance: str,
        min_instances: int,
        max_instances: int,
        max_concurrency: int,
        data_file: str | None = None,
        instance_count: int = 1,
    ) -> dict[str, Any]:
        """Deploy a function to NVCF with the specified parameters.

        Args:
            funcid (str): Function ID to deploy.
            version (str): Function version to deploy.
            backend (str): Cluster group to deploy to (will be sent as the single-element list
                           `clusters=[backend]` in the deployment spec).
            gpu (str): GPU type to use.
            instance (str): Instance type to use.
            min_instances (int): Minimum number of instances to run.
            max_instances (int): Maximum number of instances to run.
            max_concurrency (int): Maximum concurrent requests per instance.
            data_file (str): Path to a JSON file containing additional deployment configuration.
            instance_count (int, optional): Number of instances per replica. Defaults to 1.

        Returns:
            dict[str, Any]: Dictionary containing deployment details, including function
                           name, ID, version, status, and any errors encountered.

        Raises:
            RuntimeError: If the API request fails, returns an error, or if the data file
                      contains invalid configuration.

        """
        endpoint = f"/v2/nvcf/deployments/functions/{q(funcid, safe='')}/versions/{q(version, safe='')}"
        deploy_list: dict[str, Any] = {"deploymentSpecifications": []}
        deploy_data: dict[str, Any] = {
            "gpu": gpu,
            "clusters": [backend],
            "maxInstances": max_instances,
            "minInstances": min_instances,
            "instanceType": instance,
            "instanceCount": instance_count,
            "maxRequestConcurrency": max_concurrency,
            "preferredOrder": 99,
        }
        data = None
        if data_file is not None:
            try:
                with Path(data_file).open() as file:
                    data = json.load(file)
            except (FileNotFoundError, PermissionError, json.JSONDecodeError, OSError) as e:
                _raise_runtime_err(f"Failed to read data: {e!s}")

        if data is not None:
            avail = data.get("availabilityZones")
            if avail is not None:
                deploy_data["availabilityZones"] = avail

            config = data.get("configuration")
            if config is not None:
                deploy_data["configuration"] = config

            clusters = data.get("clusters")
            if clusters is not None:
                deploy_data["clusters"] = clusters

            regions = data.get("regions")
            if regions is not None:
                deploy_data["regions"] = regions

            attrs = data.get("attributes")
            if attrs is not None:
                deploy_data["attributes"] = attrs

            # If helm configuration has been passed, use it.
            if config is not None:
                # Override what's in chart/invoke with any CLI args
                deploy_data["configuration"]["replicas"] = instance_count
                labels_dict = {
                    "backend": backend,
                    "function_id": funcid,
                    "version_id": version,
                    "gpu": gpu,
                    "org": self.org,
                }
                deploy_data["configuration"]["metrics"]["extraExternalLabels"] = labels_dict

        deploy_list.get("deploymentSpecifications", []).append(deploy_data)
        resp = self.ncg_api_hdl.post(endpoint, data=deploy_list)

        if resp is None:
            raise RuntimeError(_EXCEPTION_MESSAGE)
        # Check for specific "already deployed" error first
        desc = resp.get_detail()
        if isinstance(desc, str) and (", use PUT" in desc or "already has a deployment" in desc):
            if ", use PUT" in desc:
                pos = desc.find(", use PUT")
                if pos > 0:
                    desc = desc[:pos]
            desc = f"{desc}, run '{self.exe} nvcf function undeploy-function' before (re)deploying"
            _raise_runtime_err(desc)

        # Handle any other deployment errors with full details
        if resp.is_error:
            error_msg = _extract_nvcf_error_details(
                resp, f"Function with Id '{funcid}' and version '{version}'", funcid
            )
            _raise_runtime_err(error_msg)

        dep = resp.get("deployment", {})
        if not dep:
            _raise_runtime_err(f"unexpected response: {resp}")

        return {
            "name": dep.get("functionName", ""),
            "id": funcid,
            "version": version,
            "status": (dep.get("functionStatus", "")),
            "errors": [
                {
                    "ReqId": dh.get("sisRequestId"),
                    "Error": dh.get("error"),
                }
                for dh in (dep.get("healthInfo", []))
            ],
        }

    def nvcf_helper_s3cred_function(
        self,
        funcid: str,
        version: str,
        s3credfile: str,
    ) -> None:
        """Upload S3 credentials for a function.

        Args:
            funcid (str): Function ID to apply credentials to.
            version (str): Function version to apply credentials to.
            s3credfile (str): Path to a JSON file containing S3 credentials.

        Raises:
            RuntimeError: If the API request fails, returns an error, or if the credentials
                      file contains invalid configuration or cannot be read.

        """
        endpoint = (
            f"/v2/orgs/{q(self.org or '', safe='')}"
            f"/nvcf/secrets/functions/{q(funcid, safe='')}"
            f"/versions/{q(version, safe='')}"
        )
        try:
            with Path(s3credfile).open() as file:
                cred = json.load(file)
        except (FileNotFoundError, PermissionError, json.JSONDecodeError, OSError) as e:
            _raise_runtime_err(f"Failed to read credentials: {e!s}")

        resp = self.ncg_api_hdl.put(endpoint, data=cred)

        if resp is None:
            raise RuntimeError(_EXCEPTION_MESSAGE)
        if resp.is_error:
            detail = resp.get_detail()
            if detail is not None:
                _raise_runtime_err(detail)
            _raise_runtime_err(resp.get_error(f"Function with Id '{funcid}', version '{version}'"))

    def nvcf_helper_invoke_batch(  # noqa: C901, PLR0912, PLR0913, PLR0915
        self,
        *,
        data_file: str,
        id_file: str,
        job_variant_file: str,
        ddir: str,
        s3_config: str | None = None,
        legacy_cf: bool = False,
        retry_cnt: int = 2,
        retry_delay: int = 300,
    ) -> None:
        """Execute a batch of function invocations using multiple threads.

        This method processes a batch of function invocation jobs by launching multiple
        threads, each calling nvcf_helper_invoke_wait_retry_function. The number of
        concurrent threads is determined by the minimum of available function ID/version
        pairs and I/O job variants.

        Args:
            data_file (str): Path to the base JSON file containing invocation data.
            id_file (str): Path to a JSON file containing function ID/version pairs.
            job_variant_file (str): Path to a JSON file containing job variants to process.
            ddir (str): Directory to download results to.
            s3_config (str | None, optional): S3 configuration to use. Defaults to None.
            legacy_cf (bool, optional): Whether to use the legacy client flow.
                                      Defaults to False.
            retry_cnt (int, optional): Number of times to retry on failure. Defaults to 2.
            retry_delay (int, optional): Delay in seconds between retries. Defaults to 300.

        Raises:
            ValueError: If required files cannot be read or the function ID list is empty.
            RuntimeError: If the batch execution fails.

        """
        data_file_list = []
        job_list = []
        id_len = 0
        io_len = 0
        fl = ""
        try:
            # build datafile
            fl = data_file
            with Path(data_file).open() as dfile:
                invoke_data = json.load(dfile)
            if "args" not in invoke_data:  # throw?
                invoke_data["args"] = {}

            fl = job_variant_file
            with Path(job_variant_file).open() as iofile:
                io_data = json.load(iofile)
            io_len = len(io_data)
            for idx, item in enumerate(io_data):
                invoke_temp = copy.deepcopy(invoke_data)
                for k, v in item.items():
                    invoke_temp["args"][k] = v
                fl = f"{data_file}.{idx}.json"
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    delete=False,
                    prefix=fl,
                    dir=tempfile.gettempdir(),
                ) as tf:
                    data_file_list.append(tf.name)
                    json.dump(invoke_temp, tf, indent=4)

            # just use the original data_file
            if io_len == 0:
                io_len = 1
                data_file_list.append(data_file)

            for _, item in enumerate(data_file_list):
                job: dict[str, str | None | bool | int] = {}
                job["data_file"] = item
                job["ddir"] = ddir
                job["s3_config"] = s3_config
                job["legacy_cf"] = legacy_cf
                job["wait"] = True
                job["retry_cnt"] = retry_cnt
                job["retry_delay"] = retry_delay
                job_list.append(job)

            # load function id/version files
            with Path(id_file).open() as idfile:
                id_list = json.load(idfile)
            id_len = len(id_list)
            if len(id_list) == 0:
                _raise_runtime_err("Function ID List is empty")

            # Distribute the id/version(id_len) to the job_list
            for idx, item in enumerate(job_list):
                target = id_list[idx % id_len]
                item["funcid"] = target["func"]
                item["version"] = target["vers"]

        except ValueError:
            raise
        except (FileNotFoundError, PermissionError, json.JSONDecodeError, OSError) as e:
            msg = f"Failed to read/write file {fl}: {e!s}"
            raise ValueError(msg) from e

        # Now start the dispatch
        try:
            # Limitations : the num_workers can still be very large for a small
            # system from where it is launched. Limit num_workers?
            num_workers = min(id_len, io_len)
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as futs:
                # Spawn them
                future_jobs = [futs.submit(self.nvcf_helper_invoke_wait_retry_function, **kw) for kw in job_list]  # type: ignore[arg-type]
                for done_job in concurrent.futures.as_completed(future_jobs):
                    done_job.result()  # Propagate exceptions
        finally:
            # Cleanup all the temp files
            if len(io_data) > 0:
                for _, f in enumerate(data_file_list):
                    with contextlib.suppress(Exception):
                        Path(f).unlink()

    def nvcf_helper_invoke_wait_retry_function(  # noqa: C901, PLR0912, PLR0913, PLR0915
        self,
        *,
        funcid: str,
        ddir: str,
        version: str | None = None,
        data_file: str | None = None,
        prompt_file: str | None = None,
        asset_id: str | None = None,
        s3_config: str | None = None,
        legacy_cf: bool = True,
        wait: bool = True,
        retry_cnt: int = 2,
        retry_delay: int = 300,
    ) -> None:
        """Invoke a function with retries, waiting for completion.

        This method invokes a function and waits for its completion. If the invocation
        fails, it will retry up to the specified number of times with a delay between
        retries.

        Args:
            funcid (str): Function ID to invoke.
            ddir (str): Directory to download results to.
            version (str | None, optional): Function version to invoke. Defaults to None.
            data_file (str | None, optional): Path to a JSON file containing invocation data.
                                           Defaults to None.
            prompt_file (str | None, optional): Path to a file containing a prompt text.
                                             Defaults to None.
            asset_id (str | None, optional): Asset ID to use for the invocation.
                                          Defaults to None.
            s3_config (str | None, optional): S3 configuration to use. Defaults to None.
            legacy_cf (bool, optional): Whether to use the legacy client flow.
                                      Defaults to True.
            wait (bool, optional): Whether to wait for function completion. Defaults to True.
            retry_cnt (int, optional): Number of times to retry on failure. Defaults to 2.
            retry_delay (int, optional): Delay in seconds between retries. Defaults to 300.

        Raises:
            ValueError: If required files cannot be read or for non-retryable errors.
            RuntimeError: If all retries are exhausted without success.

        """
        cnt = retry_cnt
        while cnt > 0:
            # if we get inside this, it means, this is a retry
            # need to sleep before retry
            if cnt != retry_cnt:
                msg = f"Retrying {cnt} of {retry_cnt} after a sleep of {retry_delay}s"
                self.logger.warning(msg)
                time.sleep(retry_delay)
            try:
                resp = self.nvcf_helper_invoke_function(
                    funcid=funcid,
                    version=version,
                    data_file=data_file,
                    prompt_file=prompt_file,
                    asset_id=asset_id,
                    s3_config=s3_config,
                    ddir=ddir,
                )
            except (ValueError, RuntimeError):
                # Use ValueError, RuntimeError for all cases that must not be retried
                raise
            except Exception as e:
                cnt -= 1
                if cnt == 0:
                    raise
                # All other errors are retryable
                msg = f"Could not invoke function: {e!s}"
                self.logger.exception(msg)
                continue

            reqid = None
            status = None
            reqid = resp.get("reqid")
            logs = resp.get("logs")
            status = resp.get("status")
            detail = resp.get("detail")
            # only when detail is used below
            if detail is None:
                detail = "See logs for details"

            if reqid is not None:
                if logs is not None:
                    fname = f"{tempfile.gettempdir()}/{reqid}.log"
                    try:
                        with Path(fname).open("a") as fd:
                            fd.writelines(f"{line}\n" for line in logs)
                        msg = f"Wrote the collected logs to {fname}"
                        self.logger.info(msg)
                    except (
                        TypeError,
                        OSError,
                        FileNotFoundError,
                        PermissionError,
                        AttributeError,
                        UnicodeEncodeError,
                    ) as e:
                        err_msg = f"Failed to write the logs to {fname} {e!s}: "
                        self.logger.exception(err_msg)
                if status == "fulfilled":
                    return
                if wait:
                    while cnt > 0:
                        try:
                            self.nvcf_helper_get_request_status_with_wait(
                                reqid=reqid, ddir=ddir, funcid=funcid, version=version, wait=wait, legacy_cf=legacy_cf
                            )
                        except TimeoutError as e:
                            cnt -= 1
                            msg = f"Timeout getting request status, retrys left = {cnt}: {e!s}"
                            self.logger.warning(msg)
                            if cnt == 0:
                                raise
                            continue
                        except RuntimeError as e:
                            cnt -= 1
                            msg = f"Error getting request status, retrys left = {cnt}: {e!s}"
                            self.logger.warning(msg)
                            if cnt == 0:
                                raise
                            continue
                        except CloudError as e:
                            cnt -= 1
                            if cnt == 0:
                                msg = f"{e!s}"
                                raise RuntimeError(msg) from e
                            # Need to retry the whole request, that will happen if we break out of this loop
                            # and forcibly set status
                            detail = f"Request has failed, retrying, retrys left = {cnt}: {e!s}"
                            status = "failed"
                            cnt += 1  # need to bump the count back to be reduced below
                            break
                        else:
                            return
                else:
                    msg = f"Check progress & logs: {self.exe} nvcf function get-request-status --reqid {reqid}"
                    self.logger.info(msg)
                    return

            # Check to see if we should retry the whole request
            if (status is not None and status == "failed") or reqid is None:
                cnt -= 1
                msg = f"Failed function invocation: {detail}, retrys left = {cnt}"
                self.logger.warning(msg)
                continue

        if cnt <= 0:
            _raise_runtime_err(f"Giving up after retrying for {retry_cnt} times")

    def nvcf_helper_invoke_function(  # noqa: C901, PLR0912, PLR0913, PLR0915
        self,
        *,
        funcid: str,
        ddir: str,
        version: str | None = None,
        data_file: str | None = None,
        prompt_file: str | None = None,
        asset_id: str | None = None,
        s3_config: str | None = None,
    ) -> dict[str, Any]:
        """Invoke a function with the specified parameters.

        Args:
            funcid (str): Function ID to invoke.
            ddir (str): Directory to download results to.
            version (str | None, optional): Function version to invoke. Defaults to None.
            data_file (str | None, optional): Path to a JSON file containing invocation data.
                                           Defaults to None.
            prompt_file (str | None, optional): Path to a file containing a prompt text.
                                             Defaults to None.
            asset_id (str | None, optional): Asset ID to use for the invocation.
                                          Defaults to None.
            s3_config (str | None, optional): S3 configuration to use. Defaults to None.

        Returns:
            dict[str, Any]: Dictionary containing invocation results, including request ID,
                           status, and logs.

        Raises:
            ValueError: If required files cannot be read.
            RuntimeError: If the API request fails or returns an error.

        """
        if version is not None:
            endpoint = f"/v2/nvcf/pexec/functions/{q(funcid, safe='')}/versions/{q(version, safe='')}"
        else:
            endpoint = f"/v2/nvcf/pexec/functions/{q(funcid, safe='')}"

        invoke_data = {}
        ret = {}

        if data_file is not None:
            try:
                with Path(data_file).open() as file:
                    invoke_data = json.load(file)

            except (FileNotFoundError, PermissionError, json.JSONDecodeError, OSError) as e:
                msg = f"Failed to read data: {e!s}"
                _raise_runtime_err(msg)

        if prompt_file is not None:
            try:
                with Path(prompt_file).open() as pfile:
                    prompt_text = pfile.read().strip()
                    if "args" in invoke_data and isinstance(invoke_data["args"], dict):
                        invoke_data["args"]["captioning_prompt_text"] = prompt_text
            except (FileNotFoundError, PermissionError, json.JSONDecodeError, OSError) as e:
                msg = f"Failed to read prompt file: {e!s}"
                _raise_runtime_err(msg)

        if s3_config is not None and "args" in invoke_data and isinstance(invoke_data["args"], dict):
            invoke_data["args"]["s3_config"] = s3_config

        eh = None
        if asset_id is not None:
            eh = {"NVCF-INPUT-ASSET-REFERENCES": asset_id}
        resp = self.nvcf_api_hdl.post(
            endpoint,
            data=invoke_data,
            extra_head=eh,
            timeout=self.timeout,
            addl_headers=True,
            enable_504=True,
        )

        if resp is None:
            raise RuntimeError(_EXCEPTION_MESSAGE)
        # these checks are positional and need to happen in order
        if resp.is_timeout:
            _raise_timeout_err(f"Function with Id '{funcid}', version '{version}', timed out")

        if resp.is_error:
            detail = resp.get_detail()
            if detail is not None:
                _raise_runtime_err(detail)
            _raise_runtime_err(resp.get_error(f"Function with Id '{funcid}', version '{version}'"))

        headers = resp.get("headers", {})
        if not headers:  # possibly unreachable?
            _raise_runtime_err(f"unexpected response: {resp}")

        reqid = headers.get("reqid")
        status = headers.get("status")
        pct = headers.get("pct")
        location = headers.get("location")
        if location:
            self.logger.info("Output is ready, attempting to download")
            path = Path(ddir) / f"{reqid}.zip"
            self.nvcf_api_hdl.download(url=location, dest=str(path))
            dl_msg = f"Downloaded result {path}"
            self.logger.info(dl_msg)
            msg = f"Function Id {funcid} / {version}\nReqId: {reqid}, Status: '{status}', Completed: {pct}"
            self.logger.info(msg)
            status = "fulfilled"
        else:
            info_msg = f"Function Id {funcid} / {version}\nReqId: {reqid}, Status: '{status}', Completed: {pct}"
            self.logger.info(info_msg)
        ret["status"] = status
        ret["reqid"] = reqid
        logs = resp.get("logs", "")
        if logs:
            lines = logs.split("\n")
            ret["logs"] = lines
        return ret

    def nvcf_helper_get_request_status(  # noqa: C901
        self,
        reqid: str,
        ddir: str,
        funcid: str | None = None,
        version: str | None = None,
        *,
        legacy_cf: bool | None = False,
    ) -> dict[str, Any] | None:
        """Get the status of a function request.

        This method checks the status of a function request and downloads any available
        output files if the request has completed.

        Args:
            reqid (str): Request ID to check.
            ddir (str): Directory to download output files to.
            funcid (str | None, optional): Function ID associated with the request. Defaults to None.
            version (str | None, optional): Function version associated with the request.
                                          Defaults to None.
            legacy_cf (bool | None, optional): Whether to use the legacy API or the new API.
                                             Set to None to only download assets.
                                             Defaults to False.

        Returns:
            dict[str, Any] | None: Dictionary containing request status information,
                                  or None if the status could not be determined.

        Raises:
            RuntimeError: If the API request fails or returns an error.

        """
        if legacy_cf is not None and not legacy_cf and funcid is not None:
            return self._nvcf_helper_get_request_status_new(reqid, funcid, version)

        endpoint = f"/v2/nvcf/pexec/status/{q(reqid, safe='')}"

        ret = {}
        resp = self.nvcf_api_hdl.get(endpoint, timeout=self.timeout, addl_headers=True, enable_504=True)

        if resp is None:
            raise RuntimeError(_EXCEPTION_MESSAGE)
        # these checks are positional and need to happen in order
        if resp.is_timeout:
            _raise_timeout_err(f"{reqid} May have timed out")

        if resp.is_error:
            detail = resp.get_detail()
            if detail is not None:
                _raise_runtime_err(detail)
            _raise_runtime_err(resp.get_error(f"request with funcid '{reqid}'"))

        if not resp.has_status:  # possibly unreachable?
            _raise_runtime_err(f"unexpected response: {resp}")

        headers = resp.get("headers", {})
        if len(headers) > 0:
            reqid = headers.get("reqid")
            status = headers.get("status")
            location = headers.pop("location", "")
            pct = headers.get("pct")
            if location:
                self.logger.info("Output is ready, attempting to download")
                path = Path(ddir) / f"{reqid}.zip"
                self.nvcf_api_hdl.download(url=location, dest=str(path))
                dl_msg = f"Downloaded result {path}"
                self.logger.info(dl_msg)

            # indicate that this was not called just for location
            # when get_request_status is called just for download from location
            # to support new container types, we set this to None
            elif legacy_cf is not None:
                status_msg = f"ReqId: {reqid}, Status: '{status}', Completed: {pct}"
                self.logger.info(status_msg)

            ret["reqid"] = reqid
            logs = resp.get("logs", "")
            if logs:
                lines = logs.split("\n")
                ret["logs"] = lines
            ret["status"] = status
            return ret

        issue = resp.get("issue", {})
        sts = issue.get("requestStatus") if issue else resp.get("requestStatus", {})
        if sts:
            code = sts.get("statusCode")
            desc = sts.get("statusDescription")
            self.console.print(f"{code}: {desc}")
            return None

        self.console.print(f"Unhandled Error: {resp}")
        error_msg = f"Unhandled Error: {resp}"
        self.logger.error(error_msg)
        return None

    def nvcf_helper_get_request_status_with_wait(  # noqa: C901, PLR0912, PLR0913, PLR0915
        self,
        reqid: str,
        ddir: str,
        funcid: str | None = None,
        version: str | None = None,
        *,
        legacy_cf: bool | None = False,
        wait: bool = True,
    ) -> None:
        """Get request status with periodic checking until completion.

        This method repeatedly checks the status of a function request until it
        completes or fails. It downloads logs and outputs when available.

        Args:
            reqid (str): Request ID to check.
            ddir (str): Directory to download output files to.
            funcid (str | None, optional): Function ID associated with the request. Defaults to None.
            version (str | None, optional): Function version associated with the request.
                                          Defaults to None.
            legacy_cf (bool | None, optional): Whether to use the legacy API or the new API.
                                             Defaults to False.
            wait (bool, optional): Whether to wait and repeatedly check the status
                                 until completion. Defaults to True.

        Raises:
            RunTimeError: If the API request fails or returns an error.
            TimeoutError: If there was a timeout
            CloudError: If the CF indicates failure

        """
        first_log = True
        while True:
            try:
                resp = self.nvcf_helper_get_request_status(
                    reqid, ddir=ddir, funcid=funcid, version=version, legacy_cf=legacy_cf
                )
            except (RuntimeError, TimeoutError):
                self.logger.exception("Could not get request status: ")
                raise

            # Normalized status can only be
            # fulfilled or done
            # failed
            # in-progress
            if resp is not None and legacy_cf:
                rqid = resp.get("reqid")
                logs = resp.get("logs")
                if rqid is not None and logs is not None:
                    fname = f"{tempfile.gettempdir()}/{rqid}.log"
                    try:
                        with Path(fname).open("a") as fd:
                            fd.writelines(f"{line}\n" for line in logs)
                        msg = f"Wrote the collected logs to {fname}"
                        self.logger.info(msg)
                    except RuntimeError:
                        err_msg = f"Failed to write the logs to {fname}: "
                        self.logger.exception(err_msg)

                status = resp.get("status")
                if status in {"fulfilled", "done"}:
                    break
                if status == "failed":
                    msg = f"RequestId: {reqid} has failed, check logs for details"
                    raise CloudError(msg)

            if resp is not None and not legacy_cf:
                rqid = resp.get("reqid")
                fzip = resp.get("zip")
                if rqid is not None and fzip is not None:
                    fname = f"{tempfile.gettempdir()}/{rqid}-log.zip"
                    try:
                        with Path(fname).open("wb") as fd:
                            fd.write(fzip.getvalue())
                        if first_log:
                            log_msg = f"Writing logs to {fname}"
                            self.logger.info(log_msg)
                            first_log = False
                    except RuntimeError:
                        err_msg = f"Failed to write the logs to {fname}: "
                        self.logger.exception(err_msg)

                status = resp.get("status")
                if status in {"fulfilled", "done"}:
                    # need one round of help from legacy side, but break away anyways
                    # in case the request was asset, but for non-asset cases, this is
                    # not needed. Failure to fetch the asset will not be marked error
                    # set legacy_cf to None to indicate we only care for download
                    if rqid is not None:
                        try:
                            resp = self.nvcf_helper_get_request_status(
                                rqid, ddir=ddir, funcid=funcid, version=version, legacy_cf=None
                            )
                        except (RuntimeError, TimeoutError):
                            self.logger.info("Not downloading any assets")
                    break
                if status == "failed":
                    msg = f"RequestId: {reqid} has failed, check logs for details"
                    raise CloudError(msg)

            if not wait or resp is None:
                break
            if wait:
                self.console.print()
            # dont sleep to adapt to long-polling for legacy_cf
            if not legacy_cf:
                time.sleep(120)

    def nvcf_helper_terminate_request(
        self,
        reqid: str,
        funcid: str,
        version: str | None = None,
    ) -> dict[str, Any] | None | str:
        """Terminate an in-progress function request.

        Args:
            reqid (str): Request ID to terminate.
            funcid (str): Function ID associated with the request.
            version (str | None, optional): Function version associated with the request.
                                          Defaults to None.

        Returns:
            dict[str, Any] | None | str: Termination status information or error message.

        Raises:
            RuntimeError: If the API request fails or returns an error.

        """
        eh = {"CURATOR-NVCF-REQID": reqid, "CURATOR-REQ-TERMINATE": "true"}
        if version is not None:
            endpoint = f"/v2/nvcf/pexec/functions/{q(funcid, safe='')}/versions/{q(version, safe='')}"
        else:
            endpoint = f"/v2/nvcf/pexec/functions/{q(funcid, safe='')}"

        resp = self.nvcf_api_hdl.post(endpoint, extra_head=eh, timeout=self.timeout, addl_headers=True, enable_504=True)

        if resp is None:
            raise RuntimeError(_EXCEPTION_MESSAGE)

        # these checks are positional and need to happen in order
        if resp.is_timeout:
            _raise_timeout_err(f"{reqid} May have timed out")

        if resp.is_error:
            detail = resp.get_detail()
            if detail is not None:
                _raise_runtime_err(detail)
            _raise_runtime_err(resp.get_error(f"Request with Request-Id '{reqid}'"))

        status = resp.get_term_status()
        if status is None or len(status) == 0:
            status = {"reqid": f"{reqid} could not get termination status, please check logs"}
        return status

    def nvcf_helper_delete_function(self, funcid: str, version: str) -> None:
        """Delete a function version.

        Args:
            funcid (str): Function ID to delete.
            version (str): Function version to delete.

        Raises:
            RuntimeError: If the API request fails or returns an error.

        """
        endpoint = f"/v2/nvcf/functions/{q(funcid, safe='')}/versions/{q(version, safe='')}"
        resp = self.ncg_api_hdl.delete(endpoint)

        if resp is None:
            raise RuntimeError(_EXCEPTION_MESSAGE)
        if resp.is_error:
            detail = resp.get_detail()
            if detail is not None:
                _raise_runtime_err(detail)
            _raise_runtime_err(resp.get_error(f"Function with Id '{funcid}', version '{version}'"))

        if not resp.has_status:  # possibly unreachable?
            _raise_runtime_err(f"unexpected response: {resp}")

    def nvcf_helper_get_deployment_detail(self, funcid: str, version: str) -> dict[str, Any]:
        """Get deployment details for a specific function version.

        Args:
            funcid (str): Function ID to get deployment details for.
            version (str): Function version to get deployment details for.

        Returns:
            dict[str, Any]: Dictionary containing deployment details, including function
                           name, ID, version, status, and any deployment errors.

        Raises:
            RuntimeError: If the API request fails or returns an error.

        """
        endpoint = f"/v2/nvcf/deployments/functions/{q(funcid, safe='')}/versions/{q(version, safe='')}"
        resp = self.ncg_api_hdl.get(endpoint)

        if resp is None:
            raise RuntimeError(_EXCEPTION_MESSAGE)

        if resp.is_error:
            detail = resp.get_detail()
            if detail is not None:
                _raise_runtime_err(detail)
            _raise_runtime_err(resp.get_error(f"Function with Id '{funcid}', version '{version}'"))

        dep = resp.get("deployment", {})
        if not dep:
            _raise_runtime_err(f"unexpected response: {resp}")

        return {
            "Name": dep.get("functionName"),
            "Id": dep.get("functionId"),
            "Version": dep.get("functionVersionId"),
            "Status": dep.get("functionStatus"),
            "Detail": [
                {
                    "ReqId": dh.get("sisRequestId"),
                    "Error": dh.get("error"),
                }
                for dh in dep.get("healthInfo", [])
            ],
        }

    def nvcf_helper_undeploy_function(
        self,
        funcid: str,
        version: str,
        *,
        graceful: bool,
    ) -> tuple[
        Annotated[str, "Name of the function"],
        Annotated[str, "Status of the operation"],
    ]:
        """Undeploy a function version.

        Args:
            funcid (str): Function ID to undeploy.
            version (str): Function version to undeploy.
            graceful (bool): Whether to perform a graceful undeployment.

        Returns:
            tuple[str, str]: Tuple containing:
                - The name of the function
                - The status of the undeployment operation

        Raises:
            RuntimeError: If the API request fails or returns an error.

        """
        endpoint = f"/v2/nvcf/deployments/functions/{q(funcid, safe='')}/versions/{q(version, safe='')}"
        if graceful:
            endpoint = f"{endpoint}?graceful=true"
        resp = self.ncg_api_hdl.delete(endpoint)

        if resp is None:
            raise RuntimeError(_EXCEPTION_MESSAGE)
        if resp.is_error:
            detail = resp.get_detail()
            if detail is not None:
                _raise_runtime_err(detail)
            _raise_runtime_err(resp.get_error(f"Function with Id '{funcid}', version '{version}'"))

        func = resp.get("function", {})
        if not func:
            _raise_runtime_err(f"unexpected response: {resp}")

        return func.get("name"), func.get("status")

    def _nvcf_helper_get_request_status_new(  # noqa: C901, PLR0912, PLR0915
        self,
        reqid: str,
        funcid: str,
        version: str | None = None,
    ) -> dict[str, Any] | None:
        """Get request status using the new cosmos-curator-based API.

        This is a private helper method to be used by nvcf_helper_get_request_status.

        Args:
            reqid (str): Request ID to check.
            funcid (str): Function ID associated with the request.
            version (str | None, optional): Function version associated with the request.
                                          Defaults to None.

        Returns:
            dict[str, Any] | None: Dictionary containing request status information,
                                  or None if the status could not be determined.

        Raises:
            RuntimeError: If the API request fails or returns an error.

        """
        eh = {"CURATOR-NVCF-REQID": reqid, "CURATOR-STATUS-CHECK": "true"}
        if version is not None:
            endpoint = f"/v2/nvcf/pexec/functions/{q(funcid, safe='')}/versions/{q(version, safe='')}"
        else:
            endpoint = f"/v2/nvcf/pexec/functions/{q(funcid, safe='')}"

        resp = self.nvcf_api_hdl.post(endpoint, extra_head=eh, timeout=self.timeout, addl_headers=True, enable_504=True)

        if resp is None:
            raise RuntimeError(_EXCEPTION_MESSAGE)
        # these checks are positional and need to happen in order
        if resp.is_timeout:
            _raise_timeout_err(f"{reqid} May have timed out")

        if resp.is_error:
            detail = resp.get_detail()
            if detail is not None:
                _raise_runtime_err(detail)
            _raise_runtime_err(resp.get_error(f"request with funcid '{reqid}'"))

        ret = {}

        headers = resp.get("headers", {})
        if len(headers) > 0:
            # status can be "fulfilled", "done", "in-progress", "running", "pending"
            # Note: we can get a status without any log or zip

            # done and fulfilled are similar, except with done, we may still need to
            # download assets, and running and in-progress are similar
            # nvcf flow is from pending(rare)->in-progress->running(rare)->fulfilled
            # cosmos-curator flow is from NA -> running -> ? -> done

            status = headers.get("status")
            # invoke-status from curator-pipeline-status
            # the curator-pipeline-status can be
            # pending-evaluation - The worker has not yet accepted the request.
            # not-found - Request was not found.
            # running - Request is still running.
            # in-progress - A worker is processing the request.
            # fulfilled - The process has been completed with results.
            # done - The process has been completed with results.
            # rejected - The request was rejected by the service.
            # errored - An error occurred during worker processing.
            # failed - An error occurred during worker processing.
            zsts = resp.get("invoke-based-status")
            pct = headers.get("pct", "0.0")  # wonder if we can keep the last pct

            # here we are trying to normalize curator status with nvcf status
            # curator status can be
            # fulfilled
            # failed
            # in-progress
            # not-found
            if zsts is not None:
                if zsts in ["not-found", "pending-evaluation"]:
                    _raise_runtime_err(f"request with funcid '{reqid}' not found, or not started yet")
                if zsts in ["running", "in-progress"]:
                    status = "in-progress"
                elif zsts in ["fulfilled", "done"]:
                    status = "fulfilled"
                else:
                    status = "failed"  # rejected, errored, failed

            # we can get the zip in either case but need the pct/zsts too
            if status in ["in-progress", "fulfilled", "failed"] and pct is not None:
                resp_zip = resp.get("zip", None)
                if resp_zip is not None:
                    ret["zip"] = resp_zip

            # This should no longer happen after a fix on curator side
            # but leaving it here as a safety net
            if pct is not None and float(pct) >= float(_HUNDRED_PCT):
                pct = "99.99"
            # Due to the delay in ray metrics update, it's possible curator does not
            # set the progress to 100% before it finishes & exists, so force it here
            if status == "fulfilled":
                pct = _HUNDRED_PCT

            status_msg = f"ReqId: {reqid}, Status: '{status}', Completed: {pct}%"
            self.logger.info(status_msg)

            # this is the original invoke request-id, not the current invoke
            ret["reqid"] = reqid
            ret["status"] = status
            return ret

        issue = resp.get("issue", {})
        # we have two ways to get this?
        sts = issue.get("requestStatus")
        if sts is None:  # try again?
            sts = resp.get("requestStatus", {})
        if sts is not None:
            code = sts.get("statusCode")
            desc = sts.get("statusDescription")
            # is this error ?
            if None not in [code, desc]:
                self.console.print(f"{code}: {desc}")
            return None

        self.console.print(f"Unhandled Error: {resp}")
        error_msg = f"Unhandled Error: {resp}"
        self.logger.error(error_msg)
        return None
