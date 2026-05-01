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

"""NVCF Function class."""

import time
from collections.abc import Generator
from contextlib import contextmanager
from enum import Enum
from pathlib import Path

from loguru import logger

from cosmos_curator.client.nvcf_cli.ncf.launcher.nvcf_helper import NvcfHelper

DEFAULT_BASE_NGC_URL = "https://api.ngc.nvidia.com"
DEFAULT_BASE_NVCF_URL = "https://api.nvcf.nvidia.com"
DEFAULT_NVCF_TIMEOUT = 15


class NvcfFunctionStatus(Enum):
    """NVCF Function status."""

    ACTIVE = "ACTIVE"
    INACTIVE = "INACTIVE"
    NOT_FOUND = "NOT_FOUND"


class NvcfFunctionAlreadyDeployedError(Exception):
    """Function is already deployed."""


class NvcfFunction(NvcfHelper):
    """NVCF Function.

    Example usage - deploy and invoke a function using a context manager.
    When the context manager exits, the function is undeployed. This is useful
    for handling errors or interrupts and ensuring that the function is undeployed.

    ```python
    # Use an existing NVCF function
    nvcf_function = NvcfFunction(
        funcid=funcid,
        version=version,
        key=ngc_key,
        org=ngc_org,
        team=ngc_team_in_org,
    )

    with nvcf_function.deploy(
        backend, gpu, instance_type, deploy_config, num_nodes, max_concurrency
    ):
        nvcf_function.invoke(invoke_config, s3_config_file)
    ```
    """

    def __init__(  # noqa: PLR0913
        self,
        funcid: str,
        version: str,
        # backend: str,
        # gpu: str,
        # instance_type: str,
        key: str,
        org: str,
        team: str,
        ncg_url: str = DEFAULT_BASE_NGC_URL,
        nvcf_url: str = DEFAULT_BASE_NVCF_URL,
        timeout: int = DEFAULT_NVCF_TIMEOUT,
    ) -> None:
        """Initialize NVCF function.

        Args:
            funcid: Function ID.
            version: Function version.
            key: NVCF API Key.
            org: NVCF Organization.
            team: Team name within the organization.
            ncg_url: NGC URL.
            nvcf_url: NVCF URL.
            timeout: NVCF timeout.

        """
        super().__init__(
            url=ncg_url,
            nvcf_url=nvcf_url,
            key=key,
            org=org,
            team=team,
            timeout=timeout,
        )
        self.funcid = funcid
        self.version = version

    def _deploy(  # noqa: PLR0913
        self, backend: str, gpu: str, instance_type: str, deploy_config: Path, num_nodes: int, max_concurrency: int
    ) -> None:
        self.nvcf_helper_deploy_function(
            funcid=self.funcid,
            version=self.version,
            backend=backend,
            gpu=gpu,
            instance=instance_type,
            min_instances=1,
            max_instances=1,
            max_concurrency=max_concurrency,
            data_file=str(deploy_config),
            instance_count=num_nodes,
        )

    def _undeploy(self) -> None:
        """Undeploy NVCF Function."""
        logger.info(f"Undeploying nvcf function {self.funcid}, version {self.version}")
        name, status = self.nvcf_helper_undeploy_function(
            funcid=self.funcid,
            version=self.version,
            graceful=True,
        )
        logger.info(f"Function '{name}' undeployed with status '{status}'")

    @contextmanager
    def deploy(  # noqa: PLR0913
        self, backend: str, gpu: str, instance_type: str, deploy_config: Path, num_nodes: int, max_concurrency: int
    ) -> Generator[None, None, None]:
        """Deploy function as a context manager, undeploy on context manager exit.

        Args:
            backend: Backend name for deployment.
            gpu: GPU type to use.
            instance_type: Instance type to use.
            deploy_config: Deploy configuration file path.
            num_nodes: Number of nodes to deploy.
            max_concurrency: Maximum concurrency.
            backend: Backend name for deployment.
            gpu: GPU type to use.
            instance_type: Instance type to use.

        Yields:
            None

        Raises:
            AlreadyDeployedError: If the function is already deployed.

        """

        def _log(status: NvcfFunctionStatus) -> None:
            logger.info(f"Deployment status: {status.value}")

        if self.get_status() == NvcfFunctionStatus.ACTIVE:
            msg = f"Function {self.funcid=}, {self.version=} is already deployed"
            raise NvcfFunctionAlreadyDeployedError(msg)

        try:
            self._deploy(backend, gpu, instance_type, deploy_config, num_nodes, max_concurrency)
            while (status := self.get_status()) != NvcfFunctionStatus.ACTIVE:
                _log(status)
                time.sleep(10)
            _log(status)
            yield
        finally:
            self._undeploy()
            while (status := self.get_status()) == NvcfFunctionStatus.ACTIVE:
                _log(status)
                time.sleep(10)
            _log(status)

    def invoke(  # noqa: PLR0913
        self,
        data_file: Path,
        s3_config: str,
        out_dir: Path,
        *,
        wait: bool = True,
        retry_cnt: int = 2,
        retry_delay: int = 3,
    ) -> None:
        """Invoke NVCF Function.

        Args:
            data_file: Data file with deployment configuration.
            s3_config: base64 encoded S3 config.
            out_dir: Output directory.
            wait: Whether to wait for completion.
            retry_cnt: Number of retry attempts.
            retry_delay: Delay between retries in seconds.

        """
        logger.info(f"Invoking nvcf function {self.funcid} {self.version} with {data_file}")
        self.nvcf_helper_invoke_wait_retry_function(
            funcid=self.funcid,
            version=self.version,
            data_file=str(data_file),
            s3_config=s3_config,
            ddir=str(out_dir),
            wait=wait,
            retry_cnt=retry_cnt,
            retry_delay=retry_delay,
            legacy_cf=False,
        )

    def get_status(self) -> NvcfFunctionStatus:
        """Get status of NVCF Function."""
        try:
            resp = self.nvcf_helper_get_deployment_detail(
                funcid=self.funcid,
                version=self.version,
            )
            status = NvcfFunctionStatus.ACTIVE if resp["Status"] == "ACTIVE" else NvcfFunctionStatus.INACTIVE
        except RuntimeError as e:
            if "Deployment not found" in str(e):
                status = NvcfFunctionStatus.NOT_FOUND
            else:
                raise

        return status
