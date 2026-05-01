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

"""NVIDIA Cloud Function client implementation.

This module provides the client implementation for interacting with NVIDIA Cloud Functions,
including response handling, HTTP request methods, and file operations.
"""

import copy
import json
import os
import time
from io import BytesIO
from logging import Logger
from pathlib import Path
from typing import Any, BinaryIO

import requests
from requests import Response
from tqdm import tqdm


class NVCFResponse(dict[str, Any]):
    """A dictionary-based response object for NVCF API calls.

    This class extends dict to provide additional properties and methods for handling
    NVCF API responses, including status checking and error handling.
    """

    def __init__(self, response: dict[str, Any] | None = None) -> None:
        """Initialize the NVCFResponse object.

        Args:
            response: Optional dictionary containing the response data

        """
        super().__init__(response if response else {})

    def __str__(self) -> str:
        """Get a string representation of the response.

        Returns:
            A formatted string representation of the response data

        """

        def __fmt(prefix: str, data: Any) -> str:  # noqa: ANN401
            if isinstance(data, dict):
                return ", ".join(__fmt(f"{prefix}{key}::", value) for key, value in data.items())

            return f"{prefix.rstrip(':')}={data!r}"

        return __fmt("", self)

    @property
    def status(self) -> int:
        """Get the status code of the response.

        Returns:
            Integer status code

        """
        return int(self.get("status", 500))

    @property
    def has_status(self) -> bool:
        """Check if response contains a status field.

        Returns:
            bool: True if the response contains a status field, False otherwise.

        """
        return self.get("status") is not None

    @property
    def is_not_found(self) -> bool:
        """Check if the response indicates a not found error.

        Returns:
            True if the response indicates a not found error, False otherwise

        """
        http_404: int = 404
        return self.status == http_404

    @property
    def is_not_authorized(self) -> bool:
        """Check if the response indicates unauthorized.

        Returns:
            True if the response indicates Unauthorized , False otherwise

        """
        http_401: int = 401
        return self.status == http_401

    @property
    def is_error(self) -> bool:
        """Check if the response indicates an error.

        Returns:
            True if the response indicates an error, False otherwise

        """
        http_399: int = 399
        return self.status > http_399

    @property
    def is_timeout(self) -> bool:
        """Check if the response indicates a timeout.

        Returns:
            True if the response indicates a timeout, False otherwise

        """
        return bool(self.get("timeout", False))

    def get_term_status(self) -> dict[str, Any] | str:
        """Get the terminal status of the response.

        Returns:
            Dictionary or string containing the term status

        """
        status = self.get("term-status", {})
        if len(status) == 0:
            status = self.get("body-status", {})
        return status if len(status) > 0 else ""

    def get_detail(self) -> str | dict[str, Any] | None:
        """Get the detail of the response.

        Returns:
            String or dictionary containing the detail, or None if not available

        """
        issue = self.get("issue")
        if issue is not None:
            return str(issue.get("detail"))
        return self.get("detail")

    def get_error(self, obj: str) -> str:
        """Get the error message from the response.

        Args:
            obj: The object to get the error from

        Returns:
            String containing the error message

        """
        if len(self) == 0:
            return "unexpected empty response"

        messages = {
            400: f"invalid request for {obj}",
            401: f"operation not authorized for {obj}",
            403: f"operation not allowed for {obj}",
            404: f"{obj} not found",
            412: f"{obj} precondition failed",
            429: f"{obj} too many requests",
        }

        out = messages.get(self.status, f"unknown error occurred: {self}")
        if extra := self.get("requestStatus", {}).get("statusDescription"):
            out = f"{out}: {extra}"

        return out


class NvcfClient:
    """Client for interacting with NVIDIA Cloud Functions API.

    This class provides methods for making HTTP requests to the NVCF API,
    handling responses, and managing file uploads/downloads.
    """

    def __init__(self, logger: Logger, url: str, auth: str | None = None) -> None:
        """Initialize the NVCF client.

        Args:
            logger: Logger instance for logging operations
            url: Base URL for the NVCF API
            auth: Optional authentication token

        """
        self.url = url
        self.auth = None
        self.headers: dict[str, Any] = {
            "Authorization": f"Bearer {auth}" if auth else None,
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        self.ses_basic = requests.Session()
        self.ses = requests.Session()
        self.ses.headers.update(self.headers)
        self.logger = logger
        self.trace = os.getenv("NVCF_TRACE_FILE", "trace.log")

    def _log(self, resp: Response) -> None:
        """Log request and response details to the trace file.

        Args:
            resp: The response object to log

        """
        headers = copy.deepcopy(resp.headers)

        for hdr in ("Location", "location", "Set-Cookie", "Set-cookie", "set-cookie"):
            if headers.get(hdr) is not None:
                headers[hdr] = "[REDACTED]"

        def _indent(text: bytes | str | None) -> str:
            if text is None:
                return ""
            try:
                pre_dacted = json.loads(text)
                if pre_dacted.get("args") is not None and pre_dacted["args"].get("s3_config") is not None:
                    pre_dacted["args"]["s3_config"] = "[REDACTED]"
                if pre_dacted.get("secrets") is not None:
                    pre_dacted["secrets"] = "[REDACTED]"

                text = json.dumps(pre_dacted, indent=2, sort_keys=True)
            except (json.JSONDecodeError, TypeError):
                text = f"|\n{text!s}"

            return "\n".join("    " + str(line) for line in text.splitlines())

        try:
            with Path(self.trace).open("a") as tfd:
                tfd.write(f"--- # {time.strftime('%Y-%m-%d %H:%M:%S %Z')}\n")
                req = resp.request
                tfd.write("request:\n")
                tfd.write(f"  method: '{req.method}'\n")
                tfd.write(f"  url: '{req.url}'\n")
                tfd.write("  headers:\n")
                [
                    tfd.write(f'    {k}: "{v if k != "Authorization" else "[REDACTED]"}"\n')
                    for k, v in req.headers.items()
                ]
                tfd.write(f"  body: |\n{_indent(req.body)}\n")
                tfd.write("response:\n")
                tfd.write(f"  status: {resp.status_code}\n")
                tfd.write(f"  reason: '{resp.reason or 'None'}'\n")
                tfd.write("  headers:\n")
                [tfd.write(f'    {k}: "{v}"\n') for k, v in headers.items()]
                tfd.write(f"  body:{_indent(resp.text)}\n")
        except OSError:
            err_msg = f"{self.trace}: Failed to write trace: "
            self.logger.exception(err_msg)

    def get(
        self,
        endpoint: str,
        params: dict[str, Any] | None = None,
        timeout: int = -1,
        *,
        addl_headers: bool = False,
        enable_504: bool = False,
    ) -> NVCFResponse | None:
        """Make a GET request to the NVCF API.

        Args:
            endpoint: API endpoint to request
            params: Optional query parameters
            timeout: Request timeout in seconds (-1 for no timeout)
            addl_headers: Whether to include additional headers
            enable_504: Whether to enable special handling for 504 responses

        Returns:
            NVCFResponse object or None if the request fails

        """
        hdrs = None
        if timeout > -1:
            hdrs = copy.deepcopy(self.headers)
            hdrs["NVCF-POLL-SECONDS"] = str(timeout)
        # Enabling this will require special handling for 504
        if enable_504:
            if hdrs is None:
                hdrs = copy.deepcopy(self.headers)
            hdrs["NVCF-FEATURE-ENABLE-GATEWAY-TIMEOUT"] = "true"
        url = f"{self.url}{endpoint}"
        if hdrs is None:
            resp = self.ses.get(url, params=params, allow_redirects=False)
        else:
            resp = self.ses.get(url, params=params, headers=hdrs, allow_redirects=False)

        return self._handle_resp(resp, addl_headers=addl_headers)

    def post(  # noqa: PLR0913
        self,
        endpoint: str,
        data: dict[str, Any] | None = None,
        timeout: int = -1,
        extra_head: dict[str, Any] | None = None,
        *,
        addl_headers: bool = False,
        enable_504: bool = False,
    ) -> NVCFResponse | None:
        """Make a POST request to the NVCF API.

        Args:
            endpoint: API endpoint to request
            data: Optional data to send in the request body
            timeout: Request timeout in seconds (-1 for no timeout)
            extra_head: Optional additional headers to include
            addl_headers: Whether to include additional headers
            enable_504: Whether to enable special handling for 504 responses

        Returns:
            NVCFResponse object or None if the request fails

        """
        url = f"{self.url}{endpoint}"
        hdrs = None
        if timeout > -1:
            hdrs = copy.deepcopy(self.headers)
            hdrs["NVCF-POLL-SECONDS"] = str(timeout)
        # Enabling this will require special handling for 504
        if enable_504:
            if hdrs is None:
                hdrs = copy.deepcopy(self.headers)
            hdrs["NVCF-FEATURE-ENABLE-GATEWAY-TIMEOUT"] = "true"
        if extra_head is not None:
            if hdrs is None:
                hdrs = copy.deepcopy(self.headers)
            for h, val in extra_head.items():
                hdrs[h] = val
        if hdrs is None:
            resp = self.ses.post(url, json=data, allow_redirects=False)
        else:
            resp = self.ses.post(url, json=data, headers=hdrs, allow_redirects=False)

        return self._handle_resp(resp, addl_headers=addl_headers)

    def put(
        self,
        endpoint: str,
        data: dict[str, Any] | None = None,
    ) -> NVCFResponse | None:
        """Make a PUT request to the NVCF API.

        Args:
            endpoint: API endpoint to request
            data: Optional data to send in the request body

        Returns:
            NVCFResponse object or None if the request fails

        """
        url = f"{self.url}{endpoint}"
        resp = self.ses.put(url, json=data)
        return self._handle_resp(resp)

    def put_at(self, url: str, data: BinaryIO, hdrs: dict[str, Any]) -> NVCFResponse | None:
        """Make a PUT request to a specific URL with binary data.

        Args:
            url: Full URL to send the request to
            data: Binary data to send in the request body
            hdrs: Headers to include in the request

        Returns:
            NVCFResponse object or None if the request fails

        """
        start = time.time()
        total_size = os.fstat(data.fileno()).st_size
        data.seek(0)
        hdrs = hdrs.copy()
        hdrs["Content-Length"] = str(total_size)
        resp = self.ses_basic.request("PUT", url, data=data, headers=hdrs)

        result = self._handle_resp(resp, ignore_if_not_json=True)
        self.logger.info("Upload took %.2f seconds", time.time() - start)
        return result

    def download(self, url: str, dest: str) -> None:
        """Download a file from a URL to a local destination.

        Args:
            url: URL to download from
            dest: Local path to save the file to

        """
        fname = Path(dest).name
        with self.ses_basic.request("GET", url, stream=True, allow_redirects=False) as resp:
            resp.raise_for_status()
            tsz = int(resp.headers.get("content-length", 0))
            csz = 8192
            with (
                Path(dest).open("wb") as fd,
                tqdm(
                    total=tsz,
                    unit="iB",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=fname,
                ) as sts,
            ):
                for chunk in resp.iter_content(chunk_size=csz):
                    fd.write(chunk)
                    sts.update(len(chunk))

    def delete(self, endpoint: str) -> NVCFResponse | None:
        """Make a DELETE request to the NVCF API.

        Args:
            endpoint: API endpoint to request

        Returns:
            NVCFResponse object or None if the request fails

        """
        url = f"{self.url}{endpoint}"
        resp = self.ses.delete(url)
        return self._handle_resp(resp)

    def _handle_resp(
        self,
        resp: Response,
        *,
        ignore_if_not_json: bool = False,
        addl_headers: bool = False,
    ) -> NVCFResponse | None:
        """Handle an HTTP response from the NVCF API.

        Args:
            resp: The response object to handle
            ignore_if_not_json: Whether to ignore non-JSON responses
            addl_headers: Whether to include additional headers

        Returns:
            NVCFResponse object or None if the response cannot be handled

        """
        self._log(resp)
        http_504: int = 504
        http_500: int = 500

        data: None | str | list[str] | dict[str, Any] = None
        retval: dict[str, Any] = {}

        # Try to fill out basic response headers
        if addl_headers:
            self._add_response_headers(resp, retval)

        if len(resp.content) > 0:
            # Handle binary data (image, zip, pdf, etc.)
            if self._is_binary_content(resp):
                self._handle_binary_response(resp, retval)
            else:
                # Parse JSON response, data may contain error or be none
                data = self._parse_json_response(resp, retval, ignore_if_not_json=ignore_if_not_json)
        else:
            self._create_empty_response(resp, retval)

        # Handle specific status codes
        if resp.status_code == http_504:
            retval["timeout"] = True

        # For successful responses (200-399) and timeouts
        if resp.status_code in range(200, 400) or resp.status_code == http_504:
            return NVCFResponse(retval)

        if resp.status_code >= http_500:
            self._handle_server_error(resp)

        # For other error responses
        return self._handle_client_error(resp, data)

    def _create_empty_response(self, resp: Response, retval: dict[str, Any]) -> None:
        """Add status headers to the return value.

        Args:
            resp: The response object containing headers
            retval: Dictionary to add headers to

        """
        retval["status"] = resp.status_code
        if "headers" not in retval:
            retval["headers"] = {}

    def _is_binary_content(self, resp: Response) -> bool:
        """Check if the response contains binary content.

        Args:
            resp: The response object to check

        Returns:
            True if the content is binary, False otherwise

        """
        content_type = resp.headers.get("content-type", "").lower()
        return any(k in content_type for k in ["image", "zip", "pdf", "octet-stream"])

    def _handle_binary_response(self, resp: Response, retval: dict[str, Any]) -> None:
        """Handle a binary response.

        Args:
            resp: The response object containing binary data
            retval: Dictionary of response

        Raises:
            ValueError: If the binary response cannot be processed

        """
        if "status" not in retval:
            retval["status"] = resp.status_code
        if "headers" not in retval:
            retval["headers"] = {}

        try:
            retval["zip"] = BytesIO(resp.content)
        except (ValueError, OSError, TypeError) as e:
            error_msg = f"Failed to process binary response: {e!s}"
            raise ValueError(error_msg) from e

    def _parse_json_response(
        self, resp: Response, retval: dict[str, Any], *, ignore_if_not_json: bool
    ) -> str | list[str] | dict[str, Any]:
        """Parse a JSON response.

        Args:
            resp: The response object to parse
            retval: Dictionary of response
            ignore_if_not_json: Whether to ignore non-JSON responses

        Returns:
            Either the raw JSON data or list of strings, which is not expected

        Raises:
            ValueError: If the response is not valid JSON and ignore_if_not_json is False

        """
        http_error: int = 400
        http_401: int = 401
        http_404: int = 404

        if "status" not in retval:
            retval["status"] = resp.status_code
        if "headers" not in retval:
            retval["headers"] = {}

        data: str | list[str] | dict[str, Any] = resp.text

        try:
            data = resp.json()
            if isinstance(data, dict):
                # avoid collision with http-status
                if "status" in data:
                    data["body-status"] = data.pop("status")
                retval.update(data)
        except ValueError as v:
            error_msg = str(v)
            if not ignore_if_not_json and resp.status_code < http_error:
                error_msg = f"Failed to parse response: {v!s} [{resp.text}]"
            else:
                if resp.status_code == http_401:
                    error_msg = f"Unauthorized: {resp.text}"
                if resp.status_code == http_404:
                    error_msg = f"Not Found: {resp.text}"
            raise ValueError(error_msg) from v
        return data

    def _handle_client_error(self, resp: Response, data: dict[str, Any] | list[str] | str | None) -> NVCFResponse:
        """Handle client error responses (status 400-499).

        Args:
            resp: The response object to handle
            data: The parsed JSON data or None

        Returns:
            NVCFResponse object with error details

        """
        if not data:
            data = {"status": resp.status_code, "detail": resp.reason, "title": resp.reason}
        return NVCFResponse({"status": resp.status_code, "issue": data})

    def _handle_server_error(self, resp: Response) -> None:
        """Handle server error responses (status >= 500).

        Args:
            resp: The response object to handle

        Raises:
            RuntimeError: With error details from the response

        """
        error_lines = []

        # Include request ID if available
        if reqid := resp.headers.get("nvcf-reqid"):
            error_lines.append(f'{{"reqid": "{reqid}"}}')

        # Try to get error details from response
        try:
            if detail := resp.json().get("detail"):
                error_lines.extend(detail.split("\\n"))
        except (ValueError, json.JSONDecodeError):
            if resp.text:
                error_lines.extend(resp.text.split("\\n"))

        raise RuntimeError(error_lines)

    def _add_response_headers(self, resp: Response, retval: dict[str, Any]) -> None:
        """Add response headers to the return value.

        Args:
            resp: The response object containing headers
            retval: Dictionary to add headers to

        """
        if "status" not in retval:
            retval["status"] = resp.status_code
        if "headers" not in retval:
            retval["headers"] = {}
        headers = retval["headers"]

        # Add basic headers
        if reqid := resp.headers.get("nvcf-reqid"):
            headers["reqid"] = reqid
        if status := resp.headers.get("nvcf-status"):
            headers["status"] = status
        if location := resp.headers.get("location"):
            headers["location"] = location

        # Handle pipeline status
        if sts := resp.headers.get("CURATOR-PIPELINE-STATUS"):
            if spct := resp.headers.get("CURATOR-PIPELINE-PERCENT-COMPLETE"):
                headers["pct"] = spct
                retval["invoke-based-status"] = sts
        elif sts := resp.headers.get("CURATOR-TERM-STATUS"):
            retval["term-status"] = sts

        retval["headers"] = headers
