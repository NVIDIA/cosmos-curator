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
"""CloudEvent / Kratos handling for the benchmarks."""

import uuid
from datetime import UTC, datetime
from typing import Any, cast

import requests
import tenacity

from benchmarks.secrets import KratosSecrets


@tenacity.retry(stop=tenacity.stop_after_attempt(3), wait=tenacity.wait_exponential(multiplier=1, min=4, max=15))
def push_cloudevent(cloudevent: dict[str, Any], cloudevent_endpoint: str, secrets: KratosSecrets) -> dict[str, Any]:
    """Push cloudevent to cloudevent endpoint.

    Args:
        cloudevent: cloudevent to push, must be json serializable.
        cloudevent_endpoint: cloudevent endpoint.
        secrets: Kratos secrets.

    """
    headers = {"Content-Type": "application/cloudevents-batch+json", "Authorization": f"Bearer {secrets.bearer_token}"}
    response = requests.post(cloudevent_endpoint, json=cloudevent, headers=headers, timeout=30)
    response.raise_for_status()
    return cast("dict[str, Any]", response.json())


def make_cloudevent(data: dict[str, Any]) -> dict[str, Any]:
    """Make cloudevent metrics.

    Args:
        data: data to put into the cloudevent, must be json serializable.

    Returns:
        CloudEvent dictionary

    """
    return {
        "specversion": "1.0",
        "id": f"{uuid.uuid4()!s}",
        "time": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "source": f"cosmos-curator-{uuid.uuid4()!s}",
        "type": "performance-benchmark",
        "subject": "nvcf-performance-metrics",
        "data": data,
    }
