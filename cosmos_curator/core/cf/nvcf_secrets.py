# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Read the NVCF secret store.

Deliberately stdlib-only. This lives apart from ``nvcf_utils`` (which imports ``ray``) and
from ``nvcf_model_manager`` (which imports ``ngcsdk``) so that callers needing nothing but
the secrets file -- the S3 and Azure clients, CLI tooling -- can import it directly from
environments where neither of those packages is installed.
"""

import json
import pathlib

NVCF_SECRETS_PATH = "/var/secrets/secrets.json"


def get_secrets_from_nvcf_secret_store() -> dict[str, str]:
    """Return the NVCF secrets as a dictionary, or an empty dict when not running on NVCF."""
    data: dict[str, str] = {}
    nsp = pathlib.Path(NVCF_SECRETS_PATH)
    if nsp.exists():
        with nsp.open(encoding="utf-8") as f:
            data = json.load(f)
    return data
