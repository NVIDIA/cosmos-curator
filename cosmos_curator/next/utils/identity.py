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

"""Shared stable-identity hashing for Curator Next recipes."""

import hashlib
import json
from collections.abc import Mapping
from typing import Any


def canonical_json(value: Any) -> str:  # noqa: ANN401
    """Render sorted compact UTF-8 JSON with non-finite values forbidden."""
    return json.dumps(value, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def canonical_digest(value: Any) -> str:  # noqa: ANN401
    """Hash the canonical JSON rendering of a value."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def json_snapshot(value: Mapping[str, Any] | str) -> str:
    """Freeze a JSON object as canonical text, so later caller mutations cannot change it.

    Canonical text passes through unchanged, as ``attrs.evolve`` supplies it to converters.
    """
    return canonical_json(json.loads(value) if isinstance(value, str) else dict(value))
