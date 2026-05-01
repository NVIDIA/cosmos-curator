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

"""Shared fixtures for Ray Data pipeline tests."""

from collections.abc import Generator
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def mock_get_tmp_dir(tmp_path: Path) -> Generator[None, None, None]:
    """Redirect ``get_tmp_dir`` to pytest's ``tmp_path``.

    The default ``/config/tmp`` is not writable in CI/CD environments. This
    autouse fixture mirrors the one in ``tests/.../pipelines/video/conftest.py``
    so Ray Data tests that touch video utilities don't fail on tmpdir creation.
    """
    with patch("cosmos_curator.core.utils.config.operation_context.get_tmp_dir", return_value=tmp_path):
        yield
