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
"""conftest.py for the slurm_cli package."""

from collections.abc import Generator
from unittest.mock import Mock

import pytest


@pytest.fixture(scope="session", autouse=True)
def mock_connection() -> Generator[Mock, None, None]:
    """Mock fabric module before it's imported anywhere."""
    with pytest.MonkeyPatch.context() as mp:
        mock = Mock()
        mock_connection = Mock()
        mock_connection.run.return_value = Mock()
        mock.Connection.return_value = mock_connection
        mp.setattr("fabric.Connection", mock.Connection)
        yield mock.Connection
