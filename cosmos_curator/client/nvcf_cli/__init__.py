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

"""Initialize the NVIDIA Cloud Function launcher package.

This package provides functionality for launching and managing NVIDIA Cloud Functions,
including asset management, configuration, image handling, and model deployment.
"""

from .ncf.asset import asset_manager
from .ncf.config import config_manager
from .ncf.image import image_manager
from .ncf.launcher import nvcf_driver
from .ncf.model import model_manager

__all__ = [
    "asset_manager",
    "config_manager",
    "image_manager",
    "model_manager",
    "nvcf_driver",
]
