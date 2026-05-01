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

"""Cosmos curator package."""

import warnings

# Suppress repetitive pkg_resources deprecation warnings
warnings.filterwarnings("ignore", category=UserWarning, message="pkg_resources is deprecated as an API")

# Third-party packages (ngcsdk's vendored `registry`, vllm, etc.) contain
# invalid escape sequences that trigger SyntaxWarning on Python 3.12+.
# The module pattern needs `.*` because re.match() is used internally and
# compile-time warnings use the full file path as the module name.
warnings.filterwarnings("ignore", category=SyntaxWarning, message=r"invalid escape sequence", module=r".*")
