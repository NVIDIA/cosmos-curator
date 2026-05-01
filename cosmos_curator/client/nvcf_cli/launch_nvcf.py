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

"""Launch NVIDIA Cloud Functions.

This module provides a command-line interface for launching and managing NVIDIA Cloud Functions
using Typer. It sets up the main CLI application and registers various subcommands for different
functionalities.
"""

import typer

from .ncf.common import cc_client_instances as zins

cc_client_nvcf = typer.Typer(
    context_settings={
        "max_content_width": 120,
    },
    pretty_exceptions_enable=False,
    no_args_is_help=True,
)
for name, detail in zins().items():
    help_var = detail.get("help")
    app = detail.get("app")
    cc_client_nvcf.add_typer(typer_instance=app, name=name, help=help_var)
