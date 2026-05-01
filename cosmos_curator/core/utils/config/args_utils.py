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
"""Simple utilities for easier input argument management."""

import argparse
from collections.abc import Callable
from typing import Any

_CLI_ONLY_ARGS = ["config_file"]


def fill_default_args(
    args: argparse.Namespace,
    setup_parser: Callable[[argparse.ArgumentParser], None],
) -> argparse.Namespace:
    """Fill default arguments.

    Args:
        args: The arguments to fill.
        setup_parser: The parser to setup.

    Returns:
        The filled arguments.

    """
    parser = argparse.ArgumentParser()
    setup_parser(parser)
    for action in parser._actions:  # noqa: SLF001
        if not hasattr(args, action.dest):
            if action.required and action.dest not in _CLI_ONLY_ARGS:
                error_msg = f"Required argument {action.dest} not provided"
                raise ValueError(error_msg)
            setattr(args, action.dest, action.default)
    return args


def modify_argument_choice(parser: argparse.ArgumentParser, argument_name: str, choices: list[Any]) -> None:
    """Modify the choices for an argument.

    Args:
        parser: The parser to modify.
        argument_name: The name of the argument to modify.
        choices: The choices to set.

    """
    for action in parser._actions:  # noqa: SLF001
        if action.dest == argument_name:
            action.choices = choices
            return
