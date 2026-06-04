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

"""Cosmos Curator Client Setup.

This script handles the setup and installation process for the Cosmos Curator package.
It reads metadata from pyproject.toml, prepares the build directory structure,
and configures the package for distribution.
"""

import shutil
import sys
from collections.abc import Iterable
from pathlib import Path

import tomli
from setuptools import find_packages, setup


def load_project_name() -> str:
    """Load the package name from pyproject.toml.

    Package metadata, including the dynamic version, is supplied by pyproject.toml
    and setuptools-scm. This script only needs the name for build-directory paths.

    Returns:
        Package name.

    """
    with Path("pyproject.toml").open("rb") as f:
        pyproject = tomli.load(f)
    return pyproject["project"]["name"]


name = load_project_name()

build_dir = "build"
dist_dir = "dist"
pkg_path = Path(build_dir) / name
src_env_file = Path(name) / "core" / "utils" / "environment.py"
src_storage_dir = Path(name) / "core" / "utils" / "storage"
src_init_file = Path(name) / "__init__.py"
dst_core_dir = pkg_path / "core"
dst_utils_dir = dst_core_dir / "utils"
dst_storage_dir = dst_utils_dir / "storage"
dst_client_dir = pkg_path / "client"

# Ensure build directory exists
Path(build_dir).mkdir(exist_ok=True)

# License header to be used in generated files
copyright_header = [
    "# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.",
    "# SPDX-License-Identifier: Apache-2.0",
    "#",
    '# Licensed under the Apache License, Version 2.0 (the "License");',
    "# you may not use this file except in compliance with the License.",
    "# You may obtain a copy of the License at",
    "#",
    "# http://www.apache.org/licenses/LICENSE-2.0",
    "#",
    "# Unless required by applicable law or agreed to in writing, software",
    '# distributed under the License is distributed on an "AS IS" BASIS,',
    "# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.",
    "# See the License for the specific language governing permissions and",
    "# limitations under the License.",
]


def copy_required_files(src_dir: Path, dst_dir: Path, filenames: Iterable[str]) -> None:
    """Copy selected files from a source directory, failing when any are missing."""
    if not src_dir.is_dir():
        error_msg = f"Missing source directory: {src_dir}"
        raise FileNotFoundError(error_msg)

    dst_dir.mkdir(parents=True, exist_ok=True)
    for filename in filenames:
        src_file = src_dir / filename
        if not src_file.is_file():
            error_msg = f"Missing required file: {src_file}"
            raise FileNotFoundError(error_msg)
        shutil.copy2(src_file, dst_dir)


def build_package() -> None:
    """Build the package before setup() is called."""
    # Clean build and dist directories
    shutil.rmtree(Path(build_dir), ignore_errors=True)
    shutil.rmtree(Path(dist_dir), ignore_errors=True)

    Path(build_dir).mkdir(exist_ok=True)

    dst_utils_dir.mkdir(parents=True, exist_ok=True)

    client_src = Path(name) / "client"
    if client_src.exists():
        shutil.copytree(client_src, dst_client_dir)

    if src_init_file.exists():
        shutil.copy2(src_init_file, pkg_path / "__init__.py")
    else:
        # Create a default __init__.py with license header if original doesn't exist
        with (pkg_path / "__init__.py").open("w") as f:
            for line in copyright_header:
                f.write(f"{line}\n")

    with (dst_core_dir / "__init__.py").open("w") as f:
        for line in copyright_header:
            f.write(f"{line}\n")

    with (dst_utils_dir / "__init__.py").open("w") as f:
        for line in copyright_header:
            f.write(f"{line}\n")
        f.write('"""Environment."""\n')

    if src_env_file.exists():
        shutil.copy2(src_env_file, dst_utils_dir)

    copy_required_files(src_storage_dir, dst_storage_dir, ("__init__.py", "zip_utils.py"))

    examples_src = Path("examples")
    examples_dst = pkg_path / "examples"
    if examples_src.exists():
        shutil.copytree(examples_src, examples_dst)


is_editable_install = any(command in sys.argv for command in ("develop", "editable_wheel"))
if is_editable_install:
    package_search_dir = "."
    package_dir = {}
else:
    build_package()
    package_search_dir = build_dir
    package_dir = {"": build_dir}

setup(
    name=name,
    packages=find_packages(where=package_search_dir, include=[name, f"{name}.*"]),
    include_package_data=True,
    package_dir=package_dir,
    package_data={
        name: ["examples/**/*"],
        f"{name}.client.nvcf_cli.ncf.launcher": ["helm_values/*"],
    },
)
