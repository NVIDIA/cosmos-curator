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
"""Zip archive helpers."""

import stat
import zipfile
from collections.abc import Iterable
from pathlib import Path


def _is_symlink(member: zipfile.ZipInfo) -> bool:
    return stat.S_ISLNK(member.external_attr >> 16)


def safe_extract_zip(zf: zipfile.ZipFile, dest_dir: str | Path, members: Iterable[str] | None = None) -> None:
    """Extract a zip archive without allowing path traversal or symlink entries."""
    dest_path = Path(dest_dir).resolve(strict=False)
    selected_names = set(members) if members is not None else None
    members_to_extract: list[zipfile.ZipInfo] = []

    for member in zf.infolist():
        if selected_names is not None and member.filename not in selected_names:
            continue

        if _is_symlink(member):
            error_msg = f"Refusing to extract symlink from zip archive: {member.filename}"
            raise ValueError(error_msg)

        target_path = (dest_path / member.filename).resolve(strict=False)
        try:
            target_path.relative_to(dest_path)
        except ValueError as exc:
            error_msg = f"Refusing to extract zip member outside destination: {member.filename}"
            raise ValueError(error_msg) from exc

        members_to_extract.append(member)

    zf.extractall(dest_path, members=members_to_extract)
