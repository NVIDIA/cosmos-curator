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
"""Module for processing and manipulating webdataset datasets.

WebDataset is a data format optimized for machine learning applications. Instead of using individual files or complex
database formats to store samples, WebDataset stores samples inside Tar archives. This approach offers efficiency
for distributed and large-scale training, especially when dealing with web/cloud storage.

Key Components:

- Tar files: At its core, WebDataset uses Tar archives as its primary container format. These Tar archives, or "shards",
  contain a subset of the dataset's samples. Each sample conists of one or more "files" inside a tar file. The name of
  "file" corresponds to the sample name and the suffix of the file corresponds to the file type.
- Parts: The dataset is often distrubuted into multiple "parts". Each part corresponds to a single directory, or, in the
  case of S3, a "virtual" directory. This allows us to make sure each directory only contains so many files, which has
  benefits for both real linux directories as well as "virtual" S3 directories.

Advantages of WebDataset include efficiency for streaming and distributed training, simplicity with the use of standard
Tar format, and flexibility to accommodate multiple data types in one dataset.

Right now, these utilities work at the "sample" level. They only deal with moving samples around and repackaging them,
but are ignorant to the actual contents of the samples.
"""

import collections
import io
import pathlib
import tarfile
import zipfile
from collections.abc import Iterable
from typing import Any

import attr
import webdataset  # type: ignore[import-untyped]


@attr.define
class RawSample:
    """Represent a single raw data sample."""

    key: str  # Unique identifier for the sample
    data: dict[str, Any]  # Data associated with the sample, organized by file suffix


def make_part_path_str(part_num: int) -> str:
    """Make a part path string.

    Args:
        part_num: The part number.

    Returns:
        The part path string.

    """
    return f"part_{part_num:06d}"


def make_tar_path_str(tar_num: int) -> str:
    """Make a tar path string.

    Args:
        tar_num: The tar number.

    Returns:
        The tar path string.

    """
    return f"{tar_num:06d}.tar"


def get_part_num_from_path_str(path_str: str) -> int:
    """Get the part number from a path string.

    Args:
        path_str: The path string.

    Returns:
        The part number.

    """
    prefix = "part_"
    if prefix not in path_str:
        error_msg = f"Invalid path string: {path_str}"
        raise ValueError(error_msg)
    part_num_str = path_str.split(prefix)[1]
    try:
        part_num = int(part_num_str)
    except ValueError as err:
        error_msg = f"Invalid part number: {part_num_str}"
        raise ValueError(error_msg) from err
    return part_num


def make_tar_from_samples(raw_samples: Iterable[RawSample]) -> bytes:
    """Create a webdataste TAR archive from raw samples.

    Args:
        raw_samples (Iterable[RawSample]): List or iterable of samples to be archived.

    Returns:
        bytes: TAR archive as bytes.

    """
    bytes_io = io.BytesIO()
    with webdataset.TarWriter(bytes_io) as tar_writer:
        for sample in raw_samples:
            out_dict = dict(sample.data)
            out_dict["__key__"] = sample.key
            tar_writer.write(out_dict)
    bytes_io.seek(0)
    return bytes_io.getvalue()


def read_raw_samples_from_archive(data: bytes, archive_type: str) -> list[RawSample]:
    """Extract raw samples from a webdataset TAR/ZIP archive.

    Args:
        data (bytes): TAR/ZIP archive as bytes.
        archive_type (str): Type of archive. Currently only supports "tar" and "zip".

    Returns:
        list[RawSample]: A list of samples extracted from the TAR/ZIP archive.

    """
    sample_dict: dict[str, Any] = collections.defaultdict(dict)

    if archive_type == "tar":
        with tarfile.open(fileobj=io.BytesIO(data)) as f:
            for t_member in f:
                path = pathlib.Path(t_member.name)
                file = f.extractfile(t_member.name)
                assert file is not None
                member_bytes = file.read()
                key = str(path.parent / path.stem)
                suffix = path.suffix[1:]

                # Ensure that each sample has a unique suffix
                if suffix in sample_dict[key]:
                    error_msg = f"Duplicate suffix {suffix} for sample {key}"
                    raise ValueError(error_msg)
                sample_dict[key][suffix] = member_bytes
    elif archive_type == "zip":
        with zipfile.ZipFile(io.BytesIO(data)) as f:
            for z_member in f.namelist():
                path = pathlib.Path(z_member)
                if path.is_dir():  # Skip directories
                    continue
                with f.open(z_member) as file:
                    member_bytes = file.read()
                key = str(path.parent / path.stem)
                suffix = path.suffix[1:]

                # Ensure that each sample has a unique suffix
                if suffix in sample_dict[key]:
                    error_msg = f"Duplicate suffix {suffix} for sample {key}"
                    raise ValueError(error_msg)
                sample_dict[key][suffix] = member_bytes
    else:
        error_msg = "Unsupported archive type"
        raise ValueError(error_msg)

    return [RawSample(key, value) for key, value in sample_dict.items()]
