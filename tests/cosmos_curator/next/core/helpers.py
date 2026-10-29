# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read a local workspace's staged results as publication does, without Ray."""

from pathlib import Path

import pyarrow as pa

from cosmos_curator.next.core.ray_checkpoints import STAGED_FILE_EXTENSION, read_staged_table


def staged_results(directory: str) -> pa.Table:
    """Read every staged file under a local results directory."""
    paths = sorted(Path(directory).rglob(f"*.{STAGED_FILE_EXTENSION}"))
    return pa.concat_tables([read_staged_table(str(path)) for path in paths])
