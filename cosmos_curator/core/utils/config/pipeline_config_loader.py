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
"""Load pipeline config files (JSON or YAML).

The config format matches the NVCF invoke payload used by ``nvcf_main.py``::

    {
        "pipeline": "split",
        "args": {
            "input_video_path": "/data/videos",
            "output_clip_path": "/data/output",
            "limit": 0
        }
    }

Both JSON (``.json``) and YAML (``.yaml`` / ``.yml``) files are accepted.
The ``pipeline`` key names the subcommand; it is returned under the
reserved ``_pipeline`` key.  The ``args`` dict (or the top-level mapping
when ``args`` is absent) provides the pipeline parameters.
"""

import json
import pathlib
from typing import Any

import yaml

_YAML_SUFFIXES = frozenset({".yaml", ".yml"})


def load_pipeline_config(config_path: str | pathlib.Path) -> dict[str, Any]:
    """Load a pipeline config file and return a flat parameter dict.

    JSON and YAML formats are both supported; the file extension determines
    which parser is used.

    The returned dict contains every key from the ``args`` section (or from
    the top level when ``args`` is absent).  If the config contains a
    ``pipeline`` key its value is stored under the reserved ``_pipeline``
    key so the caller can ``pop`` it to learn which subcommand was requested.

    Args:
        config_path: Path to the JSON or YAML config file.

    Returns:
        A flat dict of pipeline parameters.  The ``_pipeline`` key holds
        the pipeline name (or is absent when the config has no
        ``pipeline`` key).

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If the config file cannot be parsed.
        TypeError: If the parsed content is not a mapping.

    """
    path = pathlib.Path(config_path)
    if not path.exists():
        msg = f"Pipeline config file not found: {path}"
        raise FileNotFoundError(msg)

    try:
        with path.open() as f:
            loaded = yaml.safe_load(f) if path.suffix.lower() in _YAML_SUFFIXES else json.load(f)
    except (json.JSONDecodeError, yaml.YAMLError) as exc:
        msg = f"Failed to parse config {path}: {exc}"
        raise ValueError(msg) from exc

    if loaded is None:
        loaded = {}
    if not isinstance(loaded, dict):
        msg = f"Config file must contain a mapping at the top level, got {type(loaded).__name__}: {path}"
        raise TypeError(msg)

    raw: dict[str, Any] = loaded
    pipeline = raw.get("pipeline")

    if "args" in raw and isinstance(raw["args"], dict):
        defaults = dict(raw["args"])
    else:
        defaults = dict(raw)
        defaults.pop("pipeline", None)

    if pipeline is not None:
        defaults["_pipeline"] = pipeline

    return defaults
