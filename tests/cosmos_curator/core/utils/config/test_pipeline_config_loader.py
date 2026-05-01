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
"""Tests for the simplified flat-args pipeline config loader (JSON and YAML)."""

import json
import textwrap
from typing import TYPE_CHECKING

import pytest

from cosmos_curator.core.utils.config.pipeline_config_loader import load_pipeline_config

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# JSON loading
# ---------------------------------------------------------------------------
class TestLoadJsonConfig:
    """Tests for loading flat NVCF-compatible JSON configs."""

    def test_extracts_args_dict(self, tmp_path: "Path") -> None:
        """Return the ``args`` sub-dict with ``_pipeline`` when present."""
        cfg = {"pipeline": "split", "args": {"input_video_path": "/data", "limit": 5}}
        config_file = tmp_path / "cfg.json"
        config_file.write_text(json.dumps(cfg))
        result = load_pipeline_config(config_file)
        assert result.pop("_pipeline") == "split"
        assert result == {"input_video_path": "/data", "limit": 5}

    def test_pipeline_key_excluded_from_args(self, tmp_path: "Path") -> None:
        """The raw ``pipeline`` key must not leak into the returned dict."""
        cfg = {"pipeline": "split", "args": {"limit": 0}}
        config_file = tmp_path / "cfg.json"
        config_file.write_text(json.dumps(cfg))
        result = load_pipeline_config(config_file)
        assert "pipeline" not in result

    def test_fallback_without_args_key(self, tmp_path: "Path") -> None:
        """When no ``args`` key exists, return the whole dict minus ``pipeline``."""
        cfg = {"pipeline": "split", "input_video_path": "/data", "limit": 0}
        config_file = tmp_path / "cfg.json"
        config_file.write_text(json.dumps(cfg))
        result = load_pipeline_config(config_file)
        assert result.pop("_pipeline") == "split"
        assert result == {"input_video_path": "/data", "limit": 0}

    def test_missing_pipeline_key(self, tmp_path: "Path") -> None:
        """``_pipeline`` is absent when the config has no ``pipeline`` key."""
        cfg = {"args": {"limit": 5}}
        config_file = tmp_path / "cfg.json"
        config_file.write_text(json.dumps(cfg))
        result = load_pipeline_config(config_file)
        assert "_pipeline" not in result
        assert result == {"limit": 5}

    def test_null_values_preserved(self, tmp_path: "Path") -> None:
        """JSON null values are preserved as Python None."""
        cfg = {"pipeline": "split", "args": {"captioning_prompt_text": None, "limit": 0}}
        config_file = tmp_path / "cfg.json"
        config_file.write_text(json.dumps(cfg))
        result = load_pipeline_config(config_file)
        assert result["captioning_prompt_text"] is None
        assert result["limit"] == 0

    def test_boolean_values(self, tmp_path: "Path") -> None:
        """JSON booleans are preserved as Python bools."""
        cfg = {"pipeline": "split", "args": {"generate_captions": True, "verbose": False}}
        config_file = tmp_path / "cfg.json"
        config_file.write_text(json.dumps(cfg))
        result = load_pipeline_config(config_file)
        assert result["generate_captions"] is True
        assert result["verbose"] is False

    def test_empty_args(self, tmp_path: "Path") -> None:
        """An empty ``args`` dict returns only ``_pipeline``."""
        cfg = {"pipeline": "split", "args": {}}
        config_file = tmp_path / "cfg.json"
        config_file.write_text(json.dumps(cfg))
        result = load_pipeline_config(config_file)
        assert result.pop("_pipeline") == "split"
        assert result == {}

    def test_invalid_json_raises(self, tmp_path: "Path") -> None:
        """Raise ValueError on malformed JSON."""
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("{invalid json:")
        with pytest.raises(ValueError, match="Failed to parse"):
            load_pipeline_config(bad_file)


# ---------------------------------------------------------------------------
# YAML loading
# ---------------------------------------------------------------------------
class TestLoadYamlConfig:
    """Tests for loading YAML configs (.yaml and .yml)."""

    def test_yaml_extracts_args(self, tmp_path: "Path") -> None:
        """YAML files with an ``args`` key behave identically to JSON."""
        content = textwrap.dedent("""\
            pipeline: split
            args:
              input_video_path: /data
              limit: 5
        """)
        config_file = tmp_path / "cfg.yaml"
        config_file.write_text(content)
        result = load_pipeline_config(config_file)
        assert result.pop("_pipeline") == "split"
        assert result == {"input_video_path": "/data", "limit": 5}

    def test_yml_extension(self, tmp_path: "Path") -> None:
        """The ``.yml`` extension is also recognised."""
        content = textwrap.dedent("""\
            pipeline: shard
            args:
              input_clip_path: /clips
        """)
        config_file = tmp_path / "cfg.yml"
        config_file.write_text(content)
        result = load_pipeline_config(config_file)
        assert result.pop("_pipeline") == "shard"
        assert result == {"input_clip_path": "/clips"}

    def test_yaml_fallback_without_args(self, tmp_path: "Path") -> None:
        """Flat YAML without ``args`` returns dict minus ``pipeline``."""
        content = textwrap.dedent("""\
            pipeline: split
            input_video_path: /data
            limit: 0
        """)
        config_file = tmp_path / "cfg.yaml"
        config_file.write_text(content)
        result = load_pipeline_config(config_file)
        assert result.pop("_pipeline") == "split"
        assert result == {"input_video_path": "/data", "limit": 0}

    def test_yaml_null_and_bool(self, tmp_path: "Path") -> None:
        """YAML null and boolean values are preserved correctly."""
        content = textwrap.dedent("""\
            pipeline: split
            args:
              captioning_prompt_text: null
              generate_captions: true
              verbose: false
        """)
        config_file = tmp_path / "cfg.yaml"
        config_file.write_text(content)
        result = load_pipeline_config(config_file)
        assert result["captioning_prompt_text"] is None
        assert result["generate_captions"] is True
        assert result["verbose"] is False

    def test_invalid_yaml_raises(self, tmp_path: "Path") -> None:
        """Raise ValueError on malformed YAML."""
        bad_file = tmp_path / "bad.yaml"
        bad_file.write_text("pipeline: [\ninvalid")
        with pytest.raises(ValueError, match="Failed to parse"):
            load_pipeline_config(bad_file)

    def test_json_yaml_equivalence(self, tmp_path: "Path") -> None:
        """The same config in JSON and YAML produces identical results."""
        json_cfg = {"pipeline": "split", "args": {"limit": 5, "verbose": False}}
        json_file = tmp_path / "cfg.json"
        json_file.write_text(json.dumps(json_cfg))

        yaml_content = textwrap.dedent("""\
            pipeline: split
            args:
              limit: 5
              verbose: false
        """)
        yaml_file = tmp_path / "cfg.yaml"
        yaml_file.write_text(yaml_content)

        assert load_pipeline_config(json_file) == load_pipeline_config(yaml_file)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------
class TestErrorHandling:
    """Tests for error paths."""

    def test_missing_file_raises(self, tmp_path: "Path") -> None:
        """Raise FileNotFoundError when the config file does not exist."""
        with pytest.raises(FileNotFoundError, match="not found"):
            load_pipeline_config(tmp_path / "nonexistent.json")

    def test_empty_json_file(self, tmp_path: "Path") -> None:
        """An empty JSON file (not valid JSON) raises ValueError."""
        empty_file = tmp_path / "empty.json"
        empty_file.write_text("")
        with pytest.raises(ValueError, match="Failed to parse"):
            load_pipeline_config(empty_file)

    def test_empty_yaml_file(self, tmp_path: "Path") -> None:
        """An empty YAML file returns an empty dict with no ``_pipeline``."""
        empty_file = tmp_path / "empty.yaml"
        empty_file.write_text("")
        result = load_pipeline_config(empty_file)
        assert "_pipeline" not in result
        assert result == {}

    def test_null_only_yaml_file(self, tmp_path: "Path") -> None:
        """A YAML file containing only ``null`` returns an empty dict."""
        null_file = tmp_path / "null.yaml"
        null_file.write_text("null\n")
        result = load_pipeline_config(null_file)
        assert "_pipeline" not in result
        assert result == {}

    def test_non_mapping_json_raises(self, tmp_path: "Path") -> None:
        """A JSON file with a top-level list raises TypeError."""
        list_file = tmp_path / "list.json"
        list_file.write_text("[1, 2, 3]")
        with pytest.raises(TypeError, match="mapping at the top level"):
            load_pipeline_config(list_file)

    def test_non_mapping_yaml_raises(self, tmp_path: "Path") -> None:
        """A YAML file with a top-level list raises TypeError."""
        list_file = tmp_path / "list.yaml"
        list_file.write_text("- one\n- two\n")
        with pytest.raises(TypeError, match="mapping at the top level"):
            load_pipeline_config(list_file)

    def test_scalar_json_raises(self, tmp_path: "Path") -> None:
        """A JSON file with a bare string raises TypeError."""
        str_file = tmp_path / "scalar.json"
        str_file.write_text('"just a string"')
        with pytest.raises(TypeError, match="mapping at the top level"):
            load_pipeline_config(str_file)

    def test_scalar_yaml_raises(self, tmp_path: "Path") -> None:
        """A YAML file with a bare scalar raises TypeError."""
        str_file = tmp_path / "scalar.yaml"
        str_file.write_text("42\n")
        with pytest.raises(TypeError, match="mapping at the top level"):
            load_pipeline_config(str_file)

    def test_does_not_mutate_original(self, tmp_path: "Path") -> None:
        """Returned dict is a copy; mutating it does not affect subsequent loads."""
        cfg = {"pipeline": "split", "args": {"limit": 0}}
        config_file = tmp_path / "cfg.json"
        config_file.write_text(json.dumps(cfg))

        result1 = load_pipeline_config(config_file)
        result1["limit"] = 999

        result2 = load_pipeline_config(config_file)
        assert result2["limit"] == 0
