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
"""Test the model_cli module."""

import argparse
from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import pytest

from cosmos_curator.core.managers.model_cli import (
    _download,
    _get_default_models,
    _unpack_model_info,
    _upload,
    main,
    setup_parsers,
)

# Mock models for testing
MOCK_MODELS = {
    "bert": {
        "model_id": "google-bert/bert-large-uncased",
        "version": "6da4b6a26a1877e173fca3225479512db81a5e5b",
        "filelist": ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "vocab.txt"],
    },
    "gpt2": {
        "model_id": "openai-community/gpt2",
        "version": "607a30d783dfa663caf39e06633721c8d4cfcd7e",
        "filelist": None,
    },
    "sam3": {
        "model_id": "facebook/sam3",
        "version": None,
        "filelist": None,
    },
    "aesthetic_scorer": {
        "model_id": "ttj/sac-logos-ava1-l14-linearMSE",
        "version": "1e77fa05081323d99725fc40a9bf9f88180490e7",
        "filelist": ["model.safetensors"],
    },
}
DEFAULT_EXCLUDED_MOCK_MODELS = {"sam3"}


@pytest.fixture(scope="module", autouse=True)
def mock_get_all_models() -> Iterator[MagicMock]:
    """Mock get_all_models for all tests."""
    with patch(
        "cosmos_curator.core.managers.model_cli.get_all_models", return_value=MOCK_MODELS, autospec=True
    ) as mock:
        yield mock


class TestSetupParsers:
    """Test the setup_parsers function."""

    def test_parser_has_subparsers(self) -> None:
        """Test that parser has download and upload subcommands."""
        parser = setup_parsers()

        # Test download subcommand exists
        args_download = parser.parse_args(["download"])
        assert args_download.command == "download"

        # Test upload subcommand exists
        args_upload = parser.parse_args(["upload", "--model-weights-prefix", "s3://test"])
        assert args_upload.command == "upload"

    def test_download_parser_has_models_argument(self) -> None:
        """Test that download parser has --models argument with default."""
        parser = setup_parsers()
        args = parser.parse_args(["download"])

        # Should have models argument with default value
        assert hasattr(args, "models")
        assert isinstance(args.models, str)
        # Default should skip models that require explicit opt-in.
        expected_default = ",".join(
            model for model in MOCK_MODELS if not model.startswith("_") and model not in DEFAULT_EXCLUDED_MOCK_MODELS
        )
        assert args.models == expected_default

    def test_default_models_exclude_gated_models(self) -> None:
        """Test that gated models are available but not downloaded by default."""
        default_models = _get_default_models()

        assert "sam3" in MOCK_MODELS
        assert "sam3" not in default_models

    def test_download_parser_accepts_custom_models(self) -> None:
        """Test that download parser accepts custom --models argument."""
        parser = setup_parsers()
        args = parser.parse_args(["download", "--models", "bert,gpt2"])

        assert args.models == "bert,gpt2"

    def test_upload_parser_requires_model_weights_prefix(self) -> None:
        """Test that upload parser requires --model-weights-prefix."""
        parser = setup_parsers()

        # Should fail without --model-weights-prefix
        with pytest.raises(SystemExit):
            parser.parse_args(["upload"])

    def test_upload_parser_accepts_model_weights_prefix(self) -> None:
        """Test that upload parser accepts --model-weights-prefix."""
        parser = setup_parsers()
        args = parser.parse_args(["upload", "--model-weights-prefix", "s3://bucket/path"])

        assert args.model_weights_prefix == "s3://bucket/path"

    def test_upload_parser_has_models_argument(self) -> None:
        """Test that upload parser has --models argument."""
        parser = setup_parsers()
        args = parser.parse_args(["upload", "--model-weights-prefix", "s3://test", "--models", "bert"])

        assert args.models == "bert"

    def test_parser_sets_func_attribute_for_download(self) -> None:
        """Test that download command sets func attribute to _download."""
        parser = setup_parsers()
        args = parser.parse_args(["download"])

        assert hasattr(args, "func")
        assert args.func == _download

    def test_parser_sets_func_attribute_for_upload(self) -> None:
        """Test that upload command sets func attribute to _upload."""
        parser = setup_parsers()
        args = parser.parse_args(["upload", "--model-weights-prefix", "s3://test"])

        assert hasattr(args, "func")
        assert args.func == _upload

    def test_parser_with_no_command_returns_none(self) -> None:
        """Test that parser with no subcommand sets command to None."""
        parser = setup_parsers()
        args = parser.parse_args([])

        assert args.command is None


class TestUnpackModelInfo:
    """Test the _unpack_model_info function."""

    def test_unpacks_model_with_version_and_filelist(self) -> None:
        """Test unpacking a model that has version and filelist."""
        model_id, version, filelist = _unpack_model_info("aesthetic_scorer")

        assert model_id == MOCK_MODELS["aesthetic_scorer"]["model_id"]
        assert version == MOCK_MODELS["aesthetic_scorer"]["version"]
        assert filelist == MOCK_MODELS["aesthetic_scorer"]["filelist"]

    def test_unpacks_model_with_null_filelist(self) -> None:
        """Test unpacking a model with null filelist."""
        model_id, version, filelist = _unpack_model_info("gpt2")

        assert model_id == MOCK_MODELS["gpt2"]["model_id"]
        assert version == MOCK_MODELS["gpt2"]["version"]
        assert filelist == MOCK_MODELS["gpt2"]["filelist"]

    def test_unpacks_model_with_multiple_files(self) -> None:
        """Test unpacking a model with multiple files in filelist."""
        model_id, version, filelist = _unpack_model_info("bert")

        assert model_id == MOCK_MODELS["bert"]["model_id"]
        assert version == MOCK_MODELS["bert"]["version"]
        assert filelist == MOCK_MODELS["bert"]["filelist"]

    def test_raises_error_for_unknown_model(self) -> None:
        """Test that unknown model raises ValueError."""
        with pytest.raises(ValueError, match="Unknown model nonexistent_model"):
            _unpack_model_info("nonexistent_model")

    def test_error_message_includes_available_models(self) -> None:
        """Test that error message includes list of available models."""
        with pytest.raises(ValueError, match="Available models:"):
            _unpack_model_info("invalid_model")


class TestDownload:
    """Test the _download function."""

    @patch("cosmos_curator.core.managers.model_cli.download_model_weights_from_huggingface_to_workspace")
    def test_downloads_single_model(self, mock_download: MagicMock) -> None:
        """Ensure a single model delegates to the download helper with the right args."""
        args = argparse.Namespace(models="bert")

        _download(args)

        mock_download.assert_called_once_with(*_unpack_model_info("bert"))

    @pytest.mark.parametrize(
        ("models", "expected"),
        [
            ("bert,gpt2", ["bert", "gpt2"]),
            ("bert, gpt2", ["bert", "gpt2"]),
        ],
    )
    @patch("cosmos_curator.core.managers.model_cli.download_model_weights_from_huggingface_to_workspace")
    def test_downloads_multiple_models(self, mock_download: MagicMock, models: str, expected: list[str]) -> None:
        """Ensure each requested model is delegated with correct arguments."""
        args = argparse.Namespace(models=models)

        _download(args)

        assert [call.args for call in mock_download.call_args_list] == [_unpack_model_info(model) for model in expected]

    def test_raises_error_for_invalid_model(self) -> None:
        """Test that downloading invalid model raises error."""
        args = argparse.Namespace(models="invalid_model")

        with pytest.raises(ValueError, match="Unknown model"):
            _download(args)


class TestUpload:
    """Test the _upload function."""

    @patch("cosmos_curator.core.managers.model_cli.push_huggingface_model_to_cloud_storage")
    def test_uploads_single_model(self, mock_upload: MagicMock) -> None:
        """Ensure single-model upload delegates with the expected args and prefix."""
        args = argparse.Namespace(models="bert", model_weights_prefix="s3://bucket/path")

        _upload(args)

        mock_upload.assert_called_once()
        call = mock_upload.call_args
        assert call.args == _unpack_model_info("bert")
        assert call.kwargs == {"model_weights_prefix": "s3://bucket/path"}

    @pytest.mark.parametrize(
        ("models", "expected"),
        [
            ("bert,gpt2", ["bert", "gpt2"]),
            ("bert, gpt2", ["bert", "gpt2"]),
        ],
    )
    @patch("cosmos_curator.core.managers.model_cli.push_huggingface_model_to_cloud_storage")
    def test_uploads_multiple_models(self, mock_upload: MagicMock, models: str, expected: list[str]) -> None:
        """Ensure each model is uploaded with the correct metadata and prefix."""
        args = argparse.Namespace(models=models, model_weights_prefix="s3://bucket/path")

        _upload(args)

        actual_calls = [(call.args, call.kwargs) for call in mock_upload.call_args_list]
        assert actual_calls == [
            (_unpack_model_info(model), {"model_weights_prefix": "s3://bucket/path"}) for model in expected
        ]

    def test_raises_error_for_invalid_model(self) -> None:
        """Test that uploading invalid model raises error."""
        args = argparse.Namespace(models="invalid_model", model_weights_prefix="s3://test")

        with pytest.raises(ValueError, match="Unknown model"):
            _upload(args)


class TestMain:
    """Test the main function."""

    @patch("cosmos_curator.core.managers.model_cli.download_model_weights_from_huggingface_to_workspace")
    def test_main_calls_download(self, mock_download: MagicMock) -> None:
        """Test that main calls download function for download command."""
        main(["download", "--models", "bert"])

        mock_download.assert_called_once()

    @patch("cosmos_curator.core.managers.model_cli.push_huggingface_model_to_cloud_storage")
    def test_main_calls_upload(self, mock_upload: MagicMock) -> None:
        """Test that main calls upload function for upload command."""
        main(["upload", "--model-weights-prefix", "s3://test", "--models", "bert"])

        mock_upload.assert_called_once()

    @patch("cosmos_curator.core.managers.model_cli.setup_parsers")
    def test_main_prints_help_without_command(self, mock_setup_parsers: MagicMock) -> None:
        """Test that main prints help when no command is provided."""
        mock_parser = MagicMock()
        mock_parser.parse_args.return_value = argparse.Namespace(command=None)
        mock_setup_parsers.return_value = mock_parser

        main()

        mock_parser.print_help.assert_called_once()

    def test_main_propagates_errors(self) -> None:
        """Test that main propagates errors from underlying functions."""
        with pytest.raises(ValueError, match="Unknown model"):
            main(["download", "--models", "invalid_model"])
