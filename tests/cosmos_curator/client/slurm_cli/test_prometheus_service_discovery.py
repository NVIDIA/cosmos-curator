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
"""Test the prometheus_service_discovery module."""

import argparse
import json
import pathlib

import pytest

from cosmos_curator.client.slurm_cli.prometheus_service_discovery import (
    _setup_parser,
    create_slurm_service_discovery,
)


class TestCreateSlurmServiceDiscovery:
    """Test the create_slurm_service_discovery function."""

    def test_creates_valid_json_file(self, tmp_path: pathlib.Path) -> None:
        """Test that a valid service discovery JSON file is created."""
        # Create temporary hostfile
        hostfile = tmp_path / "hostfile.txt"
        hostfile.write_text("node1\nnode2\nnode3\n")

        # Create output path
        output_path = tmp_path / "service_discovery.json"

        # Create args namespace
        args = argparse.Namespace(
            path=output_path,
            job_user="test_user",
            job_id="12345",
            job_name="test_job",
            hostfile=hostfile,
            port=8080,
        )

        # Run the function
        create_slurm_service_discovery(args)

        # Verify output file exists
        assert output_path.exists()

        # Verify JSON structure
        with output_path.open("rt") as f:
            data = json.load(f)

        assert isinstance(data, list)
        assert len(data) == 1
        assert "labels" in data[0]
        assert "targets" in data[0]

    def test_correct_labels_in_output(self, tmp_path: pathlib.Path) -> None:
        """Test that the output contains correct labels."""
        hostfile = tmp_path / "hostfile.txt"
        hostfile.write_text("node1\n")
        output_path = tmp_path / "service_discovery.json"

        args = argparse.Namespace(
            path=output_path,
            job_user="alice",
            job_id="67890",
            job_name="my_pipeline",
            hostfile=hostfile,
            port=9090,
        )

        create_slurm_service_discovery(args)

        with output_path.open("rt") as f:
            data = json.load(f)

        labels = data[0]["labels"]
        assert labels["job"] == "cosmos-curator"
        assert labels["slurm_job_user"] == "alice"
        assert labels["slurm_job_id"] == "67890"
        assert labels["slurm_job_name"] == "my_pipeline"

    def test_correct_targets_in_output(self, tmp_path: pathlib.Path) -> None:
        """Test that the output contains correct targets."""
        hostfile = tmp_path / "hostfile.txt"
        hostfile.write_text("host1\nhost2\nhost3\n")
        output_path = tmp_path / "service_discovery.json"

        args = argparse.Namespace(
            path=output_path,
            job_user="user",
            job_id="123",
            job_name="job",
            hostfile=hostfile,
            port=7070,
        )

        create_slurm_service_discovery(args)

        with output_path.open("rt") as f:
            data = json.load(f)

        targets = data[0]["targets"]
        assert len(targets) == 3
        assert "host1:7070" in targets
        assert "host2:7070" in targets
        assert "host3:7070" in targets

    def test_handles_empty_hostfile(self, tmp_path: pathlib.Path) -> None:
        """Test that an empty hostfile results in empty targets list."""
        hostfile = tmp_path / "hostfile.txt"
        hostfile.write_text("")
        output_path = tmp_path / "service_discovery.json"

        args = argparse.Namespace(
            path=output_path,
            job_user="user",
            job_id="123",
            job_name="job",
            hostfile=hostfile,
            port=8080,
        )

        create_slurm_service_discovery(args)

        with output_path.open("rt") as f:
            data = json.load(f)

        assert data[0]["targets"] == []

    def test_handles_hostfile_with_blank_lines(self, tmp_path: pathlib.Path) -> None:
        """Test that blank lines in hostfile are ignored."""
        hostfile = tmp_path / "hostfile.txt"
        hostfile.write_text("host1\n\nhost2\n  \nhost3\n")
        output_path = tmp_path / "service_discovery.json"

        args = argparse.Namespace(
            path=output_path,
            job_user="user",
            job_id="123",
            job_name="job",
            hostfile=hostfile,
            port=8080,
        )

        create_slurm_service_discovery(args)

        with output_path.open("rt") as f:
            data = json.load(f)

        targets = data[0]["targets"]
        assert len(targets) == 3
        assert "host1:8080" in targets
        assert "host2:8080" in targets
        assert "host3:8080" in targets

    def test_handles_hostfile_with_whitespace(self, tmp_path: pathlib.Path) -> None:
        """Test that hostnames with leading/trailing whitespace are stripped."""
        hostfile = tmp_path / "hostfile.txt"
        hostfile.write_text("  host1  \n\thost2\t\n host3 \n")
        output_path = tmp_path / "service_discovery.json"

        args = argparse.Namespace(
            path=output_path,
            job_user="user",
            job_id="123",
            job_name="job",
            hostfile=hostfile,
            port=8080,
        )

        create_slurm_service_discovery(args)

        with output_path.open("rt") as f:
            data = json.load(f)

        targets = data[0]["targets"]
        assert "host1:8080" in targets
        assert "host2:8080" in targets
        assert "host3:8080" in targets

    @pytest.mark.parametrize("port", [80, 443, 8080, 9090, 65535])
    def test_handles_different_port_numbers(self, port: int, tmp_path: pathlib.Path) -> None:
        """Test that different port numbers are handled correctly."""
        hostfile = tmp_path / "hostfile.txt"
        hostfile.write_text("node1\n")
        output_path = tmp_path / "service_discovery.json"

        args = argparse.Namespace(
            path=output_path,
            job_user="user",
            job_id="123",
            job_name="job",
            hostfile=hostfile,
            port=port,
        )

        create_slurm_service_discovery(args)

        with output_path.open("rt") as f:
            data = json.load(f)

        assert data[0]["targets"][0] == f"node1:{port}"

    def test_json_formatting(self, tmp_path: pathlib.Path) -> None:
        """Test that JSON output is properly formatted with indentation."""
        hostfile = tmp_path / "hostfile.txt"
        hostfile.write_text("node1\n")
        output_path = tmp_path / "service_discovery.json"

        args = argparse.Namespace(
            path=output_path,
            job_user="user",
            job_id="123",
            job_name="job",
            hostfile=hostfile,
            port=8080,
        )

        create_slurm_service_discovery(args)

        # Read raw content to check formatting
        content = output_path.read_text()

        # Verify JSON is formatted with indent=2 (no trailing newline)
        parsed = json.loads(content)
        expected_content = json.dumps(parsed, indent=2)
        assert content == expected_content

    def test_fails_when_parent_directories_missing(self, tmp_path: pathlib.Path) -> None:
        """Test that parent directories are not automatically created (should fail)."""
        hostfile = tmp_path / "hostfile.txt"
        hostfile.write_text("node1\n")

        # Create output path in non-existent directory
        output_path = tmp_path / "nested" / "dir" / "service_discovery.json"

        args = argparse.Namespace(
            path=output_path,
            job_user="user",
            job_id="123",
            job_name="job",
            hostfile=hostfile,
            port=8080,
        )

        # Should raise FileNotFoundError since parent directory doesn't exist
        with pytest.raises(FileNotFoundError):
            create_slurm_service_discovery(args)

    def test_raises_error_for_missing_hostfile(self, tmp_path: pathlib.Path) -> None:
        """Test that missing hostfile raises appropriate error."""
        output_path = tmp_path / "service_discovery.json"
        hostfile = tmp_path / "nonexistent_hostfile.txt"

        args = argparse.Namespace(
            path=output_path,
            job_user="user",
            job_id="123",
            job_name="job",
            hostfile=hostfile,
            port=8080,
        )

        with pytest.raises(FileNotFoundError):
            create_slurm_service_discovery(args)

    def test_overwrites_existing_file(self, tmp_path: pathlib.Path) -> None:
        """Test that existing output file is overwritten."""
        hostfile = tmp_path / "hostfile.txt"
        hostfile.write_text("node1\n")
        output_path = tmp_path / "service_discovery.json"

        # Create initial file
        output_path.write_text("old content")

        args = argparse.Namespace(
            path=output_path,
            job_user="user",
            job_id="123",
            job_name="job",
            hostfile=hostfile,
            port=8080,
        )

        create_slurm_service_discovery(args)

        # Verify file was overwritten with new content
        content = output_path.read_text()
        assert "old content" not in content
        data = json.loads(content)
        assert data[0]["targets"][0] == "node1:8080"

    def test_handles_single_host(self, tmp_path: pathlib.Path) -> None:
        """Test that single host is handled correctly."""
        hostfile = tmp_path / "hostfile.txt"
        hostfile.write_text("single-node\n")
        output_path = tmp_path / "service_discovery.json"

        args = argparse.Namespace(
            path=output_path,
            job_user="user",
            job_id="999",
            job_name="single_node_job",
            hostfile=hostfile,
            port=3000,
        )

        create_slurm_service_discovery(args)

        with output_path.open("rt") as f:
            data = json.load(f)

        assert len(data[0]["targets"]) == 1
        assert data[0]["targets"][0] == "single-node:3000"

    def test_handles_many_hosts(self, tmp_path: pathlib.Path) -> None:
        """Test that many hosts are handled correctly."""
        hostfile = tmp_path / "hostfile.txt"
        hosts = [f"node{i:03d}" for i in range(100)]
        hostfile.write_text("\n".join(hosts) + "\n")
        output_path = tmp_path / "service_discovery.json"

        args = argparse.Namespace(
            path=output_path,
            job_user="user",
            job_id="123",
            job_name="large_job",
            hostfile=hostfile,
            port=8080,
        )

        create_slurm_service_discovery(args)

        with output_path.open("rt") as f:
            data = json.load(f)

        targets = data[0]["targets"]
        assert len(targets) == 100
        assert "node000:8080" in targets
        assert "node099:8080" in targets


class TestSetupParser:
    """Test the _setup_parser function."""

    def test_parser_has_all_required_arguments(self, tmp_path: pathlib.Path) -> None:
        """Test that parser has all required arguments."""
        parser = _setup_parser()

        # Test by parsing valid arguments and checking the namespace
        hostfile = tmp_path / "hostfile.txt"
        hostfile.touch()
        output_path = tmp_path / "output.json"

        args = parser.parse_args(
            [
                "--path",
                str(output_path),
                "--job-user",
                "user",
                "--job-id",
                "123",
                "--job-name",
                "job",
                "--hostfile",
                str(hostfile),
                "--port",
                "8080",
            ]
        )

        expected_args = {"path", "job_user", "job_id", "job_name", "hostfile", "port"}
        assert set(vars(args).keys()) == expected_args

    def test_parser_requires_all_arguments(self) -> None:
        """Test that parser requires all arguments."""
        parser = _setup_parser()

        # Try parsing with missing arguments
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_parser_parses_valid_arguments(self, tmp_path: pathlib.Path) -> None:
        """Test that parser correctly parses valid arguments."""
        parser = _setup_parser()

        hostfile = tmp_path / "hostfile.txt"
        hostfile.touch()
        output_path = tmp_path / "output.json"

        args = parser.parse_args(
            [
                "--path",
                str(output_path),
                "--job-user",
                "test_user",
                "--job-id",
                "12345",
                "--job-name",
                "test_job",
                "--hostfile",
                str(hostfile),
                "--port",
                "8080",
            ]
        )

        assert args.path == output_path
        assert args.job_user == "test_user"
        assert args.job_id == "12345"
        assert args.job_name == "test_job"
        assert args.hostfile == hostfile
        assert args.port == 8080

    def test_parser_path_is_pathlib_path(self, tmp_path: pathlib.Path) -> None:
        """Test that parser converts path arguments to pathlib.Path."""
        parser = _setup_parser()

        hostfile = tmp_path / "hostfile.txt"
        hostfile.touch()
        output_path = tmp_path / "output.json"

        args = parser.parse_args(
            [
                "--path",
                str(output_path),
                "--job-user",
                "user",
                "--job-id",
                "123",
                "--job-name",
                "job",
                "--hostfile",
                str(hostfile),
                "--port",
                "8080",
            ]
        )

        assert isinstance(args.path, pathlib.Path)
        assert isinstance(args.hostfile, pathlib.Path)

    def test_parser_port_is_integer(self, tmp_path: pathlib.Path) -> None:
        """Test that parser converts port to integer."""
        parser = _setup_parser()

        hostfile = tmp_path / "hostfile.txt"
        hostfile.touch()
        output_path = tmp_path / "output.json"

        args = parser.parse_args(
            [
                "--path",
                str(output_path),
                "--job-user",
                "user",
                "--job-id",
                "123",
                "--job-name",
                "job",
                "--hostfile",
                str(hostfile),
                "--port",
                "9090",
            ]
        )

        assert isinstance(args.port, int)
        assert args.port == 9090

    def test_parser_rejects_invalid_port(self, tmp_path: pathlib.Path) -> None:
        """Test that parser rejects non-integer port."""
        parser = _setup_parser()

        hostfile = tmp_path / "hostfile.txt"
        hostfile.touch()
        output_path = tmp_path / "output.json"

        with pytest.raises(SystemExit):
            parser.parse_args(
                [
                    "--path",
                    str(output_path),
                    "--job-user",
                    "user",
                    "--job-id",
                    "123",
                    "--job-name",
                    "job",
                    "--hostfile",
                    str(hostfile),
                    "--port",
                    "not_a_number",
                ]
            )
