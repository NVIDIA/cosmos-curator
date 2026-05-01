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

"""View and display clip data."""

import contextlib
import json
import shutil
import tempfile
import zipfile
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Annotated, ClassVar, override

from loguru import logger
from typer import Option

from cosmos_curator.client.utils.validations import validate_address, validate_in


class HttpServerHandler(SimpleHTTPRequestHandler):
    """Custom HTTP request handler for serving clips and metadata.

    This handler serves files from two separate directories:
    - web_dir: Contains the web UI files (index.html, etc.)
    - base_dir: Contains the actual clip data (clips/ and metas/v0/)
    """

    web_dir: ClassVar[str] = ""
    base_dir: ClassVar[str] = ""

    @classmethod
    def initialize(cls, web_dir: str, base_dir: str) -> type["HttpServerHandler"]:
        """Initialize the class variables for the handler.

        Args:
            web_dir: Directory containing web files (index.html, etc.)
            base_dir: Directory containing clips and metas

        Returns:
            The initialized handler class

        """
        cls.web_dir = web_dir
        cls.base_dir = base_dir
        return cls

    def end_headers(self) -> None:
        """Add CORS and cache control headers before ending the headers."""
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET")
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        super().end_headers()

    @staticmethod
    def get_clip_time(clip_name: str, base_dir: str) -> float:
        """Get start time for sorting clips.

        Args:
            clip_name: Name of the clip
            base_dir: Directory containing metas

        Returns:
            Start time as a float, or inf if not found

        """
        meta_path = Path(base_dir) / "metas/v0" / f"{clip_name}.json"
        try:
            if meta_path.exists():
                with Path(meta_path).open() as f:
                    metadata = json.load(f)
                    return float(metadata.get("duration_span", [float("inf"), float("inf")])[0])
            logger.warning(f"No metadata file found for {clip_name}")
        except (FileNotFoundError, PermissionError) as e:
            logger.exception(f"Cannot open metadata file '{meta_path}': {e}")
        except json.JSONDecodeError as e:
            logger.exception(f"Invalid JSON in metadata file '{meta_path}': {e}")
        except (ValueError, TypeError) as e:
            logger.exception(f"Invalid 'duration_span' format in metadata file '{meta_path}': {e}")
        return float("inf")

    @override
    def translate_path(self, path: str) -> str:
        """Override translate_path to handle both web files and clips/metas directories.

        Args:
            path: The requested path

        Returns:
            The absolute path to serve

        """
        # If path starts with /, strip it
        if path[0] == "/":
            path = path[1:]
        # If path starts with /clips or /metas, serve from base_dir
        # Otherwise serve from web_dir (where index.html is)
        abs_path = Path(self.base_dir) / path if path.startswith(("clips", "metas")) else Path(self.web_dir) / path
        logger.info(f"Serving {abs_path}")
        return str(abs_path)

    def do_GET(self) -> None:
        """Handle GET requests for clips, metadata, and web files."""
        try:
            if self.path == "/list_clips":
                # Construct the clips_dir from base_dir
                clips_dir = Path(self.base_dir) / "clips"

                logger.info("Processing request for /list_clips")
                logger.debug(f"Looking for clips in: {clips_dir}")

                # Check if directory exists
                if not clips_dir.exists():
                    logger.error(f"Clips directory '{clips_dir}' does not exist")
                    clip_names = []
                else:
                    # Get clip names without extension
                    clip_names = [f.stem for f in Path(clips_dir).rglob("*") if f.suffix.lower() == ".mp4"]
                    logger.info(f"Found {len(clip_names)} clips in {clips_dir}")

                    # Sort clips by their start time
                    clip_names.sort(key=lambda x: self.get_clip_time(x, self.base_dir))
                    logger.info("Sorted clips by start time")

                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.end_headers()

                self.wfile.write(json.dumps(clip_names).encode("utf-8"))
            else:
                try:
                    super().do_GET()
                except ConnectionResetError:
                    logger.debug("Client disconnected during file transfer")
                except BrokenPipeError:
                    logger.debug("Broken pipe during file transfer")
                except ConnectionAbortedError:
                    logger.debug("Connection aborted during file transfer")
        except (OSError, json.JSONDecodeError, ValueError) as e:
            logger.exception(f"Error handling GET request: {e}")
            # If we can't even send the error, just ignore it
            with contextlib.suppress(ConnectionResetError, BrokenPipeError, ConnectionAbortedError):
                self.send_error(500, f"Internal server error: {e}")


class ClipViewer:
    """Viewer for clip data.

    Provides methods to serve and display video clips and metadata using a local HTTP server.
    """

    def __init__(self, clip_path: str, *, cleanup_on_exit: bool = False) -> None:
        """Initialize the ClipViewer.

        Args:
            clip_path: Path to directory containing clips and metadata.
            cleanup_on_exit: If True, clean up the temporary clip directory on exit.

        """
        self.clip_path: str = clip_path
        self.cleanup_on_exit: bool = cleanup_on_exit

    def __del__(self) -> None:
        """Clean up temporary clip directory on deletion."""
        logger.info("Tearing down ClipViewer ...")
        if self.cleanup_on_exit and Path(self.clip_path).exists():
            try:
                shutil.rmtree(self.clip_path)
            except OSError:
                logger.exception(f"Failed to clean up clip directory: {self.clip_path}")

    def hdlr_factory(
        self,
        web_dir: str,
    ) -> type[SimpleHTTPRequestHandler]:
        """Create a custom HTTP request handler for serving clips and metadata.

        Args:
            web_dir: Directory containing web files (index.html, etc.)

        Returns:
            A subclass of SimpleHTTPRequestHandler

        """
        return HttpServerHandler.initialize(web_dir, self.clip_path)

    def serve(self, web_dir: str, ip: str, port: int) -> None:
        """Start the HTTP server to serve clips and metadata.

        Args:
            web_dir: Directory containing web files
            ip: IP address to bind the server
            port: Port to run the server on

        """
        try:
            self.hdlr = self.hdlr_factory(web_dir)
            self.server = HTTPServer((ip, port), self.hdlr)

            logger.info(f"Base directory: {self.clip_path}")
            logger.info(f"Clips directory: {(Path(self.clip_path) / 'clips')}")
            logger.info(f"Metadata directory: {(Path(self.clip_path) / 'metas/v0')}")

            logger.info(f"Starting server on http://{ip}:{port}")
            logger.info("Please connect your browser to the above URL to view the clips and captions")

            self.server.serve_forever()
        except KeyboardInterrupt:
            logger.info("Shutting down server ...")
            self.server.server_close()
        except Exception as e:  # noqa: BLE001
            logger.exception(f"Server error: {e}")


def clip_viewer(
    ip: Annotated[
        str,
        Option(
            help="IP to which the server binds to",
            rich_help_panel="View",
            callback=validate_address,
        ),
    ] = "localhost",
    port: Annotated[
        int,
        Option(
            help="Port to run the server on",
            rich_help_panel="View",
            callback=validate_in(range(65536)),
        ),
    ] = 8080,
    clip_path: Annotated[
        Path | None,
        Option(
            help="Path to directory containing clips/ and metas/v0/ subdirectories",
            rich_help_panel="View",
            exists=True,
            file_okay=False,
            dir_okay=True,
            writable=False,
            readable=True,
            resolve_path=True,
        ),
    ] = None,
    zip_file: Annotated[
        Path | None,
        Option(
            help="Downloaded Zip file name",
            rich_help_panel="View",
            exists=True,
            file_okay=True,
            dir_okay=False,
            writable=False,
            readable=True,
            resolve_path=True,
        ),
    ] = None,
) -> None:
    """Start a local web server to view video clips and their metadata.

    Args:
        ctx: Typer context containing the NVCF handler.
        ip: IP address to bind the server.
        port: Port number to run the server on.
        clip_path: Path to directory containing clips and metadata.
        zip_file: Path to a downloaded zip file containing clips and metadata.

    """
    logger.info("Starting clip viewer ...")
    if clip_path is None and zip_file is None:
        logger.error("Either --clip-path or --zip-file must be provided")
        return

    if clip_path is not None and zip_file is not None:
        logger.error("Only one of --clip-path or --zip-file should be provided")
        return

    logger.info(f"Clip path: {clip_path}")
    if zip_file is not None:
        # Create a temporary directory to extract the zip file
        clip_path_temp = tempfile.mkdtemp()
        try:
            with zipfile.ZipFile(zip_file, "r") as zip_ref:
                zip_ref.extractall(clip_path_temp)
            logger.info(f"Extracted zip file to {clip_path_temp}")
        except (FileNotFoundError, zipfile.BadZipFile, PermissionError, OSError) as e:
            logger.exception(f"Failed to extract zip file: {e}")
            return
        clip_path = Path(clip_path_temp)

    # Get the web directory from the package
    web_dir = Path(__file__).parent / "clip_viewer_webdocs"
    if not web_dir.exists():
        logger.error(f"Web directory not found: {web_dir}")
        return

    logger.info(f"Web directory: {web_dir}")
    viewer = ClipViewer(clip_path=str(clip_path), cleanup_on_exit=(zip_file is not None))
    logger.info(f"Starting server on http://{ip}:{port}")
    viewer.serve(web_dir=str(web_dir), ip=ip, port=port)


if __name__ == "__main__":
    clip_viewer()
