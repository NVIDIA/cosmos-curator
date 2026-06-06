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
"""Tests for Dockerfile generation utilities."""

import re
import shutil
import subprocess
from itertools import pairwise
from pathlib import Path

import pytest

from cosmos_curator.client.utils import docker_utils

REPO_ROOT = Path(__file__).resolve().parents[4]
DOCKERFILE_TEMPLATE_PATH = REPO_ROOT / "package" / "cosmos_curator" / "default.dockerfile.jinja2"


def _render_dockerfile_path(
    tmp_path: Path,
    *,
    slim: bool,
    redistributable: bool,
    conda_env_names: list[str] | None = None,
) -> Path:
    return docker_utils.generate_dockerfile(
        dockerfile_template_path=DOCKERFILE_TEMPLATE_PATH,
        conda_env_names=["default"] if conda_env_names is None else conda_env_names,
        dockerfile_output_path=tmp_path / f"Dockerfile-slim-{slim}-redistributable-{redistributable}",
        slim=slim,
        redistributable=redistributable,
    )


def _render_dockerfile(
    tmp_path: Path,
    *,
    slim: bool,
    redistributable: bool,
    conda_env_names: list[str] | None = None,
) -> str:
    return _render_dockerfile_path(
        tmp_path,
        slim=slim,
        redistributable=redistributable,
        conda_env_names=conda_env_names,
    ).read_text()


def _write_buildx_parse_check_dockerfile(source_path: Path, output_path: Path) -> None:
    """Write a BuildKit-checkable Dockerfile that avoids external image pulls."""
    lines = source_path.read_text().splitlines()
    if lines and lines[0].startswith("# syntax="):
        lines = lines[1:]
    contents = "\n".join(lines) + "\n"
    contents = re.sub(r"^FROM\s+\S+\s+AS\s+", "FROM scratch AS ", contents, flags=re.MULTILINE)
    output_path.write_text(contents)


_MIN_BUILDX_CHECK_VERSION = (0, 12)


def _buildx_supports_check(buildx_command: list[str]) -> bool:
    """Return True if buildx runs and is new enough for `build --check` (v0.12+)."""
    result = subprocess.run(  # noqa: S603
        [*buildx_command, "version"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if result.returncode != 0:
        return False
    match = re.search(r"v(\d+)\.(\d+)", result.stdout)
    return match is not None and (int(match.group(1)), int(match.group(2))) >= _MIN_BUILDX_CHECK_VERSION


def _resolve_buildx_command() -> list[str] | None:
    """Return a buildx invocation that supports `build --check`, or None.

    buildx can be a Docker CLI plugin (invoked as `docker buildx`) or a standalone
    binary (e.g. Homebrew's `docker-buildx` on macOS, which is not necessarily
    registered as a `docker buildx` plugin). `--check` requires buildx v0.12+.
    """
    docker_path = shutil.which("docker")
    if docker_path is not None and _buildx_supports_check([docker_path, "buildx"]):
        return [docker_path, "buildx"]

    standalone = shutil.which("docker-buildx")
    if standalone is not None and _buildx_supports_check([standalone]):
        return [standalone]

    return None


def _empty_continuation_lines(contents: str) -> list[int]:
    lines = contents.splitlines()
    return [
        line_number
        for line_number, (previous_line, line) in enumerate(pairwise(lines), start=2)
        if previous_line.rstrip().endswith("\\") and not line.strip()
    ]


def _run_blocks(contents: str) -> list[str]:
    blocks: list[str] = []
    current_block: list[str] = []
    in_run = False

    for line in contents.splitlines():
        if line.startswith("RUN "):
            if current_block:
                blocks.append("\n".join(current_block))
            current_block = [line]
            in_run = line.rstrip().endswith("\\")
            if not in_run:
                blocks.append("\n".join(current_block))
                current_block = []
            continue

        if in_run:
            current_block.append(line)
            in_run = line.rstrip().endswith("\\")
            if not in_run:
                blocks.append("\n".join(current_block))
                current_block = []

    if current_block:
        blocks.append("\n".join(current_block))

    return blocks


@pytest.mark.parametrize("slim", [False, True])
@pytest.mark.parametrize("redistributable", [False, True])
def test_generated_dockerfile_parses_with_buildx_check(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    slim: bool,
    redistributable: bool,
) -> None:
    """Rendered Dockerfiles should parse with the BuildKit frontend used by image builds."""
    monkeypatch.chdir(REPO_ROOT)

    dockerfile_path = _render_dockerfile_path(tmp_path, slim=slim, redistributable=redistributable)
    check_path = tmp_path / f"{dockerfile_path.name}.parse-check"
    _write_buildx_parse_check_dockerfile(dockerfile_path, check_path)

    buildx_command = _resolve_buildx_command()
    if buildx_command is None:
        pytest.skip("buildx with --check support (v0.12+) is not available")

    result = subprocess.run(  # noqa: S603
        [*buildx_command, "build", "--check", "-f", str(check_path), "."],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    # Some buildx versions return nonzero when checks emit warnings. Syntax errors
    # do not reach "Check complete", so this still catches malformed Dockerfiles.
    assert result.returncode == 0 or "Check complete" in result.stdout, result.stdout


@pytest.mark.parametrize("slim", [False, True])
@pytest.mark.parametrize("redistributable", [False, True])
def test_generated_dockerfile_has_no_empty_continuation_lines(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    slim: bool,
    redistributable: bool,
) -> None:
    """Dockerfile continuations must not include blank rendered template lines."""
    monkeypatch.chdir(REPO_ROOT)

    contents = _render_dockerfile(tmp_path, slim=slim, redistributable=redistributable)
    pkg_config_arg = contents.find("ARG PKG_CONFIG_PATH")
    pkg_config_env = contents.find('PKG_CONFIG_PATH="/opt/ffmpeg/lib/pkgconfig:${PKG_CONFIG_PATH:-}"')

    assert _empty_continuation_lines(contents) == []
    assert pkg_config_arg != -1
    assert pkg_config_env != -1
    assert pkg_config_arg < pkg_config_env


def test_full_dockerfile_rebuilds_opencv_against_local_ffmpeg(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Full images should replace PyPI OpenCV wheels with a local FFmpeg-linked build."""
    monkeypatch.chdir(REPO_ROOT)

    contents = _render_dockerfile(tmp_path, slim=False, redistributable=True)

    assert "AS opencv-builder" in contents
    assert "COPY --from=ffmpeg-builder /opt/ffmpeg /opt/ffmpeg" in contents
    assert "libvpx9" in contents
    assert "libavif16" in contents
    assert "libdrm2" in contents
    assert "libgfortran5" in contents
    assert "libwebp7" in contents
    assert "libwebpdemux2" in contents
    assert "libjpeg-turbo8" in contents
    assert "libopenblas0-pthread" in contents
    assert "libpcre2-16-0" in contents
    assert "libpng16-16t64" in contents
    assert '$(if [ "$(dpkg --print-architecture)" = "amd64" ]; then echo libquadmath0; fi)' in contents
    assert "--wheel-dir /opencv-wheelhouse /opencv-python-src" in contents
    assert 'PKG_CONFIG_PATH="/opt/ffmpeg/lib/pkgconfig"' in contents
    assert 'LD_LIBRARY_PATH="/opt/ffmpeg/lib:/usr/local/nvidia/lib' in contents
    assert 'LIBRARY_PATH="/opt/ffmpeg/lib:/usr/local/cuda/lib64' in contents
    assert "WITH_FFMPEG=ON" in contents
    assert "WITH_GTK=OFF" in contents
    assert "WITH_QT=OFF" in contents
    assert "WITH_TIFF=OFF" in contents
    assert "CMAKE_PREFIX_PATH=/opt/ffmpeg" in contents
    assert "CMAKE_BUILD_RPATH=/opt/ffmpeg/lib" in contents
    assert "pip install --no-cache-dir --no-deps /opencv-wheelhouse/opencv_python_headless-*.whl" in contents
    assert 'raise SystemExit(0 if "FFMPEG:                      YES" in info else 1)' in contents
    assert "rm -rf /pyav-build/.pixi /root/.cache/pip /tmp/pip-*" in contents
    assert "rm -rf /opencv-build/.pixi /opencv-python-src /root/.cache/pip /tmp/pip-*" in contents
    assert "COPY --from=opencv-builder /opencv-wheelhouse /opt/cosmos-curator-wheelhouse" in contents
    assert "pip uninstall -y opencv-python-headless opencv-python opencv-contrib-python" in contents
    assert (
        "pip install --no-cache-dir --no-deps /opt/cosmos-curator-wheelhouse/opencv_python_headless-*.whl" in contents
    )


def test_full_dockerfile_with_paddle_ocr_rebuilds_all_opencv_variants(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PaddleOCR images need the non-headless OpenCV wheels rebuilt too."""
    monkeypatch.chdir(REPO_ROOT)

    contents = _render_dockerfile(
        tmp_path,
        slim=False,
        redistributable=True,
        conda_env_names=["default", "paddle-ocr"],
    )

    assert "/opt/cosmos-curator-wheelhouse/opencv_python-*.whl" in contents
    assert "opencv_python-" in contents
    assert "opencv_contrib_python-" in contents
    assert "WITH_GTK=OFF" in contents
    assert "WITH_QT=OFF" in contents
    assert "WITH_TIFF=OFF" in contents


def test_full_dockerfile_default_rebuilds_all_opencv_variants(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default now includes PaddleOCR and needs the non-headless OpenCV wheels too."""
    monkeypatch.chdir(REPO_ROOT)

    contents = _render_dockerfile(
        tmp_path,
        slim=False,
        redistributable=True,
        conda_env_names=["default"],
    )

    assert "/opt/cosmos-curator-wheelhouse/opencv_python-*.whl" in contents
    assert "opencv_python-" in contents
    assert "opencv_contrib_python-" in contents
    assert "WITH_GTK=OFF" in contents
    assert "WITH_QT=OFF" in contents
    assert "WITH_TIFF=OFF" in contents


def test_full_dockerfile_replaces_bundled_video_wheels_in_pixi_install_layer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bundled PyPI video wheels must not survive in lower final-image layers."""
    monkeypatch.chdir(REPO_ROOT)

    contents = _render_dockerfile(tmp_path, slim=False, redistributable=True)
    install_blocks = [block for block in _run_blocks(contents) if "=== pixi install attempt $attempt/10 ===" in block]

    assert len(install_blocks) == 1
    install_block = install_blocks[0]
    assert "pip uninstall -y av" in install_block
    assert "pip install --no-cache-dir /opt/cosmos-curator-wheelhouse/av-17.0.0-*.whl" in install_block
    assert "pip uninstall -y opencv-python-headless opencv-python opencv-contrib-python" in install_block
    assert 'if [ "$env" = "paddle-ocr" ] || [ "$env" = "default" ]; then' in install_block
    assert "pip install --no-cache-dir --no-deps /opt/cosmos-curator-wheelhouse/opencv_python-*.whl" in install_block
    assert (
        "pip install --no-cache-dir --no-deps /opt/cosmos-curator-wheelhouse/opencv_python_headless-*.whl"
        in install_block
    )


def test_full_dockerfile_rewrites_pixi_lock_to_local_video_wheels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Full images should keep a lockfile that points at source-built video wheels."""
    monkeypatch.chdir(REPO_ROOT)

    full_contents = _render_dockerfile(tmp_path, slim=False, redistributable=True)
    slim_contents = _render_dockerfile(tmp_path, slim=True, redistributable=True)

    assert "COPY --chown=1000:1000 pixi.toml pixi.lock" not in full_contents
    assert "COPY --chown=1000:1000 --from=opencv-builder /opencv-build/pixi.lock" in full_contents
    assert "file:///opt/cosmos-curator-wheelhouse/" in full_contents
    assert "ERROR: full image runtime pixi.lock still references bundled PyPI wheels (av/opencv)" in full_contents
    assert "COPY --chown=1000:1000 pixi.toml pixi.lock" in slim_contents


def _capture_build_command(monkeypatch: pytest.MonkeyPatch, **build_kwargs: object) -> list[str]:
    """Run docker_utils.build with subprocess stubbed and return the buildx command."""
    captured: dict[str, list[str]] = {}

    def fake_check_call(cmd: list[str], *_args: object, **_kwargs: object) -> int:
        captured["cmd"] = cmd
        return 0

    monkeypatch.setattr(docker_utils.subprocess, "check_call", fake_check_call)
    docker_utils.build(
        curator_path=Path.cwd(),
        dockerfile_path=Path("Dockerfile"),
        image="reg/img:tag",
        **build_kwargs,  # type: ignore[arg-type]
    )
    return captured["cmd"]


def test_build_push_with_zstd_uses_explicit_output_exporter(monkeypatch: pytest.MonkeyPatch) -> None:
    """Compression + push should replace --push with an --output image exporter."""
    cmd = _capture_build_command(monkeypatch, push=True, load=False, compression="zstd")

    assert "--push" not in cmd
    output_value = cmd[cmd.index("--output") + 1]
    # Assert the meaningful attributes are present without pinning their order,
    # since BuildKit treats the comma-separated --output keys as unordered.
    attributes = set(output_value.split(","))
    assert {
        "type=image",
        "push=true",
        "compression=zstd",
        "compression-level=3",
        "force-compression=false",
        "oci-mediatypes=true",
    } <= attributes


def test_build_push_without_compression_uses_push_shorthand(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without compression, the fast --push shorthand is preserved."""
    cmd = _capture_build_command(monkeypatch, push=True, load=False)

    assert "--push" in cmd
    assert "--output" not in cmd


def test_build_compression_does_not_affect_local_load(monkeypatch: pytest.MonkeyPatch) -> None:
    """Compression only applies to push; load-only builds keep --load and no exporter."""
    cmd = _capture_build_command(monkeypatch, push=False, load=True, compression="zstd")

    assert "--load" in cmd
    assert "--output" not in cmd


def test_full_dockerfile_with_cuml_keeps_local_video_wheel_lockfile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The cuml install layer should not revert or delete the full-image lockfile."""
    monkeypatch.chdir(REPO_ROOT)

    contents = _render_dockerfile(
        tmp_path,
        slim=False,
        redistributable=True,
        conda_env_names=["default", "cuml"],
    )
    cuml_blocks = [block for block in _run_blocks(contents) if "=== pixi install cuml attempt $attempt/10 ===" in block]

    assert len(cuml_blocks) == 1
    cuml_block = cuml_blocks[0]
    assert "rm -f pixi.lock" not in cuml_block
    assert "source=pixi.lock,target=/tmp/cosmos-curator-pixi.lock,readonly" not in cuml_block
    assert "cp /tmp/cosmos-curator-pixi.lock pixi.lock" not in cuml_block
