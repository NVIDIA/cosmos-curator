#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# If a locally-built cosmos-xenna wheel artifact is present (downloaded from
# the build_xenna_wheels matrix job), pip-install the wheel matching the host
# architecture into each named pixi environment, replacing the cosmos-xenna
# that pixi installed from the lockfile. This is the runtime equivalent of
# the image-build path's `--use-local-xenna-build` flag, used by slim-image
# consumers (k8s_gpu_tests, gpu_tests, integration_tests) that materialise pixi envs at runtime
# from pixi.lock (which still pins the PyPI-released cosmos-xenna).
#
# This is a no-op when no matching wheel artifact is present, so it is safe
# to call unconditionally from any pixi-install step.
#
# Usage:
#   install_local_xenna_into_pixi.sh [pixi-env ...]
#
# When no pixi envs are given on the command line, the COSMOS_CURATOR_PIXI_ENVS
# env var is consulted (comma-separated). If neither is set, the script
# defaults to the single env named 'default'.
#
# Must be invoked from the directory containing pixi.toml/pixi.lock (i.e.
# inside the curator source tree where pixi installs envs).

set -euo pipefail

WHEEL_DIR="${WHEEL_DIR:-cosmos-xenna/target/wheels}"
ARCH="${ARCH:-$(uname -m)}"

shopt -s nullglob
wheels=("${WHEEL_DIR}"/cosmos_xenna-*manylinux*_"${ARCH}".whl)
shopt -u nullglob

if [[ ${#wheels[@]} -eq 0 ]]; then
    echo "No locally-built cosmos-xenna ${ARCH} wheel found at '${WHEEL_DIR}'; using the pixi-installed version."
    exit 0
fi

if [[ ${#wheels[@]} -gt 1 ]]; then
    echo "ERROR: multiple cosmos-xenna ${ARCH} wheels found in '${WHEEL_DIR}':"
    printf '  %s\n' "${wheels[@]}"
    echo "Expected exactly one. Aborting."
    exit 1
fi

wheel="${wheels[0]}"

if [[ $# -gt 0 ]]; then
    envs=("$@")
elif [[ -n "${COSMOS_CURATOR_PIXI_ENVS:-}" ]]; then
    IFS=',' read -r -a envs <<< "${COSMOS_CURATOR_PIXI_ENVS}"
else
    envs=("default")
fi

if ! command -v pixi >/dev/null 2>&1; then
    echo "ERROR: 'pixi' not on PATH; this script must run after pixi is installed and from the curator source dir."
    exit 1
fi

for env in "${envs[@]}"; do
    echo "Installing ${wheel} into pixi env '${env}' (override of pixi-installed cosmos-xenna)..."
    pixi run --as-is -e "${env}" pip install --force-reinstall --no-deps "${wheel}"
    pixi run --as-is -e "${env}" python -c '
import sys
import importlib.metadata as md
import cosmos_xenna
try:
    version = md.version("cosmos-xenna")
except md.PackageNotFoundError:
    version = "(unknown)"
label = sys.argv[1]
print(f"[{label}] cosmos-xenna {version} installed from {cosmos_xenna.__file__}")
' "${env}"
done
