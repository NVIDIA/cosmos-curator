#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Build a cosmos-xenna wheel from the submodule checkout so that the
# subsequent `cosmos-curator image build --use-local-xenna-build` step has a
# matching manylinux wheel in cosmos-xenna/target/wheels/.
#
# The ref to build is controlled by the XENNA_REF env var (default: main).
# The wheel is built natively for the host arch, so run this on a runner
# whose arch matches the target image (amd64 wheel on amd64 runner,
# aarch64 wheel on arm64 runner).

set -euo pipefail

XENNA_REF="${XENNA_REF:-main}"
XENNA_DIR="${XENNA_DIR:-cosmos-xenna}"

# XENNA_REF Guardrail: must be a non-empty git ref without a leading '-',
if [[ -z "${XENNA_REF}" || "${XENNA_REF}" == -* || "${XENNA_REF}" =~ [[:space:][:cntrl:]] ]]; then
    echo "ERROR: XENNA_REF must be a non-empty git ref without a leading '-'" \
         "or whitespace/control chars (got: '${XENNA_REF}')"
    exit 1
fi

if [ ! -d "${XENNA_DIR}" ]; then
    echo "ERROR: cosmos-xenna checkout not found at '${XENNA_DIR}'. Did the " \
         "submodule get initialised?"
    exit 1
fi

echo "=== Fetching cosmos-xenna ref: ${XENNA_REF} ==="
pushd "${XENNA_DIR}" > /dev/null
# The submodule is cloned with depth 1 in CI, so we need to fetch the
# requested ref explicitly before we can check it out.
git fetch --depth 1 origin "${XENNA_REF}"
git checkout --detach FETCH_HEAD
echo "cosmos-xenna now at: $(git --no-pager log -1 --oneline)"

# Install uv and rust into user-local paths if missing. Both installers are
# pinned to specific upstream versions and verified against a SHA256 checksum
# so a compromised CDN cannot inject arbitrary code. The pinned URLs are
# immutable, so the SHAs only change when we deliberately bump the versions
# below (rust ships every ~6 weeks; rustup is updated less often).
#
# To refresh:
#   uv:     bump UV_VERSION; sha = sha256sum on
#           https://astral.sh/uv/${UV_VERSION}/install.sh.
#   rustup: bump RUSTUP_VERSION; per-arch SHAs published next to each binary at
#           https://static.rust-lang.org/rustup/archive/${RUSTUP_VERSION}/<triple>/rustup-init.sha256
#   rust:   bump RUST_TOOLCHAIN to any released stable (>= 1.85 for edition=2024).
UV_VERSION="0.11.7"
UV_INSTALLER_SHA256="efed99618cb5c31e4e36a700ab7c3698e83c0ae0f3c336714043d0f932c8d32c"
RUSTUP_VERSION="1.28.2"
RUSTUP_INIT_SHA256_x86_64="20a06e644b0d9bd2fbdbfd52d42540bdde820ea7df86e92e533c073da0cdd43c"
RUSTUP_INIT_SHA256_aarch64="e3853c5a252fca15252d07cb23a1bdd9377a8c6f3efa01531109281ae47f841c"
RUST_TOOLCHAIN="1.95.0"

export PATH="${HOME}/.local/bin:${HOME}/.cargo/bin:${PATH}"

if ! command -v uv >/dev/null 2>&1; then
    echo "=== Installing uv ${UV_VERSION} ==="
    installer="$(mktemp)"
    curl --proto '=https' --tlsv1.2 -fsSL \
        "https://astral.sh/uv/${UV_VERSION}/install.sh" -o "${installer}"
    echo "${UV_INSTALLER_SHA256}  ${installer}" | sha256sum -c -
    sh "${installer}"
    rm -f "${installer}"
fi

if ! command -v cargo >/dev/null 2>&1; then
    echo "=== Installing rustup ${RUSTUP_VERSION} + rust ${RUST_TOOLCHAIN} ==="
    arch="$(uname -m)"
    case "${arch}" in
        x86_64)  triple="x86_64-unknown-linux-gnu";  expected_sha="${RUSTUP_INIT_SHA256_x86_64}" ;;
        aarch64) triple="aarch64-unknown-linux-gnu"; expected_sha="${RUSTUP_INIT_SHA256_aarch64}" ;;
        *) echo "ERROR: unsupported arch '${arch}' for rustup install"; exit 1 ;;
    esac
    # rustup-init dispatches on argv[0] (basename), so the file MUST be named
    # 'rustup-init' or it errors with "unknown proxy name". Stage it in a
    # private temp dir to keep the well-known name without polluting /tmp.
    rustup_dir="$(mktemp -d)"
    rustup_init="${rustup_dir}/rustup-init"
    curl --proto '=https' --tlsv1.2 -fsSL \
        "https://static.rust-lang.org/rustup/archive/${RUSTUP_VERSION}/${triple}/rustup-init" \
        -o "${rustup_init}"
    echo "${expected_sha}  ${rustup_init}" | sha256sum -c -
    chmod +x "${rustup_init}"
    "${rustup_init}" -y --profile minimal --default-toolchain "${RUST_TOOLCHAIN}" --no-modify-path
    rm -rf "${rustup_dir}"
fi

echo "=== Building cosmos-xenna wheel (maturin, native arch) ==="
rm -rf target/wheels
# dev group pulls in maturin + patchelf + ziglang + friends.
uv sync --group dev
# Let maturin auto-pick the manylinux tag based on the host glibc; the
# Dockerfile's wheel glob (`cosmos_xenna-*manylinux*_${ARCH}.whl`) accepts
# any manylinux version, and vm-builder runners have newer glibc than the
# final cuda:13.0.2-ubuntu24.04 runtime image.
uv run maturin build --release --strip

echo "=== Built wheels ==="
ls -la target/wheels/

popd > /dev/null
