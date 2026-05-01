#!/usr/bin/env bash
# Post-install for seedvr environment: flash-attn precompiled wheel.
set -euo pipefail

ARCH="$(uname -m)"
if [[ "${ARCH}" != "x86_64" ]]; then
    echo "Skipping flash-attn install on ${ARCH} (no precompiled wheel available)."
    exit 0
fi

# Precompiled flash-attn wheel for CUDA 13.0 + torch 2.10 + Python 3.12 (x86_64 only).
# From https://github.com/mjun0812/flash-attention-prebuild-wheels
FLASH_ATTN_WHL="https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/flash_attn-2.8.3+cu130torch2.10-cp312-cp312-linux_x86_64.whl"
pip install --no-cache-dir "${FLASH_ATTN_WHL}"
