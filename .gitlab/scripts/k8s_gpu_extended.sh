#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Extended GPU end-to-end coverage. Runs a single pre-canned scenario config
# (selected via the SCENARIO matrix variable) so the k8s_gpu_extended job can
# fan each scenario out to its own pod and keep wall-clock close to the slowest
# single scenario instead of the sum of all of them.

set -e

# shellcheck source=common.sh
source "$(dirname "$0")/common.sh"

: "${SCENARIO:?SCENARIO must be set to a scenario name under examples/ci/ (without .json)}"

# Resolve the repo root from this script's location so the pre-canned config
# JSONs always match the commit under test (not whatever is baked into the image).
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CONFIG_PATH="${REPO_ROOT}/examples/ci/${SCENARIO}.json"

echo "=== K8s GPU Extended Pipeline Test: ${SCENARIO} ==="

cd /opt/cosmos-curator

# Report GPU availability when running on GPU-tagged pods; split_openai runs on CPU.
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
fi

# Set up model cache (persisted via hostPath at /cache)
export NVCF_MODEL_CACHE_DIR="/cache/cosmos-models"
mkdir -p "${NVCF_MODEL_CACHE_DIR}"

# Configure S3 and NGC
setup_s3_credentials
setup_ngc_model_download
if [ "${SCENARIO}" = "split_openai" ]; then
    if [ ! -s "${CI_PROJECT_DIR}/cosmos_curator_config.yaml" ]; then
        echo "ERROR: OpenAI config was not rendered from Vault" >&2
        exit 1
    fi
    mkdir -p /cosmos_curator/config
    cp "${CI_PROJECT_DIR}/cosmos_curator_config.yaml" /cosmos_curator/config/cosmos_curator.yaml
    chmod 600 /cosmos_curator/config/cosmos_curator.yaml
    rm -f "${CI_PROJECT_DIR}/cosmos_curator_config.yaml"
fi

# The SeedVR2 super-resolution scenario needs both the `seedvr` pixi env and the
# SeedVR source repo (SEEDVR_ROOT). The slim image bakes neither -- the Dockerfile
# only provisions them for non-slim builds (default.dockerfile.jinja2 wraps the
# SeedVR COPY + ENV in `{% if not slim %}`). Materialise both at runtime so the SR
# stage can import + run, and fast-fail on the torchvision.io video I/O import
# regression before the (expensive) SeedVR2 diffusion pass.
if [ "${SCENARIO}" = "split_super_resolution" ]; then
    echo "=== Provisioning seedvr env + SeedVR source for ${SCENARIO} ==="
    pixi install --frozen -e seedvr
    WHEEL_DIR="${CI_PROJECT_DIR}/cosmos-xenna/target/wheels" \
        bash "${CI_PROJECT_DIR}/.gitlab/scripts/install_local_xenna_into_pixi.sh" seedvr

    # Run the seedvr post-install (flash-attn precompiled wheel). The Dockerfile does
    # this for non-slim builds; the slim CI image solves the env at runtime and would
    # otherwise miss flash_attn, which SeedVR's DiT imports at model-build time.
    pixi run --as-is -e seedvr bash \
        "${REPO_ROOT}/package/cosmos_curator/envs/seedvr/post_install.sh"

    # Mirror the Dockerfile's SeedVR provisioning: fetch the pinned commit, drop in
    # our color_fix.py, and export SEEDVR_ROOT. Download via the pixi Python
    # interpreter rather than curl, which the slim image intentionally omits.
    SEEDVR_COMMIT="e4de8c2"
    export SEEDVR_ROOT="${CI_PROJECT_DIR}/SeedVR"
    if [ ! -d "${SEEDVR_ROOT}" ]; then
        SEEDVR_COMMIT="${SEEDVR_COMMIT}" SEEDVR_ROOT="${SEEDVR_ROOT}" pixi run --as-is python - <<'PY'
import io
import os
import tarfile
import urllib.request

commit = os.environ["SEEDVR_COMMIT"]
dest = os.environ["SEEDVR_ROOT"]
url = f"https://github.com/ByteDance-Seed/SeedVR/archive/{commit}.tar.gz"
parent = os.path.dirname(dest) or "."
with urllib.request.urlopen(url, timeout=300) as resp:  # noqa: S310
    data = resp.read()
with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
    tar.extractall(parent)  # noqa: S202
extracted = next(
    os.path.join(parent, name)
    for name in os.listdir(parent)
    if name.startswith("SeedVR-")
)
os.rename(extracted, dest)
print(f"SeedVR source extracted to {dest}")
PY
    fi
    cp "${REPO_ROOT}/cosmos_curator/pipelines/video/super_resolution/color_fix.py" \
        "${SEEDVR_ROOT}/projects/video_diffusion_sr/color_fix.py"

    echo "=== Import regression guard (seedvr env) for ${SCENARIO} ==="
    # Invoke pytest as a module: bare `pytest` is a pixi *task* defined only in the
    # `dev` env, so `pixi run -e seedvr pytest` fails with "task not available".
    PIXI_ENVIRONMENT_NAME=seedvr pixi run --as-is -e seedvr \
        python -m pytest -m env "${REPO_ROOT}/tests/cosmos_curator/pipelines/video/super_resolution/"
fi

# Retried and concurrent matrix jobs have distinct job IDs and cannot reuse stale artifacts.
: "${CI_JOB_ID:?CI_JOB_ID must be set}"
K8S_OUTPUT_PATH="${S3_OUTPUT_PATH}/jobs/${CI_JOB_ID}/k8s-gpu-extended/${SCENARIO}"

run_pipeline_from_config "${CONFIG_PATH}" "${K8S_OUTPUT_PATH}"
pixi run --as-is python -m cosmos_curator.pipelines.video.output_validation \
    "${CONFIG_PATH}" "${K8S_OUTPUT_PATH}"

if [ "${SCENARIO}" = "split_openai" ]; then
    rm -f /cosmos_curator/config/cosmos_curator.yaml
fi

echo "✓ K8s GPU extended pipeline test (${SCENARIO}) completed successfully"
