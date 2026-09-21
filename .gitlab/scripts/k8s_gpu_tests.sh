#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -e

# shellcheck source=common.sh
source "$(dirname "$0")/common.sh"

echo "=== K8s GPU Pipeline Test ==="
echo "Running mini split pipeline with GPU model to verify full stack"

# Resolve the repo root from this script's location so the pre-canned config
# JSONs always match the commit under test (not whatever is baked into the image).
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

cd /opt/cosmos-curator

# Verify GPU is available
nvidia-smi

# Set up model cache (persisted via hostPath at /cache)
export NVCF_MODEL_CACHE_DIR="/cache/cosmos-models"
mkdir -p "${NVCF_MODEL_CACHE_DIR}"

# Configure S3 and NGC
setup_s3_credentials
setup_ngc_model_download

# A retried GitLab job gets a new job ID, so this prefix cannot reuse stale artifacts.
: "${CI_JOB_ID:?CI_JOB_ID must be set}"
K8S_OUTPUT_PATH="${S3_OUTPUT_PATH}/jobs/${CI_JOB_ID}/k8s-gpu-test"

# Run split pipeline with GPU stages (transnetv2 + embeddings + captions)
# via the config-driven run_pipeline entry point.
run_pipeline_from_config "${REPO_ROOT}/examples/ci/split_basic.json" "${K8S_OUTPUT_PATH}"
pixi run --as-is python -m cosmos_curator.pipelines.video.output_validation \
    "${REPO_ROOT}/examples/ci/split_basic.json" "${K8S_OUTPUT_PATH}"

echo "✓ K8s GPU pipeline test completed successfully"
