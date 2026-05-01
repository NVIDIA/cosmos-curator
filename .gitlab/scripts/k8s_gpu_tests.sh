#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -e

# shellcheck source=common.sh
source "$(dirname "$0")/common.sh"

echo "=== K8s GPU Pipeline Test ==="
echo "Running mini split pipeline with GPU model to verify full stack"

cd /opt/cosmos-curator

# Verify GPU is available
nvidia-smi

# Set up model cache (persisted via hostPath at /cache)
export NVCF_MODEL_CACHE_DIR="/cache/cosmos-models"
mkdir -p "${NVCF_MODEL_CACHE_DIR}"

# Configure S3 and NGC
setup_s3_credentials
setup_ngc_model_download

# Set output path for this test
K8S_OUTPUT_PATH="${S3_OUTPUT_PATH}/k8s-gpu-test"

echo "Input: ${S3_INPUT_VIDEO_PATH}"
echo "Output: ${K8S_OUTPUT_PATH}"

# Run split pipeline with GPU stages (transnetv2 + embeddings + captions)
# shellcheck disable=SC2046
pixi run --as-is python -m cosmos_curator.pipelines.video.run_pipeline split \
  --input-video-path "${S3_INPUT_VIDEO_PATH}" \
  --output-clip-path "${K8S_OUTPUT_PATH}" \
  --limit 1 \
  --splitting-algorithm transnetv2 \
  $(get_reduced_cpu_pipeline_args) \
  --embedding-algorithm cosmos-embed1-224p \
  --captioning-algorithm cosmos_r1

echo "✓ K8s GPU pipeline test completed successfully"
