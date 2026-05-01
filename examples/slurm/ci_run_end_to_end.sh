#!/usr/bin/env bash
set -euo pipefail

log() {
  printf '[%s] %s\n' "$(date -Ins)" "$*"
}

required_vars=(
  S3_INPUT_VIDEO_PATH
  SLURM_E2E_OUTPUT_CLIP_PATH
  SLURM_E2E_OUTPUT_DEDUP_PATH
  SLURM_E2E_OUTPUT_DATASET_PATH
)
for var in "${required_vars[@]}"; do
  if [[ -z "${!var:-}" ]]; then
    echo "Missing required environment variable: ${var}" >&2
    exit 1
  fi
done

SLURM_E2E_S3_PROFILE_NAME=${SLURM_E2E_S3_PROFILE_NAME:-default}

export AWS_SHARED_CREDENTIALS_FILE="${AWS_SHARED_CREDENTIALS_FILE:-/creds/s3_creds}"
export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-west-2}"
export MODEL_WEIGHTS_PREFIX="${MODEL_WEIGHTS_PREFIX:-/config/models}"
export PYTHONUNBUFFERED=1

run_split() {
  log "Running split pipeline -> ${SLURM_E2E_OUTPUT_CLIP_PATH}"
  python -m cosmos_curator.pipelines.video.run_pipeline split \
    --input-video-path "${S3_INPUT_VIDEO_PATH}" \
    --output-clip-path "${SLURM_E2E_OUTPUT_CLIP_PATH}" \
    --input-s3-profile-name "${SLURM_E2E_S3_PROFILE_NAME}" \
    --output-s3-profile-name "${SLURM_E2E_S3_PROFILE_NAME}" \
    --limit 1 \
    --execution-mode STREAMING
  log "Split pipeline completed"
}

run_dedup() {
  log "Running dedup pipeline -> ${SLURM_E2E_OUTPUT_DEDUP_PATH}"
  python -m cosmos_curator.pipelines.video.run_pipeline dedup \
    --input-embeddings-path "${SLURM_E2E_OUTPUT_CLIP_PATH}" \
    --output-path "${SLURM_E2E_OUTPUT_DEDUP_PATH}" \
    --input-s3-profile-name "${SLURM_E2E_S3_PROFILE_NAME}" \
    --output-s3-profile-name "${SLURM_E2E_S3_PROFILE_NAME}" \
    --eps-to-extract 0.01
  log "Dedup pipeline completed"
}

run_shard() {
  log "Running shard pipeline -> ${SLURM_E2E_OUTPUT_DATASET_PATH}"
  python -m cosmos_curator.pipelines.video.run_pipeline shard \
    --input-clip-path "${SLURM_E2E_OUTPUT_CLIP_PATH}" \
    --output-dataset-path "${SLURM_E2E_OUTPUT_DATASET_PATH}" \
    --input-semantic-dedup-path "${SLURM_E2E_OUTPUT_DEDUP_PATH}" \
    --input-semantic-dedup-s3-profile-name "${SLURM_E2E_S3_PROFILE_NAME}" \
    --input-s3-profile-name "${SLURM_E2E_S3_PROFILE_NAME}" \
    --output-s3-profile-name "${SLURM_E2E_S3_PROFILE_NAME}" \
    --annotation-version v0 \
    --semantic-dedup-epsilon 0.01
  log "Shard pipeline completed"
}

run_split
run_dedup
run_shard

log "SLURM end-to-end pipeline finished successfully"
