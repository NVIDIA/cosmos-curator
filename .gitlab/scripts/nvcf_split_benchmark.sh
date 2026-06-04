#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

echo "Running nvcf split benchmark"

filter_empty_items() {
  local -n items=$1
  local filtered=()
  local item
  for item in "${items[@]}"; do
    [[ -n "${item}" ]] && filtered+=("${item}")
  done
  items=("${filtered[@]}")
}

if [[ -n "${NVCF_SPLIT_BENCHMARK_SOURCE_REF:-}" ]]; then
  if [[ "${NVCF_SPLIT_BENCHMARK_SOURCE_REF}" == *"/"* ]]; then
    echo "ERROR: NVCF_SPLIT_BENCHMARK_SOURCE_REF must be an image ref without registry/org, e.g. dev-cosmos-curator:tag" >&2
    exit 1
  fi
  if [[ "${NVCF_SPLIT_BENCHMARK_SOURCE_REF}" != *":"* ||
    "${NVCF_SPLIT_BENCHMARK_SOURCE_REF%%:*}" == "" ||
    "${NVCF_SPLIT_BENCHMARK_SOURCE_REF#*:}" == "" ||
    "${NVCF_SPLIT_BENCHMARK_SOURCE_REF}" == *"@"* ||
    "${NVCF_SPLIT_BENCHMARK_SOURCE_REF}" =~ [[:space:]] ||
    "${NVCF_SPLIT_BENCHMARK_SOURCE_REF#*:}" == *":"* ]]; then
    echo "ERROR: NVCF_SPLIT_BENCHMARK_SOURCE_REF must be exactly image:tag" >&2
    exit 1
  fi
  SOURCE_IMAGE="nvcr.io/${NGC_NVCF_ORG}/${NVCF_SPLIT_BENCHMARK_SOURCE_REF}"
  PERF_IMAGE="nvcr.io/${PERF_NGC_NVCF_ORG_ID}/${NVCF_SPLIT_BENCHMARK_SOURCE_REF}"
else
  if [[ -z "${STAGING_IMAGE:-}" ]]; then
    echo "ERROR: STAGING_IMAGE is unset (needs resolve_nvcf_staging_tag dotenv or NVCF_SPLIT_BENCHMARK_SOURCE_REF)" >&2
    exit 1
  fi
  staging_prefix="nvcr.io/${NGC_NVCF_ORG}/"
  if [[ "${STAGING_IMAGE}" != "${staging_prefix}"* ]]; then
    echo "ERROR: STAGING_IMAGE must start with ${staging_prefix}" >&2
    exit 1
  fi
  SOURCE_IMAGE="${STAGING_IMAGE}"
  staging_ref="${STAGING_IMAGE#"${staging_prefix}"}"
  PERF_IMAGE="nvcr.io/${PERF_NGC_NVCF_ORG_ID}/${staging_ref}"
fi

echo "Skopeo copy ${SOURCE_IMAGE} -> ${PERF_IMAGE}"
skopeo copy --all \
  --src-creds "\$oauthtoken:${NGC_REGISTRY_KEY}" \
  --dest-creds "\$oauthtoken:${PERF_REGISTRY_KEY}" \
  "docker://${SOURCE_IMAGE}" \
  "docker://${PERF_IMAGE}"
echo "Published ${PERF_IMAGE}"

date_str=$(date +%Y%m%d%H%M%S)
LIMIT_INPUT_VIDEOS="${NVCF_SPLIT_BENCHMARK_LIMIT:-5000}"
export RUST_BACKTRACE=1

# Run benchmark
# Defaults: caption=1 (performance-critical path), nodes=4 2, algorithms=transnetv2 fixed-stride.
# Override via env vars or CI/web variables for targeted ad-hoc runs without editing the script.
vllm_sampling_temperature_args=()
if [[ -n "${VLLM_SAMPLING_TEMPERATURE:-}" ]]; then
  vllm_sampling_temperature_args=(--vllm-sampling-temperature "${VLLM_SAMPLING_TEMPERATURE}")
fi
IFS=', ' read -ra _caption_list        <<< "${NVCF_SPLIT_BENCHMARK_CAPTIONS:-${CAPTION:-1}}"
IFS=', ' read -ra _captioning_algorithm_list <<< "${NVCF_SPLIT_BENCHMARK_MODELS:-qwen}"
IFS=', ' read -ra _num_nodes_list      <<< "${NVCF_SPLIT_BENCHMARK_NUM_NODES:-${NUM_NODES_LIST:-4 2}}"
IFS=', ' read -ra _splitting_algo_list <<< "${NVCF_SPLIT_BENCHMARK_SPLITTING_ALGORITHMS:-${SPLITTING_ALGORITHM_LIST:-transnetv2 fixed-stride}}"
filter_empty_items _caption_list
filter_empty_items _captioning_algorithm_list
filter_empty_items _num_nodes_list
filter_empty_items _splitting_algo_list
if [[ ${#_caption_list[@]} -eq 0 || ${#_captioning_algorithm_list[@]} -eq 0 || ${#_num_nodes_list[@]} -eq 0 || ${#_splitting_algo_list[@]} -eq 0 ]]; then
  echo "ERROR: split benchmark scenario lists must not be empty." >&2
  exit 1
fi
for captioning_algorithm in "${_captioning_algorithm_list[@]}"; do
  for caption in "${_caption_list[@]}"; do
    case "${caption,,}" in
      1 | true | yes) caption=1 ;;
      0 | false | no) caption=0 ;;
      *) echo "Invalid caption value: ${caption}" >&2; exit 1 ;;
    esac
    for num_nodes in "${_num_nodes_list[@]}"; do
      for splitting_algorithm in "${_splitting_algo_list[@]}"; do
        PERF_S3_OUTPUT_DIR="${PERF_S3_ROOT_DIR}/${date_str}_model_${captioning_algorithm}_nodes_${num_nodes}_caption_${caption}_${splitting_algorithm}"
        echo "PERF_S3_OUTPUT_DIR: ${PERF_S3_OUTPUT_DIR}"
        python -m benchmarks.split_pipeline.nvcf_split_benchmark \
          --num-nodes "${num_nodes}" \
          --caption "${caption}" \
          --splitting-algorithm "${splitting_algorithm}" \
          --captioning-algorithm "${captioning_algorithm}" \
          --funcid "${PERF_NVCF_FUNC_ID}" \
          --version "${PERF_NVCF_FUNC_VERSION}" \
          --image "${PERF_IMAGE}" \
          --metrics-endpoint "${PERF_NVCF_METRICS_ENDPOINT}" \
          --backend "${PERF_NVCF_BACKEND}" \
          --gpu "${PERF_NVCF_GPU}" \
          --instance-type "${PERF_NVCF_INSTANCE_TYPE}" \
          --s3-input-prefix "${PERF_S3_INPUT_DIR}" \
          --s3-output-prefix "${PERF_S3_OUTPUT_DIR}" \
          --gpus-per-node 8 \
          --max-concurrency 2 \
          "${vllm_sampling_temperature_args[@]}" \
          --kratos-metrics-endpoint "${PERF_KRATOS_METRICS_ENDPOINT}" \
          --kratos-bearer-url "${PERF_KRATOS_BEARER_URL}" \
          --limit "${LIMIT_INPUT_VIDEOS}" \
          --report-metrics-to-kratos
      done
    done
  done
done
