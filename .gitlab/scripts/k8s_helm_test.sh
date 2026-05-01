#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -e

# shellcheck source=common.sh
source "$(dirname "$0")/common.sh"

echo "=== K8s Helm Chart Deployment Test ==="

# Export for cleanup script
export NAMESPACE="ci-helm-test-${CI_PIPELINE_ID:-local}"
RELEASE_NAME="cosmos-curator-test"
STATEFULSET_NAME="cosmos-curator"  # From chart's nameOverride
CHART_PATH="charts/cosmos-curator"
POLL_TIMEOUT=1800  # 30 minutes

echo "Namespace: ${NAMESPACE}"
echo "Chart path: ${CHART_PATH}"

# Create namespace
kubectl create namespace "${NAMESPACE}"

# Wait for Kyverno to create the rolebinding (auto-generated on namespace creation)
echo "Waiting for rolebinding to be created by Kyverno..."
RBAC_TIMEOUT=30
RBAC_WAITED=0
until kubectl get rolebinding ci-test-admin -n "${NAMESPACE}" --ignore-not-found | grep -q ci-test-admin; do
    if [ $RBAC_WAITED -ge $RBAC_TIMEOUT ]; then
        echo "ERROR: Rolebinding ci-test-admin not created after ${RBAC_TIMEOUT}s"
        kubectl get rolebindings -n "${NAMESPACE}" || true
        exit 1
    fi
    echo "Waiting for rolebinding... (${RBAC_WAITED}s)"
    sleep 1
    RBAC_WAITED=$((RBAC_WAITED + 1))
done
echo "Rolebinding ready"

# Decode S3 credentials for helm
S3_CREDS_DATA=$(echo -n "$AWS_CONFIG_FILE_CONTENTS" | base64 -d)

# Build helm dependencies
echo "Building helm dependencies..."
helm repo add open-telemetry https://open-telemetry.github.io/opentelemetry-helm-charts || true
helm repo update
helm dep build "${CHART_PATH}"

# Determine image tag
IMAGE_TAG="${HELM_IMAGE_TAG:-$(get_image_tag)}"
echo "Using image tag: ${IMAGE_TAG}"

# Install chart with CI-specific overrides
# Secrets are created by helm via values, not manually via kubectl
# Resource limits come from values-standalone.yaml
echo "Installing helm chart..."
helm upgrade "${RELEASE_NAME}" "${CHART_PATH}" \
    --namespace "${NAMESPACE}" \
    --install \
    --wait=false \
    -f "${CHART_PATH}/values.yaml" \
    -f "${CHART_PATH}/values-standalone.yaml" \
    --set replicas=1 \
    --set image.repository="${CURATOR_FULL_IMAGE%:*}" \
    --set image.tag="${IMAGE_TAG}" \
    --set imagePullSecret.dockerConfigJson.registry="${CI_REGISTRY}" \
    --set imagePullSecret.dockerConfigJson.username="${CI_REGISTRY_USER:-gitlab-ci-token}" \
    --set imagePullSecret.dockerConfigJson.password="${CI_REGISTRY_PASSWORD:-${CI_JOB_TOKEN}}" \
    --set s3.secret.enabled=true \
    --set s3.credsPath="/s3config/s3.config" \
    --set-string "s3.secret.data=${S3_CREDS_DATA}" \
    --set "customEnvVars.COSMOS_S3_PROFILE_PATH=/s3config/s3.config" \
    --set ngcCatalog.secret.key="${NGC_API_KEY}" \
    --set ngcCatalog.secret.org="${NGC_ORG}"

# Wait for statefulset to be ready
echo "Waiting for StatefulSet to be ready (up to ${POLL_TIMEOUT}s)..."
WAITED=0
while [ $WAITED -lt $POLL_TIMEOUT ]; do
    # jsonpath returns empty if field missing, so default empty to 0
    READY=$(kubectl get statefulset -n "${NAMESPACE}" "${STATEFULSET_NAME}" -o jsonpath='{.status.readyReplicas}' 2>/dev/null)
    READY="${READY:-0}"
    DESIRED=$(kubectl get statefulset -n "${NAMESPACE}" "${STATEFULSET_NAME}" -o jsonpath='{.spec.replicas}' 2>/dev/null)
    DESIRED="${DESIRED:-1}"

    if [ "${READY}" -eq "${DESIRED}" ] && [ "${READY}" -gt 0 ]; then
        echo "StatefulSet ready: ${READY}/${DESIRED}"
        break
    fi

    echo "Waiting... (${WAITED}s elapsed, ${READY}/${DESIRED} ready)"

    # Show pod status for debugging
    kubectl get pods -n "${NAMESPACE}" -l app.kubernetes.io/name=cosmos-curator -o wide --no-headers 2>/dev/null || true

    sleep 15
    WAITED=$((WAITED + 15))
done

if [ $WAITED -ge $POLL_TIMEOUT ]; then
    echo "ERROR: StatefulSet did not become ready in ${POLL_TIMEOUT}s"
    kubectl describe statefulset -n "${NAMESPACE}" "${STATEFULSET_NAME}" || true
    kubectl describe pods -n "${NAMESPACE}" -l app.kubernetes.io/name=cosmos-curator || true
    kubectl logs -n "${NAMESPACE}" -l app.kubernetes.io/name=cosmos-curator --tail=100 || true
    exit 1
fi

# Get pod name
POD_NAME=$(kubectl get pods -n "${NAMESPACE}" -l app.kubernetes.io/name=cosmos-curator -o jsonpath='{.items[0].metadata.name}')
echo "Pod ready: ${POD_NAME}"

# Run a mini pipeline test via kubectl exec
echo "=== Running pipeline test inside pod ==="

K8S_OUTPUT_PATH="${S3_OUTPUT_PATH}/k8s-helm-test"

# Get the reduced CPU args for the inner command
REDUCED_CPU_ARGS=$(get_reduced_cpu_pipeline_args)

kubectl exec -n "${NAMESPACE}" "${POD_NAME}" -- bash -c "
set -e
cd /opt/cosmos-curator

echo 'Verifying GPU...'
nvidia-smi

echo 'Checking env vars with shm...'
env | grep -i s3 || echo 'No s3 vars found'

echo 'Running split pipeline...'
export COSMOS_S3_PROFILE_PATH=/s3config/s3.config
export NVCF_MODEL_CACHE_DIR=/config/models
export NVCF_MULTI_NODE=true

pixi run --as-is python -m cosmos_curator.pipelines.video.run_pipeline split \
  --input-video-path '${S3_INPUT_VIDEO_PATH}' \
  --output-clip-path '${K8S_OUTPUT_PATH}' \
  --limit 1 \
  --splitting-algorithm transnetv2 \
  ${REDUCED_CPU_ARGS} \
  --embedding-algorithm cosmos-embed1-224p \
  --captioning-algorithm cosmos_r1

echo '✓ Pipeline completed successfully'
"

echo "=== K8s Helm Test Completed Successfully ==="
