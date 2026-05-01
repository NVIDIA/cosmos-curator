#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# NVCF staging image latestTag for CI (GitLab dotenv or sourcing for Helm).
#
# When executed: --write-dotenv [path] writes STAGING_TAG / STAGING_IMAGE (override: STAGING_TAG_OVERRIDE).
# When sourced: defines nvcf_registry_latest_tag() for nvcf_helm_deploy.sh (no side effects).
#
# Requires: cosmos-curator nvcf config set (configure_nvcf_cli) before list-image-detail.

set -euo pipefail

nvcf_registry_latest_tag() {
  local iname="${1:?base image name required}"
  cosmos-curator nvcf image list-image-detail --iname "${iname}" \
    | grep latestTag | sed "s/['│,]//g" | awk '{print $2}' | head -n1
}

_write_staging_dotenv() {
  local out="${1:-staging_tag.env}"
  local tag=""
  if [[ -n "${STAGING_TAG_OVERRIDE:-}" ]]; then
    tag="${STAGING_TAG_OVERRIDE}"
  else
    tag=$(nvcf_registry_latest_tag "${NVCF_STAGING_BASE_IMAGE}")
  fi
  if [[ -z "${tag}" ]]; then
    echo "ERROR: empty STAGING_TAG (NVCF list-image-detail / STAGING_TAG_OVERRIDE)" >&2
    exit 1
  fi
  {
    echo "STAGING_TAG=${tag}"
    echo "STAGING_IMAGE_NAME=${NVCF_STAGING_BASE_IMAGE}"
    echo "STAGING_IMAGE=${NVCF_STAGING_IMAGE}:${tag}"
  } >"${out}"
  echo "Wrote ${out} STAGING_TAG=${tag}"
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  case "${1:-}" in
    --write-dotenv)
      _write_staging_dotenv "${2:-staging_tag.env}"
      ;;
    *)
      echo "Usage: $0 --write-dotenv [path]" >&2
      exit 1
      ;;
  esac
fi
