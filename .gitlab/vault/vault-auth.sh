#!/bin/bash
set -e

echo "Setting up Vault authentication..."

# Install dependencies if missing
if ! which curl >/dev/null 2>&1 || ! which unzip >/dev/null 2>&1; then
  echo "Installing required dependencies..."
  if which apk >/dev/null 2>&1; then
    # Alpine
    apk add --no-cache curl unzip
  elif which apt-get >/dev/null 2>&1; then
    # Debian/Ubuntu
    DEBIAN_FRONTEND=noninteractive apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y curl unzip
  elif which yum >/dev/null 2>&1; then
    # RHEL/CentOS
    yum install -y curl unzip
  elif which dnf >/dev/null 2>&1; then
    # Fedora
    dnf install -y curl unzip
  else
    echo "Warning: Could not detect package manager. Please ensure curl and unzip are available."
  fi
fi

# Use preinstalled vault when available, otherwise download a local fallback
DOWNLOADED_VAULT="false"
if which vault >/dev/null 2>&1; then
  VAULT_BIN="$(which vault)"
  echo "Using preinstalled Vault binary: ${VAULT_BIN}"
else
  echo "Vault binary not found; downloading Vault agent fallback..."
  rm -f vault_agent.zip vault
  ARCH="$(uname -m)"
  VAULT_AGENT_URL="${VAULT_AGENT_DOWNLOAD_PATH}/latest/nvault_agent_linux_amd64.zip"
  if [ "${ARCH}" = "aarch64" ]; then
    VAULT_AGENT_URL="${VAULT_AGENT_DOWNLOAD_PATH}/latest/nvault_agent_linux_arm64.zip"
  fi
  curl "${VAULT_AGENT_URL}" -L -o vault_agent.zip
  unzip vault_agent.zip
  chmod +x vault
  VAULT_BIN="./vault"
  DOWNLOADED_VAULT="true"
fi

# Verify vault binary
"${VAULT_BIN}" -version

# Derive the Vault secret path suffix from whichever Artifactory host is
# currently selected via ARTIFACTORY_URL. This keeps the URL as the single
# source of truth for both the twine publish target and which Artifactory
# creds get pulled -- no separate var to keep in sync when flipping between
# legacy/cloud.
if [ -z "${ARTIFACTORY_URL:-}" ]; then
  echo "Error: ARTIFACTORY_URL is not set"
  exit 1
fi
VAULT_ARTIFACTORY_HOST="$(echo "${ARTIFACTORY_URL}" | sed -E 's#^[a-zA-Z][a-zA-Z0-9+.-]*://([^/]+).*#\1#')"
export VAULT_ARTIFACTORY_HOST
echo "Resolved Artifactory host for Vault lookup: ${VAULT_ARTIFACTORY_HOST}"

# Run vault agent to authenticate and render secrets
echo "Authenticating with Vault and retrieving secrets..."
"${VAULT_BIN}" agent -config=.gitlab/vault/vault-agent.config -exit-after-auth

# Verify secrets file was created
if [ ! -f "./ci_secrets.env" ]; then
  echo "Error: Vault secrets file not created"
  exit 1
fi

# Load secrets into environment
source ./ci_secrets.env
echo "Vault secrets retrieved and loaded successfully"

# Clean up downloaded vault artifacts if fallback path was used
if [ "${DOWNLOADED_VAULT}" = "true" ]; then
  rm -f vault_agent.zip vault
fi

# Clean up files now that they're environment variables
rm -f ./ci_secrets.env
echo "Secrets files cleaned up"
