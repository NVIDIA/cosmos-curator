#!/bin/bash

SLURM_LOGIN_HOST="my-slurm-login-01.my-cluster.com"

# Resolve the actual hostname via ssh -G so rclone connects to the right host.
# OpenSSH reads ~/.ssh/config (HostName, ProxyJump, etc.) but rclone's built-in
# SFTP implementation does not, so we must expand it ourselves.
RESOLVED_HOST=$(ssh -G "${SLURM_LOGIN_HOST}" 2>/dev/null | awk '/^hostname / {print $2}')
RESOLVED_HOST="${RESOLVED_HOST:-${SLURM_LOGIN_HOST}}"

RCLONE_REMOTE=":sftp,host=${RESOLVED_HOST},key_use_agent=true:"

if [ -z "${SLURM_COSMOS_CURATOR_CONFIG_DIR}" ]; then
    echo "Error: SLURM_COSMOS_CURATOR_CONFIG_DIR is not defined"
else
    echo "sync-ing cosmos_curator config yaml"
    ssh "${SLURM_LOGIN_HOST}" mkdir -p "${SLURM_COSMOS_CURATOR_CONFIG_DIR}"
    rclone copyto -P ~/.config/cosmos_curator/config.yaml "${RCLONE_REMOTE}${SLURM_COSMOS_CURATOR_CONFIG_DIR}/config.yaml"
fi

if [ -z "${SLURM_AWS_CREDS_DIR}" ]; then
    echo "SLURM_AWS_CREDS_DIR is not defined, skipping AWS creds sync"
elif [ -f ~/.aws/credentials ]; then
    echo "sync-ing aws creds"
    ssh "${SLURM_LOGIN_HOST}" mkdir -p "${SLURM_AWS_CREDS_DIR}"
    rclone copyto -P ~/.aws/credentials "${RCLONE_REMOTE}${SLURM_AWS_CREDS_DIR}/credentials"
else
    echo "AWS credentials file ~/.aws/credentials not found, skipping AWS creds sync"
fi

if [ -z "${SLURM_AZURE_CREDS_DIR}" ]; then
    echo "SLURM_AZURE_CREDS_DIR is not defined, skipping Azure creds sync"
elif [ -f ~/.azure/credentials ]; then
    echo "sync-ing azure creds"
    ssh "${SLURM_LOGIN_HOST}" mkdir -p "${SLURM_AZURE_CREDS_DIR}"
    rclone copyto -P ~/.azure/credentials "${RCLONE_REMOTE}${SLURM_AZURE_CREDS_DIR}/credentials"
else
    echo "Azure credentials file ~/.azure/credentials not found, skipping Azure creds sync"
fi

if [ -z "${SLURM_WORKSPACE}" ]; then
    echo "Error: SLURM_WORKSPACE is not defined"
else
    LOCAL_MODELS="${COSMOS_CURATOR_LOCAL_WORKSPACE_PREFIX:-$HOME}/cosmos_curator_local_workspace/models"
    if [ -d "${LOCAL_MODELS}" ] && [ "$(ls -A "${LOCAL_MODELS}" 2>/dev/null)" ]; then
        echo "sync-ing models from local workspace"
        ssh "${SLURM_LOGIN_HOST}" mkdir -p "${SLURM_WORKSPACE}/models"
        rclone copy -P "${LOCAL_MODELS}/" "${RCLONE_REMOTE}${SLURM_WORKSPACE}/models/"
    else
        echo "No local models found at ${LOCAL_MODELS}, skipping model sync."
        echo "Tip: large models (e.g. SeedVR2) are faster to download directly on the"
        echo "cluster via: pixi run --as-is -e model-download python -m cosmos_curator.core.managers.model_cli download --models seedvr2_3b"
    fi
fi
