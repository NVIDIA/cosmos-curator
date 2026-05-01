#!/bin/bash

# Validate required environment variables
for var in SLURM_COSMOS_CURATOR_CONFIG_DIR SLURM_WORKSPACE SLURM_SOURCE_DIR; do
    if [ -z "${!var}" ]; then
        echo "Error: $var is not defined. Please source source_me_env_vars.sh first."
        return
    fi
done

echo "Adding source-code mount CONTAINER_MOUNTS..."

SLURM_COSMOS_CURATOR_CONFIG_MOUNT="${SLURM_COSMOS_CURATOR_CONFIG_DIR}/config.yaml:/cosmos_curator/config/cosmos_curator.yaml"
SLURM_WORKSPACE_MOUNT="${SLURM_WORKSPACE}:/config"
SLURM_SOURCE_MOUNT="${SLURM_SOURCE_DIR}/cosmos_curator/:/opt/cosmos-curator/cosmos_curator"
MOUNTS="${SLURM_COSMOS_CURATOR_CONFIG_MOUNT},${SLURM_WORKSPACE_MOUNT},${SLURM_SOURCE_MOUNT}"

if [ -f "${SLURM_AWS_CREDS_DIR}/credentials" ]; then
    MOUNTS+=",${SLURM_AWS_CREDS_DIR}/credentials:/creds/s3_creds"
else
    echo "AWS credentials not found at ${SLURM_AWS_CREDS_DIR}/credentials; skipping S3 mount."
fi

if [ -f "${SLURM_AZURE_CREDS_DIR}/credentials" ]; then
    MOUNTS+=",${SLURM_AZURE_CREDS_DIR}/credentials:/creds/azure_creds"
else
    echo "Azure credentials not found at ${SLURM_AZURE_CREDS_DIR}/credentials; skipping Azure mount."
fi

export CONTAINER_MOUNTS="$MOUNTS"

echo "--------------------------------"
echo "CONTAINER_MOUNTS: $CONTAINER_MOUNTS"
