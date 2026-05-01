#!/bin/bash

# Parse command-line arguments
SKIP_AWS=0
SKIP_AZURE=0

for arg in "$@"; do
    case $arg in
        --no-aws)
            SKIP_AWS=1
            ;;
        --no-azure)
            SKIP_AZURE=1
            ;;
        --help|-h)
            echo "Usage: source source_me_env_vars.sh [--no-aws] [--no-azure]"
            echo "  --no-aws     Skip AWS credentials mount"
            echo "  --no-azure   Skip Azure credentials mount"
            echo "  --help, -h   Show help message"
            return 0
            ;;
        *)
            echo "Unknown option: $arg"
            echo "Use --help for usage information"
            return 1
            ;;
    esac
done

# check if env var SLURM_USER_DIR is defined
if [ -z "$SLURM_USER_DIR" ]; then
    echo "Error: SLURM_USER_DIR is not defined. Please set it to your SLURM user directory."
    return
fi

echo "Using $SLURM_USER_DIR as user directory on Slurm cluster."
echo "--------------------------------"

export SLURM_LOG_DIR="${SLURM_USER_DIR}/job_logs"
export SLURM_COSMOS_CURATOR_CONFIG_DIR="${SLURM_USER_DIR}/.config/cosmos_curator"
export SLURM_AWS_CREDS_DIR="${SLURM_USER_DIR}/.aws"
export SLURM_AZURE_CREDS_DIR="${SLURM_USER_DIR}/.azure"
export SLURM_WORKSPACE="${SLURM_USER_DIR}/cosmos_curator_local_workspace"
export SLURM_IMAGE_DIR="${SLURM_USER_DIR}/container_images"
export SLURM_SOURCE_DIR="${SLURM_USER_DIR}/src/cosmos-curator"

echo "SLURM_LOG_DIR: $SLURM_LOG_DIR"
echo "SLURM_COSMOS_CURATOR_CONFIG_DIR: $SLURM_COSMOS_CURATOR_CONFIG_DIR"
echo "SLURM_AWS_CREDS_DIR: $SLURM_AWS_CREDS_DIR"
echo "SLURM_AZURE_CREDS_DIR: $SLURM_AZURE_CREDS_DIR"
echo "SLURM_WORKSPACE: $SLURM_WORKSPACE"
echo "SLURM_IMAGE_DIR: $SLURM_IMAGE_DIR"
echo "SLURM_SOURCE_DIR: $SLURM_SOURCE_DIR"

echo "--------------------------------"
echo "Setting CONTAINER_MOUNTS..."

SLURM_COSMOS_CURATOR_CONFIG_MOUNT="${SLURM_COSMOS_CURATOR_CONFIG_DIR}/config.yaml:/cosmos_curator/config/cosmos_curator.yaml"
SLURM_WORKSPACE_MOUNT="${SLURM_WORKSPACE}:/config"
MOUNTS="${SLURM_COSMOS_CURATOR_CONFIG_MOUNT},${SLURM_WORKSPACE_MOUNT}"

# Include AWS credentials mount if not skipped (validation happens on cluster in curator_submit)
if [ "$SKIP_AWS" -eq 0 ]; then
    MOUNTS+=",${SLURM_AWS_CREDS_DIR}/credentials:/creds/s3_creds"
fi

# Include Azure credentials mount if not skipped (validation happens on cluster in curator_submit)
if [ "$SKIP_AZURE" -eq 0 ]; then
    MOUNTS+=",${SLURM_AZURE_CREDS_DIR}/credentials:/creds/azure_creds"
fi

export CONTAINER_MOUNTS="$MOUNTS"

echo "--------------------------------"
echo "CONTAINER_MOUNTS: $CONTAINER_MOUNTS"
