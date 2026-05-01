#!/bin/bash

# get directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
ROOT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
RCLONE_REMOTE=":sftp,host=my-slurm-login-01.my-cluster.com:"

if [ -z "${SLURM_SOURCE_DIR}" ]; then
    echo "Error: SLURM_SOURCE_DIR is not defined"
else
    echo "sync-ing source code"
    ssh my-slurm-login-01.my-cluster.com mkdir -p "${SLURM_SOURCE_DIR}/cosmos_curator/"
    rclone copy -P --exclude="*.pyc" --exclude="__pycache__/**" \
        "${ROOT_DIR}/cosmos_curator/" \
        "${RCLONE_REMOTE}${SLURM_SOURCE_DIR}/cosmos_curator/"
fi

