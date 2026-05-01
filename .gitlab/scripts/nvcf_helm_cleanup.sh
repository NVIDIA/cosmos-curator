#!/usr/bin/env bash
set -euo pipefail

# Cleanup NVCF function
if [[ "$HELM_DEBUG_KEEP_FAILED_DEPLOYMENTS" != "True" ]]; then
  if [ ! -f ~/.config/cosmos_curator/funcid.json ]; then
    echo "$HOME/.config/cosmos_curator/funcid.json not found, using working directory copy"
    if [ ! -f funcid_working.json ]; then
      echo "Error: funcid_working.json not found in working directory"
      exit 1
    fi
    mkdir -p ~/.config/cosmos_curator/
    cp funcid_working.json ~/.config/cosmos_curator/funcid.json
  fi
  cosmos-curator nvcf function delete-function
else
  echo "Intentionally leaving deployment behind for debugging. This must be manually cleaned up later."
fi
