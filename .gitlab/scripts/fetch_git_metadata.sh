#!/usr/bin/env bash
set -euo pipefail

repo_dir="${CI_PROJECT_DIR:-$(pwd)}"
git config --global --add safe.directory "${repo_dir}"

if [[ "$(git rev-parse --is-shallow-repository)" == "true" ]]; then
    git fetch --unshallow --tags --force
else
    git fetch --tags --force
fi
