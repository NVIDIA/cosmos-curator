#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Run vnext recipe configs in order with `run-pipeline`, stopping at the first failure.
#
# Each config is rendered before it runs: ${NAME} placeholders are replaced from the
# environment, so a committed config can take run-specific locations (input and output URIs)
# without being edited. A placeholder with no matching variable is an error.
#
# Usage: run_vnext_recipes.sh <config.yaml>...
# Run from a directory where `pixi run` resolves the default environment.

set -euo pipefail

if [ "$#" -eq 0 ]; then
    echo "usage: $0 <config.yaml>..." >&2
    exit 2
fi

run_vnext_recipe() {
    local config="$1"
    if [ ! -f "${config}" ]; then
        echo "ERROR: recipe config not found: ${config}" >&2
        return 1
    fi

    local rendered
    rendered="$(mktemp --suffix=.yaml)"
    # Remove the rendered config when the function returns (success or set -e abort).
    trap 'rm -f "${rendered}"' RETURN

    # Render with the pixi-managed Python rather than envsubst, which the slim image omits.
    CONFIG_SRC="${config}" CONFIG_DST="${rendered}" pixi run --as-is python - <<'PY'
import os
import string
from pathlib import Path

text = Path(os.environ["CONFIG_SRC"]).read_text()
Path(os.environ["CONFIG_DST"]).write_text(string.Template(text).substitute(os.environ))
PY

    echo "=== Running recipe: ${config} ==="
    pixi run --as-is run-pipeline "${rendered}" --json
}

for config in "$@"; do
    run_vnext_recipe "${config}"
done
