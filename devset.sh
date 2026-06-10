#!/bin/bash

# Install local developer hooks and run a packaging smoke test.

set -euo pipefail

pixi run -e tools pre-commit install
pixi run -e tools python -m build
