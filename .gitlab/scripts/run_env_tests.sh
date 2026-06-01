#!/usr/bin/env bash
set -euo pipefail

# Install pixi environments inside the container from the Lustre-mounted cache.
# This writes .pixi to local container storage so Python imports are fast.
ENVS=(default legacy-transformers unified)
ENV_ARGS=()
for e in "${ENVS[@]}"; do ENV_ARGS+=(-e "$e"); done
echo "Installing pixi environments: ${ENVS[*]}"
pixi install --frozen "${ENV_ARGS[@]}"

# If the build_xenna_wheels CI job uploaded a wheel artifact, override the
# pixi-installed cosmos-xenna in each env with the locally-built wheel. The
# CI job mounts ${CI_PROJECT_DIR} as /config/project inside the container, so
# the artifact lives at /config/project/cosmos-xenna/target/wheels/. The
# helper is a no-op when the artifact is absent.
WHEEL_DIR="/config/project/cosmos-xenna/target/wheels" \
    bash /config/project/.gitlab/scripts/install_local_xenna_into_pixi.sh "${ENVS[@]}"

# Run tests for each environment with unique report files and coverage
for env in "${ENVS[@]}"; do
  echo "Running tests for $env environment"
  pixi run --as-is -e $env pytest -m env -n "${PYTEST_XDIST_WORKERS}" \
    --junitxml="/config/project/$env-report.xml" \
    --cov=cosmos_curator \
    --cov-report=term \
    --cov-report=xml:/config/project/$env-coverage.xml \
    --cov-report=html:/config/project/$env-htmlcov \
    tests/cosmos_curator/pipelines tests/cosmos_curator/models

  # Save the coverage data file for each environment
  if [ -f .coverage ]; then
    cp .coverage /config/project/.coverage.gpu_$env
  fi
done
