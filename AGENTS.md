# Cosmos Curator — Agent Guidelines

## Project Overview

Cosmos Curator is a video curation system for AI training data generation, built on [Cosmos-Xenna](https://github.com/nvidia-cosmos/cosmos-xenna) (GPU-accelerated streaming pipelines using Ray).

**Architecture**: Three-layer modular design
- `cosmos_curator/client/`: Deployment CLIs (local, Slurm, NVCF, Docker image management)
- `cosmos_curator/core/`: Base interfaces (`PipelineTask`, `CuratorStage`, `ModelInterface`), managers, utilities
- `cosmos_curator/pipelines/`: Video and image pipelines + examples (start with `hello_world_pipeline.py`)

## Development

**Setup**: `git submodule update --init --recursive`; `./devset.sh` installs local git hooks and runs a build smoke test.

**Code Quality** (run globally before committing, not just on changed files):
```bash
pixi run ruff format && pixi run ruff check --fix && pixi run mypy
```

Pre-commit hooks run ruff automatically. A submodule-check hook warns before committing cosmos-xenna changes.

**Testing**:
- CPU unit tests: `pytest` (env- and integration-marked tests are excluded by default via `pytest.ini`)
- Tests that intentionally exercise CLI usage errors or other expected stderr/stdout output must capture and assert that output with `capsys`/`capfd`; passing tests should not leak expected errors into `pytest -s` CI logs.
- CPU integration tests: mark with `@pytest.mark.integration` and run `pixi run --as-is -e default integration-test`. CI runs these serially in the Kubernetes `integration_tests` job using the slim image; they must not require GPUs or staged model weights.
- GPU tests: `cosmos-curator local launch --curator-path . -- pixi run --as-is -e default gputest`. Use `@pytest.mark.env("default")` (or another required env name); CI runs these in `gpu_tests` on Slurm. Requires model weights staged in the local workspace (`model_download`).
- Do not add real Ray execution to `unit_tests`, including implicit cluster startup through Ray Data. CPU Ray tests belong behind `@pytest.mark.integration`; tests that also require GPUs or model weights use the GPU suite's env marker. Mocked Ray tests stay in the CPU unit suite. Mark the test, class, or module, not just its fixture; fixture markers do not select tests.
- Both runtime test tasks collect only `tests/cosmos_curator/{pipelines,models,next}` to avoid dev-only client/benchmark dependencies. The runtime image omits `cosmos_curator.client`; import client code only inside the unit tests that need it so runtime collection works without the deployment client.
- Place tests in `tests/` mirroring module paths. Uses `--import-mode=importlib`.

**Building**: Setuptools backend via PyPA build. `pixi run build` for client wheel, `cosmos-curator image build` for Docker.

**CLI**: `cosmos-curator [local|slurm|nvcf|image|view] --help`

## Code Style

- The current year is 2026 — use `2026` in any new copyright headers, date literals, or comments (not 2025)
- PEP 8 (4-space indent, `snake_case`/`CamelCase`), type hints, ruff formatting
- Config: `pyproject.toml` (Python 3.13, 120 chars)
- Do NOT add `from __future__ import annotations` — use native Python 3.13 type hints instead. The one exception is `cosmos_curator/client/slurm_cli/managed_ray/onnode/`, which targets Python 3.8 because those modules are uploaded into a Slurm run directory and run by a node's own `python3`; see that package's docstring
- `cosmos-xenna/` is excluded from ruff (via exclude list) and mypy (via `files = ["cosmos_curator"]`) — do not lint or fix code there

## Key Imports

Core interfaces have no `__init__.py` re-exports — always use full paths for these imports:

```python
from cosmos_curator.core.interfaces.stage_interface import CuratorStage, CuratorStageResource, CuratorStageSpec,
    PipelineTask
from cosmos_curator.core.interfaces.pipeline_interface import run_pipeline
from cosmos_curator.core.interfaces.model_interface import ModelInterface
```

## Creating Pipelines

**Tasks**: Inherit from `PipelineTask` (`@attrs.define`). Override `weight` property for load balancing.

**Stages**: Inherit from `CuratorStage`. Implement:
- `resources` property → `CuratorStageResource(cpus=X, gpus=Y)` (fractional GPUs allowed)
- `conda_env_name` property → environment name or `None` for default
- `process_data(task)` → process batch, can return different count or `None` to stop
- Optional `model` property → return `ModelInterface` instance

Stage lifecycle: `stage_setup_on_node()` (once per node) → `stage_setup()` (once per worker, runs in remote actor in target conda env) → `process_data()` (batches of `stage_batch_size`, default 1)

**Models**: Inherit from `ModelInterface`. Implement `conda_env_name`, `model_id_names`, `setup()`. Register in `cosmos_curator/models/all_models.py`.

**Running**: `run_pipeline(input_tasks, stages)` — accepts bare `CuratorStage` or `CuratorStageSpec`. Use `CuratorStageSpec(MyStage(), num_workers_per_node=N)` for tuning.

See `cosmos_curator/pipelines/examples/hello_world_pipeline.py` and `docs/curator/guides/pipeline-design.md`

## Pixi Environments

Defined in `pixi.toml`: `default` (core + vLLM/CVCUDA/PaddleOCR-CPU/advanced models), `transformers`, `legacy-transformers`, `cuml`, `model-download`, `paddle-ocr`, `seedvr`, `sam3`

Stages specify `conda_env_name` property to run in specific environments, enabling dependency isolation.

## Commits & PRs

**Commits**: Conventional Commits: `git commit -m "fix: description"`. MRs squash on merge, so commit early and often as checkpoints — individual commit messages don't need to be perfect.

**Merge Requests**:
- Create using `glab mr create` targeting branch `nvidia/main`, assign to `@me`, with `--remove-source-branch --squash-before-merge`
- Write MR descriptions for reviewers: summarize intent and behavior changes, call out impacted CLIs/pipelines/envs and non-obvious risks or follow-ups, and include issue links or manual validation only when known and useful beyond CI
- Do NOT include "Generated with Claude Code" in descriptions
- Use GitLab MCP server (if available) to retrieve CodeRabbit and Greptile review comments

**Submodule updates**: Pre-commit warns before committing. Update via `cd cosmos-xenna && git checkout VERSION && cd .. && git add cosmos-xenna`
