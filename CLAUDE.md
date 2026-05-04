# Cosmos Curator — Agent Guidelines

## Project Overview

Cosmos Curator is a video curation system for AI training data generation, built on [Cosmos-Xenna](https://github.com/nvidia-cosmos/cosmos-xenna) (GPU-accelerated streaming pipelines using Ray).

**Architecture**: Three-layer modular design
- `cosmos_curator/client/`: Deployment CLIs (local, Slurm, NVCF, Docker image management)
- `cosmos_curator/core/`: Base interfaces (`PipelineTask`, `CuratorStage`, `ModelInterface`), managers, utilities
- `cosmos_curator/pipelines/`: Video/AV pipelines + examples (start with `hello_world_pipeline.py`)

## Development

**Setup**: `git submodule update --init --recursive && ./devset.sh`

**Code Quality** (run globally before committing, not just on changed files):
```bash
ruff format && ruff check --fix && mypy
```

Pre-commit hooks run ruff automatically. A submodule-check hook warns before committing cosmos-xenna changes.

**Testing**:
- CPU tests: `pytest` (env-marked tests are excluded by default via `pytest.ini`)
- GPU tests: `cosmos-curator local launch --curator-path . -- pixi run --as-is -e [default|unified] pytest -m env tests/`
- Mark GPU tests with `@pytest.mark.env("unified")` (or other env name)
- Place tests in `tests/` mirroring module paths. Uses `--import-mode=importlib`.

**Building**: Poetry frontend with setuptools backend. `poetry build` for client wheel, `cosmos-curator image build` for Docker.

**CLI**: `cosmos-curator [local|slurm|nvcf|image|view] --help`

## Code Style

- The current year is 2026 — use `2026` in any new copyright headers, date literals, or comments (not 2025)
- PEP 8 (4-space indent, `snake_case`/`CamelCase`), type hints, ruff formatting
- Config: `pyproject.toml` (Python 3.12, 120 chars)
- Do NOT add `from __future__ import annotations` — use native Python 3.12 type hints instead (PEP 649 is Python 3.13+)
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

Defined in `pixi.toml`: `default` (core), `unified` (vllm/advanced models), `transformers`, `legacy-transformers`, `cuml`, `model-download`, `paddle-ocr`

Stages specify `conda_env_name` property to run in specific environments, enabling dependency isolation.

## Commits & PRs

**Commits**: Conventional Commits: `git commit -m "fix: description"`. MRs squash on merge, so commit early and often as checkpoints — individual commit messages don't need to be perfect.

**Merge Requests**:
- Create using `glab mr create` targeting branch `nvidia/main`, assign to author, with `--remove-source-branch --squash-before-merge`
- Summarize intent, list validation commands, note doc/submodule updates, link issues
- Do NOT include "Generated with Claude Code" in descriptions
- Use GitLab MCP server (if available) to retrieve CodeRabbit and Greptile review comments

**Submodule updates**: Pre-commit warns before committing. Update via `cd cosmos-xenna && git checkout VERSION && cd .. && git add cosmos-xenna`
