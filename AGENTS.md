# Cosmos Curator — Agent Guidelines

## Project Overview

Cosmos Curator is a toolkit for curating physical AI training data.
Start new curation workflows in [Curator Next](cosmos_curator/next/README.md): Ray Data recipes and shared components,
with canonical tabular outputs in Lance. Existing Xenna and legacy Ray Data pipelines remain available during the
transition; retire them through explicit migration to Next.

**Code map**:

- `cosmos_curator/next/`: Reference recipes and shared components
- `cosmos_curator/client/`: Deployment CLIs and pipeline config tooling
- `cosmos_curator/core/`: Sensor library, interfaces, managers, and utilities
- `cosmos_curator/models/`: Model implementations; register model artifacts in `cosmos_curator/configs/all_models.json`
- `cosmos_curator/pipelines/`: Existing Xenna/legacy Ray Data pipelines plus shared pipeline CLI registration and runtime

For changes to Next code or its mirrored tests, read [next/AGENTS.md](cosmos_curator/next/AGENTS.md).
For sensor code or its mirrored tests, read [sensors/AGENTS.md](cosmos_curator/core/sensors/AGENTS.md).

## Development

**Setup**: `git submodule update --init --recursive`; `./devset.sh` installs local git hooks and runs a build smoke test.

**Code Quality** (run globally before committing, not just on changed files):

```bash
pixi run -e tools ruff format && pixi run -e tools ruff check --fix && pixi run -e dev mypy
```

Pre-commit hooks run ruff automatically. A submodule-check hook warns before committing cosmos-xenna changes.

**Testing**:

- CPU unit tests: `pixi run unit-test` (`pixi run unit-test-fast` for four workers). Env- and integration-marked tests are excluded by default via `pytest.ini`.
- Tests that intentionally exercise CLI usage errors or other expected stderr/stdout output must capture and assert that output with `capsys`/`capfd`; passing tests should not leak expected errors into `pytest -s` CI logs.
- CPU integration tests: mark with `@pytest.mark.integration` and run `pixi run --as-is -e default integration-test`. CI runs these serially in the Kubernetes `integration_tests` job using the slim image; they must not require GPUs or staged model weights.
- GPU tests: `cosmos-curator local launch --curator-path . -- pixi run --as-is -e default gpu-test`. Use `@pytest.mark.env("default")` (or another required env name); CI runs these in `gpu_tests` on Slurm. Requires model weights staged in the local workspace (`model-download`).
- Do not add real Ray execution to `unit_tests`, including implicit cluster startup through Ray Data. CPU Ray tests belong behind `@pytest.mark.integration`; tests that also require GPUs or model weights use the GPU suite's env marker. Mocked Ray tests stay in the CPU unit suite. Mark the test, class, or module, not just its fixture; fixture markers do not select tests.
- Both runtime test tasks collect only `tests/cosmos_curator/{pipelines,models,next}` to avoid dev-only client/benchmark dependencies. The runtime image omits `cosmos_curator.client`; import client code only inside the unit tests that need it so runtime collection works without the deployment client.
- Place tests in `tests/` mirroring module paths. Uses `--import-mode=importlib`.

**Building**: Setuptools backend via PyPA build. `pixi run -e tools build` for client wheel, `cosmos-curator image build` for Docker.

**CLI**: `cosmos-curator [local|slurm|nvcf|image|pipeline|view] --help`

## Pixi Environments

Use `tools` for linting, packaging, and the deployment CLI; `dev` for CPU unit tests and mypy; and `default` or a
model-specific environment for runtime work. [pixi.toml](pixi.toml) is the source of truth for environments and tasks.
`transformers` is a feature included by environments, not a standalone environment.

After changing `pixi.toml`, regenerate the derived distributable manifest and lockfile with
`pixi run -e tools python tools/update_distributable_pixi.py`. These files are generated; the pre-commit hook checks
them with `--check`.

## Code Style

- Use `2026` for new copyright headers and examples intended to show the current year. Preserve historical dates and
  meaningful fixture dates.
- PEP 8 (4-space indent, `snake_case`/`CamelCase`), type hints, ruff formatting
- Config: `pyproject.toml` (Python 3.13, 120 chars)
- Do NOT add `from __future__ import annotations` — use native Python 3.13 type hints instead. The one exception is `cosmos_curator/client/slurm_cli/managed_ray/onnode/`, which targets Python 3.8 because those modules are uploaded into a Slurm run directory and run by a node's own `python3`; see that package's docstring
- `cosmos-xenna/` is excluded from ruff (via exclude list) and mypy (via `files = ["cosmos_curator"]`) — do not lint or fix code there

## Existing Xenna Pipelines

Use this section when maintaining Xenna pipelines. New recipes follow the Next guide above.
Core interfaces have no `__init__.py` re-exports; import them from their defining modules:

```python
from cosmos_curator.core.interfaces.model_interface import ModelInterface
from cosmos_curator.core.interfaces.pipeline_interface import run_pipeline
from cosmos_curator.core.interfaces.stage_interface import (
    CuratorStage,
    CuratorStageResource,
    CuratorStageSpec,
    PipelineTask,
)
```

**Tasks**: Inherit from `PipelineTask` (`@attrs.define`). Override `weight` property for load balancing.

**Stages**: Inherit from `CuratorStage`. Implement:

- `resources` property → `CuratorStageResource(cpus=X, gpus=Y)` (fractional GPUs allowed)
- `conda_env_name` property → environment name or `None` for default
- `process_data(task)` → process batch, can return different count or `None` to stop
- Optional `model` property → return `ModelInterface` instance

Stage lifecycle: `stage_setup_on_node()` (once per node) → `stage_setup()` (once per worker, runs in remote actor in target conda env) → `process_data()` (batches of `stage_batch_size`, default 1)

**Models**: Inherit from `ModelInterface`. Implement `conda_env_name`, `model_id_names`, and `setup()`.

**Running**: `run_pipeline(input_tasks, stages)` — accepts bare `CuratorStage` or `CuratorStageSpec`. Use `CuratorStageSpec(MyStage(), num_workers_per_node=N)` for tuning.

See `cosmos_curator/pipelines/examples/hello_world_pipeline.py` and `docs/curator/guides/pipeline-design.md`

## Commits & PRs

**Commits**: Conventional Commits: `git commit -m "fix: description"`. MRs squash on merge, so commit early and often as checkpoints — individual commit messages don't need to be perfect.

**Merge Requests**:

- Create using `glab mr create` targeting branch `nvidia/main`, assign to `@me`, with `--remove-source-branch --squash-before-merge`
- Describe the problem, resulting behavior, and significant design decisions or risks. Mention affected CLIs, recipes,
  or environments when useful to the reviewer; link relevant issues.
- Omit Testing/Validation sections and local command/pass checklists by default. CI reports routine lint, type, and test
  results. Include validation only when it adds evidence CI cannot provide, such as a manual workflow check, a benchmark
  supporting a design choice, or a material gap in CI coverage. Continue to run the required checks above.
- Do NOT include "Generated with Claude Code" in descriptions

**Submodule updates**: Pre-commit warns before committing. Update via `cd cosmos-xenna && git checkout VERSION && cd .. && git add cosmos-xenna`

## Code Reviews

CodeRabbit and Greptile already review GitLab MRs. Use `glab` or the GitLab MCP server to read MR discussions and review
comments, and avoid duplicating findings. With `glab`, start with
`glab mr view <id> --comments`; use `glab api` for additional MR or discussion details.

- Focus agent reviews on design and engineering judgment: appropriate abstractions, clear interfaces, consistency
  across recipes, user experience, unnecessary complexity, and substantive correctness or operational risks.
- Skip trivial style issues and routine mechanical checks covered by the bots or CI. Report actionable issues with
  concrete consequences; a review does not need findings just to fill a checklist.
- Review work does not require rerunning lint, type checks, or tests already covered by CI. Investigate validation only
  when a specific failure or coverage gap matters to the design or correctness of the change.
