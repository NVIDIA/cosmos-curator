# Pixi Environment Refactor Design

## Summary

Cosmos Curator should separate local developer tooling from Linux/GPU runtime environments.

The target contract is:

- The implicit Pixi default feature is a minimal cross-platform base: Python and pip only.
- The `tools` environment is a small cross-platform repo tooling shell that works on Linux and macOS.
- The `dev` environment remains the Linux development shell for type checking, CPU tests, and repo maintenance.
- The `cluster` environment is a Linux operational shell for Slurm commands and GPU-node diagnostics.
- The `default` environment remains the main runtime environment. GPU tests run inside containers in `default` or the
  specific runtime environment required by the test marker.
- Runtime environments remain Linux-only and carry Ray, Cosmos-Xenna, CUDA, vLLM, CVCUDA, Paddle, SeedVR, SAM3, cuML,
  and similar execution dependencies.
- Editable local development does not pull the runtime stack into macOS unless the developer explicitly asks for a
  runtime environment.

This design is about environment boundaries and packaging shape. It does not remove any existing runtime capability.

## Motivation

The current Pixi layout makes runtime dependencies part of the base solve:

- Workspace channels include `rapidsai`, `conda-forge`, and `nvidia`.
- Workspace platforms are Linux-only.
- Top-level PyPI dependencies include `ray` and `cosmos-xenna`.
- The `dev` environment includes runtime features such as `core`, `transformers`, `tracing`, and `profiling`.

That makes `pixi shell -e dev` a Linux runtime environment with developer tools added, rather than a lightweight local
development environment. That is useful for Linux maintainers, but it prevents a clean macOS tooling shell.

There is a second leak through packaging: if `pixi` installs the editable package in a cross-platform environment,
dependencies listed under
`[project].dependencies` in `pyproject.toml` are installed even if they are removed from Pixi's top-level feature. If
`ray` and other runtime dependencies stay in base package metadata, no editable-install tooling environment can be truly
macOS-friendly.

## Goals

- Make `pixi shell -e tools` the recommended cross-platform repo tooling shell on Linux and macOS.
- Keep `pixi shell -e dev` as the Linux development shell for CPU tests and repo maintenance.
- Add `pixi shell -e cluster` for Slurm CLI use and GPU-node diagnostics such as `nvitop` and `nvtop`.
- Keep GPU/env tests in containerized runtime environments, usually `default` and sometimes a model-specific environment.
- Keep runtime and model environments Linux-only unless there is a concrete macOS use case.
- Keep image builds and pipeline execution using Linux runtime environments.
- Keep the existing model/runtime isolation strategy.
- Make dependency ownership visible: cross-platform tooling dependencies in `tools`, Linux CPU-test and development
  dependencies in `dev`, runtime dependencies in runtime features.
- Add structural tests that prevent `ray`, `cosmos-xenna`, CUDA, and model stacks from leaking back into `tools`.

## Non-Goals

- Do not port GPU pipelines or runtime model environments to macOS.
- Do not require the entire CPU test suite to run in the macOS `tools` environment.
- Do not collapse the existing model-specific Pixi environments.
- Do not change pipeline code only to satisfy this refactor unless import behavior blocks the packaging split.

## Terminology

Pixi has two concepts that are easy to conflate:

- **Default feature**: the top-level `[dependencies]` and `[pypi-dependencies]` implicitly included in environments
  unless `no-default-feature = true` is set.
- **`default` environment**: the named runtime environment used by Cosmos Curator for normal pipeline execution.

This design makes the default feature small and cross-platform. The named `default` environment remains a Linux runtime
environment assembled from `core`, `runtime`, `transformers`, `tracing`, and `profiling`.

## Target Feature Boundaries

### Default Feature

Keep the implicit default feature boring and cross-platform:

```toml
[workspace]
channels = ["rapidsai", "conda-forge"]
platforms = ["linux-64", "linux-aarch64", "osx-arm64"]

[dependencies]
python = ">=3.12.13,<3.13"
pip = "*"
```

`rapidsai` remains a workspace channel because Pixi's strict channel priority applies workspace channels ahead of
feature channels. Keeping RAPIDS only on the `cuml` feature causes transitive RAPIDS dependencies such as `dask-cuda` to
be excluded during the `cuml` solve. Treat this as a solver constraint, not a dependency-ownership signal.

Do not put `click`, `ray`, `cosmos-xenna`, CUDA packages, or runtime model dependencies at the top level. CLI
dependencies belong in `tools` and package metadata; runtime dependencies belong in Linux-only runtime features.

### Tools

Use `tools` for cross-platform repo maintenance and the editable CLI:

```bash
pixi shell -e tools
```

The `tools` feature should be cross-platform and include:

- Lint and format tools: `ruff`
- Git hook tooling: `pre-commit`
- Build tooling: `python-build`, `setuptools-scm`
- Editable package install: `cosmos-curator = { path = ".", editable = true }`
- CLI dependencies needed for `cosmos-curator --help` and basic local client commands, including `click`, `typer`,
  `rich`, `requests`, `pyyaml`, `attrs`, `jinja2`, `loguru`, `tomli`, `tqdm`, `fabric`, `invoke`, `psutil`, `ngcsdk`,
  and `huggingface_hub`

`tools` should not promise broad `mypy` or `pytest` coverage until those commands are validated against the macOS
dependency surface. It may expose a narrower test command later, such as client/CLI/package-structure tests.

`tools` replaces the current `dev-hooks` environment. Pre-commit and lightweight CI checks should run through `tools`
instead of maintaining a separate hooks-only environment.

### Dev

Keep `dev` as the Linux development and CPU-test shell:

```bash
pixi shell -e dev
```

The `dev` environment should include:

- `tools`
- `core`
- `cluster`
- `transformers`
- `tracing`
- `profiling`
- `mypy`
- `pytest`, `pytest-mock`, and useful pytest plugins
- `twine`, if still needed locally
- CPU-compatible dependencies needed by the non-`env` test suite
- CPU-test collection/type-check dependencies imported by non-`env` tests, such as FastAPI/Uvicorn, Google GenAI,
  gpustat, safetensors, and WebDataset

`dev` should not include the main heavy GPU/model runtime stack by default. GPU tests and `@pytest.mark.env` tests should
run inside containers in the runtime environment named by the marker.

Because `dev` is Linux-only, it can keep broader repo maintenance behavior than `tools` without blocking macOS
contributors from using the smaller `tools` environment.

In `pixi.toml`, the `dev` feature should contain only the Linux development additions. The `dev` environment composes
that feature with `tools`, `cluster`, slim `core`, `transformers`, and CPU-test support features.

### Cluster

Use `cluster` as a Linux-only operational shell for Slurm commands and interactive GPU-node diagnostics:

```bash
pixi shell -e cluster
```

The `cluster` environment should include:

- `tools`, so the editable `cosmos-curator` CLI is available
- Linux-only diagnostics utilities such as `nvitop` and `nvtop`
- Any lightweight dependencies needed by `cosmos-curator slurm ...` commands

It should not include `core` or the main `runtime` feature by default. Slurm submission and job-management commands
should not require Ray, Cosmos-Xenna, Torch, vLLM, or CUDA runtime libraries at CLI import time. If they do, prefer lazy
imports in the CLI path over adding the runtime stack to `cluster`.

The `dev` environment should include the `cluster` feature so Linux developers get the same node-diagnostic tools while
working locally.

### Core Runtime

Keep `core` as a slim Linux runtime base that can be shared by multiple environments. This is where Ray and Cosmos-Xenna
should live, along with common lightweight dependencies required by runtime wiring and CPU tests:

```toml
[feature.core]
platforms = ["linux-64", "linux-aarch64"]

[feature.core.dependencies]
fastapi = "*"
starlette = "<1"
uvicorn = "*"
websockets = "<16"

[feature.core.pypi-dependencies]
cosmos-xenna = "==0.4.3"
ray = { version = "==2.55.1", extras = ["default", "data"] }
google-genai = ">=1.59.0"
webdataset = "*"
```

Because the shared `gputest` task collects pipeline and model tests before marker filtering, `core` should also carry
lightweight import dependencies that are needed across runtime-like environments, such as FastAPI/Starlette/Uvicorn for
Ray Serve import paths, WebSockets for `google-genai`, and WebDataset helpers. These are common wiring dependencies, not
the main video/model stack.

`core` should not contain the main video/model stack. Keep the default-runtime copies of Torch, vLLM, CVCUDA,
PyNvVideoCodec, Paddle, AV/OpenCV, cuML, SeedVR, SAM3, and model-specific dependencies in dedicated runtime/model
features. `dev` may still add CPU-compatible test dependencies when the non-`env` test suite needs them.

`core` should not require the `nvidia` channel. The workspace-level `rapidsai` channel exists only so the separate
`cuml` environment can solve correctly under Pixi's strict channel priority.

### Runtime

Use `runtime` for the main heavy runtime stack used by the named `default` environment. This feature replaces the current
`unified` role and should carry default video/model runtime dependencies such as:

- Torch and TorchVision
- AV/OpenCV/video dependencies
- vLLM, CVCUDA, and PyNvVideoCodec
- CUDA library packages and NVIDIA-channel packages needed by the main runtime
- Default-runtime dependencies that are too heavy or platform-specific for `core`

The `runtime` feature is Linux-only and should not be included in `tools` or `dev` by default.

### cuML

Keep the direct cuML dependencies isolated to the Linux-only `cuml` feature:

```toml
[feature.cuml]
platforms = ["linux-64", "linux-aarch64"]
channels = ["nvidia"]

[feature.cuml.dependencies]
cuml = "==26.02"
raft-dask = "*"
```

The direct cuML dependencies remain isolated to this feature even though the `rapidsai` channel must stay at workspace
priority for solver reasons.

### Runtime And Model Features

Keep the existing runtime isolation strategy:

- `transformers`
- `legacy-transformers`
- `runtime`
- `tracing`
- `profiling`
- `cuml`
- `paddle-ocr`
- `seedvr`
- `sam3`

These features should be Linux-only unless there is a concrete local developer use case for macOS. Features that do not
include `core`, such as `cuml`, need their own platform constraints if they carry Linux-only dependencies.

`model-download` should remain an environment name, but it should not have a duplicate dependency feature. Model download
work is scheduled through Ray actors, so the environment needs the `ray` dependency from `core` while still avoiding the
heavy `runtime` stack.

`legacy-transformers` is a model-specific runtime stack for Cosmos-Embed1 and InternVideo2. It should include its own
Torch, AV/OpenCV, and `timm` dependencies, but it should not include the main `runtime` feature because `runtime` carries
Cosmos3/vLLM packages that require a newer Transformers version.

## Target Environments

The environment table should move toward this shape:

```toml
[environments]
tools = ["tools"]
cluster = ["tools", "cluster"]
dev = ["tools", "cluster", "core", "transformers", "tracing", "profiling", "dev"]
default = ["core", "runtime", "transformers", "tracing", "profiling"]
cuml = ["core", "cuml", "tracing"]
legacy-transformers = ["core", "legacy-transformers", "tracing", "profiling"]
model-download = ["core", "tracing"]
paddle-ocr = ["core", "paddle-ocr", "tracing", "profiling"]
seedvr = ["core", "runtime", "transformers", "seedvr", "tracing", "profiling"]
sam3 = ["core", "runtime", "sam3", "tracing", "profiling"]
```

This relies on the implicit default feature being minimal. If that invariant changes, `tools` must use
`no-default-feature = true` to avoid inheriting runtime dependencies.

## Packaging Impact

Pixi cleanup alone is not enough because `tools` installs the package editable.

The package metadata should follow the same boundary:

- `[project].dependencies`: minimal package and CLI dependencies that should work on macOS.
- No runtime extra unless pip-based runtime installation becomes a supported and tested user flow.
- Runtime, model, and GPU dependency stacks stay managed by Pixi runtime features and Docker images.

`pyproject.toml` should not duplicate the Pixi runtime dependency graph. In particular, avoid adding a
`[project.optional-dependencies].runtime` extra just to move Ray and Cosmos-Xenna out of base dependencies. That would
create a second runtime contract that does not cover the real execution environment: CUDA, channels, system
requirements, vLLM, CVCUDA, Torch pins, Paddle, cuML, model-specific dependencies, and platform constraints.

Pixi should be the runtime source of truth:

- `tools` installs the editable package and relies on `pyproject.toml` for base package dependencies.
- `core` owns Ray, Cosmos-Xenna, and common runtime wiring dependencies.
- Runtime/model features own their isolated dependency stacks.
- Docker/image builds consume Pixi environments, not Python package extras.

Some duplication can still be justified when Pixi must pin or prefer a package for security, channel, wheel, or platform
reasons, but duplication should be intentional and local to the environment that needs it.

The important contract is that base editable install does not pull runtime dependencies into `tools`.

Any package imported at CLI startup must either remain in base dependencies or become a lazy optional import. The
validation command `pixi run -e tools cosmos-curator --help` should catch missing import-time dependencies.

## Test Strategy

The broad CPU test suite is not currently lightweight. Some tests import runtime-heavy packages during collection or
execution, including `ray`, `cosmos_xenna`, `torch`, `av`, `pyarrow`, and vLLM-adjacent modules.

The target test split is:

- Keep `tools` macOS-friendly and limit it to lint, hooks, build, CLI smoke checks, and possibly a narrow client test
  task.
- Run CPU tests in `dev`.
- Run GPU tests and `@pytest.mark.env` tests inside containers in the runtime environment named by the marker.

If broad GPU/env test collection imports lightweight packages that every runtime-like environment needs to import the
shared test tree, add those CPU-compatible dependencies to `core`. If CPU test collection imports GPU/runtime-heavy
packages, either add CPU-compatible test dependencies to `dev` or mark and scope those tests so they run in a runtime
environment. Do not add the main GPU/model runtime stack to `dev` only to support GPU test collection.

Structural tests should enforce the new dependency contract:

- Workspace platforms include `osx-arm64`.
- Workspace channels are `rapidsai` and `conda-forge`, with no `nvidia`.
- `ray` and `cosmos-xenna` are not top-level/default dependencies.
- `core` owns lightweight shared import dependencies required by broad GPU/env test collection.
- `tools` includes CLI and tooling dependencies.
- `tools` does not include runtime features.
- `cluster` is Linux-only, includes node diagnostics, and does not include `core` or `runtime`.
- `dev` remains Linux-only and can run CPU tests without the main `runtime` feature.
- GPU/env tests run inside containers through the marker-selected runtime environment.
- `dev-hooks` is not defined.
- Runtime environments and Linux-only features do not include macOS.
- Existing `nvitop`/`nvtop` assertions move to `cluster`; `dev` gets them by composing `cluster`.

## Migration Plan

1. Move workspace platforms to include macOS:

   ```toml
   platforms = ["linux-64", "linux-aarch64", "osx-arm64"]
   ```

2. Reduce top-level dependencies to `python` and `pip`.
3. Move top-level `ray` and `cosmos-xenna` into slim Linux-only `core`.
4. Move heavy video/model runtime dependencies from `core`/`unified` into a Linux-only `runtime` feature.
5. Move `nvidia` off the workspace and into Linux/runtime features that need NVIDIA packages.
6. Keep `rapidsai` on the workspace channel list for Pixi strict-channel-priority compatibility, but keep direct RAPIDS
   dependencies scoped to `cuml`.
7. Add a cross-platform `tools` feature/environment with local tooling plus editable CLI dependencies.
8. Add a Linux-only `cluster` feature/environment for Slurm CLI use and GPU-node diagnostics.
9. Split `pyproject.toml` dependencies so editable install does not pull runtime/container dependencies into `tools`.
10. Keep `dev` as the Linux CPU-test and repo maintenance shell, composed from `tools`, `cluster`, slim `core`, and
    Linux-only development features.
11. Remove `dev-hooks` and point pre-commit/CI lightweight checks at `tools`.
12. Update structural tests for the new contract.

## Validation

Run these after the refactor on Linux:

```bash
pixi lock
pixi info
pixi run -e tools ruff --version
pixi run -e tools pre-commit --version
pixi run -e tools python -m build --version
pixi run -e tools cosmos-curator --help
pixi run -e cluster cosmos-curator slurm --help
pixi run -e cluster nvitop --version
pixi run -e dev mypy --version
pixi run -e dev pytest
pixi run -e default python -m cosmos_curator.pipelines.examples.hello_world_pipeline
```

Run GPU/env tests inside a container. The `gputest` task currently scopes default-env GPU tests:

```bash
cosmos-curator local launch --curator-path . -- pixi run --as-is -e default gputest
```

For tests marked with another environment, run the corresponding Pixi environment instead.

On macOS, validate at least:

```bash
pixi install -e tools
pixi run -e tools ruff --version
pixi run -e tools pre-commit --version
pixi run -e tools python -m build --version
pixi run -e tools cosmos-curator --help
```

## Open Questions

- Should `tools` expose a scoped lightweight test command, or stay limited to lint/hooks/build/CLI smoke checks?
- What exact base package dependency set is required for `pixi run -e tools cosmos-curator --help` and basic client
  commands?
- Which CPU-compatible dependencies must stay in `dev` so non-`env` tests run without pulling in `runtime`?
- Which Slurm CLI imports need to become lazy so `cosmos-curator slurm --help` works in `cluster` without `core` or
  `runtime`?
