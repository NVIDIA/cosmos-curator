# Cosmos Curator - Developer Guide

- [Cosmos Curator - Developer Guide](#cosmos-curator---developer-guide)
  - [Development Environment Setup](#development-environment-setup)
    - [Working with Pixi Environments](#working-with-pixi-environments)
      - [Installing Pixi](#installing-pixi)
      - [Adding New Dependencies](#adding-new-dependencies)
      - [Updating Pixi Environments](#updating-pixi-environments)
      - [Viewing Available Environments](#viewing-available-environments)
      - [Viewing Packages in Pixi Environments](#viewing-packages-in-pixi-environments)
      - [Running Commands in Pixi Environments](#running-commands-in-pixi-environments)
    - [Interactive Slurm Development](#interactive-slurm-development)
  - [Code Quality Checks](#code-quality-checks)
  - [Building the Client package](#building-the-client-package)
  - [Testing](#testing)
    - [Unit Tests](#unit-tests)
    - [Model and Stage Tests](#model-and-stage-tests)
    - [End-to-End Pipeline Tests](#end-to-end-pipeline-tests)
  - [Pipeline CLI settings (attrs)](#pipeline-cli-settings-attrs)
  - [Best Practices](#best-practices)
  - [Environment Variables](#environment-variables)
  - [Contributing](#contributing)
  - [Troubleshooting](#troubleshooting)
  - [Responsible Use of AI Models](#responsible-use-of-ai-models)
  - [Support](#support)

## Development Environment Setup

Please refer to the following section in [End User Guide](./client/end-user-guide.md):
- [Prerequisites](./client/end-user-guide.md#prerequisites) for hardware and software requirements.
- [Initial Setup](./client/end-user-guide.md#initial-setup) for preparaing configurations files and workspace directories, etc.
- [Setup Environment and Install Dependencies](./client/end-user-guide.md#setup-environment-and-install-dependencies) for setting up Cosmos Curator.

For developers to contribute back to the repo, the following additional steps are needed:

To help ease with setup, from within your virtual environment you can run
```bash
./devset.sh
```

Alternatively, you should perform these additional steps.
```bash
# Install mypy types
mypy --install-types

# Set up pre-commit hooks
pre-commit install
```

### Working with Pixi Environments

Cosmos Curator uses [Pixi](https://pixi.sh) to manage Python environments inside Docker images. As a developer, you may
need to inspect or work with these environments locally.

The `pixi.toml` file at the repository root defines all Pixi environments used by Cosmos Curator.

#### Installing Pixi

Follow the [Pixi installation instructions](https://pixi.sh/latest/installation/), or run the following commands:

```bash
# Assuming your current user has write access to /usr/local/bin.
# If not, you can change ownership of /usr/local to your user: sudo chown -R $USER:$USER /usr/local
wget -qO- https://pixi.sh/install.sh | PIXI_HOME=/usr/local PIXI_NO_PATH_UPDATE=1 sh
```

This installs Pixi to `/usr/local/bin/pixi`, which matches the location in the Docker image.
You can verify the installation by running `pixi --version`.

#### Adding New Dependencies

When adding new dependencies to a pixi environment:

1. Edit the `pixi.toml` file to add your dependency under the appropriate environment
2. Run `pixi lock` to resolve the new dependency
3. Rebuild the Docker image
4. Test your changes inside the Docker container to ensure compatibility
5. Commit both `pixi.toml` and the updated `pixi.lock` file to version control to ensure reproducibility

Note: Environment names in pixi use hyphens (e.g., `model-download`) rather than underscores, as per pixi conventions.

#### Updating Pixi Environments

To update pixi environments:

1. Run `pixi update` to update the environments
2. Rebuild the Docker image
3. Test your changes inside the Docker container to ensure compatibility

#### Viewing Available Environments

To see all available pixi environments defined in `pixi.toml`:

```bash
# Show detailed information about all environments
pixi info

# Or list the environments only
pixi workspace environment list
```

#### Viewing Packages in Pixi Environments

To see all the packages and their versions in a pixi environment:

```bash
# For the default environment
pixi list

# For a specific environment
pixi list -e <env-name>

# For a specific package
pixi list -e <env-name> <package-name>

# Example: see the pytorch packages in the 'unified' environment
pixi list -e unified pytorch
```

#### Running Commands in Pixi Environments

While most pipeline execution happens inside Docker containers, you may need to run commands in pixi environments:

```bash
# Run in the default environment
pixi run python -m <module>

# Run in a specific environment
pixi run -e <env-name> python -m <module>

# Example: check if PyTorch is CUDA enabled in the unified environment
pixi run -e unified python -c "import torch; print(torch.cuda.is_available())"
```

Note: For pipeline execution, always use the Docker container as shown in the testing section.

### Interactive Slurm Development

For GPU development on a Slurm compute node, use the
[Interactive Slurm Development Guide](./curator/guides/slurm-interactive.md). It covers starting the slim container
from an interactive allocation, using a live Lustre checkout, running a pipeline command, and troubleshooting common
environment issues.

## Code Quality Checks

This project uses the following development tools:
1. **ruff**: For code formatting and linting
2. **mypy**: For static type checking
3. **poetry**: For dependency management

Before submitting any changes, format Python files and run the checks from the repository root:

```bash
pixi run format
pixi run lint
```

## Building the Client package
   - The `cosmos-curator` client can be built as a wheel and installed in a standalone mode, without the need for the rest of the source environment
```bash
poetry build
pip3 install dist/cosmos_curator*.whl
```

## Testing

Tests under the [tests/](../tests/) directory can be categorized into 3 levels:
- Unit tests: for testing critical/complex function which are typically CPU-only and can run in default conda environment.
- Model/stage tests: for testing functional correctness of a model, a pipeline stage, and a combination of a few stages, which typically require GPU and should run inside the container.
- End-to-end pipeline tests: for testing the functionality of reference pipelines.

### Unit Tests

Run the CPU-only pytest suite from the repository root:

```bash
pixi run cputest

================ test session starts ================
configfile: pytest.ini
testpaths: tests
collected 155 items / 26 deselected / 129 selected

tests/client/slurm_cli/test_slurm.py .....................................                      [ 36%]
tests/client/slurm_cli/test_start_ray.py ...................................                    [ 70%]
tests/cosmos_curator/pipelines/video/filtering/motion/test_motion_filter.py .                    [ 71%]
tests/cosmos_curator/pipelines/video/utils/test_decoder_utils.py .............................   [100%]
================ 102 passed, 6 deselected, 2 warnings in 2.98s ================
```

### Model and Stage Tests

Launch the docker container locally and simply run `pytest` command:

```bash
for conda_env in default legacy-transformers unified; do
   cosmos-curator local launch --image-name cosmos-curator --image-tag 1.0.0 --curator-path . \
   -- pixi run --as-is -e $conda_env pytest -m env tests/cosmos_curator/{pipelines,models}/;
done

================ test session starts ================
configfile: pytest.ini
collected 58 items / 40 deselected / 18 selected

tests/cosmos_curator/pipelines/video/clipping/test_fixed_stride_extraction.py ........           [ 44%]
tests/cosmos_curator/pipelines/video/clipping/test_transnetv2_extraction.py .....                [ 72%]
tests/cosmos_curator/pipelines/video/filtering/motion/test_motion_filter.py .                    [100%]
================ 18 passed, 40 deselected, 3 warning in 14.44 ================

...

================ test session starts ================
configfile: pytest.ini
collected 58 items / 52 deselected / 6 selected

tests/cosmos_curator/pipelines/video/captioning/test_t5_embedding.py .                           [ 16%]
tests/cosmos_curator/pipelines/video/filtering/aesthetics/test_aesthetic_filter.py .....         [100%]
================ 6 passed, 52 deselected, 2 warning in 30.02 ================
```

### End-to-End Pipeline Tests

Run the reference video pipeline based on instructions in [Run the Split-Annotate Pipeline](./client/end-user-guide.md#run-the-reference-video-pipeline) section to make sure everything works.

The CI will test more scenarios.

## Pipeline CLI settings (attrs)


**Where it lives**

- Shared flags (S3 profiles, execution mode, limit, model weights, profiling): [`cosmos_curator/pipelines/common_pipeline_settings.py`](../cosmos_curator/pipelines/common_pipeline_settings.py) — class `CommonPipelineSettings`, helpers `cli()`, `add_settings_cli_arguments`, and `CommonPipelineSettings.from_namespace`.
- Shard-only flags: [`cosmos_curator/pipelines/video/shard_pipeline_settings.py`](../cosmos_curator/pipelines/video/shard_pipeline_settings.py) — class `ShardPipelineSettings`, `add_shard_args`.
- Parser registration used by multiple entry points: [`cosmos_curator/pipelines/pipeline_args.py`](../cosmos_curator/pipelines/pipeline_args.py) (`add_common_args`, `add_profiling_args`).

**How to add a new flag (shard example)**

1. **Add an attrs field** on the right class (`ShardPipelineSettings` for shard-only, `CommonPipelineSettings` only if every consumer should see the flag).
2. Put **validation** on the field with `validator=` (for example `validators.ge(1)`, `validators.in_(choices)`, `validators.min_len(1)`). This runs whenever the settings instance is constructed (CLI, NVCF JSON → `Namespace`, and any code path that builds the class explicitly).
3. Attach CLI metadata with **`metadata=cli(...)`** on the same field. Typical keys are `help=`, `default=`, and when needed `arg_type=int` / `float`, `choices=frozenset(...)`, `required=True`, `action=` or `action=argparse.BooleanOptionalAction`, or a custom `flag="--my-name"`. The field name becomes argparse **`dest`** unless you override the flag string only (the dest stays the field name).
4. **Register** flags by calling `add_settings_cli_arguments(parser, YourSettingsClass)` (or the thin wrappers `add_shard_args` / `add_common_args`). Do not add a second manual `add_argument` for the same option.
5. **Build settings** at the pipeline entry point from the parsed `Namespace` (see `shard()` in [`sharding_pipeline.py`](../cosmos_curator/pipelines/video/sharding_pipeline.py): `CommonPipelineSettings.from_namespace(args)` plus shard fields keyed by field name). Pipeline logic should take **`ShardPipelineSettings`** (or the relevant settings type), not raw `args`, for the refactored paths.
6. **attrs field order**: optional fields with defaults cannot appear before required fields. Keep pipeline-specific constraints in mind (see existing comments on `ShardPipelineSettings`).

**NVCF and Slurm**

- **NVCF** builds `argparse.Namespace(**args)` from JSON. Keys must use Python **dest** names (`input_clip_path`, not `--input-clip-path`). Values are **not** re-parsed through argparse’s `type=` / `choices=`, but **attrs validators still run** when settings are constructed. Put every constraint you need for cloud invokes on **`validator=`** (and keep `choices=` in `cli()` aligned for CLI users).
- **Slurm** usually runs the same `python -m ... run_pipeline ...` CLI as local development, so argparse and attrs both apply.

**Tests**

- [`tests/cosmos_curator/pipelines/video/test_shard_pipeline_settings.py`](../tests/cosmos_curator/pipelines/video/test_shard_pipeline_settings.py) checks parser `dest` sets against settings fields and exercises attrs validation after parse.

Other large pipelines may still use legacy patterns until migrated; follow the existing module when changing them.

## Best Practices

1. **Virtual Environment**:
   - Always work in a virtual environment
   - Avoid using `$HOME/.local` for Python packages

2. **Dependencies**:
   - Maintain the `pyproject.toml` file
   - Document any new dependencies
   - Use `poetry install --extras=local` to install packages in your virtual environment
   > Note: You may need to do the following : `export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring` before running `poetry` to avoid getting stuck due to a keyring pop-up when you are running in a headless display environment
   > Note: From within your virtual environment, you may execute `./devset.sh` to complete inital setup of environment.

3. **Code Quality**:
   - Write clean, well-documented code
   - Use ruff for formatting and linting
   - Ensure type hints are properly used and checked with mypy
   - Write meaningful commit messages

4. **Testing**:
   - Write tests for new features
   - Run existing tests before submitting changes
   - Ensure all tests pass

5. **Documentation**:
   - Update documentation when adding new features
   - Keep the README files up to date
   - Document any API changes

## Environment Variables

Cosmos Curator uses several `cosmos_curator_*` environment variables for
internal communication between the driver process and Ray workers.
Most are set automatically by the framework; developers rarely need to
set them manually.

| Variable | Set by | Purpose |
|---|---|---|
| `COSMOS_CURATOR_ARTIFACTS_STAGING_DIR` | `ArtifactDelivery.create()` | Shared base directory for artifact staging on each node.  Workers inherit this from the driver so all processes agree on the same temp path.  Subdirectories (`profiling/`, `traces/`) isolate different artifact kinds. |
| `COSMOS_CURATOR_TRACE_DIR` | `enable_tracing()` | Per-process directory for OTel span files.  Defaults to `<staging>/traces/` when `COSMOS_CURATOR_ARTIFACTS_STAGING_DIR` is set, otherwise `/tmp/cosmos_curator_traces`. |
| `COSMOS_CURATOR_TRACEPARENT` | `propagate_trace_context()` | W3C traceparent header propagated from the driver's root span so all stage spans join a single distributed trace. |
| `COSMOS_CURATOR_LOCAL_WORKSPACE_PREFIX` | User (optional) | Override the home-directory prefix for the local workspace (`~/cosmos_curator_local_workspace`).  See [End User Guide](./client/end-user-guide.md). |
| `COSMOS_CURATOR_DOCKER_BUILD_ULIMIT` | User (optional) | Custom `nofile` ulimit for Docker image builds (default 65536). |

For details on the artifact delivery mechanism and why env vars are
used, see the [Artifact Transport Guide](./curator/reference/artifact-transport.md#environment-variables).

## Contributing

1. Create a new branch for your feature/fix
2. Make your changes
3. Run all code quality checks
4. Submit a pull request with a clear description of changes

## Troubleshooting

If you encounter issues during development:

1. **Environment Issues**:
   - Ensure you're using the correct Python version
   - Verify all dependencies are installed
   - Check virtual environment activation

2. **Build Issues**:
   - Clear any cached files
   - Rebuild the environment if necessary
   - Check for conflicting dependencies

3. **Pipeline Stage Debugging**:
   - Use [Stage Replay](./curator/guides/stage-replay.md) to debug specific stages without re-running entire pipelines
   - Save task inputs from problematic stages and replay them in isolation
   - Iterate rapidly on stage logic by replaying saved tasks

## Responsible Use of AI Models
[Responsible Use](../RESPONSIBLE_USE.md)

## Support

For development-related questions or issues:
- Create an issue in the repository
- Contact the development team
- Check existing documentation and issues 
