# Slurm Interactive Slim Image Guide

This guide describes the quick-development workflow for running Cosmos Curator interactively on a Slurm compute node with a slim container image.

The goal is to avoid rebuilding the image when changing Python code or Pixi environments. The slim image provides the base system and Pixi binary; your Lustre checkout provides the live source tree, `pixi.toml`, `pixi.lock`, and `.pixi` environments.

## Assumptions

- You have already built the slim image.
- You have already imported it with `enroot import`.
- You have copied the resulting `.sqsh` file to shared storage visible from the Slurm compute node.
- You are running inside an interactive Slurm allocation.
- You have a Cosmos Curator repo checkout on Lustre.

## Repo Location

`${REPO}` is the Cosmos Curator repo root. It should not be just the `cosmos_curator` Python package directory.
It should contain:

```text
${REPO}/cosmos_curator
${REPO}/pixi.toml
${REPO}/pixi.lock
${REPO}/.pixi
```

Use a path like:

```bash
USER_DIR=/path/to/${USER}
REPO=${USER_DIR}/src/cosmos-curator
mkdir -p "${REPO}/.pixi"
```

When you are intentionally changing dependencies, edit `pixi.toml` and update `pixi.lock` in this same Lustre checkout.
After the lockfile is updated, reinstall the changed environment from `${REPO}`.

## Start the Container

Set paths for the cluster filesystem:

```bash
USER_DIR=/path/to/${USER}
REPO=${USER_DIR}/src/cosmos-curator
CONTAINER_NAME=cosmos-curator+1.0.0-slim
CONTAINER_DIR=${USER_DIR}/container_images
WORKSPACE=${USER_DIR}/cosmos_curator_local_workspace
CONFIG=${HOME}/.config/cosmos_curator/config.yaml
CACHE_DIR=${USER_DIR}/cache/cosmos-curator
```

Create the workspace and cache directories if needed:

```bash
mkdir -p \
  "${WORKSPACE}" \
  "${CACHE_DIR}/pixi" \
  "${CACHE_DIR}/xdg" \
  "${CACHE_DIR}/uv" \
  "${CACHE_DIR}/pip"
```

Create the enroot container from the squashfs image if it does not exist:

```bash
if ! enroot list | grep -q "${CONTAINER_NAME}"; then
  enroot create --name "${CONTAINER_NAME}" "${CONTAINER_DIR}/${CONTAINER_NAME}.sqsh"
fi
```

Start the container. The repo is mounted twice:

- `/opt/cosmos-curator` is the path Cosmos Curator code expects inside the container.
- `${REPO}` preserves the original Lustre path, which lets scripts inside `.pixi/envs/*/bin` resolve shebangs that point
  back to the path where the environment was created.

```bash
enroot start -w \
  -m "${HOME}/.aws/credentials":/creds/s3_creds \
  -m "${WORKSPACE}":/config \
  -m "${CONFIG}":/cosmos_curator/config/cosmos_curator.yaml \
  -m "${REPO}":/opt/cosmos-curator \
  -m "${REPO}":"${REPO}" \
  -m "${CACHE_DIR}":/cache \
  -e CONDA_OVERRIDE_CUDA=13.0.2 \
  -e PIXI_CACHE_DIR=/cache/pixi \
  -e XDG_CACHE_HOME=/cache/xdg \
  -e UV_CACHE_DIR=/cache/uv \
  -e PIP_CACHE_DIR=/cache/pip \
  -e SLURM_JOBID \
  -e SLURM_NNODES \
  -e SLURM_NTASKS_PER_NODE \
  "${CONTAINER_NAME}" /bin/bash
```

Notes:

- `-w` gives the container a writable root filesystem. It is useful for caches and temporary files even though `.pixi`
  is mounted from Lustre in this quick-development flow.
- `PIXI_CACHE_DIR` moves Pixi's package cache off the small home filesystem. `XDG_CACHE_HOME`, `UV_CACHE_DIR`, and
  `PIP_CACHE_DIR` catch other installer caches that would otherwise default to `${HOME}/.cache`.
- The config mount path is `/cosmos_curator/config/cosmos_curator.yaml`.
- If you need Azure credentials, add a mount such as `-m /path/to/azure_creds:/creds/azure_creds`.

## Verify the Mounts

Inside the container:

```bash
cd /opt/cosmos-curator
ls pixi.toml pixi.lock
ls .pixi/envs
pixi info --extended | grep "Cache dir"
```

If `.pixi/envs` is missing, install the required environments from inside the container:

```bash
pixi install --frozen -e default -e unified -e transformers -e cuml -e legacy-transformers -e model-download -e paddle-ocr -e seedvr
```

For active dependency development, omit `--frozen` only when you expect Pixi to update the lockfile. Use `--frozen` when
you want the environment to match the existing `pixi.lock`.

## Run the Pipeline

Set the runtime environment variables:

```bash
export COSMOS_S3_PROFILE_PATH=/creds/s3_creds
export COSMOS_AZURE_PROFILE_PATH=/creds/azure_creds
export COSMOS_CURATOR_RAY_SLURM_JOB=True
export HEAD_NODE_PORT=$(expr 10000 + $(echo -n "${SLURM_JOBID}" | tail -c 4))
export WORLD_SIZE=$((${SLURM_NNODES} * ${SLURM_NTASKS_PER_NODE}))
export HEAD_NODE_ADDR=${HEAD_NODE_ADDR:-$(hostname)}
export PRIMARY_NODE_HOSTNAME=${PRIMARY_NODE_HOSTNAME:-${HEAD_NODE_ADDR}}
export PRIMARY_NODE_PORT=${HEAD_NODE_PORT}
export RAY_STOP_RETRIES_AFTER=10
export TQDM_MININTERVAL=9000
```

This uses the current container hostname as the Ray head node, which is the expected quick-development case when you
start the container on the node that should run the Ray head. For a multi-node launch where some containers start on
worker nodes, pass the intended head node explicitly with `-e HEAD_NODE_ADDR=<node>` and
`-e PRIMARY_NODE_HOSTNAME=<node>`.

Then launch through Pixi:

```bash
pixi run --as-is python3 -m cosmos_curator.pipelines.video.run_pipeline split \
  --input-video-path "s3://my-bucket/input-videos" \
  --output-clip-path "/config/output-qwen-test" \
  --captioning-algorithm "qwen" \
  --limit 10 \
  --verbose
```

Use `--as-is` for runtime commands. It skips Pixi environment validation and assumes the environment is already
installed. If an environment is missing, the command fails immediately instead of trying to install packages while Ray
workers are starting.

## Alternative: Shared Cache, Local `.pixi`

For longer runs or performance comparisons, mounting `.pixi` from Lustre may be slower than installing the environment
into the container writable overlay. In that mode, mount the source, `pixi.toml`, `pixi.lock`, and a shared Pixi cache,
but do not mount `${REPO}/.pixi`.

Example additional setup:

```bash
CACHE_DIR=${USER_DIR}/cache/cosmos-curator
mkdir -p "${CACHE_DIR}/pixi"
```

Example mounts and environment:

```bash
-m "${REPO}/cosmos_curator":/opt/cosmos-curator/cosmos_curator
-m "${REPO}/pixi.toml":/opt/cosmos-curator/pixi.toml
-m "${REPO}/pixi.lock":/opt/cosmos-curator/pixi.lock
-m "${CACHE_DIR}/pixi":/pixi-cache
-e PIXI_CACHE_DIR=/pixi-cache
```

Then run `pixi install --frozen ...` from `/opt/cosmos-curator` inside the container. This writes `.pixi` into the
container filesystem while reusing package downloads from `/pixi-cache`.

## Troubleshooting

If Pixi env scripts fail with a missing interpreter path, verify that the original Lustre repo path is visible inside
the container:

```bash
ls -la "${REPO}/.pixi/envs"
```

If `pixi run --as-is` fails with a missing environment, run the install step again from `/opt/cosmos-curator`:

```bash
pixi install --frozen -e default
pixi install --frozen -e unified
```

If Cosmos Curator cannot find the config file, verify the mount target:

```bash
ls -la /cosmos_curator/config/cosmos_curator.yaml
```
