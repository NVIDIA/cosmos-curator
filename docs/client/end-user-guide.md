# Cosmos Curator - End User Guide

- [Cosmos Curator - End User Guide](#cosmos-curator---end-user-guide)
  - [Overview](#overview)
  - [Prerequisites](#prerequisites)
      - [Hardware Requirement](#hardware-requirement)
      - [Software Requirement](#software-requirement)
      - [Additional Requirement](#additional-requirement)
  - [Initial Setup](#initial-setup)
  - [Quick Start for Local Run](#quick-start-for-local-run)
    - [Setup Environment and Install Dependencies](#setup-environment-and-install-dependencies)
    - [Run the Hello-World Example Pipeline](#run-the-hello-world-example-pipeline)
    - [Run the Reference Video Pipeline](#run-the-reference-video-pipeline)
    - [Use Gemini for Captioning](#use-gemini-for-captioning)
    - [Enhance Captions with OpenAI](#enhance-captions-with-openai)
    - [Use an OpenAI-Compatible Endpoint for Embedding](#use-an-openai-compatible-endpoint-for-embedding)
    - [Generate Dataset for Cosmos-Predict2 Post-Training](#generate-dataset-for-cosmos-predict2-post-training)
    - [Useful Options for Local Run](#useful-options-for-local-run)
  - [Launch Pipelines on Slurm](#launch-pipelines-on-slurm)
    - [Prerequisites for Slurm Run](#prerequisites-for-slurm-run)
      - [Setup Password-less SSH to the Cluster](#setup-password-less-ssh-to-the-cluster)
      - [Identify User Path on the Cluster](#identify-user-path-on-the-cluster)
    - [Copy Config File, Cloud Storage Credentials, and Model Files to Cluster](#copy-config-file-cloud-storage-credentials-and-model-files-to-cluster)
    - [Create sqsh Image and Copy to the Slurm Cluster](#create-sqsh-image-and-copy-to-the-slurm-cluster)
    - [Launch on Slurm](#launch-on-slurm)
    - [Find Logs](#find-logs)
    - [Processing Large Video Sets in Batches](#processing-large-video-sets-in-batches)
    - [Developing on Slurm](#developing-on-slurm)
      - [Interactive Slurm shell](#interactive-slurm-shell)
    - [Speeding up Model Load on Slurm](#speeding-up-model-load-on-slurm)
  - [Launch Pipelines on NVIDIA DGX Cloud](#launch-pipelines-on-nvidia-dgx-cloud)
  - [Launch Pipelines on K8s Cluster (coming soon)](#launch-pipelines-on-k8s-cluster-coming-soon)
  - [Observability for Pipelines](#observability-for-pipelines)
  - [Build the Client package](#build-the-client-package)
  - [Troubleshooting](#troubleshooting)
  - [Support](#support)
  - [Responsible Use of AI Models](#responsible-use-of-ai-models)

## Overview
Cosmos Curator is a powerful tool for video curation and processing. This guide will help you get started with using the application.

## Prerequisites
#### Hardware Requirement
- Minimum 32GB host memory
- Minimum 200GB disk space
- One or more GPU with
  - minimum CUDA compute capability of 8.0
  - minimum memory of
    - 4GB, to run the hello-world pipeline
    - 48GB, to run the reference video pipelines

#### Software Requirement
- Ubuntu >= 22.04
- Python >=3.12 on your host
  - We will require a specific version in your virtual environment below.
- [Docker](https://docs.docker.com/engine/install/) with [BuildKit (buildx)](https://docs.docker.com/go/buildx/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
  - **Note:** Install latest NVIDIA Container Toolkit 

**Note:** The Docker daemon needs to be restarted after the installation of NVIDIA Container Toolkit as described [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#configuring-docker).

#### Additional Requirement
- [Hugging Face account and access token](https://huggingface.co/settings/tokens) (a read token should suffice for accessing the InternVideo2 model)
- [NGC API key](https://docs.nvidia.com/ngc/gpu-cloud/ngc-private-registry-user-guide/index.html#generating-api-key) for the NGC Registry to access the [NVIDIA CUDA container image](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/cuda) (your API key should at least have catalog permission)
- Cloud storage access credentials if using cloud storage
  - full support for S3-compatible object storage
  - basic support for Azure blob storage

## Initial Setup
This guide covers launching the video curator on multiple platforms (local, DGXC, and slurm). The steps below are platform‑agnostic; complete them on whichever platform you plan to run the video curator on.

1. Create a configuration file at `~/.config/cosmos_curator/config.yaml` and put your credentials. The Hugging Face section is required for model downloads; the Gemini and OpenAI sections are optional but needed for their respective features:

```yaml
huggingface:
    user: "<your-username>"
    api_key: "<your-huggingface-token>"
gemini:
    api_key: "<your-gemini-api-key>"
openai:
    caption:
        api_key: "<your-openai-api-key>"
        base_url: "https://<optional-base-url>/v1"
    enhance:
        api_key: "<your-openai-api-key>"
        base_url: "https://<optional-base-url>/v1"
    embedding:
        api_key: "<api-key-or-dummy-for-local>"
        base_url: "http://<embedding-endpoint>/v1"
```

1. To use `InternVideo2` embedding model:
   - Visit [InternVideo2 Hugging Face page](https://huggingface.co/OpenGVLab/InternVideo2-Stage2_1B-224p-f4/tree/main)
   - Log in to your Hugging Face account
   - Click "agree" to accept the model terms; this is required before you can download this model using HuggingFace API with your token

1. By default, `~/cosmos_curator_local_workspace/` is used as the local workspace for model weights and temporary files at runtime. To configure its location, set environment variable `COSMOS_CURATOR_LOCAL_WORKSPACE_PREFIX` to move it to `${COSMOS_CURATOR_LOCAL_WORKSPACE_PREFIX}/cosmos_curator_local_workspace/`.
   - In other words, `"${COSMOS_CURATOR_LOCAL_WORKSPACE_PREFIX:-$HOME}/cosmos_curator_local_workspace"` is used as the local workspace.

1. Log into the NGC container registry via the Docker CLI. For the username, use `'$oauthtoken'` exactly as shown. It is a special name that indicates that you will authenticate with an API key. Paste your key value at the password prompt.
```bash
docker login --username '$oauthtoken' nvcr.io
```

1. If you will run pipelines with videos on cloud storage, configure `~/.aws/credentials` properly.
   - All S3-compatible cloud storage should work with Cosmos Curator.
     - Right now, since cosmos-curator only relies on `~/.aws/credentials` (not `~/.aws/config`), certain configuration entries need to be in `~/.aws/credentials`; e.g. `region` and `endpoint_url` if it's not AWS S3.
     - It should look similar to [this example file](../../examples/nvcf/creds/aws_credentials).
   - Azure blob storage is also supported but is tested much less extensively.
     - If using Azure blob storage, `~/.azure/credentials` should be configured properly.

1. Set up the environment and install dependencies (This provides a CLI for the steps described in the platform sections of this guide)
    - It is strongly recommended to use a Python virtual environment management system that can specify the Python version, such as
      [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html),
      [mamba](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html),
      [micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html),
      [uv](https://docs.astral.sh/uv/),
      etc.
      - Also it is best to prevent packages under `$HOME/.local/` being used in the virtual environment,
        you can set environment variable **`export PYTHONNOUSERSITE=1`** to exclude user site-package directory.
    - In case you are running in a headless display environment,
      you may need to set environment variable **`export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring`**
      before running `poetry` to avoid getting stuck due to a keyring pop-up.

```bash
# 1. Create virtual environment (using micromamba as an example)
micromamba create -n cosmos-curator -c conda-forge python=3.12.12 poetry
micromamba activate cosmos-curator

# 2. Clone the repository and update `cosmos-xenna` submodule
git clone --recurse-submodules https://github.com/nvidia-cosmos/cosmos-curate.git cosmos-curator
cd cosmos-curator

# 3. Install dependencies
poetry install --extras=local

# 4. Verify the CLI tool is available
cosmos-curator --help
```

Alternatively, you may execute `./devset.sh` to complete initial setup of environment **from within your virtual environment**.

The `cosmos-curator` command is a host-side deployment CLI installed by the setup steps above. Use it on the host
to build images, launch local Docker runs, submit Slurm jobs, and manage NVCF resources. Runtime container images are
focused on pipeline execution and do not guarantee the `cosmos-curator` command or the `cosmos_curator.client` package
inside the container. In-container commands should use `pixi run --as-is` with the Pixi task aliases shown below, or
`pixi run --as-is python -m cosmos_curator...` for modules without a task alias.

## Quick Start for Local Run

**The overall workflow is as follows:**
1. Build a Docker container image
1. Download model weights from Hugging Face
1. Launch a pipeline
   - locally using local-docker launcher - **focus of this section**
   - on a slurm cluster using the Slurm submit and shell commands
   - on [NVIDIA Cloud Functions (NVCF)](https://docs.nvidia.com/cloud-functions/user-guide/latest/cloud-function/overview.html) by reaching out to NVIDIA Cosmos Curator team.
   - on Kubernetes cluster (coming soon)

### Run the Hello-World Example Pipeline

The hello-world example pipeline aims to provide a minimal example to help understand the framework.

- Define the class for pipeline task as `HelloWorldTask` in [hello_world_pipeline.py](../../cosmos_curator/pipelines/examples/hello_world_pipeline.py).
- Define `GPT2` model in [gpt2](../../cosmos_curator/models/gpt2.py).
- Define 3 simple stages (`_LowerCaseStage`, `_PrintStage`, `_GPT2Stage`) in [hello_world_pipeline.py](../../cosmos_curator/pipelines/examples/hello_world_pipeline.py). So the functionality of this pipeline is:
  - stage 1: convert the input prompt in each `HelloWorldTask` to lower case;
  - stage 2: print the the converted input prompt;
  - stage 3: call GPT2 to generate some output;
- Call `cosmos_curator.core.interfaces.pipeline_interface.run_pipeline`.

There is a detailed walk-through in [Pipeline Design Guide](../curator/guides/pipeline-design.md) to help understand how to build a pipeline.
The steps below only shows how to run the pipeline.

```bash
# 1. Build a docker image for hello-world pipeline
#    - The hello-world pipeline uses the GPT-2 model
#    - GPT-2 runs in the default conda environment
cosmos-curator image build --image-name cosmos-curator --image-tag hello-world --envs default

# 2. Download the GPT-2 model weights
cosmos-curator local launch --image-name cosmos-curator --image-tag hello-world -- pixi run --as-is model_download --models gpt2

# 3. Run the hello-world pipeline
cosmos-curator local launch --image-name cosmos-curator --image-tag hello-world --curator-path . -- pixi run --as-is hello_world
```

### Run the Reference Video Pipeline

This section of the instructions references the concept of local paths. Note that these local paths are paths inside the container image, not paths on your local machine. Since `"${COSMOS_CURATOR_LOCAL_WORKSPACE_PREFIX:-$HOME}/cosmos_curator_local_workspace"` is mounted to `/config/` when launching the container, a path like `~/cosmos_curator_local_workspace/foo/` on your local machine needs to be specified as `/config/foo/` in the `cosmos-curator` commandline arguments.

1. **Build a docker image.**
   - Unlike the hello-world example, we run more than one models in this pipeline.
   - It's not always easy to run different models in the same python environment; so we need to build a new image with more conda environments included.
   - This could take up to 30 minutes for a fresh new build.

```bash
cosmos-curator image build --image-name cosmos-curator --image-tag 1.0.0
```

2. **Download model weights from Hugging Face.**
   - For the same reason as above, we need to download weights for a few more models and it will take 10+ minutes depends on your network condition.

```bash
cosmos-curator local launch --image-name cosmos-curator --image-tag 1.0.0 -- pixi run --as-is model_download
```

3. **Run the Split-Annotate Pipeline**
   - **Input and output paths**
     - `--input-video-path` and `--output-clip-path` can be either a local path inside the container or an S3 path.
       - The input videos under `input_video_path` can have any directory hierarchy.
       - If your videos are under `~/cosmos_curator_local_workspace/raw_videos/` and you want the output to be under `~/cosmos_curator_local_workspace/output_clips/`, you should specify `--input-video-path /config/raw_videos/` and `--output-clip-path /config/output_clips/`.
     - When giving an S3 path, it need to start with `s3://`.
       - Right now, space is not allowed in a video's S3 path.
       - For example, you can give `--input-video-path s3://cosmos-curator-oss/raw_videos/` & `--output-clip-path s3://cosmos-curator-oss/output_clips/` to read from `s3://cosmos-curator-oss/raw_videos/` and write to `s3://cosmos-curator-oss/output_clips/`.
       - Please make sure your `~/.aws/credentials` file is configured properly as explained in [Initial Setup](#initial-setup) section above.
   - **`--limit` option**
     - `limit` specifies how many input videos under `input_video_path` to process.
     - Note when running locally with e.g. one GPU, a small `limit` value (like `1`) is needed to avoid running out of memory or disk.
   - **CPU-constrained hosts**
     - Split-annotate default per-stage CPU counts are tuned for server-class hosts and can over-subscribe workstations. If the pipeline fails to schedule with a CPU-resource error, lower the `--*-cpus-per-worker` flags — see [Split-Annotate Pipeline Configurable Options](../curator/reference/video-pipelines.md#split-annotate-pipeline-configurable-options) for a concrete 8-core recipe.
   - **Failure recovery**
     - Failures are often inevitable due to hardware failures/glitches, kernel/driver/library bugs, etc., therefore this pipeline is carefully designed such that it can handle any crash gracefully without loss of compute time and a simple restart would resume from where it left resulting in correct behavior.

```bash
cosmos-curator local launch \
    --image-name cosmos-curator --image-tag 1.0.0 --curator-path . \
    -- pixi run --as-is video_pipeline split \
    --input-video-path <local or s3 path containing input videos> \
    --output-clip-path <local or s3 path to store output clips and metadatas> \
    --limit 1
```

At a high level, this pipeline
- splits each input video into shorter clips based on short transition
- transcodes each clip
- generates motion & aesthetic scores and filter clips (disabled by default)
- generates one descriptive caption for each 256-frame window in each clip
- stores the mp4 clips and metadatas to the specified `output_clip_path`

For more details, please refer to [Split-Annotate Pipeline](../curator/reference/video-pipelines.md#split-annotate-pipeline) section in [Reference Pipelines Guide](../curator/reference/video-pipelines.md).

### Use Gemini for Captioning

Cosmos Curator can call the Google Gemini API instead of local captioning models. To enable it:

1. Add your Gemini API key to `~/.config/cosmos_curator/config.yaml` under the `gemini` section as shown in [Initial Setup](#initial-setup). The key must also be accessible inside the container (the config file is mounted automatically when you use `--curator-path .`).
2. Select the Gemini captioning algorithm when launching the pipeline. The example below also increases `--captioning-max-output-tokens` to `4096`, which avoids Gemini truncation and has worked well in practice:

```bash
cosmos-curator local launch \
    --image-name cosmos-curator --image-tag 1.0.0 --curator-path . \
    -- pixi run --as-is video_pipeline split \
    --input-video-path <input path> \
    --output-clip-path <output path> \
    --captioning-algorithm gemini \
    --captioning-max-output-tokens 4096 \
    --gemini-model-name models/gemini-2.5-pro
```

You can further tune the behaviour with:
- `--gemini-caption-retries` / `--gemini-retry-delay-seconds` to adjust retry policy.
- `--gemini-max-inline-mb` to cap the inline MP4 size sent to Gemini (default `20.0` MB).

If Gemini returns block reasons or empty responses, the stage will surface those details in the clip errors.

### Enhance Captions with OpenAI

For a second-pass refinement of captions you can call the OpenAI API.

1. Populate the `openai` section in `~/.config/cosmos_curator/config.yaml` with your API key (and optional `base_url`).
2. Launch the pipeline with the enhance caption stage enabled and point it to your model:

```bash
cosmos-curator local launch \
    --image-name cosmos-curator --image-tag 1.0.0 --curator-path . \
    -- pixi run --as-is video_pipeline split \
    --input-video-path <input path> \
    --output-clip-path <output path> \
    --enhance-captions \
    --enhance-captions-lm-variant openai \
    --enhance-captions-max-output-tokens 2048
```

`--enhance-captions-openai-model` selects the OpenAI API model (default `auto`, which discovers the model via `/v1/models`). Set `openai.enhance.base_url` in config if you need to use a custom base URL. You can increase `--enhance-captions-max-output-tokens` if you need longer rewrites; the default `2048` works for most scenarios.

### Use an OpenAI-Compatible Endpoint for Embedding

You can generate video clip embeddings using any OpenAI-compatible embedding API (e.g. a local [vLLM](https://docs.vllm.ai/) server) instead of the built-in InternVideo2 or Cosmos-Embed1 models.

1. Download an embedding model, for example:

```bash
hf download Qwen/Qwen3-VL-Embedding-8B --local-dir /opt/models/qwen3-vl-embedding-8b
```

2. Start a vLLM server with the pooling runner:

```bash
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
vllm serve /opt/models/qwen3-vl-embedding-8b \
  --runner pooling \
  --max-model-len 131072 \
  --port 8000
```

3. Add the `openai.embedding` section to `~/.config/cosmos_curator/config.yaml`:

```yaml
openai:
    embedding:
        api_key: "dummy"
        base_url: "http://localhost:8000/v1"
```

4. Run the split-annotate pipeline with `--embedding-algorithm openai`:

```bash
cosmos-curator local launch \
    --image-name cosmos-curator --image-tag 1.0.0 --curator-path . \
    -- pixi run --as-is video_pipeline split \
    --input-video-path <input path> \
    --output-clip-path <output path> \
    --embedding-algorithm openai
```

The model name defaults to `auto`, which discovers the served model automatically via the `/v1/models` endpoint. To specify a model explicitly, use `--openai-embedding-model-name <model-name>`.

4. **Optionally, Run the Split-Annotate Pipeline via API Endpoint**

First, launch the container with a service endpoint.

```bash
cosmos-curator local launch \
   --image-name cosmos-curator --image-tag 1.0.0 --curator-path . \
   -- pixi run --as-is python cosmos_curator/scripts/onto_nvcf.py --helm False
```

After `Application startup complete.` is printed in the log, you can invoke the split-annotate with a `curl` command.

```bash
curl -X POST http://localhost:8000/v1/run_pipeline -H "NVCF-REQID: 1234-5678" -d '{
    "pipeline": "split",
    "args": {
        "input_video_path": "<local or s3 path containing input videos>",
        "output_clip_path": "<local or s3 path to store output clips and metadatas>",
        "limit": 1
    }
}'
```

To stop the service, press `Ctrl+C` in the terminal where you ran the launch command.

**Note:** When Ray starts up, it will print instructions including `ray stop` - do not use this command directly on
the host. The Ray cluster runs inside the container and is managed by the service. Always stop the service using
`Ctrl+C` for proper cleanup.

### Generate Dataset for Cosmos-Predict2 Post-Training

The [Split-Annotate Pipeline](../curator/reference/video-pipelines.md#split-annotate-pipeline) above has first-class support
for [Cosmos-Predict2 Video2World post-training](https://github.com/nvidia-cosmos/cosmos-predict2/blob/main/documentations/post-training_video2world.md).

The following arguments are needed for `split-annotate` pipeline to generate the datasets for Cosmos-Predict2:
- add `--generate-cosmos-predict-dataset` to enable the dataset creation.
- add ` --transnetv2-min-length-frames 120` to specify a minimum clip length of (e.g.) 120 frames, as Cosmos-Predict2 post-training requires 93 frames by default.

This will generate a `cosmos_predict2_video2world_dataset/` sub-directory under the output path specified by `output_clip_path`.
The `cosmos_predict2_video2world_dataset/` sub-directory has the following structure:

```bash
cosmos_predict2_video2world_dataset/
├── metas/
│   ├── {clip-uuid}_{start_frame}_{end_frame}.txt
├── videos/
│   ├── {clip-uuid}_{start_frame}_{end_frame}.mp4
├── t5_xxl/
│   ├── {clip-uuid}_{start_frame}_{end_frame}.pickle
```

Note the `T5` embedding generation are included in this pipeline, such that there is no need to
run the `python -m scripts.get_t5_embeddings --dataset_path ...` command (from `Cosmos-Predict2` repo) as stated in the
[post-training guide](https://github.com/nvidia-cosmos/cosmos-predict2/blob/main/documentations/post-training_video2world.md#post-training-guide),
unless you want to manually edit the captions which will require re-generating the T5 embeddings.

### Useful Options for Local Run

Almost all the CLI commands enable `no_args_is_help`, so running a command without any arguments will print out the help message.
For local launcher, you can simply run `cosmos-curator local launch` to see help messages for all the options.

A useful option is `--curator-path`; when this is given, the local launcher will mount the source code into the container,
such that you don't have to rebuild the container after code changes for local run.

For faster iteration, you can build a slim image and mount your host pixi environments:

```bash
# Build a slim image (lockfile + source only, no pixi install — builds in seconds)
cosmos-curator image build --slim --image-name cosmos-curator --image-tag slim

# Launch with host source code and pixi environments mounted
cosmos-curator local launch --image-name cosmos-curator --image-tag slim \
    --curator-path . --pixi-path . \
    -- pixi run --as-is hello_world
```

The `--pixi-path .` option mounts the `.pixi` directory from the given path into the container, so the container
uses your pre-installed environments directly — no environment installation at runtime.

The container runs as your host UID:GID, so files written to bind-mounted paths are owned by you on the host;
`$HOME` caches (Triton, NVIDIA kernel caches, etc.) persist in `~/.cache/cosmos-curator-home/` across runs. Set
`COSMOS_CURATOR_LOCAL_HOME_DIR` to redirect this scratch dir (e.g. to a local disk when `$HOME` is a remote mount
or has tight quotas).

#### Configuration Files

Instead of passing many CLI flags, you can define all pipeline settings in a JSON or YAML
config file. The two invocation modes are **mutually exclusive** -- use either a config
file or CLI flags, not both:

```bash
# Config mode -- all arguments come from the file
pixi run --as-is video_pipeline /path/to/config.yaml

# CLI mode -- all arguments are passed as flags
pixi run --as-is video_pipeline split --input-video-path /workspace/input --output-clip-path /workspace/output
```

The config format matches the NVCF invoke payload. The file **must** contain a `pipeline`
key naming the subcommand:

```json
{
    "pipeline": "split",
    "args": {
        "input_video_path": "/workspace/input",
        "output_clip_path": "/workspace/output",
        "generate_captions": true,
        "limit": 0
    }
}
```

The same config in YAML:

```yaml
pipeline: split
args:
  input_video_path: /workspace/input
  output_clip_path: /workspace/output
  generate_captions: true
  limit: 0
```

Example launch with a config file:

```bash
cosmos-curator local launch \
    --image-name cosmos-curator --image-tag 1.0.0 --curator-path . \
    --extra-volumes /data/models:/config/models,/data/videos:/workspace/input,/data/output:/workspace/output \
    -- pixi run --as-is video_pipeline \
    /opt/cosmos-curator/examples/osmo/split_config.json
```

Per-pipeline reference templates are provided under `examples/osmo/`:
`split_config.json`, `shard_config.json`, and `dedup_config.json`.

#### Extra Volume Mounts

The `--extra-volumes` option lets you mount additional host directories into the container.
Specify comma-separated `HOST_PATH:CONTAINER_PATH[:MODE]` entries, where `MODE` is optional:

```bash
cosmos-curator local launch \
    --extra-volumes /data/models:/config/models,/data/videos:/workspace/input,/data/static:/workspace/static:ro \
    ...
```

This is useful for mounting model weights, input data, and output directories without resorting to raw `docker run` commands.

#### Mount FFmpeg for Redistributable Images

This runtime override is only required when the image you are using has the redistributable runtime policy, such as
NVIDIA-provided Cosmos Curator images published through `nvcr.io`. If you build the image yourself, the simpler path
is to use the default non-redistributable image. If you select `--redistributable`, you must then provide ffmpeg
separately with this route for pipelines that require codecs outside the redistributable image.

Cosmos Curator images install FFmpeg at `/opt/ffmpeg` and configure `PATH`, `LD_LIBRARY_PATH`, and `PKG_CONFIG_PATH`
to prefer that prefix. To use your own FFmpeg with a redistributable image, mount a complete FFmpeg installation
prefix over `/opt/ffmpeg`:

```bash
USER_FFMPEG_PREFIX=/path/to/ffmpeg-8.1.1-custom

cosmos-curator local launch \
    --image-name cosmos-curator --image-tag slim \
    --curator-path . --pixi-path . \
    --extra-volumes "${USER_FFMPEG_PREFIX}:/opt/ffmpeg:ro" \
    -- bash -lc 'which ffmpeg && ffmpeg -hide_banner -version && ffmpeg -hide_banner -encoders'
```

The mounted prefix should include `bin/ffmpeg`, `bin/ffprobe`, shared libraries under `lib/`, and `lib/pkgconfig`
metadata. Include `include/` as well if you plan to build PyAV against that FFmpeg. For normal pipeline runs, this
overrides FFmpeg CLI/subprocess usage. Full images build PyAV against the image FFmpeg; if Python `av` code paths use
the mounted libraries, keep the mounted FFmpeg ABI-compatible with the image build. Slim images keep the PyPI PyAV
wheel and do not rebuild it against the mounted prefix.

## Launch Pipelines on Slurm

### **PLEASE READ: For end users walking through this section, the guide assumes that you have already set up a local environment and have launched a reference pipeline locally per [Initial Setup](#initial-setup) + [Run the Reference Video Pipeline](#run-the-reference-video-pipeline).**

### Prerequisites for Slurm Run

#### Setup Password-less SSH to the Cluster

Here are the [instructions](https://www.redhat.com/en/blog/passwordless-ssh) published on redhat.com.

Assume the login node of your Slurm cluster is `my-slurm-login-01.my-cluster.com`.
You can verify the password-less SSH setup by `ssh my-slurm-login-01.my-cluster.com`
and it should login directly without asking for password.

One trick to make things easier is to define `my-slurm-login-01.my-cluster.com` in your `~/.ssh/config`, like

```bash
Host my-slurm-login-01.my-cluster.com
  HostName <my-real-slurm-cluster-hostname>
```

Then you can login to your cluster literally using `ssh my-slurm-login-01.my-cluster.com`.

#### Identify User Path on the Cluster

Assume your user directory on the Slurm cluster is `/home/myusername/`. Note
- this path should be **accessible to all compute nodes**.
- this path should have enough disk quota to hold the image and model weights.

Set `$SLURM_USER_DIR` environment variable **on your local host**.

```bash
export SLURM_USER_DIR="/home/myusername"
```

Then set other dependent environment variables.

```bash
source examples/slurm/source_me_env_vars.sh
```

These helper scripts detect the presence of AWS and Azure credential files and only mount the ones that exist, so clusters that rely on a single object store do not need to stage the other provider's configuration.

### Copy Config File, Cloud Storage Credentials, and Model Files to Cluster

**Note 1**: Cloud storage credentials are only required if your input videos or output paths use S3/Azure URIs. If all data resides on the cluster's local or shared storage, you can skip the credential sync steps below.

**Note 2**: The steps below assume that you set up config files, model files, and cloud storage credentials on a different host. If you completed the steps mentioned in [Initial Setup](#initial-setup) + [Run the Reference Video Pipeline](#run-the-reference-video-pipeline) directly on the login node of the Slurm cluster OR on a mounted filesystem visible to all Slurm nodes, you can skip the steps below.

If you have defined `my-slurm-login-01.my-cluster.com` in your `/.ssh/config` like mentioned above, you can simply run

```bash
./examples/slurm/sync_config_creds_models.sh
```

Otherwise you can replace `my-slurm-login-01.my-cluster.com` with your real login hostname in the following commands.

These `rclone` examples use the SFTP backend over SSH (`:sftp,host=...`). Your Slurm login node must have an SFTP
subsystem enabled in SSHD; plain SSH shell access alone is not sufficient.

```bash
RCLONE_REMOTE=":sftp,host=my-slurm-login-01.my-cluster.com:"

# Copy ~/.config/cosmos_curator/config.yaml
ssh my-slurm-login-01.my-cluster.com mkdir -p ${SLURM_COSMOS_CURATOR_CONFIG_DIR}
rclone copyto -P ~/.config/cosmos_curator/config.yaml ${RCLONE_REMOTE}${SLURM_COSMOS_CURATOR_CONFIG_DIR}/config.yaml

# (Optional) Copy ~/.aws/credentials if using S3-compatible storage
ssh my-slurm-login-01.my-cluster.com mkdir -p ${SLURM_AWS_CREDS_DIR}
rclone copyto -P ~/.aws/credentials ${RCLONE_REMOTE}${SLURM_AWS_CREDS_DIR}/credentials

# (Optional) Copy ~/.azure/credentials if using Azure Blob Storage
ssh my-slurm-login-01.my-cluster.com mkdir -p ${SLURM_AZURE_CREDS_DIR}
rclone copyto -P ~/.azure/credentials ${RCLONE_REMOTE}${SLURM_AZURE_CREDS_DIR}/credentials

# Copy models
ssh my-slurm-login-01.my-cluster.com mkdir -p ${SLURM_WORKSPACE}/models
rclone copy -P ${COSMOS_CURATOR_LOCAL_WORKSPACE_PREFIX:-$HOME}/cosmos_curator_local_workspace/models/ ${RCLONE_REMOTE}${SLURM_WORKSPACE}/models/
```

### Create sqsh Image and Copy to the Slurm Cluster

If the Docker image is reachable from the Slurm cluster, import it directly on the cluster. The command runs
`enroot import` through `srun`, defaults to the `cpu` partition, writes to `~/container_images`, and overwrites the
default output file `cosmos-curator+1.0.0.sqsh` unless `--no-overwrite` is provided:

For private registries, create `~/.config/enroot/.credentials` on the Slurm cluster login node so it is stored in the
home directory visible to Enroot on that cluster. Ignore registries you do not use; entries can be added later.
For NGC, include both `nvcr.io` and `authn.nvidia.com`:

```text
machine nvcr.io login $oauthtoken password YOUR-NGC-API-KEY
machine authn.nvidia.com login $oauthtoken password YOUR-NGC-API-KEY
```

For other private registries, add the registry host with the username and token format required by that service:

```text
machine PRIVATE-REGISTRY-HOST login YOUR-USERNAME password YOUR-TOKEN
```

Then restrict the credentials file permissions:

```bash
chmod 0600 ~/.config/enroot/.credentials
```

```bash
cosmos-curator slurm import-image \
  -A my_slurm_account \
  --output-filename cosmos-curator_hello-world.sqsh \
  nvcr.io/your-org/your-cosmos-curator-image:your-tag
```

When running from outside the Slurm login host, add `--login-node` and `--username` as needed:

```bash
cosmos-curator slurm import-image \
  --login-node my-slurm-login-01.my-cluster.com \
  --username my_username_on_slurm_cluster_if_different_than_local_username \
  -A my_slurm_account \
  --output-filename cosmos-curator_hello-world.sqsh \
  docker://cosmos-curator:hello-world
```

Unprefixed image references are treated as `docker://` references. Use an explicit Enroot URI such as `dockerd://...`
only when the Slurm compute node can access that source.

If the image exists only on your local machine, use the manual local import and copy flow:

1. Install `enroot` on your local machine based on the [instructions here](https://github.com/NVIDIA/enroot/blob/master/doc/installation.md).

2. Import the hello world docker image built above to create a `.sqsh` file.

```bash
export COSMOS_CURATOR_IMAGE_NAME="cosmos-curator_hello-world.sqsh"
enroot import --output $COSMOS_CURATOR_IMAGE_NAME dockerd://cosmos-curator:hello-world
```

3. Copy the sqsh file to slurm cluster.

Again if you have defined `my-slurm-login-01.my-cluster.com` in your `/.ssh/config`, you can simply run

```bash
./examples/slurm/upload_image.sh
```

Otherwise replace `my-slurm-login-01.my-cluster.com` with your real login hostname in the following commands.

```bash
RCLONE_REMOTE=":sftp,host=my-slurm-login-01.my-cluster.com:"
ssh my-slurm-login-01.my-cluster.com mkdir -p ${SLURM_IMAGE_DIR}
rclone copyto -P ./$COSMOS_CURATOR_IMAGE_NAME ${RCLONE_REMOTE}${SLURM_IMAGE_DIR}/$COSMOS_CURATOR_IMAGE_NAME
```

### Launch on Slurm

From a cluster checkout, `slurm submit` uses the same container defaults as the interactive Slurm launcher: the default
image path, workspace path, cache path, local config file, AWS credentials when present, and live source mount from the
current directory when it looks like a Cosmos Curator repo.

If your cluster has default account, partition, and GPU settings, launch with:

```bash
cosmos-curator slurm submit -- pixi run --as-is hello_world
```

If your cluster requires an account, either pass `--account` or set `SBATCH_ACCOUNT` before submitting:

```bash
export SBATCH_ACCOUNT=my_slurm_account
cosmos-curator slurm submit -- pixi run --as-is hello_world
```

Add only the remaining cluster-specific scheduling options your site requires, for example partition, QoS, GPUs, or node
count. The Slurm-style short aliases are also supported for common allocation options:

```bash
cosmos-curator slurm submit \
  -A my_slurm_account \
  -p my_slurm_gpu_partition \
  -q my_slurm_qos \
  -G 8 \
  --nodes 1 \
  -J "hello-world" \
  -- pixi run --as-is hello_world
```

Use `-G 8` for the common GPU request form. `--gres=gpu:8` is also supported; the two options are mutually exclusive.

When submitting from outside the Slurm login host, add `--login-node` and `--username` as needed. Container mount paths
are still validated on the cluster, so pass `--container-image`, `--workspace-path`, `--cache-path`, `--curator-path`,
or `--extra-mounts` when your cluster uses non-default paths.
If an explicit container mount uses the same destination as an auto-detected default, the explicit mount is used.

The command above will print the slurm job id like below
```bash
Submitted batch job <slurm_job_id>
```

For a Slurm redistributable image, such as an NVIDIA-provided image imported from `nvcr.io`, make the user-provided
FFmpeg prefix visible on the cluster and append a read-only mount before submitting the job:

```bash
USER_FFMPEG_PREFIX="${SLURM_USER_DIR}/ffmpeg-8.1.1-custom"
export CONTAINER_MOUNTS="${CONTAINER_MOUNTS:+${CONTAINER_MOUNTS},}${USER_FFMPEG_PREFIX}:/opt/ffmpeg:ro"
```

The same `/opt/ffmpeg` requirements and PyAV compatibility notes from
[Mount FFmpeg for Redistributable Images](#mount-ffmpeg-for-redistributable-images) apply.

#### Email Notifications (Optional)

You can optionally receive email notifications about your SLURM jobs using the `--mail-user` and `--mail-type` parameters.

**Parameters:**
- `--mail-user`: Email address to receive notifications
- `--mail-type`: Comma-separated list of events to notify about. Options include:
  - `BEGIN`: Job start
  - `END`: Job completion
  - `FAIL`: Job failure
  - `REQUEUE`: Job requeue
  - `ALL`: All events
  - `STAGE_OUT`: Stage out (data transfer) completion
  - `TIME_LIMIT`, `TIME_LIMIT_90`, `TIME_LIMIT_80`: Time limit warnings

**Note:** If you provide `--mail-user` without `--mail-type`, SLURM will typically default to `END,FAIL`. If you provide `--mail-type`, you must also provide `--mail-user`.

**Example with email notifications:**

```bash
cosmos-curator slurm submit \
  -A my_slurm_account \
  -p my_slurm_gpu_partition \
  -q my_slurm_qos \
  -G 8 \
  --nodes 1 \
  -J "hello-world" \
  --mail-user your.email@example.com \
  --mail-type END,FAIL \
  -- pixi run --as-is hello_world
```

**⚠️ Cluster-Specific Considerations:**

Email notification functionality depends on your cluster's configuration:
- The cluster must have a properly configured mail server
- Network firewalls or security policies may affect email delivery
- Some clusters may have email notifications disabled or restricted by policy
- Email delivery may be delayed depending on the mail server configuration

If email notifications are not working as expected, please verify with your cluster administrators that this feature is enabled and properly configured for your environment.

### Find Logs

The slurm job log is at `"${SLURM_LOG_DIR}/{job_name}_{slurm_job_id}.log"` on the cluster.

You can also use the CLI to monitor the log:

```bash
cosmos-curator slurm job-log \
  --login-node my-slurm-login-01.my-cluster.com \
  --username my_username_on_slurm_cluster_if_different_than_local_username \
  --job-id slurm_job_id_printed_above
```

### Processing Large Video Sets in Batches

When processing a large number of videos (e.g., 1 million), you can split the work into smaller jobs using pre-split
manifest files with `--input-video-list-json-path`. Each manifest is a JSON array of full input video paths:

```json
[
  "s3://bucket/videos/vid_001.mp4",
  "s3://bucket/videos/vid_002.mp4"
]
```

**Note:** Every path in the manifest must be prefixed by the `--input-video-path` value.

Generate the manifests by splitting your full video list into chunks and uploading them to S3, then submit one job per
batch:

```bash
# Generate manifests and upload to S3 (one-time)
python -c "
import json, math, pathlib
videos = json.load(open('all_videos.json'))
pathlib.Path('manifests').mkdir(exist_ok=True)
chunk = 1000
for i in range(math.ceil(len(videos) / chunk)):
    with open(f'manifests/batch_{i:04d}.json', 'w') as f:
        json.dump(videos[i*chunk:(i+1)*chunk], f)
"
aws s3 sync manifests/ s3://bucket/manifests/

# Submit one job per manifest
NUM_BATCHES=1000
for i in $(seq 0 $((NUM_BATCHES - 1))); do
  MANIFEST=$(printf "s3://bucket/manifests/batch_%04d.json" "$i")
  cosmos-curator slurm submit \
    -A my_slurm_account \
    -p my_slurm_gpu_partition \
    -q my_slurm_qos \
    -G 8 \
    --nodes 1 \
    -J "split-batch-${i}" \
    -- pixi run --as-is video_pipeline split \
      --input-video-path s3://bucket/videos \
      --input-video-list-json-path "$MANIFEST" \
      --output-clip-path s3://bucket/output
done
```

This approach avoids duplicates (each video appears in exactly one manifest) and missed videos (the union of all
manifests equals the full input set). If a job fails, resubmit that specific batch — the pipeline automatically skips
already-processed videos within each job, so partial failures are handled gracefully. Native Slurm array job support
may be added in a future release to simplify this workflow.

For smaller runs or sequential processing, you can also simply use `--limit N` with a shared output directory. The
pipeline checks for completed output metadata and skips already-processed videos on each run.

### Developing on Slurm

If you plan to modify or create new pipelines on Slurm, use the interactive shell. It allocates a Slurm node with
`srun --pty`, starts the container through `srun --container-image`, mounts your live checkout into the container,
reuses a shared cache directory, and sets the common runtime environment variables needed by Slurm-backed Cosmos Curator
pipelines.

#### Interactive Slurm shell

Use this path when your cluster supports Pyxis with `srun --container-image`. It is especially useful on clusters where
direct `enroot start` is not supported.

From a full cluster checkout containing `cosmos_curator/`, `pixi.toml`, and `pixi.lock`, start a shell in a GPU
allocation and container in one step:

```bash
cosmos-curator slurm shell --curator-path .
```

This uses the default image path, workspace path, and cache path. Pass `--container-image`, `--workspace-path`, or
`--cache-path` only when your cluster uses different locations. If you are running from a local desktop instead of the
Slurm login node, add `--login-node` and `--username` as needed, and pass an explicit `--curator-path` value that is
valid on the cluster. Credential and config mounts also use cluster-visible paths; override them with
`--container-mounts` or disable credential mounts when your login node paths differ.

Use `-G 8` for the common GPU request form. `--gres=gpu:8` is also supported; the two options are mutually exclusive.
For example, on an 8-GPU node:

```bash
cosmos-curator slurm shell --curator-path . -G 8
```

To run a specific command instead of opening `bash`, put it after `--`:

```bash
cosmos-curator slurm shell --curator-path . -- pixi run --as-is hello_world
```

For slim images, the shell command installs the Pixi environments listed in the image's `COSMOS_CURATOR_SLIM_ENVS`.
To reduce startup time during focused development, warm up only the environments you need:

```bash
cosmos-curator slurm shell \
  --curator-path . \
  --pixi-envs model-download,default,unified \
```

Inside the container, run commands with `pixi run --as-is` so Pixi uses the environments installed during startup:

```bash
cd /opt/cosmos-curator
pixi run --as-is -e unified hello_world
```

Environments not listed with `--pixi-envs` are not installed during startup; install them explicitly inside the
container before using `pixi run --as-is -e <env> ...`.

The [Interactive Slurm Development Guide](../curator/guides/slurm-interactive.md) documents a manual Enroot workflow for
clusters where that setup is useful.

### Speeding up Model Load on Slurm

Model loading can sometimes be sped up by adding

```
--copy-weights-to /raid/scratch/models
```

to the pipeline command. This is because of the way that the transformers library loads safetensors from disk. Its method is efficient for local disk, but can be very slow if the model weights are stored on NFS or Lustre.

This is only useful if there is enough space on the local nodes to store the model weights.

## Launch Pipelines on NVIDIA DGX Cloud

Cosmos Curator can be deployed on [NVIDIA Cloud Function (NVCF)](https://docs.nvidia.com/cloud-functions/index.html) platform.

There are a few steps needed to get a new user onboarded to NVCF, so please reach out to NVIDIA Cosmos Curator team and we will guide you through the process.

If you have already onboarded to NVCF and have an NVCF Org, please follow this [NVCF Guide](nvcf-guide.md) to deploy Cosmos Curator on NVCF.

## Launch Pipelines on K8s Cluster (coming soon)

## Observability for Pipelines

The resource usage and bottleneck of the pipeline can vary with:
- input data, e.g. when you have ~10MB videos vs. ~10GB videos,
  or in the most difficult case you have a mix of 10MB & 10GB videos in the same input set;
- hardware configuration, e.g. ratio of CPU core count & GPU count & system memory size.

Therefore, it is critical to have good observability in place to help debug reliability problems and optimize pipeline throughput.

We have implemented a set of metrics in [Cosmos-Xenna](https://github.com/nvidia-cosmos/cosmos-xenna)
and included a [Grafana dashboard](../../examples/observability/grafana/cosmos-curator-oss.json) for `Cosmos Curator` pipelines.
More details can be found in [Observability Guide](../curator/guides/observability.md).

## Build the Client package

The `cosmos-curator` client can be built as a wheel and installed in a standalone mode, without the need for the rest of the source environment

```bash
poetry build
pip3 install dist/cosmos_curator*.whl
```

## Troubleshooting
If you encounter any issues:
1. Ensure your Hugging Face credentials are correctly configured
2. Verify that you have sufficient disk space for model downloads
3. Check that Docker is running and accessible
4. Check that [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) is installed and `docker.service` is restarted after the installation of NVIDIA Container Toolkit.
5. For debugging specific pipeline stages, see the [Stage Replay Guide](../curator/guides/stage-replay.md)
6. Ensure you have the correct Python version installed

## Support
For additional support or to report issues, please contact the development team or create an issue in the repository.

## Responsible Use of AI Models
[Responsible Use](../../RESPONSIBLE_USE.md)
