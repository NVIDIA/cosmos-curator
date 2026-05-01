# Cosmos Curator - Pipeline Design Guide

- [Cosmos Curator - Pipeline Design Guide](#cosmos-curator---pipeline-design-guide)
  - [Core Components](#core-components)
    - [Pipeline Task Class](#pipeline-task-class)
    - [Pipeline Stage Class](#pipeline-stage-class)
    - [Model Class](#model-class)
    - [Conda Environment Management](#conda-environment-management)
    - [Run the Pipeline](#run-the-pipeline)
      - [StageSpec Customization](#stagespec-customization)
    - [Writing Artifacts in Multi-Node Clusters](#writing-artifacts-in-multi-node-clusters)
  - [Quiz: Adding WordCount Stage to Hello-World Pipeline](#quiz-adding-wordcount-stage-to-hello-world-pipeline)
  - [Pipeline Performance](#pipeline-performance)
    - [Extract CPU Processing to Separate Stages](#extract-cpu-processing-to-separate-stages)
    - [Pack Multiple GPU Workers to One GPU](#pack-multiple-gpu-workers-to-one-gpu)

This guide explains how to modify existing pipelines or add new pipelines into the Cosmos Curator system.

## Core Components

Let's use the [hello_world_pipeline](../../../cosmos_curator/pipelines/examples/hello_world_pipeline.py) as an example to walk through the core components.

### Pipeline Task Class

Each new pipeline should define a class representing the tasks being passed between stages.

This class need to inherit `PipelineTask` base class defined in
[cosmos_curator/core/interfaces/stage_interface.py](../../../cosmos_curator/core/interfaces/stage_interface.py).
No function overriding is needed today but may needed later as the underlying layer improves.

For the Hello-World pipeline, it defines a simple `HelloWorldTask` class:

```python
@attrs.define
class HelloWorldTask(PipelineTask):
    prompt: str
    output: str | None = None
```

What typically happens is
- first construct a list of input tasks with a few attributes initialized
- as the task passing through pipeline stages, more fields are getting populated.

For example, it builds a list of two input tasks
```python
prompts = ["The KEY TO A CREATING GOOD art is", "Once upon a time"]
tasks: list[PipelineTask] = [HelloWorldTask(prompt=x) for x in prompts]
```

### Pipeline Stage Class

Each pipeline stage should define a class that inherits `CuratorStage` base class defined in
[cosmos_curator/core/interfaces/stage_interface.py](../../../cosmos_curator/core/interfaces/stage_interface.py).

**The following methods need to be overridden always:**

1. `process_data()`: implements the actual actions for this stage.
   - The method takes a list of pipeline tasks as input and output a list of pipeline tasks.
   - In the hello-world pipeline, the `_LowerCaseStage` simply convert the `prompt` field to lower case in each pipeline task.
   - There are two more advanced use cases:
     - The number of input tasks can be different than the number of output tasks.
       - This is the "dynamic chunking" feature discussed in the [How to handle large variation in input data?](../reference/ARCHITECTURE.md#how-to-handle-large-variation-in-input-data) section.
       - The other [demo_task_chunking_pipeline](../../../cosmos_curator/pipelines/examples/demo_task_chunking_pipeline.py) demonstrates the feature with a similar minimal example.
     - The type of input tasks can be different than the type of output tasks.

```python
class _LowerCaseStage(CuratorStage):

    def process_data(self, tasks: list[HelloWorldTask]) -> list[HelloWorldTask] | None:
        # convert the prompt to lowercase
        for task in tasks:
            task.prompt = task.prompt.lower()
        return tasks
```

2. `resources()`: specifies how many CPUs and GPUs each stage worker need
   - The `cpus` represents the number of logical CPU cores that the stage worker will use.
   - The `gpus` can be either
     - a fractional number like `0.25` if this stage's worker cannot fully utilize one GPU.
     - a `>1` number to allocate more than one GPU to enable e.g. tensor-parallelism for running larger models.

```python
    @property
    def resources(self) -> CuratorStageResource:
        return CuratorStageResource(cpus=1.0, gpus=0.0)
```

**For a stage that uses an AI model, the following methods need to be overriden:**

3. `model()`: returns a `ModelInterface` object.
   - This is needed by the underlying layer to derive pipeline runtime configuration.

```python
class _GPT2Stage(CuratorStage):

    @property
    def model(self) -> ModelInterface | None:
        return self._model
```

4. `stage_setup()`: implements the setup work when a Ray actor for this stage's worker is created.
   - A subtle but important difference between the `__init__` constructor and this `stage_setup` method is that the constructor runs in the base conda environment while `stage_setup` runs in the conda environment specified by the model. Therefore code to setup the model need to be in this method.
   - The CPU stages without a model can also implement this method if any setup work is needed.

```python
    def stage_setup(self) -> None:
        self._model.setup()
```

**For advanced use cases, the following methods can be overriden:**

5. `stage_setup_on_node()`: implements the setup work needed for this stage on a per-node basis.
   - This is guaranteed to run exactly once per node by one of the stage's actors.
   - An example usage is to copy model weights from distributed network filesystem to local SSD to speedup model loading. If we instead implement this in `stage_setup()`, some form of race-condition protection like file locking will need to be implemented.

### Model Class

Each model should be wrapped in a class, which inherits `ModelInterface` base class defined in
[cosmos_curator/core/interfaces/model_interface.py](../../../cosmos_curator/core/interfaces/model_interface.py). If you're adding support for a vLLM-based model, follow this guide: [VLLM Interface Plugin Guide](VLLM_INTERFACE_PLUGIN.md)

The hello-world pipeline uses `GPT2` model defined in [cosmos_curator/models/gpt2.py](../../../cosmos_curator/models/gpt2.py).

The following methods are required to be overriden:

```python
class GPT2(ModelInterface):
    # need override conda_env_name to tell underlying logic which conda env to use
    @property
    def conda_env_name(self) -> str:
        return "transformers"

    # need override model_id_names to faciliate model download
    @property
    def model_id_names(self) -> list[str]:
        return ["openai-community/gpt2"]

    # need override setup which is called automatically when creating stage actors
    def setup(self) -> None:
        model_dir = model_utils.get_local_dir_for_weights_name(self.model_id_names[0])
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.model = GPT2LMHeadModel.from_pretrained(model_dir)
        self.model.to("cuda")
```

To help management of models, add a section in [all_models.json](../../../cosmos_curator/configs/all_models.json).
- If only a few files are needed from the huggingface repo, the list of file names can be specified under `filelist` entry.
- When running on [NVIDIA Cloud Function](../../client/NVCF_GUIDE.md#upload-model-weights), the model ID used on NVCF model registry can be specified under `nvcf_model_id` entry.

```json
{
    ...,
    "gpt2": {
        "model_id": "openai-community/gpt2",
        "version": "607a30d783dfa663caf39e06633721c8d4cfcd7e",
        "filelist": null,
        "nvcf_model_id": "gpt2"
    },
    ...
}
```

### Conda Environment Management

The `GPT2` model above uses a conda environment called `transformers`;
that corresponds to the environment `transformers` in [pixi.toml](../../../pixi.toml).

Every `env` needs to be listed in [pixi.toml](../../../pixi.toml).
Note as a convention enforced by `pixi`, you should use `-` instead of `_` for the `env` name.

Then when building the docker image for running pipelines, use option `--envs` to specify which conda environments to be included in the build.

In case you find it hard to configure your `env` using `pixi`, you can add a `post_install.sh` script under `package/cosmos_curator/envs/<env_name>/`.
We have an example for `paddle-ocr` env which only installs the basic packages in [pixi.toml](../../../pixi.toml)
and then installs `paddlepaddle-gpu`
from [package/cosmos_curator/envs/paddle-ocr/post_install.sh](../../../package/cosmos_curator/envs/paddle-ocr/post_install.sh).
Note that if you want to use `pip` inside your `post_install.sh` script, you will need to add `pip` feature to your `env`
in [pixi.toml](../../../pixi.toml); e.g. for `paddle-ocr` env, we have `paddle-ocr = ["core", "pip"]` in [pixi.toml](../../../pixi.toml).

### Run the Pipeline

Once we have
- a list of input pipeline tasks
- a list of pipeline stages

We can call `run_pipeline()` defined in [cosmos_curator/core/interfaces/pipeline_interface.py](../../../cosmos_curator/core/interfaces/pipeline_interface.py).

```python
run_pipeline(
    input_tasks: list[PipelineTask],
    stages: Sequence[CuratorStage | CuratorStageSpec],
    model_weights_prefix: str = MODEL_WEIGHTS_PREFIX,
    runner: RunnerInterface | None = None,
    stage_save_config: StageSaveConfig | None = None,
    args: argparse.Namespace | None = None,
) -> list[T]
```

When `args` is provided and contains any `--profile-*` flags, `run_pipeline`
automatically builds a `ProfilingConfig` and wraps every stage with profiling
instrumentation -- no changes to stage code are needed.  Profiling artifacts
are written to `<output-path>/profile` (a static subdirectory derived from
the pipeline output path).  See the
[Observability Guide](OBSERVABILITY.md#profiling-and-instrumentation)
for CLI flags and the [Profiling Guide](PROFILING.md) for
full details on backend internals, file naming, and artifact delivery.

In hello-world pipeline, 

```python
    # construct a list of input pipeline tasks
    prompts = ["The KEY TO A CREATING GOOD art is", "Once upon a time"]
    tasks: list[PipelineTask] = [HelloWorldTask(prompt=x) for x in prompts]

    # construct a list of pipeline stages
    stages: list[CuratorStage | CuratorStageSpec] = [
        CuratorStageSpec(_LowerCaseStage(), num_workers_per_node=2),
        _PrintStage(),
        _GPT2Stage(),
    ]

    # run the pipeline
    run_pipeline(tasks, stages)
```

#### StageSpec Customization

When building the pipeline, each `CuratorStage` class is wrapped by a `CuratorStageSpec` class with a list of properties.
To override these properties, we can wrap the stage class directly before sending to `run_pipeline`.
Two commonly used properties are:
- `num_workers_per_node: int = None`: set a fixed number of workers per node and disable auto-scaling, typically for IO workers.
- `num_run_attempts_python: int = 1`: set number of retry attempts, if randomly failures are expected.

#### Stage Builder Functions

For pipelines with multiple logical groups of stages, each group has a dedicated **factory function** and a frozen **config dataclass**. This keeps stage construction modular and testable without introducing extra abstractions.

```python
from cosmos_curator.pipelines.video.clipping.clipping_builders import TranscodeConfig, TransNetV2SplitConfig,
    build_transcode_stages, build_transnetv2_split_stages
from cosmos_curator.pipelines.video.read_write.read_write_builders import IngestConfig, build_ingest_stages

stages: list[CuratorStage | CuratorStageSpec] = []
stages.extend(build_ingest_stages(IngestConfig(input_path=...)))
stages.extend(build_transnetv2_split_stages(TransNetV2SplitConfig()))
stages.extend(build_transcode_stages(TranscodeConfig()))

run_pipeline(input_tasks, stages)
```

Each `build_*_stages()` function takes a config and returns a `list[CuratorStage | CuratorStageSpec]`. Configs and builders live alongside the stages they wrap — for example `cosmos_curator/pipelines/video/captioning/captioning_builders.py`, `clipping/clipping_builders.py`, `embedding/embedding_builders.py`, etc.

### Writing Artifacts in Multi-Node Clusters

In multi-node Ray or Slurm deployments, each stage worker runs on a
different node and may need to write artifacts (intermediate files,
profiling captures, reports) to a shared destination that could be a
local directory, an S3 bucket, or an Azure blob container.

The framework provides `StorageWriter`
(see `cosmos_curator/core/utils/storage/storage_utils.py`) so that stage
code never needs to know the underlying storage backend.

**Quick start** -- create a `StorageWriter` once and use its methods
inside `process_data()`:

```python
from cosmos_curator.core.utils.storage.storage_utils import StorageWriter

# Construct from a local or remote path (S3/Azure).
writer = StorageWriter(output_dir)

# --- fire-and-forget writes ------------------------------------------------
writer.write_str_to("report.txt", "Pipeline complete.\n")
writer.write_bytes_to("summary.bin", raw_bytes)

# --- tools that require a local file path ----------------------------------
# resolve_path() returns a WritablePath (os.PathLike[str]); call
# path.close() to finalize (uploads to remote if needed, no-op for local).
path = writer.resolve_path("data.bin")
expensive_tool.write(path)
path.close()
```

For a detailed explanation of the storage abstractions and multi-node
behavior, see [Architecture Guide - Multi-Node Storage I/O](../reference/ARCHITECTURE.md#multi-node-storage-io).

> **Note**: `StorageWriter` is the right tool for **stage-level artifacts**
> (pipeline outputs, intermediate files, reports).  **Profiling artifacts**
> (CPU/memory/GPU profiles, OTel traces) are handled automatically by the
> framework through a separate staging + collection mechanism
> (`ArtifactDelivery` + `RayFileTransport`) -- stage authors do not need
> to manage profiling output.  See the
> [Profiling Guide - Artifact Delivery Flow](PROFILING.md#artifact-delivery-flow)
> for details.

## Quiz: Adding WordCount Stage to Hello-World Pipeline

As a simple exercise, consider adding a `WordCountStage` after the `_GPT2Stage` to count the number words that GPT2 has generated.
- Add a field `word_count` to `HelloWorldTask(PipelineTask)`
- Define a new `WordCountStage(CuratorStage)` and implement `resources()`, `process_data`, etc.
- Add the stage to `stages: list[CuratorStage | CuratorStageSpec]` before sending to `run_pipeline`.
- Run the new pipeline with instructions in [End User Guide](../../client/END_USER_GUIDE.md#run-the-hello-world-example-pipeline).

## Pipeline Performance

### Extract CPU Processing to Separate Stages

Some models require non-trivial preprocessing on CPU before sending to GPU for inference.
For example, vision models would require the input video to be first decoded into frames and then resize, normalized, etc.
If such preprocessing happens inside the GPU stage, it will waste GPU time.

This framework makes it very easy to address such problems.
We can simply extract such CPU work into a separate stage and pass the processed tensors into the GPU stage directly.
As a result, the GPU stage will push the GPU utilization higher while the CPU stage would be effectively free by hidding under the GPU processing time.
The `QwenInputPreparationStage` and `QwenCaptionStage` in [cosmos_curator/pipelines/video/captioning/captioning_stage.py](../../../cosmos_curator/pipelines/video/captioning/captioning_stages.py) are an example of such optimizations.

### Pack Multiple GPU Workers to One GPU

For small models, sometimes it is difficult to push up the GPU utilization.
One way is to request a fraction of GPU and allow multiple stage workers to be allocated on the same GPU.
The GPU memory usage metric above would help define this fractional number to maximize the GPU usage while avoiding CUDA OOM.

In the reference video pipeline, `AestheticFilterStage` in [cosmos_curator/pipelines/video/filtering/aesthetics/aesthetic_filter_stages.py](../../../cosmos_curator/pipelines/video/filtering/aesthetics/aesthetic_filter_stages.py) requests 0.25 GPUs per worker.

## Pipeline Debugging

When developing or debugging pipeline stages, you can use **Stage Replay** to run individual stages in isolation without re-executing the entire pipeline.

See the [Stage Replay Guide](STAGE_REPLAY.md) for:
- Saving task inputs from specific stages
- Replaying stages with saved data
- Rapid iteration on stage logic
- Debugging stage-specific issues

This is especially useful when:
- A specific stage is failing or producing unexpected results
- You want to test changes to a single stage quickly
- You need to debug complex multi-stage pipelines
