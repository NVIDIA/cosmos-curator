# Changelog

## Latest

## [2.0.0]

### Released

- 2026-05-27

### Breaking Changes

- Rename the project from Cosmos-Curate to Cosmos Curator:
  - Python imports and package paths move from `cosmos_curate` to `cosmos_curator`.
  - The command-line entry point moves from `cosmos-curate` to `cosmos-curator`.
  - The Helm chart path/name moves from `charts/cosmos-curate` to `charts/cosmos-curator`.
  - Examples, docs, and config paths now use `cosmos_curator` naming, including
    `~/.config/cosmos_curator/config.yaml`.
- Clean up split-pipeline CLI and config arguments:
  - Replace `--enable-sam3` with `--sam3` / `--no-sam3`.
  - Replace `--enable-event-captioning` with `--event-captioning` / `--no-event-captioning`.
  - Replace `--generate-cosmos-predict-dataset predict2` with the boolean
    `--generate-cosmos-predict-dataset`; JSON/YAML configs now use `true` or `false`.
  - Replace `--artificial-text-filter enable|disable` with
    `--artificial-text-filter` / `--no-artificial-text-filter`.
  - Replace `--video-classifier enable|disable` with `--video-classifier` /
    `--no-video-classifier`.
  - Remove the `--qwen-filter-*` and `--qwen-video-classifier-*` aliases; use
    `--vlm-filter-*` and `--video-classifier-*`.
  - Remove the unsupported `--qwen-use-async-engine` flag.
- Rename all-caps documentation filenames to lowercase kebab-case; update external links that
  target old branch-relative docs paths such as `docs/client/END_USER_GUIDE.md`.

### Added

- Qwen3.6-27B (BF16 and FP8) support for video and image captioning, registered as
  `qwen3_6_27b` and `qwen3_6_27b_fp8` variants.
- Ray Data support for Qwen captioning and TransNetV2 splitting.
- Split-output comparison tooling for summaries, captions, motion scores, and aesthetic scores.
- Run-level caption quality statistics for split-video outputs and heuristic caption quality flags.
- Video pixel budget override support for windowed vLLM captions.
- vLLM async and OpenAI backend support for SAM3 per-event captioning.
- Sensor-library overlap support, motion-vector data on `CameraSensor`, and a
  decoder-utils-compatible sampling grid.
- Interactive Slurm launch workflow.
- Pixi development tasks for linting, CPU tests, and the `cosmos-curator` CLI.

### Fixed

- Stabilize GPU stage autoscaling and upgrade `cosmos-xenna` to v0.4.2.
- Release GPU memory on stage teardown to prevent lingering CUDA contexts.
- Include the caption window end frame in CPU sampling and average caption tokens by window.
- Fix Slurm submit launches during pixi solves and guard `nvidia-smi` calls on CPU-only nodes.
- Repair client wheel packaging, including required storage utilities and clip-viewer assets.
- Serve `marked.min.js` locally in the clip viewer to avoid a CDN dependency.
- Avoid clobbering generated Dockerfiles during parallel CI builds.
- Pin `PyNvVideoCodec` to `>=2.0.4,<2.1` in the `unified` pixi environment for FFmpeg
  vulnerability remediation.

### Removed

- Remove deprecated `RemuxStage` class; remuxing remains handled inline by `VideoDownloader`.

### Changed

- Replace Slurm launch internals with a shell-based launch path and simplify submit defaults.
- Migrate vLLM async captioning to Xenna continuous mode and consolidate sync/async tuning through
  the vLLM plugin layer.
- Refactor the sensor library around `DataSource`.
- Include the `sam3` pixi environment by default.
- Build distribution-ready FFmpeg without H.264 support.
- Upgrade vLLM to 0.21.0, Ray to 2.55.1, Pixi to 0.68.0, and FFmpeg to `>8.1`.

### Documentation

- Add split-pipeline stage reference documentation.
- Refresh captioning contracts and metadata guidance.
- Add split-output comparison and Orca agentic orchestration design documents.
- Normalize documentation filenames to lowercase kebab-case.
- Clarify host CLI versus runtime container usage.
- Update MR description guidance.

## [1.4.0]

### Released

- 2026-05-01

### Added

- SAM3-based video object tracking, per-event VLM captioning, serialized SAM3 outputs, an example
  event pipeline, and a demo tool.
- Sensor library support for GPS, IMU, camera intrinsics, and camera extrinsics data.
- MP4 header validation utilities for video-index checks.
- Qwen3.5-27B support for image captioning.
- External OpenAI/Gemini endpoint support for image semantic filtering and classification stages.
- Async OpenAI/Gemini request handling with `batch_size`-controlled concurrency for image/video
  captioning and external filter/classifier stages.
- `exclusive_end_ns` support in `make_ts_grid` for half-open clip spans.

### Fixed

- Prevent Qwen from falling back to native-resolution inputs during resize.
- Isolate vLLM async per-window payload handling.
- Preserve model-variant-specific image filter errors.

### Changed

- Upgrade the `cosmos-xenna` Python package and submodule to v0.4.0.
- Add a dedicated `sam3` pixi environment for Segment Anything 3 dependencies.
- Include runtime prompt and config data files in built wheels.

### Documentation

- Reorganize curator documentation into design, guide, and reference sections.
- Add the interactive Slurm guide.
- Add GPS and IMU sensor-library design documentation.
- Update image pipeline documentation, including Qwen3.5 coverage.

## [1.3.0]

### Released

- 2026-04-27

### Added

- Image curation pipeline with semantic filtering
- Image embedding stages (Cosmos-Embed1, InternVideo2-MM, OpenAI-compatible) and image annotate pipeline
- Qwen3.5-27B support for vLLM video captioning
- OpenAI- and Gemini-compatible endpoints for image captioning, filtering, and classification
- Artificial-text detection stage for the video filtering pipeline (PaddleOCR-based)
- Sensor library (camera-only) with `SensorGroup`, mcap-based ingestion, and timestamp validation
- SeedVR-based upscaling stage
- Pipeline config files with NVCF-compatible JSON and YAML loading (`--config` for split/shard/dedup)
- Centralized pipeline argument validation via `common_pipeline_settings` and `shard_pipeline_settings`
- vLLM async captioning stage for higher captioning throughput (experimental — correctness
  issues are still being worked through; not recommended for production use)
- OpenTelemetry instrumentation for vLLM captioning
- Token-counting instrumentation to measure captioning throughput
- Caption status fields normalized across caption backends, with status-gated metadata writing
- Stage-replay validation that compares re-run output against the original recording
- S3 support for `stage-save` and `stage-replay`
- Ray Data hello-world pipeline and splitting pipeline MVP as an alternative engine alongside Xenna
- `--*-cpus-per-worker` knobs documented for CPU-constrained hosts
- Run local-launched container as the host user (including AD/SSSD/NIS UIDs) to avoid root-owned outputs
- Slim Docker image built alongside the full image, with auto-warmup honoring `--envs`
- Local Xenna build path in CI and per-pipeline Xenna overrides
- Fixed-stride coverage in the NVCF split benchmark matrix
- Real-inference smoke test for vLLM captioning health
- Upgrade to CUDA 13.0
- Upgrade vLLM to 0.19.0
- Upgrade Ray to 2.55.0 (with the `serve` extra)
- Upgrade cosmos-xenna to 0.2.3
- Bump `av` to `>=17,<18` and add the `mcap` dependency for the sensor library

### Fixed

- `SamplingGrid` produced incorrect windows for irregular grids
- `--execution-mode` CLI flag is now honored end-to-end
- Cosmos-Embed1 writes per-variant embedding directories
- Symlink the host pixi path so shebangs resolve inside the local-launched container
- Sensor library uses read-only views to avoid accidental buffer mutation
- Add Qwen3 preprocessing logic for filtering stages
- Use pre-built images for benchmark runs to avoid redundant builds
- Remove external storage dependency from `ImageSensor`
- Semantic filter updates and dedup pipeline input path cleanup
- Loosen Cosmos-Reason1 caption similarity threshold to reduce flakiness

### Changed

- Replace `CurationPhase` / `PipelineBuilder` with factory functions (`*_builders.py`); the
  `phase_interface` module and per-pipeline `phases.py` files are removed
- Add `config: VllmConfig` parameter to `VllmPlugin.make_llm_input` for image vs video
  modality selection; subclasses must update their signature
- Switch CI Slurm and k8s GPU jobs to the slim image with in-container `pixi install` and
  `pixi run --as-is`
- Change CI NVCF backend
- Normalize the `SamplingGrid` API and make sampling windows explicit (no sentinel boundaries)
- Update semantic filter stages to use `VllmCaptioning`
- Add a CPU-only Paddle option for the `unified` env
- Pixi lockfile refreshed for CVE coverage
- Add notice and disclaimer to README and Docker image

### Documentation

- Speed-of-light design doc for captioning throughput, with refined SOL baseline methodology
  using `vllm bench` as the reference
- Refined Ray Data runner design with the first implementation slice
- Document `--*-cpus-per-worker` tuning knobs
- Add `--squash-before-merge` to MR guidelines

## [1.2.2]

### Released

- 2026-03-24

### Added

- `--slim` flag and `--pixi-path` for lightweight image builds
- `--transcode-max-output-frames` to limit clip frame count
- OpenAI-compatible endpoint for video embedding
- Pre-populate timestamps on `Video` during download
- Multistage Docker builds to reduce container image size
- Docker buildx cache for faster image builds
- Set vLLM `performance_mode` for improved inference throughput
- Upgrade vLLM to 0.17.1
- Upgrade Ray to 2.54.0
- Upgrade cuML to 26.0.2
- Upgrade cosmos-xenna to 0.2.1
- Upgrade Python to 3.12.13

### Fixed

- Remove pycuda dependency, use PyNvVideoCodec built-in context
- Purge `filtered_clips` in `MetadataWriterStage` cleanup
- `Video.get_major_size()` should include filtered clips
- Type mismatch for `qwen-gpus-per-worker`
- Remove syntax warnings in stages
- Pin `importlib-metadata` to avoid version conflicts
- Output built Docker images to the local Docker image store
- Remove vllm from local extra to eliminate litellm from lock file
- Update type annotations for AV pipeline

### Removed

- `ffmpeg_gpu` decode mode

### Changed

- Replace FFmpeg source build with conda-forge package
- Remove support for [Phi-4](https://huggingface.co/microsoft/Phi-4-multimodal-instruct) captioning to keep a
  security floor of `pillow>=12.1.1` (GHSA-cfh3-3jmp-rvhc)

### Documentation

- Add batch processing guide for large video sets
- Add slim image design doc

## [1.2.1]

### Released

- 2026-03-10

### Added

- Separate OpenAI endpoints for caption and enhance stages
- Build CPU-only ffmpeg by default for LGPL-compatible images
- Allow `QwenVideoClassifier` stage to be configurable

### Fixed

- Fix tracing flush lifecycle and embed profiling inside pipeline functions
- Always use `docker buildx build` to avoid legacy builder errors
- Defer `flush_tracing()` until after traced span exits to prevent closed-file ValueError

### Changed

- Fold `RemuxStage` into `VideoDownloader`

## [1.2.0]

### Released

- 2026-03-04

### Added

- Composable pipeline API via `CurationPhase` and `PipelineBuilder` for declarative pipeline construction
- OpenAI-compatible API captioning stage for using external LLM endpoints
- LazyData for zero-copy split-field pipeline transport, reducing memory overhead
- Automatic CPU and memory profiling for all pipeline stages
- Stage replay for re-running individual stages without full pipeline re-execution
- Unified write abstraction for local and remote storage
- Multi-camera splitting pipeline (data model, task creation, download/remux, frame extraction, clip transcoding, clip
  writer, and summary writer)
- ARM64 CLI and container build support
- GB200 support for loading Qwen3-VL-235B
- Optional Ray token authentication
- Upgrade vLLM to 0.15.1
- Upgrade cosmos-xenna to 0.2.0
- Upgrade ffmpeg to 8.0.1
- `QwenVideoClassifier` stage for video classification using Qwen VL
- Remove flash-attn dependency in favor of PyTorch SDPA

### Fixed

- **Critical: fix caption ordering bug in inflight batching.** When inflight batching was enabled (the default),
  captions could be assigned to the wrong videos. The bug was introduced in v1.1.5, was dormant in v1.1.6 (inflight
  batching temporarily removed), and has been active in v1.1.7–v1.1.11. If you used VLM captioning with any of those
  releases, captions may be mismatched. Upgrade to v1.2.0 and re-run affected captioning jobs.
- Enforce exact `--limit` semantics for storage listings and add `num_input_videos_selected` metric
- Reset `LazyData.nbytes` on drop and eliminate `tobytes` copy in upload path
- Update conda environment name from `vllm` to `unified` in Qwen filter stages
- Harden NVCF split benchmark retries and count validation
- Resolve Docker build failures from NVIDIA wheel timeouts and file permissions
- Check for remote mounts in `curator_submit`
- Handle clips with no stream
- Pin setuptools<81 to preserve `pkg_resources` for ngcsdk
- Add minimum version constraints for typer dependency
- Ensure `split_video_into_windows` returns equal-length lists

### Documentation

- Add Ray Data runner design document
- Update end user guide

## [1.1.11]

### Released

- 2026-01-09

### Known Issues

- **Caption ordering bug:** Inflight batching (enabled by default) can assign captions to the wrong videos. Fixed in
  v1.2.0.

### Added

- Add support for Cosmos-Reason2-8B as an alternative VLM captioning model
- Conform shard pipeline output folder name to include duration
- Add configurable sharding parameters to the video shard pipeline
- Add a Ray Data-based hello world pipeline example

## [1.1.10]

### Released

- 2025-12-18

### Known Issues

- **Caption ordering bug:** Inflight batching (enabled by default) can assign captions to the wrong videos. Fixed in
  v1.2.0.

### Added

- Improve sharding pipeline input gathering time
- Release new helm chart 2.2.1 that improves robustness of metrics collection
- Add support for Lance outputs for clips and embeddings
- Upgrade python from 3.10 to 3.12
- Add FP8 variant of Qwen3-VL-235B which can run on 4x H100s
- Add FP8 variant of Qwen3-VL-30B which can run on a single 48GB GPU
- Upgrade cosmos-xenna to 0.1.8 with support for online-serving mode

### Fixed

- Local launch CLI when specifying GPU list
- Parquet output format for Cosmos Dataset Search (CDS)
- Race condition with --copy-weights-to by passing it only to the model captioning stage but not the prepare stage
- Upgrade vllm in develop environment to match what is used inside container
- Remove async engine code in qwen_vl
- Fix Qwen3-VL models regarding pre-processing

## [1.1.9]

### Released

- 2025-12-08

### Known Issues

- **Caption ordering bug:** Inflight batching (enabled by default) can assign captions to the wrong videos. Fixed in
  v1.2.0.

### Added

- Add support for Qwen/Qwen3-VL-235B-A22B-Instruct
- Save model_input tensor input as pngs
- Wire vllm sampling params into splitting cli
- Switch enhance captions to OpenAI V1 Responses API
- Expose setup_on_node in stage_interface

### Fixed

- Fixed Nemotron-Nano VL as the captioning algorithm.
- Upgrade vllm to 0.11.2 and add metadata field to fix nemotron-nano-v2-vl
- Replace softprops/action-gh-release with gh release command
- Nemotron: change VideoMetadata to dict, model_does_preprocess=True
- Fix a bug in windowing which made us always lose 1 frame
- Bump ray version, unset vars not used in CI
- Dimensions aligned to i4
- Race condition in --copy-weights-to

## [1.1.8]

### Released

- 2025-11-17

### Known Issues

- **Caption ordering bug:** Inflight batching (enabled by default) can assign captions to the wrong videos. Fixed in
  v1.2.0.

### Added

- Nemotron-Nano-12B-v2-VL as an alternative VLM captioning model
- Gemini API as an option for video captioning
- Improved helm chart to simplify vanilla k8s deployment
- Upgraded cosmos-xenna to 0.1.7 for better scalability
- Significantly improved test coverage

### Fixed

- Fixed a bug in clip windowing utils which caused wrong caption for later windows within a clip
- Allow underscore in S3 bucket name
- Set cudagraph mode to piecewise for Qwen-based VL models to mitigate failure with illegal memory access
- Improved exception handling in vllm-captioning stage setup and process

### Documentation

- Added documentation
  for [vllm_interface](https://github.com/nvidia-cosmos/cosmos-curate/tree/main/docs/curator/design/vllm-interface.md)
  which simplifies the integration of new vLLM-powered VLMs for captioning.

## [1.1.7]

### Released

- 2025-10-30

### Known Issues

- **Caption ordering bug:** Inflight batching (enabled by default) can assign captions to the wrong videos. Fixed in
  v1.2.0.

### Added

- Azure OpenAI API as an option to enhance captions
- Increased test coverage for vllm_interface to 100%
- Azure Blob Storage support for Slurm deployments
- Support multipart result zips
- Update python version to 3.10.19
- Retry vllm captioning on engine failure

### Fixed

- Switch torch package to pypi in unified
- Resolve hello_world pipeline execution with transformers
- vLLM stage 2 captioning bug

## [1.1.6]

### Released

- 2025-10-16

### Added

- An example workflow script to operate X nvcf function to run M jobs
- Upgrade vllm to 0.11.0
- Upgrade transformers to 4.57.0
- Agent context files for Codex, Claude, and Gemini
- Runner abstraction for pipeline execution
- Increase test coverage

### Fixed

- Allow extra environment variables to be passed to the pixi runtime env
- Let slurm env setting override defaults inside container
- Remove dependency on pynvml
- Remove `max_seq_len_to_capture` from vLLM engine creation
- Improve the speed for final summary generation
- Downgrade click dep version to fix ray and revive e2e nvcf test

## [1.1.5]

### Released

- 2025-09-26

### Known Issues

- **Caption ordering bug:** Inflight batching (enabled by default) can assign captions to the wrong videos. Fixed in
  v1.2.0.

### Added

- Upgrade to [cosmos-xenna 0.1.6](https://pypi.org/project/cosmos-xenna/0.1.6/) for improved performance.

### Changed

- Update default parameters for stages' cpu resource requests for higher throughput

## [1.1.4]

### Released

- 2025-09-17

### Added

- Add [gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b) as an option for `EnhanceCaption` stage.
- Enable batching for internvideo2 embedding stage for improved throughput.
- Upgrade to [cosmos-xenna 0.1.5](https://pypi.org/project/cosmos-xenna/0.1.5/) for improved performance and stability.

## [1.1.3]

### Released

- 2025-09-08

### Added

- Release Grafana dashboard for pipeline monitoring.
- Add inflight batching for VLM captioning throughput.

### Changed

- Merge `video-splitting` env into `unified` env.
- Improve Slurm instructions.

## [1.1.2]

### Released

- 2025-08-28

### Added

- Upgrade to [cosmos-xenna 0.1.3](https://pypi.org/project/cosmos-xenna/0.1.3/) for improved scalability and
  observability.
- Enable Semantic Deduplication on Ray and improve IO efficiency for improved throughput.

## [1.1.1]

### Released

- 2025-08-13

### Added

- Add stage2 caption support to VLLMCaptionStage
- Add Nsight Systems for CUDA profiling

### Fixed

- Avoid unnecessary post-install docker layers
- Pin Ray to the same version for both pixi and poetry
- Update slurm cli to work with pixi

## [1.1.0]

### Released

- 2025-08-12

### Added

- Use [pixi](docs/developer-guide.md#working-with-pixi-environments) to manage environments inside container image
- Use absolute URL for [cosmos-xenna](https://github.com/nvidia-cosmos/cosmos-xenna) submodule; PLEASE run
  `git submodule sync` after pulling update
- Support for [Cosmos-Reason1](https://github.com/nvidia-cosmos/cosmos-reason1) as an alternative model for captioning
- Support for running [Phi-4](https://huggingface.co/microsoft/Phi-4-multimodal-instruct)
  with [vLLM](https://docs.vllm.ai/en/latest/)

### Fixed

- Suppress warnings to make log more readable
- Make `/dev/shm` (and hence Ray object store) a fraction of system memory in local mode.

## [1.0.2]

### Released

- 2025-07-28

### Added

- Support for using multiple GPUs in captioning stage to enable large models
- Support for generating dataset to
  post-train [Cosmos-Predict2](https://github.com/nvidia-cosmos/cosmos-predict2/blob/main/documentations/post-training_video2world.md)
- Support for [Phi-4](https://huggingface.co/microsoft/Phi-4-multimodal-instruct) as an alternative model for captioning

### Fixed

- PyNvCodec path for video decoding by fixing NVIDIA_DRIVER_CAPABILITIES env var
- CLI to import existing NVCF functions

## [1.0.1]

### Released

- 2025-07-11

### Added

- Multi-camera AV video split and caption pipelines
- Semantic-deduplication pipeline
- Support for [Cosmos-Embed1](https://research.nvidia.com/labs/dir/cosmos-embed1/) embedding model
- Support for using pre-signed URLs as input and output paths

### Fixed

- Splitting & transcoding accuracy for MPEG-TS files

### Changed

- Update required python version from 3.10.14 to 3.10.18

### Security

- Upgrade base image and packages to mitigate security vulnerabilities

## [1.0.0]

### Released

- 2025-06-11

### Added

- Initial version
