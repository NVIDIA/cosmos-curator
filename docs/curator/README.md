# Cosmos Curator Curator Docs

This directory collects documentation for the curator pipeline layer. It is organized by the dominant purpose of each document:

- **Guides** teach workflows and debugging tasks.
- **Reference** documents current behavior, APIs, output formats, and operational contracts.
- **Design** captures architecture direction, rationale, plans, and open questions.

## Guides

- [Pipeline Design Guide](guides/PIPELINE_DESIGN.md) - build or modify curator pipelines.
- [Stage Replay Guide](guides/STAGE_REPLAY.md) - debug stages in isolation.
- [Interactive Slurm Development Guide](guides/SLURM_INTERACTIVE.md) - iterate from an interactive Slurm allocation.
- [Profiling Guide](guides/PROFILING.md) - collect and inspect CPU and memory profiles.
- [Observability Guide](guides/OBSERVABILITY.md) - monitor pipeline health and metrics.
- [vLLM Async Captioning Guide](guides/VLLM_ASYNC_CAPTIONING.md) - use the async vLLM captioning path.
- [vLLM Interface Plugin Guide](guides/VLLM_INTERFACE_PLUGIN.md) - add a new vLLM model plugin.
- [vLLM Interface Debugging Guide](guides/VLLM_INTERFACE_DEBUG.md) - trace and debug vLLM captioning behavior.

## Reference

- [Architecture Guide](reference/ARCHITECTURE.md) - core architecture and execution model.
- [Artifact Transport Guide](reference/ARTIFACT_TRANSPORT.md) - local and remote artifact delivery.
- [Distributed Tracing Guide](reference/DISTRIBUTED_TRACING.md) - tracing API, configuration, and output.
- [Reference Video Pipelines](reference/VIDEO_PIPELINES.md) - video pipeline behavior, options, and outputs.
- [Reference AV Pipelines](reference/AV_PIPELINES.md) - AV pipeline behavior, options, and outputs.
- [Reference Image Pipeline](reference/IMAGE_PIPELINE.md) - image pipeline behavior, options, and outputs.

## Design

- [Captioning Approaches](design/CAPTIONING_APPROACHES.md) - comparison of captioning architectures.
- [Multi-Camera Design](design/MULTICAM.md) - multi-camera data model and implementation plan.
- [Ray Data Design](design/RAY_DATA.md) - Ray Data direction and implementation notes.
- [Sensor Library Design](design/SENSOR_LIBRARY.md) - sensor data model and API direction.
- [Efficient Sparse Video Decode](design/SENSOR_LIBRARY_EFFICIENT_VIDEO_DECODE.md) - efficient decode strategy for sampled video.
- [Slim Image Design](design/SLIM_IMAGE.md) - slim container image design and rollout plan.
- [Speed-of-Light Design](design/SPEED_OF_LIGHT.md) - captioning throughput measurement and optimization direction.
- [vLLM Interface Design](design/VLLM_INTERFACE.md) - vLLM interface architecture and API design.
