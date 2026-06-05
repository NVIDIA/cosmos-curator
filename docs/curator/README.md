# Cosmos Curator Curator Docs

This directory collects documentation for the curator pipeline layer. It is organized by the dominant purpose of each document:

- **Guides** teach workflows and debugging tasks.
- **Reference** documents current behavior, APIs, output formats, and operational contracts.
- **Design** captures architecture direction, rationale, plans, and open questions.

## Guides

- [Pipeline Design Guide](guides/pipeline-design.md) - build or modify curator pipelines.
- [Stage Replay Guide](guides/stage-replay.md) - debug stages in isolation.
- [Interactive Slurm Development Guide](guides/slurm-interactive.md) - iterate from an interactive Slurm allocation.
- [Profiling Guide](guides/profiling.md) - collect and inspect CPU and memory profiles.
- [Observability Guide](guides/observability.md) - monitor pipeline health and metrics.
- [vLLM Async Captioning Guide](guides/vllm-async-captioning.md) - use the async vLLM captioning path.
- [vLLM Interface Plugin Guide](guides/vllm-interface-plugin.md) - add a new vLLM model plugin.
- [vLLM Interface Debugging Guide](guides/vllm-interface-debug.md) - trace and debug vLLM captioning behavior.

## Reference

- [Architecture Guide](reference/architecture.md) - core architecture and execution model.
- [Artifact Transport Guide](reference/artifact-transport.md) - local and remote artifact delivery.
- [Caption Quality Stats](reference/caption-quality-stats.md) - run-level caption structural-health counters artifact.
- [Distributed Tracing Guide](reference/distributed-tracing.md) - tracing API, configuration, and output.
- [Reference Video Pipelines](reference/video-pipelines.md) - video pipeline behavior, options, and outputs.
- [Split Pipeline Stage Overview](reference/split-pipeline-stages.md) - stage-by-stage split-annotate pipeline catalog.
- [Reference Image Pipeline](reference/image-pipeline.md) - image pipeline behavior, options, and outputs.

## Design

- [Captioning Approaches](design/captioning-approaches.md) - comparison of captioning architectures.
- [Deprecation and Default Changes](design/deprecation.md) - proposed cleanup of legacy features and large-run defaults.
- [Multi-Camera Design](design/multicam.md) - multi-camera data model and implementation plan.
- [Release Versioning Design](design/release-versioning.md) - tag-derived release and package versioning.
- [Ray Data Design](design/ray-data.md) - Ray Data direction and implementation notes.
- [Ray Data Captioning Design](design/ray-data-captioning.md) - Qwen captioning through Ray Data LLM.
- [Split Output Comparison Design](design/split-output-comparison.md) - feature-comparator design for split output reports.
- [Sensor Library Design](design/sensor-library.md) - sensor data model and API direction.
- [Efficient Sparse Video Decode](design/sensor-library-efficient-video-decode.md) - efficient decode strategy for sampled video.
- [Orca Agentic Curation](design/orca.md) - agentic orchestration direction for large-scale curation.
- [Slim Image Design](design/slim-image.md) - slim container image design and rollout plan.
- [Speed-of-Light Design](design/speed-of-light.md) - captioning throughput measurement and optimization direction.
- [vLLM Interface Design](design/vllm-interface.md) - vLLM interface architecture and API design.
