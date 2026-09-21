# Cosmos Curator Docs

This directory collects documentation for Cosmos Curator. It is organized by the dominant purpose of each document:

- **Guides** teach workflows and debugging tasks.
- **Reference** documents current behavior, APIs, output formats, and operational contracts.
- **Design** captures architecture direction, rationale, plans, and open questions.

## Guides

- [Pipeline Design Guide](guides/pipeline-design.md) - build or modify curator pipelines.
- [Curate Runbook](guides/curate-runbook.md) - run the Curate selection leg (fuse, cluster, dedup, balance).
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

- [Cosmos Curator Next](design/curator-next.md) - agent-friendly toolkit direction centered on reusable components and
  maintained reference recipes.
- [Curator Next Video Cut](design/curator-next-video-cut.md) - plan fixed-stride spans in Lance.
- [Curator Next Video Transcode](design/curator-next-video-transcode.md) - checkpointed CPU transcoding and atomic clip
  metadata publication.
- [Curator Next Video Captioning](design/curator-next-video-captioning.md) - checkpointed Qwen3.8 inference and
  fragment-atomic Lance enrichment.
- [Curator Next Incremental Curation](design/curator-next-incremental-curation.md) - a canonical Lance clip table that
  grows through split fragments and dynamically added curation columns.
- [Deterministic Work Sharding](design/curator-next-sharding.md) - splitting one logical run into independently launched
  jobs that cover the work with no gap or overlap, and how that composes with row checkpointing.
- [Captioning Approaches](design/captioning-approaches.md) - comparison of captioning architectures.
- [Multi-Camera Design](design/multicam.md) - multi-camera data model and implementation plan.
- [Multimodal AV Pipeline Design](design/multimodal-av-pipeline.md) - Lance-backed aligned multiview AV clip
  contract with camera, IMU, GPS/GNSS features, and curation outputs.
- [Multimodal AV Episode Publication](design/multimodal-av-episode-publication.md) - canonical complete-camera episode
  rows, artifact bindings, identity, and publication semantics.
- [Multimodal AV Captioning](design/multimodal-av-captioning.md) - single-camera free-form caption enrichment of
  canonical multimodal episodes.
- [Release Versioning Design](design/release-versioning.md) - tag-derived release and package versioning.
- [Ray Data Design](design/ray-data.md) - Ray Data direction and implementation notes.
- [Ray Data Captioning Design](design/ray-data-captioning.md) - Qwen captioning through Ray Data LLM.
- [Split Comparison — Module Architecture](design/split-comparison.md) - module DAG and architecture for the Lance-based split-output comparison.
- [Sensor Library Design](design/sensor-library.md) - sensor data model and API direction.
- [Efficient Sparse Video Decode](design/sensor-library-efficient-video-decode.md) - efficient decode strategy for sampled video.
- [Data Integrity Design](design/data-integrity-design.md) - architecture, v1 API, and phased delivery plan for the sensor data-integrity framework.
- [Data Integrity Metrics](design/data-integrity-metrics.md) - initial metric catalog for sensor data-integrity checks.
- [Data Integrity Pipeline](design/data-integrity-pipeline.md) - Ray Data recipe running the data-integrity metrics across many sessions into one store.
- [Orca Agentic Curation](design/orca.md) - agentic orchestration direction for large-scale curation.
- [Managed Ray Clusters for Curator Pipelines on Slurm](design/curator-next-slurm-ray.md) - run-scoped Ray clusters built from independent Slurm jobs.
- [Schema-Validated Pipeline Configs](design/pipeline-configs.md) - Ray Data config input contract and implementation plan.
- [Slim Image Design](design/slim-image.md) - slim container image design and rollout plan.
- [Speed-of-Light Design](design/speed-of-light.md) - captioning throughput measurement and optimization direction.
- [vLLM Interface Design](design/vllm-interface.md) - vLLM interface architecture and API design.
