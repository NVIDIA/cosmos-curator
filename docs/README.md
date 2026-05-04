# Cosmos Curator Documentation

- [Cosmos Curator Documentation](#cosmos-curator-documentation)
  - [Documentation Structure](#documentation-structure)
    - [1. End User Guide and Reference Pipelines Guide for Video, AV, and Image](#1-end-user-guide-and-reference-pipelines-guide-for-video-av-and-image)
    - [2. Architecture Guide and Pipeline Design Guide](#2-architecture-guide-and-pipeline-design-guide)
    - [3. Observability Guide and Deployment Guide for NVCF](#3-observability-guide-and-deployment-guide-for-nvcf)
    - [4. Developer Guide](#4-developer-guide)
    - [5. Curator Pipeline Docs](#5-curator-pipeline-docs)
  - [Guide Relationships](#guide-relationships)
  - [Getting Started](#getting-started)
  - [Support](#support)
  - [Contributing to Documentation](#contributing-to-documentation)

Welcome to the Cosmos Curator documentation. This documentation is organized into five main categories to help you understand and use the system effectively.

## Documentation Structure

### 1. [End User Guide](client/end-user-guide.md) and Reference Pipelines Guide for [Video](curator/reference/video-pipelines.md), [AV](curator/reference/av-pipelines.md), and [Image](curator/reference/image-pipeline.md)
The end user guide is designed for users who want to use the Cosmos Curator to process and curate videos and images:
- Prerequisites and environment setup
- Basic usage instructions
- Running pipelines
- Troubleshooting common issues

The reference pipelines guides are for users to understand the details of reference pipelines (video, AV, and image):
- Output directory structure of each reference pipeline
- Explanation of the configuration parameters

**When to use these guides:**
- If you're new to Cosmos Curator
- If you want to run existing pipelines
- If you need help with basic operations

### 2. [Architecture Guide](curator/reference/architecture.md) and [Pipeline Design Guide](curator/guides/pipeline-design.md)
The architecture and pipeline design guides are for developers who want to create new pipelines or modify existing ones:
- Core architecture
- Pipeline development
- New model integration
- Performance optimization guidelines

**When to use these guides:**
- If you're creating a new pipeline
- If you're modifying an existing pipeline
- If you need to understand the pipeline architecture

### 3. [Observability Guide](curator/guides/observability.md) and Deployment Guide for [NVCF](client/nvcf-guide.md)

The observability and deployment guides are for users who want to run curation pipelines at scale:
- List of key PromQL queries
- Monitoring dashboard
- Deployment instructions on [NVCF](https://docs.nvidia.com/cloud-functions/user-guide/latest/cloud-function/overview.html) and K8s (coming soon)

**When to use this guide:**
- If you want to run pipelines at scale
- If you want to optimize the pipeline performance

### 4. [Developer Guide](developer-guide.md)
The developer guide is for contributors who want to work on the Cosmos Curator codebase:
- Development environment setup
- Code quality standards
- Testing procedures
- Contribution guidelines

**When to use this guide:**
- If you're setting up a development environment
- If you're contributing to the codebase
- If you need to understand the development workflow

**Additional Development Resources:**
- [Stage Replay Guide](curator/guides/stage-replay.md) - Debug pipeline stages in isolation without re-running entire pipelines
- [Interactive Slurm Development Guide](curator/guides/slurm-interactive.md) - Iterate on Cosmos Curator from an interactive Slurm allocation

### 5. [Curator Pipeline Docs](curator/README.md)
The curator docs are organized by purpose:
- Guides for workflows and debugging tasks
- Reference docs for current behavior, APIs, output formats, and operational contracts
- Design docs for architecture direction, rationale, plans, and open questions

## Guide Relationships

```mermaid
graph TD
    A[End User Guide] -->|Run| B[Pipelines]
    C[Reference Pipelines Guide] -->|Configure| B[Pipelines]
    D[Architecture Guide] -->|Understand| E
    E[Pipeline Design Guide] -->|Modify/Create| B
    E -->|Follow| F[Developer Guide]
```

## Getting Started

1. **New Users**:
   - Start with the [End User Guide](client/end-user-guide.md) to learn how to use the system
   - Refer to the [Reference Video Pipelines Guide](curator/reference/video-pipelines.md) to explore the many options to configure the reference pipelines targeting general videos
   - Refer to the [Reference AV Pipelines Guide](curator/reference/av-pipelines.md) to explore the many options to configure the reference pipelines targeting autonomous vehicle
   - Refer to the [Reference Image Pipeline Guide](curator/reference/image-pipeline.md) for the image annotate pipeline, including filtering, embedding, captioning, outputs, and key configuration options
2. **Developers**:
   - Begin with the [Developer Guide](developer-guide.md) to set up your development environment
   - Read the [Architecture Guide](curator/reference/architecture.md) to understand the big picture
   - Use the [Pipeline Design Guide](curator/guides/pipeline-design.md) to modify existing pipelines or create new pipelines

## Support

If you need additional help:
- Review the examples and best practices
- Contact the development team

## Contributing to Documentation

If you find any issues or want to improve the documentation:
1. Follow the contribution guidelines in the [Developer Guide](developer-guide.md)
2. Make sure your changes are consistent with the existing documentation style
3. Update all relevant guides if your changes affect multiple areas
