<p align="center">
    <img src="docs/assets/nvidia-cosmos-header.png" alt="NVIDIA Cosmos Header">
</p>

### [Product Website](https://www.nvidia.com/en-us/ai/cosmos/)

# Cosmos Curator

A toolkit for curating physical AI training data.

## Important

Please run `git submodule sync` if you have cloned the repository before and just pulled the latest update.
We updated the URL for `cosmos-xenna` submodule on 08/04/2025.

## Overview

Cosmos Curator helps teams build physical AI data curation workflows from reusable components and reference recipes.
It supports training data preparation for [Cosmos](https://www.nvidia.com/en-us/ai/cosmos/) at NVIDIA.

Start new workflows with [Curator Next](cosmos_curator/next/README.md), which combines Ray Data recipes, shared curation
components, and Lance datasets. Its APIs and data contracts are evolving during incubation. Existing Xenna and legacy
Ray Data pipelines remain available while their workflows transition to Next; the goal is to retire both implementations
after that transition.

## Features

- **Video Processing**: Efficient video splitting, annotation, filtering, deduplication, and dataset generation
- **AI-Powered Analysis**: Advanced video analysis using multiple model families
- **Distributed Computing**: Scalable processing with Ray Data; existing pipelines also use [Cosmos-Xenna](https://github.com/nvidia-cosmos/cosmos-xenna)
- **Cloud Integration**: Support for various platforms
- **Pipeline System**: Modular and extensible pipeline architecture

## Documentation

Comprehensive documentation is available under [docs/](docs/README.md) directory.

### User Documentation

- [Curator Next](cosmos_curator/next/README.md) - start new workflows with reference recipes and shared components
- [End User Guide](docs/client/end-user-guide.md) - instructions to setup environment and run data pipelines
- [Reference Video Pipelines Guide](docs/curator/reference/video-pipelines.md) - details for general and multi-camera video processing pipelines
- [NVCF Guide](docs/client/nvcf-guide.md) - deployment instruction on [Nvidia Cloud Functions](https://docs.nvidia.com/cloud-functions/user-guide/latest/cloud-function/overview.html)

### Developer Documentation
- [Developer Guide](docs/developer-guide.md) - information for contributors
- [Architecture Guide](docs/curator/reference/architecture.md) - diagrams and description to help understand core architecture
- [Xenna Pipeline Design Guide](docs/curator/guides/pipeline-design.md) - maintain existing Xenna pipelines, with a hello-world example and performance guidance
- [Observability Guide](docs/curator/guides/observability.md) - instructions to setup and understand monitoring dashboard

### AI Agent Context Files
- [AGENTS.md](AGENTS.md) - Canonical AI agent context

## Directory Structure

Key directories:

```text
cosmos-curator/
├── cosmos_curator/         # Curator implementation
│   ├── next/               # Starting point for new curation workflows
│   │   ├── recipes/        # Reference recipes
│   │   ├── core/           # Shared config, registration, and Ray execution support
│   │   ├── embeddings/     # Shared embedding components and schemas
│   │   ├── media/          # Shared media processing
│   │   └── utils/          # Shared storage, identity, and publication utilities
│   ├── client/             # Deployment CLIs and pipeline config tooling
│   ├── core/               # Interfaces, managers, and common utilities
│   │   └── sensors/        # Self-contained sensor library
│   ├── models/             # Model inference implementations
│   ├── configs/            # Model artifact registry
│   ├── pipelines/          # Existing pipelines and shared pipeline CLI runtime
│   │   ├── image/          # Existing Xenna image pipelines
│   │   ├── video/          # Existing Xenna video pipelines
│   │   └── ray_data/       # Deprecated Ray Data pipelines
│   └── scripts/            # Deployment startup scripts
├── tests/                  # Unit, integration, and GPU tests mirroring source paths
├── docs/                   # Guides, references, and design notes
├── examples/               # Example configs and scripts
├── package/                # Dockerfile templates and image setup scripts
├── distributable/          # Generated Pixi manifest and lockfile for redistributable images
├── charts/                 # Kubernetes Helm charts
├── benchmarks/             # Performance benchmarks and analysis
├── tools/                  # Repository maintenance tools
└── cosmos-xenna/           # Git submodule supporting existing Xenna pipelines
```

Note: To initialize and update the `cosmos-xenna` submodule, run:

```bash
git submodule update --init --recursive
```

This ensures all submodule content is checked out correctly.

## Support

For support and questions:
- Check the [documentation](docs/README.md)
- Open an issue on GitHub

## Acknowledgments

- [cosmos-xenna](https://github.com/nvidia-cosmos/cosmos-xenna) team for the core library
- All contributors and users of the project

## Responsible Use of AI Models
[Responsible Use](./RESPONSIBLE_USE.md)

## Notice and Disclaimer

**NOTICE AND DISCLAIMER:** This software automatically retrieves, accesses
or interacts with external materials. Those retrieved materials are not
distributed with this software and are governed solely by separate terms,
conditions and licenses. You are solely responsible for finding, reviewing
and complying with all applicable terms, conditions, and licenses, and for
verifying the security, integrity and suitability of any retrieved materials
for your specific use case. This software is provided "AS IS", without
warranty of any kind. The author makes no representations or warranties
regarding any retrieved materials, and assumes no liability for any losses,
damages, liabilities or legal consequences from your use or inability to use
this software or any retrieved materials. Use this software and the
retrieved materials at your own risk.

## License and Contact

This project will download and install additional third-party open source software projects. Review the license terms of these open source projects before use.

NVIDIA Cosmos source code is released under the [Apache 2 License](https://www.apache.org/licenses/LICENSE-2.0).

NVIDIA Cosmos models are released under the [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license). For a custom license, please contact [cosmos-license@nvidia.com](mailto:cosmos-license@nvidia.com).
