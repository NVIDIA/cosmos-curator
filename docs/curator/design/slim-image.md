# Slim Image Design

> **Note:** "Image" throughout this document refers to the Docker/OCI container image built by
> `cosmos-curator image build`.

> **Phasing note:** Phases 1-2 (removing `ffmpeg_gpu`, switching to conda-forge FFmpeg) and Phase 4 (cleanup) are
> valuable independently of the slim image work — they unblock CUDA 13, simplify compliance, and reduce image size for
> all modes. Only Phase 3 depends on the slim mode design and its shared storage assumptions.

> **Current FFmpeg packaging note:** The conda-forge FFmpeg plan in this design has been superseded for distribution
> compliance. Pixi no longer locks conda `ffmpeg`/`openh264`; images build FFmpeg 8.1.1 into `/opt/ffmpeg` and
> full images rebuild PyAV against it. Slim images keep the locked PyPI PyAV wheel and use `/opt/ffmpeg` for CLI work.
> Image builds use the non-redistributable runtime policy by default; pass `--redistributable` for the redistributable
> variant.

## Motivation

1. **Local dev velocity**: A slim image rebuilds in seconds (just source code, no conda install). Mounting the host's
   pixi environments via `--pixi-path` avoids rebuilding the image entirely when iterating on code or dependencies.
2. **Image size**: Baked images with all environments pre-installed are enormous, causing slow push/pull on GitLab CI,
   NVCF, Slurm, and internal registries (throttling, bandwidth waste, layer timeout failures).
3. **CUDA 13 unblocked**: The custom FFmpeg source build was blocking the CUDA 13 upgrade. Replacing it with conda-forge
   FFmpeg removes that blocker.
4. **Compliance & security**: Replacing the custom FFmpeg source build with conda-forge packages eliminates
   custom-compiled binaries — smaller SBOMs, minimal license obligations, cleaner scans.
5. **Scaling**: With `.pixi` on shared storage (e.g. PVC on k8s), a pre-warm step installs environments once and all
   workers use them directly — no per-worker install overhead.

## High-Level Design

**Two image modes** via `cosmos-curator image build [--slim]`:

- Default (full) — pre-installs pixi environments at build time. Best for platforms without shared/persistent storage
  (NVCF, air-gapped deployments). With no `--envs` flag, installs all environments. With `--envs env1,env2,...`,
  installs only the specified subset (e.g. `--envs default,unified`).
- `--slim` — lockfile (`pixi.toml` + `pixi.lock`) and source code only. `--envs` selects which environments to
  install at runtime (same default as full mode). The image stores the env list in `COSMOS_CURATOR_SLIM_ENVS` and the
  launch flow runs `pixi install --frozen` before the user command. Combined with `--pixi-path .`, this is ideal for
  local development (near-instant rebuilds, the install is a fast no-op). Cluster use (Slurm/k8s with shared storage)
  is promising but needs validation.

**Dependency strategy:**

- FFmpeg from conda-forge (LGPL) via the `av` (PyAV) conda package. Currently `av` is a PyPI dependency in `pixi.toml`,
  which bundles its own FFmpeg build. To get the conda-forge FFmpeg (with NVENC support), `av` must be switched to a
  conda dependency so it pulls in conda-forge FFmpeg as a transitive dependency. conda-forge FFmpeg is a strict superset
  of the current source build's codecs (libopenh264, libdav1d, libaom, libvpx, libwebp, libvorbis, vaapi, plus extras
  like libsvtav1, libmp3lame, libopus, libjxl). The conda-forge LGPL build also includes `h264_nvenc` and `hevc_nvenc`
  via the `ffnvcodec-headers` package (MIT-licensed, LGPL-compatible). Verified: `h264_nvenc` encoding works on RTX GPUs
  with the LGPL conda-forge build. **Important:** conda-forge ships both GPL (`ffmpeg=*=gpl_*`) and LGPL
  (`ffmpeg=*=lgpl_*`) variants — pin the LGPL variant in `pixi.toml` (e.g. `ffmpeg = "=*=lgpl_*"`).
- GPU video decode via PyNvVideoCodec (`pynvc` mode, already the preferred path in `VideoFrameExtractionStage`).
  The `ffmpeg_gpu` decode mode is removed — benchmarks showed it performed the same as `ffmpeg_cpu`, while `pynvc`
  is measurably faster. Removing it eliminates the only usage of `scale_npp` (libnpp) and the need for a custom
  GPU FFmpeg build.
- GPU FFmpeg transcoding (`h264_nvenc`) retained for teams with NVENC-capable GPUs (e.g. RTX in data center). CPU
  `libopenh264` remains the default encoder for GPUs without NVENC hardware (A100, H100).

**Shared cache, local install (slim mode):** The pixi package cache is mounted from shared storage (Lustre on Slurm,
PVC on k8s) so packages are downloaded once. Each container runs `pixi install --frozen` at startup, writing `.pixi` to
the container's local writable overlay (RAM-backed on Slurm compute nodes). This gives fast sequential reads from the
shared cache and fast metadata-heavy Python imports from local storage. Mounting `.pixi` itself from shared storage was
tested but rejected — Lustre metadata latency made Python imports slower than the full image.

**Runtime invariant — `pixi run --as-is`:** All runtime `pixi run` invocations use `--as-is`, which skips environment
validation and assumes environments are already installed. This is a safety guarantee, not just a performance
optimization: if an environment is missing, the command fails immediately instead of silently installing on the fly
(which could hang, race with concurrent Ray workers, or produce non-reproducible results). The invariant holds for both
image modes — full images pre-install at build time, slim images pre-install via the warmup step. The only exception is
local dev commands outside a container (e.g. `pixi run -e unified python ...`), where environments may not yet be
installed.

## Limitations and risks

1. **Slim mode is unconventional.** The standard container practice — including pixi's own documentation — is to install
   dependencies at build time and ship a self-contained image. Deferring installation to runtime is not a well-trodden
   path. It trades image size for startup complexity and requires infrastructure (shared storage, pre-warm scripts) that
   most deployments don't have.

2. **Shared storage dependency.** For cluster use, slim mode only makes sense when shared persistent storage is available
   (PVC on k8s, Lustre on Slurm). Without it, every worker downloads and installs independently, which is slower than
   pulling a pre-built image. Platforms like NVCF have no shared storage, so they must use full mode — and those images
   are still large. (Local dev avoids this issue by mounting the host `.pixi` directory directly.)

3. **Network access at runtime.** Slim mode requires network access to conda-forge (and PyPI for some packages) during
   the pre-warm step. Air-gapped environments cannot use slim mode at all.

4. **Pre-warm adds orchestration complexity.** The pre-warm script must run before Ray workers start, on the same shared
   storage, and must complete successfully. Failures (network issues, disk full, permission errors) block the entire job.
   This is a new failure mode that doesn't exist with pre-built images.

5. **Per-container install overhead.** Each container installs its own `.pixi` from the shared cache at startup.
   With a warm cache this takes ~1-2 minutes. For short-lived jobs this overhead may be significant; for typical
   pipeline runs (10+ minutes) it is negligible and faster overall than pulling the full image.

6. **Full mode doesn't solve the size problem.** For platforms that need pre-built images (NVCF, air-gapped), the
   image size remains large. Phases 1-2 (removing the FFmpeg source build) help, but the bulk of the image size comes
   from conda environments and model dependencies, which this design does not address.

## Task List

### Phase 1: Remove `ffmpeg_gpu` decode path

Benchmarks showed `ffmpeg_gpu` performed the same as `ffmpeg_cpu` (GPU decode savings negated by CPU scaling),
while `pynvc` is measurably faster. This removal is justified independently of the conda-forge switch.

- [x] **1a. Remove `ffmpeg_gpu` decoder mode from `VideoFrameExtractionStage`**
    - Drop the `ffmpeg_gpu` choice from `--transnetv2-frame-decoder-mode` CLI arg (`splitting_pipeline.py`)
    - Remove the `use_gpu=True` branch in `get_frames_from_ffmpeg()` (`frame_extraction_stages.py`)
    - Update `VideoFrameExtractionStage` to remove the `ffmpeg_gpu` resource/logic path
    - Eliminates the only usage of `scale_npp` (libnpp) — the sole reason for a custom GPU FFmpeg build
    - `pynvc` remains as the GPU decode option; `ffmpeg_cpu` remains as CPU fallback

### Phase 2: Replace source-built FFmpeg with conda-forge

- [x] **2a. Switch `av` from PyPI to conda-forge and verify FFmpeg codec parity**
    - Move `av` from `[pypi-dependencies]` to `[dependencies]` in `pixi.toml` so it pulls conda-forge FFmpeg
    - Pin `ffmpeg = "=*=lgpl_*"` to ensure the LGPL variant is used (not GPL)
    - The PyPI `av` package bundles its own FFmpeg and does not include conda-forge FFmpeg or NVENC support
    - Build a test image without the source FFmpeg build, relying only on conda-forge FFmpeg
    - Run the CPU video pipeline end-to-end (download, split, transcode with libopenh264, write)
    - Confirm `ffprobe` is available from conda-forge FFmpeg
    - Verify all codecs used in practice: libopenh264 encode, libdav1d/libaom decode, remux
    - Verify `h264_nvenc`/`hevc_nvenc` are present (conda-forge includes them via `ffnvcodec-headers`)
    - Test GPU transcoding on an NVENC-capable GPU (e.g. RTX)

- [x] **2b. Remove `--ffmpeg-cuda` build flag and GPU FFmpeg source build**
    - Remove `--ffmpeg-cuda` option from `image_app.py`
    - Remove all `ffmpeg_cuda` conditional blocks from `default.dockerfile.jinja2`
    - No longer needed: conda-forge FFmpeg includes `h264_nvenc`/`hevc_nvenc` in the LGPL build

- [x] **2c. Remove FFmpeg source build from Dockerfile**
    - Remove the entire FFmpeg source build block from `default.dockerfile.jinja2`
    - Remove apt dependencies only needed for FFmpeg compilation (autoconf, automake, cmake, yasm, nasm, libtool, etc.)
    - Keep system deps still needed elsewhere (libsm6, libxext6 for OpenCV)

### Phase 3: Slim image (lockfile-only)

- [x] **3a. Add `--slim` flag to `cosmos-curator image build`**
    - `--slim`: skip `pixi install`, image contains only lockfile + source
    - `--envs` selects which environments to install at runtime (same default as full mode); the list is stored in the
      image as `ENV COSMOS_CURATOR_SLIM_ENVS` and `LABEL cosmos-curator.slim=true`
    - Default (no flag): pre-install pixi environments at build time, `--envs` selects subset
    - Full mode retains the NVIDIA wheel pre-download hack and retry logic

- [x] **3b. Add `--pixi-path` flag to `cosmos-curator local launch`**
    - Mount the host `.pixi` directory into the container (envs, cache, config)
    - For local dev, `--curator-path . --pixi-path .` mounts both source and `.pixi` from the project root
    - For cluster deployments, `--pixi-path /mnt/shared/pixi` can point to shared storage independently
    - Enables near-instant iteration: slim image + host envs + host source, no image rebuild needed

- [ ] **3c. Get pixi approved for open-source release and bundle source in image**
    - Get pixi (MIT-licensed, single static Rust binary) approved for inclusion in the shipped image
    - Bundle pixi source tarball at pinned version (e.g.
      `ADD https://github.com/prefix-dev/pixi/archive/refs/tags/${PIXI_VERSION}.tar.gz /opt/oss-sources/pixi.tar.gz`)
    - ~17MB compressed, source archival only — no build needed
    - Can proceed in parallel with code work; gates shipping, not development

- [x] **3d. Auto-warmup for local launch**
    - `local launch` prepends a conditional `pixi install --frozen` to the container command
    - Reads `COSMOS_CURATOR_SLIM_ENVS` from the image; no-op when the variable is unset (full images)
    - With `--pixi-path`, environments are already present so the install is a fast no-op
    - Without `--pixi-path`, installs environments into the container's ephemeral filesystem

- [x] **3e. Mount pixi cache from shared storage for cluster deployments**
    - Mount the pixi package cache (not `.pixi` itself) from Lustre into the container at `/pixi-cache`
    - Set `PIXI_CACHE_DIR=/pixi-cache` so `pixi install` reads packages from the shared cache
    - Mounting `.pixi` directly was tested but rejected — Lustre metadata latency made Python imports
      slower than the full image

- [x] **3f. Add auto-warmup to sbatch template**
    - `sbatch.sh.j2` wraps the srun command with a conditional `pixi install --frozen` preamble
    - Reads `COSMOS_CURATOR_SLIM_ENVS` from the image; no-op when unset (full images)
    - Installs environments inside the container's writable overlay (RAM-backed on compute nodes)
    - Same pattern as `local launch` auto-warmup (3d)

- [x] **3g. Validate slim image on Slurm CI**
    - Switched `gpu_tests` and `slurm_end_to_end` CI jobs to the slim image
    - Results (excluding Slurm queue wait): `gpu_tests` 7:15 vs 9:42 full (-25%),
      `slurm_end_to_end` 14:30 vs 15:53 full (-9%)
    - Faster than full image due to smaller image pull + fast local `.pixi` on RAM-backed overlay

- [x] **3h. Validate slim image on k8s CI**
    - Switched `k8s_gpu_tests` to the slim image with in-container `pixi install` from a persistent
      hostPath cache (`/cache/pixi`)
    - `pixi install` runs from `/opt/cosmos-curator` (where the slim image has source + lockfile),
      not from the GitLab checkout directory
    - Results: slim pod startup 9s + pixi install 37s = 46s overhead, total 7:43.
      Full image pod startup varies widely depending on image layer caching: 7s (cached) to 4:40
      (uncached), with 4 of 5 sampled runs uncached (2:19–4:40). Total full job durations:
      7:25–11:18. The slim image matches a best-case cached pull and consistently avoids the
      multi-minute uncached pulls

- [x] **3i. Use `pixi run --as-is` to skip per-invocation environment validation**
    - `pixi run --as-is` skips environment validation entirely — assumes environments are already
      installed. Without it, each Ray worker spawn re-validates (and potentially re-installs)
      environments, which caused worker spawn timeouts in k8s CI
    - Applied to `PixiRuntimeEnv` (Ray actor spawns), CI scripts, Dockerfile post-install steps,
      and all in-container docs examples
    - Local dev commands (developer-guide.md) left without `--as-is` since envs may not be
      pre-installed

### Phase 4: Cleanup and optimization

- [x] **4a. Remove deprecated remux stage class**
    - Already marked for removal by 2026-04-30
    - Removes obsolete stage API; inline remux remains in `VideoDownloader`

- [ ] **4b. Slim down system apt dependencies**
    - Audit which apt packages are still needed without the FFmpeg source build
    - Remove build-only tools that conda-forge FFmpeg doesn't need

### Future (lower priority)

- [ ] **Pixi base image**
    - Switch from `nvcr.io/nvidia/cuda:*-devel-ubuntu24.04` to `ubuntu:24.04` with pixi managing the CUDA toolkit
    - Move CUDA toolkit into the pixi environment itself
    - Would further reduce image size and simplify the Dockerfile
