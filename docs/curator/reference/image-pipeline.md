# Cosmos Curator - Reference Image Pipeline

- [Cosmos Curator - Reference Image Pipeline](#cosmos-curator---reference-image-pipeline)
  - [Annotate Pipeline](#annotate-pipeline)
    - [Annotate Pipeline Stages](#annotate-pipeline-stages)
    - [Annotate Pipeline Output Format](#annotate-pipeline-output-format)
    - [Already-processed Skip (Resume Behavior)](#already-processed-skip-resume-behavior)
    - [Annotate Pipeline Configurable Options](#annotate-pipeline-configurable-options)

The image reference pipeline provides:

- **Annotate pipeline** — Loads images from local disk or S3, can optionally run semantic filtering, classifier-based filtering, embedding generation, and captioning, then writes images, embeddings, metadata, and a summary to an output path.

## Annotate Pipeline

### Annotate Pipeline Stages

The annotate pipeline includes the following stages:

- **Ingest (Image Load)**: Loads image files from the input path (local or S3) into task payloads for downstream stages.
- **Filtering (optional)**:
  - **Semantic filter**: Uses a VLM prompt to determine whether an image matches rejection criteria. Can either reject matching images or run in score-only mode that annotates metadata without filtering.
  - **Image classifier**: Uses a VLM prompt plus an allow/block taxonomy to classify image type and optionally reject images that match blocked classes.
  - Like video captioning, these filtering flows are often split into a CPU prep stage, a model inference stage, and a lightweight postprocessing stage.
- **Embedding generation (optional)**: Produces one embedding vector per image using `internvideo2`, `clip`, `cosmos-embed1-*`, or an OpenAI-compatible embedding endpoint.
- **Captioning (optional)**:
  - **Caption prep**: Decodes image bytes, optionally resizes within pixel bounds, and builds model-specific input for local vLLM or OpenAI-compatible captioning paths.
  - **Caption**: Runs the selected backend to generate one caption per image. Supported backends include local vLLM variants, OpenAI-compatible endpoints, and Gemini.
- **Output (Image Writer)**: Writes passed images, filtered images, per-image metadata, optional embeddings, and a top-level `summary.json`.

One task corresponds to one image. Unlike the video split pipeline, the image annotate pipeline does not split one input asset into multiple output assets.

### Annotate Pipeline Output Format

The annotate pipeline produces the following artifacts under the path specified by `--output-path`:

```text
{output_path}/
├── images/                           # images that passed filtering
│   ├── {output_id}.jpg               # extension preserves the input suffix when possible
├── filtered_images/                  # images rejected by semantic/classifier filtering
│   ├── {output_id}.jpg
├── embeddings/
│   ├── clip/
│   │   ├── {output_id}.npy
│   ├── cosmos_embed1_336p/
│   │   ├── {output_id}.npy
│   ├── internvideo2/
│   │   ├── {output_id}.npy
│   ├── openai/
│   │   ├── {output_id}.npy
├── metas/
│   ├── {output_id}.json              # metadata for each image
├── summary.json                      # run summary for the full annotate pipeline
```

Each `metas/{output_id}.json` includes:

- `source_path`: full input path of the image
- `relative_path`: path relative to input root
- `width`, `height`: dimensions captured during image processing, typically after prep resize when a prep stage ran
- `has_caption`: `true` when `caption_status` is `success` or `truncated`
- `is_filtered`: whether the image was rejected by semantic filtering or classifier filtering
- `align_timestamp_ns`: sampled/reference timestamp in nanoseconds. Uses timestamps from `image_data` when provided; otherwise the pipeline uses synthetic timestamps.
- `sensor_timestamp_ns`: sampled sensor timestamp in nanoseconds. Uses timestamps from `image_data` when provided; otherwise the pipeline uses synthetic timestamps.
- `caption_status`: normalized caption outcome (`success`, `truncated`, `blocked`, `error`, `skipped`, or `null`) following the [normalized caption-outcome contract](../design/vllm-interface.md#caption-outcomes-and-metadata). `null` means no caption stage ran for this image row; `"skipped"` is a reserved status value and is distinct from `null`.
- `caption_failure_reason`: `exception`, `timeout`, or `null`; set only when `caption_status == "error"`
- `token_counts`: nested per-model token usage keyed by model variant, with `prompt_tokens` and `output_tokens` for each model
- `filter_caption_status`: per-model status for semantic-filter or classifier caption calls, when those stages are enabled
- `filter_caption_failure_reason`: failure reasons for filter/classifier caption calls, when present
- `qwen_type_classification`: classifier labels inferred by the image classifier postprocessing stage
- `qwen_rejection_stage`: which filtering stage rejected the image, such as `semantic` or `classifier`
- `qwen_rejection_reasons`: accumulated semantic/classifier rejection details
- `embedding_keys`: embedding backends written for the image
- `errors`: per-stage error payloads when something failed but the task still reached output
- `caption`: present only when `has_caption` is true, so `caption_status` is `success` or `truncated`

`summary.json` includes aggregate counters such as:

- `num_input_images`
- `num_output_tasks`
- `num_images_passed`
- `num_images_filtered`
- `num_images_with_caption`
- `num_images_with_embeddings`
- `embedding_backend`
- `resize_min_pixels`
- `resize_max_pixels`
- `images`
- `filtered_images`
- `captioned_images`

### Already-processed Skip (Resume Behavior)

When the same output path is used across runs, the pipeline treats images that already have output metadata as completed and skips them at input discovery time:

- At **input extraction**, the pipeline first checks `summary.json` and prefers the broadest available processed-image record:
  - `processed_images`, if present
  - otherwise `images` plus `filtered_images`
  - otherwise `captioned_images`
- If `summary.json` is missing or incomplete, the pipeline falls back to scanning `metas/*.json`.
- Any image whose output ID is already present is excluded from the new task list.


### Annotate Pipeline Configurable Options

A summary of important options is below. For the full list, run:

```bash
cosmos-curator local launch \
  --image-name cosmos-curator --image-tag 1.0.0 --curator-path . \
  -- pixi run python3 -m cosmos_curator.pipelines.image.run_pipeline annotate --help
```

**Required**

- `--input-image-path`: path (local or `s3://`) to a directory of input images.
- `--output-path`: path (local or `s3://`) for output; image files, embeddings, `metas/`, and `summary.json` are written under this path.

**Pipeline Examples**

```bash
cosmos-curator local launch \
  --image-name cosmos-curator --image-tag 1.0.0 --curator-path . \
  -- pixi run python3 -m cosmos_curator.pipelines.image.run_pipeline annotate \
  --input-image-path /path/to/images \
  --output-path /path/to/output \
  --captioning-algorithm qwen \
  --limit 10
```

Captioning with Gemini:

```bash
cosmos-curator local launch \
  --image-name cosmos-curator --image-tag 1.0.0 --curator-path . \
  -- pixi run python3 -m cosmos_curator.pipelines.image.run_pipeline annotate \
  --input-image-path /path/to/images \
  --output-path /path/to/output \
  --captioning-algorithm gemini
```

Filtering plus embeddings:

```bash
cosmos-curator local launch \
  --image-name cosmos-curator --image-tag 1.0.0 --curator-path . \
  -- pixi run python3 -m cosmos_curator.pipelines.image.run_pipeline annotate \
  --input-image-path /path/to/images \
  --output-path /path/to/output \
  --semantic-filter enable \
  --image-classifier enable \
  --embedding-algorithm cosmos-embed1-336p
```
