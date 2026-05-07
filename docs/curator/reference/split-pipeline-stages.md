# Cosmos Curator - Split Pipeline Stage Overview

The pipeline is built from logical feature blocks. Some blocks are a single
`CuratorStage`; others are multiple stages, for example a CPU preparation stage
followed by a GPU model stage and a CPU post-processing stage.

For the complete command-line surface, run:

```bash
cosmos-curator local launch --curator-path . -- \
  pixi run --as-is python -m cosmos_curator.pipelines.video.run_pipeline split --help
```

## Run Order

The blocks below are appended in this order:

| Order | Block | Runs when |
|---|---|---|
| 1 | Ingest | Always |
| 2 | Split | Always, using `--splitting-algorithm` |
| 3 | Transcode | Always |
| 4 | Super-resolution | `--super-resolution` |
| 5 | Motion filter | `--motion-filter enable` or `score-only` |
| 6 | Shared clip frame extraction | Embeddings are enabled, or `--aesthetic-threshold` is set |
| 7 | Aesthetic filter | `--aesthetic-threshold` is set |
| 8 | Artificial text filter | `--artificial-text-filter enable` |
| 9 | VLM semantic filter | `--vlm-filter enable` or `score-only` |
| 10 | Video classifier | `--video-classifier enable` |
| 11 | Embedding | Enabled by default, disabled with `--no-generate-embeddings` |
| 12 | Captioning and previews | Captioning is enabled by default; previews require captioning plus `--generate-previews` |
| 13 | Caption enhancement | Captioning is enabled and `--enhance-captions` |
| 14 | SAM3 tracking | `--enable-sam3` |
| 15 | Per-event captioning | `--enable-event-captioning`; requires `--enable-sam3` |
| 16 | T5 encoding for Cosmos-Predict | `--generate-cosmos-predict-dataset predict2` |
| 17 | Output writer | Always |

## Stage Catalog

### Ingest

| Item | Details |
|---|---|
| Stage | `VideoDownloader` |
| Code | [`read_write/download_stages.py`](../../../cosmos_curator/pipelines/video/read_write/download_stages.py), built by [`read_write_builders.py`](../../../cosmos_curator/pipelines/video/read_write/read_write_builders.py) |
| Main flags | `--input-video-path`, `--input-video-list-json-path`, `--input-presigned-s3-url`, `--num-download-workers-per-node` |
| Purpose | Reads source videos from local storage, S3-compatible storage, or a presigned ZIP URL into `SplitPipeTask` objects. |
| Output | Populates each task with video bytes and metadata. The downloader also performs inline remuxing for formats that need MP4 container normalization. |

### Split

| Item | Details |
|---|---|
| Stages | `VideoFrameExtractionStage` + `TransNetV2ClipExtractionStage`, or `FixedStrideExtractorStage` |
| Code | [`clipping/clipping_builders.py`](../../../cosmos_curator/pipelines/video/clipping/clipping_builders.py) |
| Main flags | `--splitting-algorithm`, `--transnetv2-*`, `--fixed-stride-*`, `--limit-clips` |
| Purpose | Creates clip spans from each input video. The default `transnetv2` path decodes source frames, detects scene boundaries, and applies minimum/maximum clip length constraints. The `fixed-stride` path creates fixed-duration clips without scene detection. |
| Output | Adds `Clip` objects to each `Video`, with time spans but not yet individual MP4 bytes. |

### Transcode

| Item | Details |
|---|---|
| Stage | `ClipTranscodingStage` |
| Code | [`clipping/clip_extraction_stages.py`](../../../cosmos_curator/pipelines/video/clipping/clip_extraction_stages.py) |
| Main flags | `--transcode-encoder`, `--transcode-cpus-per-worker`, `--transcode-ffmpeg-batch-size`, `--transcode-max-output-frames`, `--clip-re-chunk-size` |
| Purpose | Encodes each detected clip into standalone MP4 bytes. This is the common clip payload consumed by later stages and written by `ClipWriterStage`. |
| Output | Populates `clip.encoded_data` and rechunks clips for downstream throughput. |

### Super-Resolution

| Item | Details |
|---|---|
| Stage | `SuperResolutionStage` |
| Code | [`super_resolution/super_resolution_stage.py`](../../../cosmos_curator/pipelines/video/super_resolution/super_resolution_stage.py), built by [`super_resolution_builders.py`](../../../cosmos_curator/pipelines/video/super_resolution/super_resolution_builders.py) |
| Main flags | `--super-resolution`, `--sr-variant`, `--sr-target-height`, `--sr-target-width`, `--sr-window-frames`, `--sr-overlap-frames` |
| Purpose | Runs SeedVR2 super-resolution on transcoded clips. This is useful when users want higher-resolution clip outputs before filtering, captioning, or writing. |
| Output | Replaces or updates clip MP4 bytes with the super-resolved version. |
| Cost notes | GPU-heavy; currently scheduled as one worker per node. |

### Motion Filter

| Item | Details |
|---|---|
| Stages | `MotionVectorDecodeStage` + `MotionFilterStage` |
| Code | [`filtering/motion/motion_filter_stages.py`](../../../cosmos_curator/pipelines/video/filtering/motion/motion_filter_stages.py), built by [`motion_builders.py`](../../../cosmos_curator/pipelines/video/filtering/motion/motion_builders.py) |
| Main flags | `--motion-filter`, `--motion-global-mean-threshold`, `--motion-per-patch-min-256-threshold`, `--motion-decode-target-fps`, `--motion-score-gpus-per-worker` |
| Purpose | Detects clips with too little motion. The decode stage samples motion-vector data; the score stage computes global and per-patch motion metrics and either filters clips or records scores only. |
| Output | Sets `clip.motion_score_global_mean` and `clip.motion_score_per_patch_min_256`; in `enable` mode, low-motion clips move to `filtered_clips`. |

### Shared Clip Frame Extraction

| Item | Details |
|---|---|
| Stage | `ClipFrameExtractionStage` |
| Code | [`clipping/clip_frame_extraction_stages.py`](../../../cosmos_curator/pipelines/video/clipping/clip_frame_extraction_stages.py) |
| Main flags | `--clip-extraction-target-res`, `--clip-extraction-cpus-per-worker` |
| Purpose | Decodes sampled RGB frames from the transcoded clip bytes for downstream stages that can share the same extracted frames. |
| Output | Populates `clip.extracted_frames` with frame arrays keyed by extraction signature. |
| Runs when | At least one downstream consumer needs shared frames. Aesthetic filtering uses 1 FPS. Embedding uses 2 FPS. If both are enabled, both signatures are extracted. |

### Aesthetic Filter

| Item | Details |
|---|---|
| Stage | `AestheticFilterStage` |
| Code | [`filtering/aesthetics/aesthetic_filter_stages.py`](../../../cosmos_curator/pipelines/video/filtering/aesthetics/aesthetic_filter_stages.py) |
| Main flags | `--aesthetic-threshold`, `--aesthetic-reduction`, `--aesthetic-gpus-per-worker` |
| Purpose | Scores visual quality using sampled frames and filters clips below the configured threshold. |
| Output | Sets `clip.aesthetic_score`; filtered clips move to `filtered_clips`. |

### Artificial Text Filter

| Item | Details |
|---|---|
| Stage | `ArtificialTextFilterStage` |
| Code | [`filtering/aesthetics/artificial_text_filter_stage.py`](../../../cosmos_curator/pipelines/video/filtering/aesthetics/artificial_text_filter_stage.py) |
| Main flags | `--artificial-text-filter`, `--artificial-text-frame-interval`, `--artificial-text-detection-use-cpu`, `--no-artificial-text-corner-detection`, `--ignore-artificial-text-corner-region` |
| Purpose | Detects stable overlay or post-production text, such as subtitles, logos, watermarks, and other artificial text that may be undesirable in training clips. |
| Output | Sets `clip.has_artificial_text` and `clip.artificial_text_segments`; matching clips move to `filtered_clips`. |
| Cost notes | Can run on GPU with the Paddle OCR environment, or on CPU with `--artificial-text-detection-use-cpu`. |

### VLM Semantic Filter

| Item | Details |
|---|---|
| Stages | Local backend: `VllmPrepStage` + `VllmCaptionStage` + `VllmFilteringStage`. External backend: `ApiPrepStage` + `OpenAICaptionStage` or `GeminiCaptionStage` + `VllmFilteringStage`. |
| Code | [`filtering/aesthetics/aesthetics_builders.py`](../../../cosmos_curator/pipelines/video/filtering/aesthetics/aesthetics_builders.py), [`semantic_filter_stages.py`](../../../cosmos_curator/pipelines/video/filtering/aesthetics/semantic_filter_stages.py) |
| Main flags | `--vlm-filter`, `--vlm-filter-categories`, `--vlm-filter-endpoint`, `--vlm-filter-model-variant`, `--vlm-filter-rejection-threshold` |
| Purpose | Uses a VLM to reject clips by semantic criteria, for example categories of content the user does not want in a dataset. `score-only` records the model result without filtering. |
| Output | Adds filter windows and model responses to metadata; in filtering mode, rejected clips move to `filtered_clips` with `clip.qwen_rejection_stage = "semantic"`. |
| Backend notes | `--vlm-filter-endpoint local` runs local vLLM. `openai` calls an OpenAI-compatible endpoint configured under `openai.filter`. `gemini` calls Gemini. |

### Video Classifier

| Item | Details |
|---|---|
| Stages | Local backend: `VllmPrepStage` + `VllmCaptionStage` + `VllmVideoClassifierStage`. External backend: `ApiPrepStage` + `OpenAICaptionStage` or `GeminiCaptionStage` + `VllmVideoClassifierStage`. |
| Code | [`filtering/aesthetics/aesthetics_builders.py`](../../../cosmos_curator/pipelines/video/filtering/aesthetics/aesthetics_builders.py), [`semantic_filter_stages.py`](../../../cosmos_curator/pipelines/video/filtering/aesthetics/semantic_filter_stages.py) |
| Main flags | `--video-classifier`, `--video-classifier-allow`, `--video-classifier-block`, `--video-classifier-use-custom-categories`, `--video-classifier-endpoint` |
| Purpose | Classifies clips into video types and applies allow/block logic. This is useful when users want broad media-type filtering without writing a custom semantic prompt. |
| Output | Sets `clip.qwen_type_classification`; rejected clips move to `filtered_clips` with `clip.qwen_rejection_stage = "classifier"`. |
| Category notes | By default it uses the built-in media taxonomy. With `--video-classifier-use-custom-categories`, the allow/block lists define the taxonomy presented to the model. |

### Embedding

| Item | Details |
|---|---|
| Stages | `InternVideo2FrameCreationStage` + `InternVideo2EmbeddingStage`, `CosmosEmbed1FrameCreationStage` + `CosmosEmbed1EmbeddingStage`, or `OpenAIEmbeddingStage` |
| Code | [`embedding/embedding_builders.py`](../../../cosmos_curator/pipelines/video/embedding/embedding_builders.py) |
| Main flags | `--no-generate-embeddings`, `--embedding-algorithm`, `--embedding-gpus-per-worker`, `--embedding-batch-size`, `--openai-embedding-*` |
| Purpose | Produces one vector embedding per clip for search, retrieval, and semantic deduplication. |
| Output | Populates `clip.intern_video_2_embedding`, `clip.cosmos_embed1_embedding`, or `clip.openai_embedding`; the writer can emit per-clip pickles, grouped parquet, and optional Lance output. |
| Backend notes | `internvideo2` is the default. `cosmos-embed1-224p`, `cosmos-embed1-336p`, and `cosmos-embed1-448p` select Cosmos-Embed1 variants. `openai` calls an OpenAI-compatible embedding endpoint. |

### Captioning

| Item | Details |
|---|---|
| Stages | Local vLLM: `VllmPrepStage` + `VllmCaptionStage`. API: `ApiPrepStage` + `GeminiCaptionStage` or `OpenAICaptionStage`. vLLM async: `VllmAsyncPrepStage` + `VllmAsyncCaptionStage`. |
| Code | [`captioning/captioning_builders.py`](../../../cosmos_curator/pipelines/video/captioning/captioning_builders.py) |
| Main flags | `--no-generate-captions`, `--no-caption-quality-flags`, `--captioning-algorithm`, `--captioning-window-size`, `--captioning-sampling-fps`, `--captioning-prompt-variant`, `--captioning-max-output-tokens` |
| Purpose | Creates windowed captions for each clip. Windowing keeps long clips manageable and gives downstream datasets per-window text. |
| Output | Populates `clip.windows[*].caption` and caption status fields; the writer emits captions in per-clip metadata and `all_window_captions.json`. |
| Backend notes | Supported algorithms include `qwen`, Qwen3 variants, `nemotron`, `cosmos_r1`, `cosmos_r2`, `gemini`, `openai`, and `vllm_async`. |

### Preview Generation

| Item | Details |
|---|---|
| Stage | `PreviewStage` |
| Code | [`preview/preview_stages.py`](../../../cosmos_curator/pipelines/video/preview/preview_stages.py) |
| Main flags | `--generate-previews`, `--preview-target-fps`, `--preview-target-height` |
| Purpose | Produces lightweight WebP previews for caption windows. This is helpful for spot-checking captions without opening full MP4 clips. |
| Output | Populates `window.webp_bytes`; `ClipWriterStage` later writes those bytes as `.webp` files under `previews/`. |
| Placement | Runs inside the captioning block, after caption prep and before caption inference. |
| Dependency | Requires captioning to be enabled. |

### Caption Enhancement

| Item | Details |
|---|---|
| Stage | `EnhanceCaptionStage` |
| Code | [`captioning/captioning_stages.py`](../../../cosmos_curator/pipelines/video/captioning/captioning_stages.py) |
| Main flags | `--enhance-captions`, `--enhance-captions-lm-variant`, `--enhance-captions-openai-model`, `--enhance-captions-prompt-variant`, `--enhance-captions-max-output-tokens` |
| Purpose | Uses a language model to rewrite or enrich existing window captions. |
| Output | Populates `window.enhanced_caption`; the writer includes enhanced captions in metadata and aggregated caption JSON. |
| Dependency | Requires captioning to be enabled. |
| Backend notes | `qwen_lm` and `gpt_oss_20b` are local model options. `openai` calls an OpenAI-compatible endpoint. |

### SAM3 Object Tracking

| Item | Details |
|---|---|
| Stage | `SAM3BBoxStage` |
| Code | [`tracking/sam3_bbox_stage.py`](../../../cosmos_curator/pipelines/video/tracking/sam3_bbox_stage.py), built by [`tracking_builders.py`](../../../cosmos_curator/pipelines/video/tracking/tracking_builders.py) |
| Main flags | `--enable-sam3`, `--sam3-prompts`, `--sam3-target-fps`, `--sam3-max-clip-duration-s`, `--sam3-write-annotated-video` |
| Purpose | Tracks prompted objects through each clip with SAM3. Prompts are plain-text object descriptions such as `"a car"` or `"a pedestrian"`. |
| Output | Populates `clip.sam3_instances`, `clip.sam3_objects_by_frame`, and optionally `clip.sam3_annotated_video`; the writer emits `sam3_instances/`, `sam3_objects/`, and optionally `sam3_tracked/`. |
| Cost notes | Runs in the `sam3` Pixi environment and uses a GPU. Event captioning automatically enables annotated video because the VLM needs object ID overlays. |

### Per-Event Captioning

| Item | Details |
|---|---|
| Stage | `PerEventCaptionStage` |
| Code | [`captioning/per_event_caption_stage.py`](../../../cosmos_curator/pipelines/video/captioning/per_event_caption_stage.py) |
| Main flags | `--enable-event-captioning`, `--event-caption-backend`, `--event-caption-prompt-file`, `--event-caption-qwen-*`, `--event-caption-gemini-*` |
| Purpose | Uses SAM3 object tracks plus the annotated video to generate structured event annotations that reference SAM3 object IDs. This is intended for event-level descriptions, not generic clip captions. |
| Output | Populates `clip.sam3_events`; the writer emits `sam3_events/`. |
| Dependency | Requires `--enable-sam3` and at least one `--sam3-prompts` value. |

### T5 Encoding for Cosmos-Predict

| Item | Details |
|---|---|
| Stage | `T5StageForSplit` |
| Code | [`captioning/captioning_stages.py`](../../../cosmos_curator/pipelines/video/captioning/captioning_stages.py) |
| Main flags | `--generate-cosmos-predict-dataset predict2` |
| Purpose | Encodes generated captions with T5-XXL so the writer can emit a Cosmos-Predict2 Video2World post-training dataset. |
| Output | Populates `window.t5_xxl_embedding`; the writer emits per-window videos, captions, and T5 embeddings under `cosmos_predict2_video2world_dataset/`. |
| Dependency | Requires usable captions for the selected caption field. |

### Output Writer

| Item | Details |
|---|---|
| Stage | `ClipWriterStage` |
| Code | [`read_write/metadata_writer_stage.py`](../../../cosmos_curator/pipelines/video/read_write/metadata_writer_stage.py) |
| Main flags | `--output-clip-path`, `--no-upload-clips`, `--upload-clip-info-in-chunks`, `--upload-clip-info-in-lance`, `--upload-cds-parquet`, `--dry-run`, `--num-clip-writer-workers-per-node` |
| Purpose | Writes clips, metadata, embeddings, previews, SAM3 outputs, processed-video records, and summary files to local or cloud storage. |
| Output | Standard output directories include `clips/`, `filtered_clips/`, `metas/v0/`, `metas_jsonl/v0/`, embedding directories such as `iv2_embd/`, `ce1_embd_<variant>/`, `openai_embd/`, `previews/`, `processed_videos/`, `sam3_*` directories, and `summary.json`. |

## Common Combinations

| Goal | Useful flags |
|---|---|
| Fast smoke test with no model stages | `--splitting-algorithm fixed-stride --no-generate-embeddings --no-generate-captions --limit 1 --limit-clips 1` |
| Fast smoke test without embedding/caption models | `--no-generate-embeddings --no-generate-captions --limit 1 --limit-clips 1` |
| Generate clips, embeddings, and captions | Defaults, plus input/output paths |
| Score motion without rejecting clips | `--motion-filter score-only` |
| Reject low-quality clips | `--motion-filter enable --aesthetic-threshold <score>` |
| Reject semantic categories | `--vlm-filter enable --vlm-filter-categories <categories>` |
| Keep or reject broad video types | `--video-classifier enable --video-classifier-allow <type>` or `--video-classifier-block <type>` |
| Track prompted objects | `--enable-sam3 --sam3-prompts "a car" "a pedestrian"` |
| Generate object-grounded event annotations | `--enable-sam3 --sam3-prompts ... --enable-event-captioning` |
| Emit Cosmos-Predict2 training assets | `--generate-cosmos-predict-dataset predict2` |
