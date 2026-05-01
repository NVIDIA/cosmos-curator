# Cosmos Curator - AV Reference Pipelines

The AV pipelines are similar to the video pipelines, but are designed around dataset curation for autonomous vehicles.

- [Cosmos Curator - AV Reference Pipelines](#cosmos-curator---av-reference-pipelines)
  - [AV Split-Caption Pipeline](#av-split-caption-pipeline)
    - [AV Split-Caption Pipeline Stages](#av-split-caption-pipeline-stages)
    - [AV Split-Caption Terminology](#av-split-caption-terminology)
    - [AV Split-Caption Pipeline Output Format](#av-split-caption-pipeline-output-format)
    - [AV Split-Caption Pipeline Configurable Options](#av-split-caption-pipeline-configurable-options)

## AV Split-Caption Pipeline

### AV Split-Caption Pipeline Stages

The av-split-caption pipeline includes the following logical stages:
- **Video Download**: Downloads the videos from cloud storage or reads them from disk into memory.
- **Decoding and Splitting**: Decodes the video frames from the raw mp4 bytes and runs a fixed stride-based splitting algorithm to split the video into clips.
- **Transcoding**: Encodes each of the clips into individual mp4 files under the same encoding (H264).
- **Video Embedding**: Creates an embedding for each clip, which can be used for constructing visual semantic search and/or performing semantic deduplication across dataset.
- **Captioning**: Generates a text caption of the clip using a vision-language model (VLM).
- **Clip Writer**: Uploads the clips and their metadata back to cloud storage or writes them to local disk.

Note above lists the "logical" stages from a functionality perspective,
"physically" we would break certain logical stage into multiple stages to optimize GPU utilization and system throughput.
For example, VLM captioning typically requires non-trivial preprocessing of the video, which is typically done on CPU and can hurt GPU utilization.
To improve upon that, the preprocessing functionality is separated into a VLM input preparation stage, such that
- the VLM input preparation stage can scale independently and spawn many parallel CPU workers to keep the captioning stage's GPU workers busy;
- the VLM captioning stage is left with mostly GPU work and therefore can achieve high GPU utilization and throughput.

### AV Split-Caption Terminology

**Sessions**

A session refers to a continuous period during which data is collected from ego's sensors. These sessions can be extensive and may cover significant distances and timeframes.

**Processed Sessions**

Processed sessions are sessions that have undergone various stages of data processing, including but not limited to segmentation into chunks, and data curation.

Sessions will be broken into many clips, and the number of clips may be more than any single function call can process. These clips are batched into chunks and processed.

Processed sessions are used at the end of the pipeline to generate a summary.

**Processed Session Chunk**

A session chunk is a smaller segment of a larger  session. These chunks allow for more manageable data processing and analysis. Each chunk contains a subset of the data collected during the full session. A processed session chunk refers to a batch of sessions that has been processed.

Processed session chunks are used when restarting a pipeline so that processed session chunks are not reprocessed.

**T5 Embeddings**

T5 embeddings are generated representations of the session data that are used for downstream machine learning tasks. These embeddings are only generated if the `default` option is selected in `prompt_types`.

**Enhanced Captions**

Enhanced captions provide additional descriptive information about the sessions. They are only generated if one of the following prompt types is selected: `visibility`, `road_conditions`, or `illumination`.

**Front Window and Back Window**

The terms "front window" and "back window" refer to two distinct captioning "windows" used during the data processing of sessions. A window has a start frame and and end frame. The front window refers to a short window of frames at the start of the video clip, and the back window is a longer set of frames, which is also anchored at the start of the video clip.

**Prompt Types**

Prompt types determine the specific aspects of the session that will be focused on during data processing. The available prompt types include:

- `default`: General captions without specific focus.
- `visibility`: Captions focused on visibility conditions during the session.
- `road_conditions`: Captions that describe the state of the road, such as dryness, wetness, or presence of obstacles.
- `illumination`: Captions related to the lighting conditions, such as daylight, dusk, or night.
- `vri`: A comprehensive prompt type that includes visibility, road conditions, and illumination in a single inference call.

### AV Split-Caption Pipeline Output Format

The av-split-caption pipeline produces the following artifacts under the path specified by `--output-clip-path`:

```bash
{output_clip_path}/
├── metas/
│   ├── {clip-uuid}.json                # metadata per clip, includes captions
├── processed_session_chunks/
│   ├── {session_name}_{chunk_id}.json  # metadata for processed session chunks
├── processed_sessions/
│   ├── {session_name}_{chunk_id}.json  # metadata for processed session chunks
├── t5_embeddings/{caption_type}/
│   ├── {clip-uuid}.bin                 # pickle file with T5 embeddings for the caption type
├── raw_clips/
│   ├── {clip-uuid}.mp4                 # transcoded clips
├── summary.json                        # summary of the pipeline results
```

### AV Split-Caption Pipeline Configurable Options

Below is a summary of the important options for the av-split-caption pipeline. There are many more options available and can be seen from the help message:

```bash
cosmos-curator local launch \
    --image-name cosmos-curator --image-tag 1.0.0 --curator-path . \
    -- pixi run --as-is python3 -m cosmos_curator.pipelines.av.run_pipeline split --help
```

**Options for Input/Output**

- `--input-prefix`: path on local disk or `s3://` bucket that contains MP4 videos.
- `--output-prefix`: destination directory (local or `s3://`) for individual clip files and metadata.

With `--input-prefix` above, by default it will find all files under that path.
In case there are too many files under the same path, you can also provide a specific list of sessions in a text file in list format like bellow:

```
session1/video1.mp4
session2/video1.mp4
session3/video1.mp4
...
```

The session name that precedes the video name can have any number of levels, for example:

```
session_date/car_unique_id1/session_timestamp/video1.mp4
session_date/car_unique_id2/session_timestamp/video1.mp4
```

This text file can be passed in with
- `--session-list`: specifies the path to the video files in each session to process. These sessions are relative to the `--input-prefix`, and can be either a path inside the container or on cloud storage.

**Options for Functionality**


- `--continue-captioning`: If specified, then continue captioning after splitting the clips.
- `--dry-run`: If this flag is present, run pipeline without uploading results or updating database.
`default`. Valid values are: `default, vri, visibility, road_conditions, illumination`.
- `--fixed-stride-split-frames`: Duration of clips (in frame count) generated from the fixed stride splitting stage; default is 256.
- `--front-window-size`: Size in frame count for front window in each clip; default is 57
- `--limit`: how many videos to process. Default is 0, which is unlimited.
- `--limit-clips`: Limit the number of clips from each video to generate (for testing). Default is 0, which is unlimited.
- `--target-clip-size`: Size in frame count for each clip; default is 256
- `--prompt-types`: list of pre-defined prompts to use when captioning the videos. Default is - 
- `--output-format`: default, or cosmos_predict2.

**Options for Performance**

- `--caption-chunk-size`: Number of clips to caption in one chunk. Default is 32.
- `--captioning-max-output-tokens`: Only applies when --prompt-type is one of `vri`, `visibility`, `road_conditions`, or `illumination`. Max number of output tokens requested from enhanced captioning model. Default is 512.
- `--enhance-captions-lm-variant`: Choose `qwen_lm`, `gpt_oss_20b`, or `openai` (OpenAI API).
- `--enhance-captions-openai-model`: Model name when using the OpenAI API for caption enhancement (default: `gpt-5.1-20251113`; base URL is read from config `openai.base_url`).
- `--encoder`: Encoder backend to use for encoding the split clips. Valid values are `libopenh264` and `h264_nvenc`, default is `libopenh264`, which uses the cpu for encoding.
- `--encoder-threads`: number of cpu threads to use when encoding the split videos; default is 32. Used when encoding with `--encoder libopenh264`.
- `--encode-batch-size`: number of clips to encode in parallel; default is 16.
- `--encode-streams-per-gpu`: Number of concurrent encoding streams per GPU; default is 1.
- `--openh264-bitrate`: specifies the bitrate to use when encoding the split videos; default is 10 Mbps, used when encoding on the cpu, which is the recommended method.
- `--perf-profile`: If this flag is present, enable performance profiling.
- `--qwen-batch-size`: Batch size for Qwen model; default is 8.
- `--qwen-input-prepare-cpus-per-actor`: Number of CPUs per actor for Qwen input preparation stage; default is 4.
- `--qwen-lm-batch-size`: Only applies when --prompt-type is one of `vri`, `visibility`, `road_conditions`, or `illumination`. Batch size for Qwen-LM enhance captioning stage. Default is 128.
- `--qwen-lm-use-fp8-weights`: Only applies when --prompt-type is one of `vri`, `visibility`, `road_conditions`, or `illumination`. If this flag is present, use fp8 weights for the Qwen-LM model.
- `--verbose`: If this flag is present, print verbose logs.

**Gotchas**

* If the default prompt type is not requested in `--prompt-types`, T5 embeddings will not be generated
* There is no way to skip the embedding stage if `default` is in `--prompt-types`
* Enhanced captioning only runs when prompt_types are: `visibility`, `road_conditions`, or `illumination`
