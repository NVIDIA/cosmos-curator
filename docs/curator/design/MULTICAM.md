# Multi-Camera Design and Plan

## Scope

### In scope

- Extend the **video** splitting pipeline ([cosmos_curator/pipelines/video/splitting_pipeline.py](../../../cosmos_curator/pipelines/video/splitting_pipeline.py)) to support multi-cam sessions
- Split clips will be time aligned
- fixed-stride only for multi-cam
- session-based input (prefix + configurable session-dir pattern, no DB).
- Pipeline will continue to be capable of splitting single-camera video, no functionality change planned.

### Out of scope

- AV pipelines remain untouched (no removal, deprecation, or changes).

---

## Data Model: Multi-Cam in SplitPipeTask and Video

**File**: [cosmos_curator/pipelines/video/utils/data_model.py](../../../cosmos_curator/pipelines/video/utils/data_model.py)

- Replace `video: Video` with `videos: list[Video]` as the canonical field on `SplitPipeTask`.
- **Backward compatibility**: Keep a single-video view for existing code:
  - Add `@property def video(self) -> Video: return self.videos[0]` so all existing `task.video` and `get_video_from_task(task)` usage continues to work. `self.videos[0]` is considered the `primary` video / camera.
- Initializer:  `SplitPipeTask(video)` (single-cam) or `SplitPipeTask(videos=[...])`; when building from session-based extraction with `--multi-cam`. Normalize internally to always store `videos` (e.g. if only `video` is passed, set `videos = [video]`).
- `primary` video / camera. This is used in stages that only operate on one camera, like captioning stages. The `primary` camera will placed into `self.videos[0]`. It left to the pipeline's task building functions to set the primary camera correctly.
- `session_id`. This is a unique identifier for the "session". For single-cam, this is the path to the video. For multi-cam, this is the name of the subdirectory that contains session. This is not the full path to the session, just the name of the subdirectory. This is used in later stages to group the clips by a `session_id`

A `relative_path` attribute will be added to a video. This is used to preserve the directory structure of the clip that is being split in the output clips.

For example, if the input session has the structure:

```text
0c99dbb9-646e-44b8-9583-2448310cd6a6
├── subdir-00/
│   ├── camera_01.mp4
│   └── camera_02.mp4
└── subdir-10/
    ├── camera_03.mp4
    └── camera_04.mp4
```

The clips will retain this structure. Let's say this session is split into three clips:

```text
clips
├── clid-uuid0
│   ├── subdir-00/
│   │   ├── camera_01.mp4
│   │   └── camera_02.mp4
│   └── subdir-10/
│       ├── camera_01.mp4
│       └── camera_02.mp4
├── clid-uuid1
│   ├── subdir-00/
│   │   ├── camera_01.mp4
│   │   └── camera_02.mp4
│   └── subdir-10/
│       ├── camera_01.mp4
│       └── camera_02.mp4
└── clid-uuid2
    ├── subdir-00/
    │   ├── camera_01.mp4
    │   └── camera_02.mp4
    └── subdir-10/
        ├── camera_01.mp4
        └── camera_02.mp4
```

No change to the `Clip` type. For multi-cam, alignment is enforced by creating **one Clip per (video, clip index)** so that all clips in the same column share the same `uuid` and `span`

| video | clip 0 | clip 1 | ... | clip N-1   |
|-------|--------|--------|-----|------------|
| 0     | uuid0  | uuid1  | ... | uuid_{N-1} |
| 1     | uuid0  | uuid1  | ... | uuid_{N-1} |
| ...   | ...    | ...    | ... | ...        |
| M-1   | uuid0  | uuid1  | ... | uuid_{N-1} |

Each cell `(i, j)` is `video[i].clips[j]`: one `Clip` with `uuid=uuid_j`, `span=span_j`, and that camera’s `source_video` and (after transcode) `encoded_data`. Compute N time aligned spans and N UUIDs (e.g. `uuid_j` keyed by span; include session or path for stability). For each video index `i`, set `video[i].clips = [ Clip(uuid=uuid_j, span=span_j, source_video=video[i].input_video) for j in range(N) ]`. So there are **M×N Clip instances** in total.

A clip in this context refers to a group of clips that share time aligned boundaries. By sharing the same uuid, they can be grouped together in the ClipWriterStage.

### Time Synchronization Validation

Multi-camera tasks enforce strict time alignment through integer-based validation:

**What**: All cameras must have:

1. Identical processed clip counts (`len(clips) + len(filtered_clips)`)
2. Identical spans for clips at the same index (both `clips` and `filtered_clips`)

**When**: Multi-camera stages must explicitly call `task.assert_time_alignment()` before returning from `process_data()`.

**How**: The `assert_time_alignment()` method performs four validations:

1. Validates all `len(video[i].clips) + len(video[i].filtered_clips)` are equal
2. Validates `video[i].clips[j].span` are identical across all videos for each clip index `j` (same for filtered clips)

**Error handling**: Raises `ValueError` with specific messages for each validation failure:

- Different total clips: `"Multi-cam videos have different total clip counts: {counts}..."`
- Different processed clips: `"Multi-cam videos have processed different numbers of clips: {counts}..."`
- Exceeds total: `"Video {i} has processed {processed} clips but only has {total} total clips..."`
- Misaligned clip spans: `"Multi-cam clips at index {j} have misaligned spans: {spans}..."`
- Misaligned filtered spans: `"Multi-cam filtered clips at index {j} have misaligned spans: {spans}..."`

**Rationale**: Time-aligned clips require synchronized processing across all cameras. Since all cameras in a multi-cam task process the same time spans together, they must have the same total clips, process the same number of clips at each stage, and the clips at the same index must represent the exact same time span. Integer comparison and span equality checks ensure precise alignment without floating-point tolerance issues.

**Processing fraction**: The `SplitPipeTask.fraction` property sums all processed and filtered clips across all videos, divided by total clips. This provides an overall progress metric but does not validate alignment—stages must call `assert_time_alignment()` explicitly.

## 2. Stages with Multi-Cam Support

Not all stages will receive multi-cam support. Some stages, like VllmCaptionStage. This can be revisited in the future.

**Important**: All multi-cam stages must call `task.assert_time_alignment()` at the end of their `process_data()` method to validate that cameras remain synchronized.

| Stage | Multicam behavior |
| ------- | ------------------- |
| VideoDownloader | iterate over `task.videos`, download each (includes inline remux for mpegts). |
| FixedStrideExtractorStage | aligned spans, per-cell Clip creation (M×N), max(start)/min(end). Multicam only uses this path. |
| ClipFrameExtractionStage | iterate over `task.videos`, process clips for each video. |
| ClipTranscodingStage | transcode each `videos[i]` for each span `j`, set `videos[i].clips[j].encoded_data`. |
| ClipWriterStage | **Updated for multi-cam** — write per clip index `j`: `clips/<uuid_j>/<camera_name_i>.mp4` from `videos[i].clips[j]`; one metadata JSON per logical clip. |

All filtering, captioning, embedding, preview, and T5 stages stay primary-camera-only (use only `task.video` = primary at slot 0).

## 3. Non-stages with Multi-Cam Support

### 1. Task creation

Before the pipeline starts, a list of tasks are created to send to the pipeline. This will need to be updated to handle the session prefix path and extract sessions and related videos into tasks.

### 2. Summary writing

Because a clip now refers to a group of clips, or a multi-cam clip, summary writing will need to be updated to reflect session processed, and video hours based on groups of cameras, and not sum across all sessions and videos.

### 3. Skip processed multi-cam sessions

After full pipeline support for multi-cam has been added, it becomes possible to add support for skipping processed multi-cam sessions.

### 3. Pipeline options

New command line options will need to be added to the pipeline to enable multi-cam processing.

## Implementation Plan

Todo list of merge requests:

✅ Update data model.
✅ Task creation multi-cam support
✅ VideoDownloader multi-cam support
✅ RemuxStage multi-cam support
✅ FixedStrideExtractorStage multi-cam support
✅ ClipFrameExtractionStage multi-cam support
✅ ClipTranscodingStage multi-cam support
✅ ClipWriterStage multi-cam support
✅ Summary writer multi-cam support

Current single-cam splitting pipeline will continue to function as expected during this development cycle.
