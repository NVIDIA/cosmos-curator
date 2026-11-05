# VITRA / EPIC-KITCHENS Sample Dataset

A public sample for running the egocentric pipeline end to end: `robot-action-split` over real human
hand-manipulation video, producing a canonical `clips.lance` table. Nothing is redistributed — the
annotations and the video are both fetched from their publishers.

A full run takes about twenty minutes on one GPU node, most of it downloading.

## What this sample is

[VITRA-1M](https://huggingface.co/datasets/VITRA-VLA/VITRA-1M) publishes 1.2M egocentric episodes:
sub-second hand actions with a natural-language subtask sentence, 3D MANO hand pose, and camera
parameters. It ships annotations only; the video comes from the four datasets it annotates. This sample
uses its **EPIC-KITCHENS** split because **it needs no undistortion.** EPIC-KITCHENS was shot on a GoPro Hero7
Black in Linear mode, which corrects lens distortion in-camera, so the frames are already rectilinear.

One source video: `P30_114`, a 2-minute EPIC-100 extension video of someone washing up — rinsing,
scrubbing, moving bowls in and out of the sink. VITRA annotates 140 episodes in it, which is enough for
a demo at one 371 MB download. [Choosing different videos](#choosing-different-videos) covers swapping
it out or adding more.

## Licences

**VITRA-1M is MIT.** The episode annotations may be redistributed freely.

**EPIC-KITCHENS is CC BY-NC 4.0** — non-commercial use with attribution, and a commercial licence is
available from the authors at `uob-epic-kitchens@bristol.ac.uk`. This guide links to the video rather
than redistributing it, so the licence attaches to the download. Reuse in published material requires
attribution. Each user is responsible for checking the content of datasets and the applicable licenses
and determining if suitable for the intended use.

## 1. Fetch the annotations

The tarball is 20.5 GB. Gzip decompresses sequentially, so a ranged read of the first 239 MB yields
~1,860 episodes across 384 videos. Run all commands from the repository root.

```bash
WORKSPACE=~/cosmos_curator_local_workspace
mkdir -p "$WORKSPACE/vitra"

# Pinned revision, not main: immutable bytes.
REV=91342ea2d17022ab7069d6fccd8f8651a3ac5c5f
URL=https://huggingface.co/datasets/VITRA-VLA/VITRA-1M/resolve/$REV/epic.tar.gz

# tar exits non-zero on the truncated stream; the members it already wrote are intact.
curl -sSL -r 0-250000000 "$URL" | tee >(sha256sum > "$WORKSPACE/vitra/prefix.sha256") \
  | tar -xz -C "$WORKSPACE/vitra" || true

# Transit check: the first 250,000,000 bytes of that revision hash to a known value.
grep -q 05c5e4faff5b1c71caed5ae31d4834de093d6df6895dd816d76d799432039721 \
  "$WORKSPACE/vitra/prefix.sha256" && echo "prefix OK" || echo "PREFIX MISMATCH"

ls "$WORKSPACE/vitra/epic/episodic_annotations" | wc -l   # expect ~1,860
```

The truncation cuts the final member mid-record, so `gzip` and `tar` both report an unexpected EOF
and the pipeline exits 2 after extracting everything that arrived whole. `|| true` stops that from
killing a `set -e` shell; judge the result by the file count, not the exit code.

The prefix holds only a sample of each video's episodes. Fetch the whole tarball for complete
coverage of one video.

## 2. Fetch the video

```bash
BASE=https://data.bris.ac.uk/datasets/2g1n6qdydwa9u22shpxqzp0t8m
mkdir -p "$WORKSPACE/videos/epic_sample" "$WORKSPACE/output"

curl -L -o "$WORKSPACE/videos/epic_sample/P30_114.MP4" "$BASE/P30/videos/P30_114.MP4"
```

371 MB over roughly ten minutes; the host sustains a few hundred KB/s to just under 1 MB/s. No
registration, no credentials.

## 3. Port the annotations to LeRobot

`robot-action-split` reads LeRobot v3.0 datasets. VITRA publishes `.npy` episodes and ships no
converter, so one is included here. It uses LeRobot's own writer, so the on-disk format is produced by
the library rather than by hand:

```bash
pip install 'lerobot[dataset]'       # porting-time only, not a pipeline dependency

python tools/port_vitra_to_lerobot.py \
  --episodes-dir "$WORKSPACE/vitra/epic/episodic_annotations" \
  --videos-dir   "$WORKSPACE/videos/epic_sample" \
  --output-root  "$WORKSPACE/videos/vitra_epic_lerobot" \
  --video-ids    P30_114 \
  --fps 50 \
  --trust-annotations
```

`--trust-annotations` is required. VITRA episodes are pickled `.npy` object arrays, so loading one
executes code from that file, and the dataset publishes no non-pickle form. The flag acknowledges
that; it does not switch to a safe parser.

The prefix hash in step 1 already covers these files: those bytes determine every `.npy` extracted
from them, so a matching prefix proves the episodes are as published. No per-file manifest is needed
for the path above.

`--checksums` exists for episodes obtained another way, or to re-verify files already on disk. It
takes `sha256sum` output, checked **before any file is unpickled**; a missing or mismatched entry
aborts the run and writes nothing.

```bash
sha256sum "$WORKSPACE"/vitra/epic/episodic_annotations/epic_kitchens_P30_114_ep_*.npy > manifest.sha256
python tools/port_vitra_to_lerobot.py ... --checksums manifest.sha256 --trust-annotations
```

### What each check covers

| check | covers | how |
|---|---|---|
| Pinned revision in the URL | source — immutable bytes | `resolve/$REV/` instead of `resolve/main/` |
| Prefix SHA256 | transit of the ranged read, and every file it extracts | the `grep` above |
| Whole-tarball SHA256 | transit of a full download | HF publishes `6d0567f4...706f` via the LFS metadata; `sha256sum epic.tar.gz` |
| `--checksums` manifest | files of other provenance, or already on disk | per-episode, before any `np.load` |

The whole-file hash cannot validate a ranged read, hence the prefix hash.

One source video becomes one shard, and the shard's chunk MP4 is a **relative symlink to the source
video** — nothing is re-encoded and nothing is copied, so the pipeline does the cutting, which is the
point of running it:

```
vitra_epic_lerobot/
  shard_P30_114/
    meta/{info.json, tasks.parquet, stats.json, episodes/chunk-000/file-000.parquet}
    data/chunk-000/file-000.parquet
    videos/observation.images.main/chunk-000/file-000.mp4 -> ../../../../../epic_sample/P30_114.MP4
```

Porting more videos simply adds more `shard_*` directories; discovery picks up every one under the root.

Each VITRA episode becomes one LeRobot episode carrying one task — its subtask sentence. Discovery
bounds spans by runs of constant `task_index` when no `subtask_index` column is present, so one episode
yields one span, which matches VITRA's annotation granularity exactly.

## 4. Run the pipeline

```bash
cosmos-curator local launch --curator-path . -- \
  pixi run --as-is -e default python -m cosmos_curator.pipelines.pipeline_runtime \
    /config/vitra-epic-split.yaml
```

Copy [`examples/vitra-epic-split.yaml`](../../../examples/vitra-epic-split.yaml) into the workspace
first; the workspace is mounted at `/config` inside the container.

The one setting that matters is the duration bound:

```yaml
split:
  min_duration_s: 0.4
  max_duration_s: 3.0
```

VITRA episodes are sub-second single hand-actions — 0.58 to 2.2 s, median 0.86 s. The 4-20 s defaults
inherited from robot datasets would reject every one of them.

## What a run produces

Measured on one RTX PRO 6000 Blackwell:

| | |
|---|---|
| Input | 140 episodes, 1 video, 371 MB |
| Spans discovered | 100, after the duration filter and per-description dedup |
| Run result | **`published 1 item(s) as 100 row(s) with 0 terminal error(s); 0 item(s) pending; 0 clip(s) failed`** |
| Lance table | 100 rows, 26 columns, 100 distinct `clip_id`, none with `error_stage` set |
| Durations | 0.60-1.90 s, median 0.88 s, 93.1 s total |
| Distinct subtask sentences | 83 |
| Output | 33 MB — 100 clip MP4s and 100 action `.bin` files |

Rows carry the canonical egocentric schema: `clip_id`, `span_group_id`, `episode_id`, `subtask_name`,
`frame_start`/`frame_end`, `start_ns`/`end_ns`, `native_fps`, `episode_from_timestamp`, `clip_uri`,
`action_data_uri`, `camera_motion_annotation`, plus the source chunk (`chunk_id`, `chunk_mp4_uri`,
`data_parquet_uri`) and `error_stage`/`error_message`. A clip that could not be cut keeps its row with
null `clip_uri` and a non-null `error_stage`.

A work item is one source chunk — here the shard's one MP4 — so the run publishes one item as 100 rows.

```
Pick up the white mug by its handle.
Scrape food scraps from the bowl into the food waste container.
Scrub the inside of the bowl with a sponge.
Pick up the soapy bowl from the sink.
Place the bowl into the sink.
```

Rerunning against the same `lance_uri` does not duplicate: a chunk already in the table is not
pending, so a second run publishes `0 item(s) as 0 row(s)`. Each run publishes one selection of at most
`execution.max_items` chunks atomically, and an interrupted run resumes from its checkpoints; rerun
while the summary reports items pending. Terminal errors are not retried in the same table.

## Troubleshooting

**`Config must contain a valid 'pipeline' key`** — two CLIs exist and they take different configs. A
`kind:` config is Curator Next and runs through `cosmos_curator.pipelines.pipeline_runtime`. A
`pipeline:` config is the legacy video pipeline and runs through
`cosmos_curator.pipelines.video.run_pipeline`. This sample is the former.

**`No module named cosmos_curator.client.pipeline_cli.pipeline_runtime`** — the image's `run-pipeline`
pixi task points at a path the repository has since moved. `--curator-path` mounts the package but not
`pixi.toml`, so a stale task runs against new code. Invoke the module directly, as above.

**`ffprobe failed for '.../chunk-000/file-000.mp4': No such file or directory`** — the chunk symlink
points outside the mount. The porter writes relative symlinks for this reason; an absolute host path
will not resolve inside the container.

**`No action-binary spec registered for source dataset`** — `source_dataset` in the config must have an
entry in `ACTION_BINARY_SPEC_BY_DATASET`. This sample registers `vitra_epic` against a `vitra_hand`
spec: 21 camera-space keypoints per hand, 16 flattened 3x3 rotations per hand, plus camera pose.

**Every span dropped by the duration filter** — the defaults are 4-20 s and VITRA episodes are
sub-second. See the config above.

**`has a different producer, item ID column, or result contract; use a new output location`** — the
table was written by an earlier version of the recipe or with different `split`, `source_dataset`,
`action_format`, or `video_bitrate` settings. Point `output.media_root` (and `lance_uri`, if set) at a
new location.

**`TypeError: EngineArgs.__init__() got an unexpected keyword argument`** — repository code newer than
the image. Use a matching image or drop `--curator-path`.

**Do not pipe the launcher through `tail`** — a shell pipeline reports the pipe's exit status, so a
failed run looks successful. Redirect to a file, or set `set -o pipefail`.

## Choosing different videos

The sample's video is not special. To pick others, note that EPIC publishes per-segment
narrations, which make selection measurable without downloading any video:

```bash
BASE=https://raw.githubusercontent.com/epic-kitchens/epic-kitchens-100-annotations/master
curl -LO "$BASE/EPIC_100_train.csv"
curl -LO "$BASE/EPIC_100_validation.csv"
curl -LO "$BASE/EPIC_100_video_info.csv"
```

`EPIC_100_video_info.csv` carries duration and fps for all 700 videos; the train and validation CSVs
carry ~77,000 labelled segments with a verb and noun each. Filter to 2-5 minutes, then score candidates
by verbs that change an object (`cut`, `pour`, `stir`, `rinse`) against verbs that only move one
(`pick-up`, `open`, `close`) — footage that looks busy in thumbnails is often pure transport.

Prefer EPIC-100 extension videos, whose IDs are 100 and above: they use a flat
`{participant}/videos/{video_id}.MP4` path, while older EPIC-55 videos need a train/test split in the
URL. Then confirm VITRA covers them — not every EPIC video has episodes.
