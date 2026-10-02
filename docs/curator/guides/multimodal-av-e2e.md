# Multimodal AV: End-to-End Guide

Turn recordings into episodes, then add captions to those episodes. Run `multimodal-split` to publish one
`episodes.lance` table, inspect its rows and media, then run `multimodal-caption` on the same table.
Each caption run selects one camera through the required `context.camera_key`; separate runs add their results
to the existing episode rows.

This guide captions three representative selections: a front view with the default prompt, a non-front `video_only`
view, and a rear view with Ego context. It does not caption every camera or perform fused multiview inference.

## Prepare recordings and configs

Start with [the split example](../../../examples/multimodal-multicamera-split.yaml). Its three-camera map is
illustrative: **enumerate every camera in your own rig**, including cameras you will not caption. Make every camera
and the Ego source required so incomplete spans cannot become canonical rows. Adapt filenames and keys to your
source layout, for example:

```text
recordings/
  source-001/
    recorder-01/
      front.mp4
      front.mp4.timestamps.npy
      side.mp4
      side.mp4.timestamps.npy
      rear.mp4
      rear.mp4.timestamps.npy
      egotrajectory.mcap
```

Camera sidecars contain monotonic signed 64-bit nanosecond capture timestamps. Ego captioning requires sensor
timestamps, a `foxglove.PoseInFrame` MCAP topic, and episodes no longer than 10 seconds. Range overlap does not prove
physical camera-to-Ego clock equivalence or pose quality. Verify source clocks and coordinate meaning separately.
The example uses topic `/egotrajectory` and frame label `clip_local_enu`; adapt these to your source. The frame label
does not transform coordinates or verify a vehicle convention.

The example starts at 10-second spans, 30 fps, a 50 ms video tolerance, and 4M bitrate. These are configurable
starting values. `whole_session: false` is required for a multicamera or camera-plus-Ego rig. See the
[episode publication design](../design/multimodal-av-episode-publication.md) for input and completeness rules.

From the checkout, set your image tag and absolute host paths, then copy the examples into the local workspace.
Run the following commands in the same shell:

```bash
CURATOR_IMAGE_TAG=your-image-tag
CURATOR_RECORDINGS=/absolute/path/to/recordings
CURATOR_CURATED=/absolute/path/to/curated
CURATOR_WORKSPACE="${COSMOS_CURATOR_LOCAL_WORKSPACE_PREFIX:-$HOME}/cosmos_curator_local_workspace"
mkdir -p "$CURATOR_WORKSPACE" "$CURATOR_CURATED"
cp examples/multimodal-multicamera-*.yaml "$CURATOR_WORKSPACE/"
```

Edit these workspace copies for your rig. The examples use container paths: recordings at `/data/recordings` and
output under `/data/curated`. In every caption config, set `input.episodes_lance_uri` to
`<split output.path>/episodes.lance`; the examples already use `/data/curated/full-rig/episodes.lance`.

Choose the result keys before launching captions:

- Enable `output.tag_camera_key: true`, as the examples do, to keep selected-camera results separate.
- With tagging off, an omitted result key resolves to `default`. A second camera using that key with the same model
  and context fails closed; assign distinct keys before the first run.
- Tagging requires camera characters in `[a-z0-9_]` and a composed key of at most 48 characters. For uppercase
  letters, dots, dashes, or excessive length, turn tagging off and supply an explicit valid result key.
- Custom prompts need a base key that does not start with `default`, which is reserved for default-prompt results.
- Custom `video_only` templates need zero literal `{ego_trajectory_context}` sentinels; custom Ego templates need
  exactly one.
- Model and context already distinguish results. Retaining another sample with the same model/context needs a fresh
  base `output.result_key`, such as `run2`, on the existing episode rows.

See the [captioning design](../design/multimodal-av-captioning.md#public-configuration) for full configuration rules
and its [result contract](../design/multimodal-av-captioning.md#result-contract) for schema, provenance, collisions,
and recovery.

Validate the edited workspace copies:

```bash
pixi run -e dev cosmos-curator pipeline validate "$CURATOR_WORKSPACE/multimodal-multicamera-split.yaml"
pixi run -e dev cosmos-curator pipeline validate "$CURATOR_WORKSPACE/multimodal-multicamera-front-caption.yaml"
pixi run -e dev cosmos-curator pipeline validate "$CURATOR_WORKSPACE/multimodal-multicamera-side-caption.yaml"
pixi run -e dev cosmos-curator pipeline validate "$CURATOR_WORKSPACE/multimodal-multicamera-rear-ego-caption.yaml"
```

## Prepare the runtime

Use a full (non-slim) image built from this checkout. See the
[local launcher guide](../../client/end-user-guide.md#quick-start-for-local-run) for image and host setup.
For a slim image, build with `--slim --envs ''` to select only `default` and `model-download`, then run
`pixi install --frozen -e default -e model-download` in the checkout. Add `--pixi-path .` after `--curator-path .`
in the function below. See [Useful Options for Local Run](../../client/end-user-guide.md#useful-options-for-local-run).

Define the launcher once. It mounts the workspace at `/config`, recordings read-only at `/data/recordings`, and
writable output at `/data/curated`:

```bash
curator_launch() {
  pixi run -e dev cosmos-curator local launch \
    --image-name cosmos-curator --image-tag "$CURATOR_IMAGE_TAG" --curator-path . \
    --extra-volumes "$CURATOR_RECORDINGS:/data/recordings:ro,$CURATOR_CURATED:/data/curated" \
    "$@"
}
```

Download the selected model's weights; they land under `/config/models`:

```bash
curator_launch -- pixi run --as-is model-download --models qwen3_8_27b_fp8
```

## Publish the complete rig

```bash
curator_launch -- pixi run --as-is -e default python -m cosmos_curator.next.recipes.multimodal_split.pipeline \
  /config/multimodal-multicamera-split.yaml
```

## Inspect the split

The split prints a JSON summary and fails if its episode accounting does not balance. Check `episodes_dropped`,
`failures`, and `failures_by_category`; inspect detailed reports under `<output.path>/errors/`.
Those reports are diagnostic history; interpret them alongside the current accounting and published rows.
Record actual episode counts and source exclusions.

Read the published table through the same runtime. If you changed the split output path, update the table path here
and in the caption inspection below:

```bash
curator_launch -- pixi run --as-is -e default python -c '
import lance

table = lance.dataset("/data/curated/full-rig/episodes.lance")
columns = ["episode_id", "camera_media", "sensor_media", "row_count", "sampling_fps"]
for row in table.to_table(columns=columns).to_pylist():
    print(row["episode_id"],
          [camera["camera_key"] for camera in row["camera_media"]],
          [sensor["sensor_key"] for sensor in row["sensor_media"]],
          row["row_count"], row["sampling_fps"])
    for camera in row["camera_media"]:
        print(camera["media_location"])
'
```

Every published row must contain exactly your configured cameras and Ego source, with recorded hashes and valid
frame counts/rates. Decode representative published MP4s before captioning. Set `CURATOR_CAMERA_CLIP` to one of the
printed media locations, then repeat for other representative views:

```bash
CURATOR_CAMERA_CLIP='/data/curated/full-rig/sessions/.../front_wide.mp4'
curator_launch -- pixi run --as-is -e default ffmpeg -v error -i "$CURATOR_CAMERA_CLIP" -f null -
```

## Caption three selections

All three caption configs use the same episode table:

| Example | Camera | Context | Prompt | Effective result key |
| --- | --- | --- | --- | --- |
| [Front](../../../examples/multimodal-multicamera-front-caption.yaml) | `front_wide` | `video_only` | default | `default_front_wide` |
| [Side](../../../examples/multimodal-multicamera-side-caption.yaml) | `left_fisheye` | `video_only` | default | `default_left_fisheye` |
| [Rear](../../../examples/multimodal-multicamera-rear-ego-caption.yaml) | `rear_fisheye` | `ego_trajectory_summary` | custom | `rear_view_custom_v2_rear_fisheye` |

Each uses `qwen3_8_27b_fp8`, `tag_camera_key: true`, and `execution.inference_batch_size: 4`. High-resolution video
can exhaust host memory at the recipe default batch size of 32; 4 is a conservative starting setting. Tune down if
needed for your memory budget. Execution tuning does not change caption identity.

The rear example's custom prompt identifies objects as behind the vehicle and uses image-relative positions. It asks
for any supported Ego distance or speed in a separate sentence, without inferring forward or reverse travel from the
rear images. Ego describes motion of the vehicle carrying the camera; it supplies neither camera extrinsics nor a
mapping of Ego axes to image directions.

```bash
curator_launch -- pixi run --as-is -e default python -m cosmos_curator.next.recipes.multimodal_caption.pipeline \
  /config/multimodal-multicamera-front-caption.yaml
curator_launch -- pixi run --as-is -e default python -m cosmos_curator.next.recipes.multimodal_caption.pipeline \
  /config/multimodal-multicamera-side-caption.yaml
curator_launch -- pixi run --as-is -e default python -m cosmos_curator.next.recipes.multimodal_caption.pipeline \
  /config/multimodal-multicamera-rear-ego-caption.yaml
```

## Inspect captions and recover

Read caption metadata through the runtime:

```bash
curator_launch -- pixi run --as-is -e default python -c '
import lance

table = lance.dataset("/data/curated/full-rig/episodes.lance")
rows = table.to_table().to_pylist()
for row in rows:
    print(row["episode_id"])
    for name, metadata in row.items():
        if name.startswith("caption_") and name.endswith("_metadata") and metadata is not None:
            print(name, metadata["status"], metadata["camera_key"],
                  metadata["result_key"], metadata["media_sha256"])
'
```

Inspect the three results' statuses and selected-camera media hashes against their episode descriptors. Each result
also binds the episode's materialization digest. Rear metadata records the exact prompt template in `prompt`,
together with `ego_media_sha256`, `decoder_mapping_digest`, and `context_digest`.

Inspect rear captions against sampled rear frames for unsupported ahead/forward claims about objects behind the
vehicle. A custom prompt is guidance, not a guarantee of viewpoint or general caption accuracy.

Rerunning the same config resumes interrupted work and skips valid terminal results, including `success`, `error`,
and `truncated`; it does not retry them. To recompute, choose a fresh base `output.result_key`, such as `run2`.
This creates new caption and metadata columns for all episodes in the table and preserves earlier results.
