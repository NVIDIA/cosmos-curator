#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Port VITRA-1M episode annotations into a LeRobot v3.0 dataset.

VITRA-1M (https://huggingface.co/datasets/VITRA-VLA/VITRA-1M) publishes
per-episode ``.npy`` files, not a LeRobot dataset, and ships no converter. This
tool reads those episodes plus their source videos and writes a LeRobot v3.0
dataset that ``robot-action-split`` can discover.

It follows the pattern of LeRobot's own ``examples/port_datasets/port_droid.py``:
build a feature schema, emit frames, let ``LeRobotDataset`` own the v3 layout
(chunking, ``meta/tasks.parquet``, ``meta/episodes/``, ``stats.json``,
``info.json``). Nothing about the on-disk format is written by hand.

One VITRA episode becomes one LeRobot episode carrying one task — its subtask
sentence. ``robot-action-split`` bounds spans by runs of constant ``task_index``
when no ``subtask_index`` column is present, so one episode yields one span,
which is what VITRA's annotation granularity already describes.

Use the EPIC-KITCHENS subset: it is the VITRA source that needs no undistortion.

Usage (needs the ``lerobot`` package, which is not a runtime dependency):
    python tools/port_vitra_to_lerobot.py \
        --episodes-dir  /data/vitra/epic/episodic_annotations \
        --videos-dir    ~/cosmos_curator_local_workspace/videos/epic_sample \
        --output-root   ~/cosmos_curator_local_workspace/videos/vitra_epic_lerobot \
        --repo-id       local/vitra-epic-sample \
        --fps 50 --trust-annotations

``--trust-annotations`` is required: VITRA episodes are pickled ``.npy`` object
arrays, so loading one executes code from that file. The dataset publishes no
non-pickle form, so the flag acknowledges that rather than selecting a safe parser.
"""

import argparse
import hashlib
import logging
import os
import pathlib
import re
import shutil
import sys
from collections.abc import Iterator
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger("port_vitra")

# VITRA stores 21 hand keypoints in camera space per hand, and MANO rotations as
# one global orientation plus 15 joint rotations, all 3x3.
_NUM_JOINTS = 21
_NUM_ROTS = 16  # 1 global orientation + 15 MANO joint rotations
_HAND_POS_LEN = _NUM_JOINTS * 3  # 63
_HAND_ROT_LEN = _NUM_ROTS * 9  # 144, flattened rotation matrices

# Episode file names look like ``epic_kitchens_P01_108_ep_000023.npy``; the video
# id is the part the source MP4 is named for.
_EPISODE_RE = re.compile(r"^epic_kitchens_(?P<video_id>P\d+_\d+)_ep_(?P<index>\d+)\.npy$")

VITRA_FEATURES: dict[str, dict[str, Any]] = {
    # Hand keypoints in camera space, flattened. These are the fields the
    # action-binary encoder reads to produce per-frame action records.
    "observation.state.hand_left_cam": {"dtype": "float32", "shape": (_HAND_POS_LEN,)},
    "observation.state.hand_right_cam": {"dtype": "float32", "shape": (_HAND_POS_LEN,)},
    "observation.state.hand_left_cam_rotation": {"dtype": "float32", "shape": (_HAND_ROT_LEN,)},
    "observation.state.hand_right_cam_rotation": {"dtype": "float32", "shape": (_HAND_ROT_LEN,)},
    # Camera pose per frame, recovered from the episode's 4x4 extrinsics.
    "observation.state.camera_position": {"dtype": "float32", "shape": (3,)},
    "observation.state.camera_rotation": {"dtype": "float32", "shape": (9,)},
    # Frame index into the ORIGINAL source video. Carried so a consumer can map a
    # ported frame back to the untouched EPIC MP4; LeRobot renumbers its own
    # frame_index from zero per episode.
    "observation.source_frame": {"dtype": "int64", "shape": (1,)},
}


def _rotation_block(hand: dict[str, Any], num_frames: int) -> np.ndarray:
    """Flatten a hand's global orientation and joint rotations per frame.

    VITRA gives ``global_orient_camspace`` as (T, 3, 3) and ``hand_pose`` as
    (T, 15, 3, 3). They are concatenated into one (T, 144) block rather than
    converted to quaternions, so nothing is lost and no packing convention is
    assumed.
    """
    glob = np.asarray(hand["global_orient_camspace"], dtype=np.float32).reshape(num_frames, 1, 3, 3)
    pose = np.asarray(hand["hand_pose"], dtype=np.float32).reshape(num_frames, -1, 3, 3)
    return np.concatenate([glob, pose], axis=1).reshape(num_frames, -1)


def _hand_arrays(episode: dict[str, Any], side: str, num_frames: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (positions, rotations) for one hand, zero-filled when absent.

    An episode is annotated for one hand or both. The unannotated side is
    emitted as zeros so every frame has the same shape, which the action-binary
    layout requires.
    """
    hand = episode.get(side)
    if not isinstance(hand, dict) or "joints_camspace" not in hand:
        return (
            np.zeros((num_frames, _HAND_POS_LEN), dtype=np.float32),
            np.zeros((num_frames, _HAND_ROT_LEN), dtype=np.float32),
        )
    pos = np.asarray(hand["joints_camspace"], dtype=np.float32).reshape(num_frames, -1)
    return pos, _rotation_block(hand, num_frames)


def _task_sentence(episode: dict[str, Any]) -> str | None:
    """Return the episode's subtask sentence, preferring its annotated hand.

    ``text`` maps a side to ``[(sentence, (start, end)), ...]``. Episodes carry
    one sentence in practice; the first is taken and the rest ignored.
    """
    text = episode.get("text") or {}
    sides = [episode.get("anno_type"), "right", "left"]
    for side in sides:
        if not side:
            continue
        entries = text.get(side) or []
        if entries:
            sentence = entries[0][0]
            if isinstance(sentence, str) and sentence.strip():
                return sentence.strip()
    return None


def _camera_pose(episode: dict[str, Any], num_frames: int) -> tuple[np.ndarray, np.ndarray]:
    """Split (T, 4, 4) extrinsics into translation and flattened rotation."""
    ext = np.asarray(episode["extrinsics"], dtype=np.float32).reshape(num_frames, 4, 4)
    return ext[:, :3, 3], ext[:, :3, :3].reshape(num_frames, 9)


def iter_episode_frames(episode: dict[str, Any]) -> Iterator[dict[str, np.ndarray[Any, Any]]]:
    """Yield one LeRobot frame dict per annotated frame of a VITRA episode."""
    frames = np.asarray(episode["video_decode_frame"], dtype=np.int64)
    n = len(frames)
    left_pos, left_rot = _hand_arrays(episode, "left", n)
    right_pos, right_rot = _hand_arrays(episode, "right", n)
    cam_t, cam_r = _camera_pose(episode, n)
    for i in range(n):
        yield {
            "observation.state.hand_left_cam": left_pos[i],
            "observation.state.hand_right_cam": right_pos[i],
            "observation.state.hand_left_cam_rotation": left_rot[i],
            "observation.state.hand_right_cam_rotation": right_rot[i],
            "observation.state.camera_position": cam_t[i],
            "observation.state.camera_rotation": cam_r[i],
            "observation.source_frame": np.asarray([frames[i]], dtype=np.int64),
        }


def _select_episodes(
    episodes_dir: pathlib.Path,
    video_ids: set[str] | None,
) -> list[tuple[str, pathlib.Path]]:
    """Return (video_id, path) for every matching episode, sorted for determinism.

    ``--limit`` is deliberately not applied here: episodes whose source video is
    missing are dropped afterwards, and capping first would let them consume the
    quota and silently yield fewer episodes than asked for.
    """
    selected: list[tuple[str, pathlib.Path]] = []
    for path in sorted(episodes_dir.glob("*.npy")):
        m = _EPISODE_RE.match(path.name)
        if m is None:
            continue
        vid = m.group("video_id")
        if video_ids is not None and vid not in video_ids:
            continue
        selected.append((vid, path))
    return selected


def _verify_checksums(selected: list[tuple[str, pathlib.Path]], manifest: pathlib.Path) -> int | None:
    """Check every selected episode against a sha256 manifest before loading any.

    The manifest is ``sha256sum`` format, matched on basename. Verified up front
    rather than per file: unpickling runs code, so nothing loads until the whole
    set is accounted for.
    """
    try:
        text = manifest.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        # --checksums is a safety control: a mistyped path must fail loudly rather
        # than crash ambiguously.
        logger.error("--checksums could not be read: %s", exc)  # noqa: TRY400 - the message carries the cause
        return 2

    expected: dict[str, str] = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 2:  # noqa: PLR2004 - "<hex> <name>", the sha256sum format
            expected[pathlib.Path(parts[-1]).name] = parts[0].lower()

    unlisted: list[str] = []
    mismatched: list[str] = []
    for _, path in selected:
        want = expected.get(path.name)
        if want is None:
            unlisted.append(path.name)
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != want:
            mismatched.append(path.name)

    if unlisted:
        logger.error("%d episode(s) absent from %s, e.g. %s", len(unlisted), manifest, unlisted[:3])
    if mismatched:
        logger.error("%d episode(s) failed checksum, e.g. %s", len(mismatched), mismatched[:3])
    if unlisted or mismatched:
        return 2
    logger.info("verified %d episode(s) against %s", len(selected), manifest.name)
    return None


def _attach_video(shard_root: pathlib.Path, source_video: pathlib.Path, view: str) -> None:
    """Expose the source video as this shard's single chunk file.

    ``robot-action-split`` addresses a span as a frame offset into a chunk MP4.
    Pointing the chunk at the untouched source video means the pipeline does the
    cutting, which is the point of running it, and the port copies no video.
    """
    chunk_dir = shard_root / "videos" / view / "chunk-000"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    link = chunk_dir / "file-000.mp4"
    if link.is_symlink() or link.exists():
        link.unlink()
    # Relative, not absolute: the dataset is read inside a container where the
    # workspace is mounted under a different prefix, and an absolute host path
    # would not resolve there.
    target = os.path.relpath(source_video.resolve(), chunk_dir.resolve())
    link.symlink_to(target)


def _add_span_columns(
    shard_root: pathlib.Path,
    first_source_frame: dict[int, int],
    fps: int,
    view: str,
) -> None:
    """Add the per-episode columns discovery reads to meta/episodes.

    LeRobot renumbers ``frame_index`` from zero per episode, so an episode's
    position inside the chunk has to be recorded separately. ``from_timestamp``
    is where the episode starts in the source video; discovery turns it back
    into a frame offset as ``round(from_timestamp * fps)``.
    """
    for path in sorted((shard_root / "meta" / "episodes").rglob("*.parquet")):
        table = pq.read_table(path)
        episode_indices = table.column("episode_index").to_pylist()
        starts = [float(first_source_frame.get(int(e), 0)) / fps for e in episode_indices]
        n = len(episode_indices)
        table = table.append_column("from_timestamp", pa.array(starts, type=pa.float64()))
        table = table.append_column(f"videos/{view}/chunk_index", pa.array([0] * n, type=pa.int64()))
        table = table.append_column(f"videos/{view}/file_index", pa.array([0] * n, type=pa.int64()))
        table = table.append_column(f"videos/{view}/from_timestamp", pa.array(starts, type=pa.float64()))
        pq.write_table(table, path)


def _port_shard(
    shard_root: pathlib.Path,
    video_id: str,
    paths: list[pathlib.Path],
    videos_dir: pathlib.Path,
    args: argparse.Namespace,
) -> tuple[int, int]:
    """Write one shard: every episode of one source video, plus its chunk link.

    Returns (episodes_written, episodes_skipped).
    """
    # Deferred: lerobot is a porting-time dependency, absent from the runtime envs.
    from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: PLC0415

    if shard_root.exists():
        shutil.rmtree(shard_root)

    dataset = LeRobotDataset.create(
        repo_id=f"{args.repo_id}-{video_id}",
        fps=args.fps,
        features=VITRA_FEATURES,
        root=str(shard_root),
        robot_type="human_hand",
        use_videos=False,
    )

    written = 0
    skipped = 0
    first_source_frame: dict[int, int] = {}
    for path in paths:
        # Gated by --trust-annotations, checked in main(): unpickling runs code
        # from the file, and VITRA publishes no non-pickle form of these episodes.
        episode = np.load(path, allow_pickle=args.trust_annotations).item()
        sentence = _task_sentence(episode)
        frames = list(iter_episode_frames(episode)) if sentence is not None else []
        if not frames:
            skipped += 1
            continue
        first_source_frame[written] = int(frames[0]["observation.source_frame"][0])
        for frame in frames:
            # LeRobot reads the task from the frame itself and builds
            # meta/tasks.parquet from the distinct values it sees.
            frame["task"] = sentence
            dataset.add_frame(frame)
        dataset.save_episode()
        written += 1

    dataset.finalize()
    _attach_video(shard_root, videos_dir / f"{video_id}.MP4", args.view)
    _add_span_columns(shard_root, first_source_frame, args.fps, args.view)
    return written, skipped


def _preflight(args: argparse.Namespace, episodes_dir: pathlib.Path) -> int | None:
    """Return an exit code when the run cannot proceed, else None."""
    if args.limit < 0:
        # Left unchecked, a negative limit would slice from the end and quietly
        # port a different set than asked for rather than failing.
        logger.error("--limit must be 0 (no limit) or positive, got %d", args.limit)
        return 2
    if not args.trust_annotations:
        logger.error(
            "Refusing to unpickle without --trust-annotations. VITRA episodes are pickled .npy "
            "object arrays, so loading one runs code from that file. There is no non-pickle form "
            "of this dataset; pass --trust-annotations to proceed."
        )
        return 2
    try:
        import lerobot.datasets.lerobot_dataset  # noqa: F401, PLC0415
    except ImportError:
        logger.error(  # noqa: TRY400 - the traceback adds nothing to a missing-package message
            "lerobot is not installed. It is a porting-time dependency only, not a pipeline "
            "runtime dependency: pip install 'lerobot[dataset]'"
        )
        return 2
    if not episodes_dir.is_dir():
        logger.error("--episodes-dir does not exist: %s", episodes_dir)
        return 2
    return None


def _resolve_episodes(
    args: argparse.Namespace,
    episodes_dir: pathlib.Path,
    videos_dir: pathlib.Path,
) -> list[tuple[str, pathlib.Path]] | int:
    """Return the episodes to port, or an exit code when none can be."""
    video_ids = set(args.video_ids) if args.video_ids else None
    selected = _select_episodes(episodes_dir, video_ids)
    if not selected:
        logger.error("No episodes matched under %s", episodes_dir)
        return 1

    missing = sorted({vid for vid, _ in selected if not (videos_dir / f"{vid}.MP4").is_file()})
    if missing:
        logger.warning("No source video for %d video id(s); their episodes are skipped: %s", len(missing), missing[:5])
        selected = [(vid, p) for vid, p in selected if vid not in set(missing)]
    if not selected:
        logger.error("Every matched episode lacks its source video.")
        return 1

    # Capped only now, so episodes dropped for a missing video do not eat the quota.
    # Only a positive limit caps; 0 means no limit, and negatives are rejected in _preflight.
    if args.limit > 0:
        selected = selected[: args.limit]

    if args.checksums:
        failure = _verify_checksums(selected, pathlib.Path(args.checksums).expanduser())
        if failure is not None:
            return failure
    return selected


def main(argv: list[str] | None = None) -> int:
    """Port VITRA episodes into a LeRobot v3.0 dataset."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episodes-dir", required=True, help="VITRA episodic_annotations directory.")
    parser.add_argument("--videos-dir", required=True, help="Directory of source videos, named <video_id>.MP4.")
    parser.add_argument("--output-root", required=True, help="Directory to write the LeRobot dataset into.")
    parser.add_argument("--repo-id", default="local/vitra-epic-sample", help="LeRobot repo id (naming only).")
    parser.add_argument("--fps", type=int, default=50, help="Source frame rate; EPIC-100 extension videos are 50.")
    parser.add_argument(
        "--video-ids",
        nargs="*",
        default=None,
        help="Restrict to these video ids, e.g. P01_108 P30_114. Default: every video found.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Stop after this many episodes. 0 = no limit.")
    parser.add_argument(
        "--view",
        default="observation.images.main",
        help="View name for the videos/ tree. Must match the view the split config selects.",
    )
    parser.add_argument(
        "--checksums",
        default=None,
        help=(
            "Path to a sha256sum-format manifest. Every selected episode is verified against it "
            "before any file is unpickled; a missing or mismatched entry aborts the run."
        ),
    )
    parser.add_argument(
        "--trust-annotations",
        action="store_true",
        help=("Required. VITRA ships episodes as pickled .npy; unpickling executes code from the file."),
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    episodes_dir = pathlib.Path(args.episodes_dir).expanduser()
    videos_dir = pathlib.Path(args.videos_dir).expanduser()
    output_root = pathlib.Path(args.output_root).expanduser()

    failure = _preflight(args, episodes_dir)
    if failure is not None:
        return failure

    resolved = _resolve_episodes(args, episodes_dir, videos_dir)
    if isinstance(resolved, int):
        return resolved
    selected = resolved

    by_video: dict[str, list[pathlib.Path]] = {}
    for vid, path in selected:
        by_video.setdefault(vid, []).append(path)
    logger.info("porting %d episode(s) from %d video(s)", len(selected), len(by_video))

    total_written = 0
    total_skipped = 0
    for vid, paths in sorted(by_video.items()):
        written, skipped = _port_shard(output_root / f"shard_{vid}", vid, paths, videos_dir, args)
        total_written += written
        total_skipped += skipped
        logger.info("  shard_%s: %d episode(s)", vid, written)

    logger.info("wrote %d episode(s) across %d shard(s) to %s", total_written, len(by_video), output_root)
    if total_skipped:
        logger.warning("skipped %d episode(s) carrying no subtask sentence", total_skipped)
    logger.info("Chunk MP4s are symlinks to the source videos; nothing was re-encoded.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
