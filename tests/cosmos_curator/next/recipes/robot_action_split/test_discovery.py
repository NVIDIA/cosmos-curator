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

"""Label resolution in ``robot-action-split`` span discovery.

The human-readable label in ``meta/tasks.parquet`` and ``meta/subtasks.parquet``
reaches Arrow in three shapes, all of them produced by real exporters::

    _write_arrow_native          ->  ['task_index', 'task']
    _write_pandas_named_index    ->  ['task_index', 'task']               (Mecka)
    _write_pandas_unnamed_index  ->  ['task_index', '__index_level_0__']  (LIBERO)

The two pandas shapes store the label as the DataFrame index, so it is absent
from ``.to_pandas().columns`` while present in the Arrow schema. Only a
pandas-written file reproduces that, which is why these fixtures use pandas
rather than composing the columns directly in Arrow.

``_write_pandas_named_index`` is the default because it is the shape production
uses. Defaulting to ``arrow_native`` would let most of this module pass against
a reader that cannot see a pandas index at all.

The fixtures write no video. Discovery decodes none, so these tests stay free of
the ffmpeg dependency that every test in ``test_pipeline_integration`` skips
without -- a label contract must never be reported green while skipped.
"""

import json
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from loguru import logger

from cosmos_curator.next.recipes.robot_action_split.config import (
    ResolvedRobotActionSplitConfig,
    SpanFilterConfig,
)
from cosmos_curator.next.recipes.robot_action_split.discovery import (
    SpanWorkItem,
    _read_shard_meta,
    discover_spans,
)

FPS = 24
FRAMES_PER_SPAN = 100

# The span geometry above is arbitrary, so these tests own the bounds it has to
# clear. Riding the shipped SpanFilterConfig defaults instead would let a retune
# of min_duration_s red this whole module with a message naming neither. The skip
# lists are deliberately left at their defaults: one test's subject is that they
# fire on real label text.
TEST_DURATION_BOUNDS: Mapping[str, float] = {"min_duration_s": 1.0, "max_duration_s": 60.0}

# The view discovery falls back to when a shard has no videos/ directory.
VIEW = "observation.images.main"

TASK_INDEX = 0
DEFAULT_TASKS: Mapping[int, str] = {TASK_INDEX: "make coffee"}
DEFAULT_SUBTASKS: Mapping[int, str] = {0: "pick up coffee pod", 1: "open machine lid"}
DEFAULT_TASK_DESCRIPTION = "A person makes coffee at a counter."

LabelWriter = Callable[[Path, Mapping[int, str], str], None]


def _label_name(key: str) -> str:
    """Return the label column paired with an index key (``task_index`` -> ``task``)."""
    return key.removesuffix("_index")


def _write_arrow_native(path: Path, labels: Mapping[int, str], key: str) -> None:
    """Write the label as a real Arrow column, carrying no pandas metadata."""
    pq.write_table(
        pa.table(
            {
                key: pa.array(list(labels), type=pa.int64()),
                _label_name(key): pa.array(list(labels.values())),
            }
        ),
        str(path),
    )


def _write_pandas_named_index(path: Path, labels: Mapping[int, str], key: str) -> None:
    """Write the label as a named pandas index, the Mecka export shape."""
    name = _label_name(key)
    pd.DataFrame({key: list(labels), name: list(labels.values())}).set_index(name).to_parquet(path)


def _write_pandas_unnamed_index(path: Path, labels: Mapping[int, str], key: str) -> None:
    """Write the label as an unnamed pandas index, the LIBERO export shape."""
    pd.DataFrame({key: list(labels)}, index=pd.Index(list(labels.values()))).to_parquet(path)


every_writer_shape = pytest.mark.parametrize(
    "write_labels",
    [_write_arrow_native, _write_pandas_named_index, _write_pandas_unnamed_index],
    ids=["arrow_native", "pandas_named_index", "pandas_unnamed_index"],
)


def _write_shard(
    shard_dir: Path,
    *,
    write_labels: LabelWriter = _write_pandas_named_index,
    tasks: Mapping[int, str] = DEFAULT_TASKS,
    subtasks: Mapping[int, str] | None = DEFAULT_SUBTASKS,
    span_subtask_indices: Sequence[int] | None = None,
    task_description: str | None = DEFAULT_TASK_DESCRIPTION,
    include_task_description: bool = True,
) -> Path:
    """Write one video-free LeRobot/Mecka shard and return its directory.

    Args:
        shard_dir: Directory to create; its parent is the dataset root.
        write_labels: Meta-parquet writer, defaulting to the production shape.
        tasks: ``task_index -> label`` for ``meta/tasks.parquet``.
        subtasks: ``subtask_index -> label``; ``None`` omits both
            ``meta/subtasks.parquet`` and the data file's ``subtask_index``
            column, the layout of a dataset with no subtask annotation.
        span_subtask_indices: One contiguous span per entry, in order. Defaults
            to one span per key of *subtasks*. Naming an index absent from
            *subtasks* builds a span no label can resolve.
        task_description: Episode-level description value, including ``None``.
        include_task_description: Whether the episode parquet declares the
            optional ``task_description`` column.

    Returns:
        The shard directory.

    """
    meta = shard_dir / "meta"
    meta.mkdir(parents=True)
    (meta / "info.json").write_text(json.dumps({"fps": FPS}), encoding="utf-8")

    write_labels(meta / "tasks.parquet", tasks, "task_index")
    if subtasks is not None:
        write_labels(meta / "subtasks.parquet", subtasks, "subtask_index")

    episodes = meta / "episodes" / "chunk-000"
    episodes.mkdir(parents=True)
    episode_columns = {
        "episode_index": pa.array([0], type=pa.int64()),
        # Distinct per shard: span_group_id hashes (source_id, episode_id,
        # subtask_index, frame_start) and source_id covers the whole
        # dataset root, so a shared episode_id would collide span_group_id
        # and clip_id between shards.
        "episode_id": pa.array([f"{shard_dir.name}_ep_000"]),
        f"videos/{VIEW}/chunk_index": pa.array([0], type=pa.int64()),
        f"videos/{VIEW}/file_index": pa.array([0], type=pa.int64()),
        f"videos/{VIEW}/from_timestamp": pa.array([0.0]),
    }
    if include_task_description:
        episode_columns["task_description"] = pa.array([task_description], type=pa.large_string())
    pq.write_table(
        pa.table(episode_columns),
        str(episodes / "file-000.parquet"),
    )

    # Without subtask annotation the constant task_index bounds a single span.
    spans = list(span_subtask_indices) if span_subtask_indices is not None else list(subtasks or tasks)
    per_frame = [index for index in spans for _ in range(FRAMES_PER_SPAN)]
    total = len(per_frame)
    columns = {
        "episode_index": pa.array([0] * total, type=pa.int64()),
        "frame_index": pa.array(range(total), type=pa.int64()),
        "task_index": pa.array([TASK_INDEX] * total, type=pa.int64()),
    }
    if subtasks is not None:
        columns["subtask_index"] = pa.array(per_frame, type=pa.int64())

    data = shard_dir / "data" / "chunk-000"
    data.mkdir(parents=True)
    pq.write_table(pa.table(columns), str(data / "file-000.parquet"))
    return shard_dir


def _make_config(dataset_root: Path) -> ResolvedRobotActionSplitConfig:
    """Build a resolved config over *dataset_root* with test-owned duration bounds.

    Output locations are never written by discovery, so they point at an unused
    sibling directory.
    """
    output = dataset_root.parent / "output"
    return ResolvedRobotActionSplitConfig.model_validate(
        {
            "schema_version": 1,
            "kind": "robot-action-split",
            "input": {"uris": [str(dataset_root)], "source_dataset": "test_dataset"},
            "output": {"media_root": str(output), "lance_uri": str(output / "clips.lance")},
            "execution": {"storage_profile": "default", "discovery_workers": 1},
            "split": dict(TEST_DURATION_BOUNDS),
        }
    )


def _discovered(config: ResolvedRobotActionSplitConfig) -> list[SpanWorkItem]:
    """Flatten every SpanWorkItem discovery emits for *config*."""
    return [item for batch in discover_spans(config) for item in batch.items]


@pytest.fixture
def logged_warnings() -> Iterator[list[str]]:
    """Collect the WARNING messages discovery emits during one test.

    The run-level zero-span failure names no cause of its own; it defers to the
    per-file warnings. Asserting on those is the only way to tell which branch
    emptied a run, since every branch returns the same empty list.
    """
    messages: list[str] = []
    sink = logger.add(lambda message: messages.append(message.record["message"]), level="WARNING")
    try:
        yield messages
    finally:
        logger.remove(sink)


def _overwrite_tasks_parquet(shard: Path, columns: Mapping[str, pa.Array]) -> Path:
    """Replace a shard's ``meta/tasks.parquet`` with *columns* and return its path."""
    path = shard / "meta" / "tasks.parquet"
    pq.write_table(pa.table(dict(columns)), str(path))
    return path


def _reorder_data_rows(shard: Path, order: Sequence[int]) -> None:
    """Rewrite a shard's data parquet with its rows permuted into *order*."""
    path = shard / "data" / "chunk-000" / "file-000.parquet"
    # Typed explicitly so an empty *order* stays int64; a bare [] is null-typed,
    # which take() rejects.
    pq.write_table(pq.read_table(path).take(pa.array(order, type=pa.int64())), str(path))


def _renumber_the_only_episode(shard: Path, episode_index: int) -> None:
    """Point a shard's ``meta/episodes/`` at *episode_index*, orphaning its data rows."""
    path = shard / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    table = pq.read_table(path)
    renumbered = table.set_column(
        table.schema.get_field_index("episode_index"),
        "episode_index",
        pa.array([episode_index], type=pa.int64()),
    )
    pq.write_table(renumbered, str(path))


def _replace_episode_task_description(shard: Path, values: pa.Array) -> None:
    """Replace the only episode parquet's ``task_description`` column."""
    path = shard / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    table = pq.read_table(path)
    field_index = table.schema.get_field_index("task_description")
    replaced = table.set_column(field_index, "task_description", values)
    pq.write_table(replaced, str(path))


@every_writer_shape
def test_shard_meta_recovers_labels_from_every_writer_shape(write_labels: LabelWriter, tmp_path: Path) -> None:
    """Both label maps resolve whatever writer produced the meta parquet."""
    shard = _write_shard(tmp_path / "dataset" / "shard_00", write_labels=write_labels)

    subtask_map, task_map, _ = _read_shard_meta(str(shard))

    assert task_map == dict(DEFAULT_TASKS)
    assert subtask_map == dict(DEFAULT_SUBTASKS)


@every_writer_shape
def test_emitted_spans_carry_source_labels_from_every_writer_shape(write_labels: LabelWriter, tmp_path: Path) -> None:
    """Discovered spans carry the source instruction text, not a stringified index."""
    shard = _write_shard(tmp_path / "dataset" / "shard_00", write_labels=write_labels)

    items = _discovered(_make_config(shard.parent))

    assert {item.subtask_name for item in items} == set(DEFAULT_SUBTASKS.values())
    assert {item.task_name for item in items} == set(DEFAULT_TASKS.values())


def test_tasks_parquet_without_a_label_column_raises(tmp_path: Path) -> None:
    """A tasks.parquet holding only ``task_index`` fails loudly instead of fabricating."""
    shard = _write_shard(tmp_path / "dataset" / "shard_00")
    tasks_parquet = _overwrite_tasks_parquet(shard, {"task_index": pa.array([0], type=pa.int64())})

    with pytest.raises(ValueError, match=r"tasks\.parquet") as excinfo:
        _read_shard_meta(str(shard))

    message = str(excinfo.value)
    assert str(tasks_parquet) in message
    assert "task_index" in message
    # The index column is present, so claiming it is missing sends the operator
    # to the wrong half of the contract.
    assert "expected an index column" not in message


def test_tasks_parquet_without_an_index_column_reports_only_the_missing_index(tmp_path: Path) -> None:
    """A resolvable label with no ``task_index`` names the index as the missing half.

    The two halves of the contract are checked separately so the message names
    whichever is absent. Reporting both would send the operator looking for a
    label column that is right there, and without the check at all the lookup
    raises a bare KeyError naming neither the column nor the file.
    """
    shard = _write_shard(tmp_path / "dataset" / "shard_00")
    tasks_parquet = _overwrite_tasks_parquet(shard, {"task": pa.array(["make coffee"], type=pa.string())})

    with pytest.raises(ValueError, match=r"expected an index column named 'task_index'") as excinfo:
        _read_shard_meta(str(shard))

    message = str(excinfo.value)
    assert str(tasks_parquet) in message
    assert "label column" not in message


def test_numeric_preserved_index_is_not_accepted_as_a_label(tmp_path: Path) -> None:
    """``__index_level_0__`` holding row numbers is an identifier, so it is rejected.

    The column name only says pandas preserved the index, not that the index was
    the label. Accepting it on name alone would stringify row numbers into labels,
    which is the outcome this contract exists to prevent.
    """
    shard = _write_shard(tmp_path / "dataset" / "shard_00")
    pd.DataFrame({"task_index": [0]}, index=[10]).to_parquet(shard / "meta" / "tasks.parquet")

    with pytest.raises(ValueError, match=r"tasks\.parquet"):
        _read_shard_meta(str(shard))


@pytest.mark.parametrize(
    "label_type",
    [pa.string(), pa.large_string(), pa.string_view(), pa.dictionary(pa.int32(), pa.string())],
    ids=["string", "large_string", "string_view", "dictionary"],
)
def test_every_arrow_text_encoding_resolves(label_type: pa.DataType, tmp_path: Path) -> None:
    """All of Arrow's string encodings are labels, and each survives parquet as itself.

    The type gate is the only thing standing between a healthy shard and an
    aborted run, so narrowing it to one encoding would fail datasets it should
    read.
    """
    shard = _write_shard(tmp_path / "dataset" / "shard_00")
    _overwrite_tasks_parquet(
        shard,
        {
            "task_index": pa.array([TASK_INDEX], type=pa.int64()),
            "task": pa.array(["make coffee"], type=label_type),
        },
    )

    _, task_map, _ = _read_shard_meta(str(shard))

    assert task_map == dict(DEFAULT_TASKS)


@pytest.mark.parametrize("label_column", ["task", "task_name"])
def test_either_spelling_of_the_task_label_column_resolves(label_column: str, tmp_path: Path) -> None:
    """Both explicit spellings are accepted, as they were before the contract was enumerated."""
    shard = _write_shard(tmp_path / "dataset" / "shard_00")
    _overwrite_tasks_parquet(
        shard,
        {
            "task_index": pa.array([TASK_INDEX], type=pa.int64()),
            label_column: pa.array(["make coffee"]),
        },
    )

    _, task_map, _ = _read_shard_meta(str(shard))

    assert task_map == dict(DEFAULT_TASKS)


def test_a_real_label_column_wins_over_a_preserved_index(tmp_path: Path) -> None:
    """Precedence puts ``__index_level_0__`` last, so a named column is never shadowed."""
    shard = _write_shard(tmp_path / "dataset" / "shard_00")
    pd.DataFrame({"task_index": [TASK_INDEX], "task": ["make coffee"]}, index=["ignore me"]).to_parquet(
        shard / "meta" / "tasks.parquet"
    )

    _, task_map, _ = _read_shard_meta(str(shard))

    assert task_map == dict(DEFAULT_TASKS)


def test_binary_label_column_is_rejected(tmp_path: Path) -> None:
    """Bytes are not text: stringified they would read ``b'make coffee'``."""
    shard = _write_shard(tmp_path / "dataset" / "shard_00")
    _overwrite_tasks_parquet(
        shard,
        {
            "task_index": pa.array([TASK_INDEX], type=pa.int64()),
            "task": pa.array([b"make coffee"], type=pa.binary()),
        },
    )

    with pytest.raises(ValueError, match=r"tasks\.parquet"):
        _read_shard_meta(str(shard))


def test_one_task_index_mapped_to_two_labels_raises(tmp_path: Path) -> None:
    """A lookup table that answers one index two ways is corrupt, not last-wins."""
    shard = _write_shard(tmp_path / "dataset" / "shard_00")
    _overwrite_tasks_parquet(
        shard,
        {
            "task_index": pa.array([0, 0], type=pa.int64()),
            "task": pa.array(["make coffee", "make tea"]),
        },
    )

    with pytest.raises(ValueError, match="maps to both"):
        _read_shard_meta(str(shard))


def test_tasks_parquet_whose_labels_are_all_blank_raises(tmp_path: Path) -> None:
    """A populated file resolving to nothing raises instead of dropping every span."""
    shard = _write_shard(tmp_path / "dataset" / "shard_00")
    _overwrite_tasks_parquet(
        shard,
        {
            "task_index": pa.array([0, 1], type=pa.int64()),
            "task": pa.array(["", "   "]),
        },
    )

    with pytest.raises(ValueError, match="none carry a label"):
        _read_shard_meta(str(shard))


def test_tasks_parquet_with_an_unrecognised_label_column_raises(tmp_path: Path) -> None:
    """Real prose under an unexpected column name is rejected rather than guessed at."""
    shard = _write_shard(tmp_path / "dataset" / "shard_00")
    _overwrite_tasks_parquet(
        shard,
        {
            "task_index": pa.array([0], type=pa.int64()),
            "notes": pa.array(["make coffee"]),
        },
    )

    with pytest.raises(ValueError, match=r"tasks\.parquet"):
        _read_shard_meta(str(shard))


def test_same_task_index_in_two_shards_resolves_against_its_own_shard(tmp_path: Path) -> None:
    """``task_index`` restarts at 0 per shard, so each shard's rows use its own map."""
    root = tmp_path / "dataset"
    _write_shard(root / "shard_00", tasks={0: "applying stickers"}, subtasks={0: "peel the sticker"})
    _write_shard(root / "shard_01", tasks={0: "applying nail polish"}, subtasks={0: "open the bottle"})

    items = _discovered(_make_config(root))

    assert {item.task_name for item in items} == {"applying stickers", "applying nail polish"}


@pytest.mark.parametrize("description", ["", "  Keep source whitespace.\n"])
def test_episode_task_description_preserves_exact_text(description: str, tmp_path: Path) -> None:
    """Verify discovery preserves description text without stripping or substitution."""
    shard = _write_shard(
        tmp_path / "dataset" / "shard_00",
        task_description=description,
    )

    items = _discovered(_make_config(shard.parent))

    assert {item.task_description for item in items} == {description}


def test_episode_without_task_description_column_emits_none(tmp_path: Path) -> None:
    """Verify a missing optional description column emits ``None``."""
    shard = _write_shard(
        tmp_path / "dataset" / "shard_00",
        include_task_description=False,
    )

    items = _discovered(_make_config(shard.parent))

    assert {item.task_description for item in items} == {None}


def test_null_episode_task_description_emits_none(tmp_path: Path) -> None:
    """Verify a source-null episode description remains ``None`` on every span."""
    shard = _write_shard(
        tmp_path / "dataset" / "shard_00",
        task_description=None,
    )

    items = _discovered(_make_config(shard.parent))

    assert {item.task_description for item in items} == {None}


def test_non_text_episode_task_description_raises(tmp_path: Path) -> None:
    """Verify a non-text episode description column fails clearly."""
    shard = _write_shard(tmp_path / "dataset" / "shard_00")
    _replace_episode_task_description(shard, pa.array([7], type=pa.int64()))

    with pytest.raises(ValueError, match=r"task_description.*Arrow text"):
        _discovered(_make_config(shard.parent))


def test_episode_task_descriptions_are_isolated_per_shard(tmp_path: Path) -> None:
    """Verify each shard resolves episode descriptions against its own metadata."""
    root = tmp_path / "dataset"
    _write_shard(root / "shard_00", task_description="First shard description.")
    _write_shard(root / "shard_01", task_description="Second shard description.")

    items = _discovered(_make_config(root))

    assert len(items) == 4
    descriptions_by_row = {
        (item.episode_id, item.subtask_index, item.view_name): item.task_description for item in items
    }
    assert len(descriptions_by_row) == len(items)
    assert descriptions_by_row == {
        ("shard_00_ep_000", 0, VIEW): "First shard description.",
        ("shard_00_ep_000", 1, VIEW): "First shard description.",
        ("shard_01_ep_000", 0, VIEW): "Second shard description.",
        ("shard_01_ep_000", 1, VIEW): "Second shard description.",
    }


def test_changing_only_task_description_leaves_ids_unchanged(tmp_path: Path) -> None:
    """Verify descriptive metadata does not participate in span or clip identity."""
    shard = _write_shard(
        tmp_path / "dataset" / "shard_00",
        task_description="Original description.",
    )
    original = _discovered(_make_config(shard.parent))

    _replace_episode_task_description(shard, pa.array(["Revised description."], type=pa.large_string()))
    revised = _discovered(_make_config(shard.parent))

    assert {item.task_description for item in original} == {"Original description."}
    assert {item.task_description for item in revised} == {"Revised description."}
    assert [(item.span_group_id, item.clip_id) for item in revised] == [
        (item.span_group_id, item.clip_id) for item in original
    ]


def test_default_filters_drop_their_configured_labels(tmp_path: Path) -> None:
    """Real labels reach the span filters, so each default skip rule actually fires.

    The dropped labels are built from the shipped rules rather than written out,
    so retuning a rule cannot leave this test passing against text nothing
    matches. The preconditions fail first, naming the survivor, if a retune ever
    captures it.
    """
    cfg = SpanFilterConfig()
    survivor = "pour water"
    assert survivor not in cfg.skip_labels
    assert not survivor.startswith(cfg.skip_label_prefixes)
    assert not any(substring in survivor for substring in cfg.skip_label_substrings)

    shard = _write_shard(
        tmp_path / "dataset" / "shard_00",
        subtasks={
            0: cfg.skip_labels[0],
            1: f"{cfg.skip_label_prefixes[0]} wrench",
            2: f"machine {cfg.skip_label_substrings[0]}",
            3: survivor,
        },
    )

    items = _discovered(_make_config(shard.parent))

    assert {item.subtask_name for item in items} == {survivor}


def test_filtering_every_span_out_is_not_a_failure(tmp_path: Path) -> None:
    """A run whose spans all resolve and are all filtered is empty, not broken.

    Distinguishes the two ways to reach zero spans: this one is the configured
    rules doing their job, so it must stay outside the zero-span guard below.
    """
    shard = _write_shard(tmp_path / "dataset" / "shard_00", subtasks={0: SpanFilterConfig().skip_labels[0]})

    assert _discovered(_make_config(shard.parent)) == []


def test_task_label_is_used_when_subtasks_parquet_is_absent(tmp_path: Path) -> None:
    """A dataset with no subtask annotation labels its spans from the task map."""
    shard = _write_shard(tmp_path / "dataset" / "shard_00", tasks={0: "close the top drawer"}, subtasks=None)

    items = _discovered(_make_config(shard.parent))

    assert [item.subtask_name for item in items] == ["close the top drawer"]


def test_task_label_is_used_when_subtask_indices_have_no_label_table(tmp_path: Path) -> None:
    """Subtask indices with no table to resolve against fall back, they do not drop.

    A shard whose ``data/`` names subtasks carries strictly more information than
    one that does not, so it must not yield fewer spans. Dropping here would empty
    the run while reporting success.
    """
    shard = _write_shard(tmp_path / "dataset" / "shard_00", subtasks={0: "peel it", 1: "press it"})
    (shard / "meta" / "subtasks.parquet").unlink()

    items = _discovered(_make_config(shard.parent))

    assert [item.subtask_name for item in items] == ["make coffee", "make coffee"]


def test_fallback_labelled_spans_do_not_compete_for_one_dedup_allowance(tmp_path: Path) -> None:
    """Losing the subtask labels must not also lose spans.

    Per-episode dedup keeps at most ``max_keep_per_description`` spans per
    description. Every span in this shard carries the task label as a stand-in, so
    bucketing on the label alone would cap the whole episode at that allowance --
    deleting a label table would silently reduce the span count while the geometry
    it is derived from never changed.
    """
    runs = SpanFilterConfig().max_keep_per_description + 2
    shard = _write_shard(tmp_path / "dataset" / "shard_00", subtasks={i: f"step {i}" for i in range(runs)})
    (shard / "meta" / "subtasks.parquet").unlink()

    items = _discovered(_make_config(shard.parent))

    assert [item.subtask_name for item in items] == ["make coffee"] * runs


def test_one_subtask_label_shared_by_two_indices_still_dedups(tmp_path: Path) -> None:
    """Distinct indices resolving to one label remain one description.

    The carve-out above keys fallback spans by index, which must not leak into the
    resolved path: spans that genuinely describe the same action still compete for
    a single allowance however many indices carry that description.
    """
    allowance = SpanFilterConfig().max_keep_per_description
    runs = allowance + 2
    shard = _write_shard(tmp_path / "dataset" / "shard_00", subtasks=dict.fromkeys(range(runs), "stir it"))

    items = _discovered(_make_config(shard.parent))

    assert [item.subtask_name for item in items] == ["stir it"] * allowance


def test_one_unreadable_shard_fails_the_whole_run(tmp_path: Path) -> None:
    """The label raise crosses the discovery thread pool rather than being swallowed.

    Neighbouring helpers in discovery catch their exceptions and return ``[]``. A
    shard whose labels cannot be resolved must not take that path and quietly
    emit only the healthy shards' spans.
    """
    root = tmp_path / "dataset"
    _write_shard(root / "shard_00")
    unreadable = _write_shard(root / "shard_01")
    _overwrite_tasks_parquet(unreadable, {"task_index": pa.array([0], type=pa.int64())})

    with pytest.raises(ValueError, match="shard_01"):
        _discovered(_make_config(root))


def test_span_whose_index_is_absent_from_meta_is_dropped(tmp_path: Path) -> None:
    """An unresolvable ``data/`` index drops its span while the resolvable one survives."""
    shard = _write_shard(
        tmp_path / "dataset" / "shard_00",
        subtasks={0: "pick up coffee pod"},
        span_subtask_indices=[0, 1],
    )

    items = _discovered(_make_config(shard.parent))

    assert [item.subtask_name for item in items] == ["pick up coffee pod"]


def test_data_files_that_yield_no_span_at_all_fail_the_run(tmp_path: Path, logged_warnings: list[str]) -> None:
    """A subtask map overlapping no data index empties the run, so the run fails.

    A stale ``subtasks.parquet``, or one numbered from 1 against 0-based data,
    drops every span. Returning nothing is indistinguishable from a dataset with
    nothing to do, which the caller reports as a successful empty run -- so total
    data loss would exit 0. The failure defers to the per-file warning for the
    cause, so the unresolvable indices have to appear there.
    """
    shard = _write_shard(
        tmp_path / "dataset" / "shard_00",
        subtasks={5: "peel it", 6: "press it"},
        span_subtask_indices=[0, 1],
    )

    with pytest.raises(ValueError, match="yielded no span"):
        _discovered(_make_config(shard.parent))

    assert [message for message in logged_warnings if "dropped" in message and "subtask_index=[0, 1]" in message]


@pytest.mark.parametrize("padding", ["chunk-0/file-0.parquet", "chunk-0000/file-0000.parquet"])
def test_a_data_file_listed_but_not_readable_fails_the_run(padding: str, tmp_path: Path) -> None:
    """A file that ``data/`` listed and the reader cannot open fails instead of being skipped.

    Skipping drops that file's spans while the run still reports success, so a
    file removed mid-run costs data silently. The listing parses any digit width
    while every path is rebuilt with three, so an unconventionally padded name
    takes this branch for every file in the shard -- deterministically, and with
    nothing else to reveal it.
    """
    shard = _write_shard(tmp_path / "dataset" / "shard_00")
    listed = shard / "data" / "chunk-000" / "file-000.parquet"
    renamed = shard / "data" / padding
    renamed.parent.mkdir(parents=True, exist_ok=True)
    listed.rename(renamed)
    listed.parent.rmdir()

    with pytest.raises(ValueError, match=r"was listed under data/ but cannot be read"):
        _discovered(_make_config(shard.parent))


def test_a_dataset_of_only_empty_data_files_fails_the_run(tmp_path: Path, logged_warnings: list[str]) -> None:
    """A data file with no rows counts as read, so a dataset of them fails.

    Returning early on an empty file without counting it would leave the run
    reporting success on no output. Nothing was resolved or dropped here, so the
    emptiness is the only cause the file can report -- without it the failure
    sends the operator looking for warnings that were never written.
    """
    shard = _write_shard(tmp_path / "dataset" / "shard_00")
    _reorder_data_rows(shard, [])

    with pytest.raises(ValueError, match="yielded no span"):
        _discovered(_make_config(shard.parent))

    assert [message for message in logged_warnings if "no rows" in message]
    assert not [message for message in logged_warnings if "dropped" in message]


def test_row_order_in_the_data_file_does_not_change_the_spans(tmp_path: Path) -> None:
    """Frames arriving out of order inside one episode still cut the same spans.

    Runs are found on adjacent rows and a span's bounds come from its run's first
    and last row, so unordered frames fragment each run and mis-bound whatever
    survives. ``frame_start`` also feeds ``span_group_id``, so the identities move
    with it.
    """
    shard = _write_shard(tmp_path / "dataset" / "shard_00")
    ordered = _discovered(_make_config(shard.parent))
    assert len(ordered) == len(DEFAULT_SUBTASKS), "fixture precondition: one span per subtask"

    # Alternate the two subtask runs row by row: same rows, same episode, order
    # alone changed. Every pair of neighbours now differs in subtask_index.
    halves = zip(range(FRAMES_PER_SPAN), range(FRAMES_PER_SPAN, 2 * FRAMES_PER_SPAN), strict=True)
    _reorder_data_rows(shard, [row for pair in halves for row in pair])

    assert _discovered(_make_config(shard.parent)) == ordered


def test_data_referencing_no_known_episode_fails_the_run(tmp_path: Path) -> None:
    """An episode absent from ``meta/episodes/`` drops its spans, and a run of only those fails.

    The run-level error names this cause alongside unresolvable labels, so it has
    to actually reach it.
    """
    shard = _write_shard(tmp_path / "dataset" / "shard_00")
    _renumber_the_only_episode(shard, 7)

    with pytest.raises(ValueError, match="yielded no span"):
        _discovered(_make_config(shard.parent))


def test_non_integer_index_column_names_the_file_that_holds_it(tmp_path: Path) -> None:
    """A non-integer ``task_index`` raises against the file it came from.

    The bare coercion error names no file, and this raise crosses a thread pool
    covering every shard in the run.
    """
    shard = _write_shard(tmp_path / "dataset" / "shard_00")
    _overwrite_tasks_parquet(
        shard,
        {"task_index": pa.array(["not-a-number"]), "task": pa.array(["make coffee"])},
    )

    with pytest.raises(ValueError, match=r"tasks\.parquet: task_index value"):
        _discovered(_make_config(shard.parent))
