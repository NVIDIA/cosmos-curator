# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU smoke test through actual FFmpeg, clip upload, and atomic Lance enrichment."""

import shutil
import subprocess
from pathlib import Path

import lance
import pyarrow as pa
import pytest

from cosmos_curator.next.enrichment.execution import selection_datasource
from cosmos_curator.next.enrichment.inputs import open_enrichment_input
from cosmos_curator.next.media.ffmpeg import assert_video_encoder_available, probe_video_path
from cosmos_curator.next.recipes.video_transcode import processing
from cosmos_curator.next.recipes.video_transcode.bindings import resolve_bindings
from cosmos_curator.next.recipes.video_transcode.config import ResolvedVideoTranscodeConfig
from cosmos_curator.next.recipes.video_transcode.contracts import READ_COLUMNS
from cosmos_curator.next.recipes.video_transcode.pipeline import publication_properties
from tests.cosmos_curator.next.enrichment.helpers import attempt_all, publish_attempt


@pytest.mark.integration
def test_real_ffmpeg_clips_publish_and_probe(
    config: ResolvedVideoTranscodeConfig, dataset: lance.LanceDataset, source_path: Path
) -> None:
    """A four-second MP4 becomes four readable clips while span rows stay unchanged."""
    ffmpeg = shutil.which("ffmpeg")
    assert ffmpeg is not None, "FFmpeg is not installed"
    assert_video_encoder_available("libopenh264")
    subprocess.run(  # noqa: S603 -- fixed test generator and an explicitly resolved executable
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:size=64x64:rate=10:duration=4",
            "-c:v",
            "libopenh264",
            "-b:v",
            "100K",
            str(source_path),
        ],
        check=True,
        capture_output=True,
        timeout=30,
    )
    spans = dataset.to_table()
    index = spans.schema.get_field_index("source_size_bytes")
    spans = spans.set_column(index, spans.schema.field(index), pa.array([source_path.stat().st_size] * spans.num_rows))
    original = lance.write_dataset(
        spans, dataset.uri, mode="overwrite", data_storage_version="2.2", max_rows_per_file=2
    )
    attempt = attempt_all(open_enrichment_input(resolve_bindings(config)))
    staged_schema = attempt.input.bindings.spec.staged_schema
    datasource = selection_datasource(attempt, columns=READ_COLUMNS)
    inputs = pa.concat_tables([batch for task in datasource.get_read_tasks(2) for batch in task()])
    downloaded = processing.download_source(processing.group_source_spans(inputs).to_pylist()[0], config=config)
    encoded = pa.Table.from_pylist(list(processing.transcode_source(downloaded, config=config)))
    results = processing.upload_clips(encoded, config=config, staged_schema=staged_schema)
    assert results.schema.equals(staged_schema, check_metadata=True)
    completed = publish_attempt(results, attempt, properties=publication_properties).dataset
    assert completed.version == original.version + 1
    assert completed.count_rows() == 4
    for row in completed.to_table(columns=["clip_uri", "clip_size_bytes", "clip_duration_ns"]).to_pylist():
        clip = Path(row["clip_uri"])
        assert clip.stat().st_size == row["clip_size_bytes"]
        metadata = probe_video_path(clip)
        assert metadata.duration_ns == row["clip_duration_ns"] == 1_000_000_000
        assert metadata.video_codec == "h264"
