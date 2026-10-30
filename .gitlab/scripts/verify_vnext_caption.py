#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fail unless ``video-caption`` published real captions into the canonical clip table.

Used by the ``vnext_fixed_stride_caption`` k8s_gpu_extended scenario. A zero exit code from the recipe is
not enough evidence that Ray Data LLM and vLLM ran: this reads the Lance table the recipe
publishes to and requires every real span to have become a clip with a non-empty, non-error caption.
"""
# ruff: noqa: T201

import argparse
import sys

import lance
import pyarrow.compute as pc

from cosmos_curator.core.utils.storage.storage_utils import get_lance_storage_options
from cosmos_curator.next.recipes.video_caption.contracts import resolve_model_spec

# "truncated" is a legitimate terminal outcome of a long generation, not an engine fault.
_ACCEPTED_STATUSES = frozenset({"success", "truncated"})


def main() -> int:
    """Validate that every real span has a clip and a caption."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("clips_lance_uri", help="Canonical clip table written by video-transcode.")
    parser.add_argument("--variant", required=True, help="video-caption model variant that was run.")
    parser.add_argument("--storage-profile", default="default")
    args = parser.parse_args()

    spec = resolve_model_spec(args.variant)
    storage_options = get_lance_storage_options(args.clips_lance_uri, profile_name=args.storage_profile)
    table = lance.dataset(args.clips_lance_uri, storage_options=storage_options).to_table(
        columns=["span_id", "clip_id", spec.caption_field_name, spec.metadata_field_name],
    )
    # Empty/error source outcomes from video-cut have no span and are never captioned. Select on span_id
    # rather than clip_id: a span that never became a clip has a null clip_id and would be filtered out.
    spans = table.filter(pc.is_valid(table["span_id"])).to_pylist()

    failures: list[str] = []
    if not spans:
        failures.append("the table has no spans, so nothing was captioned")
    for span in spans:
        clip_id = span["clip_id"]
        if clip_id is None:
            failures.append(f"span {span['span_id']}: no clip_id, so it was never transcoded or captioned")
            continue
        caption = span[spec.caption_field_name]
        metadata = span[spec.metadata_field_name]
        if metadata is None:
            failures.append(f"{clip_id}: no terminal metadata, the clip was never captioned")
            continue
        if metadata["status"] not in _ACCEPTED_STATUSES:
            failures.append(
                f"{clip_id}: status={metadata['status']!r} "
                f"error_type={metadata['error_type']!r} error_message={metadata['error_message']!r}"
            )
        elif not (caption or "").strip():
            failures.append(f"{clip_id}: status={metadata['status']!r} but the caption is empty")
        if metadata["model_id"] != spec.model_id:
            failures.append(f"{clip_id}: model_id={metadata['model_id']!r}, expected {spec.model_id!r}")

    if failures:
        print(f"FAIL: {len(failures)} problem(s) across {len(spans)} span(s):", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1

    print(f"OK: {len(spans)} span(s) captioned by {spec.model_id}")
    print(f"first caption: {(spans[0][spec.caption_field_name] or '')[:200]!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
