#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stage a registry model's weights where the vnext recipes expect them.

Downloads the model from NGC into the persistent cache (``NVCF_MODEL_CACHE_DIR``) with the same
function the helm/NVCF flow uses. That function verifies checksums, moves a finished download
into place only once it is complete, and records the model in ``.downloaded_models.json``, so a
model that is already recorded and on disk is not downloaded again. The result is linked at
``/config/models/<model_id>``, where recipes read weights; they never download them.

Needs ``NGC_NVCF_API_KEY`` and ``NGC_NVCF_ORG`` (see ``setup_ngc_model_download`` in common.sh)
and the ``model-download`` pixi environment.
"""
# ruff: noqa: T201

import argparse
import os
import pathlib
import shutil
import sys

from cosmos_curator.core.cf.nvcf_utils import download_model_weights_from_nvcf_to_workspace
from cosmos_curator.models.all_models import get_all_models

_MODELS_DIR = pathlib.Path("/config/models")


def main() -> int:
    """Download the model if needed and link it into the recipe model directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant", help="Key in cosmos_curator/configs/all_models.json.")
    args = parser.parse_args()

    entry = get_all_models()[args.variant]
    nvcf_model_id, version, model_id = entry["nvcf_model_id"], entry["version"], entry["model_id"]

    download_dir = pathlib.Path(os.environ["NVCF_MODEL_CACHE_DIR"]) / "models"
    # The download lands in <nvcf_model_id>_v<version> and is renamed to <model_id> once verified,
    # so this directory only survives an interrupted download. Clear it before retrying.
    shutil.rmtree(download_dir / f"{nvcf_model_id}_v{version}", ignore_errors=True)
    download_model_weights_from_nvcf_to_workspace([model_id], download_dir=download_dir)

    staged = download_dir / model_id
    if not (staged / "config.json").is_file():
        print(f"NGC download did not produce {staged}/config.json", file=sys.stderr)
        return 1

    link = _MODELS_DIR / model_id
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink():
        link.unlink()
    elif link.exists():
        print(f"Refusing to replace {link}: it is not a symlink", file=sys.stderr)
        return 1
    link.symlink_to(staged, target_is_directory=True)
    print(f"Linked {link} -> {staged}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
