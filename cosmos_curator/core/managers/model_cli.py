# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""CLI when you call a model file."""

import argparse
import contextlib
import json
import os
import pathlib
import tempfile
from typing import Any

import requests
from loguru import logger
from ngcbase.errors import ResourceNotFoundException

from cosmos_curator.core.cf.nvcf_model_manager import get_nvcf_download_manager
from cosmos_curator.core.utils.environment import (
    CONTAINER_PATHS_CODE_DIR,
    MODEL_WEIGHTS_PREFIX,
)
from cosmos_curator.core.utils.model.model_checksums import (
    describe_verification_failure,
    resolve_checksums_for,
    verify_local_files,
)
from cosmos_curator.core.utils.model.model_utils import (
    download_model_weights_from_huggingface_to_workspace,
    push_huggingface_model_to_cloud_storage,
)
from cosmos_curator.models.all_models import get_all_models

_ALL_MODELS_JSON = CONTAINER_PATHS_CODE_DIR / "cosmos_curator" / "configs" / "all_models.json"


def _get_default_models() -> list[str]:
    all_models = get_all_models()
    default_models_to_download = []
    for model in all_models:
        if model.startswith("_"):
            # internal temporary entries
            continue
        if model in {
            "cosmos3_nano",
            "cosmos3_super",
            "qwen3_5_27b",
            "qwen3_6_27b",
            "qwen3_6_27b_fp8",
            "qwen3_8_27b",
            "qwen3_8_27b_fp8",
            "qwen3_vl_30b",
            "qwen3_vl_30b_fp8",
            "qwen3_vl_235b",
            "qwen3_vl_235b_fp8",
        }:
            # too large for new users (cosmos3_{nano,super} are 32 GB / 128 GB omnimodels)
            continue
        if model == "gpt_oss_20b":
            # large and only for enhanced caption
            continue
        if model.startswith("seedvr2"):
            # large SR models, opt-in only
            continue
        if model == "sam3":
            # gated by Meta approval, opt-in only
            continue
        default_models_to_download.append(model)
    return default_models_to_download


def _get_verifiable_models() -> list[str]:
    """Return every model that is published to NVCF and has a checksum to verify against.

    A model with no ``nvcf_model_id`` is not in the registry to fetch, and one with no
    ``sha256`` has nothing to compare to, so neither can be verified.
    """
    return [
        name
        for name, details in get_all_models().items()
        if not name.startswith("_") and details.get("nvcf_model_id") and details.get("sha256")
    ]


def setup_parsers() -> argparse.ArgumentParser:
    """Set up the parsers for the model CLI.

    Returns:
        The setup parser.

    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Download and upload model weights",
    )

    subparsers = parser.add_subparsers(dest="command")

    download_parser = subparsers.add_parser(
        "download",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Download model weights from huggingface to local workspace",
        help="Download model weights from huggingface to local workspace",
    )
    download_parser.set_defaults(func=_download)

    upload_parser = subparsers.add_parser(
        "upload",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Download model weights from huggingface and upload to S3 at prefix specified by --model-weights-prefix"
        ),
        help="Download model weights from huggingface and upload to S3 at prefix specified by --model-weights-prefix",
    )
    upload_parser.set_defaults(func=_upload)

    checksums_parser = subparsers.add_parser(
        "checksums",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Fetch authoritative sha256 checksums from Hugging Face for every file the model "
            "downloads, at the revision it is pinned to, and optionally record them in "
            "all_models.json. Only small non-LFS files are transferred."
        ),
        help="Fetch sha256 checksums from Hugging Face and record them in all_models.json",
    )
    checksums_parser.set_defaults(func=_checksums)
    checksums_parser.add_argument(
        "--write",
        action="store_true",
        help="Write results into all_models.json instead of only printing them.",
    )

    verify_parser = subparsers.add_parser(
        "verify-ngc",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Download only the checksummed file(s) for each model from the NGC/NVCF registry and "
            "verify they match the sha256 recorded in all_models.json. Requires NGC_NVCF_API_KEY "
            "and NGC_NVCF_ORG."
        ),
        help="Download the checksummed file(s) from NGC and verify their sha256",
    )
    verify_parser.set_defaults(func=_verify_ngc)
    verify_parser.add_argument(
        "--dest",
        type=str,
        default="",
        help="Directory to download into. Defaults to a temporary directory that is cleaned up.",
    )
    # Deliberately NOT _get_default_models(): that list is tuned for what a new user should
    # download, so it omits the large models. Validating an org means checking everything
    # that is actually published there and has a checksum to check against.
    verify_parser.add_argument(
        "--models",
        type=str,
        default=",".join(_get_verifiable_models()),
        help=(
            "comma-separated list of models to verify. Defaults to every model with both an "
            "nvcf_model_id and a recorded sha256."
        ),
    )

    for x_parser in [download_parser, upload_parser, checksums_parser]:
        x_parser.add_argument(
            "--models",
            type=str,
            default=",".join(_get_default_models()),
            help=f"comma-separated list of models to download. Available models: {','.join(get_all_models().keys())}",
        )

    upload_parser.add_argument(
        "--model-weights-prefix",
        type=str,
        required=True,
        help=f"Cloud storage prefix for model weights, e.g. {MODEL_WEIGHTS_PREFIX}",
    )

    return parser


def main(argv: list[str] | None = None) -> None:
    """Run the model CLI.

    Args:
        argv: Optional list of arguments to parse. If None, uses sys.argv.

    """
    parser = setup_parsers()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return
    args.func(args)


def _unpack_model_info(model: str) -> tuple[str, str | None, list[str] | None]:
    all_models = get_all_models()
    if model not in all_models:
        err_msg = f"Unknown model {model}. Available models: {','.join(all_models.keys())}"
        logger.error(err_msg)
        raise ValueError(err_msg)
    model_id = all_models[model]["model_id"]
    version = all_models[model]["version"]
    filelist = all_models[model]["filelist"]
    assert isinstance(model_id, str)
    assert isinstance(version, str) or version is None
    assert isinstance(filelist, list) or filelist is None
    return model_id, version, filelist


def _download(args: argparse.Namespace) -> None:
    models = [model.strip() for model in args.models.split(",")]
    for model in models:
        model_id, version, filelist = _unpack_model_info(model)
        download_model_weights_from_huggingface_to_workspace(
            model_id,
            version,
            filelist,
        )


def _upload(args: argparse.Namespace) -> None:
    models = [model.strip() for model in args.models.split(",")]
    for model in models:
        model_id, version, filelist = _unpack_model_info(model)
        push_huggingface_model_to_cloud_storage(
            model_id,
            version,
            filelist,
            model_weights_prefix=args.model_weights_prefix,
        )


def _resolve_one_model_checksums(model: str, token: str | None) -> dict[str, str]:
    """Refresh the sha256 values for the files this model already lists, or an empty dict.

    The filenames come from the model's existing ``sha256`` keys in all_models.json, which a
    human sets when adding the model -- picking the artifact the loader actually reads is a
    judgement call, so this never chooses files on its own.
    """
    model_id, version, _ = _unpack_model_info(model)
    if version is None:
        logger.warning(f"{model}: no pinned version, skipping")
        return {}
    wanted = sorted(get_all_models()[model].get("sha256") or {})
    if not wanted:
        logger.warning(
            f"{model}: no sha256 entry in all_models.json. Add the filename(s) to checksum there "
            "first; this command only resolves hashes for files you have named.",
        )
        return {}
    try:
        selected = resolve_checksums_for(model_id, version, wanted, token=token)
    except (requests.RequestException, KeyError, ValueError) as e:
        # Gated repos answer 401/403 without a token; that is expected, not fatal.
        logger.warning(f"{model}: could not read {model_id}@{version[:8]} from Hugging Face: {e}")
        return {}

    if not selected:
        logger.warning(f"{model}: no files resolved, cannot record checksums")
        return {}
    unresolved = sorted(set(wanted) - set(selected))
    if unresolved:
        # Writing a subset back would silently drop these filenames from all_models.json and
        # stop verifying them. Skip the model entirely so the recorded data stays intact.
        logger.warning(
            f"{model}: could not resolve {unresolved}; leaving its checksums unchanged rather "
            "than recording a partial set.",
        )
        return {}
    logger.info(f"{model}: resolved {len(selected)} checksum(s)")
    return selected


def _checksums(args: argparse.Namespace) -> None:
    """Fetch sha256 checksums from Hugging Face for each requested model."""
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    resolved: dict[str, dict[str, str]] = {}
    for model in (m.strip() for m in args.models.split(",")):
        selected = _resolve_one_model_checksums(model, token)
        if selected:
            resolved[model] = selected

    if not args.write:
        logger.info(f"Resolved {len(resolved)} model(s). Re-run with --write to record them.")
        return

    models_json = json.loads(_ALL_MODELS_JSON.read_text("utf-8"))
    for model, selected in resolved.items():
        entry = models_json[model]
        # Rebuild so sha256 lands next to filelist rather than at the end of the entry.
        # Plain dicts preserve insertion order (3.7+), so no OrderedDict is needed.
        rebuilt: dict[str, object] = {}
        for key, value in entry.items():
            if key != "sha256":
                rebuilt[key] = value
            if key == "filelist":
                rebuilt["sha256"] = selected
        rebuilt.setdefault("sha256", selected)
        models_json[model] = rebuilt

    # Write via a temp file and rename: all_models.json is checked-in config that the
    # pipeline reads at startup, so a crash mid-write must not leave it truncated.
    tmp_path = _ALL_MODELS_JSON.with_suffix(".json.tmp")
    try:
        tmp_path.write_text(json.dumps(models_json, indent=4) + "\n", encoding="utf-8")
        tmp_path.replace(_ALL_MODELS_JSON)
    finally:
        tmp_path.unlink(missing_ok=True)
    logger.info(f"Recorded checksums for {len(resolved)} model(s) in {_ALL_MODELS_JSON}")


def _verify_one_model_against_ngc(model: str, entry: dict[str, Any], dest_root: pathlib.Path) -> bool:
    """Download the checksummed file(s) for one model from NGC and verify them.

    Fetches only the checksummed file(s) rather than the whole model: pulling a
    multi-hundred-GB model to check one artifact is not worth the transfer.

    Returns:
        True if the model verified (or was skipped as not-applicable), False on mismatch.

    """
    expected = entry.get("sha256") or {}
    nvcf_model_id, version = entry.get("nvcf_model_id"), entry.get("version")
    if not expected:
        logger.warning(f"{model}: no sha256 recorded, skipping (run `checksums --write` first)")
        return True
    if not nvcf_model_id or not version:
        logger.warning(f"{model}: not published to NVCF (no nvcf_model_id), skipping")
        return True

    manager = get_nvcf_download_manager()
    dest = dest_root / model
    dest.mkdir(parents=True, exist_ok=True)
    logger.info(f"{model}: fetching {sorted(expected)} from {nvcf_model_id}:{version}")
    try:
        manager.download_model(f"{nvcf_model_id}:{version}", str(dest), file_patterns=sorted(expected))
    except ResourceNotFoundException:
        # Not every model listed in all_models.json is published to every org, so an absent
        # one is reported and skipped rather than failing the whole sweep. A model that IS
        # published but fails to download still counts as a failure below.
        logger.warning(f"{model}: {nvcf_model_id}:{version} is not in this org's registry, skipping")
        return True

    # Scope verification to the exact directory the SDK just wrote. `dest` is not cleared
    # between runs when the caller supplies --dest, so an older <name>_v<version> tree could
    # otherwise satisfy the hash via the bare-filename fallback in verify_local_files.
    downloaded = dest / f"{nvcf_model_id}_v{version}"
    mismatches = verify_local_files(downloaded, expected)
    if mismatches:
        logger.error(f"{model}: CHECKSUM MISMATCH -- {describe_verification_failure(downloaded, mismatches)}")
        return False
    logger.info(f"{model}: OK ({len(expected)} file(s) verified)")
    return True


def _verify_ngc(args: argparse.Namespace) -> None:
    """Verify NGC-hosted copies against the checksums recorded in all_models.json."""
    all_models = get_all_models()
    requested = [m.strip() for m in args.models.split(",")]
    for model in requested:
        if model not in all_models:
            error_msg = f"Unknown model {model}. Available models: {','.join(all_models.keys())}"
            raise ValueError(error_msg)

    with contextlib.ExitStack() as stack:
        # Only allocate a temp dir when the caller did not name one; a caller-supplied
        # --dest is theirs to keep, so it is deliberately not cleaned up.
        dest_root = (
            pathlib.Path(args.dest) if args.dest else pathlib.Path(stack.enter_context(tempfile.TemporaryDirectory()))
        )
        failures = [
            model for model in requested if not _verify_one_model_against_ngc(model, dict(all_models[model]), dest_root)
        ]

    if failures:
        error_msg = f"Checksum verification failed for: {', '.join(failures)}"
        raise RuntimeError(error_msg)
    logger.info(f"All {len(requested)} model(s) verified against their recorded checksums")


if __name__ == "__main__":
    main()
