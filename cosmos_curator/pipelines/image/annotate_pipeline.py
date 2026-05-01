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

"""Image annotate pipeline: load images and write to output (load → write)."""

import argparse
import time
from typing import Any

from loguru import logger

from cosmos_curator.core.interfaces.pipeline_interface import run_pipeline
from cosmos_curator.core.interfaces.stage_interface import CuratorStage, CuratorStageSpec
from cosmos_curator.core.utils.infra.performance_utils import dump_and_write_perf_stats
from cosmos_curator.core.utils.misc.retry_utils import do_with_retries
from cosmos_curator.core.utils.storage.storage_utils import (
    create_path,
    get_full_path,
    get_storage_client,
    is_path_nested,
    verify_path,
)
from cosmos_curator.core.utils.storage.writer_utils import write_json
from cosmos_curator.pipelines.image.captioning.captioning_builders import (
    IMAGE_CAPTION_ALGOS,
    ImageCaptioningConfig,
    build_image_captioning_stages,
)
from cosmos_curator.pipelines.image.embedding.embedding_builders import (
    CLIPImageEmbeddingConfig,
    CosmosEmbed1ImageEmbeddingConfig,
    ImageEmbeddingBackendConfig,
    ImageEmbeddingConfig,
    InternVideo2ImageEmbeddingConfig,
    OpenAIImageEmbeddingConfig,
    build_image_embedding_stages,
)
from cosmos_curator.pipelines.image.filtering.filtering_builders import (
    ImageClassifierConfig,
    ImageSemanticFilterConfig,
    build_image_filter_classifier_stages,
)
from cosmos_curator.pipelines.image.read_write.image_writer_stage import get_image_output_id
from cosmos_curator.pipelines.image.read_write.read_write_builders import (
    ImageIngestConfig,
    ImageOutputConfig,
    build_image_ingest_stages,
    build_image_output_stages,
)
from cosmos_curator.pipelines.image.utils.data_model import ImagePipeTask
from cosmos_curator.pipelines.image.utils.image_pipe_input import extract_image_tasks
from cosmos_curator.pipelines.pipeline_args import add_common_args


def write_summary(
    args: argparse.Namespace,
    num_tasks: int,
    output_tasks: list[ImagePipeTask],
    pipeline_run_time_min: float,
) -> None:
    """Write summary.json and optionally per-stage performance stats to terminal."""
    num_with_caption = sum(1 for t in output_tasks if t.image.has_caption())
    num_with_embeddings = sum(1 for t in output_tasks if t.image.embeddings)
    passed_tasks = [t for t in output_tasks if not t.image.is_filtered]
    filtered_tasks = [t for t in output_tasks if t.image.is_filtered]
    # Use prep-stage defaults when not set via CLI so summary reflects actual values used
    _default_min = 128 * 28 * 28
    _default_max = 768 * 28 * 28
    resize_min_pixels = getattr(args, "caption_prep_min_pixels", None)
    resize_max_pixels = getattr(args, "caption_prep_max_pixels", None)
    if resize_min_pixels is None:
        resize_min_pixels = _default_min
    if resize_max_pixels is None:
        resize_max_pixels = _default_max
    captioned_images = [get_image_output_id(t.session_id) for t in output_tasks if t.image.has_caption()]
    passed_images = [get_image_output_id(t.session_id) for t in passed_tasks]
    filtered_images = [get_image_output_id(t.session_id) for t in filtered_tasks]

    summary_data: dict[str, Any] = {
        "num_input_images": num_tasks,
        "num_output_tasks": len(output_tasks),
        "num_images_passed": len(passed_tasks),
        "num_images_filtered": len(filtered_tasks),
        "pipeline_run_time": round(pipeline_run_time_min, 4),
        "num_images_with_caption": num_with_caption,
        "num_images_with_embeddings": num_with_embeddings,
        "embedding_backend": getattr(args, "embedding_algorithm", None)
        if getattr(args, "generate_embeddings", True)
        else None,
        "resize_min_pixels": resize_min_pixels,
        "resize_max_pixels": resize_max_pixels,
        "images": passed_images,
        "filtered_images": filtered_images,
        "captioned_images": captioned_images,
    }

    client_output = get_storage_client(
        args.output_path,
        profile_name=args.output_s3_profile_name,
        can_overwrite=True,
    )

    def func_write_summary() -> None:
        summary_dest = get_full_path(args.output_path, "summary.json")
        write_json(
            summary_data,
            summary_dest,
            "summary",
            "all images",
            verbose=True,
            client=client_output,
            backup_and_overwrite=True,
        )
        logger.info(f"Wrote summary to {summary_dest}")

    do_with_retries(func_write_summary)

    if args.perf_profile and output_tasks:
        dump_and_write_perf_stats(
            [t.stage_perf for t in output_tasks],
            args.output_path,
            args.output_s3_profile_name,
        )


def build_input_data(args: argparse.Namespace) -> tuple[list[ImagePipeTask], int]:
    """Build input tasks for the image pipeline.

    Validates paths, creates output directory, and discovers image files.

    Args:
        args: Parsed CLI namespace (must have input_image_path, output_path, limit, etc.).

    Returns:
        (list of ImagePipeTask, number of tasks).

    """
    verify_path(args.input_image_path)
    verify_path(args.output_path, level=1)
    create_path(args.output_path)
    if is_path_nested(args.input_image_path, args.output_path):
        msg = "Do not make input and output paths nested"
        raise ValueError(msg)

    tasks = extract_image_tasks(
        args.input_image_path,
        args.input_s3_profile_name,
        limit=args.limit,
        output_path_and_profile=(args.output_path, args.output_s3_profile_name),
        verbose=args.verbose,
    )
    n = len(tasks)
    logger.info(f"About to process {n} image(s) ...")
    return tasks, n


def _assemble_stages(args: argparse.Namespace) -> list[CuratorStage | CuratorStageSpec]:
    """Build the image stage list via the current builder architecture."""
    stages: list[CuratorStage | CuratorStageSpec] = []
    stages.extend(
        build_image_ingest_stages(
            ImageIngestConfig(
                input_path=args.input_image_path,
                input_s3_profile_name=args.input_s3_profile_name,
                num_workers_per_node=args.num_ingest_workers_per_node,
                verbose=args.verbose,
                perf_profile=args.perf_profile,
            )
        )
    )
    if args.image_classifier == "enable" or args.semantic_filter == "enable":
        stages.extend(
            build_image_filter_classifier_stages(
                filter_config=(
                    ImageSemanticFilterConfig(
                        enabled=True,
                        score_only=args.semantic_filter_score_only,
                        model_variant=args.semantic_filter_model_variant,
                        filter_categories=args.semantic_filter_categories,
                        prompt_variant=args.semantic_filter_prompt_variant,
                        rejection_threshold=args.semantic_filter_rejection_threshold,
                        batch_size=args.semantic_filter_batch_size,
                        max_output_tokens=args.semantic_filter_max_output_tokens,
                        num_gpus=args.semantic_filter_num_gpus,
                        openai_model_name=args.semantic_filter_openai_model_name,
                        openai_max_caption_retries=args.semantic_filter_openai_retries,
                        openai_retry_delay_seconds=args.semantic_filter_openai_retry_delay_seconds,
                        gemini_model_name=args.semantic_filter_gemini_model_name,
                        gemini_max_caption_retries=args.semantic_filter_gemini_retries,
                        gemini_retry_delay_seconds=args.semantic_filter_gemini_retry_delay_seconds,
                        caption_prep_min_pixels=args.caption_prep_min_pixels,
                        caption_prep_max_pixels=args.caption_prep_max_pixels,
                        num_prep_workers_per_node=args.num_caption_prep_workers_per_node,
                        verbose=args.verbose,
                        perf_profile=args.perf_profile,
                    )
                    if args.semantic_filter == "enable"
                    else None
                ),
                classifier_config=(
                    ImageClassifierConfig(
                        enabled=True,
                        model_variant=args.image_classifier_model_variant,
                        rejection_threshold=args.image_classifier_rejection_threshold,
                        batch_size=args.image_classifier_batch_size,
                        max_output_tokens=args.image_classifier_max_output_tokens,
                        num_gpus=args.image_classifier_num_gpus,
                        openai_model_name=args.image_classifier_openai_model_name,
                        openai_max_caption_retries=args.image_classifier_openai_retries,
                        openai_retry_delay_seconds=args.image_classifier_openai_retry_delay_seconds,
                        gemini_model_name=args.image_classifier_gemini_model_name,
                        gemini_max_caption_retries=args.image_classifier_gemini_retries,
                        gemini_retry_delay_seconds=args.image_classifier_gemini_retry_delay_seconds,
                        caption_prep_min_pixels=args.caption_prep_min_pixels,
                        caption_prep_max_pixels=args.caption_prep_max_pixels,
                        num_prep_workers_per_node=args.num_caption_prep_workers_per_node,
                        verbose=args.verbose,
                        perf_profile=args.perf_profile,
                        type_allow=",".join(args.image_classifier_allow) if args.image_classifier_allow else None,
                        type_block=",".join(args.image_classifier_block) if args.image_classifier_block else None,
                        custom_categories=args.image_classifier_use_custom_categories,
                        type_allow_file=args.image_classifier_allow_file,
                        type_block_file=args.image_classifier_block_file,
                    )
                    if args.image_classifier == "enable"
                    else None
                ),
            )
        )
    embedding_algorithm = getattr(args, "embedding_algorithm", "internvideo2")
    if getattr(args, "generate_embeddings", True):
        emb_backend: ImageEmbeddingBackendConfig
        if embedding_algorithm.startswith("cosmos-embed1-"):
            emb_backend = CosmosEmbed1ImageEmbeddingConfig(variant=embedding_algorithm.removeprefix("cosmos-embed1-"))
        elif embedding_algorithm == "internvideo2":
            emb_backend = InternVideo2ImageEmbeddingConfig()
        elif embedding_algorithm == "clip":
            emb_backend = CLIPImageEmbeddingConfig()
        else:
            emb_backend = OpenAIImageEmbeddingConfig(
                model_name=args.openai_embedding_model_name,
                max_concurrent_requests=args.openai_embedding_max_concurrent_requests,
            )
        stages.extend(
            build_image_embedding_stages(
                ImageEmbeddingConfig(
                    backend=emb_backend,
                    gpus_per_worker=args.embedding_gpus_per_worker,
                    verbose=args.verbose,
                    perf_profile=args.perf_profile,
                )
            )
        )
    if getattr(args, "generate_captions", True):
        stages.extend(
            build_image_captioning_stages(
                ImageCaptioningConfig(
                    caption_algo=args.captioning_algorithm,
                    num_gpus=args.caption_num_gpus,
                    num_prep_workers_per_node=args.num_caption_prep_workers_per_node,
                    batch_size=args.caption_batch_size,
                    max_output_tokens=args.caption_max_output_tokens,
                    prompt_variant=args.caption_prompt_variant,
                    prompt_text=args.caption_prompt_text or None,
                    stage2_caption=False,
                    stage2_prompt_text=None,
                    caption_prep_min_pixels=args.caption_prep_min_pixels,
                    caption_prep_max_pixels=args.caption_prep_max_pixels,
                    openai_raw_image=args.openai_caption_raw_image,
                    openai_model_name=args.openai_caption_model,
                    openai_caption_retries=args.openai_caption_retries,
                    openai_retry_delay_seconds=args.openai_retry_delay_seconds,
                    gemini_model_name=args.gemini_caption_model,
                    gemini_caption_retries=args.gemini_caption_retries,
                    gemini_retry_delay_seconds=args.gemini_retry_delay_seconds,
                    verbose=args.verbose,
                    perf_profile=args.perf_profile,
                )
            )
        )
    stages.extend(
        build_image_output_stages(
            ImageOutputConfig(
                output_path=args.output_path,
                output_s3_profile_name=args.output_s3_profile_name,
                num_workers_per_node=args.num_output_workers_per_node,
                verbose=args.verbose,
                perf_profile=args.perf_profile,
            )
        )
    )
    return stages


def annotate(args: argparse.Namespace) -> None:
    """Run the image annotate pipeline (load → write)."""
    zero_start = time.time()
    input_tasks, num_tasks = build_input_data(args)
    if num_tasks == 0:
        logger.warning("No images to process; exiting.")
        return

    stages = _assemble_stages(args)
    pipeline_start = time.time()
    output_tasks: list[ImagePipeTask] = run_pipeline(
        input_tasks,
        stages,
        args.model_weights_path,
        args=args,
    )
    pipeline_run_time_min = (time.time() - pipeline_start) / 60
    total_elapsed = (time.time() - zero_start) / 60
    write_summary(args, num_tasks, output_tasks, pipeline_run_time_min=pipeline_run_time_min)
    logger.info(
        f"Image annotate pipeline: {pipeline_run_time_min:.2f} min processing, {total_elapsed:.2f} min total "
        f"for {num_tasks} image(s), {len(output_tasks)} task(s) returned."
    )


def _setup_parser(parser: argparse.ArgumentParser) -> None:  # noqa: PLR0915
    """Add image annotate arguments to the parser."""
    parser.add_argument(
        "--input-image-path",
        type=str,
        required=True,
        help="Local or S3 path to a directory of input images.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Local or S3 path for output (images/ and metas/ created under this).",
    )
    parser.add_argument(
        "--num-ingest-workers-per-node",
        type=int,
        default=4,
        help="Number of image load workers per node.",
    )
    parser.add_argument(
        "--num-output-workers-per-node",
        type=int,
        default=8,
        help="Number of image writer workers per node.",
    )
    parser.add_argument(
        "--no-generate-captions",
        dest="generate_captions",
        action="store_false",
        default=True,
        help="Skip captioning (load and write only).",
    )
    parser.add_argument(
        "--captioning-algorithm",
        type=str,
        default="qwen",
        choices=sorted(IMAGE_CAPTION_ALGOS),
        help="Captioning algorithm for images (local vLLM variants, OpenAI-compatible, or Gemini).",
    )
    parser.add_argument(
        "--caption-num-gpus",
        type=int,
        default=1,
        help="GPUs per node for caption stage.",
    )
    parser.add_argument(
        "--num-caption-prep-workers-per-node",
        type=int,
        default=2,
        help="Workers per node for caption prep stage.",
    )
    parser.add_argument(
        "--caption-prep-min-pixels",
        type=int,
        default=None,
        metavar="N",
        help="Min total pixels for prep resize (default: video-style 128*28*28).",
    )
    parser.add_argument(
        "--caption-prep-max-pixels",
        type=int,
        default=None,
        metavar="N",
        help="Max total pixels for prep resize (default: video-style 768*28*28).",
    )
    parser.add_argument(
        "--caption-batch-size",
        type=int,
        default=16,
        help="Batch size for vLLM caption stage.",
    )
    parser.add_argument(
        "--caption-max-output-tokens",
        type=int,
        default=8192,
        help="Max output tokens for caption generation.",
    )
    parser.add_argument(
        "--caption-prompt-variant",
        type=str,
        default="image",
        help="Prompt variant for captioning (e.g. 'image', 'default').",
    )
    parser.add_argument(
        "--caption-prompt-text",
        type=str,
        default=None,
        help="Custom prompt text for captioning (overrides prompt variant).",
    )
    parser.add_argument(
        "--gemini-caption-model",
        type=str,
        default="models/gemini-2.5-pro",
        help="Gemini model name when --captioning-algorithm gemini is selected.",
    )
    parser.add_argument(
        "--gemini-caption-retries",
        type=int,
        default=3,
        help="Retry count for Gemini image captioning requests.",
    )
    parser.add_argument(
        "--gemini-retry-delay-seconds",
        type=float,
        default=1.0,
        help="Delay between Gemini image captioning retries.",
    )
    parser.add_argument(
        "--openai-caption-model",
        type=str,
        default="auto",
        help="OpenAI-compatible model name when --captioning-algorithm openai is selected.",
    )
    parser.add_argument(
        "--openai-caption-retries",
        type=int,
        default=3,
        help="Retry count for OpenAI-compatible image captioning requests.",
    )
    parser.add_argument(
        "--openai-retry-delay-seconds",
        type=float,
        default=1.0,
        help="Delay between OpenAI-compatible image captioning retries.",
    )
    parser.add_argument(
        "--openai-caption-raw-image",
        action="store_true",
        help="Send original image bytes to the OpenAI-compatible endpoint instead of local-preprocessed PNGs.",
    )
    parser.add_argument(
        "--semantic-filter",
        dest="semantic_filter",
        choices=["enable", "disable"],
        default="disable",
        help="Whether to enable local semantic filtering for images.",
    )
    parser.add_argument(
        "--semantic-filter-score-only",
        action="store_true",
        default=False,
        help="Annotate semantic filter outputs but do not reject images.",
    )
    parser.add_argument(
        "--semantic-filter-model-variant",
        type=str,
        default="qwen",
        choices=sorted(IMAGE_CAPTION_ALGOS),
        help="Model/backend variant for image semantic filtering.",
    )
    parser.add_argument(
        "--semantic-filter-prompt-variant",
        type=str,
        default="default",
        help="Prompt variant for image semantic filtering.",
    )
    parser.add_argument(
        "--semantic-filter-categories",
        type=str,
        default=None,
        help="Comma-separated custom semantic filter categories for images.",
    )
    parser.add_argument(
        "--semantic-filter-rejection-threshold",
        type=float,
        default=0.5,
        help="Fraction of matched filter prompts required to reject an image.",
    )
    parser.add_argument(
        "--semantic-filter-batch-size",
        type=int,
        default=16,
        help="Batch size for local image semantic filtering caption generation.",
    )
    parser.add_argument(
        "--semantic-filter-max-output-tokens",
        type=int,
        default=8192,
        help="Max output tokens for local image semantic filtering.",
    )
    parser.add_argument(
        "--semantic-filter-num-gpus",
        type=int,
        default=1,
        help="Number of GPUs per worker for local image semantic filtering.",
    )
    parser.add_argument(
        "--semantic-filter-openai-model-name",
        type=str,
        default="auto",
        help="Model name for the OpenAI-compatible semantic-filter endpoint.",
    )
    parser.add_argument(
        "--semantic-filter-openai-retries",
        type=int,
        default=3,
        help="Max retries per image for the OpenAI-compatible semantic-filter endpoint.",
    )
    parser.add_argument(
        "--semantic-filter-openai-retry-delay-seconds",
        type=float,
        default=1.0,
        help="Delay in seconds between retries for the OpenAI-compatible semantic-filter endpoint.",
    )
    parser.add_argument(
        "--semantic-filter-gemini-model-name",
        type=str,
        default="models/gemini-2.5-pro",
        help="Gemini model name for the semantic-filter endpoint.",
    )
    parser.add_argument(
        "--semantic-filter-gemini-retries",
        type=int,
        default=3,
        help="Max retries per image for the Gemini semantic-filter endpoint.",
    )
    parser.add_argument(
        "--semantic-filter-gemini-retry-delay-seconds",
        type=float,
        default=1.0,
        help="Delay in seconds between retries for the Gemini semantic-filter endpoint.",
    )
    parser.add_argument(
        "--image-classifier",
        dest="image_classifier",
        choices=["enable", "disable"],
        default="disable",
        help="Whether to enable local image classifier filtering.",
    )
    parser.add_argument(
        "--image-classifier-model-variant",
        type=str,
        default="qwen",
        choices=sorted(IMAGE_CAPTION_ALGOS),
        help="Model/backend variant for image classifier filtering.",
    )
    parser.add_argument(
        "--image-classifier-rejection-threshold",
        type=float,
        default=0.5,
        help="Fraction of blocked classifier outputs required to reject an image.",
    )
    parser.add_argument(
        "--image-classifier-allow",
        nargs="*",
        default=None,
        help="Image categories to allow.",
    )
    parser.add_argument(
        "--image-classifier-block",
        nargs="*",
        default=None,
        help="Image categories to block.",
    )
    parser.add_argument(
        "--image-classifier-use-custom-categories",
        action="store_true",
        default=False,
        help="Treat allow/block values as the full category set instead of the default taxonomy.",
    )
    parser.add_argument(
        "--image-classifier-allow-file",
        type=str,
        default=None,
        help="File containing newline-separated allow-list categories.",
    )
    parser.add_argument(
        "--image-classifier-block-file",
        type=str,
        default=None,
        help="File containing newline-separated block-list categories.",
    )
    parser.add_argument(
        "--image-classifier-batch-size",
        type=int,
        default=16,
        help="Batch size for local image classifier caption generation.",
    )
    parser.add_argument(
        "--image-classifier-max-output-tokens",
        type=int,
        default=8192,
        help="Max output tokens for local image classifier caption generation.",
    )
    parser.add_argument(
        "--image-classifier-num-gpus",
        type=int,
        default=1,
        help="Number of GPUs per worker for local image classifier caption generation.",
    )
    parser.add_argument(
        "--image-classifier-openai-model-name",
        type=str,
        default="auto",
        help="Model name for the OpenAI-compatible classifier endpoint.",
    )
    parser.add_argument(
        "--image-classifier-openai-retries",
        type=int,
        default=3,
        help="Max retries per image for the OpenAI-compatible classifier endpoint.",
    )
    parser.add_argument(
        "--image-classifier-openai-retry-delay-seconds",
        type=float,
        default=1.0,
        help="Delay in seconds between retries for the OpenAI-compatible classifier endpoint.",
    )
    parser.add_argument(
        "--image-classifier-gemini-model-name",
        type=str,
        default="models/gemini-2.5-pro",
        help="Gemini model name for the classifier endpoint.",
    )
    parser.add_argument(
        "--image-classifier-gemini-retries",
        type=int,
        default=3,
        help="Max retries per image for the Gemini classifier endpoint.",
    )
    parser.add_argument(
        "--image-classifier-gemini-retry-delay-seconds",
        type=float,
        default=1.0,
        help="Delay in seconds between retries for the Gemini classifier endpoint.",
    )
    parser.add_argument(
        "--no-generate-embeddings",
        dest="generate_embeddings",
        action="store_false",
        default=True,
        help="Skip embedding generation even if --embedding-algorithm is set.",
    )
    parser.add_argument(
        "--embedding-algorithm",
        type=str,
        default="internvideo2",
        choices=["clip", "cosmos-embed1-224p", "cosmos-embed1-336p", "cosmos-embed1-448p", "internvideo2", "openai"],
        help=(
            "Embedding algorithm to use. The `cosmos-embed1-*` suffix selects the input resolution "
            "(224p, 336p, or 448p): 224p is the fastest with 256-dim output vectors, while 336p and "
            "448p are slower but score higher on retrieval/classification benchmarks and produce "
            "768-dim vectors."
        ),
    )
    parser.add_argument(
        "--embedding-gpus-per-worker",
        type=float,
        default=1.0,
        help="GPUs per worker for local embedding stages (CLIP, CosmosEmbed1, InternVideo2).",
    )
    parser.add_argument(
        "--openai-embedding-model-name",
        type=str,
        default="auto",
        help="Model name for OpenAI-compatible embedding endpoint (only used when --embedding-algorithm=openai).",
    )
    parser.add_argument(
        "--openai-embedding-max-concurrent-requests",
        type=int,
        default=8,
        help="Max concurrent requests to the OpenAI embedding endpoint.",
    )
    add_common_args(parser)


def add_annotate_command(
    subparsers: argparse._SubParsersAction,  # type: ignore[type-arg]
) -> argparse.ArgumentParser:
    """Register the annotate subcommand on the given subparsers."""
    parser = subparsers.add_parser(
        "annotate",
        help="Load images and write to output (images/ + metas/).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.set_defaults(func=annotate)
    _setup_parser(parser)
    return parser  # type: ignore[no-any-return]
