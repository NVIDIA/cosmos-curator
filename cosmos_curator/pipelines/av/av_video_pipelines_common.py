# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
"""Builds the captioning pipeline stages."""

import argparse
import uuid

from cosmos_curator.core.interfaces.stage_interface import CuratorStage, CuratorStageSpec
from cosmos_curator.core.utils.db.database_types import PostgresDB
from cosmos_curator.pipelines.av.captioning.captioning_stages import (
    PROMPT_TYPES_FOR_ENHANCE_CAPTIONS,
    PROMPT_TYPES_FOR_T5_EMBEDDING,
    EnhanceCaptionStage,
    QwenCaptionStage,
    QwenInputPreparationStage,
    T5Stage,
)
from cosmos_curator.pipelines.av.writers.annotation_writer_stage import (
    AnnotationDbWriterStage,
    AnnotationJsonWriterStage,
)
from cosmos_curator.pipelines.av.writers.cosmos_predict2_writer_stage import (
    CosmosPredict2WriterStage,
)
from cosmos_curator.pipelines.av.writers.t5_writer_stage import (
    T5WriterStage,
)


def build_caption_pipeline_stages(
    args: argparse.Namespace, db: PostgresDB | None, run_uuid: uuid.UUID
) -> list[CuratorStage | CuratorStageSpec]:
    """Build the captioning pipeline stages.

    Args:
        args: The arguments for the pipeline.
        db: The database for the pipeline.
        run_uuid: The run UUID for the pipeline.

    Returns:
        The list of stages for the pipeline.

    """
    stages: list[CuratorStage | CuratorStageSpec] = []

    t5_embedding_stage = any(x in PROMPT_TYPES_FOR_T5_EMBEDDING for x in args.prompt_types)
    enhance_caption = any(x in PROMPT_TYPES_FOR_ENHANCE_CAPTIONS for x in args.prompt_types)
    caption_chain_lens = {
        prompt_type: 2 if prompt_type in PROMPT_TYPES_FOR_ENHANCE_CAPTIONS else 1 for prompt_type in args.prompt_types
    }

    stages.append(
        QwenInputPreparationStage(
            camera_format_id=args.camera_format_id,
            model_variant="qwen",
            target_clip_size=args.target_clip_size,
            front_window_size=args.front_window_size,
            prompt_variants=args.prompt_types,
            prompt_text=args.prompt_text,
            sampling_fps=2.0,
            num_cpus_per_actor=args.qwen_input_prepare_cpus_per_actor,
            preprocess_dtype="float16",
            model_does_preprocess=False,
            keep_mp4=(args.output_format == "cosmos_predict2"),
            verbose=args.verbose,
            log_stats=args.perf_profile,
        )
    )

    stages.append(
        QwenCaptionStage(
            model_variant="qwen",
            prompt_variants=args.prompt_types,
            batch_size=args.qwen_batch_size,
            fp8_enable=False,
            max_output_tokens=2048,
            model_does_preprocess=False,
            verbose=args.verbose,
            log_stats=args.perf_profile,
        )
    )

    if t5_embedding_stage:
        stages.append(
            T5Stage(
                prompt_variants=args.prompt_types,
                verbose=args.verbose,
                log_stats=args.perf_profile,
            )
        )

    if enhance_caption:
        stages.append(
            EnhanceCaptionStage(
                model_variant=args.enhance_captions_lm_variant,
                batch_size=args.qwen_lm_batch_size,
                fp8_enable=args.qwen_lm_use_fp8_weights,
                max_output_tokens=args.captioning_max_output_tokens,
                openai_model=args.enhance_captions_openai_model,
                prompt_variants=args.prompt_types,
                prompt_text=None,
                verbose=args.verbose,
                log_stats=args.perf_profile,
            )
        )

    if not args.dry_run:
        if args.output_format == "cosmos_predict2":
            # cosmos_predict2 format only works with single prompt type
            if len(args.prompt_types) != 1:
                error_msg = (
                    f"cosmos_predict2 output format requires exactly one prompt type, "
                    f"got {len(args.prompt_types)}: {args.prompt_types}. "
                    f"Please specify a single prompt type with --prompt-types."
                )
                raise ValueError(error_msg)

            stages.append(
                CuratorStageSpec(
                    CosmosPredict2WriterStage(
                        output_prefix=args.output_prefix,
                        dataset_name=args.dataset_name,
                        camera_format_id=args.camera_format_id,
                        verbose=args.verbose,
                        log_stats=args.perf_profile,
                    ),
                    num_workers_per_node=4,
                )
            )
        else:
            # Use existing writers (current behavior)
            if t5_embedding_stage:
                stages.append(
                    CuratorStageSpec(
                        T5WriterStage(
                            env=args.db_profile,
                            output_prefix=args.output_prefix,
                            run_id=run_uuid,
                            version=args.caption_version,
                            prompt_variants=args.prompt_types,
                            verbose=args.verbose,
                        ),
                        num_workers_per_node=4,
                    )
                )

            if db is None:
                stages.append(
                    CuratorStageSpec(
                        AnnotationJsonWriterStage(
                            output_prefix=args.output_prefix,
                            verbose=args.verbose,
                            log_stats=args.perf_profile,
                        ),
                        num_workers_per_node=4,
                    )
                )
            else:
                stages.append(
                    CuratorStageSpec(
                        AnnotationDbWriterStage(
                            db,
                            output_prefix=args.output_prefix,
                            run_id=run_uuid,
                            version=args.caption_version,
                            caption_prompt_types=args.prompt_types,
                            caption_chain_lens=caption_chain_lens,
                            verbose=args.verbose,
                            log_stats=args.perf_profile,
                        ),
                        num_workers_per_node=4,
                    )
                )

    return stages
