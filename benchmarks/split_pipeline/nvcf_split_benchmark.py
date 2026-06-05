# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
"""Benchmark tests for the split pipeline using NVCF."""

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any, cast

import boto3
import smart_open  # type: ignore[import-untyped]
import tenacity
from loguru import logger
from rich import print_json

from benchmarks.cloudevent import make_cloudevent, push_cloudevent
from benchmarks.secrets import KratosSecrets, NvcfSecrets, S3Secrets
from benchmarks.summary import make_summary_metrics
from cosmos_curator.client.nvcf_cli.ncf.launcher.nvcf_driver import _get_s3_config_str
from cosmos_curator.client.nvcf_cli.ncf.launcher.nvcf_function import NvcfFunction, NvcfFunctionAlreadyDeployedError

# Qwen's default temp is 0.000001
DEFAULT_BENCHMARK_VLLM_SAMPLING_TEMPERATURE = 0.000001


class RetryableBenchmarkAttemptError(RuntimeError):
    """Retryable benchmark-attempt failure."""


def _split_image(image: str) -> tuple[str, str]:
    """Split a full image reference into repository and tag."""
    if image != image.strip() or any(ch.isspace() for ch in image):
        msg = f"image must not contain whitespace: {image!r}"
        raise ValueError(msg)
    image_repository, sep, image_tag = image.rpartition(":")
    if not sep or not image_repository or not image_tag or "/" in image_tag:
        msg = f"image must be a full image reference with tag: {image}"
        raise ValueError(msg)
    return image_repository, image_tag


def _log_retryable_attempt_failure(retry_state: tenacity.RetryCallState) -> None:
    """Log retryable attempt failures before the next retry."""
    if retry_state.outcome is None:
        return
    exc = retry_state.outcome.exception()
    if exc is not None:
        logger.warning(str(exc))


def _read_summary_json(summary_path: str, transport_params: dict[str, Any]) -> dict[str, Any]:
    """Load and return summary.json content."""
    with smart_open.open(summary_path, transport_params=transport_params) as f:
        return cast("dict[str, Any]", json.load(f))


def _summary_counts_are_valid(summary_path: str, transport_params: dict[str, Any], limit: int) -> bool:
    """Validate summary count fields used for benchmark integrity checks."""
    try:
        summary_data = _read_summary_json(summary_path, transport_params)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Failed to read summary from {summary_path}: {e!s}")
        return False

    num_input_videos = summary_data.get("num_input_videos")
    has_explicit_selected_count = "num_input_videos_selected" in summary_data
    num_input_videos_selected = summary_data.get("num_input_videos_selected", num_input_videos)
    num_processed_videos = summary_data.get("num_processed_videos")
    is_valid = True

    if not isinstance(num_input_videos_selected, int) or not isinstance(num_processed_videos, int):
        logger.warning(
            f"Invalid summary counts in {summary_path}: {num_input_videos_selected=}, {num_processed_videos=}. "
            "Expected integer values."
        )
        is_valid = False
    elif num_input_videos is not None and not isinstance(num_input_videos, int):
        logger.warning(f"Invalid summary counts in {summary_path}: {num_input_videos=}. Expected integer value.")
        is_valid = False
    elif isinstance(num_input_videos, int) and num_input_videos_selected > num_input_videos:
        logger.warning(
            f"Invalid summary counts in {summary_path}: {num_input_videos_selected=} exceeds {num_input_videos=}."
        )
        is_valid = False
    elif has_explicit_selected_count and num_input_videos_selected > limit:
        logger.warning(
            f"Invalid summary counts in {summary_path}: {num_input_videos_selected=} exceeds configured {limit=}."
        )
        is_valid = False
    elif num_processed_videos > limit:
        logger.warning(
            f"Invalid summary counts in {summary_path}: {num_processed_videos=} exceeds configured {limit=}."
        )
        is_valid = False
    elif num_processed_videos > num_input_videos_selected:
        logger.warning(
            f"Invalid summary counts in {summary_path}: {num_processed_videos=} exceeds {num_input_videos_selected=}."
        )
        is_valid = False

    return is_valid


def _run_benchmark_attempt(  # noqa: PLR0913
    attempt: int,
    max_attempts: int,
    *,
    attempt_output_prefix: str,
    invoke_data: dict[str, Any],
    invoke_config: Path,
    nvcf_function: NvcfFunction,
    backend: str,
    gpu: str,
    instance_type: str,
    deploy_config: Path,
    num_nodes: int,
    max_concurrency: int,
    s3_config_str: str,
    tmpdir_path: Path,
    transport_params: dict[str, Any],
    limit: int,
) -> str:
    """Run one benchmark attempt and return summary path when counts are valid."""
    attempt_summary_path = f"{attempt_output_prefix}/summary.json"
    invoke_data["args"]["output_clip_path"] = attempt_output_prefix
    invoke_config.write_text(json.dumps(invoke_data, indent=2))
    logger.info(f"Invoke data for attempt {attempt}/{max_attempts}:")
    print_json(json.dumps(invoke_data, indent=2))

    logger.info(f"Attempt {attempt}/{max_attempts} with output: {attempt_output_prefix}")
    try:
        with nvcf_function.deploy(backend, gpu, instance_type, deploy_config, num_nodes, max_concurrency):
            nvcf_function.invoke(invoke_config, s3_config_str, out_dir=tmpdir_path, retry_cnt=1)
    except NvcfFunctionAlreadyDeployedError as e:
        msg = "Function is already deployed, this should not happen, previous benchmark may be running."
        raise RuntimeError(msg) from e
    except Exception as e:
        msg = f"Attempt {attempt}/{max_attempts} failed: {e!s}"
        raise RetryableBenchmarkAttemptError(msg) from e

    if not _summary_counts_are_valid(attempt_summary_path, transport_params, limit):
        msg = (
            f"Attempt {attempt}/{max_attempts} produced invalid summary counts at {attempt_summary_path}. "
            "Trying next attempt."
        )
        raise RetryableBenchmarkAttemptError(msg)
    return attempt_summary_path


def _run_benchmark(  # noqa: PLR0913
    max_attempts: int,
    *,
    run_output_prefix: str,
    invoke_data: dict[str, Any],
    invoke_config: Path,
    nvcf_function: NvcfFunction,
    backend: str,
    gpu: str,
    instance_type: str,
    deploy_config: Path,
    num_nodes: int,
    max_concurrency: int,
    s3_config_str: str,
    tmpdir_path: Path,
    transport_params: dict[str, Any],
    limit: int,
) -> str:
    """Run the benchmark with retries and return a validated summary path."""
    retryer = tenacity.Retrying(
        stop=tenacity.stop_after_attempt(max_attempts),
        wait=tenacity.wait_none(),
        retry=tenacity.retry_if_exception_type(RetryableBenchmarkAttemptError),
        before_sleep=_log_retryable_attempt_failure,
        reraise=True,
    )

    try:
        for retry_attempt in retryer:
            with retry_attempt:
                attempt = retry_attempt.retry_state.attempt_number
                attempt_output_prefix = f"{run_output_prefix}/attempt_{attempt}"
                summary_path = _run_benchmark_attempt(
                    attempt=attempt,
                    max_attempts=max_attempts,
                    attempt_output_prefix=attempt_output_prefix,
                    invoke_data=invoke_data,
                    invoke_config=invoke_config,
                    nvcf_function=nvcf_function,
                    backend=backend,
                    gpu=gpu,
                    instance_type=instance_type,
                    deploy_config=deploy_config,
                    num_nodes=num_nodes,
                    max_concurrency=max_concurrency,
                    s3_config_str=s3_config_str,
                    tmpdir_path=tmpdir_path,
                    transport_params=transport_params,
                    limit=limit,
                )
                logger.info(f"Using validated summary at {summary_path}")
                return summary_path
    except RetryableBenchmarkAttemptError as e:
        logger.warning(str(e))
        msg = f"No valid summary was produced after {max_attempts} attempts."
        raise RuntimeError(msg) from e

    msg = "Unexpected retry flow: benchmark attempts exhausted without a terminal exception."
    raise RuntimeError(msg)


def report_metrics(  # noqa: PLR0913
    summary_path: str,
    transport_params: dict[str, Any],
    num_nodes: int,
    gpus_per_node: int,
    *,
    caption: bool,
    splitting_algorithm: str,
    metrics_metadata: dict[str, Any] | None = None,
    kratos_metrics_endpoint: str | None = None,
    kratos_secrets: KratosSecrets | None = None,
    metrics_path: str | None = None,
) -> None:
    """Report metrics to Kratos or save to file.

    Args:
        summary_path: path to summary.json file.
        transport_params: smart_open transport parameters.
        num_nodes: Number of nodes used in the benchmark.
        gpus_per_node: Number of GPUs per node.
        caption: Whether captions are enabled.
        splitting_algorithm: Splitting algorithm used.
        metrics_metadata: Additional metadata to include with the uploaded metrics.
        kratos_metrics_endpoint: Endpoint for sending metrics.
            Must be provided if reporting metrics to Kratos.
        kratos_secrets: Authentication secrets for metrics endpoint.
            If None, metrics are not reported to Kratos.
        metrics_path: path to save metrics to.
            If None, metrics are not saved to a file.

    Raises:
        ValueError: If reporting metrics to Kratos and kratos_metrics_endpoint is not provided.

    """
    logger.info(f"Getting summary metrics from {summary_path}")
    summary_data = _read_summary_json(summary_path, transport_params)

    summary_metrics = make_summary_metrics(
        summary_data, num_nodes, gpus_per_node, caption=caption, env="nvcf", splitting_algorithm=splitting_algorithm
    )
    if metrics_metadata:
        summary_metrics.update(metrics_metadata)

    logger.info("Summary metrics:")
    print_json(json.dumps(summary_metrics, indent=2))

    if metrics_path is not None:
        logger.info(f"Saving metrics to {metrics_path}")
        _transport_params = transport_params if str(metrics_path).startswith("s3://") else None
        with smart_open.open(str(metrics_path), transport_params=_transport_params, mode="w") as f:
            json.dump(summary_metrics, f, indent=2)

    if kratos_secrets is not None:
        if kratos_metrics_endpoint is None:
            msg = "Kratos metrics endpoint is required when reporting metrics to Kratos."
            raise ValueError(msg)

        cloudevent = make_cloudevent(summary_metrics)
        logger.info(f"Pushing metrics to {kratos_metrics_endpoint}")
        response = push_cloudevent(cloudevent, kratos_metrics_endpoint, kratos_secrets)
        logger.info("Response:")
        print_json(json.dumps(response, indent=2))


def nvcf_split_benchmark(  # noqa: PLR0913
    funcid: str,
    version: str,
    nvcf_secrets: NvcfSecrets,
    s3_secrets: S3Secrets,
    captioning_algorithm: str,
    kratos_metrics_token_env: str,
    kratos_bearer_url: str,
    image_repository: str,
    image_tag: str,
    metrics_endpoint: str,
    backend: str,
    gpu: str,
    instance_type: str,
    s3_input_prefix: str,
    s3_output_prefix: str,
    max_concurrency: int,
    limit: int,
    caption: int,
    splitting_algorithm: str,
    num_nodes: int,
    gpus_per_node: int,
    kratos_metrics_endpoint: str,
    metrics_path: str | None,
    max_attempts: int,
    *,
    clip_re_chunk_size: int,
    qwen_use_fp8_weights: bool,
    report_metrics_to_kratos: bool,
    vllm_sampling_temperature: float,
    vllm_use_inflight_batching: bool,
) -> None:
    """Run benchmark tests."""
    nvcf_function = NvcfFunction(
        funcid=funcid,
        version=version,
        key=nvcf_secrets.ngc_key,
        org=nvcf_secrets.ngc_org,
        team="no-team",
    )

    # Load and customize configuration templates
    template_dir = Path(__file__).parent

    with (template_dir / "deploy.json").open() as f:
        deploy_data = json.load(f)

    with (template_dir / "invoke.json").open() as f:
        invoke_data = json.load(f)

    # Update deploy configuration
    deploy_data["configuration"]["image"]["repository"] = image_repository
    deploy_data["configuration"]["image"]["tag"] = image_tag
    deploy_data["configuration"]["metrics"]["remoteWrite"]["endpoint"] = metrics_endpoint

    # Update invoke configuration
    invoke_data["args"].update(
        {
            "input_video_path": s3_input_prefix,
            "captioning_algorithm": captioning_algorithm,
            "splitting_algorithm": splitting_algorithm,
            "qwen_preprocess_dtype": "float16",
            "generate_captions": caption == 1,
            "limit": limit,
            "clip_re_chunk_size": clip_re_chunk_size,
            "qwen_use_fp8_weights": qwen_use_fp8_weights,
            "vllm_sampling_temperature": vllm_sampling_temperature,
            "vllm_use_inflight_batching": vllm_use_inflight_batching,
        }
    )

    logger.info("Deploy data:")
    print_json(json.dumps(deploy_data, indent=2))

    # Prepare S3 credentials
    s3_config = f"""[default]
aws_access_key_id = {s3_secrets.aws_access_key_id}
aws_secret_access_key = {s3_secrets.aws_secret_access_key}
aws_region = {s3_secrets.aws_region}
"""

    transport_params = {
        "client": boto3.client(
            "s3",
            aws_access_key_id=s3_secrets.aws_access_key_id,
            aws_secret_access_key=s3_secrets.aws_secret_access_key,
            region_name=s3_secrets.aws_region,
        )
    }
    if report_metrics_to_kratos:
        # Verify that the kratos secret can be successfully obtained before a long running benchmark.
        KratosSecrets.from_env(
            kratos_metrics_token_env,
            kratos_bearer_url,
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        logger.info(
            f"Benchmarking with {caption=} {num_nodes=} {captioning_algorithm=} {splitting_algorithm=}, "
            f"input: {s3_input_prefix}, output: {s3_output_prefix}"
        )
        tmpdir_path = Path(tmpdir)

        deploy_config = tmpdir_path / "deploy.json"
        invoke_config = tmpdir_path / "invoke.json"
        s3_config_file = tmpdir_path / "s3_cred"

        deploy_config.write_text(json.dumps(deploy_data, indent=2))
        s3_config_file.write_text(s3_config)
        s3_config_str = _get_s3_config_str(s3_config_file)

        if s3_config_str is None:
            msg = "Failed to get S3 config string"
            raise ValueError(msg)

        run_output_prefix = f"{s3_output_prefix}/run_{tmpdir_path.name}"
        accepted_summary_path = _run_benchmark(
            max_attempts=max_attempts,
            run_output_prefix=run_output_prefix,
            invoke_data=invoke_data,
            invoke_config=invoke_config,
            nvcf_function=nvcf_function,
            backend=backend,
            gpu=gpu,
            instance_type=instance_type,
            deploy_config=deploy_config,
            num_nodes=num_nodes,
            max_concurrency=max_concurrency,
            s3_config_str=s3_config_str,
            tmpdir_path=tmpdir_path,
            transport_params=transport_params,
            limit=limit,
        )

        kratos_secrets: KratosSecrets | None = None
        if report_metrics_to_kratos:
            # Get secrets immediately before reporting - benchmarking time may exceed the token's expiration date.
            kratos_secrets = KratosSecrets.from_env(
                kratos_metrics_token_env,
                kratos_bearer_url,
            )

        report_metrics(
            summary_path=accepted_summary_path,
            transport_params=transport_params,
            num_nodes=num_nodes,
            gpus_per_node=gpus_per_node,
            caption=bool(caption),
            splitting_algorithm=splitting_algorithm,
            metrics_metadata={
                "image": image_repository,
                "tag": image_tag,
                "scheduled": os.getenv("CI_PIPELINE_SOURCE") == "schedule",
                "backend": backend,
                "gpu": gpu,
                "instance_type": instance_type,
                "captioning_algorithm": captioning_algorithm,
                "gpus_per_node": gpus_per_node,
                "max_concurrency": max_concurrency,
                "limit": limit,
                "funcid": funcid,
                "version": version,
                "ci_pipeline_source": os.getenv("CI_PIPELINE_SOURCE", "unknown"),
                "gitlab_user_login": os.getenv("GITLAB_USER_LOGIN", "unknown"),
                "input_path": s3_input_prefix,
                "output_path": s3_output_prefix,
                "vllm_sampling_temperature": vllm_sampling_temperature,
            },
            kratos_secrets=kratos_secrets,
            kratos_metrics_endpoint=kratos_metrics_endpoint,
            metrics_path=metrics_path,
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run benchmark tests on NVCF cluster.")
    parser.add_argument("--num-nodes", type=int, default=1, help="Number of nodes to use.")
    parser.add_argument("--caption", type=int, default=0, help="Whether to use captioning for the benchmark.")
    parser.add_argument("--funcid", type=str, required=True, help="Function ID to run.")
    parser.add_argument("--version", type=str, required=True, help="Function version to use.")
    parser.add_argument("--captioning-algorithm", type=str, required=True, help="Captioning algorithm to use")
    parser.add_argument(
        "--splitting-algorithm",
        type=str,
        required=True,
        choices=["transnetv2", "fixed-stride"],
        help="Splitting algorithm to use.",
    )
    parser.add_argument(
        "--ngc-org-env",
        type=str,
        required=False,
        default="PERF_NGC_NVCF_ORG_ID",
        help="NGC organization ID environment variable.",
    )
    parser.add_argument(
        "--ngc-key-env",
        type=str,
        required=False,
        default="PERF_NGC_NVCF_API_KEY",
        help="NGC API key environment variable.",
    )
    parser.add_argument("--image", type=str, required=False, help="Full image reference to use for the benchmark.")
    parser.add_argument(
        "--image-repository",
        type=str,
        required=False,
        help="Image repository to use for the benchmark. Prefer --image for new callers.",
    )
    parser.add_argument(
        "--image-tag",
        type=str,
        required=False,
        help="Image tag to use for the benchmark. Prefer --image for new callers.",
    )
    parser.add_argument(
        "--metrics-endpoint", type=str, required=True, help="Metrics endpoint to use for the benchmark."
    )
    parser.add_argument("--backend", type=str, required=True, help="Backend to use for the benchmark.")
    parser.add_argument("--gpu", type=str, required=True, help="GPU")
    parser.add_argument("--instance-type", type=str, required=True, help="Instance type..")
    parser.add_argument("--s3-input-prefix", type=str, required=True, help="S3 input prefix.")
    parser.add_argument("--s3-output-prefix", type=str, required=True, help="S3 output prefix.")
    parser.add_argument("--max-concurrency", type=int, required=True, default=2, help="Max concurrency.")
    parser.add_argument(
        "--aws-access-key-id-env",
        type=str,
        required=False,
        default="PERF_AWS_ACCESS_KEY_ID",
        help="AWS access key ID environment variable.",
    )
    parser.add_argument(
        "--aws-secret-access-key-env",
        type=str,
        required=False,
        default="PERF_AWS_SECRET_ACCESS_KEY",
        help="AWS secret access key environment variable.",
    )
    parser.add_argument(
        "--aws-region-env", type=str, required=False, default="PERF_AWS_REGION", help="AWS region environment variable."
    )
    parser.add_argument("--limit", type=int, required=True, default=5000, help="Limit the number of videos to process.")
    parser.add_argument(
        "--kratos-metrics-endpoint",
        type=str,
        required=False,
        default=None,
        help="URL of destination for the metrics to push to Kratos.",
    )
    parser.add_argument(
        "--kratos-bearer-url",
        type=str,
        required=False,
        default=None,
        help="URL of the bearer token endpoint for Kratos.",
    )
    parser.add_argument(
        "--kratos-metrics-token-env",
        type=str,
        required=False,
        default="PERF_KRATOS_METRICS_TOKEN",
        help="Environment variable that contains the token to use to push metrics to Kratos.",
    )
    parser.add_argument("--gpus-per-node", type=int, required=True, default=8, help="Number of GPUs per node.")
    parser.add_argument(
        "--clip-re-chunk-size",
        type=int,
        required=False,
        default=32,
        help="Number of clips per chunk after transcoding stage.",
    )
    parser.add_argument(
        "--metrics-path",
        type=str,
        required=False,
        default=None,
        help="Path to save metrics json to. Can be used to save metrics to a file instead of reporting to Kratos.",
    )
    parser.add_argument(
        "--qwen-use-fp8-weights",
        type=int,
        required=False,
        default=0,
        help="Whether to use FP8 weights for Qwen.",
    )
    parser.add_argument(
        "--report-metrics-to-kratos",
        action="store_true",
        help="Whether to report metrics to Kratos.",
    )
    parser.add_argument(
        "--vllm-use-inflight-batching",
        type=int,
        required=False,
        default=1,
        help="Whether to use inflight batching with vllm.",
    )
    parser.add_argument(
        "--vllm-sampling-temperature",
        type=float,
        required=False,
        default=DEFAULT_BENCHMARK_VLLM_SAMPLING_TEMPERATURE,
        help="Temperature for vLLM sampling in benchmark invoke args.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        required=False,
        default=2,
        help="Maximum number of benchmark attempts with unique output paths.",
    )
    args = parser.parse_args()
    if args.image:
        if args.image_repository or args.image_tag:
            parser.error("--image cannot be used with --image-repository or --image-tag.")
        try:
            args.image_repository, args.image_tag = _split_image(args.image)
        except ValueError as e:
            parser.error(str(e))
    elif not args.image_repository or not args.image_tag:
        parser.error("either --image or both --image-repository and --image-tag are required.")
    return args


def main() -> None:
    """Run benchmark tests."""
    args = _parse_args()
    nvcf_secrets = NvcfSecrets.from_env(
        args.ngc_org_env,
        args.ngc_key_env,
    )

    s3_secrets = S3Secrets.from_env(
        args.aws_access_key_id_env,
        args.aws_secret_access_key_env,
        args.aws_region_env,
    )

    args.qwen_use_fp8_weights = bool(args.qwen_use_fp8_weights)
    args.vllm_use_inflight_batching = bool(args.vllm_use_inflight_batching)
    if args.max_attempts < 1:
        msg = "max-attempts must be at least 1."
        raise ValueError(msg)

    if args.metrics_path:
        logger.info(f"Saving metrics to {args.metrics_path}")
        if not str(args.metrics_path).startswith("s3://"):
            args.metrics_path = Path(args.metrics_path)
            args.metrics_path.parent.mkdir(parents=True, exist_ok=True)

    if args.report_metrics_to_kratos:
        if not args.kratos_metrics_endpoint:
            msg = "Kratos metrics endpoint is required when reporting metrics to Kratos."
            raise ValueError(msg)
        if not args.kratos_bearer_url:
            msg = "Kratos bearer URL is required when reporting metrics to Kratos."
            raise ValueError(msg)
        if not args.kratos_metrics_token_env:
            msg = "Kratos metrics token environment variable is required when reporting metrics to Kratos."
            raise ValueError(msg)

    nvcf_split_benchmark(
        funcid=args.funcid,
        version=args.version,
        nvcf_secrets=nvcf_secrets,
        s3_secrets=s3_secrets,
        captioning_algorithm=args.captioning_algorithm,
        kratos_metrics_token_env=args.kratos_metrics_token_env,
        kratos_bearer_url=args.kratos_bearer_url,
        image_repository=args.image_repository,
        image_tag=args.image_tag,
        metrics_endpoint=args.metrics_endpoint,
        backend=args.backend,
        gpu=args.gpu,
        instance_type=args.instance_type,
        s3_input_prefix=args.s3_input_prefix,
        s3_output_prefix=args.s3_output_prefix,
        max_concurrency=args.max_concurrency,
        limit=args.limit,
        caption=args.caption,
        splitting_algorithm=args.splitting_algorithm,
        num_nodes=args.num_nodes,
        gpus_per_node=args.gpus_per_node,
        kratos_metrics_endpoint=args.kratos_metrics_endpoint,
        metrics_path=args.metrics_path,
        max_attempts=args.max_attempts,
        report_metrics_to_kratos=args.report_metrics_to_kratos,
        clip_re_chunk_size=args.clip_re_chunk_size,
        qwen_use_fp8_weights=args.qwen_use_fp8_weights,
        vllm_sampling_temperature=args.vllm_sampling_temperature,
        vllm_use_inflight_batching=args.vllm_use_inflight_batching,
    )


if __name__ == "__main__":
    main()
