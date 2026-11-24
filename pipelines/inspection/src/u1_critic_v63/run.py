from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping

from critic_workflows import PillowImageTool

from .input_manifest import (
    discover_images,
    manifest_rows,
    select_images_from_jsonl,
    to_image_asset,
)
from .consolidation import (
    ConsolidationValidationError,
    ResultConsolidator,
)
from .model_adapter import OpenAICompatibleModelClient, TraceWriter
from .adaptive_image import overview_image
from .workflows import TAXONOMY, build_critics, critic_context


WORKFLOW_NAMES = (
    "specialist_cascade",
    "evidence_seeking_agent",
    "adversarial_consensus",
)


def append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, default=str) + "\n")
        handle.flush()


def completed_units(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    output: set[tuple[str, str]] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            workflow = str(row.get("workflow", ""))
            if row.get("status") != "ok":
                continue
            if (
                workflow == "adversarial_consensus"
                and not isinstance(row.get("consolidated_result"), Mapping)
            ):
                continue
            output.add((str(row["sample_id"]), workflow))
    return output


def attach_cumulative_unit_timing(
    row: dict[str, Any],
    attempt_elapsed_seconds: list[float],
) -> dict[str, Any]:
    """Attach retry-inclusive unit latency without changing legacy fields."""

    timing = row.setdefault("timing", {})
    timing["unit_attempts_total"] = len(attempt_elapsed_seconds)
    timing["unit_total_elapsed_seconds"] = round(
        sum(attempt_elapsed_seconds),
        4,
    )
    return row


async def iter_completed_units(
    pending,
    run_unit,
    *,
    max_units_in_flight: int,
):
    if max_units_in_flight <= 0:
        raise ValueError("max_units_in_flight must be positive")
    limiter = asyncio.Semaphore(max_units_in_flight)

    async def tagged(record, workflow):
        async with limiter:
            try:
                outcome = await run_unit(record, workflow)
            except Exception as exc:
                outcome = exc
            return record, workflow, outcome

    tasks = [
        asyncio.create_task(tagged(record, workflow))
        for record, workflow in pending
    ]
    try:
        for task in asyncio.as_completed(tasks):
            yield await task
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    temporary.replace(path)


async def consolidate_if_needed(
    workflow: str,
    image,
    result: Mapping[str, Any],
    consolidator,
    *,
    skip: bool,
) -> dict[str, Any] | None:
    if workflow != "adversarial_consensus" or skip:
        return None
    consolidated = await consolidator.consolidate(image, result)
    metadata = result.get("metadata", {})
    coverage_status = (
        str(metadata.get("coverage_status", ""))
        if isinstance(metadata, Mapping)
        else ""
    )
    if coverage_status and coverage_status != "ok":
        consolidated["quality_verdict"] = consolidated.get(
            "verdict",
            "uncertain",
        )
        consolidated["verdict"] = "uncertain"
        consolidated["coverage_status"] = coverage_status
        consolidated["findings_non_exhaustive"] = True
        consolidated["harness_abstention_reason"] = str(
            metadata.get(
                "harness_abstention_reason",
                "内容覆盖账本未全部核销。",
            )
        )
    return consolidated


async def consolidate_with_failure_record(
    *,
    workflow: str,
    image,
    result: Mapping[str, Any],
    consolidator,
    skip: bool,
    error_path: Path,
    sample_id: str,
    model: str,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    try:
        consolidated = await consolidate_if_needed(
            workflow,
            image,
            result,
            consolidator,
            skip=skip,
        )
    except ConsolidationValidationError as exc:
        error = {
            "sample_id": sample_id,
            "workflow": workflow,
            "status": "error",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "validation_errors": list(exc.errors),
            "model": model,
        }
        append_jsonl(error_path, error)
        return None, error
    return consolidated, None


async def consolidate_with_retries(
    *,
    workflow: str,
    image,
    result: Mapping[str, Any],
    consolidator,
    skip: bool,
    max_retries: int,
) -> dict[str, Any] | None:
    """Finish consolidation inside the live unit or fail the whole unit."""

    if max_retries < 0:
        raise ValueError("max_retries must be non-negative")
    last_error: ConsolidationValidationError | None = None
    for _ in range(max_retries + 1):
        try:
            return await consolidate_if_needed(
                workflow,
                image,
                result,
                consolidator,
                skip=skip,
            )
        except ConsolidationValidationError as exc:
            last_error = exc
    assert last_error is not None
    raise last_error


def v41_flash_enabled(args: argparse.Namespace) -> bool:
    """Return whether the public Flash profile or its legacy alias is set."""

    return bool(
        getattr(args, "v41_flash", False)
        or getattr(args, "v423_latency_experiment", False)
    )


def v61_enabled(args: argparse.Namespace) -> bool:
    """Return whether the V6.1 experimental profile is selected."""

    return bool(getattr(args, "v61", False))


def v62_enabled(args: argparse.Namespace) -> bool:
    """Return whether the V6.2 experimental profile is selected."""

    return bool(getattr(args, "v62", False))


def v63_enabled(args: argparse.Namespace) -> bool:
    """Return whether the V6.3 coverage-authoritative profile is selected."""

    return bool(getattr(args, "v63", False))


def structured_recovery_enabled(args: argparse.Namespace) -> bool:
    """Use request-local structured recovery for V6.2-derived profiles."""

    return v62_enabled(args) or v63_enabled(args)


def component_repair_enabled(args: argparse.Namespace) -> bool:
    """Use component-local online repair for profiles that validated it."""

    return bool(
        getattr(args, "coverage_mode", None) == "harness_v42"
        and (v61_enabled(args) or v62_enabled(args) or v63_enabled(args))
    )


def flash_optimizations_enabled(args: argparse.Namespace) -> bool:
    """Return whether validated V6.0 scheduling is part of the profile."""

    return (
        v41_flash_enabled(args)
        or v61_enabled(args)
        or v62_enabled(args)
        or v63_enabled(args)
    )


def pipeline_version_for_args(args: argparse.Namespace) -> str | None:
    """Resolve the public pipeline version persisted in new artifacts."""

    if getattr(args, "v63_dense_path_latency", False):
        return "v6.3-i"
    if v63_enabled(args):
        return "v6.3"
    if v62_enabled(args):
        return "v6.2"
    if v61_enabled(args):
        return "v6.1"
    if v41_flash_enabled(args):
        return "v4.1-flash"
    if getattr(args, "v42_quality_first", False):
        return "v4.2.1-quality-first"
    coverage_mode = getattr(args, "coverage_mode", None)
    if coverage_mode == "harness_v42":
        return "v4.2-experimental"
    if coverage_mode == "harness_v41":
        return "v4.1"
    if coverage_mode == "harness_v5":
        return "v5"
    return None


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        help=(
            "Optional JSONL allow-list. Only discovered images referenced by "
            "its image path field are evaluated, preserving JSONL order."
        ),
    )
    parser.add_argument(
        "--input-jsonl-path-field",
        default="image_absolute_path",
        help="JSONL field containing each absolute image path.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="Qwen3.6-27B-agen-0722-agentv4")
    parser.add_argument("--api-key-env", default="CRITIC_API_KEY")
    parser.add_argument(
        "--workflow",
        action="append",
        choices=WORKFLOW_NAMES,
        dest="selected_workflows",
        help="Workflow to run; repeat to select multiple. Defaults to all three.",
    )
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--limit-images", type=int)
    selection.add_argument("--all", action="store_true")
    parser.add_argument("--confirm-run-all", default="")
    parser.add_argument("--max-in-flight", type=int, default=6)
    parser.add_argument(
        "--max-images-in-flight",
        type=int,
        default=1,
        help=(
            "Maximum image/workflow units evaluated concurrently. Model "
            "requests remain globally bounded by --max-in-flight."
        ),
    )
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--max-tokens", type=int, default=2400)
    parser.add_argument(
        "--consolidation-max-tokens",
        type=int,
        default=0,
        help=(
            "Optional per-request token ceiling for final consolidation; "
            "0 inherits --max-tokens."
        ),
    )
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument(
        "--consolidation-retries",
        type=int,
        default=2,
        help="Retries final consolidation inside the same pipeline unit.",
    )
    parser.add_argument(
        "--unit-retries",
        type=int,
        default=2,
        help="Restarts an incomplete image/workflow unit from its beginning.",
    )
    parser.add_argument(
        "--coverage-mode",
        choices=(
            "grid",
            "adaptive",
            "harness",
            "harness_v41",
            "harness_v42",
            "harness_v5",
        ),
        default="grid",
        help=(
            "Coverage strategy for adversarial consensus: V2 fixed grid "
            "V3 adaptive content regions, or V4 reliability harness."
        ),
    )
    parser.add_argument(
        "--v42-quality-first",
        action="store_true",
        help=(
            "V4.2.1 quality-preserving profile: retain V4.1 multi-role "
            "discovery, model Judge/Skeptic semantics, and online model "
            "consolidation; enable only bounded micro-batching, concurrent "
            "inventories, and an extra text crosscheck."
        ),
    )
    parser.add_argument(
        "--v422-low-latency",
        action="store_true",
        help=(
            "V4.2.2 scheduling-only profile: keep the V4.2.1 quality path, "
            "require one image in flight, and allow dedicated intra-image "
            "Skeptic batch concurrency."
        ),
    )
    parser.add_argument(
        "--v41-flash",
        action="store_true",
        help=(
            "Public V4.1 Flash profile: preserve the complete V4.1 quality "
            "chain while enabling the validated V4.2.3 intra-image "
            "scheduling and latency optimizations."
        ),
    )
    parser.add_argument(
        "--v61",
        action="store_true",
        help=(
            "V6.1 profile: retain the complete V6.0 quality chain and add "
            "candidate-streamed Skeptic scheduling plus safe component-local "
            "online consolidation repair."
        ),
    )
    parser.add_argument(
        "--v62",
        action="store_true",
        help=(
            "V6.2 profile: retain the V6.0 stage boundary while enabling "
            "adaptive atomic region batching, node-local structured-output "
            "recovery, and component-local online consolidation repair."
        ),
    )
    parser.add_argument(
        "--v63",
        action="store_true",
        help=(
            "V6.3 coverage-authoritative profile: keep exhaustive local "
            "text/Artifact coverage and online model consolidation while "
            "removing redundant full-image proposer, Skeptic, and model-Judge "
            "calls."
        ),
    )
    parser.add_argument(
        "--v423-latency-experiment",
        action="store_true",
        help=(
            "Deprecated compatibility alias for --v41-flash."
        ),
    )
    parser.add_argument(
        "--v63-selective-review",
        action="store_true",
        help=(
            "V6.3-D ablation: inventory every region but locally inspect only "
            "regions triaged as suspicious or uncertain."
        ),
    )
    parser.add_argument(
        "--v63-entity-safe-review",
        action="store_true",
        help=(
            "V6.3-E ablation: retain selective text/layout review but force "
            "lossless local review for every independently inventoried "
            "non-text entity."
        ),
    )
    parser.add_argument(
        "--v63-artifact-scout",
        action="store_true",
        help=(
            "V6.3-F ablation: add one independently prompted full-image "
            "Artifact scout whose findings retain Skeptic verification and "
            "online consolidation."
        ),
    )
    parser.add_argument(
        "--v63-dual-artifact-scout",
        action="store_true",
        help=(
            "V6.3-G ablation: run independent structure/interaction and "
            "rendering/material Artifact scouts, then trust spatial peer "
            "agreement while sending singleton findings to Skeptic."
        ),
    )
    parser.add_argument(
        "--v63-dense-path-latency",
        action="store_true",
        help=(
            "V6.3-I ablation: retain the complete V6.3-H quality chain while "
            "optimizing dense-region batching, inventory payloads, and "
            "online consolidation recovery."
        ),
    )
    parser.add_argument("--v42-text-batch-size", type=int, default=2)
    parser.add_argument(
        "--v42-text-crosscheck-batch-size",
        type=int,
        default=0,
        help=(
            "Text regions per independent second-pass request; 0 reuses "
            "--v42-text-batch-size."
        ),
    )
    parser.add_argument("--v42-artifact-batch-size", type=int, default=4)
    parser.add_argument(
        "--v42-inspection-concurrency",
        type=int,
        default=12,
        help="Concurrent V4.2 region-inspection batches within one image.",
    )
    parser.add_argument("--v42-challenge-batch-size", type=int, default=4)
    parser.add_argument(
        "--v42-challenge-concurrency",
        type=int,
        default=0,
        help=(
            "Concurrent V4.2 Skeptic batches within one image; 0 selects 16 "
            "for V4.2.2 and 4 otherwise."
        ),
    )
    parser.add_argument("--v42-trust-preverified", action="store_true")
    parser.add_argument("--v42-deterministic-judge", action="store_true")
    parser.add_argument("--v42-combined-inventory", action="store_true")
    parser.add_argument("--v42-combined-coverage-audit", action="store_true")
    parser.add_argument(
        "--v42-eager-inventory",
        action="store_true",
        help=(
            "Experimental: start text/entity inventories concurrently with "
            "semantic-zone enumeration, then reconcile them spatially before "
            "the unchanged coverage ledger and recovery gate."
        ),
    )
    parser.add_argument(
        "--v42-text-crosscheck",
        action="store_true",
        help=(
            "Experimental: run an independent batched second-pass character "
            "scan for every text region in parallel with the primary text "
            "inspection, then union supported visible findings."
        ),
    )
    parser.add_argument(
        "--v42-decompose-broad-text",
        action="store_true",
        help=(
            "Experimental: after a broad text panel is confirmed defective, "
            "run a localization-only crop pass to split spatially independent "
            "repaint targets before consensus and consolidation."
        ),
    )
    parser.add_argument(
        "--v42-coverage-only",
        action="store_true",
        help=(
            "Experimental: use the exhaustive dual-channel coverage auditor "
            "without the three redundant full-image proposer roles."
        ),
    )
    parser.add_argument(
        "--v42-single-artifact-proposer",
        action="store_true",
        help=(
            "Experimental: replace the three full-image proposer roles with "
            "one unified non-text Artifact proposer."
        ),
    )
    parser.add_argument(
        "--v42-review-original-max-side",
        type=int,
        default=0,
        help=(
            "Downscale only the repeated full-image overview in V4.2 batch "
            "inspection/challenge calls; detail crops remain lossless."
        ),
    )
    parser.add_argument(
        "--v42-consolidation-image-max-side",
        type=int,
        default=0,
        help=(
            "Experimental: downscale only the full image supplied to V4.2 "
            "final consolidation. Bboxes and source findings are unchanged."
        ),
    )
    parser.add_argument(
        "--v42-inventory-overview-max-side",
        type=int,
        default=0,
        help=(
            "Experimental: downscale V4.2 semantic-zone and non-text "
            "inventory overviews. Text inventory remains at original "
            "resolution."
        ),
    )
    parser.add_argument(
        "--v42-deterministic-consolidation",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--v42-deterministic-source-consolidation",
        action="store_true",
        help=(
            "Experimental: build final edit regions directly from supported "
            "V4.2 findings, then apply the same duplicate/nested-region "
            "normalization and strict validation without a model "
            "consolidation call. Invalid cases fall back to model "
            "consolidation."
        ),
    )
    parser.add_argument(
        "--skip-consolidation",
        action="store_true",
        help="Debug only: omit the post-Judge consolidation stage.",
    )
    args = parser.parse_args(argv)
    args.legacy_version_alias = (
        "v4.1-flash.1" if v61_enabled(args) else None
    )
    if flash_optimizations_enabled(args):
        args.v422_low_latency = True
    if args.v422_low_latency:
        args.v42_quality_first = True
    if v63_enabled(args):
        args.v42_coverage_only = not (
            args.v63_artifact_scout or args.v63_dual_artifact_scout
        )
        args.v42_single_artifact_proposer = args.v63_artifact_scout
        args.v42_trust_preverified = True
        args.v42_deterministic_judge = True
    if args.v42_challenge_concurrency == 0:
        args.v42_challenge_concurrency = (
            16 if args.v422_low_latency else 4
        )
    if args.limit_images is not None and args.limit_images <= 0:
        parser.error("--limit-images must be positive")
    if args.all and args.confirm_run_all != "YES":
        parser.error("--all requires --confirm-run-all YES")
    if args.max_in_flight <= 0:
        parser.error("--max-in-flight must be positive")
    if args.max_tokens <= 0:
        parser.error("--max-tokens must be positive")
    if args.consolidation_max_tokens < 0:
        parser.error("--consolidation-max-tokens must be non-negative")
    if args.max_images_in_flight <= 0:
        parser.error("--max-images-in-flight must be positive")
    if args.v42_challenge_concurrency <= 0:
        parser.error("--v42-challenge-concurrency must be positive")
    if args.v42_inspection_concurrency <= 0:
        parser.error("--v42-inspection-concurrency must be positive")
    if args.consolidation_retries < 0:
        parser.error("--consolidation-retries must be non-negative")
    if args.unit_retries < 0:
        parser.error("--unit-retries must be non-negative")
    if (
        args.coverage_mode
        in {"harness_v41", "harness_v42", "harness_v5"}
        and args.skip_consolidation
    ):
        parser.error("V4.1/V4.2/V5 requires final consolidation")
    if args.v42_quality_first:
        if args.coverage_mode != "harness_v42":
            parser.error("--v42-quality-first requires --coverage-mode harness_v42")
        unsafe_options = {
            "--v42-trust-preverified": (
                args.v42_trust_preverified and not v63_enabled(args)
            ),
            "--v42-deterministic-judge": (
                args.v42_deterministic_judge and not v63_enabled(args)
            ),
            "--v42-combined-inventory": (
                args.v42_combined_inventory and not v63_enabled(args)
            ),
            "--v42-combined-coverage-audit": (
                args.v42_combined_coverage_audit and not v63_enabled(args)
            ),
            "--v42-decompose-broad-text": args.v42_decompose_broad_text,
            "--v42-coverage-only": (
                args.v42_coverage_only and not v63_enabled(args)
            ),
            "--v42-single-artifact-proposer": (
                args.v42_single_artifact_proposer
                and not args.v63_artifact_scout
            ),
            "--v42-deterministic-source-consolidation": (
                args.v42_deterministic_source_consolidation
            ),
            "--v42-review-original-max-side": (
                args.v42_review_original_max_side > 0
                and not v62_enabled(args)
            ),
            "--v42-consolidation-image-max-side": (
                args.v42_consolidation_image_max_side > 0
                and not flash_optimizations_enabled(args)
            ),
            "--v42-inventory-overview-max-side": (
                args.v42_inventory_overview_max_side > 0
            ),
        }
        enabled_unsafe = [
            name for name, enabled in unsafe_options.items() if enabled
        ]
        if enabled_unsafe:
            parser.error(
                "--v42-quality-first forbids quality-changing options: "
                + ", ".join(enabled_unsafe)
            )
        discovery_batch_limit = (
            6 if flash_optimizations_enabled(args) else 2
        )
        challenge_batch_limit = 3 if v62_enabled(args) else 2
        if max(
            args.v42_text_batch_size,
            args.v42_text_crosscheck_batch_size,
            args.v42_artifact_batch_size,
        ) > discovery_batch_limit or (
            args.v42_challenge_batch_size > challenge_batch_limit
        ):
            parser.error(
                "--v42-quality-first limits challenge batches to "
                f"{challenge_batch_limit} and "
                f"discovery batches to {discovery_batch_limit}"
            )
        if args.max_tokens < 16000:
            parser.error(
                "--v42-quality-first requires --max-tokens 16000 or higher"
            )
        if 0 < args.consolidation_max_tokens < 16000:
            parser.error(
                "--v42-quality-first requires --consolidation-max-tokens "
                "16000 or higher"
            )
        # These two optimizations change only scheduling/coverage redundancy.
        # Semantic decisions still go through the model Skeptic, Judge, and
        # final online consolidation.
        args.v42_eager_inventory = True
        args.v42_text_crosscheck = True
        # Model consolidation remains mandatory. Deterministic logic is used
        # only after model output for duplicate/nested/source-accounting
        # cleanup; the independent-source and preconsolidated shortcuts are
        # explicitly disabled below for this profile.
        args.v42_deterministic_consolidation = True
    if (
        args.v422_low_latency
        and not flash_optimizations_enabled(args)
        and args.max_images_in_flight != 1
    ):
        parser.error(
            "--v422-low-latency requires --max-images-in-flight 1"
        )
    if args.v42_coverage_only and args.v42_single_artifact_proposer:
        parser.error(
            "--v42-coverage-only and --v42-single-artifact-proposer are "
            "mutually exclusive"
        )
    if args.v63_selective_review and not v63_enabled(args):
        parser.error("--v63-selective-review requires --v63")
    if args.v63_entity_safe_review and not (
        v63_enabled(args) and args.v63_selective_review
    ):
        parser.error(
            "--v63-entity-safe-review requires --v63 and "
            "--v63-selective-review"
        )
    if args.v63_artifact_scout and not v63_enabled(args):
        parser.error("--v63-artifact-scout requires --v63")
    if args.v63_artifact_scout and args.v63_entity_safe_review:
        parser.error(
            "--v63-artifact-scout and --v63-entity-safe-review are "
            "mutually exclusive"
        )
    if args.v63_dual_artifact_scout and not v63_enabled(args):
        parser.error("--v63-dual-artifact-scout requires --v63")
    if args.v63_dual_artifact_scout and (
        args.v63_artifact_scout or args.v63_entity_safe_review
    ):
        parser.error(
            "--v63-dual-artifact-scout is mutually exclusive with "
            "--v63-artifact-scout and --v63-entity-safe-review"
        )
    if args.v63_dense_path_latency and not (
        v63_enabled(args)
        and args.v63_selective_review
        and args.v63_dual_artifact_scout
    ):
        parser.error(
            "--v63-dense-path-latency requires --v63, "
            "--v63-selective-review, and --v63-dual-artifact-scout"
        )
    if args.v42_review_original_max_side < 0:
        parser.error("--v42-review-original-max-side must be non-negative")
    if args.v42_consolidation_image_max_side < 0:
        parser.error(
            "--v42-consolidation-image-max-side must be non-negative"
        )
    if args.v42_inventory_overview_max_side < 0:
        parser.error(
            "--v42-inventory-overview-max-side must be non-negative"
        )
    if args.v42_text_crosscheck_batch_size == 0:
        args.v42_text_crosscheck_batch_size = args.v42_text_batch_size
    for name in (
        "v42_text_batch_size",
        "v42_text_crosscheck_batch_size",
        "v42_artifact_batch_size",
        "v42_challenge_batch_size",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    args.workflows = tuple(
        dict.fromkeys(args.selected_workflows or WORKFLOW_NAMES)
    )
    args.api_key = os.environ.get(args.api_key_env)
    return args


async def run_evaluation(args) -> None:
    records = discover_images(args.input_root)
    if args.input_jsonl is not None:
        records = select_images_from_jsonl(
            records,
            args.input_jsonl,
            path_field=args.input_jsonl_path_field,
        )
    if args.limit_images is not None:
        records = records[: args.limit_images]
    args.output_root.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(
        args.output_root / "manifest.json",
        {
            "input_root": str(args.input_root.resolve()),
            "images": manifest_rows(records),
        },
    )
    _write_json_atomic(
        args.output_root / "run_config.json",
        {
            "model": args.model,
            "base_url": args.base_url,
            "image_count": len(records),
            "input_jsonl": (
                str(args.input_jsonl.resolve()) if args.input_jsonl else None
            ),
            "input_jsonl_path_field": (
                args.input_jsonl_path_field if args.input_jsonl else None
            ),
            "workflows": list(args.workflows),
            "taxonomy": list(TAXONOMY),
            "max_in_flight": args.max_in_flight,
            "max_images_in_flight": args.max_images_in_flight,
            "timeout": args.timeout,
            "max_tokens": args.max_tokens,
            "consolidation_max_tokens": args.consolidation_max_tokens,
            "retries": args.retries,
            "consolidation_retries": args.consolidation_retries,
            "unit_retries": args.unit_retries,
            "coverage_mode": args.coverage_mode,
            "pipeline_version": pipeline_version_for_args(args),
            "legacy_version_alias": args.legacy_version_alias,
            "v42_text_batch_size": args.v42_text_batch_size,
            "v42_quality_first": args.v42_quality_first,
            "v42_model_consolidation_required": args.v42_quality_first,
            "v42_online_validation_repair_rounds": (
                3 if args.v42_quality_first else 1
            ),
            "v42_text_crosscheck_batch_size": (
                args.v42_text_crosscheck_batch_size
            ),
            "v42_artifact_batch_size": args.v42_artifact_batch_size,
            "v42_inspection_concurrency": args.v42_inspection_concurrency,
            "v42_challenge_batch_size": args.v42_challenge_batch_size,
            "v42_challenge_concurrency": (
                args.v42_challenge_concurrency
            ),
            "v422_low_latency": args.v422_low_latency,
            "v41_flash": v41_flash_enabled(args),
            "v61": v61_enabled(args),
            "v61_streaming_skeptic": v61_enabled(args),
            "v61_component_repair": v61_enabled(args),
            "v62": v62_enabled(args),
            "v62_component_repair": v62_enabled(args),
            "v62_structured_recovery": structured_recovery_enabled(args),
            "v62_adaptive_batching": structured_recovery_enabled(args),
            "v63": v63_enabled(args),
            "v63_selective_review": bool(args.v63_selective_review),
            "v63_entity_safe_review": bool(
                args.v63_entity_safe_review
            ),
            "v63_artifact_scout": bool(args.v63_artifact_scout),
            "v63_dual_artifact_scout": bool(
                args.v63_dual_artifact_scout
            ),
            "v63_dense_path_latency": bool(
                args.v63_dense_path_latency
            ),
            "v423_latency_experiment": bool(
                args.v423_latency_experiment
            ),
            "v423_overlap_coverage_inspection": (
                flash_optimizations_enabled(args)
            ),
            "v423_deterministic_issue_hydration": (
                flash_optimizations_enabled(args)
            ),
            "v42_trust_preverified": args.v42_trust_preverified,
            "v42_deterministic_judge": args.v42_deterministic_judge,
            "v42_combined_inventory": args.v42_combined_inventory,
            "v42_combined_coverage_audit": (
                args.v42_combined_coverage_audit
            ),
            "v42_eager_inventory": args.v42_eager_inventory,
            "v42_text_crosscheck": args.v42_text_crosscheck,
            "v42_decompose_broad_text": args.v42_decompose_broad_text,
            "v42_coverage_only": args.v42_coverage_only,
            "v42_single_artifact_proposer": (
                args.v42_single_artifact_proposer
            ),
            "v42_review_original_max_side": (
                args.v42_review_original_max_side
            ),
            "v42_consolidation_image_max_side": (
                args.v42_consolidation_image_max_side
            ),
            "v42_inventory_overview_max_side": (
                args.v42_inventory_overview_max_side
            ),
            "v42_deterministic_consolidation": (
                args.v42_deterministic_consolidation
            ),
            "v42_deterministic_source_consolidation": (
                args.v42_deterministic_source_consolidation
            ),
            "bbox_coordinate_space": "bbox_1000",
            "consolidation_enabled": not args.skip_consolidation,
        },
    )
    results_path = args.output_root / "results.jsonl"
    errors_path = args.output_root / "errors.jsonl"
    trace_writer = TraceWriter(args.output_root / "traces.jsonl")
    completed = completed_units(results_path)
    limiter = asyncio.Semaphore(args.max_in_flight)
    pending = [
        (record, workflow)
        for record in records
        for workflow in args.workflows
        if (record.image_id, workflow) not in completed
    ]
    total = len(records) * len(args.workflows)
    done_before = total - len(pending)
    started = time.monotonic()
    completed_now = 0

    async def run_unit_once(
        record,
        workflow: str,
        *,
        unit_attempt: int,
    ) -> dict[str, Any]:
        unit_started = time.monotonic()
        model = OpenAICompatibleModelClient(
            base_url=args.base_url,
            model=args.model,
            sample_id=record.image_id,
            workflow=workflow,
            trace_writer=trace_writer,
            limiter=limiter,
            api_key=args.api_key,
            timeout=args.timeout,
            max_tokens=args.max_tokens,
            max_retries=args.retries,
            retry_structured_output=structured_recovery_enabled(args),
        )
        critic = build_critics(
            model,
            PillowImageTool(),
            coverage_mode=args.coverage_mode,
            v42_text_batch_size=args.v42_text_batch_size,
            v42_text_crosscheck_batch_size=(
                args.v42_text_crosscheck_batch_size
            ),
            v42_artifact_batch_size=args.v42_artifact_batch_size,
            v42_inspection_concurrency=args.v42_inspection_concurrency,
            v42_challenge_batch_size=args.v42_challenge_batch_size,
            v42_challenge_concurrency=(
                args.v42_challenge_concurrency
            ),
            v42_trust_preverified=args.v42_trust_preverified,
            v42_deterministic_judge=args.v42_deterministic_judge,
            v42_combined_inventory=args.v42_combined_inventory,
            v42_combined_coverage_audit=(
                args.v42_combined_coverage_audit
            ),
            v42_eager_inventory=args.v42_eager_inventory,
            v42_text_crosscheck=args.v42_text_crosscheck,
            v42_decompose_broad_text=args.v42_decompose_broad_text,
            v42_coverage_only=args.v42_coverage_only,
            v42_single_artifact_proposer=(
                args.v42_single_artifact_proposer
            ),
            v42_review_original_max_side=(
                args.v42_review_original_max_side
            ),
            v42_inventory_overview_max_side=(
                args.v42_inventory_overview_max_side
            ),
            v61_streaming=v61_enabled(args),
            v62_adaptive=structured_recovery_enabled(args),
            v63_selective_review=args.v63_selective_review,
            v63_entity_safe_review=args.v63_entity_safe_review,
            v63_artifact_scout=args.v63_artifact_scout,
            v63_dual_artifact_scout=args.v63_dual_artifact_scout,
            v63_dense_path_latency=args.v63_dense_path_latency,
            v42_pipeline_version=(
                "v6.3-i"
                if args.v63_dense_path_latency
                else "v6.3"
                if v63_enabled(args)
                else "v6.2"
                if v62_enabled(args)
                else "v6.1"
                if v61_enabled(args)
                else "v4.1-flash"
                if v41_flash_enabled(args)
                else "v4.2.2-scheduling-only"
                if args.v422_low_latency
                else "v4.2.1-quality-first"
                if args.v42_quality_first
                else "v4.2-experimental"
            ),
        )[workflow]
        image = to_image_asset(record)
        critic_started = time.monotonic()
        result = await critic.critique(
            image,
            critic_context(),
        )
        critic_elapsed = time.monotonic() - critic_started
        result_dict = result.as_dict()
        consolidation_started = time.monotonic()
        consolidation_image = image
        if (
            args.coverage_mode == "harness_v42"
            and args.v42_consolidation_image_max_side > 0
        ):
            consolidation_image = await overview_image(
                image,
                max_side=args.v42_consolidation_image_max_side,
            )
        consolidated = await consolidate_with_retries(
            workflow=workflow,
            image=consolidation_image,
            result=result_dict,
            consolidator=ResultConsolidator(
                model,
                reject_nested_containers=(
                    args.coverage_mode
                    in {
                        "harness",
                        "harness_v41",
                        "harness_v42",
                        "harness_v5",
                    }
                ),
                enforce_materiality_gate=(
                    args.coverage_mode
                    in {"harness_v41", "harness_v42", "harness_v5"}
                ),
                deterministic_independent_fast_path=(
                    args.coverage_mode == "harness_v42"
                    and args.v42_deterministic_consolidation
                    and not args.v42_quality_first
                ),
                deterministic_nested_text_pruning=(
                    args.coverage_mode == "harness_v42"
                    and args.v42_deterministic_consolidation
                ),
                deterministic_preconsolidated_fast_path=(
                    args.coverage_mode == "harness_v42"
                    and not v63_enabled(args)
                    and (
                        args.v42_coverage_only
                        or args.v42_deterministic_source_consolidation
                    )
                    and args.v42_deterministic_consolidation
                ),
                deterministic_issue_hydration=(
                    args.coverage_mode == "harness_v42"
                    and flash_optimizations_enabled(args)
                ),
                component_repair=(
                    component_repair_enabled(args)
                ),
                online_source_atomization=bool(
                    args.v63_dense_path_latency
                ),
                max_validation_repairs=(
                    3 if args.v42_quality_first else 1
                ),
                max_tokens_override=(
                    args.consolidation_max_tokens or None
                ),
            ),
            skip=args.skip_consolidation,
            max_retries=args.consolidation_retries,
        )
        consolidation_elapsed = time.monotonic() - consolidation_started
        row = {
            "sample_id": record.image_id,
            "workflow": workflow,
            "status": "ok",
            "result": result_dict,
            "model": args.model,
            "timing": {
                "unit_attempt": unit_attempt,
                "unit_elapsed_seconds": round(
                    time.monotonic() - unit_started,
                    4,
                ),
                "critic_elapsed_seconds": round(critic_elapsed, 4),
                "consolidation_elapsed_seconds": round(
                    consolidation_elapsed,
                    4,
                ),
            },
        }
        if consolidated is not None:
            row["consolidated_result"] = consolidated
        return row

    async def run_unit(record, workflow: str) -> dict[str, Any]:
        last_error: Exception | None = None
        attempt_elapsed_seconds: list[float] = []
        for unit_attempt in range(1, args.unit_retries + 2):
            attempt_started = time.monotonic()
            try:
                row = await run_unit_once(
                    record,
                    workflow,
                    unit_attempt=unit_attempt,
                )
                attempt_elapsed = time.monotonic() - attempt_started
                attempt_elapsed_seconds.append(attempt_elapsed)
                attach_cumulative_unit_timing(
                    row,
                    attempt_elapsed_seconds,
                )
                trace_writer.append(
                    {
                        "sample_id": record.image_id,
                        "workflow": workflow,
                        "stage": "pipeline.unit",
                        "attempt": unit_attempt,
                        "status": "ok",
                        "recorded_at_unix": time.time(),
                        "unit_elapsed_seconds": round(
                            attempt_elapsed,
                            4,
                        ),
                        "timing": row.get("timing", {}),
                    }
                )
                return row
            except Exception as exc:
                last_error = exc
                attempt_elapsed = time.monotonic() - attempt_started
                attempt_elapsed_seconds.append(attempt_elapsed)
                trace_writer.append(
                    {
                        "sample_id": record.image_id,
                        "workflow": workflow,
                        "stage": "pipeline.unit",
                        "attempt": unit_attempt,
                        "status": "error",
                        "recorded_at_unix": time.time(),
                        "unit_elapsed_seconds": round(
                            attempt_elapsed,
                            4,
                        ),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
        assert last_error is not None
        raise last_error

    async for record, workflow, outcome in iter_completed_units(
        pending,
        run_unit,
        max_units_in_flight=args.max_images_in_flight,
    ):
        completed_now += 1
        if isinstance(outcome, Exception):
            append_jsonl(
                errors_path,
                {
                    "sample_id": record.image_id,
                    "workflow": workflow,
                    "status": "error",
                    "error_type": type(outcome).__name__,
                    "error": str(outcome),
                    "model": args.model,
                },
            )
            status = "error"
        else:
            append_jsonl(results_path, outcome)
            status = "ok"
        elapsed = time.monotonic() - started
        rate = elapsed / completed_now
        remaining = max(0, len(pending) - completed_now)
        print(
            json.dumps(
                {
                    "sample_id": record.image_id,
                    "workflow": workflow,
                    "status": status,
                    "completed_units": done_before + completed_now,
                    "total_units": total,
                    "elapsed_seconds": round(elapsed, 1),
                    "eta_seconds": round(rate * remaining, 1),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )


def v63_profile_argv(argv=None) -> list[str]:
    """Apply the validated V6.3-I defaults without blocking user overrides."""

    raw = list(sys.argv[1:] if argv is None else argv)

    def has_option(name: str) -> bool:
        return any(value == name or value.startswith(f"{name}=") for value in raw)

    prefix: list[str] = []
    for flag in (
        "--v63",
        "--v63-selective-review",
        "--v63-dual-artifact-scout",
        "--v63-dense-path-latency",
    ):
        if not has_option(flag):
            prefix.append(flag)
    defaults = (
        ("--coverage-mode", "harness_v42"),
        ("--max-in-flight", "64"),
        ("--max-images-in-flight", "4"),
        ("--timeout", "900"),
        ("--max-tokens", "16000"),
        ("--consolidation-max-tokens", "16000"),
        ("--v42-text-batch-size", "2"),
        ("--v42-text-crosscheck-batch-size", "2"),
        ("--v42-artifact-batch-size", "2"),
        ("--v42-inspection-concurrency", "32"),
        ("--v42-challenge-batch-size", "2"),
        ("--v42-challenge-concurrency", "16"),
    )
    for name, value in defaults:
        if not has_option(name):
            prefix.extend((name, value))
    if not has_option("--workflow"):
        prefix.extend(("--workflow", "adversarial_consensus"))
    return prefix + raw


def main(argv=None) -> None:
    asyncio.run(run_evaluation(parse_args(v63_profile_argv(argv))))


if __name__ == "__main__":
    main()
