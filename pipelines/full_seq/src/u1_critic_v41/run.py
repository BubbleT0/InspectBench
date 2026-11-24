from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any, Mapping

from critic_workflows import PillowImageTool

from .input_manifest import (
    discover_images,
    load_images_from_jsonl,
    manifest_rows,
    to_image_asset,
)
from .consolidation import (
    ConsolidationValidationError,
    ResultConsolidator,
)
from .model_adapter import OpenAICompatibleModelClient, TraceWriter
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
    parser.add_argument(
        "--api-format",
        choices=("openai_chat", "anthropic_messages"),
        default="openai_chat",
        help="Wire protocol used by the model endpoint.",
    )
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
        choices=("grid", "adaptive", "harness", "harness_v41"),
        default="grid",
        help=(
            "Coverage strategy for adversarial consensus: V2 fixed grid "
            "V3 adaptive content regions, or V4 reliability harness."
        ),
    )
    parser.add_argument(
        "--skip-consolidation",
        action="store_true",
        help="Debug only: omit the post-Judge consolidation stage.",
    )
    args = parser.parse_args(argv)
    if args.limit_images is not None and args.limit_images <= 0:
        parser.error("--limit-images must be positive")
    if args.all and args.confirm_run_all != "YES":
        parser.error("--all requires --confirm-run-all YES")
    if args.max_in_flight <= 0:
        parser.error("--max-in-flight must be positive")
    if args.max_images_in_flight <= 0:
        parser.error("--max-images-in-flight must be positive")
    if args.consolidation_retries < 0:
        parser.error("--consolidation-retries must be non-negative")
    if args.unit_retries < 0:
        parser.error("--unit-retries must be non-negative")
    if args.coverage_mode == "harness_v41" and args.skip_consolidation:
        parser.error("V4.1 requires final consolidation")
    args.workflows = tuple(
        dict.fromkeys(args.selected_workflows or WORKFLOW_NAMES)
    )
    args.api_key = os.environ.get(args.api_key_env)
    return args


async def run_evaluation(args) -> None:
    if args.input_jsonl is not None:
        records = load_images_from_jsonl(
            args.input_jsonl,
            path_field=args.input_jsonl_path_field,
        )
    else:
        records = discover_images(args.input_root)
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
            "api_format": args.api_format,
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
            "retries": args.retries,
            "consolidation_retries": args.consolidation_retries,
            "unit_retries": args.unit_retries,
            "coverage_mode": args.coverage_mode,
            "pipeline_version": (
                "v4.1" if args.coverage_mode == "harness_v41" else None
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

    async def run_unit_once(record, workflow: str) -> dict[str, Any]:
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
            api_format=args.api_format,
        )
        critic = build_critics(
            model,
            PillowImageTool(),
            coverage_mode=args.coverage_mode,
        )[workflow]
        image = to_image_asset(record)
        result = await critic.critique(
            image,
            critic_context(),
        )
        result_dict = result.as_dict()
        consolidated = await consolidate_with_retries(
            workflow=workflow,
            image=image,
            result=result_dict,
            consolidator=ResultConsolidator(
                model,
                reject_nested_containers=(
                    args.coverage_mode in {"harness", "harness_v41"}
                ),
                enforce_materiality_gate=(
                    args.coverage_mode == "harness_v41"
                ),
            ),
            skip=args.skip_consolidation,
            max_retries=args.consolidation_retries,
        )
        row = {
            "sample_id": record.image_id,
            "workflow": workflow,
            "status": "ok",
            "result": result_dict,
            "model": args.model,
        }
        if consolidated is not None:
            row["consolidated_result"] = consolidated
        return row

    async def run_unit(record, workflow: str) -> dict[str, Any]:
        last_error: Exception | None = None
        for _ in range(args.unit_retries + 1):
            try:
                return await run_unit_once(record, workflow)
            except Exception as exc:
                last_error = exc
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


def main(argv=None) -> None:
    asyncio.run(run_evaluation(parse_args(argv)))


if __name__ == "__main__":
    main()
