from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any, Mapping

from critic_workflows import ImageAsset

from .consolidation import ResultConsolidator
from .model_adapter import OpenAICompatibleModelClient, TraceWriter
from .run import append_jsonl


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text("utf-8").splitlines()
        if line.strip()
    ]


def completed_samples(path: Path) -> set[str]:
    return {
        str(row["sample_id"])
        for row in _jsonl(path)
        if row.get("status") == "ok"
    }


def reject_nested_containers(output_root: Path) -> bool:
    config_path = output_root / "run_config.json"
    if not config_path.exists():
        return False
    config = json.loads(config_path.read_text("utf-8"))
    return str(config.get("coverage_mode", "")) == "harness"


async def _tagged(sample_id: str, operation):
    try:
        return sample_id, await operation, None
    except Exception as exc:
        return sample_id, None, exc


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--model",
        default="Qwen3.6-27B-agen-0722-agentv4",
    )
    parser.add_argument("--api-key-env", default="CRITIC_API_KEY")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--sample-id")
    selection.add_argument("--limit-images", type=int)
    selection.add_argument("--all", action="store_true")
    parser.add_argument("--confirm-run-all", default="")
    parser.add_argument("--max-in-flight", type=int, default=6)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--max-tokens", type=int, default=8000)
    parser.add_argument("--retries", type=int, default=2)
    args = parser.parse_args(argv)
    if args.all and args.confirm_run_all != "YES":
        parser.error("--all requires --confirm-run-all YES")
    if args.limit_images is not None and args.limit_images <= 0:
        parser.error("--limit-images must be positive")
    if args.max_in_flight <= 0:
        parser.error("--max-in-flight must be positive")
    args.api_key = os.environ.get(args.api_key_env)
    return args


async def consolidate_existing(args) -> None:
    config_path = args.output_root / "run_config.json"
    config = (
        json.loads(config_path.read_text("utf-8"))
        if config_path.exists()
        else {}
    )
    if str(config.get("pipeline_version", "")) == "v4.1":
        raise RuntimeError(
            "V4.1 forbids offline consolidation; rerun the incomplete "
            "pipeline unit instead."
        )
    manifest = json.loads(
        (args.output_root / "manifest.json").read_text("utf-8")
    )
    images = {
        str(row["image_id"]): row
        for row in manifest.get("images", [])
    }
    results = {
        str(row["sample_id"]): row
        for row in _jsonl(args.output_root / "results.jsonl")
        if row.get("status") == "ok"
        and row.get("workflow") == "adversarial_consensus"
    }
    sample_ids = sorted(set(images) & set(results))
    if args.sample_id:
        if args.sample_id not in sample_ids:
            raise ValueError(
                f"Sample {args.sample_id!r} has no image and successful result."
            )
        sample_ids = [args.sample_id]
    elif args.limit_images is not None:
        sample_ids = sample_ids[: args.limit_images]

    output_path = args.output_root / "consolidated_results.jsonl"
    errors_path = args.output_root / "consolidation_errors.jsonl"
    completed = completed_samples(output_path)
    pending = [
        sample_id
        for sample_id in sample_ids
        if sample_id not in completed
    ]
    limiter = asyncio.Semaphore(args.max_in_flight)
    trace_writer = TraceWriter(args.output_root / "traces.jsonl")
    reject_nested = reject_nested_containers(args.output_root)
    started = time.monotonic()

    async def run_one(sample_id: str) -> dict[str, Any]:
        image_row = images[sample_id]
        image = ImageAsset(
            id=sample_id,
            data=Path(image_row["source_path"]).read_bytes(),
            mime_type=str(image_row.get("mime_type", "image/jpeg")),
            metadata={},
        )
        model = OpenAICompatibleModelClient(
            base_url=args.base_url,
            model=args.model,
            sample_id=sample_id,
            workflow="adversarial_consensus",
            trace_writer=trace_writer,
            limiter=limiter,
            api_key=args.api_key,
            timeout=args.timeout,
            max_tokens=args.max_tokens,
            max_retries=args.retries,
        )
        consolidated = await ResultConsolidator(
            model,
            reject_nested_containers=reject_nested,
        ).consolidate(image, results[sample_id]["result"])
        return {
            "sample_id": sample_id,
            "workflow": "adversarial_consensus",
            "status": "ok",
            "consolidated_result": consolidated,
            "model": args.model,
        }

    tasks = [
        asyncio.create_task(_tagged(sample_id, run_one(sample_id)))
        for sample_id in pending
    ]
    completed_now = 0
    failures = 0
    for task in asyncio.as_completed(tasks):
        sample_id, row, error = await task
        completed_now += 1
        if error is not None:
            failures += 1
            append_jsonl(
                errors_path,
                {
                    "sample_id": sample_id,
                    "workflow": "adversarial_consensus",
                    "status": "error",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "model": args.model,
                },
            )
            status = "error"
        else:
            append_jsonl(output_path, row)
            status = "ok"
        elapsed = time.monotonic() - started
        rate = elapsed / completed_now
        remaining = len(pending) - completed_now
        print(
            json.dumps(
                {
                    "sample_id": sample_id,
                    "status": status,
                    "completed": len(completed) + completed_now,
                    "total": len(sample_ids),
                    "failures": failures,
                    "elapsed_seconds": round(elapsed, 1),
                    "eta_seconds": round(rate * remaining, 1),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )


def main(argv=None) -> None:
    asyncio.run(consolidate_existing(parse_args(argv)))


if __name__ == "__main__":
    main()
