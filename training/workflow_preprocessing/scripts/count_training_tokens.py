#!/usr/bin/env python3
"""Count exact rendered training tokens with the target model tokenizer/template."""

import argparse
import json
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


TOKENIZER = None


def init_worker(tokenizer_path):
    global TOKENIZER
    from transformers import AutoTokenizer
    TOKENIZER = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)


def scan_dataset(task):
    dataset, path, repeat = task
    lengths = []
    errors = []
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                ids = TOKENIZER.apply_chat_template(
                    record["messages"], tokenize=True, add_generation_prompt=False
                )
                lengths.append((len(ids), line_no))
            except Exception as exc:
                errors.append({"line": line_no, "error": str(exc)})
    return dataset, path, repeat, lengths, errors


def percentile(sorted_values, pct):
    if not sorted_values:
        return 0
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = pct / 100 * (len(sorted_values) - 1)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    frac = rank - lo
    return int(round(sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac))


def stats(values):
    values = sorted(values)
    thresholds = (32768, 65536, 98304, 131072, 163840, 196608, 229376, 262144)
    return {
        "records": len(values),
        "total_tokens": sum(values),
        "min": values[0] if values else 0,
        "mean": round(sum(values) / len(values), 2) if values else 0,
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "p99_5": percentile(values, 99.5),
        "p99_9": percentile(values, 99.9),
        "max": values[-1] if values else 0,
        "over_threshold": {
            str(threshold): sum(value > threshold for value in values)
            for threshold in thresholds
        },
        "coverage": {
            str(threshold): round(
                100 * sum(value <= threshold for value in values) / len(values), 4
            ) if values else 0
            for threshold in thresholds
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    meta = json.loads(args.meta.read_text(encoding="utf-8"))
    tasks = [
        (name, item["annotation"], int(item.get("repeat_time", 1)))
        for name, item in meta.items()
    ]
    datasets = {}
    raw_lengths = []
    weighted_lengths = []
    longest = []
    errors = {}
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=init_worker,
        initargs=(args.tokenizer,),
    ) as executor:
        futures = [executor.submit(scan_dataset, task) for task in tasks]
        for future in as_completed(futures):
            dataset, path, repeat, rows, dataset_errors = future.result()
            lengths = [length for length, _ in rows]
            datasets[dataset] = {
                "annotation": path,
                "repeat_time": repeat,
                **stats(lengths),
            }
            raw_lengths.extend(lengths)
            for length in lengths:
                weighted_lengths.extend([length] * repeat)
            longest.extend(
                {"dataset": dataset, "line": line_no, "tokens": length}
                for length, line_no in rows
            )
            if dataset_errors:
                errors[dataset] = dataset_errors
            print(f"done {dataset}: {len(rows)}", flush=True)
    report = {
        "meta": str(args.meta),
        "tokenizer": args.tokenizer,
        "rendering": "tokenizer.apply_chat_template(add_generation_prompt=False)",
        "raw": stats(raw_lengths),
        "repeat_weighted": stats(weighted_lengths),
        "datasets": dict(sorted(datasets.items())),
        "longest_records": sorted(longest, key=lambda item: item["tokens"], reverse=True)[:50],
        "errors": errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "raw": report["raw"],
        "repeat_weighted": report["repeat_weighted"],
        "longest_records": report["longest_records"][:10],
        "errors": errors,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
