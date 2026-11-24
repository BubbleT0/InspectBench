#!/usr/bin/env python3
"""Build the packing repeat map from the converted Swift meta (train only)."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()

    with Path(args.meta).open(encoding="utf-8") as f:
        meta = json.load(f)

    rows = []
    total_original = 0
    total_effective = 0
    for name, config in sorted(meta.items()):
        if not name.endswith("_train"):
            continue
        path = Path(config["annotation"])
        if path.stem != name:
            raise ValueError(f"shard basename mismatch: {name} != {path.stem}")
        length = int(config["length"])
        actual = sum(1 for line in path.open(encoding="utf-8") if line.strip())
        if actual != length:
            raise ValueError(f"{name}: meta length={length}, actual={actual}")
        repeat = float(config.get("repeat_time", 1.0))
        effective = max(1, int(round(length * repeat)))
        rows.append((name, str(path), effective, length, repeat))
        total_original += length
        total_effective += effective

    if len(rows) != 4:
        raise ValueError(f"expected exactly four train shards, got {len(rows)}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(("shard_name", "jsonl_path", "dataset_sample_N", "orig_length", "repeat_time"))
        writer.writerows(rows)

    summary = {
        "meta": args.meta,
        "repeat_map": str(output),
        "train_shards": len(rows),
        "original_train_records": total_original,
        "effective_train_records": total_effective,
        "entries": [
            {"shard_name": r[0], "jsonl_path": r[1], "dataset_sample_N": r[2], "orig_length": r[3], "repeat_time": r[4]}
            for r in rows
        ],
    }
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
