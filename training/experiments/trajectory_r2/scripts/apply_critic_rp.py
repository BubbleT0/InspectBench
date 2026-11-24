#!/usr/bin/env python3
"""Freeze the user-approved repeat policy into a new stage-4 metadata file."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def split_of(name: str) -> str:
    for split in ("train", "validation", "test"):
        if name.endswith("_" + split):
            return split
    raise ValueError(f"dataset name has no recognized split suffix: {name}")


def label_of(name: str) -> str:
    if name.startswith("negative_"):
        return "negative"
    if name.startswith("positive_"):
        return "positive"
    raise ValueError(f"dataset name has no recognized label prefix: {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-meta", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--negative-train-repeat", type=float, default=2.0)
    args = parser.parse_args()

    input_meta_path = Path(args.input_meta)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with input_meta_path.open(encoding="utf-8") as f:
        meta = json.load(f)
    if not isinstance(meta, dict):
        raise TypeError("meta must be a dataset-name keyed object")

    rows = []
    totals = {
        "train": {"positive": 0, "negative": 0, "effective_positive": 0, "effective_negative": 0},
        "validation": {"positive": 0, "negative": 0, "effective_positive": 0, "effective_negative": 0},
        "test": {"positive": 0, "negative": 0, "effective_positive": 0, "effective_negative": 0},
    }
    for name, config in sorted(meta.items()):
        split = split_of(name)
        label = label_of(name)
        repeat = args.negative_train_repeat if split == "train" and label == "negative" else 1.0
        length = int(config["length"])
        effective = max(1, int(round(length * repeat)))
        config["repeat_time"] = repeat
        totals[split][label] += length
        totals[split][f"effective_{label}"] += effective
        rows.append((name, config["annotation"], effective, length, repeat, split, label))

    output_meta = output_dir / "meta_middle_format.json"
    with output_meta.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")

    repeat_map = output_dir / "repeat_map_openai.tsv"
    with repeat_map.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(("shard_name", "jsonl_path", "dataset_sample_N", "orig_length", "repeat_time", "split", "label"))
        writer.writerows(rows)

    decision = {
        "status": "approved_by_user",
        "policy": {
            "negative_train_repeat_time": args.negative_train_repeat,
            "positive_train_repeat_time": 1.0,
            "validation_repeat_time": 1.0,
            "test_repeat_time": 1.0,
        },
        "input_meta": str(input_meta_path),
        "output_meta": str(output_meta),
        "repeat_map": str(repeat_map),
        "datasets": len(meta),
        "totals": totals,
    }
    with (output_dir / "rp_decision.json").open("w", encoding="utf-8") as f:
        json.dump(decision, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
