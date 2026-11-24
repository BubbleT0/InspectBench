#!/usr/bin/env python3
"""Materialize the public InspectBench SFT datasets as model-input JSONL."""
import argparse
import json
from pathlib import Path


def rows(path):
    with path.open(encoding="utf-8") as stream:
        for raw in stream:
            if raw.strip():
                yield json.loads(raw)


def materialize(root, output, apply_repeats=False, available_only=False):
    root, output = Path(root).resolve(), Path(output).resolve()
    stats = json.loads((root / "dataset_statistics.json").read_text())
    kind = stats["dataset"]
    if output == root or root in output.parents:
        raise ValueError("Choose an output directory outside the downloaded dataset.")
    if kind == "decision_sft":
        if apply_repeats:
            raise ValueError("--apply-repeats applies only to Workflow-SFT.")
        missing = sum(s["missing_images"] for s in stats["splits"].values())
        if missing and not available_only:
            raise ValueError(
                f"{missing} original Decision-SFT images are unavailable. "
                "Use --available-only to explicitly select the incomplete subset. "
                "This does not reproduce the original training mixture."
            )
        jobs = {split: [(root / "data" / (split + ".jsonl"), 1)]
                for split in stats["splits"]}
    elif kind == "workflow_sft":
        if available_only:
            raise ValueError("--available-only applies only to Decision-SFT.")
        sampling = json.loads((root / "sampling.json").read_text())
        jobs = {"train": [(root / shard["file"], shard["repeat_count"] if apply_repeats else 1)
                          for shard in sampling["shards"]]}
    else:
        raise ValueError("Unknown dataset: " + kind)
    output.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split, shards in jobs.items():
        target = output / (split + ".jsonl")
        temporary = target.with_suffix(".jsonl.tmp")
        count = 0
        try:
            with temporary.open("w", encoding="utf-8") as out:
                for source, repeat in shards:
                    for _ in range(repeat):
                        for row in rows(source):
                            record = {"messages": row["messages"]}
                            if kind == "decision_sft":
                                if not row["image_available"]:
                                    continue
                                if len(row["images"]) != 1:
                                    raise ValueError("Expected one image: " + row["sample_id"])
                                image = (root / row["images"][0]).resolve()
                                if root not in image.parents or not image.is_file():
                                    raise ValueError("Missing/invalid local image: " + row["sample_id"])
                                record["images"] = [str(image)]
                            out.write(json.dumps(record, ensure_ascii=False) + "\n")
                            count += 1
            expected = (stats["splits"][split]["available_images"] if kind == "decision_sft"
                        else stats["effective_train_records" if apply_repeats else "train_records"])
            if count != expected:
                raise ValueError(f"{split}: expected {expected} rows, found {count}")
            temporary.replace(target)
        finally:
            if temporary.exists():
                temporary.unlink()
        counts[split] = count
    return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--apply-repeats", action="store_true",
                        help="Workflow: expand 8,834 records to the saved 12,312-record mixture")
    parser.add_argument("--available-only", action="store_true",
                        help="Decision: explicitly accept the incomplete 3,000/600 subset")
    args = parser.parse_args()
    try:
        result = materialize(args.dataset_dir, args.output_dir, args.apply_repeats, args.available_only)
    except ValueError as error:
        parser.exit(2, str(error) + "\n")
    print(json.dumps(result, indent=2))
