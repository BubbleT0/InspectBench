#!/usr/bin/env python3
"""Build agent-data-tools dict meta from structured_qc cleaned shards."""

import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--structured-meta", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    entries = json.loads(Path(args.structured_meta).read_text(encoding="utf-8"))
    if not isinstance(entries, list) or not entries:
        raise ValueError("structured meta must be a non-empty list")
    meta = {}
    for entry in entries:
        path = str(entry["path"])
        basename = Path(path).stem
        if basename.endswith("_train"):
            split = "train"
        elif basename.endswith("_validation"):
            split = "validation"
        elif basename.endswith("_test"):
            split = "test"
        else:
            raise ValueError(f"cannot infer split: {basename}")
        label = "positive" if basename.startswith("positive_") else "negative"
        name = basename
        meta[name] = {
            "annotation": path,
            "author": "research",
            "class": "critic",
            "comment": f"critic RP1 stratified-validation SFT; split={split}; label={label}; semantic QC waived by prior user decision",
            "enable_thinking": True,
            "expect_name": name,
            "from": "critic_rp1_stratval_4src_16gpu_neo1_trajectory_20260820",
            "length": int(entry["length"]),
            "repeat_time": 1.0,
            "team": "research",
            "tools": [],
        }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    tmp.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, output)
    print(json.dumps({"datasets": len(meta), "records": sum(x["length"] for x in meta.values()), "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
