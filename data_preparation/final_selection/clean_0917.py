#!/usr/bin/env python3
"""Clean the 0917 annotation against the 0807 canonical image-path set.

Rule: any 0917 record whose image path does not exist in the old (0807)
annotation set is dropped. Output files keep source-recognizable names so
vlm_judge_bench source bucketing still works.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("/path/to/storage/research/WorkSpace3/critic/benchmark/data_0804")
OLD_DIR = ROOT / "annotation/0807_composed"
NEW_DIR = ROOT / "annotation/0917"
OUT_DIR = ROOT / "new_results/annotation_0917_clean"

OLD_FILES = [
    "GPT_completed_211_20260807_090625.jsonl",
    "External4_completed_positive_511_20260807.jsonl",
    "U1.5_completed_141_20260807_090612.jsonl",
    "U1.5_completed_clean_81_20260807_090538.jsonl",
    "Natural_clean_110.jsonl",
]

# 0917 source file -> cleaned output name (must keep the lowercase patterns
# 'gpt_completed_', 'external4_completed_', 'u1.5_completed_', 'natural_clean_110'
# that metrics._annotation_source matches on)
NEW_FILES = {
    "GPT_completed_208_20260917_072651.jsonl": "GPT_completed_208_20260917_cleaned.jsonl",
    "External4_completed_552_20260910_095951.jsonl": "External4_completed_499_20260917_cleaned.jsonl",
    "U1.5_completed_141_20260910_095959.jsonl": "U1.5_completed_141_20260910_cleaned.jsonl",
    "U15_clean_completed_81_20260910_095937.jsonl": "U1.5_completed_clean_81_20260910_cleaned.jsonl",
    "Natural_clean_110.jsonl": "Natural_clean_110_cleaned.jsonl",
}


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    old_paths: set[str] = set()
    for name in OLD_FILES:
        for row in read_rows(OLD_DIR / name):
            old_paths.add(str(Path(row["image_absolute_path"]).resolve()))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    total_kept = 0
    total_dropped = 0
    for src_name, out_name in NEW_FILES.items():
        rows = read_rows(NEW_DIR / src_name)
        kept = [r for r in rows if str(Path(r["image_absolute_path"]).resolve()) in old_paths]
        dropped = len(rows) - len(kept)
        out_path = OUT_DIR / out_name
        with out_path.open("w", encoding="utf-8") as handle:
            for row in kept:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        pos = sum(1 for r in kept if r["annotation"]["boxes"])
        boxes = sum(len(r["annotation"]["boxes"]) for r in kept)
        print(f"{src_name}: kept {len(kept)} (pos {pos}, clean {len(kept)-pos}, gt {boxes}), dropped {dropped} -> {out_name}")
        total_kept += len(kept)
        total_dropped += dropped

    print(f"TOTAL kept {total_kept}, dropped {total_dropped}")


if __name__ == "__main__":
    main()
