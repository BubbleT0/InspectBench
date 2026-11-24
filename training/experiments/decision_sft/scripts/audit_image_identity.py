#!/usr/bin/env python3
"""Detect exact image reuse across R1/R2/R3 train and validation splits.

The fast fingerprint hashes file size plus the first, middle, and last 64 KiB.
Only matching fingerprint groups are then fully SHA-256 hashed, so large images
do not all need to be read twice.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


ROOT = Path(
    "/path/to/storage/multimodal_ckpt/research/autotrain/experiments/"
    "critic_singlemodel_r123_8gpu_20260824/data/dev_4000_800"
)
CHUNK = 64 * 1024


def load_rows(split: str):
    rows = []
    with (ROOT / f"{split}.jsonl").open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            row = json.loads(line)
            target = json.loads(row["messages"][-1]["content"])
            rows.append((split, index, "clean" if target["no_error"] else "error", row["images"][0]))
    return rows


def fast_fingerprint(item):
    split, index, label, path = item
    size = os.path.getsize(path)
    digest = hashlib.sha256()
    digest.update(str(size).encode())
    with open(path, "rb") as handle:
        digest.update(handle.read(CHUNK))
        if size > CHUNK * 2:
            handle.seek(max(0, size // 2 - CHUNK // 2))
            digest.update(handle.read(CHUNK))
        if size > CHUNK:
            handle.seek(max(0, size - CHUNK))
            digest.update(handle.read(CHUNK))
    return digest.hexdigest(), item


def full_sha(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    rows = load_rows("train") + load_rows("validation")
    by_realpath = defaultdict(list)
    for item in rows:
        by_realpath[os.path.realpath(item[3])].append(item)
    repeated_paths = [group for group in by_realpath.values() if len(group) > 1]

    by_fast = defaultdict(list)
    with ThreadPoolExecutor(max_workers=16) as pool:
        for fingerprint, item in pool.map(fast_fingerprint, rows):
            by_fast[fingerprint].append(item)

    exact_groups = []
    candidates = [group for group in by_fast.values() if len(group) > 1]
    for group in candidates:
        by_sha = defaultdict(list)
        unique_paths = sorted({item[3] for item in group})
        with ThreadPoolExecutor(max_workers=min(8, len(unique_paths))) as pool:
            hashes = dict(zip(unique_paths, pool.map(full_sha, unique_paths)))
        for item in group:
            by_sha[hashes[item[3]]].append(item)
        exact_groups.extend(items for items in by_sha.values() if len(items) > 1)

    cross_split = [g for g in exact_groups if len({item[0] for item in g}) > 1]
    cross_label = [g for g in exact_groups if len({item[2] for item in g}) > 1]
    result = {
        "rows": len(rows),
        "same_realpath_groups": len(repeated_paths),
        "fast_collision_groups_checked": len(candidates),
        "exact_duplicate_groups": len(exact_groups),
        "exact_cross_split_groups": len(cross_split),
        "exact_cross_label_groups": len(cross_label),
    }
    print(json.dumps(result, indent=2))
    if cross_split or cross_label or repeated_paths:
        for group in (cross_split + cross_label + repeated_paths)[:20]:
            print(json.dumps(group, ensure_ascii=False))
        raise SystemExit("image identity audit: FAIL")
    print("image identity audit: PASS")


if __name__ == "__main__":
    main()
