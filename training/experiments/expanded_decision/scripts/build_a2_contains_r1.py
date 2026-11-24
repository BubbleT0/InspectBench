#!/usr/bin/env python3
"""Build A2 as a minimal, source-matched modification of A that contains R1."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


SEED = 20260824
R1_DIR = Path(
    "/path/to/storage/multimodal_ckpt/research/autotrain/experiments/"
    "critic_singlemodel_r123_8gpu_20260824/data/dev_4000_800"
)
ABC_DATA = Path(
    "/path/to/storage/multimodal_ckpt/research/autotrain/experiments/"
    "critic_r1_full_ratio_abc_16gpu_20260826/data"
)
A_DIR = ABC_DATA / "a_balanced"
OUT_DIR = ABC_DATA / "a2_balanced_contains_r1"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def image_of(row: dict[str, Any]) -> str:
    return str(row["images"][0])


def label_of(row: dict[str, Any]) -> str:
    target = json.loads(row["messages"][1]["content"])
    return "no_error" if target.get("no_error") is True else "error"


def load_aligned(directory: Path, split: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = read_jsonl(directory / f"{split}.jsonl")
    audit = [x for x in read_jsonl(directory / "audit_manifest.jsonl") if x["split"] == split]
    audit.sort(key=lambda x: int(x["ordinal"]))
    if len(rows) != len(audit):
        raise ValueError(f"{directory}:{split}: row/audit mismatch")
    for index, (row, meta) in enumerate(zip(rows, audit)):
        if image_of(row) != str(meta["image_path"]):
            raise ValueError(f"{directory}:{split}:{index}: image/audit mismatch")
    return rows, audit


def stable_rank(image: str) -> str:
    return hashlib.sha256(f"{SEED}|a2-remove|{image}".encode()).hexdigest()


def validate(rows: list[dict[str, Any]], audit: list[dict[str, Any]]) -> dict[str, Any]:
    if len(rows) != 13026 or len(audit) != 13026:
        raise ValueError("A2 must have exactly 13026 training rows")
    images: set[str] = set()
    labels: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    for index, (row, meta) in enumerate(zip(rows, audit)):
        if set(row) != {"messages", "images"}:
            raise ValueError(f"row {index}: unexpected model-visible columns")
        image = image_of(row)
        if image in images:
            raise ValueError(f"row {index}: duplicate image {image}")
        images.add(image)
        label = label_of(row)
        if label != meta["label"] or image != meta["image_path"]:
            raise ValueError(f"row {index}: row/audit label or image mismatch")
        if int(meta["ordinal"]) != index or meta["split"] != "train":
            raise ValueError(f"row {index}: invalid audit ordinal/split")
        labels[label] += 1
        sources[str(meta["source"])] += 1
    expected_labels = Counter({"no_error": 6513, "error": 6513})
    expected_sources = Counter(
        {
            "clean_pinterest": 3257,
            "clean_wikimedia": 3256,
            "error_v41_flash": 6413,
            "error_v41_hp": 100,
        }
    )
    if labels != expected_labels or sources != expected_sources:
        raise ValueError(f"unexpected labels/sources: {labels}, {sources}")
    return {"labels": dict(labels), "sources": dict(sources), "unique_images": len(images)}


def main() -> None:
    if OUT_DIR.exists():
        raise FileExistsError(f"refusing to overwrite existing A2 directory: {OUT_DIR}")
    OUT_DIR.mkdir(parents=True)

    a_rows, a_audit = load_aligned(A_DIR, "train")
    r1_rows, r1_audit = load_aligned(R1_DIR, "train")
    validation_rows, validation_audit = load_aligned(A_DIR, "validation")

    a_by_image = {image_of(row): (row, meta) for row, meta in zip(a_rows, a_audit)}
    r1_by_image = {image_of(row): (row, meta) for row, meta in zip(r1_rows, r1_audit)}
    missing = set(r1_by_image) - set(a_by_image)
    missing_by_source: dict[str, list[str]] = {}
    for image in missing:
        row, meta = r1_by_image[image]
        if label_of(row) != "no_error":
            raise ValueError(f"A unexpectedly misses an R1 error row: {image}")
        missing_by_source.setdefault(str(meta["source"]), []).append(image)

    expected_missing = {"clean_pinterest": 509, "clean_wikimedia": 485}
    if {k: len(v) for k, v in missing_by_source.items()} != expected_missing:
        raise ValueError(f"R1/A overlap drift: { {k: len(v) for k, v in missing_by_source.items()} }")

    remove_images: set[str] = set()
    for source, needed in expected_missing.items():
        candidates = [
            image_of(row)
            for row, meta in zip(a_rows, a_audit)
            if meta["label"] == "no_error"
            and meta["source"] == source
            and image_of(row) not in r1_by_image
        ]
        candidates.sort(key=stable_rank)
        if len(candidates) < needed:
            raise ValueError(f"not enough replaceable A rows for {source}")
        remove_images.update(candidates[:needed])

    replacements_by_source = {
        source: sorted(images, key=lambda x: int(r1_by_image[x][1]["ordinal"]))
        for source, images in missing_by_source.items()
    }
    replacement_cursor = Counter()
    new_rows: list[dict[str, Any]] = []
    new_audit: list[dict[str, Any]] = []
    for index, (row, meta) in enumerate(zip(a_rows, a_audit)):
        if image_of(row) in remove_images:
            source = str(meta["source"])
            replacement_image = replacements_by_source[source][replacement_cursor[source]]
            replacement_cursor[source] += 1
            row, meta = r1_by_image[replacement_image]
        new_rows.append(dict(row))
        updated_meta = dict(meta)
        updated_meta["split"] = "train"
        updated_meta["ordinal"] = index
        new_audit.append(updated_meta)

    stats = validate(new_rows, new_audit)
    new_by_image = {image_of(row): row for row in new_rows}
    r1_exact = sum(new_by_image.get(image) == row for image, (row, _) in r1_by_image.items())
    if set(r1_by_image) - set(new_by_image) or r1_exact != 4000:
        raise ValueError("A2 does not exactly contain every R1 row")
    validation_images = {image_of(row) for row in validation_rows}
    if set(new_by_image) & validation_images:
        raise ValueError("A2 train/validation overlap")

    atomic_write_jsonl(OUT_DIR / "train.jsonl", new_rows)
    atomic_write_jsonl(OUT_DIR / "validation.jsonl", validation_rows)
    full_audit = new_audit + [dict(row) for row in validation_audit]
    atomic_write_jsonl(OUT_DIR / "audit_manifest.jsonl", full_audit)
    shutil.copyfile(A_DIR / "neutral_prompt.txt", OUT_DIR / "neutral_prompt.txt")

    a_images = set(a_by_image)
    summary = {
        "seed": SEED,
        "variant": "a2_balanced_contains_r1",
        "construction": "minimal source-matched clean-image replacement from A",
        "train": {"rows": len(new_rows), **stats},
        "validation": {
            "rows": len(validation_rows),
            "labels": {"no_error": 400, "error": 400},
            "source_file": str(R1_DIR / "validation.jsonl"),
        },
        "containment": {
            "r1_rows": len(r1_rows),
            "r1_unique_images": len(r1_by_image),
            "r1_images_in_a2": len(set(r1_by_image) & set(new_by_image)),
            "r1_exact_rows_in_a2": r1_exact,
            "r1_missing_from_a2": 0,
            "a_rows_retained": len(a_images & set(new_by_image)),
            "a_rows_replaced": len(a_images - set(new_by_image)),
            "replacement_by_source": expected_missing,
        },
        "train_validation_image_overlap": 0,
        "model_visible_columns": ["messages", "images"],
        "repeat_time": 1.0,
        "qc": False,
    }
    atomic_write_json(OUT_DIR / "summary.json", summary)
    atomic_write_json(OUT_DIR / "containment_audit.json", summary["containment"])
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
