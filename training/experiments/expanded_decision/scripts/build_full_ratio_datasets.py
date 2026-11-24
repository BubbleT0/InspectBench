#!/usr/bin/env python3
"""Build frozen A/B/C full-scale ratio datasets for the R1 critic ablation."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
from pathlib import Path
from typing import Any, Iterable


SEED = 20260824
SOURCE_EXP = Path(
    "/path/to/storage/multimodal_ckpt/research/autotrain/experiments/"
    "critic_singlemodel_r123_8gpu_20260824"
)
SOURCE_BUILDER = SOURCE_EXP / "scripts/build_dev_dataset.py"
SOURCE_DATA = SOURCE_EXP / "data/dev_4000_800"


def load_source_module():
    spec = importlib.util.spec_from_file_location("r123_source_builder", SOURCE_BUILDER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {SOURCE_BUILDER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def atomic_write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def shuffled(items: list[dict[str, Any]], salt: int) -> list[dict[str, Any]]:
    result = list(items)
    random.Random(SEED + salt).shuffle(result)
    return result


def spread_classes(
    clean: list[dict[str, Any]], error: list[dict[str, Any]], salt: int
) -> list[dict[str, Any]]:
    """Evenly spread both labels through a deterministic one-epoch stream."""
    clean = shuffled(clean, salt)
    error = shuffled(error, salt + 1)
    keyed = [((index + 0.5) / len(clean), 0, row) for index, row in enumerate(clean)]
    keyed += [((index + 0.5) / len(error), 1, row) for index, row in enumerate(error)]
    keyed.sort(key=lambda item: (item[0], item[1]))
    return [item[2] for item in keyed]


def validate_rows(
    rows: list[dict[str, Any]], expected_clean: int, expected_error: int, name: str
) -> None:
    if len(rows) != expected_clean + expected_error:
        raise ValueError(f"{name}: unexpected row count {len(rows)}")
    labels = {"no_error": 0, "error": 0}
    images = set()
    for index, row in enumerate(rows):
        if set(row) != {"messages", "images"}:
            raise ValueError(f"{name}:{index}: model-visible columns {set(row)}")
        target = json.loads(row["messages"][1]["content"])
        label = "no_error" if target.get("no_error") is True else "error"
        labels[label] += 1
        if (label == "no_error") != (target.get("boxes") == []):
            raise ValueError(f"{name}:{index}: target contradiction")
        image = row["images"][0]
        if image in images:
            raise ValueError(f"{name}:{index}: duplicate image path {image}")
        images.add(image)
        prompt = str(row["messages"][0]["content"]).lower()
        for marker in (
            "generation_prompt",
            "source_name",
            "source_model",
            "base_prediction",
            "reference generation instruction",
        ):
            if marker in prompt:
                raise ValueError(f"{name}:{index}: forbidden marker {marker}")
    expected = {"no_error": expected_clean, "error": expected_error}
    if labels != expected:
        raise ValueError(f"{name}: labels {labels} != {expected}")


def source_counts(items: list[dict[str, Any]]) -> dict[str, int]:
    result: dict[str, int] = {}
    for item in items:
        source = str(item["source"])
        result[source] = result.get(source, 0) + 1
    return result


def target_character_stats(items: list[dict[str, Any]]) -> dict[str, Any]:
    values: dict[str, list[int]] = {"no_error": [], "error": []}
    for item in items:
        target = json.dumps(item["target"], ensure_ascii=False, separators=(",", ":"))
        values[str(item["label"])].append(len(target))
    return {
        label: {
            "samples": len(lengths),
            "characters": sum(lengths),
            "mean_characters": round(sum(lengths) / len(lengths), 3) if lengths else None,
        }
        for label, lengths in values.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    source = load_source_module()

    validation_rows = read_jsonl(SOURCE_DATA / "validation.jsonl")
    source_audit = read_jsonl(SOURCE_DATA / "audit_manifest.jsonl")
    validation_audit = [row for row in source_audit if row["split"] == "validation"]
    validation_audit.sort(key=lambda row: int(row["ordinal"]))
    validation_images = {row["images"][0] for row in validation_rows}
    if len(validation_rows) != 800 or len(validation_images) != 800:
        raise ValueError("frozen validation is not exactly 800 unique images")

    pools = {
        "clean_wikimedia": source.load_candidates(
            source.CLEAN_WIKIMEDIA, "clean_wikimedia", "no_error"
        ),
        "clean_pinterest": source.load_candidates(
            source.CLEAN_PINTEREST, "clean_pinterest", "no_error"
        ),
        "error_v41_hp": source.load_candidates(
            source.ERROR_V41_HP, "error_v41_hp", "error"
        ),
        "error_v41_flash": source.load_candidates(
            source.ERROR_FLASH, "error_v41_flash", "error"
        ),
    }
    for key, items in pools.items():
        pools[key] = [item for item in items if item["image_path"] not in validation_images]

    expected_pool_sizes = {
        "clean_wikimedia": 6877,
        "clean_pinterest": 7111,
        "error_v41_hp": 100,
        "error_v41_flash": 6413,
    }
    actual_pool_sizes = {key: len(value) for key, value in pools.items()}
    if actual_pool_sizes != expected_pool_sizes:
        raise ValueError(f"eligible pool drift: {actual_pool_sizes} != {expected_pool_sizes}")

    cw = shuffled(pools["clean_wikimedia"], 10)
    cp = shuffled(pools["clean_pinterest"], 20)
    hp = shuffled(pools["error_v41_hp"], 30)
    flash = shuffled(pools["error_v41_flash"], 40)

    variants = {
        "a_balanced": {
            "clean": cw[:3256] + cp[:3257],
            "error": hp + flash,
            "expected": (6513, 6513),
        },
        "b_natural_ratio_controlled": {
            "clean": cw[:4444] + cp[:4444],
            "error": hp[:64] + flash[:4074],
            "expected": (8888, 4138),
        },
        "c_all_unique": {
            "clean": cw + cp,
            "error": hp + flash,
            "expected": (13988, 6513),
        },
    }

    global_summary: dict[str, Any] = {
        "seed": SEED,
        "frozen_validation": str(SOURCE_DATA / "validation.jsonl"),
        "eligible_after_validation": actual_pool_sizes,
        "variants": {},
    }
    for index, (name, config) in enumerate(variants.items()):
        clean = config["clean"]
        error = config["error"]
        expected_clean, expected_error = config["expected"]
        candidates = spread_classes(clean, error, 100 + index * 10)
        train_rows = [source.model_row(item) for item in candidates]
        validate_rows(train_rows, expected_clean, expected_error, name)
        if {row["images"][0] for row in train_rows} & validation_images:
            raise ValueError(f"{name}: train/validation image overlap")

        variant_dir = args.output_root / name
        variant_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_jsonl(variant_dir / "train.jsonl", train_rows)
        atomic_write_jsonl(variant_dir / "validation.jsonl", validation_rows)
        audit = [source.audit_row(item, "train", i) for i, item in enumerate(candidates)]
        audit += [dict(row) for row in validation_audit]
        atomic_write_jsonl(variant_dir / "audit_manifest.jsonl", audit)
        (variant_dir / "neutral_prompt.txt").write_text(
            source.NEUTRAL_PROMPT + "\n", encoding="utf-8"
        )
        summary = {
            "seed": SEED,
            "variant": name,
            "train": {
                "rows": len(train_rows),
                "labels": {"no_error": expected_clean, "error": expected_error},
                "sources": source_counts(candidates),
                "target_character_stats": target_character_stats(candidates),
            },
            "validation": {
                "rows": 800,
                "labels": {"no_error": 400, "error": 400},
                "source_file": str(SOURCE_DATA / "validation.jsonl"),
            },
            "train_validation_image_overlap": 0,
            "model_visible_columns": ["messages", "images"],
            "repeat_time": 1.0,
            "qc": False,
        }
        atomic_write_json(variant_dir / "summary.json", summary)
        global_summary["variants"][name] = summary

    atomic_write_json(args.output_root / "summary_all.json", global_summary)
    print(json.dumps(global_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
