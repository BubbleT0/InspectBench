#!/usr/bin/env python3
"""Build a fixed, source-auditable R1/R2/R3 critic development dataset.

The four source JSONL files and all images are read-only. Model-visible rows contain
only messages and images. Source information is written to a separate audit manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any, Iterable


SEED = 20260824

CLEAN_WIKIMEDIA = Path(
    "/path/to/storage/multimodal_data/research/images/image-noerror2/"
    "prefilter_candidates_vlm_judge_errors.jsonl"
)
CLEAN_PINTEREST = Path(
    "/path/to/storage/multimodal_data/research/images/pinterest_tianbo/raw-27b/"
    "raw_pinterest_vlm_judge_errors.jsonl"
)
ERROR_V41_HP = Path(
    "/path/to/storage/multimodal_data/research/images/image-error/output/v4.1/"
    "filtered_nontext_high_precision/images_with_errors_v4.1_high_precision_nontext.jsonl"
)
ERROR_FLASH = Path(
    "/path/to/storage/multimodal_data/research/images/image-error/output/v4.1_flash_50000/"
    "images_with_errors_v4.1_flash_50000.jsonl"
)

FAMILY_MAP = {
    "rendering_artifact": "D1_Artifact/A1_RenderingArtifact",
    "repeated_texture": "D1_Artifact/A1_RenderingArtifact",
    "structural_artifact": "D1_Artifact/A2_StructuralArtifact",
    "topology_artifact": "D1_Artifact/A3_TopologyArtifact",
    "relational_physical_artifact": "D1_Artifact/A4_RelationalAndPhysicalArtifact",
    "material_failure": "D1_Artifact/A4_RelationalAndPhysicalArtifact",
    "text_visual_integrity": "D2_TextCorrectness/A1_TextVisualIntegrity",
}

FAMILY_PRIORITY = {
    "structural_artifact": 0,
    "topology_artifact": 1,
    "relational_physical_artifact": 2,
    "rendering_artifact": 3,
    "repeated_texture": 4,
    "material_failure": 5,
    "text_visual_integrity": 6,
}

NEUTRAL_PROMPT = """<image>
Inspect the complete image for qualifying localized visual anomalies. Check local rendering corruption, malformed or missing entity parts, topology/connectivity errors, relational or physical impossibilities, and visibly malformed text glyphs. Do not report ordinary depth-of-field blur, compression, artistic style, aesthetic preference, or instruction compliance.

Both outcomes are equally valid. Do not assume an anomaly must exist. Set `no_error` from visible evidence in the image. If no well-supported anomaly is visible, return `no_error=true` and an empty `boxes` list. Otherwise return `no_error=false` and only tight, non-overlapping boxes for clearly supported anomalies.

Allowed family values:
- D1_Artifact/A1_RenderingArtifact
- D1_Artifact/A2_StructuralArtifact
- D1_Artifact/A3_TopologyArtifact
- D1_Artifact/A4_RelationalAndPhysicalArtifact
- D2_TextCorrectness/A1_TextVisualIntegrity

Coordinates use `[x_min,y_min,x_max,y_max]` on a 0..1000 image coordinate system, with 0 <= x_min < x_max <= 1000 and 0 <= y_min < y_max <= 1000.

Return exactly one JSON object with this schema:
`no_error`: boolean; `boxes`: a list. Each box has `bbox_1000_xyxy`, `family`, and a brief visible-evidence `description`. Return no analysis or additional text."""

FORBIDDEN_INPUT_MARKERS = (
    "reference generation instruction",
    "generation_prompt",
    "source_name",
    "source_model",
    "base_prediction",
    "no target text or generation instruction",
)


def iter_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if line.strip():
                yield line_no, json.loads(line)


def atomic_write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(tmp, path)


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def stable_id(source: str, identity: str) -> str:
    digest = hashlib.sha256(f"{source}\0{identity}".encode()).hexdigest()[:20]
    return f"{source}/{digest}"


def clean_image_path(row: dict[str, Any]) -> str | None:
    source = row.get("source") or {}
    if isinstance(source, dict):
        value = source.get("local_path")
        if value:
            return str(value)
        primary = source.get("primary_source") or {}
        if isinstance(primary, dict):
            metadata = primary.get("metadata") or {}
            if isinstance(metadata, dict) and metadata.get("image_path"):
                return str(metadata["image_path"])
    for key in ("image_path", "local_path"):
        if row.get(key):
            return str(row[key])
    return None


def valid_box(box: Any) -> bool:
    if not isinstance(box, list) or len(box) != 4:
        return False
    try:
        x1, y1, x2, y2 = (int(round(float(v))) for v in box)
    except (TypeError, ValueError):
        return False
    return 0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000


def region_to_target(region: dict[str, Any]) -> dict[str, Any] | None:
    box = region.get("bbox_norm")
    if not valid_box(box):
        return None
    issues = [
        issue for issue in (region.get("issues") or [])
        if issue.get("artifact_type") in FAMILY_MAP
    ]
    if not issues:
        return None
    issue = sorted(
        issues,
        key=lambda x: (
            -float(x.get("confidence") or 0),
            FAMILY_PRIORITY.get(str(x.get("artifact_type")), 99),
        ),
    )[0]
    description = str(
        issue.get("observation")
        or issue.get("abnormality")
        or region.get("edit_region_rationale")
        or region.get("edit_target")
        or "localized visible anomaly"
    )
    description = " ".join(description.split())[:300]
    return {
        "bbox_1000_xyxy": [int(round(float(v))) for v in box],
        "family": FAMILY_MAP[str(issue["artifact_type"])],
        "description": description,
    }


def make_clean_candidate(source: str, line_no: int, row: dict[str, Any]) -> dict[str, Any] | None:
    image_path = clean_image_path(row)
    if not image_path or not Path(image_path).is_file():
        return None
    identity = str((row.get("source") or {}).get("sha1") or (row.get("source") or {}).get("sha256") or image_path)
    return {
        "sample_id": stable_id(source, identity),
        "source": source,
        "source_line": line_no,
        "image_path": image_path,
        "label": "no_error",
        "target": {"no_error": True, "boxes": []},
    }


def make_error_candidate(source: str, line_no: int, row: dict[str, Any]) -> dict[str, Any] | None:
    image_path = str(row.get("image_path") or "")
    if not image_path or not Path(image_path).is_file():
        return None
    regions = row.get("v41_edit_regions") or row.get("v41_flash_edit_regions") or []
    boxes = [box for region in regions if (box := region_to_target(region)) is not None]
    if not boxes:
        return None
    identity = str(row.get("sample_id") or row.get("id") or image_path)
    return {
        "sample_id": stable_id(source, identity),
        "source": source,
        "source_line": line_no,
        "image_path": image_path,
        "label": "error",
        "target": {"no_error": False, "boxes": boxes},
    }


def load_candidates(path: Path, source: str, label: str) -> list[dict[str, Any]]:
    maker = make_clean_candidate if label == "no_error" else make_error_candidate
    candidates = []
    seen_images = set()
    for line_no, row in iter_jsonl(path):
        candidate = maker(source, line_no, row)
        if candidate is None or candidate["image_path"] in seen_images:
            continue
        seen_images.add(candidate["image_path"])
        candidates.append(candidate)
    return candidates


def take_disjoint(items: list[dict[str, Any]], n_train: int, n_val: int, rng: random.Random):
    items = list(items)
    rng.shuffle(items)
    if len(items) < n_train + n_val:
        raise ValueError(f"not enough eligible rows: have={len(items)} need={n_train + n_val}")
    return items[:n_train], items[n_train:n_train + n_val]


def interleave(clean: list[dict[str, Any]], error: list[dict[str, Any]], rng: random.Random):
    if len(clean) != len(error):
        raise ValueError(f"class counts must match: clean={len(clean)} error={len(error)}")
    clean = list(clean)
    error = list(error)
    rng.shuffle(clean)
    rng.shuffle(error)
    rows = []
    for clean_row, error_row in zip(clean, error):
        rows.extend((clean_row, error_row))
    return rows


def model_row(candidate: dict[str, Any]) -> dict[str, Any]:
    target = json.dumps(candidate["target"], ensure_ascii=False, separators=(",", ":"))
    return {
        "messages": [
            {"role": "user", "content": NEUTRAL_PROMPT},
            {"role": "assistant", "content": target},
        ],
        "images": [candidate["image_path"]],
    }


def audit_row(candidate: dict[str, Any], split: str, ordinal: int) -> dict[str, Any]:
    return {
        "sample_id": candidate["sample_id"],
        "split": split,
        "ordinal": ordinal,
        "label": candidate["label"],
        "source": candidate["source"],
        "source_line": candidate["source_line"],
        "image_path": candidate["image_path"],
        "box_count": len(candidate["target"]["boxes"]),
    }


def validate_model_rows(rows: list[dict[str, Any]], expected: int) -> None:
    if len(rows) != expected:
        raise ValueError(f"unexpected row count: {len(rows)} != {expected}")
    image_paths = set()
    labels = {"no_error": 0, "error": 0}
    for index, row in enumerate(rows):
        if set(row) != {"messages", "images"}:
            raise ValueError(f"model-visible extra columns at row {index}: {set(row)}")
        prompt = row["messages"][0]["content"]
        lower = prompt.lower()
        for marker in FORBIDDEN_INPUT_MARKERS:
            if marker in lower:
                raise ValueError(f"forbidden input marker {marker!r} at row {index}")
        target = json.loads(row["messages"][1]["content"])
        label = "no_error" if target["no_error"] else "error"
        labels[label] += 1
        if bool(target["no_error"]) != (target["boxes"] == []):
            raise ValueError(f"inconsistent target at row {index}")
        image_path = row["images"][0]
        if image_path in image_paths:
            raise ValueError(f"duplicate image in split: {image_path}")
        image_paths.add(image_path)
    if labels["no_error"] * 2 != expected or labels["error"] * 2 != expected:
        raise ValueError(f"unbalanced labels: {labels}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rng = random.Random(SEED)

    pools = {
        "clean_wikimedia": load_candidates(CLEAN_WIKIMEDIA, "clean_wikimedia", "no_error"),
        "clean_pinterest": load_candidates(CLEAN_PINTEREST, "clean_pinterest", "no_error"),
        "error_v41_hp": load_candidates(ERROR_V41_HP, "error_v41_hp", "error"),
        "error_v41_flash": load_candidates(ERROR_FLASH, "error_v41_flash", "error"),
    }

    cw_train, cw_val = take_disjoint(pools["clean_wikimedia"], 1000, 200, rng)
    cp_train, cp_val = take_disjoint(pools["clean_pinterest"], 1000, 200, rng)
    hp_train, hp_val = take_disjoint(pools["error_v41_hp"], 100, 17, rng)
    flash_train, flash_val = take_disjoint(pools["error_v41_flash"], 1900, 383, rng)

    train_candidates = interleave(cw_train + cp_train, hp_train + flash_train, rng)
    val_candidates = interleave(cw_val + cp_val, hp_val + flash_val, rng)
    train_rows = [model_row(row) for row in train_candidates]
    val_rows = [model_row(row) for row in val_candidates]
    validate_model_rows(train_rows, 4000)
    validate_model_rows(val_rows, 800)

    train_images = {row["images"][0] for row in train_rows}
    val_images = {row["images"][0] for row in val_rows}
    overlap = train_images & val_images
    if overlap:
        raise ValueError(f"train/validation image overlap: {len(overlap)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_jsonl(args.output_dir / "train.jsonl", train_rows)
    atomic_write_jsonl(args.output_dir / "validation.jsonl", val_rows)
    atomic_write_jsonl(
        args.output_dir / "audit_manifest.jsonl",
        [audit_row(row, "train", i) for i, row in enumerate(train_candidates)]
        + [audit_row(row, "validation", i) for i, row in enumerate(val_candidates)],
    )
    (args.output_dir / "neutral_prompt.txt").write_text(NEUTRAL_PROMPT + "\n", encoding="utf-8")

    def source_counts(items: list[dict[str, Any]]) -> dict[str, int]:
        result: dict[str, int] = {}
        for item in items:
            result[item["source"]] = result.get(item["source"], 0) + 1
        return result

    summary = {
        "seed": SEED,
        "model_visible_columns": ["messages", "images"],
        "forbidden_input_markers_checked": list(FORBIDDEN_INPUT_MARKERS),
        "text_content_fidelity_enabled": False,
        "reasoning_content_enabled": False,
        "train": {"rows": len(train_rows), "sources": source_counts(train_candidates)},
        "validation": {"rows": len(val_rows), "sources": source_counts(val_candidates)},
        "eligible_pool_sizes": {key: len(value) for key, value in pools.items()},
        "train_validation_image_overlap": 0,
        "source_files": {
            "clean_wikimedia": str(CLEAN_WIKIMEDIA),
            "clean_pinterest": str(CLEAN_PINTEREST),
            "error_v41_hp": str(ERROR_V41_HP),
            "error_v41_flash": str(ERROR_FLASH),
        },
    }
    atomic_write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
