#!/usr/bin/env python3
"""Build source-stratified OpenAI-format critic SFT splits.

Source JSONL and images are read-only. The script emits new per-source shards,
split/audit manifests, and meta_openai.json under a new experiment directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections import Counter
from pathlib import Path
from typing import Any


SEED = 20260819
SPLIT_RATIOS = {"train": 0.90, "validation": 0.05, "test": 0.05}

POS_WIKIMEDIA = Path("/path/to/storage/multimodal_data/research/images/image-noerror2/prefilter_candidates_vlm_judge_errors.jsonl")
POS_PINTEREST = Path("/path/to/storage/multimodal_data/research/images/pinterest_tianbo/raw-27b/raw_pinterest_vlm_judge_errors.jsonl")
NEG_OLD = Path("/path/to/storage/multimodal_data/research/images/image-error/output/images_without_errors.jsonl")
NEG_OLD_HP = Path("/path/to/storage/multimodal_data/research/images/image-error/output/v4.1/filtered_nontext_high_precision/images_with_errors_v4.1_high_precision_nontext.jsonl")
NEG_FLASH = Path("/path/to/storage/multimodal_data/research/images/image-error/output/v4.1_flash_50000/images_with_errors_v4.1_flash_50000.jsonl")

FAMILY_MAP = {
    "rendering_artifact": "D1_Artifact/A1_RenderingArtifact",
    "repeated_texture": "D1_Artifact/A1_RenderingArtifact",
    "structural_artifact": "D1_Artifact/A2_StructuralArtifact",
    "topology_artifact": "D1_Artifact/A3_TopologyArtifact",
    "relational_physical_artifact": "D1_Artifact/A4_RelationalAndPhysicalArtifact",
    "material_failure": "D1_Artifact/A4_RelationalAndPhysicalArtifact",
    "text_visual_integrity": "D2_TextCorrectness/A1_TextVisualIntegrity",
    "text_content_fidelity": "D2_TextCorrectness/A2_TextContentFidelity",
}

FAMILY_PRIORITY = {
    "structural_artifact": 0,
    "topology_artifact": 1,
    "relational_physical_artifact": 2,
    "rendering_artifact": 3,
    "repeated_texture": 4,
    "material_failure": 5,
    "text_visual_integrity": 6,
    "text_content_fidelity": 7,
}

PROMPT_PREFIX = """You are a meticulous visual anomaly judge. Inspect the entire image for localized generation defects. Evaluate rendering artifacts, malformed entity structure, topology/connectivity errors, relational or physical impossibilities, visually broken text, and text-content mismatch only when target text is explicitly supplied. Do not report ordinary blur from depth of field, compression, artistic style, aesthetic preference, or instruction compliance. Return tight, non-overlapping boxes: one spatial anomaly per box, without duplicates or a large box merging separate defects.

Allowed family values:
- D1_Artifact/A1_RenderingArtifact: local blending, smearing, texture/detail/color corruption
- D1_Artifact/A2_StructuralArtifact: extra, missing, fused, split, deformed, misplaced, disconnected or proportionally wrong parts
- D1_Artifact/A3_TopologyArtifact: broken/wrong paths, connections, branches, closures, crossings, arrows or symbols
- D1_Artifact/A4_RelationalAndPhysicalArtifact: penetration, occlusion/contact/support/scale/perspective/light/reflection/pose/material inconsistency
- D2_TextCorrectness/A1_TextVisualIntegrity: malformed, fused, overlapping, misaligned, blurred, occluded or clipped glyphs/text
- D2_TextCorrectness/A2_TextContentFidelity: readable text differs from an explicitly provided target"""

PROMPT_SUFFIX = """Coordinates use corner coordinates on a 0..1000 image coordinate system. `bbox_1000_xyxy` must be exactly `[x_min, y_min, x_max, y_max]`: the first pair is the top-left corner and the second pair is the bottom-right corner. Do NOT output width or height. Require 0 <= x_min < x_max <= 1000 and 0 <= y_min < y_max <= 1000. If no anomaly is visible, return an empty boxes list.

Return ONLY this JSON shape:
{"no_error": false, "boxes": [{"bbox_1000_xyxy": [100, 200, 300, 400], "family": "one allowed family", "description": "brief visible evidence"}]}"""


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if line.strip():
                yield line_no, json.loads(line)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(tmp, path)


def stable_id(source: str, identity: str) -> str:
    digest = hashlib.sha256(f"{source}\0{identity}".encode()).hexdigest()[:20]
    return f"{source}/{digest}"


def prompt(generation_prompt: str | None) -> str:
    if generation_prompt:
        context = f"Reference generation instruction/target, when useful:\n{generation_prompt}"
    else:
        context = "No target text or generation instruction is available; do not infer text-content mismatch from spelling plausibility alone."
    return f"<image>\n{PROMPT_PREFIX}\n\n{context}\n\n{PROMPT_SUFFIX}"


def valid_box(box: Any) -> bool:
    if not isinstance(box, list) or len(box) != 4:
        return False
    try:
        x1, y1, x2, y2 = (int(round(float(value))) for value in box)
    except (TypeError, ValueError):
        return False
    return 0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000


def region_to_box(region: dict[str, Any]) -> dict[str, Any]:
    bbox = [int(round(float(value))) for value in region["bbox_norm"]]
    issues = [issue for issue in (region.get("issues") or []) if issue.get("artifact_type") in FAMILY_MAP]
    if not issues:
        raise ValueError("region has no mappable issues")
    issue = sorted(
        issues,
        key=lambda value: (
            -float(value.get("confidence") or 0),
            FAMILY_PRIORITY.get(value.get("artifact_type"), 99),
        ),
    )[0]
    description = str(issue.get("observation") or issue.get("abnormality") or region.get("edit_region_rationale") or region.get("edit_target") or "localized visual anomaly")
    return {
        "bbox_1000_xyxy": bbox,
        "family": FAMILY_MAP[issue["artifact_type"]],
        "description": " ".join(description.split())[:500],
    }


def make_record(
    *, source_name: str, source_jsonl: Path, image_path: str, identity: str,
    label: str, generation_prompt: str | None, regions: list[dict[str, Any]] | None,
    source_line: int, source_sha: str | None, source_model: str | None,
    source_theme: str | None, annotation_basis: str,
) -> dict[str, Any]:
    if label == "positive":
        target = {"no_error": True, "boxes": []}
        reasoning = "The image was checked for qualifying localized anomalies. Under the user-approved source-label policy, this curated positive sample contains no reportable boxes."
        base_prediction = "error"
        error_relation = "base_false_positive"
    else:
        boxes = []
        for region in regions or []:
            if not valid_box(region.get("bbox_norm")):
                raise ValueError(f"invalid bbox for {image_path}: {region.get('bbox_norm')}")
            boxes.append(region_to_box(region))
        if not boxes:
            raise ValueError(f"negative sample has no boxes: {image_path}")
        target = {"no_error": False, "boxes": boxes}
        reasoning = "The image was checked for localized visual anomalies. Only independently localized regions retained by the approved v4.1 annotation gate are included."
        base_prediction = "no_error"
        error_relation = "base_false_negative"

    sample_id = stable_id(source_name, identity)
    return {
        "messages": [
            {"role": "user", "content": prompt(generation_prompt)},
            {
                "role": "assistant",
                "content": json.dumps(target, ensure_ascii=False, separators=(",", ":")),
                "reasoning_content": reasoning,
            },
        ],
        "metadata": {
            "sample_id": sample_id,
            "identity": identity,
            "ground_truth_label": label,
            "ground_truth": "no_error" if label == "positive" else "error",
            "base_prediction": base_prediction,
            "error_relation": error_relation,
            "source_name": source_name,
            "source_jsonl": str(source_jsonl),
            "source_line": source_line,
            "source_sha": source_sha,
            "source_model": source_model,
            "source_theme": source_theme,
            "annotation_basis": annotation_basis,
            "semantic_qc": "waived_by_user",
            "selection_seed": SEED,
        },
        "image": [image_path],
    }


def load_positive(source_name: str, path: Path) -> list[dict[str, Any]]:
    records = []
    for line_no, row in iter_jsonl(path):
        source = row.get("source") or {}
        image_path = source.get("local_path")
        if not image_path or row.get("status") != "ok":
            raise ValueError(f"invalid positive row {path}:{line_no}")
        source_sha = source.get("sha256") or source.get("sha1")
        identity = f"sha:{source_sha}" if source_sha else f"path:{os.path.normpath(image_path)}"
        records.append(make_record(
            source_name=source_name,
            source_jsonl=path,
            image_path=os.path.normpath(image_path),
            identity=identity,
            label="positive",
            generation_prompt=None,
            regions=None,
            source_line=line_no,
            source_sha=source_sha,
            source_model=None,
            source_theme=(source.get("rule_category") or ((source.get("primary_source") or {}).get("metadata") or {}).get("search_target")),
            annotation_basis="user_assigned_actual_no_error; semantic_qc_waived",
        ))
    return records


def load_old_negative() -> list[dict[str, Any]]:
    original_by_path = {
        os.path.normpath(row["image_path"]): (line_no, row)
        for line_no, row in iter_jsonl(NEG_OLD)
        if row.get("image_path")
    }
    records = []
    for line_no, annotation in iter_jsonl(NEG_OLD_HP):
        image_path = os.path.normpath(annotation["image_path"])
        if image_path not in original_by_path:
            raise ValueError(f"old high-precision path absent from source: {image_path}")
        source_line, original = original_by_path[image_path]
        records.append(make_record(
            source_name="negative_old_v41_high_precision",
            source_jsonl=NEG_OLD,
            image_path=image_path,
            identity=f"path:{image_path}",
            label="negative",
            generation_prompt=original.get("generation_prompt"),
            regions=annotation.get("v41_edit_regions") or [],
            source_line=source_line,
            source_sha=original.get("sha256"),
            source_model=annotation.get("source_model") or original.get("model"),
            source_theme=annotation.get("source_theme") or original.get("theme"),
            annotation_basis="v4.1_high_precision_nontext_keep",
        ))
    return records


def load_flash_negative() -> list[dict[str, Any]]:
    records = []
    for line_no, row in iter_jsonl(NEG_FLASH):
        image_path = os.path.normpath(row["image_path"])
        regions = row.get("v41_flash_edit_regions") or []
        if row.get("v41_flash_has_error") is not True or not regions:
            raise ValueError(f"invalid v4.1-flash row {line_no}")
        records.append(make_record(
            source_name="negative_v41_flash_50000",
            source_jsonl=NEG_FLASH,
            image_path=image_path,
            identity=f"path:{image_path}",
            label="negative",
            generation_prompt=row.get("generation_prompt"),
            regions=regions,
            source_line=line_no,
            source_sha=None,
            source_model=row.get("source_model"),
            source_theme=row.get("theme"),
            annotation_basis=f"v4.1-flash:{row.get('v41_flash_pipeline_version')};confidence={row.get('v41_flash_confidence')}",
        ))
    return records


def stratified_split(records: list[dict[str, Any]], source_name: str) -> dict[str, list[dict[str, Any]]]:
    shuffled = list(records)
    random.Random(f"{SEED}:{source_name}").shuffle(shuffled)
    n = len(shuffled)
    n_validation = round(n * SPLIT_RATIOS["validation"])
    n_test = round(n * SPLIT_RATIOS["test"])
    n_train = n - n_validation - n_test
    splits = {
        "train": shuffled[:n_train],
        "validation": shuffled[n_train:n_train + n_validation],
        "test": shuffled[n_train + n_validation:],
    }
    for split, rows in splits.items():
        for row in rows:
            row["metadata"]["split"] = split
    return splits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"refusing non-empty output dir: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    sources = {
        "positive_wikimedia": load_positive("positive_wikimedia", POS_WIKIMEDIA),
        "positive_pinterest": load_positive("positive_pinterest", POS_PINTEREST),
        "negative_old_v41_high_precision": load_old_negative(),
        "negative_v41_flash_50000": load_flash_negative(),
    }

    all_ids = [row["metadata"]["sample_id"] for rows in sources.values() for row in rows]
    all_paths = [row["image"][0] for rows in sources.values() for row in rows]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("duplicate sample_id after conversion")
    if len(all_paths) != len(set(all_paths)):
        raise ValueError("duplicate image path after conversion")

    meta = {}
    split_manifest = []
    counts = Counter()
    for source_name, records in sources.items():
        for split, rows in stratified_split(records, source_name).items():
            shard = f"{source_name}_{split}"
            # agen-sft-pipeline structured_qc discovers <agentType>/*.jsonl.
            shard_path = output_dir / "critic" / f"{shard}.jsonl"
            write_jsonl(shard_path, rows)
            label = rows[0]["metadata"]["ground_truth_label"] if rows else (
                "positive" if source_name.startswith("positive") else "negative"
            )
            counts[f"{split}:{label}"] += len(rows)
            counts[f"source:{source_name}"] += len(rows)
            meta[shard] = {
                "annotation": str(shard_path),
                "author": "research",
                "class": "critic",
                "comment": f"formal critic SFT; source={source_name}; split={split}; label={label}; semantic QC waived by user",
                "enable_thinking": True,
                "expect_name": shard,
                "from": "critic_full_4src_16gpu_neo1_trajectory_20260819",
                "length": len(rows),
                "repeat_time": 1.0,
                "team": "research",
                "tools": [],
            }
            split_manifest.extend({
                "sample_id": row["metadata"]["sample_id"],
                "identity": row["metadata"]["identity"],
                "image_path": row["image"][0],
                "source_name": source_name,
                "label": label,
                "split": split,
                "shard": shard,
            } for row in rows)

    write_json(output_dir / "meta_openai.json", meta)
    write_jsonl(output_dir / "split_manifest.jsonl", split_manifest)
    summary = {
        "seed": SEED,
        "split_ratios": SPLIT_RATIOS,
        "source_counts": {key: len(value) for key, value in sources.items()},
        "counts": dict(sorted(counts.items())),
        "total_records": len(split_manifest),
        "unique_sample_ids": len(set(all_ids)),
        "unique_image_paths": len(set(all_paths)),
        "semantic_qc": "waived_by_user",
        "old_negative_policy": "117 v4.1 high-precision localized records only",
    }
    write_json(output_dir / "conversion_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
