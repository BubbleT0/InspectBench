#!/usr/bin/env python3
"""Stage-0 audit for the four-source critic SFT dataset.

This script is read-only with respect to source JSONL/images. It writes an audit
summary and row-level rejection/duplicate reports into a new experiment folder.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from PIL import Image


SOURCES = {
    "positive_wikimedia": {
        "path": "/path/to/storage/multimodal_data/research/images/image-noerror2/prefilter_candidates_vlm_judge_errors.jsonl",
        "label": "positive_no_error",
    },
    "positive_pinterest": {
        "path": "/path/to/storage/multimodal_data/research/images/pinterest_tianbo/raw-27b/raw_pinterest_vlm_judge_errors.jsonl",
        "label": "positive_no_error",
    },
    "negative_old_base_false_negative": {
        "path": "/path/to/storage/multimodal_data/research/images/image-error/output/images_without_errors.jsonl",
        "label": "negative_error",
    },
    "negative_v41_flash_50000": {
        "path": "/path/to/storage/multimodal_data/research/images/image-error/output/v4.1_flash_50000/images_with_errors_v4.1_flash_50000.jsonl",
        "label": "negative_error",
    },
}

OLD_V41 = Path("/path/to/storage/multimodal_data/research/images/image-error/output/v4.1/images_with_errors_v4.1.jsonl")
OLD_V41_HP = Path("/path/to/storage/multimodal_data/research/images/image-error/output/v4.1/filtered_nontext_high_precision/images_with_errors_v4.1_high_precision_nontext.jsonl")


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if line.strip():
                yield line_no, json.loads(line)


def image_path(source_name: str, row: dict[str, Any]) -> str | None:
    if source_name == "positive_wikimedia":
        return (row.get("source") or {}).get("local_path")
    if source_name == "positive_pinterest":
        return (row.get("source") or {}).get("local_path")
    return row.get("image_path")


def stable_identity(source_name: str, row: dict[str, Any], path: str) -> str:
    source = row.get("source") or {}
    if source_name == "positive_wikimedia" and source.get("sha1"):
        return f"sha1:{source['sha1']}"
    if source_name == "positive_pinterest" and source.get("sha256"):
        return f"sha256:{source['sha256']}"
    if row.get("sha256"):
        return f"sha256:{row['sha256']}"
    return f"path:{os.path.normpath(path)}"


def valid_box(box: Any) -> bool:
    if not isinstance(box, list) or len(box) != 4:
        return False
    try:
        x1, y1, x2, y2 = (float(value) for value in box)
    except (TypeError, ValueError):
        return False
    return 0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000


def inspect_image(path: str) -> dict[str, Any]:
    result: dict[str, Any] = {"path": path, "exists": False, "valid": False}
    try:
        stat = os.stat(path)
        result.update({"exists": True, "bytes": stat.st_size})
        with Image.open(path) as image:
            result.update({
                "format": image.format,
                "width": image.width,
                "height": image.height,
                "mode": image.mode,
            })
            image.verify()
        result["valid"] = True
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(tmp, path)


def load_paths(path: Path) -> dict[str, dict[str, Any]]:
    return {
        str(row["image_path"]): row
        for _, row in iter_jsonl(path)
        if row.get("image_path")
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    old_v41 = load_paths(OLD_V41)
    old_v41_hp = load_paths(OLD_V41_HP)
    source_stats: dict[str, Any] = {}
    audit_rows: list[dict[str, Any]] = []
    parse_rejections: list[dict[str, Any]] = []
    selected_paths: set[str] = set()
    all_paths: set[str] = set()
    identity_to_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for source_name, config in SOURCES.items():
        source_path = Path(config["path"])
        stats = Counter()
        status_counts = Counter()
        prediction_counts = Counter()
        box_counts = Counter()
        for line_no, row in iter_jsonl(source_path):
            stats["rows"] += 1
            status_counts[str(row.get("status"))] += 1
            prediction_counts[str(row.get("no_error"))] += 1
            path = image_path(source_name, row)
            reasons = []
            if not path or not isinstance(path, str):
                reasons.append("missing_image_path")
            if row.get("status") not in (None, "ok", "success"):
                reasons.append("source_status_not_ok")

            annotation = None
            if source_name == "negative_old_base_false_negative" and path:
                if path in old_v41_hp:
                    annotation = old_v41_hp[path]
                    stats["old_v41_high_precision"] += 1
                elif path in old_v41:
                    stats["old_v41_non_high_precision"] += 1
                    reasons.append("old_negative_not_high_precision")
                else:
                    stats["old_v41_not_detected"] += 1
                    reasons.append("old_negative_missing_localized_annotation")
            elif source_name == "negative_v41_flash_50000":
                annotation = row
                regions = row.get("v41_flash_edit_regions") or []
                if row.get("v41_flash_has_error") is not True or not regions:
                    reasons.append("v41_flash_missing_error_regions")

            if annotation:
                regions = annotation.get("v41_flash_edit_regions") or annotation.get("v41_edit_regions") or []
                stats["annotated_regions"] += len(regions)
                for region in regions:
                    box = region.get("bbox_norm")
                    if valid_box(box):
                        box_counts["valid"] += 1
                    else:
                        box_counts["invalid"] += 1
                        reasons.append("invalid_bbox")
                    issues = region.get("issues") or []
                    if not issues:
                        reasons.append("region_without_issues")
                    stats["annotated_issues"] += len(issues)

            selected = not reasons
            if path:
                normalized = os.path.normpath(path)
                all_paths.add(normalized)
                identity = stable_identity(source_name, row, normalized)
                identity_to_rows[identity].append({
                    "source": source_name,
                    "line_no": line_no,
                    "path": normalized,
                    "selected": selected,
                })
                if selected:
                    selected_paths.add(normalized)
            else:
                normalized = None
                identity = None

            record = {
                "source": source_name,
                "source_jsonl": str(source_path),
                "line_no": line_no,
                "ground_truth": config["label"],
                "image_path": normalized,
                "identity": identity,
                "selected_by_structural_gate": selected,
                "reasons": sorted(set(reasons)),
            }
            audit_rows.append(record)
            if selected:
                stats["selected_by_structural_gate"] += 1
            else:
                stats["rejected_by_structural_gate"] += 1
                parse_rejections.append(record)

        source_stats[source_name] = {
            **dict(stats),
            "status_counts": dict(status_counts),
            "source_no_error_counts": dict(prediction_counts),
            "bbox_counts": dict(box_counts),
            "source_jsonl": str(source_path),
            "assigned_ground_truth": config["label"],
        }

    duplicate_groups = [rows for rows in identity_to_rows.values() if len(rows) > 1]
    selected_identity_duplicates = [
        rows for rows in duplicate_groups if sum(bool(row["selected"]) for row in rows) > 1
    ]

    image_results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(inspect_image, path): path for path in sorted(selected_paths)}
        for completed, future in enumerate(as_completed(futures), 1):
            image_results.append(future.result())
            if completed % 1000 == 0:
                print(json.dumps({"image_checks_completed": completed, "total": len(futures)}), flush=True)
    image_results.sort(key=lambda row: row["path"])
    invalid_images = [row for row in image_results if not row["valid"]]

    selected_by_label = Counter(
        row["ground_truth"] for row in audit_rows if row["selected_by_structural_gate"]
    )
    selected_by_source = Counter(
        row["source"] for row in audit_rows if row["selected_by_structural_gate"]
    )
    summary = {
        "stage": "stage0_audit",
        "policy": {
            "positive_sources": "assigned no_error=true, boxes=[]; semantic label QC still required",
            "negative_old": "only existing v4.1 high-precision localized annotations enter main SFT",
            "negative_v41_flash_50000": "requires v41_flash_has_error=true and at least one valid localized edit region",
            "source_files_modified": False,
        },
        "source_stats": source_stats,
        "all_source_rows": len(audit_rows),
        "selected_by_structural_gate": sum(selected_by_label.values()),
        "selected_by_label": dict(selected_by_label),
        "selected_by_source": dict(selected_by_source),
        "unique_paths_all_sources": len(all_paths),
        "unique_paths_selected": len(selected_paths),
        "duplicate_identity_groups": len(duplicate_groups),
        "selected_duplicate_identity_groups": len(selected_identity_duplicates),
        "images_checked": len(image_results),
        "invalid_or_missing_selected_images": len(invalid_images),
        "semantic_qc_complete": False,
        "ready_for_split": not invalid_images and not selected_identity_duplicates,
    }

    write_json(output_dir / "audit_summary.json", summary)
    write_jsonl(output_dir / "audit_rows.jsonl", audit_rows)
    write_jsonl(output_dir / "structural_gate_rejections.jsonl", parse_rejections)
    write_jsonl(output_dir / "invalid_images.jsonl", invalid_images)
    write_json(output_dir / "duplicate_identity_groups.json", duplicate_groups)
    write_json(output_dir / "selected_duplicate_identity_groups.json", selected_identity_duplicates)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not summary["ready_for_split"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
