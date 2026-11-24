from __future__ import annotations

import csv
import json
from pathlib import Path


def _fmt(value, scale: float = 1.0) -> str:
    if value is None:
        return "—"
    return f"{float(value) * scale:.3f}"


def build_report(run_dir: str | Path) -> Path:
    run_path = Path(run_dir)
    model_metrics: dict[str, dict] = {}
    for path in sorted(run_path.glob("*/metrics.json")):
        with path.open("r", encoding="utf-8") as handle:
            model_metrics[path.parent.name] = json.load(handle)
    if not model_metrics:
        raise FileNotFoundError(f"no */metrics.json files found under {run_path}")

    columns = [
        "model", "main", "positive_elastic", "clean_soft", "clean_specificity", "clean_fppi",
        "recall_0_1", "recall_0_5", "recall_0_75", "found_quality", "count_mae", "count_bias",
        "coverage", "tightness", "failed",
    ]
    rows: list[dict] = []
    for model, metrics in model_metrics.items():
        h, d = metrics["headline"], metrics["detection"]
        rows.append({
            "model": model, "main": h["elastic_bbox_score"], "positive_elastic": h["positive_elastic"],
            "clean_soft": h["clean_soft"], "clean_specificity": h["clean_specificity"], "clean_fppi": h["clean_fppi"],
            "recall_0_1": d["recall_0_1"], "recall_0_5": d["recall_0_5"], "recall_0_75": d["recall_0_75"],
            "found_quality": d["found_quality"], "count_mae": metrics["counts"]["all"]["mae"],
            "count_bias": metrics["counts"]["all"]["bias"], "coverage": d["mean_coverage"],
            "tightness": d["mean_tightness"], "failed": metrics["dataset"]["failed_or_missing_predictions"],
        })
    csv_path = run_path / "summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# VLM Judge BBox Benchmark", "",
        "## Overview", "",
        "| Model | Main ↑ | Positive Elastic ↑ | Clean Soft ↑ | Clean Specificity ↑ | Clean FPPI ↓ | Failed |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(f"| {row['model']} | {_fmt(row['main'])} | {_fmt(row['positive_elastic'])} | {_fmt(row['clean_soft'])} | {_fmt(row['clean_specificity'])} | {_fmt(row['clean_fppi'])} | {row['failed']} |")
    lines += [
        "", "## Detection and localization", "",
        "| Model | Recall@0.1 ↑ | Recall@0.5 ↑ | Recall@0.75 ↑ | Found Quality ↑ | Coverage ↑ | Tightness ↑ |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(f"| {row['model']} | {_fmt(row['recall_0_1'])} | {_fmt(row['recall_0_5'])} | {_fmt(row['recall_0_75'])} | {_fmt(row['found_quality'])} | {_fmt(row['coverage'])} | {_fmt(row['tightness'])} |")
    lines += [
        "", "## Count behavior", "",
        "| Model | Count MAE ↓ | Count Bias →0 |", "|---|---:|---:|",
    ]
    for row in rows:
        lines.append(f"| {row['model']} | {_fmt(row['count_mae'])} | {_fmt(row['count_bias'])} |")
    lines += [
        "", "> Main is the 0–100 Elastic BBox Score. All other scores are on 0–1 unless their name denotes a count.",
        "> Text Content Fidelity is retained in the bbox-only main score, but the current annotations contain no reference prompts; interpret that family diagnostically.", "",
    ]
    output = run_path / "summary.md"
    output.write_text("\n".join(lines), encoding="utf-8")
    return output
