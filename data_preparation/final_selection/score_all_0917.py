#!/usr/bin/env python3
"""Rescore all results.md runs against the cleaned 0917 annotation (1039 images).

Each config's existing normalized predictions.jsonl is reused verbatim; only the
ground truth changes. Writes new_results/<config>/metrics.json (bootstrap 2000)
and new_results/summary_0917.csv with the 16 requested metrics.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

from vlm_judge_bench.data import load_annotations, load_predictions
from vlm_judge_bench.metrics import compute_metrics

ROOT = Path("/path/to/storage/research/WorkSpace3/critic/benchmark/data_0804")
OUT = ROOT / "new_results"
ANNOTATIONS = [
    OUT / "annotation_0917_clean/GPT_completed_208_20260917_cleaned.jsonl",
    OUT / "annotation_0917_clean/External4_completed_499_20260917_cleaned.jsonl",
    OUT / "annotation_0917_clean/U1.5_completed_141_20260910_cleaned.jsonl",
    OUT / "annotation_0917_clean/U1.5_completed_clean_81_20260910_cleaned.jsonl",
    OUT / "annotation_0917_clean/Natural_clean_110_cleaned.jsonl",
]

# results.md label -> predictions.jsonl location
CONFIGS = {
    "Qwen3.6-27b": ROOT / "runs/direct_full_qwen36_27b_zeroshot_1054/local_qwen_27b_zeroshot_1054/predictions.jsonl",
    "0722_ck145": ROOT / "runs/direct_full_0722_ck145_1054/0722_ck145_direct_1054/predictions.jsonl",
    "0722_ck145 + v4": ROOT / "runs/scored_u1_v41_original_1054/u1_v41_original_27b_pipeline/predictions.jsonl",
    "0722_ck145 + v5": ROOT / "runs/scored_u1_v5_local_0807_host8000_full/u1_v5_local_27b_pipeline/predictions.jsonl",
    "0722_ck145 + v6": ROOT / "runs/scored_u1_v41_flash_0812/u1_v41_flash_27b_pipeline/predictions.jsonl",
    "0722_ck145 + v6.3": ROOT / "runs/scored_u1_v63_20260814_full1054/u1_v63_pipeline/predictions.jsonl",
    "0820_ck1001": ROOT / "runs/direct_full_ck1001_0820_zeroshot_1054/ck1001_0820_zeroshot_1054/predictions.jsonl",
    "0820_ck1001 + v6.3": ROOT / "runs/scored_u1_v63_ck1001_0820_1054/u1_v63_ck1001_0820_pipeline/predictions.jsonl",
    "0820_ck740": ROOT / "runs/direct_full_ck740_0820_zeroshot_1054/ck740_0820_zeroshot_1054/predictions.jsonl",
    "0826_ck136": ROOT / "runs/0826-ck136/0826-ck136/predictions.jsonl",
    "0826_ck136 + v6.3": ROOT / "runs/scored_v63_0826_ck136_1054/v63+0826_ck136/predictions.jsonl",
    "0828_ck444": ROOT / "runs/direct_full_0828_ck444_1054/0828_ck444/predictions.jsonl",
    "0828_ck688": ROOT / "runs/direct_full_0828_ck688_1054/0828_ck688/predictions.jsonl",
    "Qwen3.8-27b": ROOT / "runs/direct_full_qwen38_27b_zeroshot_1054/local_qwen38_27b_zeroshot_1054/predictions.jsonl",
}

CSV_COLUMNS = [
    "配置",
    "正式区域框主分 ↑",
    "Artifact ↑",
    "文字 ↑",
    "有错图弹性分 ↑",
    "无错图软分 ↑",
    "有错图判无错率 ↓",
    "无错图判有错率 ↓",
    "总体误判率 ↓",
    "无错图平均预测框数 ↓",
    "质量>0.1的召回率 ↑",
    "质量>0.5的召回率 ↑",
    "框平均质量 ↑",
    "完全漏检率 ↓",
    "数量平均绝对误差 ↓",
    "数量偏差（绝对值）↓",
    "背景误报框数 ↓",
    "失败或缺失预测数",
    "主分95%CI下界",
    "主分95%CI上界",
]

# results.md 0917 zero-shot models (direct runs on cleaned-0917 annotation)
ZEROSHOT_0917_CONFIGS = {
    "qwen3.7-plus": ROOT / "runs/direct_0917_zeroshot_1039/qwen37_0917/predictions.jsonl",
    "gpt-5.6-sol": ROOT / "runs/direct_0917_zeroshot_1039/gpt56_0917/predictions.jsonl",
    "gemini-3.5-flash": ROOT / "runs/direct_0917_zeroshot_1039/gemini35_0917/predictions.jsonl",
}


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", help="score only these config labels (e.g. the 0917 zero-shot models)")
    parser.add_argument("--csv", default="summary_0917.csv", help="output csv filename under new_results/")
    parser.add_argument("--append", action="store_true", help="append rows to the csv instead of overwriting")
    args = parser.parse_args()

    configs = dict(CONFIGS)
    if args.only:
        configs = {k: v for k, v in configs.items() if k in set(args.only)}
    # zero-shot 0917 runs can be scored alone or together with the legacy configs
    if not args.only or any(k in set(args.only) for k in ZEROSHOT_0917_CONFIGS):
        configs.update({k: v for k, v in ZEROSHOT_0917_CONFIGS.items() if not args.only or k in set(args.only)})

    samples = load_annotations(ANNOTATIONS)
    print(f"loaded {len(samples)} cleaned-0917 samples; scoring {len(configs)} configs")

    rows = []
    for label, pred_path in configs.items():
        assert pred_path.is_file(), pred_path
        predictions = load_predictions(pred_path)
        missing = [s.sample_id for s in samples if s.sample_id not in predictions]
        t0 = time.time()
        metrics, sample_rows = compute_metrics(samples, predictions, bootstrap=2000, seed=20260804)
        h = metrics["headline"]
        d = metrics["detection"]
        gt_boxes = metrics["dataset"]["gt_boxes"]

        config_dir = OUT / label.replace(" ", "_").replace("/", "_")
        atomic_json(config_dir / "metrics.json", metrics)
        atomic_json(config_dir / "per_sample_metrics.json", [
            {k: v for k, v in row.items() if k not in {"gt_pairs", "matched"}}
            for row in sample_rows
        ])

        ci = metrics.get("bootstrap", {}).get("ci95", {}).get("elastic_bbox_score", [None, None])
        def rnd(value, digits=4):
            return round(value, digits) if value is not None else None

        rows.append({
            "配置": label,
            "正式区域框主分 ↑": round(h["elastic_bbox_score"], 3),
            "Artifact ↑": round(h["artifact_region_bbox_score"], 3),
            "文字 ↑": round(h["text_region_bbox_score"], 3),
            "有错图弹性分 ↑": round(h["positive_elastic"], 4),
            "无错图软分 ↑": round(h["clean_soft"], 4),
            "有错图判无错率 ↓": round(metrics["binary_decision"]["positive_false_negative_rate"], 4),
            "无错图判有错率 ↓": round(metrics["binary_decision"]["clean_false_positive_rate"], 4),
            "总体误判率 ↓": round(metrics["binary_decision"]["overall_error_rate"], 4),
            "无错图平均预测框数 ↓": round(h["clean_fppi"], 4),
            "质量>0.1的召回率 ↑": round(d["recall_0_1"], 4),
            "质量>0.5的召回率 ↑": round(d["recall_0_5"], 4),
            "框平均质量 ↑": rnd(d["found_quality"]),
            "完全漏检率 ↓": round(d["hard_miss"] / gt_boxes, 4),
            "数量平均绝对误差 ↓": round(metrics["counts"]["all"]["mae"], 4),
            "数量偏差（绝对值）↓": round(abs(metrics["counts"]["all"]["bias"]), 4),
            "背景误报框数 ↓": metrics["multi_box"]["background_fp"],
            "失败或缺失预测数": metrics["dataset"]["failed_or_missing_predictions"],
            "主分95%CI下界": round(ci[0], 3) if ci and ci[0] is not None else None,
            "主分95%CI上界": round(ci[1], 3) if ci and ci[1] is not None else None,
        })
        print(f"{label}: main={h['elastic_bbox_score']:.3f} pos={h['positive_elastic']:.4f} "
              f"clean={h['clean_soft']:.4f} missing_pred={len(missing)} "
              f"failed={metrics['dataset']['failed_or_missing_predictions']} ({time.time()-t0:.1f}s)")
        if missing:
            print(f"  WARNING: {len(missing)} samples without prediction, scored as empty")

    csv_path = OUT / args.csv
    if args.append and csv_path.exists():
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            existing = [row for row in csv.DictReader(handle)]
        existing_labels = {row["配置"] for row in existing}
        rows = [row for row in existing if row["配置"] not in {r["配置"] for r in rows}] + rows
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {csv_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
