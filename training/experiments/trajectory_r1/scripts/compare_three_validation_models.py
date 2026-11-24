#!/usr/bin/env python3
"""Compare paired critic predictions from several endpoints on one split."""

import argparse
import json
import math
from collections import Counter
from pathlib import Path


def load(path):
    rows = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            key = (row["source"], row["source_index"], row["image"])
            rows[key] = row
    return rows


def correct(row):
    return bool(row["target"]["no_error"]) == bool(row["prediction"]["no_error"])


def exact_mcnemar_two_sided(b, c):
    n = b + c
    if not n:
        return 1.0
    k = min(b, c)
    logs = [
        math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) - n * math.log(2)
        for i in range(k + 1)
    ]
    anchor = max(logs)
    lower = math.exp(anchor) * sum(math.exp(value - anchor) for value in logs)
    return min(1.0, 2 * lower)


def pairwise(a, b, keys):
    counts = Counter((correct(a[k]), correct(b[k])) for k in keys)
    only_a = counts[(True, False)]
    only_b = counts[(False, True)]
    return {
        "both_correct": counts[(True, True)],
        "only_first_correct": only_a,
        "only_second_correct": only_b,
        "both_wrong": counts[(False, False)],
        "mcnemar_exact_two_sided_p": exact_mcnemar_two_sided(only_a, only_b),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", nargs=3, metavar=("NAME", "PREDICTIONS", "SUMMARY"), required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    predictions = {name: load(pred) for name, pred, _ in args.model}
    summaries = {name: json.loads(Path(summary).read_text(encoding="utf-8")) for name, _, summary in args.model}
    key_sets = [set(rows) for rows in predictions.values()]
    if not key_sets or any(keys != key_sets[0] for keys in key_sets[1:]):
        raise SystemExit("prediction key mismatch")
    keys = sorted(key_sets[0])

    metrics = {}
    for name in predictions:
        summary = summaries[name]
        cls = summary["classification"]
        metrics[name] = {
            "samples": summary["samples"],
            "valid_predictions": summary["valid_predictions"],
            "accuracy": cls["accuracy"],
            "error_recall": cls["recall"],
            "clean_specificity": cls["specificity"],
            "balanced_accuracy": cls["balanced_accuracy"],
            "classification_f1": cls["f1"],
            "bbox_f1": summary["bbox_iou_0_5"]["f1"],
            "bbox_family_f1": summary["bbox_iou_0_5_and_family"]["f1"],
            "mean_latency_s": summary["mean_latency_s"],
            "prediction_labels": Counter(
                "no_error" if row["prediction"]["no_error"] else "error"
                for row in predictions[name].values() if row["prediction"] is not None
            ),
        }

    names = list(predictions)
    pairs = {}
    for i, first in enumerate(names):
        for second in names[i + 1:]:
            pairs[f"{first}__vs__{second}"] = pairwise(predictions[first], predictions[second], keys)

    best_ba = max(metrics[name]["balanced_accuracy"] for name in names)
    close = [name for name in names if best_ba - metrics[name]["balanced_accuracy"] <= 0.01]
    selected = max(
        close,
        key=lambda name: (
            metrics[name]["clean_specificity"],
            metrics[name]["bbox_family_f1"],
        ),
    )
    result = {
        "selection_rule": "highest balanced accuracy; within 0.01 prefer clean specificity; bbox+family F1 tertiary",
        "selected": selected,
        "metrics": metrics,
        "pairwise": pairs,
    }

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "three_model_comparison.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# checkpoint-740、checkpoint-1001 与 base 分层 validation 对照",
        "",
        f"- 样本数：{len(keys)}",
        f"- 选择规则：{result['selection_rule']}",
        f"- 当前选中：`{selected}`",
        "",
        "|模型|Balanced accuracy|无错特异度|有错召回率|Accuracy|BBox F1|BBox+family F1|错误预测/有效预测|",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in names:
        m = metrics[name]
        labels = m["prediction_labels"]
        lines.append(
            f"|{name}|{m['balanced_accuracy']:.4%}|{m['clean_specificity']:.4%}|"
            f"{m['error_recall']:.4%}|{m['accuracy']:.4%}|{m['bbox_f1']:.4%}|"
            f"{m['bbox_family_f1']:.4%}|{labels.get('error', 0)}/{m['valid_predictions']}|"
        )
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
