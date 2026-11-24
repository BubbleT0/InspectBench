#!/usr/bin/env python3
"""Standalone comprehensive analysis of the cleaned 0917 annotation.

Every bucket (per-image gt count, bbox area, error family, per-image stats)
is broken down by source sub-dataset and positive/clean split.
Writes new_results/annotation_0917_analysis.md and .json.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path("/path/to/storage/research/WorkSpace3/critic/benchmark/data_0804")
OUT = ROOT / "new_results"

ANNOTATIONS = [
    OUT / "annotation_0917_clean/GPT_completed_208_20260917_cleaned.jsonl",
    OUT / "annotation_0917_clean/External4_completed_499_20260917_cleaned.jsonl",
    OUT / "annotation_0917_clean/U1.5_completed_141_20260910_cleaned.jsonl",
    OUT / "annotation_0917_clean/U1.5_completed_clean_81_20260910_cleaned.jsonl",
    OUT / "annotation_0917_clean/Natural_clean_110_cleaned.jsonl",
]

FAMILY_LABELS = {
    "D1_Artifact/A1_RenderingArtifact": "D1/A1 渲染瑕疵",
    "D1_Artifact/A2_StructuralArtifact": "D1/A2 结构异常",
    "D1_Artifact/A3_TopologyArtifact": "D1/A3 拓扑异常",
    "D1_Artifact/A4_RelationalAndPhysicalArtifact": "D1/A4 关系与物理异常",
    "D2_TextCorrectness/A1_TextVisualIntegrity": "D2/A1 文字视觉完整性",
    "D2_TextCorrectness/A2_TextContentFidelity": "D2/A2 文字内容保真",
}
SOURCE_ORDER = ["gpt", "external4", "u1_5", "u1_5_clean", "natural_clean_110"]
SOURCE_LABELS = {
    "gpt": "gpt",
    "external4": "external4",
    "u1_5": "U1.5",
    "u1_5_clean": "U1.5 clean",
    "natural_clean_110": "natural_110",
}
AREA_BUCKETS = [
    ("tiny(<0.1%)", lambda f: f < 0.001),
    ("small(0.1-0.5%)", lambda f: 0.001 <= f < 0.005),
    ("medium(0.5-2%)", lambda f: 0.005 <= f <= 0.02),
    ("large(>2%)", lambda f: f > 0.02),
]
COUNT_BUCKETS = [1, 2, 3, 4, "5+"]


def source_of(filename: str) -> str:
    name = filename.lower()
    if "natural_clean_110" in name:
        return "natural_clean_110"
    if "gpt_completed_" in name:
        return "gpt"
    if "u1.5_completed_clean" in name:
        return "u1_5_clean"
    if "u1.5_completed_" in name:
        return "u1_5"
    if "external4_completed_" in name:
        return "external4"
    return "other"


def load() -> list[dict]:
    rows = []
    for p in ANNOTATIONS:
        src = source_of(p.name)
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                row["_source"] = src
                rows.append(row)
    return rows


def fmt_pct(values: list[float]) -> str:
    if not values:
        return "—"
    values_s = sorted(values)
    n = len(values_s)
    mean = sum(values_s) / n
    median = values_s[n // 2] if n % 2 else (values_s[n // 2 - 1] + values_s[n // 2]) / 2
    return f"{mean*100:.2f}% / {median*100:.2f}%"


def source_stats(rows: list[dict]) -> dict:
    return {
        "images": len(rows),
        "positive": sum(1 for r in rows if r["annotation"]["boxes"]),
        "clean": sum(1 for r in rows if not r["annotation"]["boxes"]),
        "gt_boxes": sum(len(r["annotation"]["boxes"]) for r in rows),
        "per_image_counts": [len(r["annotation"]["boxes"]) for r in rows if r["annotation"]["boxes"]],
    }


def analyze(rows: list[dict]) -> dict:
    by_source: dict[str, list[dict]] = {s: [] for s in SOURCE_ORDER}
    for r in rows:
        by_source[r["_source"]].append(r)

    def enrich(st: dict, srows: list[dict]) -> dict:
        counts = st["per_image_counts"]
        st["avg_boxes_per_positive"] = round(sum(counts) / len(counts), 3) if counts else 0
        st["max_boxes_per_image"] = max(counts) if counts else 0
        st["min_boxes_per_image"] = min(counts) if counts else 0
        st["gt_count_buckets"] = {
            str(k): (sum(1 for c in counts if c >= 5) if k == "5+" else sum(1 for c in counts if c == k))
            for k in COUNT_BUCKETS
        }
        area_counter: Counter = Counter()
        fracs: list[float] = []
        family_counter: Counter = Counter()
        for r in srows:
            area = r["image_width"] * r["image_height"]
            for b in r["annotation"]["boxes"]:
                x, y, w, h = b["bbox_xywh"]
                frac = (w * h) / area
                fracs.append(frac)
                for label, pred in AREA_BUCKETS:
                    if pred(frac):
                        area_counter[label] += 1
                        break
                family_counter[b["error_type"]["code"]] += 1
        st["area_buckets"] = dict(area_counter)
        st["area_mean_median_pct"] = fmt_pct(fracs)
        st["families"] = dict(family_counter)
        return st

    result = {"overall": enrich(source_stats(rows), rows), "by_source": {}}
    for src, srows in by_source.items():
        result["by_source"][src] = enrich(source_stats(srows), srows)
    return result


def main() -> None:
    rows = load()
    assert len(rows) == 1039, len(rows)
    result = analyze(rows)
    (OUT / "annotation_0917_analysis.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    o = result["overall"]
    lines: list[str] = []
    a = lines.append
    a("# 0917 新版标注（清洗后）全方位分析")
    a("")
    a("标注文件：`new_results/annotation_0917_clean/`（5 个来源文件，已通过 `vlm_judge_bench validate` 校验）。")
    a("")
    a("## 1. 总体规模")
    a("")
    a("| 指标 | 数值 |")
    a("| --- | ---: |")
    a(f"| 图片总数 | {o['images']} |")
    a(f"| 有错图 | {o['positive']} |")
    a(f"| 无错图 | {o['clean']} |")
    a(f"| 人工真值框总数 | {o['gt_boxes']} |")
    a(f"| 有错图平均框数 | {o['avg_boxes_per_positive']} |")
    a(f"| 单图最多 / 最少框数 | {o['max_boxes_per_image']} / {o['min_boxes_per_image']} |")
    a(f"| 框面积占全图比例（均值 / 中位数） | {o['area_mean_median_pct']} |")
    a("")
    a("## 2. 各来源子数据集构成")
    a("")
    a("| 来源 | 图片数 | 有错图 | 无错图 | 真值框 | 有错图平均框数 | 单图最多 / 最少 |")
    a("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for src in SOURCE_ORDER:
        s = result["by_source"][src]
        a(f"| {SOURCE_LABELS[src]} | {s['images']} | {s['positive']} | {s['clean']} | {s['gt_boxes']} | "
          f"{s['avg_boxes_per_positive']} | {s['max_boxes_per_image']} / {s['min_boxes_per_image']} |")
    a(f"| 合计 | {o['images']} | {o['positive']} | {o['clean']} | {o['gt_boxes']} | "
      f"{o['avg_boxes_per_positive']} | {o['max_boxes_per_image']} / {o['min_boxes_per_image']} |")
    a("")
    a("注：U1.5 clean 与 natural_110 为纯无错子集（0 真值框），其框统计不适用；gpt、U1.5 为有错/无错混合来源；external4 为纯有错来源。")
    a("")

    a("## 3. 有错图按每图真值框数量分布（按来源）")
    a("")
    a("| 来源 | " + " | ".join(str(k) for k in COUNT_BUCKETS) + " | 有错图合计 |")
    a("| --- " * (len(COUNT_BUCKETS) + 2) + "|")
    for src in SOURCE_ORDER:
        s = result["by_source"][src]
        a(f"| {SOURCE_LABELS[src]} | " + " | ".join(str(s["gt_count_buckets"][str(k)]) for k in COUNT_BUCKETS) +
          f" | {s['positive']} |")
    a(f"| 合计 | " + " | ".join(str(o["gt_count_buckets"][str(k)]) for k in COUNT_BUCKETS) + f" | {o['positive']} |")
    a("")

    a("## 4. 真值框按面积分桶（按来源，纯无错来源不适用）")
    a("")
    area_labels = [label for label, _ in AREA_BUCKETS]
    a("| 来源 | " + " | ".join(area_labels) + " | 框数合计 |")
    a("| --- " * (len(area_labels) + 2) + "|")
    for src in SOURCE_ORDER:
        s = result["by_source"][src]
        if not s["gt_boxes"]:
            continue
        a(f"| {SOURCE_LABELS[src]} | " + " | ".join(str(s["area_buckets"].get(l, 0)) for l in area_labels) +
          f" | {s['gt_boxes']} |")
    a(f"| 合计 | " + " | ".join(str(o["area_buckets"].get(l, 0)) for l in area_labels) + f" | {o['gt_boxes']} |")
    a("")

    a("## 5. 错误类别分布（按来源，框数）")
    a("")
    fams = [f for f in FAMILY_LABELS if o["families"].get(f)]
    a("| 来源 | " + " | ".join(FAMILY_LABELS[f] for f in fams) + " | 框数合计 |")
    a("| --- " * (len(fams) + 2) + "|")
    for src in SOURCE_ORDER:
        s = result["by_source"][src]
        if not s["gt_boxes"]:
            continue
        a(f"| {SOURCE_LABELS[src]} | " + " | ".join(str(s["families"].get(f, 0)) for f in fams) +
          f" | {s['gt_boxes']} |")
    a(f"| 合计 | " + " | ".join(str(o["families"].get(f, 0)) for f in fams) + f" | {o['gt_boxes']} |")
    a("")

    a("## 6. 各来源内部结构说明")
    a("")
    for src in SOURCE_ORDER:
        s = result["by_source"][src]
        pos_share = s["positive"] / s["images"] * 100
        if s["gt_boxes"]:
            top_family = max(s["families"], key=s["families"].get)
            a(f"- {SOURCE_LABELS[src]}：{s['images']} 张（有错 {s['positive']} / 无错 {s['clean']}，有错占比 {pos_share:.1f}%），"
              f"{s['gt_boxes']} 框，最主要类别为 {FAMILY_LABELS[top_family]}（{s['families'][top_family]} 框，"
              f"{s['families'][top_family]/s['gt_boxes']*100:.1f}%）。")
        else:
            a(f"- {SOURCE_LABELS[src]}：{s['images']} 张纯无错图（有错占比 0%），0 真值框，仅用于考察误报克制。")
    a("")
    (OUT / "annotation_0917_analysis.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
