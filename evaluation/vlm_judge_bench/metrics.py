from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .data import GTBox, PredictionBox, PredictionRecord, Sample

METRIC_VERSION = "merged-ground-truth-ownership-v2"
REGION_GROUP_SCORE_VERSION = "ownership-v2-gt-family-diagnostic-v1"
REGION_GROUPS = {
    "artifact": ("D1_Artifact/",),
    "text_correctness": ("D2_TextCorrectness/",),
}


@dataclass(frozen=True)
class PairStats:
    score: float
    coverage: float
    tightness: float
    dice: float
    area_ratio: float


def box_pair_stats(pred_xywh: Iterable[float], gt_xywh: Iterable[float], image_area: float) -> PairStats:
    px, py, pw, ph = map(float, pred_xywh)
    gx, gy, gw, gh = map(float, gt_xywh)
    pred_area, gt_area = pw * ph, gw * gh
    ix = max(0.0, min(px + pw, gx + gw) - max(px, gx))
    iy = max(0.0, min(py + ph, gy + gh) - max(py, gy))
    intersection = ix * iy
    coverage = intersection / gt_area if gt_area > 0 else 0.0
    tightness = intersection / pred_area if pred_area > 0 else 0.0
    dice = 2 * coverage * tightness / (coverage + tightness) if intersection > 0 else 0.0
    ratio = pred_area / gt_area if gt_area > 0 else math.inf
    gt_fraction = gt_area / image_area
    tolerance = 1.25 + 0.35 * math.exp(-gt_fraction / 0.01)
    excess = max(0.0, abs(math.log(ratio)) - math.log(tolerance)) if ratio > 0 else math.inf
    score = dice ** 0.7 * math.exp(-4.0 * excess**2) if dice > 0 else 0.0
    return PairStats(score, coverage, tightness, dice, ratio)


def _prediction_boxes(record: PredictionRecord | None) -> list[PredictionBox]:
    return record.boxes if record is not None else []


def _rectangle_union_area(rectangles: list[tuple[float, float, float, float]]) -> float:
    """Exact union area for axis-aligned xywh rectangles."""
    if not rectangles:
        return 0.0
    x_edges = sorted({x for x, _, w, _ in rectangles for x in (x, x + w)})
    area = 0.0
    for left, right in zip(x_edges, x_edges[1:]):
        if right <= left:
            continue
        intervals = sorted(
            (y, y + h) for x, y, w, h in rectangles
            if x < right and x + w > left and h > 0
        )
        covered_y = 0.0
        if intervals:
            start, end = intervals[0]
            for next_start, next_end in intervals[1:]:
                if next_start > end:
                    covered_y += end - start
                    start, end = next_start, next_end
                else:
                    end = max(end, next_end)
            covered_y += end - start
        area += (right - left) * covered_y
    return area


def _group_pair_stats(
    pred_xywh: tuple[float, float, float, float],
    gt_boxes: list[tuple[int, GTBox]],
    image_area: float,
) -> list[tuple[int, PairStats]]:
    """Score one prediction against every GT it owns using their union as its target area."""
    px, py, pw, ph = pred_xywh
    pred_area = pw * ph
    gt_rectangles = [gt.bbox_xywh for _, gt in gt_boxes]
    union_area = _rectangle_union_area(gt_rectangles)
    clipped: list[tuple[float, float, float, float]] = []
    intersections: dict[int, float] = {}
    for gt_idx, gt in gt_boxes:
        gx, gy, gw, gh = gt.bbox_xywh
        left, top = max(px, gx), max(py, gy)
        right, bottom = min(px + pw, gx + gw), min(py + ph, gy + gh)
        width, height = max(0.0, right - left), max(0.0, bottom - top)
        intersections[gt_idx] = width * height
        if width and height:
            clipped.append((left, top, width, height))
    union_intersection = _rectangle_union_area(clipped)
    tightness = union_intersection / pred_area if pred_area > 0 else 0.0
    ratio = pred_area / union_area if union_area > 0 else math.inf
    gt_fraction = union_area / image_area
    base_tolerance = 1.25 + 0.35 * math.exp(-gt_fraction / 0.01)
    tolerance = base_tolerance * math.sqrt(len(gt_boxes))
    excess = max(0.0, abs(math.log(ratio)) - math.log(tolerance)) if ratio > 0 else math.inf
    size_penalty = math.exp(-4.0 * excess**2)
    output = []
    for gt_idx, gt in gt_boxes:
        gt_area = gt.bbox_xywh[2] * gt.bbox_xywh[3]
        coverage = intersections[gt_idx] / gt_area if gt_area > 0 else 0.0
        dice = 2 * coverage * tightness / (coverage + tightness) if coverage and tightness else 0.0
        score = dice ** 0.7 * size_penalty if dice > 0 else 0.0
        output.append((gt_idx, PairStats(score, coverage, tightness, dice, ratio)))
    return output


def evaluate_sample(sample: Sample, prediction: PredictionRecord | None) -> dict:
    preds = _prediction_boxes(prediction)
    gts = sample.boxes
    m, n = len(gts), len(preds)
    image_area = sample.image_width * sample.image_height
    matrix = np.zeros((n, m), dtype=float)
    pair_details: list[list[PairStats]] = []
    for pred_idx, pred in enumerate(preds):
        absolute = pred.absolute_xywh(sample.image_width, sample.image_height)
        row = []
        for gt_idx, gt in enumerate(gts):
            stats = box_pair_stats(absolute, gt.bbox_xywh, image_area)
            matrix[pred_idx, gt_idx] = stats.score
            row.append(stats)
        pair_details.append(row)

    # Each GT chooses one owner. A prediction may own several GTs, which permits a
    # single region to describe a merged anomaly while still rejecting duplicates.
    owned_by_prediction: list[list[int]] = [[] for _ in range(n)]
    for gt_idx in range(m):
        candidates = [pred_idx for pred_idx in range(n) if pair_details[pred_idx][gt_idx].coverage > 0]
        if candidates:
            owner = max(
                candidates,
                key=lambda pred_idx: (
                    pair_details[pred_idx][gt_idx].coverage,
                    pair_details[pred_idx][gt_idx].score,
                    -pred_idx,
                ),
            )
            owned_by_prediction[owner].append(gt_idx)

    meaningful: list[tuple[int, int, PairStats]] = []
    for pred_idx, gt_indices in enumerate(owned_by_prediction):
        if not gt_indices:
            continue
        absolute = preds[pred_idx].absolute_xywh(sample.image_width, sample.image_height)
        group = [(gt_idx, gts[gt_idx]) for gt_idx in gt_indices]
        meaningful.extend(
            (pred_idx, gt_idx, stats)
            for gt_idx, stats in _group_pair_stats(absolute, group, image_area)
            if stats.score > 0
        )
    q = sum(item.score for _, _, item in meaningful)
    gt_scores = [0.0] * m
    gt_pairs: list[PairStats | None] = [None] * m
    for _, gt_idx, pair in meaningful:
        gt_scores[gt_idx] = pair.score
        gt_pairs[gt_idx] = pair
    matched_preds = {pred_idx for pred_idx, _, _ in meaningful}
    matched_gts = {gt_idx for _, gt_idx, _ in meaningful}
    unmatched_preds = [idx for idx in range(n) if idx not in matched_preds]
    prediction_credits = [0.0] * n
    for pred_idx in matched_preds:
        owned_scores = [pair.score for owner, _, pair in meaningful if owner == pred_idx]
        prediction_credits[pred_idx] = float(np.mean(owned_scores))
    soft_recall = q / m if m else 0.0
    soft_precision = sum(prediction_credits) / n if n else 0.0
    image_score = (
        2 * soft_precision * soft_recall / (soft_precision + soft_recall)
        if m and soft_precision + soft_recall else (math.exp(-n) if not m else 0.0)
    )
    gt_hard_miss = [bool(not n or not np.any(matrix[:, gt_idx] > 0)) for gt_idx in range(m)]
    hard_miss = sum(gt_hard_miss)
    merge_preds = sum(len(gt_indices) >= 2 for gt_indices in owned_by_prediction)
    split_gts = sum(1 for gt_idx in range(m) if np.sum(matrix[:, gt_idx] > 0.1) >= 2)
    duplicate_fp = sum(1 for pred_idx in unmatched_preds if any(matrix[pred_idx, gt_idx] > 0.1 for gt_idx in matched_gts))
    background_fp = len(unmatched_preds) - duplicate_fp
    return {
        "sample_id": sample.sample_id,
        "m": m,
        "n": n,
        "q": q,
        "prediction_credit": sum(prediction_credits),
        "prediction_credits": prediction_credits,
        "soft_precision": soft_precision,
        "soft_recall": soft_recall,
        "image_score": image_score,
        "gt_scores": gt_scores,
        "gt_families": [gt.family for gt in gts],
        "gt_pairs": gt_pairs,
        "gt_hard_miss": gt_hard_miss,
        "hard_miss": hard_miss,
        "merge_predictions": merge_preds,
        "split_gts": split_gts,
        "duplicate_fp": duplicate_fp,
        "background_fp": background_fp,
        "matched": meaningful,
        "prediction_status": prediction.status if prediction else "missing",
    }


def _safe_mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _quantiles(values: list[float]) -> dict:
    if not values:
        return {"p25": None, "median": None, "p75": None}
    q = np.quantile(values, [0.25, 0.5, 0.75])
    return {"p25": float(q[0]), "median": float(q[1]), "p75": float(q[2])}


def _count_metrics(rows: list[dict]) -> dict:
    if not rows:
        return {"images": 0, "mae": None, "bias": None, "exact": None, "within_1": None, "within_2": None}
    diffs = np.array([row["n"] - row["m"] for row in rows], dtype=float)
    return {
        "images": len(rows),
        "mae": float(np.mean(np.abs(diffs))),
        "bias": float(np.mean(diffs)),
        "exact": float(np.mean(np.abs(diffs) == 0)),
        "within_1": float(np.mean(np.abs(diffs) <= 1)),
        "within_2": float(np.mean(np.abs(diffs) <= 2)),
    }


def _region_group_stats(rows: list[dict], family_prefixes: tuple[str, ...]) -> dict:
    """Aggregate ownership-v2 GT quality for one human-annotated error group."""
    weighted_score = 0.0
    total_weight = 0.0
    image_count = 0
    gt_count = 0
    for row in rows:
        selected = [
            score
            for score, family in zip(row["gt_scores"], row.get("gt_families", ()))
            if family.startswith(family_prefixes)
        ]
        if not selected:
            continue
        weight = math.sqrt(len(selected))
        weighted_score += weight * float(np.mean(selected))
        total_weight += weight
        image_count += 1
        gt_count += len(selected)
    return {
        "region_bbox_score": 100 * weighted_score / total_weight if total_weight else None,
        "images": image_count,
        "gt_boxes": gt_count,
    }


def _headline(rows: list[dict]) -> dict:
    positive = [r for r in rows if r["m"] > 0]
    clean = [r for r in rows if r["m"] == 0]
    pos_weight = sum(math.sqrt(r["m"]) for r in positive)
    positive_elastic = sum(math.sqrt(r["m"]) * r["image_score"] for r in positive) / pos_weight if pos_weight else 0.0
    clean_soft = _safe_mean([r["image_score"] for r in clean]) or 0.0
    # Weighted power mean with w=0.7 and r=0.5.  Keep the two components on
    # their native [0, 1] scale and convert only the final headline to [0, 100].
    main = (0.7 * math.sqrt(positive_elastic) + 0.3 * math.sqrt(clean_soft)) ** 2
    macro = _safe_mean([r["image_score"] for r in positive]) or 0.0
    q_sum = sum(r["q"] for r in positive)
    credit_sum = sum(r["prediction_credit"] for r in positive)
    m_sum, n_sum = sum(r["m"] for r in positive), sum(r["n"] for r in positive)
    micro_recall = q_sum / m_sum if m_sum else 0.0
    micro_precision = credit_sum / n_sum if n_sum else 0.0
    micro = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if micro_precision + micro_recall else 0.0
    )
    artifact = _region_group_stats(rows, REGION_GROUPS["artifact"])
    text = _region_group_stats(rows, REGION_GROUPS["text_correctness"])
    return {
        "elastic_bbox_score": 100 * main,
        "artifact_region_bbox_score": artifact["region_bbox_score"],
        "text_region_bbox_score": text["region_bbox_score"],
        "positive_elastic": positive_elastic,
        "clean_soft": clean_soft,
        "clean_specificity": _safe_mean([float(r["n"] == 0) for r in clean]),
        "clean_fppi": _safe_mean([float(r["n"]) for r in clean]),
        "positive_macro": macro,
        "positive_micro": micro,
        "positive_micro_precision": micro_precision,
        "positive_micro_recall": micro_recall,
    }


def _gt_bucket(samples: list[Sample], rows: list[dict], selector) -> dict:
    scores: list[float] = []
    pairs: list[PairStats] = []
    hard = 0
    for sample, row in zip(samples, rows):
        for idx, gt in enumerate(sample.boxes):
            if selector(sample, gt, idx):
                score = row["gt_scores"][idx]
                scores.append(score)
                if row["gt_pairs"][idx] is not None and score >= 0.1:
                    pairs.append(row["gt_pairs"][idx])
                if row["gt_hard_miss"][idx]:
                    hard += 1
    return {
        "gt": len(scores),
        "recall_0_1": _safe_mean([float(s >= 0.1) for s in scores]),
        "recall_0_5": _safe_mean([float(s >= 0.5) for s in scores]),
        "recall_0_75": _safe_mean([float(s >= 0.75) for s in scores]),
        "found_quality": _safe_mean([p.score for p in pairs]),
        "mean_coverage": _safe_mean([p.coverage for p in pairs]),
        "mean_tightness": _safe_mean([p.tightness for p in pairs]),
        "scale_bias": float(np.median([math.log(p.area_ratio) for p in pairs])) if pairs else None,
        "hard_miss_rate": hard / len(scores) if scores else None,
    }


def _annotation_source(sample: Sample) -> str:
    name = sample.source_file.lower()
    if "natural_clean_110" in name:
        return "natural_clean_110"
    if "gpt_completed_" in name:
        return "gpt"
    if "u1.5_completed_" in name:
        return "u1_5"
    if "external4_completed_" in name:
        return "external4"
    return "other"


def _source_bucket(samples: list[Sample], rows: list[dict], source: str) -> dict:
    selected = [(sample, row) for sample, row in zip(samples, rows) if _annotation_source(sample) == source]
    subset_samples = [sample for sample, _ in selected]
    subset_rows = [row for _, row in selected]
    gt_stats = _gt_bucket(
        samples,
        rows,
        lambda sample, gt, index: _annotation_source(sample) == source,
    )
    headline = _headline(subset_rows)
    counts = _count_metrics(subset_rows)
    return {
        "images": len(subset_samples),
        "positive_images": sum(bool(sample.boxes) for sample in subset_samples),
        "clean_images": sum(not sample.boxes for sample in subset_samples),
        "gt": sum(len(sample.boxes) for sample in subset_samples),
        "elastic_bbox_score": headline["elastic_bbox_score"],
        "positive_elastic": headline["positive_elastic"],
        "clean_soft": headline["clean_soft"],
        "clean_specificity": headline["clean_specificity"],
        "clean_fppi": headline["clean_fppi"],
        "recall_0_1": gt_stats["recall_0_1"],
        "recall_0_5": gt_stats["recall_0_5"],
        "found_quality": gt_stats["found_quality"],
        "hard_miss_rate": gt_stats["hard_miss_rate"],
        "count_mae": counts["mae"],
        "count_bias": counts["bias"],
        "background_fp": sum(row["background_fp"] for row in subset_rows),
    }


def compute_metrics(samples: list[Sample], predictions: dict[str, PredictionRecord], bootstrap: int = 0, seed: int = 20260804) -> tuple[dict, list[dict]]:
    rows = [evaluate_sample(sample, predictions.get(sample.sample_id)) for sample in samples]
    headline = _headline(rows)
    all_scores = [score for row in rows for score in row["gt_scores"]]
    found_pairs = [pair for row in rows for _, _, pair in row["matched"] if pair.score >= 0.1]
    area_ratios = [pair.area_ratio for pair in found_pairs]
    scale_bins = {
        "under_0_5": _safe_mean([float(r < 0.5) for r in area_ratios]),
        "0_5_to_0_8": _safe_mean([float(0.5 <= r < 0.8) for r in area_ratios]),
        "0_8_to_1_25": _safe_mean([float(0.8 <= r <= 1.25) for r in area_ratios]),
        "1_25_to_2": _safe_mean([float(1.25 < r <= 2) for r in area_ratios]),
        "over_2": _safe_mean([float(r > 2) for r in area_ratios]),
    }
    matched_family = []
    for sample, row in zip(samples, rows):
        record = predictions.get(sample.sample_id)
        if not record:
            continue
        for pred_idx, gt_idx, pair in row["matched"]:
            if pair.score >= 0.1 and record.boxes[pred_idx].family is not None:
                matched_family.append(record.boxes[pred_idx].family == sample.boxes[gt_idx].family)
    result = {
        "metric_version": METRIC_VERSION,
        "headline": headline,
        "region_group_scores": {
            "version": REGION_GROUP_SCORE_VERSION,
            "artifact": _region_group_stats(rows, REGION_GROUPS["artifact"]),
            "text_correctness": _region_group_stats(rows, REGION_GROUPS["text_correctness"]),
        },
        "dataset": {
            "images": len(samples), "positive_images": sum(bool(s.boxes) for s in samples), "clean_images": sum(not s.boxes for s in samples),
            "gt_boxes": sum(len(s.boxes) for s in samples), "predicted_boxes": sum(r["n"] for r in rows),
            "failed_or_missing_predictions": sum(r["prediction_status"] != "ok" for r in rows),
        },
        "detection": {
            "hard_miss": sum(r["hard_miss"] for r in rows),
            "effective_miss": sum(s < 0.1 for s in all_scores),
            "recall_0_1": _safe_mean([float(s >= 0.1) for s in all_scores]),
            "recall_0_5": _safe_mean([float(s >= 0.5) for s in all_scores]),
            "recall_0_75": _safe_mean([float(s >= 0.75) for s in all_scores]),
            "found_quality": _safe_mean([p.score for p in found_pairs]),
            "found_score_quantiles": _quantiles([p.score for p in found_pairs]),
            "mean_coverage": _safe_mean([p.coverage for p in found_pairs]),
            "mean_tightness": _safe_mean([p.tightness for p in found_pairs]),
        },
        "counts": {
            "all": _count_metrics(rows),
            "positive": _count_metrics([r for r in rows if r["m"] > 0]),
            "clean": _count_metrics([r for r in rows if r["m"] == 0]),
        },
        "binary_decision": {
            # Image-level yes/no: at least one predicted box means error present.
            "positive_false_negative_rate": _safe_mean([
                float(r["n"] == 0) for r in rows if r["m"] > 0
            ]),
            "clean_false_positive_rate": _safe_mean([
                float(r["n"] > 0) for r in rows if r["m"] == 0
            ]),
            "overall_error_rate": _safe_mean([
                float((r["m"] > 0) != (r["n"] > 0)) for r in rows
            ]),
            "positive_false_negatives": sum(
                r["m"] > 0 and r["n"] == 0 for r in rows
            ),
            "clean_false_positives": sum(
                r["m"] == 0 and r["n"] > 0 for r in rows
            ),
            "total_errors": sum(
                (r["m"] > 0) != (r["n"] > 0) for r in rows
            ),
        },
        "scale": {
            "area_ratio": _quantiles(area_ratios),
            "scale_bias": float(np.median([math.log(r) for r in area_ratios])) if area_ratios else None,
            "bins": scale_bins,
        },
        "multi_box": {
            "merge_predictions": sum(r["merge_predictions"] for r in rows),
            "merge_rate": sum(r["merge_predictions"] for r in rows) / sum(r["n"] for r in rows if r["m"] > 0) if sum(r["n"] for r in rows if r["m"] > 0) else 0.0,
            "split_gts": sum(r["split_gts"] for r in rows),
            "split_rate": sum(r["split_gts"] for r in rows) / sum(r["m"] for r in rows) if sum(r["m"] for r in rows) else 0.0,
            "duplicate_fp": sum(r["duplicate_fp"] for r in rows),
            "background_fp": sum(r["background_fp"] for r in rows),
            "duplicate_fp_rate": sum(r["duplicate_fp"] for r in rows) / sum(r["duplicate_fp"] + r["background_fp"] for r in rows) if sum(r["duplicate_fp"] + r["background_fp"] for r in rows) else 0.0,
        },
        "family_accuracy_on_found": _safe_mean([float(x) for x in matched_family]),
    }

    difficulty_defs = {
        "single": lambda m: m == 1, "light": lambda m: m == 2,
        "medium": lambda m: 3 <= m <= 4, "dense": lambda m: m >= 5,
    }
    result["buckets_by_gt_count"] = {}
    for name, selector in difficulty_defs.items():
        subset = [row for row in rows if selector(row["m"])]
        result["buckets_by_gt_count"][name] = {**_headline(subset), **_count_metrics(subset), "images": len(subset), "gt": sum(r["m"] for r in subset)}
    area_defs = {
        "tiny": lambda a: a < 0.001, "small": lambda a: 0.001 <= a < 0.005,
        "medium": lambda a: 0.005 <= a <= 0.02, "large": lambda a: a > 0.02,
    }
    result["buckets_by_area"] = {name: _gt_bucket(samples, rows, lambda s, g, i, f=selector: f(g.bbox_xywh[2] * g.bbox_xywh[3] / (s.image_width * s.image_height))) for name, selector in area_defs.items()}
    families = sorted({gt.family for sample in samples for gt in sample.boxes})
    result["buckets_by_family"] = {family: _gt_bucket(samples, rows, lambda s, g, i, target=family: g.family == target) for family in families}
    sources = [
        name for name in ("gpt", "u1_5", "external4", "natural_clean_110", "other")
        if any(_annotation_source(sample) == name for sample in samples)
    ]
    result["buckets_by_source"] = {name: _source_bucket(samples, rows, name) for name in sources}

    if bootstrap:
        rng = np.random.default_rng(seed)
        pos_indices = [i for i, r in enumerate(rows) if r["m"] > 0]
        clean_indices = [i for i, r in enumerate(rows) if r["m"] == 0]
        draws: dict[str, list[float]] = {k: [] for k in (
            "elastic_bbox_score", "artifact_region_bbox_score", "text_region_bbox_score",
            "positive_elastic", "clean_soft", "positive_macro", "positive_micro",
        )}
        for _ in range(bootstrap):
            indices = list(rng.choice(pos_indices, len(pos_indices), replace=True)) + list(rng.choice(clean_indices, len(clean_indices), replace=True))
            h = _headline([rows[int(i)] for i in indices])
            for key in draws:
                if h[key] is not None:
                    draws[key].append(h[key])
        result["bootstrap"] = {
            "iterations": bootstrap,
            "seed": seed,
            "ci95": {
                key: [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]
                if values else []
                for key, values in draws.items()
            },
        }
    return result, rows


def paired_bootstrap(rows_by_model: dict[str, list[dict]], iterations: int = 2000, seed: int = 20260804) -> dict:
    """Paired, positive/clean-stratified confidence intervals for model deltas."""
    names = sorted(rows_by_model)
    if len(names) < 2 or iterations <= 0:
        return {"iterations": iterations, "seed": seed, "pairs": {}}
    lengths = {len(rows_by_model[name]) for name in names}
    if len(lengths) != 1:
        raise ValueError("paired bootstrap requires aligned model rows")
    reference = rows_by_model[names[0]]
    pos_indices = [i for i, row in enumerate(reference) if row["m"] > 0]
    clean_indices = [i for i, row in enumerate(reference) if row["m"] == 0]
    rng = np.random.default_rng(seed)
    keys = (
        "elastic_bbox_score", "artifact_region_bbox_score", "text_region_bbox_score",
        "positive_elastic", "clean_soft", "positive_macro", "positive_micro",
    )
    draws = [list(rng.choice(pos_indices, len(pos_indices), replace=True)) + list(rng.choice(clean_indices, len(clean_indices), replace=True)) for _ in range(iterations)]
    output: dict[str, dict] = {}
    for left_idx, left in enumerate(names):
        for right in names[left_idx + 1 :]:
            differences = {key: [] for key in keys}
            for indices in draws:
                left_h = _headline([rows_by_model[left][int(i)] for i in indices])
                right_h = _headline([rows_by_model[right][int(i)] for i in indices])
                for key in keys:
                    if left_h[key] is not None and right_h[key] is not None:
                        differences[key].append(left_h[key] - right_h[key])
            output[f"{left}_minus_{right}"] = {
                key: {
                    "mean_delta": float(np.mean(values)),
                    "ci95": [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))],
                    "probability_gt_0": float(np.mean(np.asarray(values) > 0)),
                }
                for key, values in differences.items() if values
            }
    return {"iterations": iterations, "seed": seed, "pairs": output}
