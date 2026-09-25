#!/usr/bin/env python3
"""Standalone, open-sourceable bbox scorer for the image-anomaly-critic benchmark.

Single file, standard library only. Scores model-predicted bounding boxes
against human ground-truth boxes with the metric version
``merged-ground-truth-ownership-v2``.

Inputs
------
Annotations (JSONL, one image per line):
    {"sample_id": str, "image_absolute_path": str, "image_width": int,
     "image_height": int, "prompt": str,
     "annotation": {"no_error": bool, "boxes": [
         {"bbox_xywh": [x, y, w, h],            # absolute pixels, top-left + size
          "error_type": {"code": FAMILY},        # one of FAMILIES
          "detail": str}]}}
Predictions (JSONL, one image per line):
    {"sample_id": str, "model": str, "status": "ok"|"error"|"missing",
     "boxes": [{"bbox_1000_xywh": [x, y, w, h],  # normalized to 0..1000
                "family": str|null, "description": str}],
     "no_error": bool, ...}

Usage
-----
    python bbox_scorer.py \
        --annotations gt1.jsonl gt2.jsonl \
        --predictions preds.jsonl \
        --output metrics.json \
        --per-sample-out per_sample.json \
        --bootstrap 2000 --seed 20260804
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

METRIC_VERSION = "merged-ground-truth-ownership-v2"
REGION_GROUP_SCORE_VERSION = "ownership-v2-gt-family-diagnostic-v1"

FAMILIES = (
    "D1_Artifact/A1_RenderingArtifact",
    "D1_Artifact/A2_StructuralArtifact",
    "D1_Artifact/A3_TopologyArtifact",
    "D1_Artifact/A4_RelationalAndPhysicalArtifact",
    "D2_TextCorrectness/A1_TextVisualIntegrity",
    "D2_TextCorrectness/A2_TextContentFidelity",
)
REGION_GROUPS = {
    "artifact": ("D1_Artifact/",),
    "text_correctness": ("D2_TextCorrectness/",),
}

# Area-penalty constants (see README section "面积惩罚函数").
TAU_BASE = 1.25          # asymptotic tolerance for very large GT boxes
TAU_SMALL_EXCESS = 0.35  # extra tolerance for infinitesimally small GT boxes
TAU_DECAY_FRACTION = 0.01  # decay scale: 1% of the full image area
SIZE_PENALTY_K = 4.0     # Gaussian width in log-space
DICE_EXPONENT = 0.7      # score = dice ** 0.7 * size_penalty


class DataValidationError(ValueError):
    pass


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class GTBox:
    bbox_xywh: tuple[float, float, float, float]  # absolute pixels
    family: str
    detail: str = ""


@dataclass(frozen=True)
class Sample:
    sample_id: str
    image_path: str
    image_width: int
    image_height: int
    prompt: str
    no_error: bool
    boxes: tuple[GTBox, ...]


@dataclass(frozen=True)
class PredictionBox:
    bbox_1000_xywh: tuple[float, float, float, float]  # 0..1000 normalized
    family: str | None = None
    description: str = ""

    def absolute_xywh(self, width: int, height: int) -> tuple[float, float, float, float]:
        x, y, w, h = self.bbox_1000_xywh
        return x * width / 1000, y * height / 1000, w * width / 1000, h * height / 1000


@dataclass
class PredictionRecord:
    sample_id: str
    model: str
    status: str
    boxes: list[PredictionBox]
    no_error: bool
    error: str | None = None

    @classmethod
    def from_dict(cls, payload: dict) -> "PredictionRecord":
        boxes = [
            PredictionBox(tuple(b["bbox_1000_xywh"]), b.get("family"), b.get("description", ""))
            for b in payload.get("boxes", [])
        ]
        return cls(
            sample_id=payload["sample_id"],
            model=payload.get("model", ""),
            status=payload.get("status", "ok"),
            boxes=boxes,
            no_error=bool(payload.get("no_error", not boxes)),
            error=payload.get("error"),
        )


# --------------------------------------------------------------------------- #
# Loading and validation
# --------------------------------------------------------------------------- #

def _validate_bbox(value: object, width: int, height: int, where: str) -> tuple[float, float, float, float]:
    if not isinstance(value, list) or len(value) != 4:
        raise DataValidationError(f"{where}: bbox_xywh must contain four numbers")
    try:
        x, y, w, h = (float(v) for v in value)
    except (TypeError, ValueError) as exc:
        raise DataValidationError(f"{where}: bbox_xywh contains a non-number") from exc
    if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > width + 0.01 or y + h > height + 0.01:
        raise DataValidationError(f"{where}: bbox_xywh is outside the {width}x{height} image")
    return x, y, w, h


def load_annotations(paths: Iterable[str | Path]) -> list[Sample]:
    samples: list[Sample] = []
    seen: dict[str, str] = {}
    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise DataValidationError(f"annotation file not found: {path}")
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                where = f"{path}:{line_no}"
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise DataValidationError(f"{where}: invalid JSON: {exc}") from exc
                sample_id = str(row.get("sample_id", "")).strip()
                if not sample_id:
                    raise DataValidationError(f"{where}: missing sample_id")
                if sample_id in seen:
                    raise DataValidationError(f"{where}: duplicate sample_id {sample_id!r}; first seen at {seen[sample_id]}")
                seen[sample_id] = where
                try:
                    width, height = int(row["image_width"]), int(row["image_height"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise DataValidationError(f"{where}: invalid image dimensions") from exc
                annotation = row.get("annotation") or {}
                no_error = annotation.get("no_error")
                if not isinstance(no_error, bool):
                    raise DataValidationError(f"{where}: annotation.no_error must be boolean")
                raw_boxes = annotation.get("boxes")
                if not isinstance(raw_boxes, list):
                    raise DataValidationError(f"{where}: annotation.boxes must be a list")
                if no_error != (len(raw_boxes) == 0):
                    raise DataValidationError(f"{where}: no_error is inconsistent with boxes")
                boxes = []
                for idx, box in enumerate(raw_boxes):
                    box_where = f"{where}:box[{idx}]"
                    bbox = _validate_bbox(box.get("bbox_xywh"), width, height, box_where)
                    family = str((box.get("error_type") or {}).get("code", ""))
                    if family not in FAMILIES:
                        raise DataValidationError(f"{box_where}: unsupported primary family {family!r}")
                    boxes.append(GTBox(bbox, family, str(box.get("detail", ""))))
                samples.append(Sample(
                    sample_id, str(row.get("image_absolute_path", "")), width, height,
                    str(row.get("prompt") or ""), no_error, tuple(boxes),
                ))
    if not samples:
        raise DataValidationError("no annotation records were loaded")
    return samples


def load_predictions(path: str | Path) -> dict[str, PredictionRecord]:
    records: dict[str, PredictionRecord] = {}
    file_path = Path(path)
    if not file_path.exists():
        return records
    with file_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            payload = json.loads(line)
            record = PredictionRecord.from_dict(payload)
            if record.sample_id in records:
                raise DataValidationError(f"{file_path}:{line_no}: duplicate prediction for {record.sample_id}")
            records[record.sample_id] = record
    return records


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class PairStats:
    score: float
    coverage: float
    tightness: float
    dice: float
    area_ratio: float


def area_tolerance(gt_fraction: float) -> float:
    """Tolerance tau(f) on the area ratio, f = GT area / image area.

    tau(f) = 1.25 + 0.35 * exp(-f / 0.01): a smooth exponential decay between
    1.60 (infinitesimally small GT) and 1.25 (GT covering the whole image),
    with a decay scale of 1% of the image area. NOT piecewise interpolation.
    """
    return TAU_BASE + TAU_SMALL_EXCESS * math.exp(-gt_fraction / TAU_DECAY_FRACTION)


def size_penalty(excess: float) -> float:
    """psi = exp(-4 * e^2), a Gaussian falloff in log-ratio space."""
    return math.exp(-SIZE_PENALTY_K * excess * excess)


def box_pair_stats(pred_xywh: Sequence[float], gt_xywh: Sequence[float], image_area: float) -> PairStats:
    """Base quality of one prediction against one GT (1-to-1, pre-ownership)."""
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
    tolerance = area_tolerance(gt_fraction)
    excess = max(0.0, abs(math.log(ratio)) - math.log(tolerance)) if ratio > 0 else math.inf
    score = dice ** DICE_EXPONENT * size_penalty(excess) if dice > 0 else 0.0
    return PairStats(score, coverage, tightness, dice, ratio)


def rectangle_union_area(rectangles: list[tuple[float, float, float, float]]) -> float:
    """Exact union area of axis-aligned xywh rectangles (coordinate sweep)."""
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


def group_pair_stats(
    pred_xywh: tuple[float, float, float, float],
    gt_boxes: list[tuple[int, GTBox]],
    image_area: float,
) -> list[tuple[int, PairStats]]:
    """Final quality of one prediction against every GT it owns.

    The owned GTs' union U is the prediction's target area: tightness and the
    area ratio are computed against U, and the tolerance is widened by sqrt(K).
    Each GT keeps its own coverage (intersection with that GT / GT area).
    """
    px, py, pw, ph = pred_xywh
    pred_area = pw * ph
    gt_rectangles = [gt.bbox_xywh for _, gt in gt_boxes]
    union_area = rectangle_union_area(gt_rectangles)
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
    union_intersection = rectangle_union_area(clipped)
    tightness = union_intersection / pred_area if pred_area > 0 else 0.0
    ratio = pred_area / union_area if union_area > 0 else math.inf
    gt_fraction = union_area / image_area
    tolerance = area_tolerance(gt_fraction) * math.sqrt(len(gt_boxes))
    excess = max(0.0, abs(math.log(ratio)) - math.log(tolerance)) if ratio > 0 else math.inf
    penalty = size_penalty(excess)
    output = []
    for gt_idx, gt in gt_boxes:
        gx, gy, gw, gh = gt.bbox_xywh
        gt_area = gw * gh
        coverage = intersections[gt_idx] / gt_area if gt_area > 0 else 0.0
        dice = 2 * coverage * tightness / (coverage + tightness) if coverage and tightness else 0.0
        score = dice ** DICE_EXPONENT * penalty if dice > 0 else 0.0
        output.append((gt_idx, PairStats(score, coverage, tightness, dice, ratio)))
    return output


# --------------------------------------------------------------------------- #
# Per-sample evaluation
# --------------------------------------------------------------------------- #

def evaluate_sample(sample: Sample, prediction: PredictionRecord | None) -> dict:
    """Score one image. Missing/failed predictions are treated as empty."""
    preds = prediction.boxes if prediction is not None else []
    gts = sample.boxes
    m, n = len(gts), len(preds)
    image_area = sample.image_width * sample.image_height

    # Base 1-to-1 matrix (pre-ownership), used for owner election and diagnostics.
    matrix: list[list[float]] = [[0.0] * m for _ in range(n)]
    pair_details: list[list[PairStats]] = []
    for pred_idx, pred in enumerate(preds):
        absolute = pred.absolute_xywh(sample.image_width, sample.image_height)
        row = []
        for gt_idx, gt in enumerate(gts):
            stats = box_pair_stats(absolute, gt.bbox_xywh, image_area)
            matrix[pred_idx][gt_idx] = stats.score
            row.append(stats)
        pair_details.append(row)

    # --- Ownership: each GT elects exactly one owning prediction. ---
    # Candidates = predictions with positive coverage on that GT.
    # Tie-break: (1) higher coverage, (2) higher base 1-to-1 score,
    #            (3) earlier position in the model's output order.
    owned_by_prediction: list[list[int]] = [[] for _ in range(n)]
    for gt_idx in range(m):
        candidates = [p for p in range(n) if pair_details[p][gt_idx].coverage > 0]
        if candidates:
            owner = max(
                candidates,
                key=lambda p: (
                    pair_details[p][gt_idx].coverage,
                    pair_details[p][gt_idx].score,
                    -p,
                ),
            )
            owned_by_prediction[owner].append(gt_idx)

    # --- Final quality: merged-group recomputation for owned GTs. ---
    meaningful: list[tuple[int, int, PairStats]] = []  # (pred_idx, gt_idx, stats)
    for pred_idx, gt_indices in enumerate(owned_by_prediction):
        if not gt_indices:
            continue
        absolute = preds[pred_idx].absolute_xywh(sample.image_width, sample.image_height)
        group = [(gt_idx, gts[gt_idx]) for gt_idx in gt_indices]
        meaningful.extend(
            (pred_idx, gt_idx, stats)
            for gt_idx, stats in group_pair_stats(absolute, group, image_area)
            if stats.score > 0
        )
    q = sum(item.score for _, _, item in meaningful)
    gt_scores = [0.0] * m
    for _, gt_idx, pair in meaningful:
        gt_scores[gt_idx] = pair.score
    matched_preds = {p for p, _, _ in meaningful}
    matched_gts = {g for _, g, _ in meaningful}
    unmatched_preds = [p for p in range(n) if p not in matched_preds]

    # Prediction credit: a matched prediction earns the mean of its owned GT
    # scores; unmatched (duplicate/background) predictions earn 0.
    prediction_credits = [0.0] * n
    for pred_idx in matched_preds:
        owned = [pair.score for owner, _, pair in meaningful if owner == pred_idx]
        prediction_credits[pred_idx] = sum(owned) / len(owned)

    soft_recall = q / m if m else 0.0
    soft_precision = sum(prediction_credits) / n if n else 0.0
    image_score = (
        2 * soft_precision * soft_recall / (soft_precision + soft_recall)
        if m and soft_precision + soft_recall
        else (math.exp(-n) if not m else 0.0)
    )

    hard_miss = sum(1 for g in range(m) if not any(matrix[p][g] > 0 for p in range(n)))
    merge_predictions = sum(1 for owned in owned_by_prediction if len(owned) >= 2)
    split_gts = sum(1 for g in range(m) if sum(1 for p in range(n) if matrix[p][g] > 0.1) >= 2)
    duplicate_fp = sum(
        1 for p in unmatched_preds
        if any(matrix[p][g] > 0.1 for g in matched_gts)
    )
    background_fp = len(unmatched_preds) - duplicate_fp

    return {
        "sample_id": sample.sample_id,
        "m": m,
        "n": n,
        "q": q,
        "prediction_credit": sum(prediction_credits),
        "soft_precision": soft_precision,
        "soft_recall": soft_recall,
        "image_score": image_score,
        "gt_scores": gt_scores,
        "gt_families": [gt.family for gt in gts],
        "hard_miss": hard_miss,
        "merge_predictions": merge_predictions,
        "split_gts": split_gts,
        "duplicate_fp": duplicate_fp,
        "background_fp": background_fp,
        "prediction_status": prediction.status if prediction else "missing",
    }


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #

def _safe_mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _count_metrics(rows: list[dict]) -> dict:
    if not rows:
        return {"images": 0, "mae": None, "bias": None, "exact": None, "within_1": None, "within_2": None}
    diffs = [row["n"] - row["m"] for row in rows]
    return {
        "images": len(rows),
        "mae": sum(abs(d) for d in diffs) / len(diffs),
        "bias": sum(diffs) / len(diffs),
        "exact": sum(1 for d in diffs if d == 0) / len(diffs),
        "within_1": sum(1 for d in diffs if abs(d) <= 1) / len(diffs),
        "within_2": sum(1 for d in diffs if abs(d) <= 2) / len(diffs),
    }


def _region_group_stats(rows: list[dict], family_prefixes: tuple[str, ...]) -> dict:
    """GT-family diagnostic split (artifact vs text): per-image mean of the
    selected GT scores, weighted by sqrt(K_i) where K_i is the number of
    selected GTs in that image."""
    weighted_score = 0.0
    total_weight = 0.0
    image_count = 0
    gt_count = 0
    for row in rows:
        selected = [
            score for score, family in zip(row["gt_scores"], row["gt_families"])
            if family.startswith(family_prefixes)
        ]
        if not selected:
            continue
        weight = math.sqrt(len(selected))
        weighted_score += weight * (sum(selected) / len(selected))
        total_weight += weight
        image_count += 1
        gt_count += len(selected)
    return {
        "region_bbox_score": 100 * weighted_score / total_weight if total_weight else None,
        "images": image_count,
        "gt_boxes": gt_count,
    }


def headline(rows: list[dict]) -> dict:
    positive = [r for r in rows if r["m"] > 0]
    clean = [r for r in rows if r["m"] == 0]
    pos_weight = sum(math.sqrt(r["m"]) for r in positive)
    positive_elastic = (
        sum(math.sqrt(r["m"]) * r["image_score"] for r in positive) / pos_weight
        if pos_weight else 0.0
    )
    clean_soft = _safe_mean([r["image_score"] for r in clean]) or 0.0
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
    return {
        "elastic_bbox_score": 100 * main,
        "artifact_region_bbox_score": _region_group_stats(rows, REGION_GROUPS["artifact"])["region_bbox_score"],
        "text_region_bbox_score": _region_group_stats(rows, REGION_GROUPS["text_correctness"])["region_bbox_score"],
        "positive_elastic": positive_elastic,
        "clean_soft": clean_soft,
        "clean_specificity": _safe_mean([float(r["n"] == 0) for r in clean]),
        "clean_fppi": _safe_mean([float(r["n"]) for r in clean]),
        "positive_macro": macro,
        "positive_micro": micro,
        "positive_micro_precision": micro_precision,
        "positive_micro_recall": micro_recall,
    }


def compute_metrics(
    samples: list[Sample],
    predictions: dict[str, PredictionRecord],
    bootstrap: int = 2000,
    seed: int = 20260804,
) -> tuple[dict, list[dict]]:
    rows = [evaluate_sample(s, predictions.get(s.sample_id)) for s in samples]
    h = headline(rows)
    all_scores = [s for row in rows for s in row["gt_scores"]]
    found_scores = [s for s in all_scores if s >= 0.1]

    result = {
        "metric_version": METRIC_VERSION,
        "headline": h,
        "region_group_scores": {
            "version": REGION_GROUP_SCORE_VERSION,
            "artifact": _region_group_stats(rows, REGION_GROUPS["artifact"]),
            "text_correctness": _region_group_stats(rows, REGION_GROUPS["text_correctness"]),
        },
        "dataset": {
            "images": len(samples),
            "positive_images": sum(bool(s.boxes) for s in samples),
            "clean_images": sum(not s.boxes for s in samples),
            "gt_boxes": sum(len(s.boxes) for s in samples),
            "predicted_boxes": sum(r["n"] for r in rows),
            "failed_or_missing_predictions": sum(r["prediction_status"] != "ok" for r in rows),
        },
        "detection": {
            "hard_miss": sum(r["hard_miss"] for r in rows),
            "effective_miss": sum(1 for s in all_scores if s < 0.1),
            "recall_0_1": _safe_mean([float(s >= 0.1) for s in all_scores]),
            "recall_0_5": _safe_mean([float(s >= 0.5) for s in all_scores]),
            "recall_0_75": _safe_mean([float(s >= 0.75) for s in all_scores]),
            "found_quality": _safe_mean(found_scores),
        },
        "counts": {
            "all": _count_metrics(rows),
            "positive": _count_metrics([r for r in rows if r["m"] > 0]),
            "clean": _count_metrics([r for r in rows if r["m"] == 0]),
        },
        "binary_decision": {
            "positive_false_negative_rate": _safe_mean([
                float(r["n"] == 0) for r in rows if r["m"] > 0
            ]),
            "clean_false_positive_rate": _safe_mean([
                float(r["n"] > 0) for r in rows if r["m"] == 0
            ]),
            "overall_error_rate": _safe_mean([
                float((r["m"] > 0) != (r["n"] > 0)) for r in rows
            ]),
        },
        "multi_box": {
            "merge_predictions": sum(r["merge_predictions"] for r in rows),
            "split_gts": sum(r["split_gts"] for r in rows),
            "duplicate_fp": sum(r["duplicate_fp"] for r in rows),
            "background_fp": sum(r["background_fp"] for r in rows),
        },
    }

    if bootstrap:
        rng = random.Random(seed)
        pos_indices = [i for i, r in enumerate(rows) if r["m"] > 0]
        clean_indices = [i for i, r in enumerate(rows) if r["m"] == 0]
        keys = (
            "elastic_bbox_score", "artifact_region_bbox_score", "text_region_bbox_score",
            "positive_elastic", "clean_soft", "positive_macro", "positive_micro",
        )
        draws: dict[str, list[float]] = {k: [] for k in keys}
        for _ in range(bootstrap):
            indices = (
                [rng.choice(pos_indices) for _ in pos_indices]
                + [rng.choice(clean_indices) for _ in clean_indices]
            )
            bh = headline([rows[i] for i in indices])
            for key in keys:
                if bh[key] is not None:
                    draws[key].append(bh[key])
        result["bootstrap"] = {
            "iterations": bootstrap,
            "seed": seed,
            "ci95": {
                key: [_quantile(values, 0.025), _quantile(values, 0.975)] if values else []
                for key, values in draws.items()
            },
            "note": "stratified (positive/clean) resampling; RNG differs from the "
                    "numpy reference implementation, so CI bounds may differ "
                    "slightly while point estimates match exactly",
        }
    return result, rows


def _quantile(sorted_values: list[float], q: float) -> float:
    """Linear-interpolated quantile (numpy default method) on sorted data."""
    values = sorted(sorted_values)
    if not values:
        return float("nan")
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * q
    lower = math.floor(pos)
    upper = math.ceil(pos)
    if lower == upper:
        return values[int(pos)]
    return values[lower] + (values[upper] - values[lower]) * (pos - lower)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--annotations", nargs="+", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", help="metrics JSON path (default: stdout)")
    parser.add_argument("--per-sample-out", help="optional per-sample JSON path")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260804)
    args = parser.parse_args(argv)

    samples = load_annotations(args.annotations)
    predictions = load_predictions(args.predictions)
    missing = [s.sample_id for s in samples if s.sample_id not in predictions]
    metrics, rows = compute_metrics(samples, predictions, bootstrap=args.bootstrap, seed=args.seed)

    payload = dict(metrics)
    payload["missing_prediction_samples"] = len(missing)
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    if args.per_sample_out:
        slim_rows = [{k: v for k, v in row.items()} for row in rows]
        Path(args.per_sample_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.per_sample_out).write_text(
            json.dumps(slim_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    h = metrics["headline"]
    sys.stderr.write(
        f"main={h['elastic_bbox_score']:.3f} positive_elastic={h['positive_elastic']:.4f} "
        f"clean_soft={h['clean_soft']:.4f} failed_or_missing="
        f"{metrics['dataset']['failed_or_missing_predictions']}\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
