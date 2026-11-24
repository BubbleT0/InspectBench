#!/usr/bin/env python3
"""Evaluate an OpenAI-compatible VLM endpoint on the frozen stratified split."""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests
from PIL import Image, ImageOps


WRITE_LOCK = threading.Lock()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--audit-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--max-pixels", type=int, default=200704)
    parser.add_argument("--smoke-per-label", type=int, default=0)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_samples(validation: Path, audit_manifest: Path) -> list[dict[str, Any]]:
    rows = read_jsonl(validation)
    audit = [r for r in read_jsonl(audit_manifest) if r.get("split") == "validation"]
    audit.sort(key=lambda r: int(r["ordinal"]))
    if len(rows) != len(audit):
        raise ValueError(f"validation/audit length mismatch: {len(rows)} != {len(audit)}")
    samples = []
    for row, meta in zip(rows, audit):
        image = str(row["images"][0])
        if image != str(meta["image_path"]):
            raise ValueError(f"image mismatch at ordinal {meta['ordinal']}")
        messages = row["messages"]
        prompt = str(messages[0]["content"]).replace("<image>\n", "", 1)
        gold = json.loads(messages[1]["content"])
        samples.append({"meta": meta, "image": image, "prompt": prompt, "gold": gold})
    return samples


def select_smoke(samples: list[dict[str, Any]], per_label: int) -> list[dict[str, Any]]:
    if per_label <= 0:
        return samples
    selected: list[dict[str, Any]] = []
    counts = {"no_error": 0, "error": 0}
    for sample in samples:
        label = str(sample["meta"]["label"])
        if counts.get(label, 0) < per_label:
            selected.append(sample)
            counts[label] = counts.get(label, 0) + 1
        if counts["no_error"] >= per_label and counts["error"] >= per_label:
            break
    return selected


def image_data_url(path: str, max_pixels: int) -> tuple[str, dict[str, Any]]:
    Image.MAX_IMAGE_PIXELS = None
    with Image.open(path) as original:
        image = ImageOps.exif_transpose(original).convert("RGB")
        source_size = image.size
        pixels = image.width * image.height
        if pixels > max_pixels:
            scale = math.sqrt(max_pixels / pixels)
            size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
            image = image.resize(size, Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=95, subsampling=0, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}", {
        "source_size": list(source_size),
        "inference_size": list(image.size),
        "encoded_bytes": len(buffer.getvalue()),
    }


def extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        value = json.loads(text[start : end + 1])
        if isinstance(value, dict):
            return value
    raise ValueError("assistant response does not contain one JSON object")


def call_api(sample: dict[str, Any], args: argparse.Namespace, model: str) -> dict[str, Any]:
    data_url, image_meta = image_data_url(sample["image"], args.max_pixels)
    payload = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": sample["prompt"]},
            ],
        }],
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 2048,
        "stream": False,
        "seed": 20260824,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    url = args.endpoint.rstrip("/") + "/v1/chat/completions"
    started = time.time()
    error = None
    for attempt in range(1, args.max_retries + 1):
        try:
            response = requests.post(
                url,
                headers={"Authorization": "Bearer dummy", "Content-Type": "application/json"},
                json=payload,
                timeout=args.timeout,
            )
            response.raise_for_status()
            body = response.json()
            raw = str(body["choices"][0]["message"].get("content", ""))
            parsed = extract_json(raw)
            return {
                "sample_id": sample["meta"]["sample_id"],
                "source": sample["meta"]["source"],
                "label": sample["meta"]["label"],
                "image": sample["image"],
                "gold": sample["gold"],
                "prediction": parsed,
                "raw_response": raw,
                "request_ok": True,
                "parse_ok": True,
                "attempts": attempt,
                "latency_seconds": round(time.time() - started, 4),
                "image_meta": image_meta,
                "usage": body.get("usage"),
            }
        except Exception as exc:  # preserve failures for denominator accounting
            error = f"{type(exc).__name__}: {exc}"
            if attempt < args.max_retries:
                time.sleep(min(2 ** attempt, 8))
    return {
        "sample_id": sample["meta"]["sample_id"],
        "source": sample["meta"]["source"],
        "label": sample["meta"]["label"],
        "image": sample["image"],
        "gold": sample["gold"],
        "prediction": None,
        "raw_response": "",
        "request_ok": False,
        "parse_ok": False,
        "attempts": args.max_retries,
        "latency_seconds": round(time.time() - started, 4),
        "image_meta": image_meta,
        "error": error,
    }


def iou(a: list[float], b: list[float]) -> float:
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, right - left) * max(0.0, bottom - top)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def valid_boxes(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    result = []
    for item in value:
        box = item.get("bbox_1000_xyxy") if isinstance(item, dict) else None
        if isinstance(box, list) and len(box) == 4 and all(isinstance(x, (int, float)) for x in box):
            result.append(item)
    return result


def bbox_matches(gold: list[dict[str, Any]], pred: list[dict[str, Any]], threshold: float, family: bool) -> int:
    candidates = []
    for pi, p in enumerate(pred):
        for gi, g in enumerate(gold):
            if family and p.get("family") != g.get("family"):
                continue
            overlap = iou(p["bbox_1000_xyxy"], g["bbox_1000_xyxy"])
            if overlap >= threshold:
                candidates.append((overlap, pi, gi))
    used_p, used_g = set(), set()
    matched = 0
    for _, pi, gi in sorted(candidates, reverse=True):
        if pi not in used_p and gi not in used_g:
            used_p.add(pi)
            used_g.add(gi)
            matched += 1
    return matched


def safe_div(num: float, den: float) -> float | None:
    return round(num / den, 6) if den else None


def compute_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    totals = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    per_source: dict[str, dict[str, int]] = {}
    parse_ok = contradictions = 0
    gold_boxes = pred_boxes = 0
    loc = {"iou30_any": 0, "iou50_any": 0, "iou30_family": 0, "iou50_family": 0}
    latencies = []
    for record in records:
        source = record["source"]
        source_counts = per_source.setdefault(source, {"tp": 0, "tn": 0, "fp": 0, "fn": 0, "count": 0})
        source_counts["count"] += 1
        gold_error = not bool(record["gold"].get("no_error", False))
        prediction = record.get("prediction")
        valid = isinstance(prediction, dict) and isinstance(prediction.get("no_error"), bool) and isinstance(prediction.get("boxes"), list)
        parse_ok += int(valid)
        if valid:
            pboxes = valid_boxes(prediction["boxes"])
            pred_error = not prediction["no_error"]
            contradictions += int(prediction["no_error"] and bool(prediction["boxes"]))
        else:
            pboxes = []
            pred_error = True  # failed output is not credited as a clean abstention
        key = "tp" if gold_error and pred_error else "fn" if gold_error else "fp" if pred_error else "tn"
        totals[key] += 1
        source_counts[key] += 1
        gboxes = valid_boxes(record["gold"].get("boxes", []))
        gold_boxes += len(gboxes)
        pred_boxes += len(pboxes)
        loc["iou30_any"] += bbox_matches(gboxes, pboxes, 0.3, False)
        loc["iou50_any"] += bbox_matches(gboxes, pboxes, 0.5, False)
        loc["iou30_family"] += bbox_matches(gboxes, pboxes, 0.3, True)
        loc["iou50_family"] += bbox_matches(gboxes, pboxes, 0.5, True)
        latencies.append(float(record.get("latency_seconds", 0)))

    sensitivity = safe_div(totals["tp"], totals["tp"] + totals["fn"])
    specificity = safe_div(totals["tn"], totals["tn"] + totals["fp"])
    balanced = round((sensitivity + specificity) / 2, 6) if sensitivity is not None and specificity is not None else None
    source_metrics = {}
    for source, counts in per_source.items():
        source_metrics[source] = {
            **counts,
            "accuracy": safe_div(counts["tp"] + counts["tn"], counts["count"]),
            "sensitivity": safe_div(counts["tp"], counts["tp"] + counts["fn"]),
            "specificity": safe_div(counts["tn"], counts["tn"] + counts["fp"]),
        }
    return {
        "samples": len(records),
        "confusion": totals,
        "accuracy": safe_div(totals["tp"] + totals["tn"], len(records)),
        "balanced_accuracy": balanced,
        "error_sensitivity": sensitivity,
        "clean_specificity": specificity,
        "clean_false_positive_rate": safe_div(totals["fp"], totals["tn"] + totals["fp"]),
        "schema_valid_rate": safe_div(parse_ok, len(records)),
        "schema_contradictions": contradictions,
        "bbox": {
            "gold_boxes": gold_boxes,
            "predicted_boxes": pred_boxes,
            **loc,
            "recall_iou30_any": safe_div(loc["iou30_any"], gold_boxes),
            "recall_iou50_any": safe_div(loc["iou50_any"], gold_boxes),
            "recall_iou30_family": safe_div(loc["iou30_family"], gold_boxes),
            "recall_iou50_family": safe_div(loc["iou50_family"], gold_boxes),
        },
        "latency_seconds": {
            "mean": round(sum(latencies) / len(latencies), 4) if latencies else None,
            "max": round(max(latencies), 4) if latencies else None,
        },
        "per_source": source_metrics,
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = select_smoke(load_samples(Path(args.validation), Path(args.audit_manifest)), args.smoke_per_label)

    models_response = requests.get(args.endpoint.rstrip("/") + "/v1/models", timeout=30)
    models_response.raise_for_status()
    model_record = models_response.json()["data"][0]
    model = model_record["id"]
    metadata = {
        "endpoint": args.endpoint,
        "served_model": model,
        "served_model_root": model_record.get("root"),
        "validation": args.validation,
        "audit_manifest": args.audit_manifest,
        "sample_count": len(samples),
        "workers": args.workers,
        "max_pixels": args.max_pixels,
        "decode": {"temperature": 0.0, "top_p": 1.0, "seed": 20260824, "enable_thinking": False},
        "started_at_unix": time.time(),
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    predictions_path = output_dir / "predictions.jsonl"
    completed = set()
    records = []
    if predictions_path.exists():
        records = read_jsonl(predictions_path)
        completed = {r["sample_id"] for r in records}
    pending = [s for s in samples if s["meta"]["sample_id"] not in completed]
    with predictions_path.open("a", encoding="utf-8") as output:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(call_api, sample, args, model): sample for sample in pending}
            done = len(records)
            for future in as_completed(futures):
                result = future.result()
                with WRITE_LOCK:
                    output.write(json.dumps(result, ensure_ascii=False) + "\n")
                    output.flush()
                records.append(result)
                done += 1
                if done % 10 == 0 or done == len(samples):
                    print(f"progress={done}/{len(samples)}", flush=True)

    records.sort(key=lambda r: r["sample_id"])
    metrics = compute_metrics(records)
    metadata["completed_at_unix"] = time.time()
    metadata["duration_seconds"] = round(metadata["completed_at_unix"] - metadata["started_at_unix"], 3)
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
