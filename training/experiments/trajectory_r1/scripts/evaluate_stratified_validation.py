#!/usr/bin/env python3
"""Evaluate an OpenAI-compatible VLM critic endpoint on labelled JSONL data."""

import argparse
import base64
import json
import mimetypes
import os
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def request_json(url, payload=None, timeout=300):
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode()
    req = urllib.request.Request(url, data=data)
    if payload is not None:
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read())


def extract_json(text):
    text = text.strip()
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1].strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I)
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            value, _ = decoder.raw_decode(text[match.start():])
            if isinstance(value, dict) and "no_error" in value:
                return value
        except json.JSONDecodeError:
            pass
    raise ValueError("No valid prediction JSON found")


def image_data_url(path):
    mime = mimetypes.guess_type(path)[0] or "image/jpeg"
    with open(path, "rb") as handle:
        encoded = base64.b64encode(handle.read()).decode()
    return f"data:{mime};base64,{encoded}"


def load_examples(paths, limit=None):
    examples = []
    for path in paths:
        source = Path(path).stem.removesuffix("_test").removesuffix("_validation")
        with open(path, encoding="utf-8") as handle:
            for source_index, line in enumerate(handle):
                row = json.loads(line)
                prompt = row["messages"][0]["content"].replace("<image>\n", "", 1)
                target_raw = row["messages"][-1]["content"]
                examples.append({
                    "index": len(examples),
                    "source_index": source_index,
                    "source": source,
                    "image": row["images"][0],
                    "prompt": prompt,
                    "target_raw": target_raw,
                    "target": extract_json(target_raw),
                })
                if limit and len(examples) >= limit:
                    return examples
    return examples


def infer_one(example, endpoint, model, max_tokens, retries):
    payload = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_data_url(example["image"])}},
                {"type": "text", "text": example["prompt"]},
            ],
        }],
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }
    started = time.time()
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            response = request_json(f"{endpoint}/v1/chat/completions", payload, timeout=600)
            raw = response["choices"][0]["message"]["content"]
            prediction = extract_json(raw)
            return {
                "index": example["index"],
                "source_index": example["source_index"],
                "source": example["source"],
                "image": example["image"],
                "target": example["target"],
                "prediction": prediction,
                "raw_response": raw,
                "latency_s": round(time.time() - started, 3),
                "attempts": attempt,
                "error": None,
            }
        except Exception as exc:  # record server and parse failures per sample
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < retries:
                time.sleep(2 * attempt)
    return {
        "index": example["index"],
        "source_index": example["source_index"],
        "source": example["source"],
        "image": example["image"],
        "target": example["target"],
        "prediction": None,
        "raw_response": None,
        "latency_s": round(time.time() - started, 3),
        "attempts": retries,
        "error": last_error,
    }


def iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    iw = max(0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0, min(ay2, by2) - max(ay1, by1))
    intersection = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - intersection
    return intersection / union if union > 0 else 0.0


def match_boxes(target_boxes, predicted_boxes, require_family):
    candidates = []
    for ti, target in enumerate(target_boxes):
        for pi, predicted in enumerate(predicted_boxes):
            if require_family and target.get("family") != predicted.get("family"):
                continue
            try:
                overlap = iou(target["bbox_1000_xyxy"], predicted["bbox_1000_xyxy"])
            except (KeyError, TypeError, ValueError):
                continue
            if overlap >= 0.5:
                candidates.append((overlap, ti, pi))
    matched_target, matched_predicted = set(), set()
    for _, ti, pi in sorted(candidates, reverse=True):
        if ti not in matched_target and pi not in matched_predicted:
            matched_target.add(ti)
            matched_predicted.add(pi)
    return len(matched_target), len(target_boxes), len(predicted_boxes)


def safe_ratio(numerator, denominator):
    return round(numerator / denominator, 6) if denominator else None


def summarize(results):
    valid = [row for row in results if row["prediction"] is not None]
    tp = tn = fp = fn = 0
    spatial = [0, 0, 0]
    spatial_family = [0, 0, 0]
    for row in valid:
        target_error = not bool(row["target"]["no_error"])
        predicted_error = not bool(row["prediction"]["no_error"])
        if target_error and predicted_error:
            tp += 1
        elif not target_error and not predicted_error:
            tn += 1
        elif not target_error and predicted_error:
            fp += 1
        else:
            fn += 1
        target_boxes = row["target"].get("boxes", [])
        predicted_boxes = row["prediction"].get("boxes", [])
        for accumulator, require_family in ((spatial, False), (spatial_family, True)):
            matched, target_count, predicted_count = match_boxes(
                target_boxes, predicted_boxes, require_family
            )
            accumulator[0] += matched
            accumulator[1] += target_count
            accumulator[2] += predicted_count

    precision = safe_ratio(tp, tp + fp)
    recall = safe_ratio(tp, tp + fn)
    f1 = None if precision is None or recall is None or precision + recall == 0 else round(
        2 * precision * recall / (precision + recall), 6
    )

    def box_metrics(counts):
        matched, targets, predictions = counts
        box_precision = safe_ratio(matched, predictions)
        box_recall = safe_ratio(matched, targets)
        box_f1 = None
        if box_precision is not None and box_recall is not None and box_precision + box_recall:
            box_f1 = round(2 * box_precision * box_recall / (box_precision + box_recall), 6)
        return {
            "matched": matched,
            "target_boxes": targets,
            "predicted_boxes": predictions,
            "precision": box_precision,
            "recall": box_recall,
            "f1": box_f1,
        }

    return {
        "samples": len(results),
        "valid_predictions": len(valid),
        "failed_predictions": len(results) - len(valid),
        "classification": {
            "positive_definition": "actual_error",
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "accuracy": safe_ratio(tp + tn, len(valid)),
            "precision": precision,
            "recall": recall,
            "specificity": safe_ratio(tn, tn + fp),
            "balanced_accuracy": safe_ratio(
                safe_ratio(tp, tp + fn) + safe_ratio(tn, tn + fp), 2
            ) if (tp + fn) and (tn + fp) else None,
            "f1": f1,
        },
        "bbox_iou_0_5": box_metrics(spatial),
        "bbox_iou_0_5_and_family": box_metrics(spatial_family),
        "mean_latency_s": round(sum(row["latency_s"] for row in results) / len(results), 3) if results else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--dataset", required=True, action="append")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--model")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    endpoint = args.endpoint.rstrip("/")
    model = args.model or request_json(f"{endpoint}/v1/models")["data"][0]["id"]
    examples = load_examples(args.dataset, args.limit)
    os.makedirs(args.output_dir, exist_ok=True)
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(infer_one, item, endpoint, model, args.max_tokens, args.retries): item
            for item in examples
        }
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda row: row["index"])
    summary = summarize(results)
    summary.update({
        "split": args.split,
        "datasets": args.dataset,
        "endpoint": endpoint,
        "model": model,
        "evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "bbox_match_rule": "greedy one-to-one match, IoU >= 0.5",
    })
    summary["by_source"] = {
        source: summarize([row for row in results if row["source"] == source])
        for source in sorted({row["source"] for row in results})
    }

    predictions_path = Path(args.output_dir) / f"{args.split}_predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8") as handle:
        for row in results:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary_path = Path(args.output_dir) / f"{args.split}_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
