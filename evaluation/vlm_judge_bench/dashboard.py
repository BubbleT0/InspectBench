from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageOps

from .data import load_annotations, load_predictions
from .metrics import compute_metrics, paired_bootstrap


MODEL_LABELS = {
    "0828_ck444": "0828_ck444",
    "0828_ck688": "0828_ck688",
    "qwen37": "qwen3.7-plus",
    "gemini31": "gemini-3.1-pro-preview",
    "local_qwen": "Qwen3.6-27B-agen-0722-agentv4",
    "gpt56_direct": "gpt-5.6-sol",
    "claude_opus48_direct": "claude-opus-4-8-thinking",
    "u1_local_27b_pipeline": "hj的管线",
    "u1_claude_opus48_pipeline": "hj的管线 Claude",
    "u1_gpt56_pipeline": "hj的管线 GPT56",
    "u1_v5_local_27b_pipeline": "hj的管线 V5 Local",
    "u1_v69_agen0722_pipeline": "v69+agen0722",
    "agentic_local_27b_pipeline": "xy的管线",
}

MODEL_DETAILS = {
    "0828_ck444": "单轮直调；checkpoint-444；Qwen3.6-27B-agen-0722-agentv4",
    "0828_ck688": "单轮直调；checkpoint-688；Qwen3.6-27B-agen-0722-agentv4",
    "qwen37": "单轮直答",
    "gemini31": "单轮直答",
    "local_qwen": "单轮直答",
    "gpt56_direct": "单轮直答",
    "claude_opus48_direct": "单轮直答",
    "u1_local_27b_pipeline": (
        "管线来源：/path/to/storage/research/workspace/U2/critic/deliverables；"
        "实现：U1 Image Critic V4.1；视觉语言模型：Qwen3.6-27B-agen-0722-agentv4"
    ),
    "u1_claude_opus48_pipeline": (
        "管线来源：/path/to/storage/research/workspace/U2/critic/deliverables；"
        "实现：U1 Image Critic V4.1；视觉语言模型：claude-opus-4-8-thinking"
    ),
    "u1_gpt56_pipeline": (
        "管线来源：integrations/u1_critic_v41_test；"
        "实现：U1 Image Critic V4.1；视觉语言模型：gpt-5.6-sol"
    ),
    "u1_v5_local_27b_pipeline": (
        "管线来源：/path/to/storage/research/workspace/U2/critic/deliverables/u1_critic_v5_20260807；"
        "实现：U1 Image Critic V5；视觉语言模型：Qwen3.6-27B-agen-0722-agentv4"
    ),
    "u1_v69_agen0722_pipeline": (
        "管线来源：/path/to/storage/research/workspace/U2/critic/deliverables/"
        "u1_critic_v69_20260827；实现：U1 Image Critic V6.9；"
        "视觉语言模型：Qwen3.6-27B-agen-0722-agentv4"
    ),
    "agentic_local_27b_pipeline": (
        "管线来源：/path/to/storage/research/agent_workspace/codex/agentic_critic；"
        "视觉语言模型：Qwen3.6-27B-agen-0722-agentv4"
    ),
}

MODEL_COLORS = {
    "0828_ck444": "#f97316",
    "0828_ck688": "#06b6d4",
    "qwen37": "#f59e0b",
    "gemini31": "#a78bfa",
    "local_qwen": "#38bdf8",
    "gpt56_direct": "#60a5fa",
    "claude_opus48_direct": "#c084fc",
    "u1_local_27b_pipeline": "#f59e0b",
    "u1_claude_opus48_pipeline": "#fb7185",
    "u1_gpt56_pipeline": "#14b8a6",
    "u1_v5_local_27b_pipeline": "#22c55e",
    "u1_v69_agen0722_pipeline": "#e879f9",
    "agentic_local_27b_pipeline": "#34d399",
}

FAMILY_LABELS = {
    "D1_Artifact/A1_RenderingArtifact": "局部渲染错误",
    "D1_Artifact/A2_StructuralArtifact": "实体结构错误",
    "D1_Artifact/A3_TopologyArtifact": "拓扑连接错误",
    "D1_Artifact/A4_RelationalAndPhysicalArtifact": "关系与物理错误",
    "D2_TextCorrectness/A1_TextVisualIntegrity": "文字视觉完整性错误",
    "D2_TextCorrectness/A2_TextContentFidelity": "文字内容保真错误",
}


def _source_key(source_file: str) -> str:
    name = Path(source_file).name.lower()
    if "natural_clean_110" in name:
        return "natural_clean_110"
    if "external4" in name:
        return "external4"
    if "u1.5" in name or "u1_5" in name:
        return "u1_5"
    if "gpt" in name:
        return "gpt"
    return "other"


def _load_run(run_dir: str | Path) -> list[tuple[str, Path, dict]]:
    path = Path(run_dir).resolve()
    manifest_path = path / "run_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"run manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    models = manifest.get("models") or []
    if not models:
        raise ValueError(f"dashboard run has no models: {path}")
    loaded = []
    for model in models:
        model_dir = path / model
        for filename in ("predictions.jsonl", "metrics.json", "per_sample_metrics.json"):
            if not (model_dir / filename).exists():
                raise FileNotFoundError(f"missing scored run artifact: {model_dir / filename}")
        loaded.append((model, model_dir, manifest))
    return loaded


def _thumbnail(source: str, destination: Path, max_side: int) -> None:
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        image.save(destination, format="WEBP", quality=84, method=6)


def build_dashboard(
    run_dirs: list[str | Path],
    output_path: str | Path,
    thumbnail_max_side: int = 900,
    paired_iterations: int = 0,
    seed: int = 20260804,
) -> Path:
    if len(run_dirs) < 1:
        raise ValueError("at least one run directory is required")
    loaded = [item for path in run_dirs for item in _load_run(path)]
    model_names = [model for model, _, _ in loaded]
    if len(model_names) != len(set(model_names)):
        duplicates = sorted({model for model in model_names if model_names.count(model) > 1})
        raise ValueError(f"duplicate model names across dashboard runs: {duplicates}")
    first_manifest = loaded[0][2]
    annotation_paths = first_manifest["annotations"]
    samples = load_annotations(annotation_paths)
    first_ids = first_manifest.get("sample_ids") or []
    if first_ids:
        wanted = set(first_ids)
        samples = [sample for sample in samples if sample.sample_id in wanted]
    expected_ids = [sample.sample_id for sample in samples]

    output = Path(output_path).resolve()
    assets_dir = output.parent / f"{output.stem}_assets"
    model_payload: dict[str, dict] = {}
    rows_by_model: dict[str, list[dict]] = {}
    prediction_sets = {}
    per_sample_sets = {}
    for model, model_dir, manifest in loaded:
        ids = manifest.get("sample_ids") or expected_ids
        # Runs built from the same 1054 annotations may preserve a different
        # source-file order. Predictions and per-sample metrics are keyed by ID,
        # so require the same unique ID set rather than identical list order.
        if len(ids) != len(expected_ids) or len(set(ids)) != len(ids) or set(ids) != set(expected_ids):
            missing = len(set(expected_ids) - set(ids))
            extra = len(set(ids) - set(expected_ids))
            raise ValueError(
                f"sample IDs are not aligned for model {model}: "
                f"missing={missing}, extra={extra}, rows={len(ids)}"
            )
        predictions = load_predictions(model_dir / "predictions.jsonl")
        missing = sorted(set(expected_ids) - set(predictions))
        if missing:
            raise ValueError(f"model {model} is missing {len(missing)} predictions")
        metrics = json.loads((model_dir / "metrics.json").read_text(encoding="utf-8"))
        per_rows = json.loads((model_dir / "per_sample_metrics.json").read_text(encoding="utf-8"))
        per_by_id = {row["sample_id"]: row for row in per_rows}
        _, metric_rows = compute_metrics(samples, predictions, bootstrap=0)
        positive_rows = [row for row in metric_rows if row["m"] > 0]
        clean_rows = [row for row in metric_rows if row["m"] == 0]
        positive_false_negatives = sum(row["n"] == 0 for row in positive_rows)
        clean_false_positives = sum(row["n"] > 0 for row in clean_rows)
        metrics["binary_decision"] = {
            "positive_false_negative_rate": (
                positive_false_negatives / len(positive_rows) if positive_rows else None
            ),
            "clean_false_positive_rate": (
                clean_false_positives / len(clean_rows) if clean_rows else None
            ),
            "overall_error_rate": (
                (positive_false_negatives + clean_false_positives) / len(metric_rows)
                if metric_rows else None
            ),
            "positive_false_negatives": positive_false_negatives,
            "clean_false_positives": clean_false_positives,
            "total_errors": positive_false_negatives + clean_false_positives,
        }
        rows_by_model[model] = metric_rows
        prediction_sets[model] = predictions
        per_sample_sets[model] = per_by_id
        model_payload[model] = {
            "label": MODEL_LABELS.get(model, model),
            "detail": MODEL_DETAILS.get(model, ""),
            "color": MODEL_COLORS.get(model, "#fb7185"),
            "metrics": metrics,
        }

    comparisons = (
        paired_bootstrap(rows_by_model, paired_iterations, seed)
        if len(rows_by_model) > 1 and paired_iterations > 0
        else None
    )
    sample_payload = []
    for index, sample in enumerate(samples, 1):
        asset_name = f"{index:04d}_{sample.sample_id}.webp"
        _thumbnail(sample.image_path, assets_dir / asset_name, thumbnail_max_side)
        model_results = {}
        for model in model_payload:
            prediction = prediction_sets[model][sample.sample_id]
            sample_metrics = per_sample_sets[model][sample.sample_id]
            prediction_credits = sample_metrics.get("prediction_credits") or []
            boxes = []
            for box_idx, box in enumerate(prediction.boxes):
                x, y, width, height = box.absolute_xywh(sample.image_width, sample.image_height)
                boxes.append({
                    "index": box_idx + 1,
                    "xywh": [x, y, width, height],
                    "family": box.family,
                    "family_label": FAMILY_LABELS.get(box.family or "", box.family or "未提供类别"),
                    "description": box.description,
                    "credit": prediction_credits[box_idx] if box_idx < len(prediction_credits) else 0.0,
                })
            model_results[model] = {
                "status": prediction.status,
                "boxes": boxes,
                "metrics": sample_metrics,
            }
        sample_payload.append({
            "index": index,
            "sample_id": sample.sample_id,
            "image": f"{assets_dir.name}/{asset_name}",
            "width": sample.image_width,
            "height": sample.image_height,
            "prompt": sample.prompt,
            "no_error": sample.no_error,
            "source": _source_key(sample.source_file),
            "gt": [{
                "index": gt_idx + 1,
                "xywh": list(gt.bbox_xywh),
                "family": gt.family,
                "family_label": FAMILY_LABELS.get(gt.family, gt.family),
                "detail": gt.detail,
            } for gt_idx, gt in enumerate(sample.boxes)],
            "models": model_results,
        })

    payload = {
        "generated_for": f"{len(samples)} 张图片异常区域评测",
        "models": model_payload,
        "samples": sample_payload,
        "comparisons": comparisons,
        "families": FAMILY_LABELS,
        "dataset": {
            "images": len(samples),
            "positive_images": sum(bool(sample.boxes) for sample in samples),
            "clean_images": sum(not sample.boxes for sample in samples),
            "gt_boxes": sum(len(sample.boxes) for sample in samples),
        },
    }
    template_path = Path(__file__).with_name("dashboard_template.html")
    template = template_path.read_text(encoding="utf-8")
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    html = template.replace("__DASHBOARD_DATA__", encoded)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(output.suffix + ".tmp")
    temp.write_text(html, encoding="utf-8")
    temp.replace(output)
    return output
