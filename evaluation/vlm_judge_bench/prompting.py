from __future__ import annotations

import json
import re

from .data import FAMILIES, PredictionBox


SYSTEM_PROMPT = """You are a meticulous visual anomaly judge. Inspect the entire image for localized generation defects. Evaluate rendering artifacts, malformed entity structure, topology/connectivity errors, relational or physical impossibilities, visually broken text, and text-content mismatch only when target text is explicitly supplied. Do not report ordinary blur from depth of field, compression, artistic style, aesthetic preference, or instruction compliance. Return tight, non-overlapping boxes: one spatial anomaly per box, without duplicates or a large box merging separate defects."""

FAMILY_GUIDE = """Allowed family values:
- D1_Artifact/A1_RenderingArtifact: local blending, smearing, texture/detail/color corruption
- D1_Artifact/A2_StructuralArtifact: extra, missing, fused, split, deformed, misplaced, disconnected or proportionally wrong parts
- D1_Artifact/A3_TopologyArtifact: broken/wrong paths, connections, branches, closures, crossings, arrows or symbols
- D1_Artifact/A4_RelationalAndPhysicalArtifact: penetration, occlusion/contact/support/scale/perspective/light/reflection/pose/material inconsistency
- D2_TextCorrectness/A1_TextVisualIntegrity: malformed, fused, overlapping, misaligned, blurred, occluded or clipped glyphs/text
- D2_TextCorrectness/A2_TextContentFidelity: readable text differs from an explicitly provided target"""


def build_prompt(reference_prompt: str, format_retry: bool = False) -> str:
    reference = reference_prompt.strip()
    context = f"\nReference generation instruction/target, when useful:\n{reference}" if reference else "\nNo target text or generation instruction is available; do not infer text-content mismatch from spelling plausibility alone."
    retry = "\nYour previous response could not be parsed. Return only one valid JSON object, with no Markdown or commentary." if format_retry else ""
    return f"""{SYSTEM_PROMPT}

{FAMILY_GUIDE}
{context}

Coordinates use corner coordinates on a 0..1000 image coordinate system. `bbox_1000_xyxy` must be exactly `[x_min, y_min, x_max, y_max]`: the first pair is the top-left corner and the second pair is the bottom-right corner. Do NOT output width or height. Require 0 <= x_min < x_max <= 1000 and 0 <= y_min < y_max <= 1000. If no anomaly is visible, return an empty boxes list.

Return ONLY this JSON shape:
{{"no_error": false, "boxes": [{{"bbox_1000_xyxy": [100, 200, 300, 400], "family": "one allowed family", "description": "brief visible evidence"}}]}}{retry}"""


class PredictionParseError(ValueError):
    pass


def _extract_json(text: str) -> dict:
    cleaned = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        cleaned = fenced.group(1)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end <= start:
            raise PredictionParseError("response does not contain a JSON object")
        try:
            value = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError as exc:
            raise PredictionParseError(f"invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise PredictionParseError("top-level response must be an object")
    return value


def parse_prediction(text: str) -> tuple[bool, list[PredictionBox], list[str]]:
    value = _extract_json(text)
    raw_boxes = value.get("boxes")
    if not isinstance(raw_boxes, list):
        raise PredictionParseError("boxes must be a list")
    warnings: list[str] = []
    boxes: list[PredictionBox] = []
    for idx, raw in enumerate(raw_boxes):
        if not isinstance(raw, dict):
            warnings.append(f"box[{idx}] dropped: not an object")
            continue
        bbox_xyxy = raw.get("bbox_1000_xyxy")
        legacy_xywh = raw.get("bbox_1000_xywh")
        corner_coordinates = False
        try:
            if isinstance(bbox_xyxy, dict):
                corner_coordinates = True
                x, y = float(bbox_xyxy["x_min"]), float(bbox_xyxy["y_min"])
                x_max, y_max = float(bbox_xyxy["x_max"]), float(bbox_xyxy["y_max"])
                width, height = x_max - x, y_max - y
            elif isinstance(bbox_xyxy, list) and len(bbox_xyxy) == 4:
                corner_coordinates = True
                x, y, x_max, y_max = [float(v) for v in bbox_xyxy]
                width, height = x_max - x, y_max - y
                pass
            elif isinstance(legacy_xywh, list) and len(legacy_xywh) == 4:
                x, y, width, height = [float(v) for v in legacy_xywh]
                warnings.append(f"box[{idx}] legacy bbox_1000_xywh accepted")
            else:
                warnings.append(f"box[{idx}] dropped: invalid bbox_1000_xyxy")
                continue
        except KeyError:
            warnings.append(f"box[{idx}] dropped: bbox_1000_xyxy is missing a named corner")
            continue
        except (TypeError, ValueError):
            warnings.append(f"box[{idx}] dropped: non-numeric coordinates")
            continue
        if not all(v == v and abs(v) != float("inf") for v in (x, y, width, height)) or width <= 0 or height <= 0:
            warnings.append(f"box[{idx}] dropped: non-finite or non-positive coordinates")
            continue
        original = (x, y, width, height)
        if corner_coordinates:
            x_max, y_max = x + width, y + height
            x, y = max(0.0, min(1000.0, x)), max(0.0, min(1000.0, y))
            x_max, y_max = max(0.0, min(1000.0, x_max)), max(0.0, min(1000.0, y_max))
            width, height = x_max - x, y_max - y
        else:
            x, y = max(0.0, min(1000.0, x)), max(0.0, min(1000.0, y))
            width, height = min(width, 1000.0 - x), min(height, 1000.0 - y)
        if width <= 0 or height <= 0:
            warnings.append(f"box[{idx}] dropped: outside image")
            continue
        if original != (x, y, width, height):
            warnings.append(f"box[{idx}] clamped to image bounds")
        family = raw.get("family")
        if family not in FAMILIES:
            warnings.append(f"box[{idx}] unknown family ignored: {family!r}")
            family = None
        boxes.append(PredictionBox((x, y, width, height), family, str(raw.get("description", ""))))
    if raw_boxes and not boxes:
        raise PredictionParseError("response reported boxes but none had valid named corner coordinates")
    declared = value.get("no_error")
    no_error = len(boxes) == 0
    if not isinstance(declared, bool) or declared != no_error:
        warnings.append("no_error normalized from boxes")
    return no_error, boxes, warnings
