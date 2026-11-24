from __future__ import annotations

import asyncio
import copy
import difflib
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from critic_workflows import BBox, ImageAsset, ModelRequest
from critic_workflows.utils import parse_response, serialize_json

from .edit_consensus import is_near_full_frame
from .quality_policy import (
    CORE_CONFIDENCE_THRESHOLD,
    CORE_MATERIALITIES,
)


_CHINESE = re.compile(r"[\u4e00-\u9fff]")

_BBOX_SCHEMA: Mapping[str, Any] = {
    "type": "array",
    "items": {"type": "integer", "minimum": 0, "maximum": 1000},
    "minItems": 4,
    "maxItems": 4,
}

_ISSUE_SCHEMA: Mapping[str, Any] = {
    "type": "object",
    "required": [
        "artifact_type",
        "observation",
        "abnormality",
        "confidence",
        "materiality",
        "source_finding_ids",
    ],
    "properties": {
        "artifact_type": {"type": "string"},
        "observation": {"type": "string"},
        "abnormality": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "materiality": {
            "type": "string",
            "enum": ["critical", "major"],
        },
        "source_finding_ids": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
}

_EDIT_REGION_SCHEMA: Mapping[str, Any] = {
    "type": "object",
    "required": [
        "edit_region_id",
        "bbox_norm",
        "edit_target",
        "edit_region_rationale",
        "source_finding_ids",
        "issues",
        "confidence",
        "verification",
    ],
    "properties": {
        "edit_region_id": {"type": "string"},
        "bbox_norm": _BBOX_SCHEMA,
        "edit_target": {"type": "string"},
        "edit_region_rationale": {"type": "string"},
        "source_finding_ids": {
            "type": "array",
            "items": {"type": "string"},
        },
        "issues": {"type": "array", "items": _ISSUE_SCHEMA},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "verification": {
            "type": "string",
            "enum": ["supported"],
        },
    },
}

CONSOLIDATED_RESULT_SCHEMA: Mapping[str, Any] = {
    "type": "object",
    "required": [
        "verdict",
        "confidence",
        "edit_regions",
        "discarded_finding_ids",
        "discard_reasons",
        "rationale",
    ],
    "properties": {
        "verdict": {
            "type": "string",
            "enum": ["artifact_present", "no_artifact"],
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "edit_regions": {
            "type": "array",
            "items": _EDIT_REGION_SCHEMA,
        },
        "discarded_finding_ids": {
            "type": "array",
            "items": {"type": "string"},
        },
        "discard_reasons": {
            "type": "object",
            "additionalProperties": {"type": "string"},
        },
        "rationale": {"type": "string"},
    },
}


class ConsolidationValidationError(ValueError):
    def __init__(self, errors: Sequence[str]):
        self.errors = list(errors)
        super().__init__("; ".join(self.errors))

def source_findings(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for status, key in (
        ("accepted", "findings"),
        ("uncertain", "uncertain_findings"),
    ):
        values = result.get(key, [])
        if not isinstance(values, list):
            continue
        for value in values:
            if not isinstance(value, Mapping):
                continue
            item = copy.deepcopy(dict(value))
            item["source_finding_id"] = f"f{len(output):03d}"
            item["source_status"] = status
            output.append(item)
    return output


def materiality_gate_reason(
    source: Mapping[str, Any],
) -> str | None:
    try:
        confidence = float(source.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    if confidence <= CORE_CONFIDENCE_THRESHOLD:
        return "置信度不高于0.6，不进入明显核心错误结果。"
    metadata = source.get("metadata", {})
    if not isinstance(metadata, Mapping):
        metadata = {}
    materiality = str(metadata.get("materiality", "")).lower()
    if materiality and materiality not in CORE_MATERIALITIES:
        return "该问题属于轻微瑕疵或非核心质量问题。"
    evidence_basis = str(metadata.get("evidence_basis", "")).lower()
    if evidence_basis and evidence_basis != "visible_only":
        return "该问题并非仅由原图直接可见证据支持。"
    return None


def _intersection(left: Sequence[float], right: Sequence[float]) -> float:
    width = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
    height = max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
    return width * height


def _area(bbox: Sequence[float]) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def spatial_hints(
    findings: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for left_index, left in enumerate(findings):
        left_bbox = left["bbox_norm"]
        left_area = _area(left_bbox)
        for right in findings[left_index + 1:]:
            right_bbox = right["bbox_norm"]
            right_area = _area(right_bbox)
            intersection = _intersection(left_bbox, right_bbox)
            if intersection <= 0:
                continue
            union = left_area + right_area - intersection
            output.append(
                {
                    "left_id": left["source_finding_id"],
                    "right_id": right["source_finding_id"],
                    "iou": round(intersection / union, 4),
                    "left_covered": round(intersection / left_area, 4),
                    "right_covered": round(intersection / right_area, 4),
                }
            )
    return output


def partition_source_findings(
    findings: Sequence[Mapping[str, Any]],
    *,
    max_batch_size: int,
) -> list[list[dict[str, Any]]]:
    """Create bounded semantic/spatial batches for a small consolidator model."""

    if max_batch_size <= 0:
        raise ValueError("max_batch_size must be positive")
    keyed: dict[str, list[dict[str, Any]]] = {}
    order: list[str] = []
    for raw in findings:
        finding = copy.deepcopy(dict(raw))
        metadata = finding.get("metadata", {})
        if not isinstance(metadata, Mapping):
            metadata = {}
        semantic_key = str(
            metadata.get("semantic_group_id")
            or metadata.get("edit_target")
            or finding.get("region_id")
            or finding.get("source_finding_id")
        ).strip()
        if semantic_key not in keyed:
            keyed[semantic_key] = []
            order.append(semantic_key)
        keyed[semantic_key].append(finding)
    batches: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for key in order:
        group = keyed[key]
        while group:
            capacity = max_batch_size - len(current)
            if capacity == 0:
                batches.append(current)
                current = []
                capacity = max_batch_size
            current.extend(group[:capacity])
            group = group[capacity:]
            if len(current) == max_batch_size:
                batches.append(current)
                current = []
    if current:
        batches.append(current)
    return batches


def _normalized_target(value: Any) -> str:
    return re.sub(r"[\W_]+", "", str(value).lower())


def _source_edit_target(source: Mapping[str, Any]) -> str:
    metadata = source.get("metadata", {})
    if isinstance(metadata, Mapping):
        target = str(metadata.get("edit_target", "")).strip()
        if target:
            return target
    return str(source.get("observation", "")).strip()


def _same_semantic_text_target(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> bool:
    left_target = _normalized_target(_source_edit_target(left))
    right_target = _normalized_target(_source_edit_target(right))
    if not left_target or not right_target:
        return False
    directional_conflicts = (
        ("左侧", "右侧"),
        ("左上", "右上"),
        ("左下", "右下"),
        ("上方", "下方"),
        ("顶部", "底部"),
    )
    if any(
        (first in left_target and second in right_target)
        or (second in left_target and first in right_target)
        for first, second in directional_conflicts
    ):
        return False
    left_anchors = set(re.findall(r"[a-z0-9]{2,}", left_target))
    right_anchors = set(re.findall(r"[a-z0-9]{2,}", right_target))
    if len(left_anchors & right_anchors) >= 2:
        return True
    return difflib.SequenceMatcher(
        None,
        left_target,
        right_target,
    ).ratio() >= 0.72


def _bbox_iou(left: Sequence[float], right: Sequence[float]) -> float:
    intersection = _intersection(left, right)
    union = _area(left) + _area(right) - intersection
    return intersection / union if union > 0 else 0.0


def is_duplicate_edit_region(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> bool:
    left_bbox = left.get("bbox_norm")
    right_bbox = right.get("bbox_norm")
    if not (
        isinstance(left_bbox, Sequence)
        and not isinstance(left_bbox, (str, bytes))
        and isinstance(right_bbox, Sequence)
        and not isinstance(right_bbox, (str, bytes))
        and len(left_bbox) == 4
        and len(right_bbox) == 4
    ):
        return False
    if tuple(round(float(value), 4) for value in left_bbox) == tuple(
        round(float(value), 4) for value in right_bbox
    ):
        return True
    if _bbox_iou(left_bbox, right_bbox) < 0.9:
        return False
    left_target = _normalized_target(left.get("edit_target"))
    right_target = _normalized_target(right.get("edit_target"))
    directional_conflicts = (
        ("左侧", "右侧"),
        ("左上", "右上"),
        ("左下", "右下"),
        ("上方", "下方"),
        ("顶部", "底部"),
        ("前景", "背景"),
    )
    if any(
        (first in left_target and second in right_target)
        or (second in left_target and first in right_target)
        for first, second in directional_conflicts
    ):
        return False
    similarity = difflib.SequenceMatcher(
        None,
        left_target,
        right_target,
    ).ratio()
    return bool(left_target) and similarity >= 0.6


def _nonempty_chinese(value: Any) -> bool:
    return bool(str(value).strip()) and bool(_CHINESE.search(str(value)))


def text_sources_are_atomic(
    source_ids: Sequence[str],
    source_by_id: Mapping[str, Mapping[str, Any]],
) -> bool:
    text_sources = [
        source_by_id[source_id]
        for source_id in source_ids
        if source_id in source_by_id
        and str(source_by_id[source_id].get("artifact_type", "")).startswith(
            "text_"
        )
    ]
    if len(text_sources) <= 1:
        return True
    boxes: list[Sequence[float]] = []
    for source in text_sources:
        bbox = source.get("bbox_norm")
        if not (
            isinstance(bbox, Sequence)
            and not isinstance(bbox, (str, bytes))
            and len(bbox) == 4
        ):
            return False
        boxes.append(bbox)
    connected = {0}
    changed = True
    while changed:
        changed = False
        for left_index in tuple(connected):
            left = boxes[left_index]
            for right_index, right in enumerate(boxes):
                if right_index in connected:
                    continue
                intersection = _intersection(left, right)
                spatially_connected = False
                if intersection > 0:
                    containment = intersection / min(
                        _area(left),
                        _area(right),
                    )
                    spatially_connected = (
                        _bbox_iou(left, right) >= 0.3
                        or containment >= 0.6
                    )
                if spatially_connected or _same_semantic_text_target(
                    text_sources[left_index],
                    text_sources[right_index],
                ):
                    connected.add(right_index)
                    changed = True
    return len(connected) == len(boxes)


def validate_consolidated_result(
    value: Mapping[str, Any],
    source_findings: Sequence[Mapping[str, Any]],
    *,
    reject_nested_containers: bool = False,
) -> list[str]:
    errors: list[str] = []
    source_by_id = {
        str(source["source_finding_id"]): source
        for source in source_findings
        if isinstance(source, Mapping)
        and str(source.get("source_finding_id", "")).strip()
    }
    source_ids = set(source_by_id)
    regions = value.get("edit_regions", [])
    if not isinstance(regions, list):
        return ["edit_regions must be a list"]
    discarded = value.get("discarded_finding_ids", [])
    if not isinstance(discarded, list):
        discarded = []
        errors.append("discarded_finding_ids must be a list")
    discard_reasons = value.get("discard_reasons", {})
    if not isinstance(discard_reasons, Mapping):
        discard_reasons = {}
        errors.append("discard_reasons must be an object")

    handled: list[str] = [str(item) for item in discarded]
    region_ids: list[str] = []
    for region_index, region in enumerate(regions):
        if not isinstance(region, Mapping):
            errors.append(f"edit region {region_index} must be an object")
            continue
        region_id = str(region.get("edit_region_id", "")).strip()
        if not region_id:
            errors.append(f"edit region {region_index} has empty edit_region_id")
        region_ids.append(region_id)
        if not _nonempty_chinese(region.get("edit_target")):
            errors.append(f"edit region {region_id} has empty or non-Chinese edit_target")
        if not _nonempty_chinese(region.get("edit_region_rationale")):
            errors.append(
                f"edit region {region_id} has empty or non-Chinese rationale"
            )

        bbox_value = region.get("bbox_norm")
        try:
            bbox = BBox.from_value(bbox_value)
        except (TypeError, ValueError):
            errors.append(f"edit region {region_id} has invalid bbox")
        else:
            if is_near_full_frame(bbox):
                errors.append(f"edit region {region_id} has near-full-frame bbox")
        try:
            region_confidence = float(region.get("confidence", 0.0))
        except (TypeError, ValueError):
            region_confidence = 0.0
        if region_confidence <= CORE_CONFIDENCE_THRESHOLD:
            errors.append(
                f"edit region {region_id} confidence must be > "
                f"{CORE_CONFIDENCE_THRESHOLD}"
            )
        if region.get("verification") != "supported":
            errors.append(
                f"edit region {region_id} verification must be supported"
            )

        region_sources_raw = region.get("source_finding_ids", [])
        region_sources = (
            [str(item) for item in region_sources_raw]
            if isinstance(region_sources_raw, list)
            else []
        )
        if not region_sources:
            errors.append(f"edit region {region_id} has no source finding ids")
        elif not text_sources_are_atomic(
            region_sources,
            source_by_id,
        ):
            errors.append(
                f"edit region {region_id} merges spatially separate text entities"
            )
        handled.extend(region_sources)
        issues = region.get("issues", [])
        if not isinstance(issues, list) or not issues:
            errors.append(f"edit region {region_id} has no issues")
            continue
        issue_sources: set[str] = set()
        for issue_index, issue in enumerate(issues):
            if not isinstance(issue, Mapping):
                errors.append(
                    f"edit region {region_id} issue {issue_index} must be an object"
                )
                continue
            if not _nonempty_chinese(issue.get("observation")):
                errors.append(
                    f"edit region {region_id} issue {issue_index} "
                    "has empty or non-Chinese observation"
                )
            if not _nonempty_chinese(issue.get("abnormality")):
                errors.append(
                    f"edit region {region_id} issue {issue_index} "
                    "has empty or non-Chinese abnormality"
                )
            if str(issue.get("materiality", "")).lower() not in (
                CORE_MATERIALITIES
            ):
                errors.append(
                    f"edit region {region_id} issue {issue_index} "
                    "has non-core materiality"
                )
            try:
                issue_confidence = float(issue.get("confidence", 0.0))
            except (TypeError, ValueError):
                issue_confidence = 0.0
            if issue_confidence <= CORE_CONFIDENCE_THRESHOLD:
                errors.append(
                    f"edit region {region_id} issue {issue_index} "
                    f"confidence must be > {CORE_CONFIDENCE_THRESHOLD}"
                )
            raw_issue_sources = issue.get("source_finding_ids", [])
            current_issue_sources = (
                [str(item) for item in raw_issue_sources]
                if isinstance(raw_issue_sources, list)
                else []
            )
            if not current_issue_sources:
                errors.append(
                    f"edit region {region_id} issue {issue_index} has no source ids"
                )
            for source_id in current_issue_sources:
                issue_sources.add(source_id)
                if source_id not in region_sources:
                    errors.append(
                        f"issue references id outside its edit region: {source_id}"
                    )
        for source_id in region_sources:
            if source_id not in issue_sources:
                errors.append(
                    f"edit region source id is not represented by an issue: {source_id}"
                )

    for source_id in handled:
        if source_id not in source_ids:
            errors.append(f"unknown source finding id: {source_id}")
    counts = Counter(handled)
    for source_id in sorted(source_ids):
        if counts[source_id] == 0:
            errors.append(f"source finding id not handled: {source_id}")
        elif counts[source_id] > 1:
            errors.append(
                f"source finding id appears more than once: {source_id}"
            )
    for source_id in discarded:
        if source_id in source_ids and not _nonempty_chinese(
            discard_reasons.get(source_id)
        ):
            errors.append(
                f"discarded source finding id has no Chinese reason: {source_id}"
            )
    for reason_id in discard_reasons:
        if str(reason_id) not in set(map(str, discarded)):
            errors.append(
                f"discard reason references non-discarded id: {reason_id}"
            )
    for region_index, left in enumerate(regions):
        if not isinstance(left, Mapping):
            continue
        for right in regions[region_index + 1:]:
            if isinstance(right, Mapping) and is_duplicate_edit_region(left, right):
                errors.append(
                    "duplicate edit regions: "
                    f"{left.get('edit_region_id')} and "
                    f"{right.get('edit_region_id')}"
                )
    if reject_nested_containers:
        valid_region_boxes: list[tuple[Mapping[str, Any], list[float]]] = []
        for region in regions:
            if not isinstance(region, Mapping):
                continue
            try:
                bbox = BBox.from_value(region.get("bbox_norm")).as_list()
            except (TypeError, ValueError):
                continue
            valid_region_boxes.append((region, bbox))
        for outer, outer_bbox in valid_region_boxes:
            outer_area = _area(outer_bbox)
            contained_ids: list[str] = []
            for inner, inner_bbox in valid_region_boxes:
                if inner is outer:
                    continue
                inner_area = _area(inner_bbox)
                if inner_area <= 0 or outer_area <= inner_area:
                    continue
                containment = _intersection(
                    outer_bbox,
                    inner_bbox,
                ) / inner_area
                if containment >= 0.95:
                    contained_ids.append(
                        str(inner.get("edit_region_id", ""))
                    )
            if contained_ids:
                errors.append(
                    "redundant nested edit container: "
                    f"{outer.get('edit_region_id')} contains "
                    + ", ".join(contained_ids)
                )
    for region_id, count in Counter(region_ids).items():
        if region_id and count > 1:
            errors.append(f"duplicate edit_region_id: {region_id}")
    if not _nonempty_chinese(value.get("rationale")):
        errors.append("result has empty or non-Chinese rationale")
    return errors


def _restore_missing_issue_sources(
    value: Mapping[str, Any],
    sources: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    output = copy.deepcopy(dict(value))
    source_by_id = {
        str(source.get("source_finding_id")): source
        for source in sources
        if str(source.get("source_finding_id", "")).strip()
    }
    regions = output.get("edit_regions", [])
    if not isinstance(regions, list):
        return output
    for region in regions:
        if not isinstance(region, dict):
            continue
        region_sources = region.get("source_finding_ids", [])
        if not isinstance(region_sources, list):
            continue
        issues = region.get("issues", [])
        if not isinstance(issues, list):
            issues = []
            region["issues"] = issues
        represented = {
            str(source_id)
            for issue in issues
            if isinstance(issue, Mapping)
            for source_id in (
                issue.get("source_finding_ids", [])
                if isinstance(issue.get("source_finding_ids", []), list)
                else []
            )
        }
        for source_id in (str(item) for item in region_sources):
            if source_id in represented or source_id not in source_by_id:
                continue
            source = source_by_id[source_id]
            metadata = source.get("metadata", {})
            if not isinstance(metadata, Mapping):
                metadata = {}
            materiality = str(
                metadata.get("materiality", "major")
            ).lower()
            if materiality not in CORE_MATERIALITIES:
                materiality = "major"
            issues.append(
                {
                    "artifact_type": str(
                        source.get("artifact_type", "")
                    ),
                    "observation": str(
                        source.get("observation", "")
                    ).strip(),
                    "abnormality": str(
                        source.get("abnormality", "")
                    ).strip(),
                    "confidence": float(
                        source.get("confidence", 0.0)
                    ),
                    "materiality": materiality,
                    "source_finding_ids": [source_id],
                }
            )
            represented.add(source_id)
    return output


def _reconcile_missing_broad_container_sources(
    value: Mapping[str, Any],
    sources: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    output = copy.deepcopy(dict(value))
    regions = output.get("edit_regions", [])
    if not isinstance(regions, list):
        return output
    discarded = output.get("discarded_finding_ids", [])
    if not isinstance(discarded, list):
        discarded = []
        output["discarded_finding_ids"] = discarded
    discard_reasons = output.get("discard_reasons", {})
    if not isinstance(discard_reasons, dict):
        discard_reasons = {}
        output["discard_reasons"] = discard_reasons
    handled = {
        str(source_id)
        for region in regions
        if isinstance(region, Mapping)
        for source_id in (
            region.get("source_finding_ids", [])
            if isinstance(region.get("source_finding_ids", []), list)
            else []
        )
    } | {str(source_id) for source_id in discarded}
    valid_regions: list[tuple[list[int], bool]] = []
    for region in regions:
        if not isinstance(region, Mapping):
            continue
        try:
            bbox = BBox.from_value(region.get("bbox_norm")).as_list()
        except (TypeError, ValueError):
            continue
        issues = region.get("issues", [])
        contains_text_issue = any(
            isinstance(issue, Mapping)
            and str(issue.get("artifact_type", "")).startswith("text_")
            for issue in issues
        ) if isinstance(issues, list) else False
        valid_regions.append((bbox, contains_text_issue))
    for source in sources:
        source_id = str(source.get("source_finding_id", ""))
        if not source_id or source_id in handled:
            continue
        try:
            source_bbox = BBox.from_value(source.get("bbox_norm")).as_list()
        except (TypeError, ValueError):
            continue
        source_is_text = str(source.get("artifact_type", "")).startswith(
            "text_"
        )
        contained_regions = 0
        for region_bbox, contains_text_issue in valid_regions:
            if source_is_text and not contains_text_issue:
                continue
            region_area = _area(region_bbox)
            if region_area <= 0 or _area(source_bbox) <= region_area:
                continue
            containment = _intersection(source_bbox, region_bbox) / region_area
            if containment >= 0.9:
                contained_regions += 1
        if contained_regions < 2:
            continue
        discarded.append(source_id)
        discard_reasons[source_id] = (
            "该来源是覆盖多个独立编辑区域的宽泛重复框，具体问题已由更精确区域"
            "分别处理。"
        )
        handled.add(source_id)
    return output


class ResultConsolidator:
    def __init__(
        self,
        model,
        *,
        reject_nested_containers: bool = False,
        max_batch_size: int = 10,
        enforce_materiality_gate: bool = False,
    ):
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        self.model = model
        self.reject_nested_containers = reject_nested_containers
        self.max_batch_size = max_batch_size
        self.enforce_materiality_gate = enforce_materiality_gate

    async def consolidate(
        self,
        image: ImageAsset,
        result: Mapping[str, Any],
    ) -> dict[str, Any]:
        all_sources = source_findings(result)
        if not all_sources:
            return {
                "verdict": str(result.get("verdict", "no_artifact")),
                "confidence": float(result.get("confidence", 0.5)),
                "edit_regions": [],
                "discarded_finding_ids": [],
                "discard_reasons": {},
                "rationale": "原始结果中没有需要整编的局部问题。",
            }
        rejected: list[tuple[dict[str, Any], str]] = []
        sources: list[dict[str, Any]] = []
        for source in all_sources:
            reason = (
                materiality_gate_reason(source)
                if self.enforce_materiality_gate
                else None
            )
            if reason is None:
                sources.append(source)
            else:
                rejected.append((source, reason))
        if not sources:
            return {
                "verdict": "no_artifact",
                "confidence": float(result.get("confidence", 0.5)),
                "edit_regions": [],
                "discarded_finding_ids": [
                    str(source["source_finding_id"])
                    for source, _ in rejected
                ],
                "discard_reasons": {
                    str(source["source_finding_id"]): reason
                    for source, reason in rejected
                },
                "rationale": "候选问题均未通过明显核心错误门槛。",
            }
        batches = partition_source_findings(
            sources,
            max_batch_size=self.max_batch_size,
        )
        if len(batches) == 1:
            consolidated = await self._consolidate_batch(
                image,
                result,
                batches[0],
            )
            return self._attach_policy_discards(
                consolidated,
                rejected,
                all_sources,
            )
        partials = await asyncio.gather(
            *(
                self._consolidate_batch(image, result, batch)
                for batch in batches
            )
        )
        combined = self._combine_partials(partials)
        errors = validate_consolidated_result(
            combined,
            sources,
            reject_nested_containers=self.reject_nested_containers,
        )
        if not errors:
            final = combined
        else:
            final = await self._repair_combined(
                image=image,
                result=result,
                sources=sources,
                combined=combined,
                errors=errors,
            )
        return self._attach_policy_discards(
            final,
            rejected,
            all_sources,
        )

    def _attach_policy_discards(
        self,
        value: Mapping[str, Any],
        rejected: Sequence[tuple[Mapping[str, Any], str]],
        all_sources: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        output = copy.deepcopy(dict(value))
        discarded = output.setdefault("discarded_finding_ids", [])
        reasons = output.setdefault("discard_reasons", {})
        for source, reason in rejected:
            source_id = str(source["source_finding_id"])
            if source_id not in discarded:
                discarded.append(source_id)
            reasons[source_id] = reason
        errors = validate_consolidated_result(
            output,
            all_sources,
            reject_nested_containers=self.reject_nested_containers,
        )
        if errors:
            raise ConsolidationValidationError(errors)
        return output

    async def _consolidate_batch(
        self,
        image: ImageAsset,
        result: Mapping[str, Any],
        sources: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        packet = {
            "source_result": {
                "verdict": result.get("verdict"),
                "confidence": result.get("confidence"),
            },
            "source_findings": sources,
            "spatial_hints": spatial_hints(sources),
        }
        nested_container_rule = (
            "若一个大框包住任何已经独立定位的小框，必须按一次局部重绘的真实目标"
            "重新选择：保留独立小框并移除冗余大框，或在确需整体重绘时合并为一个框；"
            "禁止大框与被其包含的小框并存。"
            if self.reject_nested_containers
            else ""
        )
        initial = await self._complete(
            stage="consensus.consolidate",
            image=image,
            system_prompt=(
                "你是图像局部重绘任务的最终结果整编员。只整编已有问题，不新增诊断。"
                "所有自然语言字段使用简体中文。"
            ),
            user_prompt=(
                "按“一次局部重绘能否共同解决”整编下面的 finding。"
                "同一实体或文字块上的多类问题合并为一个 edit region，并在 issues 中"
                "完整保留全部问题；相邻但需要分别重绘的实体必须保持独立。bbox 覆盖完整"
                "编辑目标和必要上下文，允许少量冗余，不追求最紧；禁止整图框，不按面积"
                "大小拒绝独立主体。每个 source_finding_id 必须恰好进入一个 region，或"
                "进入 discarded_finding_ids 并给出中文原因。空间分离的文字标签必须"
                "分别输出 region，不得合成覆盖整片场景的大框。只保留 major/critical、"
                "置信度严格大于0.6且 verification=supported 的问题，每个 issue 必须"
                "返回 materiality。不得输出未知来源 ID。"
                + nested_container_rule
                + "\n\n"
                f"待整编数据：\n{serialize_json(packet)}"
            ),
            metadata={"source_finding_count": len(sources)},
        )
        initial = _restore_missing_issue_sources(initial, sources)
        initial = _reconcile_missing_broad_container_sources(
            initial,
            sources,
        )
        errors = validate_consolidated_result(
            initial,
            sources,
            reject_nested_containers=self.reject_nested_containers,
        )
        if not errors:
            return initial
        repair = await self._complete(
            stage="consensus.consolidate_repair",
            image=image,
            system_prompt=(
                "你是最终整编结果修正员。必须修复所有校验错误，不新增来源问题，"
                "所有自然语言字段使用简体中文。"
            ),
            user_prompt=(
                f"校验错误：\n{serialize_json(errors)}\n\n"
                f"来源数据：\n{serialize_json(packet)}\n\n"
                f"待修正结果：\n{serialize_json(initial)}\n\n"
                "返回完整修正结果。尤其确保每个来源 ID 只处理一次、同一重绘区域只保留"
                "一个 bbox、一个区域中的全部问题写入 issues，且不使用整图框。空间"
                "分离的文字实体必须拆成独立 region；只保留 major/critical 且置信度"
                "严格大于0.6的问题。"
                + nested_container_rule
            ),
            metadata={
                "source_finding_count": len(sources),
                "validation_errors": errors,
            },
        )
        repair = _restore_missing_issue_sources(repair, sources)
        repair = _reconcile_missing_broad_container_sources(
            repair,
            sources,
        )
        repaired_errors = validate_consolidated_result(
            repair,
            sources,
            reject_nested_containers=self.reject_nested_containers,
        )
        if repaired_errors:
            raise ConsolidationValidationError(repaired_errors)
        return repair

    @staticmethod
    def _combine_partials(
        partials: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        regions: list[dict[str, Any]] = []
        discarded: list[str] = []
        reasons: dict[str, str] = {}
        confidences: list[float] = []
        for partial in partials:
            for raw in partial.get("edit_regions", []):
                if isinstance(raw, Mapping):
                    region = copy.deepcopy(dict(raw))
                    region["edit_region_id"] = f"r{len(regions) + 1:03d}"
                    regions.append(region)
            raw_discarded = partial.get("discarded_finding_ids", [])
            if isinstance(raw_discarded, list):
                for source_id in map(str, raw_discarded):
                    if source_id not in discarded:
                        discarded.append(source_id)
            raw_reasons = partial.get("discard_reasons", {})
            if isinstance(raw_reasons, Mapping):
                for source_id in discarded:
                    reason = str(raw_reasons.get(source_id, "")).strip()
                    if reason:
                        reasons[source_id] = reason
            try:
                confidences.append(float(partial.get("confidence", 0.5)))
            except (TypeError, ValueError):
                pass
        return {
            "verdict": "artifact_present" if regions else "no_artifact",
            "confidence": max(confidences, default=0.5),
            "edit_regions": regions,
            "discarded_finding_ids": discarded,
            "discard_reasons": reasons,
            "rationale": (
                "先按语义目标分批整编，再合并为全图局部重绘结果。"
            ),
        }

    async def _repair_combined(
        self,
        *,
        image: ImageAsset,
        result: Mapping[str, Any],
        sources: Sequence[Mapping[str, Any]],
        combined: Mapping[str, Any],
        errors: Sequence[str],
    ) -> dict[str, Any]:
        packet = {
            "source_result": {
                "verdict": result.get("verdict"),
                "confidence": result.get("confidence"),
            },
            "source_findings": sources,
            "spatial_hints": spatial_hints(sources),
        }
        repaired = await self._complete(
            stage="consensus.consolidate_global_repair",
            image=image,
            system_prompt=(
                "你是分层整编的全图终审员，只修复已有整编结果中的重复、"
                "嵌套、来源闭环或编辑目标划分错误，不新增图像问题。所有"
                "自然语言字段使用简体中文。"
            ),
            user_prompt=(
                f"程序校验错误：\n{serialize_json(list(errors))}\n\n"
                f"完整来源：\n{serialize_json(packet)}\n\n"
                f"分批整编结果：\n{serialize_json(combined)}\n\n"
                "返回完整终审结果。任何大框都不得与其包含的小框并存；同一"
                "编辑目标的多个问题合并一个框，空间独立目标分别保留。每个"
                "source ID 必须恰好处理一次，discard_reasons 只能包含确实"
                "discarded 的 ID。"
            ),
            metadata={
                "source_finding_count": len(sources),
                "validation_errors": list(errors),
                "batch_count": len(
                    partition_source_findings(
                        sources,
                        max_batch_size=self.max_batch_size,
                    )
                ),
            },
        )
        repaired = _restore_missing_issue_sources(repaired, sources)
        repaired = _reconcile_missing_broad_container_sources(
            repaired,
            sources,
        )
        repaired_errors = validate_consolidated_result(
            repaired,
            sources,
            reject_nested_containers=self.reject_nested_containers,
        )
        if repaired_errors:
            raise ConsolidationValidationError(repaired_errors)
        return repaired

    async def _complete(
        self,
        *,
        stage: str,
        image: ImageAsset,
        system_prompt: str,
        user_prompt: str,
        metadata: Mapping[str, Any],
    ) -> dict[str, Any]:
        response = await self.model.complete(
            ModelRequest(
                stage=stage,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                images=(image,),
                response_schema=CONSOLIDATED_RESULT_SCHEMA,
                metadata={
                    "workflow": "adversarial_consensus",
                    **dict(metadata),
                },
            )
        )
        return dict(parse_response(response))
