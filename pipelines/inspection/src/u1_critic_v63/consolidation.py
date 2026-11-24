from __future__ import annotations

import asyncio
import copy
import difflib
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from critic_workflows import BBox, ImageAsset, ModelRequest
from critic_workflows.utils import parse_response, serialize_json

from .edit_consensus import is_near_full_frame
from .quality_policy import (
    CORE_CONFIDENCE_THRESHOLD,
    CORE_MATERIALITIES,
)


_CHINESE = re.compile(r"[\u4e00-\u9fff]")
_NESTED_CONTAINMENT_THRESHOLD = 0.9


@dataclass(frozen=True)
class ConsolidationConflictComponent:
    region_ids: tuple[str, ...]
    source_ids: tuple[str, ...]
    errors: tuple[str, ...]


@dataclass(frozen=True)
class ConsolidationConflictAnalysis:
    components: tuple[ConsolidationConflictComponent, ...]
    unmapped_errors: tuple[str, ...]

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

_GROUPING_EDIT_REGION_SCHEMA = copy.deepcopy(_EDIT_REGION_SCHEMA)
_GROUPING_EDIT_REGION_SCHEMA["required"] = [
    key
    for key in _GROUPING_EDIT_REGION_SCHEMA["required"]
    if key != "issues"
]
_GROUPING_EDIT_REGION_SCHEMA["properties"].pop("issues", None)
CONSOLIDATION_GROUPING_SCHEMA = copy.deepcopy(CONSOLIDATED_RESULT_SCHEMA)
CONSOLIDATION_GROUPING_SCHEMA["properties"]["edit_regions"][
    "items"
] = _GROUPING_EDIT_REGION_SCHEMA


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


def compact_consolidation_source(
    source: Mapping[str, Any],
) -> dict[str, Any]:
    """Keep every consolidation decision field while dropping trace echoes."""

    top_level_keys = (
        "source_finding_id",
        "source_status",
        "region_id",
        "bbox_norm",
        "artifact_type",
        "observation",
        "abnormality",
        "confidence",
        "verification",
    )
    output = {
        key: copy.deepcopy(source[key])
        for key in top_level_keys
        if key in source
    }
    metadata = source.get("metadata", {})
    if isinstance(metadata, Mapping):
        metadata_keys = (
            "edit_target",
            "edit_region_rationale",
            "semantic_group_id",
            "content_kind",
            "issues",
            "materiality",
            "evidence_basis",
            "grounding_status",
        )
        compact_metadata = {
            key: copy.deepcopy(metadata[key])
            for key in metadata_keys
            if key in metadata
        }
        if compact_metadata:
            output["metadata"] = compact_metadata
    return output


def canonicalize_edit_region_ids(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Normalize arbitrary model labels without changing any edit semantics."""

    output = copy.deepcopy(dict(value))
    regions = output.get("edit_regions")
    if not isinstance(regions, list):
        return output
    identifiers = [
        str(region.get("edit_region_id", "")).strip()
        for region in regions
        if isinstance(region, Mapping)
    ]
    if (
        len(identifiers) == len(regions)
        and all(identifiers)
        and len(set(identifiers)) == len(identifiers)
    ):
        return output
    for index, region in enumerate(regions, 1):
        if isinstance(region, dict):
            region["edit_region_id"] = f"r{index:03d}"
    return output


def drop_stale_discard_reasons(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Remove reasons for findings the model ultimately kept in a region."""

    output = copy.deepcopy(dict(value))
    discarded = output.get("discarded_finding_ids")
    reasons = output.get("discard_reasons")
    if not isinstance(discarded, list) or not isinstance(reasons, Mapping):
        return output
    allowed = {str(source_id) for source_id in discarded}
    output["discard_reasons"] = {
        str(source_id): copy.deepcopy(reason)
        for source_id, reason in reasons.items()
        if str(source_id) in allowed
    }
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
        # Keep an explicit semantic group intact whenever it fits in a fresh
        # batch. Splitting it merely to fill the tail of the current batch
        # makes two model calls independently create the same repaint target,
        # which then produces duplicate/nested regions during global merge.
        if (
            current
            and len(group) <= max_batch_size
            and len(group) > max_batch_size - len(current)
        ):
            batches.append(current)
            current = []
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
    left_metadata = left.get("metadata", {})
    right_metadata = right.get("metadata", {})
    if isinstance(left_metadata, Mapping) and isinstance(
        right_metadata,
        Mapping,
    ) and (
        str(left_metadata.get("semantic_unit_kind", ""))
        == "independent_label"
        or str(right_metadata.get("semantic_unit_kind", ""))
        == "independent_label"
    ):
        # Independent labels may reuse a generic model ID (for example
        # "dimensions") but remain separate repaint targets. True duplicate
        # detections still connect through their overlapping bboxes before
        # this semantic fallback is consulted.
        return False
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
    if isinstance(left_metadata, Mapping) and isinstance(
        right_metadata,
        Mapping,
    ):
        left_group = str(
            left_metadata.get("semantic_group_id", "")
        ).strip()
        right_group = str(
            right_metadata.get("semantic_group_id", "")
        ).strip()
        left_region = str(left.get("region_id", "")).strip()
        right_region = str(right.get("region_id", "")).strip()
        if (
            left_group
            and left_group == right_group
            and left_region
            and left_region == right_region
        ):
            # The coverage inventory intentionally assigns the same pair only
            # when multiple spatial fragments belong to one complete semantic
            # edit unit (for example a panel title plus its body).  Treating
            # those fragments as unrelated caused deterministic validation to
            # reject a correct one-pass repaint and trigger repeated model
            # repair loops.
            return True
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
    # Two repaint regions with at least 0.9 IoU target effectively the same
    # pixels. Even if their descriptions mention different issues or layers,
    # the edit model should receive one region with all issues attached.
    return True


def _merge_nested_cluster_with_duplicate_outer(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Merge one repaint cluster proven by two near-identical outer boxes.

    A broad box must not absorb precise children merely because it is larger.
    The safe exception is when two independent findings already select almost
    the same complete repaint footprint and every other conflicting region is
    contained by that footprint.  Repainting either outer target necessarily
    repaints all children, so keeping nested boxes would be both redundant and
    impossible for the downstream editor to execute independently.
    """

    output = copy.deepcopy(dict(value))
    raw_regions = output.get("edit_regions", [])
    if not isinstance(raw_regions, list) or len(raw_regions) < 3:
        return output
    regions = [
        dict(region) if isinstance(region, Mapping) else region
        for region in raw_regions
    ]
    changed = False
    while True:
        valid: list[tuple[int, dict[str, Any], list[int]]] = []
        for index, region in enumerate(regions):
            if not isinstance(region, dict):
                continue
            try:
                bbox = BBox.from_value(region.get("bbox_norm")).as_list()
            except (TypeError, ValueError):
                continue
            valid.append((index, region, bbox))
        selected: tuple[int, list[int], list[int]] | None = None
        for left_pos, (left_index, _, left_bbox) in enumerate(valid):
            left_area = _area(left_bbox)
            if left_area <= 0:
                continue
            for right_index, _, right_bbox in valid[left_pos + 1 :]:
                right_area = _area(right_bbox)
                if right_area <= 0:
                    continue
                intersection = _intersection(left_bbox, right_bbox)
                smaller = min(left_area, right_area)
                larger = max(left_area, right_area)
                if (
                    intersection / smaller < 0.9
                    or smaller / larger < 0.75
                ):
                    continue
                anchor_bbox = [
                    min(left_bbox[0], right_bbox[0]),
                    min(left_bbox[1], right_bbox[1]),
                    max(left_bbox[2], right_bbox[2]),
                    max(left_bbox[3], right_bbox[3]),
                ]
                anchor_area = _area(anchor_bbox)
                cluster = {left_index, right_index}
                for child_index, _, child_bbox in valid:
                    if child_index in cluster:
                        continue
                    child_area = _area(child_bbox)
                    if child_area <= 0 or child_area >= anchor_area:
                        continue
                    if (
                        _intersection(anchor_bbox, child_bbox) / child_area
                        >= _NESTED_CONTAINMENT_THRESHOLD
                    ):
                        cluster.add(child_index)
                if len(cluster) < 3:
                    continue
                merged_bbox = list(anchor_bbox)
                for child_index, _, child_bbox in valid:
                    if child_index not in cluster:
                        continue
                    merged_bbox = [
                        min(merged_bbox[0], child_bbox[0]),
                        min(merged_bbox[1], child_bbox[1]),
                        max(merged_bbox[2], child_bbox[2]),
                        max(merged_bbox[3], child_bbox[3]),
                    ]
                if is_near_full_frame(BBox.from_value(merged_bbox)):
                    continue
                base_index = (
                    left_index if left_area >= right_area else right_index
                )
                selected = (base_index, sorted(cluster), merged_bbox)
                break
            if selected is not None:
                break
        if selected is None:
            break

        base_index, cluster_indices, merged_bbox = selected
        base = regions[base_index]
        if not isinstance(base, dict):
            break
        base["bbox_norm"] = merged_bbox
        base_sources = base.get("source_finding_ids", [])
        base["source_finding_ids"] = (
            list(map(str, base_sources))
            if isinstance(base_sources, list)
            else []
        )
        base_issues = base.get("issues", [])
        base["issues"] = (
            copy.deepcopy(base_issues)
            if isinstance(base_issues, list)
            else []
        )
        for index in cluster_indices:
            if index == base_index:
                continue
            child = regions[index]
            if not isinstance(child, Mapping):
                continue
            child_sources = child.get("source_finding_ids", [])
            if isinstance(child_sources, list):
                for source_id in map(str, child_sources):
                    if source_id not in base["source_finding_ids"]:
                        base["source_finding_ids"].append(source_id)
            child_issues = child.get("issues", [])
            if isinstance(child_issues, list):
                base["issues"].extend(copy.deepcopy(child_issues))
            child_target = str(child.get("edit_target", "")).strip()
            if child_target and child_target not in str(
                base.get("edit_target", "")
            ):
                base["edit_target"] = (
                    str(base.get("edit_target", "")).rstrip("。")
                    + "，并同时重绘其中的"
                    + child_target
                ).lstrip("，")
            try:
                base["confidence"] = max(
                    float(base.get("confidence", 0.0)),
                    float(child.get("confidence", 0.0)),
                )
            except (TypeError, ValueError):
                pass
        base["edit_region_rationale"] = (
            str(base.get("edit_region_rationale", "")).rstrip("。")
            + "；两个独立候选确认同一完整重绘范围，已将其包含的局部问题"
            "并入一次重绘。"
        ).lstrip("；")
        for index in sorted(cluster_indices, reverse=True):
            if index != base_index:
                regions.pop(index)
        changed = True

    output["edit_regions"] = regions
    if changed:
        output["rationale"] = (
            str(output.get("rationale", "")).rstrip("。")
            + "；已确定性合并由重复外层证据确认的单次重绘区域。"
        ).lstrip("；")
    return output


def _merge_duplicate_edit_regions(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Coalesce near-identical repaint regions before strict validation."""

    output = _merge_nested_cluster_with_duplicate_outer(value)
    raw_regions = output.get("edit_regions", [])
    if not isinstance(raw_regions, list) or len(raw_regions) < 2:
        return output
    regions = [
        dict(region) if isinstance(region, Mapping) else region
        for region in raw_regions
    ]
    changed = False
    left_index = 0
    while left_index < len(regions):
        left = regions[left_index]
        if not isinstance(left, dict):
            left_index += 1
            continue
        right_index = left_index + 1
        while right_index < len(regions):
            right = regions[right_index]
            if not (
                isinstance(right, Mapping)
                and is_duplicate_edit_region(left, right)
            ):
                right_index += 1
                continue
            try:
                left_bbox = BBox.from_value(left.get("bbox_norm"))
                right_bbox = BBox.from_value(right.get("bbox_norm"))
            except (TypeError, ValueError):
                right_index += 1
                continue
            left["bbox_norm"] = [
                min(left_bbox.x1, right_bbox.x1),
                min(left_bbox.y1, right_bbox.y1),
                max(left_bbox.x2, right_bbox.x2),
                max(left_bbox.y2, right_bbox.y2),
            ]
            left_sources = left.get("source_finding_ids", [])
            right_sources = right.get("source_finding_ids", [])
            left["source_finding_ids"] = list(
                dict.fromkeys(
                    [
                        *(
                            map(str, left_sources)
                            if isinstance(left_sources, list)
                            else []
                        ),
                        *(
                            map(str, right_sources)
                            if isinstance(right_sources, list)
                            else []
                        ),
                    ]
                )
            )
            left_issues = left.get("issues", [])
            right_issues = right.get("issues", [])
            left["issues"] = [
                *(
                    copy.deepcopy(left_issues)
                    if isinstance(left_issues, list)
                    else []
                ),
                *(
                    copy.deepcopy(right_issues)
                    if isinstance(right_issues, list)
                    else []
                ),
            ]
            right_target = str(right.get("edit_target", "")).strip()
            if right_target and right_target not in str(
                left.get("edit_target", "")
            ):
                left["edit_target"] = (
                    str(left.get("edit_target", "")).rstrip("。")
                    + "，并同时重绘"
                    + right_target
                ).lstrip("，")
            left["edit_region_rationale"] = (
                str(left.get("edit_region_rationale", "")).rstrip("。")
                + "；两个候选覆盖同一重绘区域，已合并全部问题。"
            ).lstrip("；")
            try:
                left["confidence"] = max(
                    float(left.get("confidence", 0.0)),
                    float(right.get("confidence", 0.0)),
                )
            except (TypeError, ValueError):
                pass
            regions.pop(right_index)
            changed = True
        left_index += 1
    output["edit_regions"] = regions
    if changed:
        output["rationale"] = (
            str(output.get("rationale", "")).rstrip("。")
            + "；已合并近乎重合的重复局部重绘区域。"
        ).lstrip("；")
    return output


def _nonempty_chinese(value: Any) -> bool:
    return bool(str(value).strip()) and bool(_CHINESE.search(str(value)))


def _sources_with_online_atomization(
    value: Mapping[str, Any],
    sources: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Expand parent findings according to an online model split lineage."""

    raw_mapping = value.get("online_source_atomization", {})
    if not isinstance(raw_mapping, Mapping):
        return [copy.deepcopy(dict(source)) for source in sources]
    output: list[dict[str, Any]] = []
    for raw_source in sources:
        source = copy.deepcopy(dict(raw_source))
        parent_id = str(source.get("source_finding_id", ""))
        raw_children = raw_mapping.get(parent_id)
        if not isinstance(raw_children, list) or not raw_children:
            output.append(source)
            continue
        for raw_child in raw_children:
            if not isinstance(raw_child, Mapping):
                continue
            child_id = str(raw_child.get("source_finding_id", "")).strip()
            if not child_id:
                continue
            child = copy.deepcopy(source)
            child["source_finding_id"] = child_id
            child["bbox_norm"] = copy.deepcopy(raw_child.get("bbox_norm"))
            metadata = child.setdefault("metadata", {})
            if not isinstance(metadata, dict):
                metadata = {}
                child["metadata"] = metadata
            metadata["source_atomized_from"] = parent_id
            metadata["edit_target"] = str(
                raw_child.get("edit_target", "")
            ).strip()
            metadata["edit_region_rationale"] = str(
                raw_child.get("edit_region_rationale", "")
            ).strip()
            output.append(child)
    return output


def atomize_disjoint_repeated_sources(
    value: Mapping[str, Any],
    sources: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, list[str]]] | None:
    """Turn a model's valid disjoint split of one broad source into children."""

    raw_regions = value.get("edit_regions", [])
    if not isinstance(raw_regions, list):
        return None
    regions = [
        copy.deepcopy(dict(region))
        for region in raw_regions
        if isinstance(region, Mapping)
    ]
    if len(regions) != len(raw_regions):
        return None
    source_by_id = {
        str(source.get("source_finding_id", "")): source
        for source in sources
        if isinstance(source, Mapping)
        and str(source.get("source_finding_id", "")).strip()
    }
    memberships: dict[str, list[int]] = {}
    for index, region in enumerate(regions):
        raw_ids = region.get("source_finding_ids", [])
        if not isinstance(raw_ids, list):
            return None
        for source_id in map(str, raw_ids):
            memberships.setdefault(source_id, []).append(index)
    repeated = {
        source_id: indexes
        for source_id, indexes in memberships.items()
        if len(indexes) > 1
    }
    if not repeated or any(
        source_id not in source_by_id for source_id in repeated
    ):
        return None
    raw_discarded = value.get("discarded_finding_ids", [])
    discarded = (
        set(map(str, raw_discarded))
        if isinstance(raw_discarded, list)
        else set()
    )
    if any(source_id in discarded for source_id in repeated):
        return None

    lineage: dict[str, list[str]] = {}
    mapping: dict[str, list[dict[str, Any]]] = {}
    for parent_id, indexes in repeated.items():
        boxes: list[list[int]] = []
        for index in indexes:
            try:
                boxes.append(
                    BBox.from_value(
                        regions[index].get("bbox_norm")
                    ).as_list()
                )
            except (TypeError, ValueError):
                return None
        for left_index, left in enumerate(boxes):
            for right in boxes[left_index + 1 :]:
                if _intersection(left, right) > 0:
                    return None
        children: list[dict[str, Any]] = []
        child_ids: list[str] = []
        for atom_index, (region_index, bbox) in enumerate(
            zip(indexes, boxes),
            start=1,
        ):
            region = regions[region_index]
            child_id = f"{parent_id}:a{atom_index:02d}"
            child_ids.append(child_id)
            children.append(
                {
                    "source_finding_id": child_id,
                    "bbox_norm": bbox,
                    "edit_target": str(region.get("edit_target", "")).strip(),
                    "edit_region_rationale": str(
                        region.get("edit_region_rationale", "")
                    ).strip(),
                }
            )
            region["source_finding_ids"] = [
                child_id if str(source_id) == parent_id else str(source_id)
                for source_id in region.get("source_finding_ids", [])
            ]
            raw_issues = region.get("issues", [])
            if isinstance(raw_issues, list):
                for issue_value in raw_issues:
                    if not isinstance(issue_value, dict):
                        continue
                    raw_issue_ids = issue_value.get("source_finding_ids", [])
                    if isinstance(raw_issue_ids, list):
                        issue_value["source_finding_ids"] = [
                            child_id
                            if str(source_id) == parent_id
                            else str(source_id)
                            for source_id in raw_issue_ids
                        ]
        lineage[parent_id] = child_ids
        mapping[parent_id] = children

    output = copy.deepcopy(dict(value))
    output["edit_regions"] = regions
    existing_mapping = output.get("online_source_atomization", {})
    merged_mapping = (
        copy.deepcopy(dict(existing_mapping))
        if isinstance(existing_mapping, Mapping)
        else {}
    )
    merged_mapping.update(mapping)
    output["online_source_atomization"] = merged_mapping
    output["consolidation_mode"] = "online_source_atomization"
    expanded_sources = _sources_with_online_atomization(output, sources)
    return output, expanded_sources, lineage


def text_sources_are_atomic(
    source_ids: Sequence[str],
    source_by_id: Mapping[str, Mapping[str, Any]],
    region_bbox: BBox | None = None,
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
    if len(connected) == len(boxes):
        return True

    # Spatially separate text labels may share one edit region only when a
    # supported non-text source independently requires repainting that whole
    # region. This is the structural proof used by the duplicate-outer cluster
    # merger above; without it, independent text labels remain separate.
    if region_bbox is None:
        return False
    region_values = region_bbox.as_list()
    region_area = _area(region_values)
    if region_area <= 0:
        return False
    non_text_sources = [
        source_by_id[source_id]
        for source_id in source_ids
        if source_id in source_by_id
        and not str(
            source_by_id[source_id].get("artifact_type", "")
        ).startswith("text_")
    ]
    for source in non_text_sources:
        try:
            outer_bbox = BBox.from_value(source.get("bbox_norm")).as_list()
        except (TypeError, ValueError):
            continue
        outer_area = _area(outer_bbox)
        if outer_area <= 0:
            continue
        if (
            _intersection(region_values, outer_bbox)
            / max(region_area, outer_area)
            < 0.85
        ):
            continue
        if all(
            _intersection(outer_bbox, text_bbox)
            / max(_area(text_bbox), 1.0)
            >= _NESTED_CONTAINMENT_THRESHOLD
            for text_bbox in boxes
        ):
            return True
    return False


def _split_spatially_disconnected_text_regions(
    value: Mapping[str, Any],
    sources: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Split only model regions that provably join disconnected text sources.

    This is a post-model structural safeguard, not semantic consolidation:
    it never joins sources or invents findings. A region is split only when
    all of its sources are text findings and the same connectivity test used
    by strict validation proves that they form multiple components.
    """

    output = copy.deepcopy(dict(value))
    raw_regions = output.get("edit_regions", [])
    if not isinstance(raw_regions, list):
        return output
    source_by_id = {
        str(source.get("source_finding_id")): source
        for source in sources
        if str(source.get("source_finding_id", "")).strip()
    }
    rebuilt: list[Any] = []
    changed = False
    for raw_region in raw_regions:
        if not isinstance(raw_region, Mapping):
            rebuilt.append(raw_region)
            continue
        region = copy.deepcopy(dict(raw_region))
        raw_ids = region.get("source_finding_ids", [])
        source_ids = (
            list(dict.fromkeys(map(str, raw_ids)))
            if isinstance(raw_ids, list)
            else []
        )
        region_sources = [
            source_by_id[source_id]
            for source_id in source_ids
            if source_id in source_by_id
        ]
        if (
            len(region_sources) != len(source_ids)
            or len(region_sources) <= 1
            or any(
                not str(source.get("artifact_type", "")).startswith("text_")
                for source in region_sources
            )
            or text_sources_are_atomic(source_ids, source_by_id)
        ):
            rebuilt.append(region)
            continue

        components: list[list[str]] = []
        remaining = set(source_ids)
        while remaining:
            seed = next(source_id for source_id in source_ids if source_id in remaining)
            component = {seed}
            remaining.remove(seed)
            expanded = True
            while expanded:
                expanded = False
                for candidate_id in list(remaining):
                    candidate = source_by_id[candidate_id]
                    if any(
                        text_sources_are_atomic(
                            [member_id, candidate_id],
                            source_by_id,
                        )
                        for member_id in component
                    ):
                        component.add(candidate_id)
                        remaining.remove(candidate_id)
                        expanded = True
            components.append(
                [source_id for source_id in source_ids if source_id in component]
            )
        if len(components) <= 1:
            rebuilt.append(region)
            continue

        original_issues = region.get("issues", [])
        for component_index, component_ids in enumerate(components, start=1):
            component_set = set(component_ids)
            component_sources = [source_by_id[item] for item in component_ids]
            boxes = [BBox.from_value(source["bbox_norm"]) for source in component_sources]
            split_region = copy.deepcopy(region)
            split_region["edit_region_id"] = (
                str(region.get("edit_region_id", "region"))
                if component_index == 1
                else f"{region.get('edit_region_id', 'region')}_s{component_index}"
            )
            split_region["bbox_norm"] = [
                min(box.x1 for box in boxes),
                min(box.y1 for box in boxes),
                max(box.x2 for box in boxes),
                max(box.y2 for box in boxes),
            ]
            split_region["source_finding_ids"] = component_ids
            targets = list(
                dict.fromkeys(
                    target
                    for target in (
                        _source_edit_target(source) for source in component_sources
                    )
                    if target
                )
            )
            if targets:
                split_region["edit_target"] = "、".join(targets)
            split_region["edit_region_rationale"] = (
                "原模型整编区域包含空间不连通的文字实体；按独立局部重绘目标拆分，"
                "保留该实体已有问题和原始定位范围。"
            )
            split_issues: list[dict[str, Any]] = []
            if isinstance(original_issues, list):
                for raw_issue in original_issues:
                    if not isinstance(raw_issue, Mapping):
                        continue
                    issue_ids = raw_issue.get("source_finding_ids", [])
                    if not isinstance(issue_ids, list):
                        continue
                    retained = [
                        str(source_id)
                        for source_id in issue_ids
                        if str(source_id) in component_set
                    ]
                    if not retained:
                        continue
                    issue = copy.deepcopy(dict(raw_issue))
                    issue["source_finding_ids"] = retained
                    split_issues.append(issue)
            split_region["issues"] = split_issues
            rebuilt.append(split_region)
        changed = True
    output["edit_regions"] = rebuilt
    if changed:
        output["rationale"] = (
            str(output.get("rationale", "")).rstrip("。")
            + "；已将模型误合并的空间不连通文字实体拆为独立重绘区域。"
        ).lstrip("；")
    return output


def validate_consolidated_result(
    value: Mapping[str, Any],
    source_findings: Sequence[Mapping[str, Any]],
    *,
    reject_nested_containers: bool = False,
) -> list[str]:
    errors: list[str] = []
    source_findings = _sources_with_online_atomization(
        value,
        source_findings,
    )
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
        bbox: BBox | None = None
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
            bbox,
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
                if containment >= _NESTED_CONTAINMENT_THRESHOLD:
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


def build_consolidation_conflict_components(
    errors: Sequence[str],
    value: Mapping[str, Any],
    source_findings: Sequence[Mapping[str, Any]],
) -> ConsolidationConflictAnalysis:
    """Map strict validation failures to conservative repair components.

    Any error that cannot be tied to known regions or source findings remains
    unmapped and therefore forces the caller onto the full global repair path.
    """

    known_sources = {
        str(source.get("source_finding_id", "")).strip()
        for source in source_findings
        if isinstance(source, Mapping)
        and str(source.get("source_finding_id", "")).strip()
    }
    raw_regions = value.get("edit_regions", [])
    regions = (
        [region for region in raw_regions if isinstance(region, Mapping)]
        if isinstance(raw_regions, list)
        else []
    )
    region_ids = {
        str(region.get("edit_region_id", "")).strip()
        for region in regions
        if str(region.get("edit_region_id", "")).strip()
    }
    adjacency: dict[str, set[str]] = {}

    def region_node(region_id: str) -> str:
        return f"r:{region_id}"

    def source_node(source_id: str) -> str:
        return f"s:{source_id}"

    def connect(left: str, right: str) -> None:
        adjacency.setdefault(left, set()).add(right)
        adjacency.setdefault(right, set()).add(left)

    for region in regions:
        region_id = str(region.get("edit_region_id", "")).strip()
        if not region_id:
            continue
        node = region_node(region_id)
        adjacency.setdefault(node, set())
        raw_source_ids = region.get("source_finding_ids", [])
        if not isinstance(raw_source_ids, list):
            continue
        for raw_source_id in raw_source_ids:
            source_id = str(raw_source_id)
            if source_id in known_sources:
                connect(node, source_node(source_id))
    for source_id in known_sources:
        adjacency.setdefault(source_node(source_id), set())

    mapped_errors: list[tuple[str, set[str]]] = []
    unmapped_errors: list[str] = []
    duplicate_pattern = re.compile(
        r"^duplicate edit regions: (\S+) and (\S+)$"
    )
    nested_pattern = re.compile(
        r"^redundant nested edit container: (\S+) contains (.+)$"
    )
    region_pattern = re.compile(r"^edit region (\S+)")
    duplicate_id_pattern = re.compile(r"^duplicate edit_region_id: (\S+)$")
    source_patterns = (
        re.compile(r"^source finding id not handled: (\S+)$"),
        re.compile(r"^source finding id appears more than once: (\S+)$"),
        re.compile(r"^issue references id outside its edit region: (\S+)$"),
        re.compile(
            r"^edit region source id is not represented by an issue: (\S+)$"
        ),
        re.compile(
            r"^discarded source finding id has no Chinese reason: (\S+)$"
        ),
        re.compile(r"^discard reason references non-discarded id: (\S+)$"),
    )

    for raw_error in errors:
        error = str(raw_error)
        seeds: set[str] = set()
        duplicate_match = duplicate_pattern.match(error)
        nested_match = nested_pattern.match(error)
        duplicate_id_match = duplicate_id_pattern.match(error)
        region_match = region_pattern.match(error)
        if duplicate_match:
            left, right = duplicate_match.groups()
            if left in region_ids and right in region_ids:
                left_node = region_node(left)
                right_node = region_node(right)
                connect(left_node, right_node)
                seeds.update((left_node, right_node))
        elif nested_match:
            outer, raw_inner = nested_match.groups()
            inner_ids = [item.strip() for item in raw_inner.split(",")]
            if outer in region_ids and all(
                item in region_ids for item in inner_ids
            ):
                outer_node = region_node(outer)
                seeds.add(outer_node)
                for inner in inner_ids:
                    inner_node = region_node(inner)
                    connect(outer_node, inner_node)
                    seeds.add(inner_node)
        elif duplicate_id_match:
            region_id = duplicate_id_match.group(1)
            if region_id in region_ids:
                seeds.add(region_node(region_id))
        elif region_match:
            region_id = region_match.group(1)
            if region_id in region_ids:
                seeds.add(region_node(region_id))
        else:
            for pattern in source_patterns:
                source_match = pattern.match(error)
                if source_match:
                    source_id = source_match.group(1)
                    if source_id in known_sources:
                        seeds.add(source_node(source_id))
                    break
        if seeds:
            mapped_errors.append((error, seeds))
        else:
            unmapped_errors.append(error)

    node_to_component: dict[str, frozenset[str]] = {}
    for node in adjacency:
        if node in node_to_component:
            continue
        connected = {node}
        frontier = [node]
        while frontier:
            current = frontier.pop()
            for neighbor in adjacency.get(current, ()):
                if neighbor not in connected:
                    connected.add(neighbor)
                    frontier.append(neighbor)
        frozen = frozenset(connected)
        for member in connected:
            node_to_component[member] = frozen

    errors_by_component: dict[frozenset[str], list[str]] = {}
    for error, seeds in mapped_errors:
        closures = {
            node_to_component.get(seed, frozenset((seed,)))
            for seed in seeds
        }
        if len(closures) != 1:
            unmapped_errors.append(error)
            continue
        closure = next(iter(closures))
        errors_by_component.setdefault(closure, []).append(error)

    components = [
        ConsolidationConflictComponent(
            region_ids=tuple(
                sorted(node[2:] for node in nodes if node.startswith("r:"))
            ),
            source_ids=tuple(
                sorted(node[2:] for node in nodes if node.startswith("s:"))
            ),
            errors=tuple(component_errors),
        )
        for nodes, component_errors in errors_by_component.items()
    ]
    components.sort(
        key=lambda item: (
            item.region_ids[0] if item.region_ids else "",
            item.source_ids[0] if item.source_ids else "",
        )
    )
    return ConsolidationConflictAnalysis(
        components=tuple(components),
        unmapped_errors=tuple(unmapped_errors),
    )


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


def hydrate_issues_from_sources(
    value: Mapping[str, Any],
    sources: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Attach exact accepted issue evidence after model-owned region grouping."""

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
        raw_source_ids = region.get("source_finding_ids", [])
        source_ids = (
            [str(item) for item in raw_source_ids]
            if isinstance(raw_source_ids, list)
            else []
        )
        hydrated: list[dict[str, Any]] = []
        hydrated_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
        for source_id in source_ids:
            source = source_by_id.get(source_id)
            if source is None:
                continue
            metadata = source.get("metadata", {})
            if not isinstance(metadata, Mapping):
                metadata = {}
            raw_issues = metadata.get("issues", [])
            issue_values = (
                [item for item in raw_issues if isinstance(item, Mapping)]
                if isinstance(raw_issues, list)
                else []
            )
            if not issue_values:
                issue_values = [source]
            default_materiality = str(
                metadata.get("materiality", "major")
            ).lower()
            if default_materiality not in CORE_MATERIALITIES:
                default_materiality = "major"
            for raw_issue in issue_values:
                try:
                    confidence = float(
                        raw_issue.get(
                            "confidence",
                            source.get("confidence", 0.0),
                        )
                    )
                except (TypeError, ValueError):
                    confidence = float(source.get("confidence", 0.0))
                materiality = str(
                    raw_issue.get("materiality", default_materiality)
                ).lower()
                if materiality not in CORE_MATERIALITIES:
                    materiality = default_materiality
                artifact_type = str(
                    raw_issue.get(
                        "artifact_type",
                        source.get("artifact_type", ""),
                    )
                )
                observation = str(
                    raw_issue.get(
                        "observation",
                        source.get("observation", ""),
                    )
                ).strip()
                abnormality = str(
                    raw_issue.get(
                        "abnormality",
                        source.get("abnormality", ""),
                    )
                ).strip()
                key = (
                    artifact_type,
                    " ".join(observation.split()),
                    " ".join(abnormality.split()),
                )
                existing = hydrated_by_key.get(key)
                if existing is not None:
                    if source_id not in existing["source_finding_ids"]:
                        existing["source_finding_ids"].append(source_id)
                    existing["confidence"] = max(
                        float(existing["confidence"]),
                        confidence,
                    )
                    continue
                issue = {
                    "artifact_type": artifact_type,
                    "observation": observation,
                    "abnormality": abnormality,
                    "confidence": confidence,
                    "materiality": materiality,
                    "source_finding_ids": [source_id],
                }
                hydrated_by_key[key] = issue
                hydrated.append(issue)
        region["issues"] = hydrated
    return output


def hydrate_edit_region_text_from_sources(
    value: Mapping[str, Any],
    sources: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Recover presentational text without changing model grouping or bboxes."""

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
        raw_ids = region.get("source_finding_ids", [])
        source_ids = (
            [str(item) for item in raw_ids]
            if isinstance(raw_ids, list)
            else []
        )
        assigned = [
            source_by_id[source_id]
            for source_id in source_ids
            if source_id in source_by_id
        ]
        if not _nonempty_chinese(region.get("edit_target")):
            candidates: list[str] = []
            for source in assigned:
                candidate = _source_edit_target(source)
                if _nonempty_chinese(candidate) and candidate not in candidates:
                    candidates.append(candidate)
            region["edit_target"] = (
                "、".join(candidates)
                if candidates
                else "待局部重绘的已验证问题区域"
            )
        if not _nonempty_chinese(region.get("edit_region_rationale")):
            rationales = []
            for source in assigned:
                metadata = source.get("metadata", {})
                if not isinstance(metadata, Mapping):
                    continue
                rationale = str(
                    metadata.get("edit_region_rationale", "")
                ).strip()
                if _nonempty_chinese(rationale) and rationale not in rationales:
                    rationales.append(rationale)
            region["edit_region_rationale"] = (
                rationales[0]
                if rationales
                else "该区域对应已验证的独立编辑目标，可通过一次局部重绘修复。"
            )
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


def _prune_redundant_nested_text_containers(
    value: Mapping[str, Any],
    sources: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Drop a broad text-only region when precise text regions supersede it.

    This is intentionally narrower than generic nested-box repair.  It only
    handles an outer region whose every source finding is itself a broad text
    box containing at least two independently consolidated text regions.  A
    mixed text/artifact region or a large region with a unique local source is
    left to the model so code cannot erase a legitimate large edit target.
    """

    output = copy.deepcopy(dict(value))
    regions = output.get("edit_regions", [])
    if not isinstance(regions, list) or len(regions) < 3:
        return output
    source_by_id = {
        str(source.get("source_finding_id")): source
        for source in sources
        if str(source.get("source_finding_id", "")).strip()
    }
    valid: list[tuple[int, Mapping[str, Any], list[int]]] = []
    for index, region in enumerate(regions):
        if not isinstance(region, Mapping):
            continue
        try:
            bbox = BBox.from_value(region.get("bbox_norm")).as_list()
        except (TypeError, ValueError):
            continue
        valid.append((index, region, bbox))

    remove_indices: set[int] = set()
    discard_ids: list[str] = []
    for outer_index, outer, outer_bbox in valid:
        outer_source_ids = (
            [str(item) for item in outer.get("source_finding_ids", [])]
            if isinstance(outer.get("source_finding_ids", []), list)
            else []
        )
        outer_sources = [
            source_by_id[source_id]
            for source_id in outer_source_ids
            if source_id in source_by_id
        ]
        if (
            not outer_sources
            or len(outer_sources) != len(outer_source_ids)
            or any(
                not str(source.get("artifact_type", "")).startswith("text_")
                for source in outer_sources
            )
        ):
            continue

        precise_text_boxes: list[list[int]] = []
        for inner_index, inner, inner_bbox in valid:
            if inner_index == outer_index or inner_index in remove_indices:
                continue
            inner_issues = inner.get("issues", [])
            if not (
                isinstance(inner_issues, list)
                and inner_issues
                and all(
                    isinstance(issue, Mapping)
                    and str(issue.get("artifact_type", "")).startswith("text_")
                    for issue in inner_issues
                )
            ):
                continue
            inner_area = _area(inner_bbox)
            if inner_area <= 0 or _area(outer_bbox) <= inner_area:
                continue
            if _intersection(outer_bbox, inner_bbox) / inner_area >= 0.9:
                precise_text_boxes.append(inner_bbox)
        if len(precise_text_boxes) < 2:
            continue

        every_source_is_broad = True
        for source in outer_sources:
            try:
                source_bbox = BBox.from_value(
                    source.get("bbox_norm")
                ).as_list()
            except (TypeError, ValueError):
                every_source_is_broad = False
                break
            contained_count = sum(
                1
                for precise_bbox in precise_text_boxes
                if _intersection(source_bbox, precise_bbox)
                / max(_area(precise_bbox), 1.0)
                >= 0.9
            )
            if contained_count < 2:
                every_source_is_broad = False
                break
        if every_source_is_broad:
            remove_indices.add(outer_index)
            discard_ids.extend(outer_source_ids)

    if not remove_indices:
        return output
    output["edit_regions"] = [
        region for index, region in enumerate(regions) if index not in remove_indices
    ]
    discarded = output.setdefault("discarded_finding_ids", [])
    reasons = output.setdefault("discard_reasons", {})
    for source_id in discard_ids:
        if source_id not in discarded:
            discarded.append(source_id)
        reasons[source_id] = (
            "该来源是覆盖多个独立文字编辑区域的宽泛重复框，具体问题已由更精确"
            "的文字区域分别处理。"
        )
    output["rationale"] = (
        str(output.get("rationale", "")).rstrip("。")
        + "；已确定性移除被多个精确文字区域取代的宽泛重复框。"
    ).lstrip("；")
    return output


def _refine_nested_regions_from_precise_sources(
    value: Mapping[str, Any],
    sources: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Replace a duplicate-source union box with its precise grounded source.

    A model sometimes merges two findings for the same target by taking their
    broader bbox even when the narrower source bbox is the correct edit area.
    Refinement is allowed only when every source bbox in that region is
    spatially duplicate (one covers at least 75% of the smaller box) and a
    grounded source bbox removes the nesting conflict.  Distinct subregions of
    a genuinely large entity therefore remain model decisions.
    """

    output = copy.deepcopy(dict(value))
    regions = output.get("edit_regions", [])
    if not isinstance(regions, list) or len(regions) < 2:
        return output
    source_by_id = {
        str(source.get("source_finding_id")): source
        for source in sources
        if str(source.get("source_finding_id", "")).strip()
    }
    changed = False
    for outer_index, outer in enumerate(regions):
        if not isinstance(outer, dict):
            continue
        source_ids = (
            [str(item) for item in outer.get("source_finding_ids", [])]
            if isinstance(outer.get("source_finding_ids", []), list)
            else []
        )
        if len(source_ids) < 2 or any(
            source_id not in source_by_id for source_id in source_ids
        ):
            continue
        source_boxes: list[list[int]] = []
        for source_id in source_ids:
            try:
                source_boxes.append(
                    BBox.from_value(
                        source_by_id[source_id].get("bbox_norm")
                    ).as_list()
                )
            except (TypeError, ValueError):
                source_boxes = []
                break
        if len(source_boxes) < 2:
            continue
        spatially_duplicate = all(
            _intersection(left, right)
            / max(min(_area(left), _area(right)), 1.0)
            >= 0.75
            for left_index, left in enumerate(source_boxes)
            for right in source_boxes[left_index + 1 :]
        )
        if not spatially_duplicate:
            continue
        other_boxes: list[list[int]] = []
        for inner_index, inner in enumerate(regions):
            if inner_index == outer_index or not isinstance(inner, Mapping):
                continue
            try:
                other_boxes.append(
                    BBox.from_value(inner.get("bbox_norm")).as_list()
                )
            except (TypeError, ValueError):
                pass
        try:
            current_bbox = BBox.from_value(outer.get("bbox_norm")).as_list()
        except (TypeError, ValueError):
            continue
        current_contains_other = any(
            _area(current_bbox) > _area(other)
            and _intersection(current_bbox, other)
            / max(_area(other), 1.0)
            >= _NESTED_CONTAINMENT_THRESHOLD
            for other in other_boxes
        )
        if not current_contains_other:
            continue
        for candidate in sorted(source_boxes, key=_area):
            try:
                candidate_bbox = BBox.from_value(candidate)
            except (TypeError, ValueError):
                continue
            if is_near_full_frame(candidate_bbox):
                continue
            candidate_contains_other = any(
                _area(candidate) > _area(other)
                and _intersection(candidate, other)
                / max(_area(other), 1.0)
                >= _NESTED_CONTAINMENT_THRESHOLD
                for other in other_boxes
            )
            if candidate_contains_other:
                continue
            outer["bbox_norm"] = candidate
            outer["edit_region_rationale"] = (
                str(outer.get("edit_region_rationale", "")).rstrip("。")
                + "；采用同一编辑目标中更精确的已验证来源框，避免与独立区域嵌套。"
            ).lstrip("；")
            changed = True
            break
    if changed:
        output["rationale"] = (
            str(output.get("rationale", "")).rstrip("。")
            + "；已用精确来源框修正重复来源合并造成的嵌套。"
        ).lstrip("；")
    return output


def _merge_nested_repaint_regions(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Merge nested children when the outer defect requires one repaint.

    A text-only outer container may absorb a sole nested text child, but never
    several independent children: in that case the broad text container must
    be pruned and the precise children retained.  An independently defective
    non-text entity may absorb contained text children because editing that
    entity already repaints those pixels.  A supported structural, topology,
    or relational/physical outer defect likewise absorbs every nested issue:
    repairing the malformed whole necessarily repaints those child pixels, so
    keeping separate boxes would violate the edit-model contract.  Rendering-
    only artifact nesting remains a model decision because it can represent
    distinct objects that merely overlap in the image plane.
    """

    output = copy.deepcopy(dict(value))
    regions = output.get("edit_regions", [])
    if not isinstance(regions, list) or len(regions) < 2:
        return output
    changed = False
    while True:
        valid: list[tuple[int, dict[str, Any], list[int]]] = []
        for index, region in enumerate(regions):
            if not isinstance(region, dict):
                continue
            try:
                bbox = BBox.from_value(region.get("bbox_norm")).as_list()
            except (TypeError, ValueError):
                continue
            valid.append((index, region, bbox))
        merged = False
        for outer_index, outer, outer_bbox in valid:
            contained: list[tuple[int, dict[str, Any]]] = []
            for inner_index, inner, inner_bbox in valid:
                if inner_index == outer_index:
                    continue
                if _area(outer_bbox) <= _area(inner_bbox):
                    continue
                containment = _intersection(outer_bbox, inner_bbox) / max(
                    _area(inner_bbox),
                    1.0,
                )
                if containment >= _NESTED_CONTAINMENT_THRESHOLD:
                    contained.append((inner_index, inner))
            if not contained:
                continue
            outer_issues = outer.get("issues", [])
            if not (
                isinstance(outer_issues, list)
                and outer_issues
            ):
                continue
            outer_text_only = all(
                isinstance(issue, Mapping)
                and str(issue.get("artifact_type", "")).startswith("text_")
                for issue in outer_issues
            )
            outer_requires_whole_repaint = any(
                isinstance(issue, Mapping)
                and str(issue.get("artifact_type", ""))
                in {
                    "structural_artifact",
                    "topology_artifact",
                    "relational_physical_artifact",
                }
                for issue in outer_issues
            )
            contained_details: list[
                tuple[int, dict[str, Any], list[Mapping[str, Any]]]
            ] = []
            for inner_index, inner in contained:
                inner_issues = inner.get("issues", [])
                if not (
                    isinstance(inner_issues, list)
                    and inner_issues
                    and all(isinstance(issue, Mapping) for issue in inner_issues)
                ):
                    contained_details = []
                    break
                contained_details.append(
                    (inner_index, inner, list(inner_issues))
                )
            if not contained_details:
                continue
            inner_text_only = all(
                all(
                    str(issue.get("artifact_type", "")).startswith("text_")
                    for issue in inner_issues
                )
                for _, _, inner_issues in contained_details
            )
            if len(contained_details) > 1 and outer_text_only:
                # Keep the independent text children; the broad text-only
                # outer box is handled by deterministic pruning below.
                continue
            if not (
                outer_text_only
                or inner_text_only
                or outer_requires_whole_repaint
            ):
                continue
            outer_sources = outer.setdefault("source_finding_ids", [])
            if not isinstance(outer_sources, list):
                continue
            merged_indexes: list[int] = []
            for inner_index, inner, inner_issues in contained_details:
                inner_sources = inner.get("source_finding_ids", [])
                if not isinstance(inner_sources, list):
                    merged_indexes = []
                    break
                for source_id in map(str, inner_sources):
                    if source_id not in outer_sources:
                        outer_sources.append(source_id)
                try:
                    inner_bbox = BBox.from_value(inner.get("bbox_norm"))
                    outer_bbox = [
                        min(outer_bbox[0], inner_bbox.x1),
                        min(outer_bbox[1], inner_bbox.y1),
                        max(outer_bbox[2], inner_bbox.x2),
                        max(outer_bbox[3], inner_bbox.y2),
                    ]
                    outer["bbox_norm"] = outer_bbox
                except (TypeError, ValueError):
                    pass
                outer_issues.extend(copy.deepcopy(inner_issues))
                try:
                    outer["confidence"] = max(
                        float(outer.get("confidence", 0.0)),
                        float(inner.get("confidence", 0.0)),
                    )
                except (TypeError, ValueError):
                    pass
                inner_target = str(inner.get("edit_target", "")).strip()
                if inner_target and inner_target not in str(
                    outer.get("edit_target", "")
                ):
                    outer["edit_target"] = (
                        str(outer.get("edit_target", "")).rstrip("。")
                        + "，并同时重绘其中的"
                        + inner_target
                    )
                merged_indexes.append(inner_index)
            if not merged_indexes:
                continue
            outer["edit_region_rationale"] = (
                str(outer.get("edit_region_rationale", "")).rstrip("。")
                + "；内部问题与该区域的整体重绘空间不可分，由同一次局部重绘"
                "共同修复。"
            ).lstrip("；")
            for inner_index in sorted(merged_indexes, reverse=True):
                regions.pop(inner_index)
            changed = True
            merged = True
            break
        if not merged:
            break
    output["edit_regions"] = regions
    if changed:
        output["rationale"] = (
            str(output.get("rationale", "")).rstrip("。")
            + "；已将与外层整体修复空间不可分的嵌套问题并入同一次局部重绘。"
        ).lstrip("；")
    return output


class ResultConsolidator:
    def __init__(
        self,
        model,
        *,
        reject_nested_containers: bool = False,
        max_batch_size: int = 10,
        enforce_materiality_gate: bool = False,
        deterministic_independent_fast_path: bool = False,
        deterministic_nested_text_pruning: bool = False,
        deterministic_preconsolidated_fast_path: bool = False,
        deterministic_issue_hydration: bool = False,
        component_repair: bool = False,
        online_source_atomization: bool = False,
        max_validation_repairs: int = 1,
        max_tokens_override: int | None = None,
    ):
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        self.model = model
        self.reject_nested_containers = reject_nested_containers
        self.max_batch_size = max_batch_size
        self.enforce_materiality_gate = enforce_materiality_gate
        self.deterministic_independent_fast_path = (
            deterministic_independent_fast_path
        )
        self.deterministic_nested_text_pruning = (
            deterministic_nested_text_pruning
        )
        self.deterministic_preconsolidated_fast_path = (
            deterministic_preconsolidated_fast_path
        )
        self.deterministic_issue_hydration = deterministic_issue_hydration
        self.component_repair = component_repair
        self.online_source_atomization = online_source_atomization
        self._last_component_repair: dict[str, Any] = {}
        if max_validation_repairs <= 0:
            raise ValueError("max_validation_repairs must be positive")
        self.max_validation_repairs = max_validation_repairs
        if max_tokens_override is not None and max_tokens_override <= 0:
            raise ValueError("max_tokens_override must be positive")
        self.max_tokens_override = max_tokens_override

    async def consolidate(
        self,
        image: ImageAsset,
        result: Mapping[str, Any],
    ) -> dict[str, Any]:
        self._last_component_repair = {}
        all_sources = source_findings(result)
        if not all_sources:
            return {
                "verdict": str(result.get("verdict", "no_artifact")),
                "confidence": float(result.get("confidence", 0.5)),
                "edit_regions": [],
                "discarded_finding_ids": [],
                "discard_reasons": {},
                "rationale": "原始结果中没有需要整编的局部问题。",
                "consolidation_mode": "deterministic_no_findings",
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
                "consolidation_mode": "deterministic_policy_rejection",
            }
        if self.deterministic_preconsolidated_fast_path:
            deterministic = self._consolidate_independent_sources(
                sources,
                allow_semantic_overlap=True,
            )
            if deterministic is not None:
                return self._attach_policy_discards(
                    deterministic,
                    rejected,
                    all_sources,
                )
        if self.deterministic_independent_fast_path:
            deterministic = self._consolidate_independent_sources(
                sources,
            )
            if deterministic is not None:
                return self._attach_policy_discards(
                    deterministic,
                    rejected,
                    all_sources,
                )
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
        effective_sources = _sources_with_online_atomization(
            combined,
            sources,
        )
        if self.deterministic_nested_text_pruning:
            combined = _merge_duplicate_edit_regions(combined)
            combined = _refine_nested_regions_from_precise_sources(
                combined,
                effective_sources,
            )
            combined = _merge_nested_repaint_regions(combined)
            combined = _prune_redundant_nested_text_containers(
                combined,
                effective_sources,
            )
            combined = _split_spatially_disconnected_text_regions(
                combined,
                effective_sources,
            )
        errors = validate_consolidated_result(
            combined,
            effective_sources,
            reject_nested_containers=self.reject_nested_containers,
        )
        if not errors:
            final = combined
        else:
            final = None
            if self.component_repair:
                final = await self._repair_conflict_components(
                    image=image,
                    result=result,
                    sources=effective_sources,
                    combined=combined,
                    errors=errors,
                )
            if final is None:
                final = await self._repair_combined(
                    image=image,
                    result=result,
                    sources=effective_sources,
                    combined=combined,
                    errors=errors,
                )
                if self.component_repair:
                    final = dict(final)
                    final["consolidation_mode"] = (
                        "v61_global_repair_fallback"
                    )
                    final["v61_component_repair"] = copy.deepcopy(
                        self._last_component_repair
                    )
        return self._attach_policy_discards(
            final,
            rejected,
            all_sources,
        )

    def _consolidate_independent_sources(
        self,
        sources: Sequence[Mapping[str, Any]],
        *,
        allow_semantic_overlap: bool = False,
    ) -> dict[str, Any] | None:
        """Avoid a model call only when no semantic merge is possible."""

        if any(
            str(source.get("verification", "")).lower() != "supported"
            for source in sources
        ):
            return None
        for left_index, left in enumerate(sources):
            left_bbox = left.get("bbox_norm")
            if not isinstance(left_bbox, list) or len(left_bbox) != 4:
                return None
            left_metadata = left.get("metadata", {})
            if not isinstance(left_metadata, Mapping):
                left_metadata = {}
            left_semantic_key = str(
                left_metadata.get("semantic_group_id")
                or left.get("region_id")
                or ""
            ).strip()
            for right in sources[left_index + 1 :]:
                right_bbox = right.get("bbox_norm")
                if not isinstance(right_bbox, list) or len(right_bbox) != 4:
                    return None
                right_metadata = right.get("metadata", {})
                if not isinstance(right_metadata, Mapping):
                    right_metadata = {}
                right_semantic_key = str(
                    right_metadata.get("semantic_group_id")
                    or right.get("region_id")
                    or ""
                ).strip()
                if not allow_semantic_overlap:
                    if _intersection(left_bbox, right_bbox) > 0:
                        return None
                    if left_semantic_key and left_semantic_key == right_semantic_key:
                        return None
                    if _normalized_target(
                        _source_edit_target(left)
                    ) == _normalized_target(_source_edit_target(right)):
                        return None
        regions: list[dict[str, Any]] = []
        for index, source in enumerate(sources, 1):
            source_id = str(source["source_finding_id"])
            metadata = source.get("metadata", {})
            if not isinstance(metadata, Mapping):
                metadata = {}
            materiality = str(metadata.get("materiality", "major")).lower()
            if materiality not in CORE_MATERIALITIES:
                materiality = "major"
            confidence = float(source.get("confidence", 0.0))
            edit_target = _source_edit_target(source)
            if not _nonempty_chinese(edit_target):
                quoted = f"“{edit_target}”" if edit_target else ""
                edit_target = (
                    f"文字编辑区域{quoted}"
                    if str(source.get("artifact_type", "")).startswith(
                        "text_"
                    )
                    else f"图像编辑区域{quoted}"
                )
            regions.append(
                {
                    "edit_region_id": f"r{index:03d}",
                    "bbox_norm": list(source["bbox_norm"]),
                    "edit_target": edit_target,
                    "edit_region_rationale": (
                        str(metadata.get("edit_region_rationale", "")).strip()
                        or "该区域对应独立编辑目标，可通过一次局部重绘完成修复。"
                    ),
                    "source_finding_ids": [source_id],
                    "issues": [
                        {
                            "artifact_type": str(source.get("artifact_type", "")),
                            "observation": str(source.get("observation", "")).strip(),
                            "abnormality": str(source.get("abnormality", "")).strip(),
                            "confidence": confidence,
                            "materiality": materiality,
                            "source_finding_ids": [source_id],
                        }
                    ],
                    "confidence": confidence,
                    "verification": "supported",
                }
            )
        value = {
            "verdict": "artifact_present",
            "confidence": max(
                (float(source.get("confidence", 0.0)) for source in sources),
                default=0.5,
            ),
            "edit_regions": regions,
            "discarded_finding_ids": [],
            "discard_reasons": {},
            "rationale": "候选区域彼此独立，无需语义合并，按原始局部重绘目标确定性整编。",
            "consolidation_mode": (
                "deterministic_preconsolidated"
                if allow_semantic_overlap
                else "deterministic_independent"
            ),
        }
        if allow_semantic_overlap and self.deterministic_nested_text_pruning:
            value = _merge_duplicate_edit_regions(value)
            value = _refine_nested_regions_from_precise_sources(
                value,
                sources,
            )
            value = _merge_nested_repaint_regions(value)
            value = _prune_redundant_nested_text_containers(
                value,
                sources,
            )
            value = _split_spatially_disconnected_text_regions(
                value,
                sources,
            )
        errors = validate_consolidated_result(
            value,
            sources,
            reject_nested_containers=self.reject_nested_containers,
        )
        return None if errors else value

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
            "source_findings": [
                compact_consolidation_source(source) for source in sources
            ],
            "spatial_hints": spatial_hints(sources),
        }
        nested_container_rule = (
            "若一个大框包住任何已经独立定位的小框，必须按一次局部重绘的真实目标"
            "重新选择：保留独立小框并移除冗余大框，或在确需整体重绘时合并为一个框；"
            "禁止大框与被其包含的小框并存。"
            if self.reject_nested_containers
            else ""
        )
        issue_output_rule = (
            "同一实体或文字块上的多类问题合并为一个 edit region；模型只负责"
            "区域分组、bbox、编辑目标和来源 ID 闭环，issues 将由程序依据"
            "source_finding_ids 无损回填，不要返回 issues。"
            if self.deterministic_issue_hydration
            else (
                "同一实体或文字块上的多类问题合并为一个 edit region，并在 issues 中"
                "完整保留全部问题。"
            )
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
                + issue_output_rule
                + "相邻但需要分别重绘的实体必须保持独立。bbox 覆盖完整"
                "编辑目标和必要上下文，允许少量冗余，不追求最紧；禁止整图框，不按面积"
                "大小拒绝独立主体。每个 source_finding_id 必须恰好进入一个 region，或"
                "进入 discarded_finding_ids 并给出中文原因。空间分离的文字标签必须"
                "分别输出 region，不得合成覆盖整片场景的大框。只保留 major/critical、"
                "置信度严格大于0.6且 verification=supported 的问题。不得输出未知来源 ID。"
                + nested_container_rule
                + "\n\n"
                f"待整编数据：\n{serialize_json(packet)}"
            ),
            metadata={"source_finding_count": len(sources)},
        )
        if self.deterministic_issue_hydration:
            initial = hydrate_issues_from_sources(initial, sources)
        initial = _restore_missing_issue_sources(initial, sources)
        initial = _reconcile_missing_broad_container_sources(
            initial,
            sources,
        )
        if self.deterministic_nested_text_pruning:
            initial = _merge_duplicate_edit_regions(initial)
            initial = _refine_nested_regions_from_precise_sources(
                initial,
                sources,
            )
            initial = _merge_nested_repaint_regions(initial)
            initial = _prune_redundant_nested_text_containers(
                initial,
                sources,
            )
            initial = _split_spatially_disconnected_text_regions(
                initial,
                sources,
            )
        if self.deterministic_issue_hydration:
            initial = hydrate_edit_region_text_from_sources(
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
        if self.online_source_atomization:
            atomized = atomize_disjoint_repeated_sources(initial, sources)
            if atomized is not None:
                atomized_value, _, _ = atomized
                atomized_errors = validate_consolidated_result(
                    atomized_value,
                    sources,
                    reject_nested_containers=self.reject_nested_containers,
                )
                if not atomized_errors:
                    return atomized_value
        current = initial
        current_errors = errors
        for repair_round in range(1, self.max_validation_repairs + 1):
            current = await self._complete(
                stage="consensus.consolidate_repair",
                image=image,
                system_prompt=(
                    "你是最终整编结果修正员。必须修复所有校验错误，不新增来源问题，"
                    "所有自然语言字段使用简体中文。"
                ),
                user_prompt=(
                    f"校验错误：\n{serialize_json(current_errors)}\n\n"
                    f"来源数据：\n{serialize_json(packet)}\n\n"
                    f"待修正结果：\n{serialize_json(current)}\n\n"
                    "返回完整修正结果。尤其确保每个来源 ID 只处理一次、同一重绘区域只保留"
                    "一个 bbox，且不使用整图框。空间"
                    "分离的文字实体必须拆成独立 region；只保留 major/critical 且置信度"
                    "严格大于0.6的问题。"
                    + (
                        "issues 由程序按来源 ID 回填，不要返回 issues。"
                        if self.deterministic_issue_hydration
                        else "一个区域中的全部问题必须写入 issues。"
                    )
                    + nested_container_rule
                ),
                metadata={
                    "source_finding_count": len(sources),
                    "validation_errors": current_errors,
                    "validation_repair_round": repair_round,
                },
            )
            if self.deterministic_issue_hydration:
                current = hydrate_issues_from_sources(current, sources)
            current = _restore_missing_issue_sources(current, sources)
            current = _reconcile_missing_broad_container_sources(
                current,
                sources,
            )
            if self.deterministic_nested_text_pruning:
                current = _merge_duplicate_edit_regions(current)
                current = _refine_nested_regions_from_precise_sources(
                    current,
                    sources,
                )
                current = _merge_nested_repaint_regions(current)
                current = _prune_redundant_nested_text_containers(
                    current,
                    sources,
                )
                current = _split_spatially_disconnected_text_regions(
                    current,
                    sources,
                )
            if self.deterministic_issue_hydration:
                current = hydrate_edit_region_text_from_sources(
                    current,
                    sources,
                )
            current_errors = validate_consolidated_result(
                current,
                sources,
                reject_nested_containers=self.reject_nested_containers,
            )
            if not current_errors:
                return current
        raise ConsolidationValidationError(current_errors)

    @staticmethod
    def _combine_partials(
        partials: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        regions: list[dict[str, Any]] = []
        discarded: list[str] = []
        reasons: dict[str, str] = {}
        confidences: list[float] = []
        atomization: dict[str, Any] = {}
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
            raw_atomization = partial.get("online_source_atomization", {})
            if isinstance(raw_atomization, Mapping):
                atomization.update(copy.deepcopy(dict(raw_atomization)))
        output = {
            "verdict": "artifact_present" if regions else "no_artifact",
            "confidence": max(confidences, default=0.5),
            "edit_regions": regions,
            "discarded_finding_ids": discarded,
            "discard_reasons": reasons,
            "rationale": (
                "先按语义目标分批整编，再合并为全图局部重绘结果。"
            ),
        }
        if atomization:
            output["online_source_atomization"] = atomization
            output["consolidation_mode"] = "online_source_atomization"
        return output

    async def _repair_conflict_components(
        self,
        *,
        image: ImageAsset,
        result: Mapping[str, Any],
        sources: Sequence[Mapping[str, Any]],
        combined: Mapping[str, Any],
        errors: Sequence[str],
    ) -> dict[str, Any] | None:
        analysis = build_consolidation_conflict_components(
            errors,
            combined,
            sources,
        )
        self._last_component_repair = {
            "attempted": False,
            "component_count": len(analysis.components),
            "component_attempt_count": 0,
            "component_success_count": 0,
            "unmapped_errors": list(analysis.unmapped_errors),
            "fallback_reason": "",
        }
        if analysis.unmapped_errors or not analysis.components:
            self._last_component_repair["fallback_reason"] = (
                "validation_errors_not_fully_localizable"
            )
            return None

        source_by_id = {
            str(source.get("source_finding_id", "")): source
            for source in sources
            if str(source.get("source_finding_id", "")).strip()
        }
        raw_regions = combined.get("edit_regions", [])
        regions = (
            [region for region in raw_regions if isinstance(region, Mapping)]
            if isinstance(raw_regions, list)
            else []
        )
        region_by_id = {
            str(region.get("edit_region_id", "")): region
            for region in regions
            if str(region.get("edit_region_id", "")).strip()
        }
        all_conflict_region_ids = {
            region_id
            for component in analysis.components
            for region_id in component.region_ids
        }
        async def repair_component(
            component_index: int,
            component: ConsolidationConflictComponent,
        ) -> dict[str, Any]:
            component_sources = [
                source_by_id[source_id]
                for source_id in component.source_ids
                if source_id in source_by_id
            ]
            component_regions = [
                region_by_id[region_id]
                for region_id in component.region_ids
                if region_id in region_by_id
            ]
            immutable_regions = [
                {
                    "edit_region_id": str(
                        region.get("edit_region_id", "")
                    ),
                    "bbox_norm": region.get("bbox_norm"),
                    "edit_target": region.get("edit_target"),
                    "source_finding_ids": region.get(
                        "source_finding_ids", []
                    ),
                }
                for region in regions
                if str(region.get("edit_region_id", ""))
                not in set(component.region_ids)
            ]
            packet = {
                "source_result": {
                    "verdict": result.get("verdict"),
                    "confidence": result.get("confidence"),
                },
                "component_source_findings": [
                    compact_consolidation_source(source)
                    for source in component_sources
                ],
                "component_spatial_hints": spatial_hints(
                    component_sources
                ),
                "component_edit_regions": component_regions,
                "immutable_region_boundaries": immutable_regions,
                "validation_errors": list(component.errors),
            }
            repaired = await self._complete(
                stage="consensus.consolidate_component_repair",
                image=image,
                system_prompt=(
                    "你是 V6.1 在线整编的冲突分量修复员。只修复当前分量中的"
                    "重复、嵌套、来源闭环或编辑目标划分错误，不新增图像问题，"
                    "不得修改只读的分量外区域。所有自然语言字段使用简体中文。"
                ),
                user_prompt=(
                    f"当前冲突分量：\n{serialize_json(packet)}\n\n"
                    "返回当前分量的完整整编结果，而不是全图结果。当前分量的每个"
                    "source ID 必须恰好处理一次；只能引用 component_source_findings"
                    " 中的 ID。不得返回或改写 immutable_region_boundaries 中的区域。"
                    "同一局部重绘目标的多个问题合成一个框，空间独立目标分别保留。"
                ),
                metadata={
                    "component_index": component_index,
                    "component_count": len(analysis.components),
                    "component_region_ids": list(component.region_ids),
                    "component_source_ids": list(component.source_ids),
                    "component_validation_errors": list(component.errors),
                    "immutable_region_count": len(immutable_regions),
                },
            )
            if self.deterministic_issue_hydration:
                repaired = hydrate_issues_from_sources(
                    repaired,
                    component_sources,
                )
            repaired = _restore_missing_issue_sources(
                repaired,
                component_sources,
            )
            if self.deterministic_nested_text_pruning:
                repaired = _merge_duplicate_edit_regions(repaired)
                repaired = _refine_nested_regions_from_precise_sources(
                    repaired,
                    component_sources,
                )
                repaired = _merge_nested_repaint_regions(repaired)
                repaired = _prune_redundant_nested_text_containers(
                    repaired,
                    component_sources,
                )
                repaired = _split_spatially_disconnected_text_regions(
                    repaired,
                    component_sources,
                )
            if self.deterministic_issue_hydration:
                repaired = hydrate_edit_region_text_from_sources(
                    repaired,
                    component_sources,
                )
            component_errors = validate_consolidated_result(
                repaired,
                component_sources,
                reject_nested_containers=self.reject_nested_containers,
            )
            if component_errors:
                raise ConsolidationValidationError(component_errors)
            self._last_component_repair["component_success_count"] += 1
            return repaired

        self._last_component_repair["attempted"] = True
        self._last_component_repair["component_attempt_count"] = len(
            analysis.components
        )
        repair_tasks = [
            asyncio.create_task(repair_component(index, component))
            for index, component in enumerate(
                analysis.components,
                start=1,
            )
        ]
        try:
            repaired_components = await asyncio.gather(*repair_tasks)
        except Exception as exc:
            for task in repair_tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(
                *repair_tasks,
                return_exceptions=True,
            )
            self._last_component_repair["fallback_reason"] = (
                f"component_request_or_validation_failed:{type(exc).__name__}"
            )
            return None

        conflict_region_ids = all_conflict_region_ids
        conflict_source_ids = {
            source_id
            for component in analysis.components
            for source_id in component.source_ids
        }
        merged = copy.deepcopy(dict(combined))
        merged["edit_regions"] = [
            copy.deepcopy(dict(region))
            for region in regions
            if str(region.get("edit_region_id", ""))
            not in conflict_region_ids
        ]
        raw_discarded = combined.get("discarded_finding_ids", [])
        merged["discarded_finding_ids"] = [
            str(source_id)
            for source_id in raw_discarded
            if str(source_id) not in conflict_source_ids
        ] if isinstance(raw_discarded, list) else []
        raw_reasons = combined.get("discard_reasons", {})
        merged["discard_reasons"] = {
            str(source_id): str(reason)
            for source_id, reason in raw_reasons.items()
            if str(source_id) not in conflict_source_ids
        } if isinstance(raw_reasons, Mapping) else {}
        for repaired in repaired_components:
            repaired_regions = repaired.get("edit_regions", [])
            if isinstance(repaired_regions, list):
                merged["edit_regions"].extend(
                    copy.deepcopy(repaired_regions)
                )
            repaired_discarded = repaired.get(
                "discarded_finding_ids", []
            )
            if isinstance(repaired_discarded, list):
                merged["discarded_finding_ids"].extend(
                    str(source_id) for source_id in repaired_discarded
                )
            repaired_reasons = repaired.get("discard_reasons", {})
            if isinstance(repaired_reasons, Mapping):
                merged["discard_reasons"].update(
                    {
                        str(source_id): str(reason)
                        for source_id, reason in repaired_reasons.items()
                    }
                )
        merged["verdict"] = (
            "artifact_present" if merged["edit_regions"] else "no_artifact"
        )
        merged["rationale"] = (
            "保留无冲突区域，并行修复冲突连通分量后完成全图在线整编。"
        )
        if self.deterministic_nested_text_pruning:
            merged = _merge_duplicate_edit_regions(merged)
            merged = _refine_nested_regions_from_precise_sources(
                merged,
                sources,
            )
            merged = _merge_nested_repaint_regions(merged)
            merged = _prune_redundant_nested_text_containers(
                merged,
                sources,
            )
            merged = _split_spatially_disconnected_text_regions(
                merged,
                sources,
            )
        if self.deterministic_issue_hydration:
            merged = hydrate_edit_region_text_from_sources(
                merged,
                sources,
            )
        merged_errors = validate_consolidated_result(
            merged,
            sources,
            reject_nested_containers=self.reject_nested_containers,
        )
        if merged_errors:
            self._last_component_repair["fallback_reason"] = (
                "global_validation_failed_after_component_merge"
            )
            self._last_component_repair["post_merge_errors"] = (
                merged_errors
            )
            return None
        self._last_component_repair.update(
            {
                "fallback_reason": "",
                "repaired": True,
                "components": [
                    {
                        "region_ids": list(component.region_ids),
                        "source_ids": list(component.source_ids),
                        "errors": list(component.errors),
                    }
                    for component in analysis.components
                ],
            }
        )
        merged["consolidation_mode"] = "v61_component_repair"
        merged["v61_component_repair"] = copy.deepcopy(
            self._last_component_repair
        )
        return merged

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
            "source_findings": [
                compact_consolidation_source(source) for source in sources
            ],
            "spatial_hints": spatial_hints(sources),
        }
        current = dict(combined)
        atomization = combined.get("online_source_atomization")
        current_errors = list(errors)
        for repair_round in range(1, self.max_validation_repairs + 1):
            current = await self._complete(
                stage="consensus.consolidate_global_repair",
                image=image,
                system_prompt=(
                    "你是分层整编的全图终审员，只修复已有整编结果中的重复、"
                    "嵌套、来源闭环或编辑目标划分错误，不新增图像问题。所有"
                    "自然语言字段使用简体中文。"
                ),
                user_prompt=(
                    f"程序校验错误：\n{serialize_json(current_errors)}\n\n"
                    f"完整来源：\n{serialize_json(packet)}\n\n"
                    f"分批整编结果：\n{serialize_json(current)}\n\n"
                    "返回完整终审结果。任何大框都不得与其包含的小框并存；同一"
                    "编辑目标的多个问题合并一个框，空间独立目标分别保留。每个"
                    "source ID 必须恰好处理一次，discard_reasons 只能包含确实"
                    "discarded 的 ID。"
                ),
                metadata={
                    "source_finding_count": len(sources),
                    "validation_errors": current_errors,
                    "validation_repair_round": repair_round,
                    "batch_count": len(
                        partition_source_findings(
                            sources,
                            max_batch_size=self.max_batch_size,
                        )
                    ),
                },
            )
            if isinstance(atomization, Mapping):
                current["online_source_atomization"] = copy.deepcopy(
                    dict(atomization)
                )
            if self.deterministic_issue_hydration:
                current = hydrate_issues_from_sources(current, sources)
            current = _restore_missing_issue_sources(current, sources)
            current = _reconcile_missing_broad_container_sources(
                current,
                sources,
            )
            if self.deterministic_nested_text_pruning:
                current = _merge_duplicate_edit_regions(current)
                current = _refine_nested_regions_from_precise_sources(
                    current,
                    sources,
                )
                current = _merge_nested_repaint_regions(current)
                current = _prune_redundant_nested_text_containers(
                    current,
                    sources,
                )
                current = _split_spatially_disconnected_text_regions(
                    current,
                    sources,
                )
            if self.deterministic_issue_hydration:
                current = hydrate_edit_region_text_from_sources(
                    current,
                    sources,
                )
            current_errors = validate_consolidated_result(
                current,
                sources,
                reject_nested_containers=self.reject_nested_containers,
            )
            if not current_errors:
                return current
        raise ConsolidationValidationError(current_errors)

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
                response_schema=(
                    CONSOLIDATION_GROUPING_SCHEMA
                    if self.deterministic_issue_hydration
                    else CONSOLIDATED_RESULT_SCHEMA
                ),
                metadata={
                    "workflow": "adversarial_consensus",
                    **(
                        {"max_tokens_override": self.max_tokens_override}
                        if self.max_tokens_override is not None
                        else {}
                    ),
                    **dict(metadata),
                },
            )
        )
        parsed = canonicalize_edit_region_ids(dict(parse_response(response)))
        return drop_stale_discard_reasons(parsed)
