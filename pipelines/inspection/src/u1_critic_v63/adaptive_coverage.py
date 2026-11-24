from __future__ import annotations

import asyncio
import copy
import json
from typing import Any, Mapping, Sequence

from critic_workflows import (
    BBox,
    CriticContext,
    Finding,
    ImageAsset,
    ModelRequest,
    Region,
)
from critic_workflows.utils import parse_response

from .adaptive_image import expand_bbox
from .quality_policy import candidate_is_core


_BBOX_SCHEMA: Mapping[str, Any] = {
    "type": "array",
    "items": {"type": "integer", "minimum": 0, "maximum": 1000},
    "minItems": 4,
    "maxItems": 4,
}

_CONTENT_REGION_SCHEMA: Mapping[str, Any] = {
    "type": "object",
    "required": [
        "region_id",
        "semantic_group_id",
        "semantic_unit_kind",
        "bbox_norm",
        "content_kind",
        "description",
        "designed_as_text",
        "boundary_complete",
    ],
    "properties": {
        "region_id": {"type": "string"},
        "zone_id": {"type": "string"},
        "semantic_group_id": {"type": "string"},
        "semantic_unit_kind": {
            "type": "string",
            "enum": [
                "complete_title",
                "card",
                "list_row",
                "table_row",
                "independent_label",
                "continuous_panel",
                "entity",
                "layout_block",
                "other",
            ],
        },
        "bbox_norm": _BBOX_SCHEMA,
        "content_kind": {
            "type": "string",
            "enum": ["text", "entity", "layout"],
        },
        "description": {"type": "string"},
        "designed_as_text": {"type": "boolean"},
        "boundary_complete": {"type": "boolean"},
    },
}

_INVENTORY_SCHEMA: Mapping[str, Any] = {
    "type": "object",
    "required": ["regions", "inventory_rationale"],
    "properties": {
        "regions": {
            "type": "array",
            "items": _CONTENT_REGION_SCHEMA,
        },
        "confirmed_empty": {"type": "boolean"},
        "inventory_rationale": {"type": "string"},
    },
}

_COVERAGE_AUDIT_SCHEMA: Mapping[str, Any] = {
    "type": "object",
    "required": ["missing_regions", "coverage_complete", "rationale"],
    "properties": {
        "missing_regions": {
            "type": "array",
            "items": _CONTENT_REGION_SCHEMA,
        },
        "coverage_complete": {"type": "boolean"},
        "rationale": {"type": "string"},
    },
}

_ISSUE_SCHEMA: Mapping[str, Any] = {
    "type": "object",
    "required": [
        "artifact_type",
        "observation",
        "abnormality",
        "confidence",
        "evidence_kind",
    ],
    "properties": {
        "artifact_type": {
            "type": "string",
            "enum": [
                "rendering_artifact",
                "structural_artifact",
                "topology_artifact",
                "relational_physical_artifact",
                "text_visual_integrity",
                "text_content_fidelity",
            ],
        },
        "observation": {"type": "string"},
        "abnormality": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "evidence_kind": {
            "type": "string",
            "enum": [
                "visible_corruption",
                "visible_geometry",
                "direct_format",
                "same_image_contradiction",
                "crop_boundary_artifact",
                "external_completion",
                "aesthetic_preference",
                "minor_imperfection",
            ],
        },
    },
}

_INSPECTION_SCHEMA: Mapping[str, Any] = {
    "type": "object",
    "required": [
        "crop_boundary_complete",
        "revised_content_bbox_norm",
        "defect_present",
        "materiality",
        "evidence_basis",
        "confidence",
        "grounding_status",
        "edit_bbox_full_norm",
        "issues",
        "rationale",
    ],
    "properties": {
        "crop_boundary_complete": {"type": "boolean"},
        "revised_content_bbox_norm": _BBOX_SCHEMA,
        "defect_present": {"type": "boolean"},
        "materiality": {
            "type": "string",
            "enum": ["critical", "major", "minor", "none"],
        },
        "evidence_basis": {
            "type": "string",
            "enum": [
                "visible_only",
                "external_knowledge",
                "aesthetic_preference",
            ],
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "grounding_status": {
            "type": "string",
            "enum": ["valid", "invalid"],
        },
        "edit_bbox_full_norm": _BBOX_SCHEMA,
        "issues": {
            "type": "array",
            "items": _ISSUE_SCHEMA,
        },
        "rationale": {"type": "string"},
    },
}

_COMPLETE_TEXT_UNIT_KINDS = frozenset(
    {
        "complete_title",
        "card",
        "list_row",
        "table_row",
        "continuous_panel",
    }
)

# Regions thinner than this are not reliably observable after crop encoding
# and some vision servers reject the resulting extreme-aspect image outright.
# Reject them before inspection so the coverage ledger can recover the zone
# instead of restarting the whole image unit after an HTTP 400.
_MIN_OBSERVABLE_BBOX_SPAN = 4


def _area(box: BBox) -> float:
    return (box.x2 - box.x1) * (box.y2 - box.y1)


def _intersection(left: BBox, right: BBox) -> float:
    width = max(0.0, min(left.x2, right.x2) - max(left.x1, right.x1))
    height = max(0.0, min(left.y2, right.y2) - max(left.y1, right.y1))
    return width * height


def _same_content_region(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> bool:
    if left.get("content_kind") != right.get("content_kind"):
        return False
    left_group = str(left.get("semantic_group_id", "")).strip()
    right_group = str(right.get("semantic_group_id", "")).strip()
    if (
        left_group
        and left_group == right_group
        and left.get("content_kind") == "text"
        and left.get("semantic_unit_kind")
        == right.get("semantic_unit_kind")
        and left.get("semantic_unit_kind")
        in _COMPLETE_TEXT_UNIT_KINDS
        and (
            left.get("boundary_complete") is not True
            or right.get("boundary_complete") is not True
        )
    ):
        return True
    left_box = BBox.from_value(left["bbox_norm"])
    right_box = BBox.from_value(right["bbox_norm"])
    intersection = _intersection(left_box, right_box)
    if intersection <= 0:
        return False
    union = _area(left_box) + _area(right_box) - intersection
    iou = intersection / union if union else 0.0
    containment = intersection / min(_area(left_box), _area(right_box))
    area_ratio = max(_area(left_box), _area(right_box)) / min(
        _area(left_box),
        _area(right_box),
    )
    return iou >= 0.5 or (
        containment >= 0.8 and area_ratio <= 3.0
    )


def _merge_content_regions(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> dict[str, Any]:
    output = copy.deepcopy(dict(left))
    left_box = BBox.from_value(left["bbox_norm"])
    right_box = BBox.from_value(right["bbox_norm"])
    output["bbox_norm"] = [
        min(left_box.x1, right_box.x1),
        min(left_box.y1, right_box.y1),
        max(left_box.x2, right_box.x2),
        max(left_box.y2, right_box.y2),
    ]
    sources = list(output.get("inventory_sources", []))
    sources.extend(right.get("inventory_sources", []))
    output["inventory_sources"] = list(
        dict.fromkeys(str(item) for item in sources)
    )
    descriptions = [
        str(item).strip()
        for item in (
            output.get("description", ""),
            right.get("description", ""),
        )
        if str(item).strip()
    ]
    output["description"] = "；".join(dict.fromkeys(descriptions))
    output["boundary_complete"] = (
        output.get("boundary_complete") is True
        and right.get("boundary_complete") is True
    )
    return output


def deduplicate_content_regions(
    regions: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for raw in regions:
        try:
            BBox.from_value(raw["bbox_norm"])
        except (KeyError, TypeError, ValueError):
            continue
        for index, current in enumerate(output):
            if _same_content_region(current, raw):
                output[index] = _merge_content_regions(current, raw)
                break
        else:
            output.append(copy.deepcopy(dict(raw)))
    return output


def suppress_aggregate_content_regions(
    regions: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    complete_text_parents = [
        region
        for region in regions
        if (
            region.get("content_kind") == "text"
            and region.get("boundary_complete") is True
            and region.get("semantic_unit_kind")
            in _COMPLETE_TEXT_UNIT_KINDS
        )
    ]
    output: list[dict[str, Any]] = []
    for index, candidate in enumerate(regions):
        candidate_box = BBox.from_value(candidate["bbox_norm"])
        candidate_is_complete_text = any(
            candidate is parent
            for parent in complete_text_parents
        )
        if (
            candidate.get("content_kind") == "text"
            and not candidate_is_complete_text
            and candidate.get("semantic_unit_kind")
            != "independent_label"
        ):
            contained_by_complete_parent = any(
                (
                    _intersection(
                        BBox.from_value(parent["bbox_norm"]),
                        candidate_box,
                    )
                    / _area(candidate_box)
                    >= 0.8
                )
                for parent in complete_text_parents
            )
            if contained_by_complete_parent:
                continue
        contained: list[Mapping[str, Any]] = []
        for other_index, other in enumerate(regions):
            if other_index == index:
                continue
            other_box = BBox.from_value(other["bbox_norm"])
            intersection = _intersection(candidate_box, other_box)
            if intersection <= 0:
                continue
            containment = intersection / _area(other_box)
            area_ratio = _area(candidate_box) / _area(other_box)
            if containment >= 0.8 and area_ratio >= 3.0:
                contained.append(other)
        separate_children = any(
            not _same_content_region(left, right)
            for left_index, left in enumerate(contained)
            for right in contained[left_index + 1:]
        )
        broad_layout = candidate.get("content_kind") == "layout"
        same_kind_children = all(
            item.get("content_kind") == candidate.get("content_kind")
            for item in contained
        )
        if (
            len(contained) >= 2
            and separate_children
            and (broad_layout or same_kind_children)
            and not candidate_is_complete_text
        ):
            continue
        output.append(copy.deepcopy(dict(candidate)))
    return output


def _normalize_regions(
    values: Any,
    *,
    source: str,
) -> list[dict[str, Any]]:
    if not isinstance(values, list):
        return []
    region_id_counts: dict[str, int] = {}
    for raw in values:
        if isinstance(raw, Mapping):
            raw_region_id = str(raw.get("region_id", "")).strip()
            if raw_region_id:
                region_id_counts[raw_region_id] = (
                    region_id_counts.get(raw_region_id, 0) + 1
                )
    output: list[dict[str, Any]] = []
    for index, raw in enumerate(values):
        if not isinstance(raw, Mapping):
            continue
        try:
            bbox = BBox.from_value(raw.get("bbox_norm"))
        except (TypeError, ValueError):
            continue
        if (
            bbox.x2 - bbox.x1 < _MIN_OBSERVABLE_BBOX_SPAN
            or bbox.y2 - bbox.y1 < _MIN_OBSERVABLE_BBOX_SPAN
        ):
            continue
        kind = str(raw.get("content_kind", ""))
        if kind not in {"text", "entity", "layout"}:
            continue
        region_id = str(raw.get("region_id", "")).strip()
        semantic_group_id = str(
            raw.get("semantic_group_id", "")
        ).strip()
        resolved_region_id = region_id or f"{source}-{index:03d}"
        if region_id and region_id_counts.get(region_id, 0) > 1:
            resolved_region_id = f"{region_id}-{index:03d}"
            if semantic_group_id == region_id:
                semantic_group_id = resolved_region_id
        unit_kind = str(raw.get("semantic_unit_kind", "")).strip()
        allowed_unit_kinds = {
            "complete_title",
            "card",
            "list_row",
            "table_row",
            "independent_label",
            "continuous_panel",
            "entity",
            "layout_block",
            "other",
        }
        if unit_kind not in allowed_unit_kinds:
            unit_kind = "other"
        output.append(
            {
                "region_id": resolved_region_id,
                "zone_id": str(raw.get("zone_id", "")).strip(),
                "semantic_group_id": (
                    semantic_group_id
                    or resolved_region_id
                ),
                "semantic_unit_kind": unit_kind,
                "bbox_norm": bbox.as_list(),
                "content_kind": kind,
                "description": str(raw.get("description", "")).strip()
                or f"{kind} 区域 {index + 1}",
                "designed_as_text": kind == "text",
                "boundary_complete": (
                    raw.get("boundary_complete") is True
                ),
                "inventory_sources": [
                    f"{source}:{resolved_region_id}"
                ],
            }
        )
    return output


def _issue_is_core(issue: Mapping[str, Any]) -> bool:
    try:
        confidence = float(issue.get("confidence", 0.0))
    except (TypeError, ValueError):
        return False
    if confidence <= 0.6:
        return False
    artifact_type = str(issue.get("artifact_type", ""))
    evidence_kind = str(issue.get("evidence_kind", ""))
    common_allowed = {
        "visible_corruption",
        "visible_geometry",
        "direct_format",
        "same_image_contradiction",
    }
    if evidence_kind not in common_allowed:
        return False
    if artifact_type == "text_content_fidelity":
        return evidence_kind in {
            "visible_corruption",
            "direct_format",
            "same_image_contradiction",
        }
    if artifact_type == "text_visual_integrity":
        return evidence_kind == "visible_corruption"
    return artifact_type in {
        "rendering_artifact",
        "structural_artifact",
        "topology_artifact",
        "relational_physical_artifact",
    }


class AdaptiveContentCoverageAuditor:
    def __init__(
        self,
        model: Any,
        image_tool: Any,
        *,
        max_concurrency: int = 12,
    ):
        if max_concurrency <= 0:
            raise ValueError("max_concurrency must be positive")
        self.model = model
        self.image_tool = image_tool
        self.max_concurrency = max_concurrency
        self.overlap_coverage_inspection = False

    async def audit(
        self,
        image: ImageAsset,
        context: CriticContext,
    ) -> tuple[list[Finding], dict[str, Any]]:
        text_raw, entity_raw = await asyncio.gather(
            self._inventory(image, "text"),
            self._inventory(image, "entity"),
        )
        initial = [*text_raw, *entity_raw]
        initial = suppress_aggregate_content_regions(
            deduplicate_content_regions(initial)
        )
        if self.overlap_coverage_inspection:
            coverage_task = asyncio.create_task(
                self._coverage_audit(image, initial)
            )
            initial_inspection_task = asyncio.create_task(
                self._inspect_regions_during_coverage_audit(
                    image,
                    initial,
                )
            )
            try:
                missing = await coverage_task
            except BaseException:
                initial_inspection_task.cancel()
                await asyncio.gather(
                    initial_inspection_task,
                    return_exceptions=True,
                )
                raise
        else:
            missing = await self._coverage_audit(
                image,
                initial,
            )
        regions = suppress_aggregate_content_regions(
            deduplicate_content_regions([*initial, *missing])
        )
        if self.overlap_coverage_inspection:
            initial_keys = {
                self._inspection_region_key(region) for region in initial
            }
            additional_regions = [
                region
                for region in regions
                if self._inspection_region_key(region) not in initial_keys
            ]
            self._prepare_completed_inventory_regions(image, regions)
            additional_task = asyncio.create_task(
                self._inspect_additional_regions_after_coverage_audit(
                    image,
                    additional_regions,
                )
            )
            initial_inspected, additional_inspected = await asyncio.gather(
                initial_inspection_task,
                additional_task,
            )
            inspected_by_key = {
                self._inspection_region_key(region): value
                for region, value in zip(initial, initial_inspected)
            }
            inspected_by_key.update(
                {
                    self._inspection_region_key(region): value
                    for region, value in zip(
                        additional_regions,
                        additional_inspected,
                    )
                }
            )
            inspected = [
                inspected_by_key[self._inspection_region_key(region)]
                for region in regions
            ]
        else:
            inspected = await self._inspect_regions(image, regions)
        findings = [
            finding
            for finding, _, _ in inspected
            if finding is not None
        ]
        inspection_outcomes = [
            {
                "region_id": str(region["region_id"]),
                "zone_id": str(region.get("zone_id", "")),
                "content_kind": str(region["content_kind"]),
                "status": "failed" if failed else "checked",
                "boundary_retry_count": retry_count,
            }
            for region, (_, retry_count, failed) in zip(
                regions,
                inspected,
            )
        ]
        return findings, {
            "verdict": (
                "artifact_present" if findings else "no_artifact"
            ),
            "confidence": max(
                (finding.confidence for finding in findings),
                default=0.9,
            ),
            "inventory_region_count": len(regions),
            "inspected_region_count": len(regions),
            "suspicious_region_count": len(findings),
            "boundary_retry_count": sum(
                retry_count
                for _, retry_count, _ in inspected
            ),
            "boundary_failure_count": sum(
                1
                for _, _, failed in inspected
                if failed
            ),
            "content_regions": regions,
            "inspection_outcomes": inspection_outcomes,
        }

    @staticmethod
    def _inspection_region_key(
        region: Mapping[str, Any],
    ) -> tuple[Any, ...]:
        return (
            str(region.get("region_id", "")),
            str(region.get("content_kind", "")),
            *tuple(region.get("bbox_norm", [])),
        )

    async def _inspect_regions_during_coverage_audit(
        self,
        image: ImageAsset,
        regions: Sequence[Mapping[str, Any]],
    ) -> list[tuple[Finding | None, int, bool]]:
        return await self._inspect_regions(image, regions)

    def _prepare_completed_inventory_regions(
        self,
        image: ImageAsset,
        regions: Sequence[Mapping[str, Any]],
    ) -> None:
        del image, regions

    async def _inspect_additional_regions_after_coverage_audit(
        self,
        image: ImageAsset,
        regions: Sequence[Mapping[str, Any]],
    ) -> list[tuple[Finding | None, int, bool]]:
        return await self._inspect_regions(image, regions)

    async def _inspect_regions(
        self,
        image: ImageAsset,
        regions: Sequence[Mapping[str, Any]],
    ) -> list[tuple[Finding | None, int, bool]]:
        """Inspect content regions with an overridable batching strategy."""

        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def inspect(region: Mapping[str, Any]):
            async with semaphore:
                return await self._inspect_region(image, region)

        return list(
            await asyncio.gather(*(inspect(region) for region in regions))
        )

    async def _inventory(
        self,
        image: ImageAsset,
        kind: str,
    ) -> list[dict[str, Any]]:
        text_mode = kind == "text"
        stage = (
            "content.inventory.text"
            if text_mode
            else "content.inventory.entity"
        )
        system_prompt = (
            "你是内容覆盖清单员，只依据图片像素建立区域清单，不做缺陷判断。"
            "不得读取或推断文件名、目录标签、参考答案或其他输入元数据。"
            "不得使用世界知识、审美偏好或疑似错误作为选择区域的条件。"
            "所有自然语言字段使用简体中文。"
        )
        user_prompt = (
            (
                "枚举原图中所有被设计为文字内容的完整语义编辑单元，乱码、"
                "伪文字和当前完全不可读的文字也必须列出并设置 designed_as_text=true。"
                "以一次局部重绘应共同修复的目标划分：完整主标题为一个单元；"
                "同一信息卡片的标题与正文为一个单元；表格、榜单或时间线完整一行"
                "为一个单元，包含该行的序号、名称、年份和数值；同一说明面板内"
                "连续正文为一个单元；空间独立标签各自为完整单元。禁止按换行、"
                "单词、几个字符或固定面积拆碎。若同一语义单元仍被拆成多个候选，"
                "必须使用同一 semantic_group_id。boundary_complete 仅在 bbox "
                "覆盖完整语义单元时为 true。不要判断文字是否正确。"
            )
            if text_mode
            else (
                "枚举原图中所有主要独立实体和语义完整的版面区块，包括人物、"
                "动物、物体、建筑、图表、插图、信息卡片等。正常实体也必须列出。"
                "不要把整张图作为一个区域，也不要仅列疑似有问题的实体。"
            )
        )
        response = await self.model.complete(
            ModelRequest(
                stage=stage,
                system_prompt=system_prompt,
                user_prompt=(
                    user_prompt
                    + " bbox_norm 使用原图整数 [0,1000] 归一化 "
                    "[x1,y1,x2,y2] 坐标。"
                ),
                images=(image,),
                response_schema=_INVENTORY_SCHEMA,
                metadata={
                    "workflow": "adversarial_consensus",
                    "coverage_mode": "adaptive",
                    "inventory_kind": kind,
                },
            )
        )
        parsed = parse_response(response)
        return _normalize_regions(
            parsed.get("regions", []),
            source=stage,
        )

    async def _coverage_audit(
        self,
        image: ImageAsset,
        regions: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        inventory = [
            {
                "region_id": region["region_id"],
                "semantic_group_id": region["semantic_group_id"],
                "semantic_unit_kind": region["semantic_unit_kind"],
                "content_kind": region["content_kind"],
                "description": region["description"],
                "bbox_norm": region["bbox_norm"],
            }
            for region in regions
        ]
        response = await self.model.complete(
            ModelRequest(
                stage="content.coverage.audit",
                system_prompt=(
                    "你是内容覆盖终审员，只检查清单是否漏掉原图中意图可读的"
                    "独立文字块或主要独立实体。不得判断缺陷、审美或外部事实，"
                    "不得输出已在清单中的区域。所有自然语言字段使用简体中文。"
                ),
                user_prompt=(
                    "下面是已有内容清单。对照原图，只补充明显漏掉的独立区域；"
                    "若没有漏项，missing_regions 返回空数组。\n"
                    + json.dumps(inventory, ensure_ascii=False)
                ),
                images=(image,),
                response_schema=_COVERAGE_AUDIT_SCHEMA,
                metadata={
                    "workflow": "adversarial_consensus",
                    "coverage_mode": "adaptive",
                },
            )
        )
        parsed = parse_response(response)
        return _normalize_regions(
            parsed.get("missing_regions", []),
            source="content.coverage.audit",
        )

    async def _inspect_region(
        self,
        image: ImageAsset,
        content: Mapping[str, Any],
    ) -> tuple[Finding | None, int, bool]:
        initial_content_bbox = BBox.from_value(content["bbox_norm"])
        content_bbox = initial_content_bbox
        review_bbox = expand_bbox(content_bbox, 0.08)
        boundary_retry_count = 0
        inspection_roles = self._inspection_roles(content)
        inspections: list[tuple[Mapping[str, Any], ImageAsset]] = []
        for attempt in range(2):
            inspections = list(
                await asyncio.gather(
                    *(
                        self._request_inspection(
                            image,
                            content,
                            content_bbox,
                            review_bbox,
                            inspection_role=inspection_role,
                            boundary_retry_count=boundary_retry_count,
                        )
                        for inspection_role in inspection_roles
                    )
                )
            )
            incomplete = [
                parsed
                for parsed, _ in inspections
                if parsed.get("crop_boundary_complete") is not True
            ]
            if not incomplete:
                break
            if attempt == 1:
                return None, boundary_retry_count, True
            revisions: list[BBox] = []
            for parsed in incomplete:
                try:
                    revised = BBox.from_value(
                        parsed.get("revised_content_bbox_norm")
                    )
                except (TypeError, ValueError):
                    return None, boundary_retry_count, True
                if not (
                    revised.x1 <= initial_content_bbox.x1
                    and revised.y1 <= initial_content_bbox.y1
                    and revised.x2 >= initial_content_bbox.x2
                    and revised.y2 >= initial_content_bbox.y2
                ):
                    return None, boundary_retry_count, True
                revisions.append(revised)
            content_bbox = BBox(
                min(revised.x1 for revised in revisions),
                min(revised.y1 for revised in revisions),
                max(revised.x2 for revised in revisions),
                max(revised.y2 for revised in revisions),
            )
            review_bbox = expand_bbox(content_bbox, 0.08)
            boundary_retry_count += 1
        if not inspections:
            return None, boundary_retry_count, True
        accepted: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
        seen_issues: set[tuple[str, str, str]] = set()
        for parsed, _ in inspections:
            if (
                not candidate_is_core(parsed)
                or parsed.get("grounding_status") != "valid"
            ):
                continue
            raw_issues = parsed.get("issues", [])
            if not isinstance(raw_issues, list):
                continue
            for issue in raw_issues:
                if not isinstance(issue, Mapping) or not _issue_is_core(issue):
                    continue
                key = (
                    str(issue.get("artifact_type", "")),
                    str(issue.get("observation", "")).strip(),
                    str(issue.get("abnormality", "")).strip(),
                )
                if key in seen_issues:
                    continue
                seen_issues.add(key)
                accepted.append((parsed, copy.deepcopy(dict(issue))))
        issues = [issue for _, issue in accepted]
        if not issues:
            return None, boundary_retry_count, False
        has_text_issue = any(
            str(issue.get("artifact_type", "")).startswith("text_")
            for issue in issues
        )
        if content.get("content_kind") == "text" or has_text_issue:
            edit_bbox = content_bbox
        else:
            parsed = accepted[0][0]
            try:
                edit_bbox = BBox.from_value(
                    parsed.get("edit_bbox_full_norm")
                )
            except (TypeError, ValueError):
                return None, boundary_retry_count, False
            intersection = _intersection(edit_bbox, review_bbox)
            if intersection / _area(edit_bbox) < 0.8:
                return None, boundary_retry_count, False
        observation = "；".join(
            str(issue.get("observation", "")).strip()
            for issue in issues
            if str(issue.get("observation", "")).strip()
        )
        abnormality = "；".join(
            str(issue.get("abnormality", "")).strip()
            for issue in issues
            if str(issue.get("abnormality", "")).strip()
        )
        confidence = max(
            min(
                float(parsed.get("confidence", 0.0)),
                float(issue["confidence"]),
            )
            for parsed, issue in accepted
        )
        if confidence <= 0.6:
            return None, boundary_retry_count, False
        materiality = (
            "critical"
            if any(
                str(parsed.get("materiality", "")).lower() == "critical"
                for parsed, _ in accepted
            )
            else "major"
        )
        crop = inspections[0][1]
        return Finding(
            region_id=str(content["semantic_group_id"]),
            bbox=edit_bbox,
            artifact_type=str(issues[0]["artifact_type"]),
            observation=observation,
            abnormality=abnormality,
            confidence=confidence,
            evidence_ids=[
                image.id,
                *[
                    str(item)
                    for item in content.get("inventory_sources", [])
                ],
            ],
            source="role:text_coverage",
            metadata={
                "edit_target": str(content["description"]),
                "edit_region_rationale": (
                    "覆盖该完整语义编辑单元中需要一次局部重绘的全部核心问题，"
                    "保留少量上下文，不按行或词拆碎，也不包含空间分离的其他实体。"
                ),
                "original_bbox_norm": edit_bbox.as_list(),
                "grounding_status": "pending",
                "defect_present": True,
                "materiality": materiality,
                "evidence_basis": "visible_only",
                "text_entity_atomic": (
                    content.get("content_kind") == "text"
                ),
                "content_kind": content["content_kind"],
                "content_description": content["description"],
                "semantic_group_id": content["semantic_group_id"],
                "semantic_unit_kind": content["semantic_unit_kind"],
                "designed_as_text": content["designed_as_text"],
                "initial_content_bbox_norm": (
                    initial_content_bbox.as_list()
                ),
                "content_bbox_norm": content_bbox.as_list(),
                "review_bbox_norm": review_bbox.as_list(),
                "inventory_sources": list(
                    content.get("inventory_sources", [])
                ),
                "boundary_retry_count": boundary_retry_count,
                "original_crop_size": crop.metadata.get(
                    "original_crop_size"
                ),
                "review_crop_size": crop.metadata.get(
                    "review_crop_size"
                ),
                "zoom_scale": crop.metadata.get("zoom_scale", 1),
                "issues": issues,
                "inspection_roles": list(inspection_roles),
                "inspection_rationale": "；".join(
                    dict.fromkeys(
                        str(parsed.get("rationale", "")).strip()
                        for parsed, _ in accepted
                        if str(parsed.get("rationale", "")).strip()
                    )
                ),
            },
        ), boundary_retry_count, False

    def _inspection_roles(
        self,
        content: Mapping[str, Any],
    ) -> tuple[str, ...]:
        if content.get("content_kind") == "text":
            return ("text_visual", "text_fidelity")
        return ("artifact", "embedded_text")

    async def _request_inspection(
        self,
        image: ImageAsset,
        content: Mapping[str, Any],
        content_bbox: BBox,
        review_bbox: BBox,
        *,
        inspection_role: str,
        boundary_retry_count: int,
    ) -> tuple[Mapping[str, Any], ImageAsset]:
        region = Region(
            id=f"content-{content['region_id']}",
            bbox=review_bbox,
            label=str(content["content_kind"]),
            metadata={
                "content_region_id": content["region_id"],
                "content_kind": content["content_kind"],
                "content_bbox_norm": content_bbox.as_list(),
                "review_bbox_norm": review_bbox.as_list(),
            },
        )
        crop = await self.image_tool.crop(image, region)
        if inspection_role == "text_visual":
            inspection_rule = (
                "你是文字视觉完整性专员。逐字覆盖该完整语义单元，检查明显乱码、"
                "伪文字、笔画断裂、粘连、扭曲、缺损和无法辨认的字符。不得按行、"
                "单词或局部片段缩小检查范围。清晰可读的艺术字体正常。"
            )
        elif inspection_role == "text_visual_negative_verification":
            inspection_rule = (
                "你是独立的文字视觉负结论复核员。不要假设其他审查已经正确，"
                "重新逐字扫描该完整语义单元，检查明显乱码、伪文字、笔画断裂、"
                "粘连、扭曲、缺损和无法辨认字符。只报告直接可见的核心错误；"
                "清晰可读的艺术字体正常，不得依赖正确文案或外部知识猜错字。"
            )
        elif inspection_role == "text_fidelity":
            inspection_rule = (
                "你是文字内容保真专员。逐块检查只有同一文字单元内可直接验证的"
                "明显格式错误，或与本图其他明确文字逐字矛盾的问题。若需要知道"
                "片名、名称、年份、事实、排名、人物身份、正确文案或靠语言补全"
                "猜测缺字，必须判为无问题；不得因为片段不像熟悉短语就补出预期文字。"
            )
        elif inspection_role == "embedded_text":
            inspection_rule = (
                "你是实体与版面中的文字兜底专员。先检查该独立实体或版面区块内"
                "是否存在被设计为可读内容的文字；没有文字则返回无问题。若有文字，"
                "必须覆盖其中每一块，检查明显乱码、伪文字、笔画断裂粘连、扭曲，"
                "以及无需外部知识即可直接验证的格式错误。不得依赖正确文案、片名、"
                "年份、事实或语言补全。若发现问题，编辑框以当前完整实体或版面单元"
                "为边界，不拆成单行、单词或字符碎片。"
            )
        else:
            inspection_rule = (
                "你是非文字 Artifact 专员。检查明显渲染破损、实体结构错误、"
                "拓扑错误以及明显关系或物理错误。正常透视、遮挡、景深和轻微 "
                "AI 瑕疵正常。"
            )
        response = await self.model.complete(
            ModelRequest(
                stage="content.inspect",
                system_prompt=(
                    "你是面向局部重绘的核心缺陷审查员。只报告肉眼明显、"
                    "通过一次局部重绘能实质改善的 major/critical 问题。"
                    "忽略世界知识、审美偏好和轻微瑕疵。图像0是原图，"
                    "图像1是同一区域保留上下文后的真实像素裁剪，可能仅做"
                    "整数倍像素复制放大，未增加新细节。所有自然语言使用简体中文。"
                ),
                user_prompt=(
                    f"区域类型：{content['content_kind']}；"
                    f"语义单元：{content['semantic_unit_kind']}；"
                    f"区域说明：{content['description']}。\n"
                    "先比较图像0和图像1，确认图像1是否覆盖完整语义单元。"
                    "若文字、卡片、榜单行或实体仍延伸到裁剪边界之外，设置 "
                    "crop_boundary_complete=false、issues=[]、defect_present=false，"
                    "并在 revised_content_bbox_norm 中给出覆盖完整语义单元的"
                    "原图坐标，程序会在其外围重新增加上下文。"
                    "裁剪造成的断头词、半行、孤立数字或字符残片不是原图缺陷，"
                    "evidence_kind 不得用它支撑问题。边界完整后才能继续检查。\n"
                    + inspection_rule
                    + "\n若同一区域有多个问题，在 issues 中全部列出，但只给一个"
                    "可供 edit 模型一次局部重绘的 edit_bbox_full_norm，所有 bbox "
                    "统一使用图像0原图整数 [0,1000] 归一化坐标，禁止返回"
                    "图像1局部坐标。"
                    "文字语义单元若有问题，程序会使用完整语义单元 bbox，不按行或词"
                    "缩小；实体区域才使用 edit_bbox_full_norm 定位局部缺陷。"
                    "没有核心问题时 defect_present=false、issues=[]。"
                ),
                images=(image, crop),
                response_schema=_INSPECTION_SCHEMA,
                metadata={
                    "workflow": "adversarial_consensus",
                    "coverage_mode": "adaptive",
                    "content_region_id": content["region_id"],
                    "semantic_group_id": content["semantic_group_id"],
                    "semantic_unit_kind": content["semantic_unit_kind"],
                    "content_kind": content["content_kind"],
                    "inspection_role": inspection_role,
                    "content_bbox_norm": content_bbox.as_list(),
                    "review_bbox_norm": review_bbox.as_list(),
                    "original_crop_size": crop.metadata.get(
                        "original_crop_size"
                    ),
                    "review_crop_size": crop.metadata.get(
                        "review_crop_size"
                    ),
                    "zoom_scale": crop.metadata.get("zoom_scale", 1),
                    "boundary_retry_count": boundary_retry_count,
                },
            )
        )
        parsed = parse_response(response)
        return parsed, crop
