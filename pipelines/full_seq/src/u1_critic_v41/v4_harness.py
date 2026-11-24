from __future__ import annotations

import asyncio
import copy
import json
from collections.abc import Mapping, Sequence
from typing import Any

from critic_workflows import (
    BBox,
    CriticContext,
    CriticRole,
    Finding,
    ImageAsset,
    ModelRequest,
    Region,
    Verdict,
)
from critic_workflows.utils import parse_response

from .adaptive_coverage import (
    _COVERAGE_AUDIT_SCHEMA,
    _INVENTORY_SCHEMA,
    _INSPECTION_SCHEMA,
    _normalize_regions,
    AdaptiveContentCoverageAuditor,
)
from .adaptive_image import expand_bbox
from .edit_consensus import EditGroundedAdversarialConsensusCritic
from .quality_policy import candidate_is_core


_ZONE_SCHEMA: Mapping[str, Any] = {
    "type": "object",
    "required": [
        "zone_id",
        "bbox_norm",
        "description",
        "expected_content_kinds",
    ],
    "properties": {
        "zone_id": {"type": "string"},
        "bbox_norm": {
            "type": "array",
            "items": {
                "type": "integer",
                "minimum": 0,
                "maximum": 1000,
            },
            "minItems": 4,
            "maxItems": 4,
        },
        "description": {"type": "string"},
        "expected_content_kinds": {
            "type": "array",
            "items": {
                "type": "string",
                "enum": ["text", "entity", "layout"],
            },
        },
    },
}

_ZONE_INVENTORY_SCHEMA: Mapping[str, Any] = {
    "type": "object",
    "required": ["zones", "rationale"],
    "properties": {
        "zones": {"type": "array", "items": _ZONE_SCHEMA},
        "rationale": {"type": "string"},
    },
}

_ZONE_RECOVERY_SCHEMA: Mapping[str, Any] = {
    "type": "object",
    "required": ["regions", "confirmed_empty", "rationale"],
    "properties": {
        "regions": _INVENTORY_SCHEMA["properties"]["regions"],
        "confirmed_empty": {"type": "boolean"},
        "rationale": {"type": "string"},
    },
}

_V41_INSPECTION_SCHEMA = copy.deepcopy(_INSPECTION_SCHEMA)
_V41_INSPECTION_SCHEMA["required"] = [
    *list(_V41_INSPECTION_SCHEMA["required"]),
    "confirmed_in_original",
    "confirmed_in_context",
]
_V41_INSPECTION_SCHEMA["properties"] = {
    **dict(_V41_INSPECTION_SCHEMA["properties"]),
    "confirmed_in_original": {"type": "boolean"},
    "confirmed_in_context": {"type": "boolean"},
}
_V41_INVENTORY_SCHEMA = copy.deepcopy(_INVENTORY_SCHEMA)
_V41_INVENTORY_SCHEMA["required"] = [
    *list(_V41_INVENTORY_SCHEMA["required"]),
    "confirmed_empty",
]


def _bbox_area(value: Sequence[int]) -> int:
    return max(0, value[2] - value[0]) * max(0, value[3] - value[1])


def _bbox_intersection(
    left: Sequence[int],
    right: Sequence[int],
) -> int:
    return max(
        0,
        min(left[2], right[2]) - max(left[0], right[0]),
    ) * max(
        0,
        min(left[3], right[3]) - max(left[1], right[1]),
    )


def _coverage_channel(content_kind: str) -> str:
    return "text" if content_kind == "text" else "artifact"


def summarize_coverage_channels(
    ledger: Sequence[Mapping[str, Any]],
    inventory_health: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Keep text and non-text coverage independently auditable."""

    output: dict[str, dict[str, Any]] = {}
    for channel in ("text", "artifact"):
        entries = [
            copy.deepcopy(dict(item))
            for item in ledger
            if _coverage_channel(str(item.get("content_kind", "")))
            == channel
        ]
        relevant_health: list[str] = []
        for key, value in inventory_health.items():
            key_text = str(key)
            applies = key_text.startswith("zone_")
            if channel == "text":
                applies = applies or key_text == "text" or key_text.endswith(
                    ":text"
                )
            else:
                applies = (
                    applies
                    or key_text == "entity"
                    or key_text.endswith(":artifact")
                    or key_text.endswith(":entity")
                    or key_text.endswith(":layout")
                )
            if applies:
                relevant_health.append(str(value.get("status", "failed")))
        states = {str(item.get("state", "pending")) for item in entries}
        if "failed" in states or "failed" in relevant_health:
            status = "failed"
        elif "pending" in states or "degraded" in relevant_health:
            status = "degraded"
        else:
            status = "ok"
        output[channel] = {
            "status": status,
            "ledger": entries,
            "required_region_count": sum(
                len(item.get("region_ids", [])) for item in entries
            ),
            "confirmed_empty_count": sum(
                item.get("state") == "confirmed_empty" for item in entries
            ),
        }
    return output


def build_multiscale_review_bboxes(
    content_bbox: BBox,
    zone_bbox: BBox | None = None,
) -> tuple[BBox, BBox]:
    """Generate deterministic detail/context views around one edit target."""

    detail = expand_bbox(content_bbox, 0.08)
    context = expand_bbox(content_bbox, 0.35)
    if zone_bbox is not None:
        context = BBox(
            (
                min(context.x1, zone_bbox.x1)
                if abs(content_bbox.x1 - zone_bbox.x1) <= 24
                else context.x1
            ),
            (
                min(context.y1, zone_bbox.y1)
                if abs(content_bbox.y1 - zone_bbox.y1) <= 24
                else context.y1
            ),
            (
                max(context.x2, zone_bbox.x2)
                if abs(content_bbox.x2 - zone_bbox.x2) <= 24
                else context.x2
            ),
            (
                max(context.y2, zone_bbox.y2)
                if abs(content_bbox.y2 - zone_bbox.y2) <= 24
                else context.y2
            ),
        )
    return detail, context


def _normalize_zones(
    values: Any,
    *,
    source: str,
) -> list[dict[str, Any]]:
    if not isinstance(values, list):
        return []
    output: list[dict[str, Any]] = []
    for index, raw in enumerate(values):
        if not isinstance(raw, Mapping):
            continue
        try:
            bbox = BBox.from_value(raw.get("bbox_norm"))
        except (TypeError, ValueError):
            continue
        kinds = sorted(
            {
                str(kind)
                for kind in raw.get("expected_content_kinds", [])
                if str(kind) in {"text", "entity", "layout"}
            }
        )
        if not kinds:
            continue
        zone_id = str(raw.get("zone_id", "")).strip()
        output.append(
            {
                "zone_id": zone_id or f"{source}-zone-{index:03d}",
                "bbox_norm": bbox.as_list(),
                "description": (
                    str(raw.get("description", "")).strip()
                    or f"语义版面区 {index + 1}"
                ),
                "expected_content_kinds": kinds,
                "coverage_sources": [source],
            }
        )
    return output


def reconcile_semantic_zones(
    primary: Sequence[Mapping[str, Any]],
    blind: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Union independent zone observations without letting absence cancel content."""

    output: list[dict[str, Any]] = []
    for source, values in (("primary", primary), ("blind", blind)):
        for incoming in _normalize_zones(list(values), source=source):
            incoming_box = incoming["bbox_norm"]
            match: dict[str, Any] | None = None
            for existing in output:
                existing_box = existing["bbox_norm"]
                intersection = _bbox_intersection(
                    existing_box,
                    incoming_box,
                )
                if intersection <= 0:
                    continue
                smaller = min(
                    _bbox_area(existing_box),
                    _bbox_area(incoming_box),
                )
                union = (
                    _bbox_area(existing_box)
                    + _bbox_area(incoming_box)
                    - intersection
                )
                containment = intersection / smaller if smaller else 0.0
                iou = intersection / union if union else 0.0
                if containment >= 0.6 or iou >= 0.4:
                    match = existing
                    break
            if match is None:
                output.append(incoming)
                continue
            match["bbox_norm"] = [
                min(match["bbox_norm"][0], incoming_box[0]),
                min(match["bbox_norm"][1], incoming_box[1]),
                max(match["bbox_norm"][2], incoming_box[2]),
                max(match["bbox_norm"][3], incoming_box[3]),
            ]
            match["expected_content_kinds"] = sorted(
                {
                    *match["expected_content_kinds"],
                    *incoming["expected_content_kinds"],
                }
            )
            match["coverage_sources"] = list(
                dict.fromkeys(
                    [
                        *match["coverage_sources"],
                        *incoming["coverage_sources"],
                    ]
                )
            )
            descriptions = [
                match["description"],
                incoming["description"],
            ]
            match["description"] = "；".join(
                dict.fromkeys(
                    item for item in descriptions if item
                )
            )
    return output


def _region_matches_zone(
    region: Mapping[str, Any],
    zone: Mapping[str, Any],
) -> bool:
    if (
        str(region.get("zone_id", "")).strip()
        == str(zone.get("zone_id", "")).strip()
    ):
        return True
    try:
        region_box = BBox.from_value(region.get("bbox_norm")).as_list()
        zone_box = BBox.from_value(zone.get("bbox_norm")).as_list()
    except (TypeError, ValueError):
        return False
    intersection = _bbox_intersection(region_box, zone_box)
    area = _bbox_area(region_box)
    return area > 0 and intersection / area >= 0.5


def build_coverage_ledger(
    zones: Sequence[Mapping[str, Any]],
    regions: Sequence[Mapping[str, Any]],
    outcomes: Sequence[Mapping[str, Any]],
    *,
    confirmed_empty: set[tuple[str, str]] | None = None,
) -> list[dict[str, Any]]:
    confirmed_empty = confirmed_empty or set()
    outcome_by_region = {
        str(item.get("region_id", "")): item
        for item in outcomes
        if not item.get("superseded")
    }
    active_region_ids = set(outcome_by_region)
    ledger: list[dict[str, Any]] = []
    for zone in zones:
        zone_id = str(zone["zone_id"])
        for kind in zone.get("expected_content_kinds", []):
            matched = [
                region
                for region in regions
                if (
                    str(region.get("content_kind", "")) == str(kind)
                    and _region_matches_zone(region, zone)
                    and str(region.get("region_id", ""))
                    in active_region_ids
                )
            ]
            statuses = [
                str(
                    outcome_by_region.get(
                        str(region.get("region_id", "")),
                        {},
                    ).get("status", "pending")
                )
                for region in matched
            ]
            if (zone_id, str(kind)) in confirmed_empty and not matched:
                state = "confirmed_empty"
            elif any(status == "failed" for status in statuses):
                state = "failed"
            elif matched and all(status == "checked" for status in statuses):
                state = "checked"
            else:
                state = "pending"
            ledger.append(
                {
                    "zone_id": zone_id,
                    "zone_description": str(zone.get("description", "")),
                    "content_kind": str(kind),
                    "state": state,
                    "region_ids": [
                        str(region.get("region_id", ""))
                        for region in matched
                    ],
                }
            )
    return ledger


def build_channel_coverage_ledger(
    zones: Sequence[Mapping[str, Any]],
    regions: Sequence[Mapping[str, Any]],
    outcomes: Sequence[Mapping[str, Any]],
    *,
    confirmed_empty: set[tuple[str, str]] | None = None,
) -> list[dict[str, Any]]:
    """V4.1 ledger: one text and one artifact requirement per zone."""

    confirmed_empty = confirmed_empty or set()
    outcome_by_region = {
        str(item.get("region_id", "")): item
        for item in outcomes
        if not item.get("superseded")
    }
    ledger: list[dict[str, Any]] = []
    for zone in zones:
        expected_channels = {
            _coverage_channel(str(kind))
            for kind in zone.get("expected_content_kinds", [])
        }
        for channel in sorted(expected_channels):
            matched = [
                region
                for region in regions
                if (
                    _coverage_channel(
                        str(region.get("content_kind", ""))
                    )
                    == channel
                    and _region_matches_zone(region, zone)
                    and str(region.get("region_id", ""))
                    in outcome_by_region
                )
            ]
            statuses = [
                str(
                    outcome_by_region[str(region["region_id"])].get(
                        "status",
                        "pending",
                    )
                )
                for region in matched
            ]
            key = (str(zone["zone_id"]), channel)
            if key in confirmed_empty and not matched:
                state = "confirmed_empty"
            elif any(status == "failed" for status in statuses):
                state = "failed"
            elif matched and all(status == "checked" for status in statuses):
                state = "checked"
            else:
                state = "pending"
            ledger.append(
                {
                    "zone_id": str(zone["zone_id"]),
                    "zone_description": str(zone.get("description", "")),
                    "content_kind": channel,
                    "coverage_channel": channel,
                    "state": state,
                    "region_ids": [
                        str(region.get("region_id", ""))
                        for region in matched
                    ],
                }
            )
    return ledger


def apply_coverage_gate(
    summary: Mapping[str, Any],
    *,
    ledger: Sequence[Mapping[str, Any]],
    inventory_health: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    output = copy.deepcopy(dict(summary))
    health_states = {
        str(item.get("status", "failed"))
        for item in inventory_health.values()
    }
    ledger_states = {
        str(item.get("state", "pending"))
        for item in ledger
    }
    if (
        not ledger
        or "failed" in health_states
        or "failed" in ledger_states
    ):
        coverage_status = "failed"
    elif "degraded" in health_states or "pending" in ledger_states:
        coverage_status = "degraded"
    else:
        coverage_status = "ok"
    output["quality_verdict"] = str(
        output.get("quality_verdict", output.get("verdict", "uncertain"))
    )
    output["coverage_status"] = coverage_status
    output["harness_coverage_status"] = coverage_status
    output["coverage_ledger"] = copy.deepcopy(list(ledger))
    output["harness_inventory_health"] = copy.deepcopy(
        dict(inventory_health)
    )
    output["findings_non_exhaustive"] = coverage_status != "ok"
    if coverage_status != "ok":
        output["verdict"] = "uncertain"
        output["confidence"] = min(
            float(output.get("confidence", 0.5)),
            0.5,
        )
        output["harness_abstention_reason"] = (
            "内容覆盖账本未全部核销，以下已发现问题不是穷尽结果。"
        )
    return output


class HarnessedAdaptiveContentCoverageAuditor(
    AdaptiveContentCoverageAuditor
):
    """V4 semantic reliability harness around adaptive content coverage."""

    inventory_response_schema = _INVENTORY_SCHEMA
    coverage_mode_name = "harness"

    def __init__(
        self,
        model: Any,
        image_tool: Any,
        *,
        max_concurrency: int = 12,
        max_inventory_retries: int = 2,
        minimum_valid_ratio: float = 0.5,
    ):
        super().__init__(
            model,
            image_tool,
            max_concurrency=max_concurrency,
        )
        if max_inventory_retries < 0:
            raise ValueError("max_inventory_retries must be non-negative")
        if not 0 < minimum_valid_ratio <= 1:
            raise ValueError("minimum_valid_ratio must be in (0, 1]")
        self.max_inventory_retries = max_inventory_retries
        self.minimum_valid_ratio = minimum_valid_ratio
        self._inventory_health: dict[str, dict[str, Any]] = {}
        self._primary_zones: list[dict[str, Any]] = []
        self._blind_zones: list[dict[str, Any]] = []
        self._semantic_zones: list[dict[str, Any]] = []
        self.last_summary: dict[str, Any] = {}

    async def audit(
        self,
        image: ImageAsset,
        context: CriticContext,
    ) -> tuple[list[Finding], dict[str, Any]]:
        self._inventory_health = {}
        self._primary_zones, self._blind_zones = await asyncio.gather(
            self._zone_inventory(image, blind=False),
            self._zone_inventory(image, blind=True),
        )
        self._semantic_zones = reconcile_semantic_zones(
            self._primary_zones,
            self._blind_zones,
        )
        findings, summary = await super().audit(image, context)
        health = copy.deepcopy(self._inventory_health)
        regions = list(summary.get("content_regions", []))
        outcomes = list(summary.get("inspection_outcomes", []))
        ledger = self._build_coverage_ledger(
            self._semantic_zones,
            regions,
            outcomes,
        )
        recovery_targets = [
            item
            for item in ledger
            if item["state"] in {"pending", "failed"}
        ]
        confirmed_empty: set[tuple[str, str]] = set()
        if recovery_targets:
            recovery_results = await asyncio.gather(
                *(
                    self._recover_zone_requirement(
                        image,
                        item["zone_id"],
                        item["content_kind"],
                    )
                    for item in recovery_targets
                )
            )
            for target, recovered in zip(
                recovery_targets,
                recovery_results,
            ):
                recovered_regions, is_empty, recovery_health = recovered
                recovery_key = (
                    f"recovery:{target['zone_id']}:"
                    f"{target['content_kind']}"
                )
                health[recovery_key] = recovery_health
                if recovered_regions:
                    recovery_inspected = await asyncio.gather(
                        *(
                            self._inspect_region(image, region)
                            for region in recovered_regions
                        )
                    )
                    for region, (
                        finding,
                        retry_count,
                        failed,
                    ) in zip(recovered_regions, recovery_inspected):
                        regions.append(region)
                        outcomes.append(
                            {
                                "region_id": str(region["region_id"]),
                                "zone_id": str(region.get("zone_id", "")),
                                "content_kind": str(
                                    region["content_kind"]
                                ),
                                "status": (
                                    "failed" if failed else "checked"
                                ),
                                "boundary_retry_count": retry_count,
                            }
                        )
                        if finding is not None:
                            findings.append(finding)
                    if not any(
                        item["status"] == "failed"
                        for item in outcomes
                        if item.get("region_id")
                        in {
                            region["region_id"]
                            for region in recovered_regions
                        }
                    ):
                        recovered_ids = {
                            str(region["region_id"])
                            for region in recovered_regions
                        }
                        target_zone = next(
                            zone
                            for zone in self._semantic_zones
                            if zone["zone_id"] == target["zone_id"]
                        )
                        region_by_id = {
                            str(region["region_id"]): region
                            for region in regions
                        }
                        for item in outcomes:
                            region_id = str(item.get("region_id", ""))
                            original_region = region_by_id.get(region_id)
                            if (
                                not item.get("superseded")
                                and _coverage_channel(
                                    str(item.get("content_kind", ""))
                                )
                                == _coverage_channel(
                                    str(target["content_kind"])
                                )
                                and region_id not in recovered_ids
                                and original_region is not None
                                and _region_matches_zone(
                                    original_region,
                                    target_zone,
                                )
                            ):
                                item["superseded"] = True
                elif is_empty:
                    confirmed_empty.add(
                        (
                            str(target["zone_id"]),
                            str(target["content_kind"]),
                        )
                    )
                    target_zone = next(
                        zone
                        for zone in self._semantic_zones
                        if zone["zone_id"] == target["zone_id"]
                    )
                    region_by_id = {
                        str(region["region_id"]): region for region in regions
                    }
                    for item in outcomes:
                        original_region = region_by_id.get(
                            str(item.get("region_id", ""))
                        )
                        if (
                            not item.get("superseded")
                            and _coverage_channel(
                                str(item.get("content_kind", ""))
                            )
                            == _coverage_channel(
                                str(target["content_kind"])
                            )
                            and original_region is not None
                            and _region_matches_zone(
                                original_region,
                                target_zone,
                            )
                        ):
                            item["superseded"] = True
        ledger = self._build_coverage_ledger(
            self._semantic_zones,
            regions,
            outcomes,
            confirmed_empty=confirmed_empty,
        )
        summary["content_regions"] = regions
        summary["inspection_outcomes"] = outcomes
        summary["inventory_region_count"] = len(regions)
        summary["inspected_region_count"] = len(outcomes)
        summary["suspicious_region_count"] = len(findings)
        summary["boundary_failure_count"] = sum(
            1 for item in outcomes if item["status"] == "failed"
        )
        summary["semantic_zones"] = copy.deepcopy(self._semantic_zones)
        summary["primary_semantic_zones"] = copy.deepcopy(
            self._primary_zones
        )
        summary["blind_semantic_zones"] = copy.deepcopy(
            self._blind_zones
        )
        summary["harness_inventory_retries"] = sum(
            int(item.get("attempts", 1)) - 1
            for item in health.values()
        )
        summary = apply_coverage_gate(
            summary,
            ledger=ledger,
            inventory_health=health,
        )
        self.last_summary = copy.deepcopy(summary)
        return findings, summary

    def _build_coverage_ledger(
        self,
        zones: Sequence[Mapping[str, Any]],
        regions: Sequence[Mapping[str, Any]],
        outcomes: Sequence[Mapping[str, Any]],
        *,
        confirmed_empty: set[tuple[str, str]] | None = None,
    ) -> list[dict[str, Any]]:
        return build_coverage_ledger(
            zones,
            regions,
            outcomes,
            confirmed_empty=confirmed_empty,
        )

    async def _zone_inventory(
        self,
        image: ImageAsset,
        *,
        blind: bool,
    ) -> list[dict[str, Any]]:
        source = "blind" if blind else "primary"
        base_stage = (
            "content.coverage.blind_inventory"
            if blind
            else "content.zones.primary"
        )
        best: list[dict[str, Any]] = []
        last_errors: list[str] = []
        attempts = 0
        for attempt in range(self.max_inventory_retries + 1):
            attempts += 1
            stage = (
                base_stage
                if attempt == 0
                else f"{base_stage}.harness_retry"
            )
            feedback = (
                ""
                if not last_errors
                else "\n前一轮结构校验失败："
                + "；".join(last_errors)
                + "。请重新观察整张原图并提交完整语义版面区。"
            )
            response = await self.model.complete(
                ModelRequest(
                    stage=stage,
                    system_prompt=(
                        "你是独立的语义版面覆盖清单员，只依据当前原图划分"
                        "自适应语义版面区，不判断图片质量。不得读取文件名、"
                        "目录标签、参考答案、其他清单或已有缺陷结论。"
                        "禁止固定网格。bbox 使用原图整数 [0,1000] 坐标。"
                        "所有自然语言字段使用简体中文。"
                    ),
                    user_prompt=(
                        "独立枚举覆盖所有可见主要内容的语义版面区，并为每个区"
                        "标注肉眼可见或应当进一步清点的 content kind：text、"
                        "entity、layout。标题、标签、表格、图例、角落信息和"
                        "密集文字区域不得遗漏。背景空白不必单独建区。"
                        + feedback
                    ),
                    images=(image,),
                    response_schema=_ZONE_INVENTORY_SCHEMA,
                    metadata={
                        "workflow": "adversarial_consensus",
                        "coverage_mode": self.coverage_mode_name,
                        "zone_inventory": source,
                        "harness_semantic_attempt": attempt + 1,
                    },
                )
            )
            parsed = parse_response(response)
            raw_values = parsed.get("zones", [])
            raw_count = (
                len(raw_values) if isinstance(raw_values, list) else 0
            )
            normalized = _normalize_zones(
                raw_values,
                source=source,
            )
            valid_ratio = (
                len(normalized) / raw_count if raw_count else 0.0
            )
            if len(normalized) > len(best):
                best = normalized
            last_errors = self._inventory_validation_errors(
                raw_count=raw_count,
                valid_count=len(normalized),
                valid_ratio=valid_ratio,
            )
            if not last_errors:
                self._inventory_health[f"zone_{source}"] = {
                    "status": "ok",
                    "attempts": attempts,
                    "raw_region_count": raw_count,
                    "valid_region_count": len(normalized),
                    "valid_ratio": round(valid_ratio, 4),
                    "validation_errors": [],
                }
                return normalized
        self._inventory_health[f"zone_{source}"] = {
            "status": "degraded" if best else "failed",
            "attempts": attempts,
            "best_valid_region_count": len(best),
            "validation_errors": last_errors,
        }
        return best

    async def _recover_zone_requirement(
        self,
        image: ImageAsset,
        zone_id: str,
        content_kind: str,
    ) -> tuple[list[dict[str, Any]], bool, dict[str, Any]]:
        zone = next(
            (
                item
                for item in self._semantic_zones
                if item["zone_id"] == zone_id
            ),
            None,
        )
        if zone is None:
            return [], False, {
                "status": "failed",
                "attempts": 0,
                "validation_errors": ["目标 zone 不存在"],
            }
        crop = await self.image_tool.crop(
            image,
            Region(
                id=f"coverage-recovery:{zone_id}:{content_kind}",
                bbox=BBox.from_value(zone["bbox_norm"]),
                label=f"coverage_{content_kind}",
            ),
        )
        response = await self.model.complete(
            ModelRequest(
                stage="content.inventory.zone_recovery",
                system_prompt=(
                    "你是覆盖失败恢复清单员。图像0是原图，图像1是目标语义"
                    "版面区的无损裁剪或放大。只重建指定内容通道的完整清单，"
                    "不判断质量。所有 bbox 必须返回图像0原图整数 [0,1000] "
                    "坐标，不得使用图像1局部坐标。"
                ),
                user_prompt=(
                    f"目标 zone：{json.dumps(zone, ensure_ascii=False)}\n"
                    f"目标 content kind：{content_kind}。"
                    "枚举该 zone 内所有对应的完整语义单元；如果复核后确认"
                    "确实不存在该类内容，regions=[] 且 confirmed_empty=true。"
                ),
                images=(image, crop),
                response_schema=_ZONE_RECOVERY_SCHEMA,
                metadata={
                    "workflow": "adversarial_consensus",
                    "coverage_mode": self.coverage_mode_name,
                    "zone_id": zone_id,
                    "content_kind": content_kind,
                },
            )
        )
        parsed = parse_response(response)
        regions = [
            {
                **region,
                "zone_id": zone_id,
            }
            for region in _normalize_regions(
                parsed.get("regions", []),
                source=(
                    f"content.inventory.zone_recovery:"
                    f"{zone_id}:{content_kind}"
                ),
            )
            if (
                _coverage_channel(str(region.get("content_kind", "")))
                == _coverage_channel(content_kind)
                and _region_matches_zone(region, zone)
            )
        ]
        confirmed_empty = (
            parsed.get("confirmed_empty") is True and not regions
        )
        status = "ok" if regions or confirmed_empty else "failed"
        return regions, confirmed_empty, {
            "status": status,
            "attempts": 1,
            "raw_region_count": len(parsed.get("regions", []))
            if isinstance(parsed.get("regions"), list)
            else 0,
            "valid_region_count": len(regions),
            "confirmed_empty": confirmed_empty,
            "validation_errors": (
                []
                if status == "ok"
                else ["zone 恢复既未返回有效区域，也未确认空区"]
            ),
        }

    def _inspection_roles(
        self,
        content: Mapping[str, Any],
    ) -> tuple[str, ...]:
        if content.get("content_kind") == "text":
            return (
                "text_visual",
                "text_visual_negative_verification",
                "text_fidelity",
            )
        return super()._inspection_roles(content)

    async def _inventory(
        self,
        image: ImageAsset,
        kind: str,
    ) -> list[dict[str, Any]]:
        text_mode = kind == "text"
        base_stage = (
            "content.inventory.text"
            if text_mode
            else "content.inventory.entity"
        )
        best_regions: list[dict[str, Any]] = []
        best_score = (-1.0, -1)
        last_errors: list[str] = []
        attempts = 0
        last_raw_count = 0
        last_valid_count = 0
        for attempt in range(self.max_inventory_retries + 1):
            attempts += 1
            stage = (
                base_stage
                if attempt == 0
                else f"{base_stage}.harness_retry"
            )
            parsed = await self._request_inventory_attempt(
                image,
                kind,
                stage=stage,
                attempt=attempt + 1,
                validation_errors=last_errors,
            )
            raw_values = parsed.get("regions", [])
            raw_regions = (
                [item for item in raw_values if isinstance(item, Mapping)]
                if isinstance(raw_values, list)
                else []
            )
            normalized = _normalize_regions(
                raw_values,
                source=base_stage,
            )
            raw_count = len(raw_regions)
            valid_count = len(normalized)
            confirmed_empty = (
                parsed.get("confirmed_empty") is True and raw_count == 0
            )
            if confirmed_empty:
                self._inventory_health[kind] = {
                    "status": "ok",
                    "attempts": attempts,
                    "raw_region_count": 0,
                    "valid_region_count": 0,
                    "valid_ratio": 1.0,
                    "confirmed_empty": True,
                    "validation_errors": [],
                }
                return []
            valid_ratio = valid_count / raw_count if raw_count else 0.0
            score = (valid_ratio, valid_count)
            if score > best_score:
                best_score = score
                best_regions = normalized
            last_raw_count = raw_count
            last_valid_count = valid_count
            last_errors = self._inventory_validation_errors(
                raw_count=raw_count,
                valid_count=valid_count,
                valid_ratio=valid_ratio,
            )
            if not last_errors:
                self._inventory_health[kind] = {
                    "status": "ok",
                    "attempts": attempts,
                    "raw_region_count": raw_count,
                    "valid_region_count": valid_count,
                    "valid_ratio": round(valid_ratio, 4),
                    "validation_errors": [],
                }
                return normalized
        status = "degraded" if best_regions else "failed"
        self._inventory_health[kind] = {
            "status": status,
            "attempts": attempts,
            "raw_region_count": last_raw_count,
            "valid_region_count": last_valid_count,
            "best_valid_region_count": len(best_regions),
            "best_valid_ratio": round(max(best_score[0], 0.0), 4),
            "validation_errors": last_errors,
        }
        return best_regions

    def _inventory_validation_errors(
        self,
        *,
        raw_count: int,
        valid_count: int,
        valid_ratio: float,
    ) -> list[str]:
        errors: list[str] = []
        if raw_count == 0:
            errors.append("regions 为空，未枚举任何内容区域")
        if raw_count > 0 and valid_count == 0:
            errors.append("所有区域 bbox 均退化、越界或无法解析")
        if raw_count > 0 and valid_ratio < self.minimum_valid_ratio:
            errors.append(
                f"有效 bbox 比例 {valid_count}/{raw_count} 低于 "
                f"{self.minimum_valid_ratio:.0%}"
            )
        return errors

    async def _request_inventory_attempt(
        self,
        image: ImageAsset,
        kind: str,
        *,
        stage: str,
        attempt: int,
        validation_errors: list[str],
    ) -> Mapping[str, Any]:
        text_mode = kind == "text"
        system_prompt = (
            "你是内容覆盖清单员，只依据图片像素建立区域清单，不做缺陷判断。"
            "不得读取或推断文件名、目录标签、参考答案或其他输入元数据。"
            "所有 bbox 必须使用原图整数归一化坐标 [x1,y1,x2,y2]，严格满足 "
            "0<=x1<x2<=1000 且 0<=y1<y2<=1000。所有自然语言字段使用简体中文。"
        )
        task = (
            "枚举所有被设计为文字内容的完整语义编辑单元。乱码、伪文字和当前"
            "不可读文字也必须列出。以一次局部重绘共同修复的完整标题、卡片、"
            "榜单行、连续面板或空间独立标签为边界，不按行、词或固定面积拆碎。"
            if text_mode
            else
            "枚举所有主要独立实体和语义完整的版面区块，包括人物、物体、建筑、"
            "图表、插图和信息卡片。正常实体也必须列出，不得用整图占位。"
        )
        task += (
            " 若完整检查原图后确认不存在该类内容，必须返回 regions=[] 且 "
            "confirmed_empty=true；只要存在任何该类内容就返回 "
            "confirmed_empty=false。"
        )
        if self._semantic_zones:
            task += (
                "\n以下是由两次独立版面观察对账得到的语义 zone。每个区域"
                "必须填写所属 zone_id；逐 zone 检查，不能只覆盖其中一部分：\n"
                + json.dumps(self._semantic_zones, ensure_ascii=False)
            )
        feedback = ""
        if validation_errors:
            feedback = (
                "\n前一轮结果未通过程序校验："
                + "；".join(validation_errors)
                + "。请从原图重新枚举全部区域，不要只修一个字段；特别检查 bbox "
                "宽高均大于 0，禁止 [1,0,1,1] 等退化坐标。"
            )
        response = await self.model.complete(
            ModelRequest(
                stage=stage,
                system_prompt=system_prompt,
                user_prompt=task + feedback,
                images=(image,),
                response_schema=self.inventory_response_schema,
                metadata={
                    "workflow": "adversarial_consensus",
                    "coverage_mode": self.coverage_mode_name,
                    "inventory_kind": kind,
                    "harness_semantic_attempt": attempt,
                    "harness_validation_errors": validation_errors,
                },
            )
        )
        return parse_response(response)


class V41AdaptiveContentCoverageAuditor(
    HarnessedAdaptiveContentCoverageAuditor
):
    """V4.1 coverage with independent channels and code-owned review views."""

    inventory_response_schema = _V41_INVENTORY_SCHEMA
    coverage_mode_name = "harness_v41"

    def _build_coverage_ledger(
        self,
        zones: Sequence[Mapping[str, Any]],
        regions: Sequence[Mapping[str, Any]],
        outcomes: Sequence[Mapping[str, Any]],
        *,
        confirmed_empty: set[tuple[str, str]] | None = None,
    ) -> list[dict[str, Any]]:
        return build_channel_coverage_ledger(
            zones,
            regions,
            outcomes,
            confirmed_empty=confirmed_empty,
        )

    async def _coverage_audit(
        self,
        image: ImageAsset,
        regions: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        text_regions = [
            region
            for region in regions
            if str(region.get("content_kind", "")) == "text"
        ]
        artifact_regions = [
            region
            for region in regions
            if str(region.get("content_kind", "")) != "text"
        ]
        text_missing, artifact_missing = await asyncio.gather(
            self._coverage_audit_channel(
                image,
                text_regions,
                channel="text",
            ),
            self._coverage_audit_channel(
                image,
                artifact_regions,
                channel="artifact",
            ),
        )
        return [*text_missing, *artifact_missing]

    async def _coverage_audit_channel(
        self,
        image: ImageAsset,
        regions: Sequence[Mapping[str, Any]],
        *,
        channel: str,
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
        if channel == "text":
            target = (
                "只检查是否漏掉任何被设计为可读内容的独立文字语义单元，"
                "包括乱码、伪文字、角落标签和密集小字。"
            )
        else:
            target = (
                "只检查是否漏掉主要独立实体或语义完整的非文字版面区块。"
                "不要在本通道补充纯文字区域。"
            )
        response = await self.model.complete(
            ModelRequest(
                stage=f"content.coverage.audit.{channel}",
                system_prompt=(
                    "你是独立覆盖终审员，只核对内容清单漏项，不判断图像"
                    "缺陷、审美或外部事实。所有自然语言使用简体中文。"
                ),
                user_prompt=(
                    target
                    + "已有清单如下；只返回明显漏掉的区域，没有漏项时返回"
                    "空数组。\n"
                    + json.dumps(inventory, ensure_ascii=False)
                ),
                images=(image,),
                response_schema=_COVERAGE_AUDIT_SCHEMA,
                metadata={
                    "workflow": "adversarial_consensus",
                    "coverage_mode": "harness_v41",
                    "coverage_channel": channel,
                },
            )
        )
        parsed = parse_response(response)
        normalized = _normalize_regions(
            parsed.get("missing_regions", []),
            source=f"content.coverage.audit.{channel}",
        )
        return [
            region
            for region in normalized
            if (
                (channel == "text" and region["content_kind"] == "text")
                or (
                    channel == "artifact"
                    and region["content_kind"] != "text"
                )
            )
        ]

    async def audit(
        self,
        image: ImageAsset,
        context: CriticContext,
    ) -> tuple[list[Finding], dict[str, Any]]:
        findings, summary = await super().audit(image, context)
        channels = summarize_coverage_channels(
            summary.get("coverage_ledger", []),
            summary.get("harness_inventory_health", {}),
        )
        summary["coverage_channels"] = channels
        summary["text_coverage"] = copy.deepcopy(channels["text"])
        summary["artifact_coverage"] = copy.deepcopy(
            channels["artifact"]
        )
        channel_states = {
            value["status"] for value in channels.values()
        }
        if "failed" in channel_states:
            coverage_status = "failed"
        elif "degraded" in channel_states:
            coverage_status = "degraded"
        else:
            coverage_status = "ok"
        summary["coverage_status"] = coverage_status
        summary["harness_coverage_status"] = coverage_status
        summary["findings_non_exhaustive"] = coverage_status != "ok"
        if coverage_status != "ok":
            summary["quality_verdict"] = str(
                summary.get("quality_verdict", summary.get("verdict"))
            )
            summary["verdict"] = "uncertain"
            summary["confidence"] = min(
                float(summary.get("confidence", 0.5)),
                0.5,
            )
            summary["harness_abstention_reason"] = (
                "文字或非文字覆盖通道未完成独立核销。"
            )
        self.last_summary = copy.deepcopy(summary)
        return findings, summary

    async def _request_inventory_attempt(
        self,
        image: ImageAsset,
        kind: str,
        *,
        stage: str,
        attempt: int,
        validation_errors: list[str],
    ) -> Mapping[str, Any]:
        parsed = await super()._request_inventory_attempt(
            image,
            kind,
            stage=stage,
            attempt=attempt,
            validation_errors=validation_errors,
        )
        return parsed

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
        del review_bbox, boundary_retry_count
        zone = next(
            (
                item
                for item in self._semantic_zones
                if str(item.get("zone_id", ""))
                == str(content.get("zone_id", ""))
            ),
            None,
        )
        zone_bbox = None
        if zone is not None:
            try:
                zone_bbox = BBox.from_value(zone.get("bbox_norm"))
            except (TypeError, ValueError):
                zone_bbox = None
        detail_bbox, context_bbox = build_multiscale_review_bboxes(
            content_bbox,
            zone_bbox,
        )
        detail_region = Region(
            id=f"detail-{content['region_id']}",
            bbox=detail_bbox,
            label=str(content["content_kind"]),
        )
        context_region = Region(
            id=f"context-{content['region_id']}",
            bbox=context_bbox,
            label=str(content["content_kind"]),
        )
        detail_crop, context_crop = await asyncio.gather(
            self.image_tool.crop(image, detail_region),
            self.image_tool.crop(image, context_region),
        )
        if inspection_role == "text_visual":
            rule = (
                "逐字检查明显乱码、伪文字、笔画断裂、粘连、扭曲、缺损和"
                "不可辨认字符；清晰可读的艺术字体正常。"
            )
        elif inspection_role == "text_visual_negative_verification":
            rule = (
                "独立复核上一轮可能遗漏的明显文字视觉错误，只报告直接可见的"
                "核心错误，不依据正确文案或外部知识补全。"
            )
        elif inspection_role == "text_fidelity":
            rule = (
                "只检查同一文字单元内直接可验证的明显格式错误或本图内部逐字"
                "矛盾；片名、年份、事实和语言补全均不检查。"
            )
        elif inspection_role == "embedded_text":
            rule = (
                "检查实体内所有被设计为可读内容的文字，只报告明显乱码、伪文字、"
                "断裂粘连或扭曲；没有文字时返回无问题。"
            )
        else:
            rule = (
                "检查明显渲染、结构、拓扑、关系或物理错误；正常透视、遮挡、"
                "景深和轻微 AI 瑕疵不报告。"
            )
        response = await self.model.complete(
            ModelRequest(
                stage="content.inspect.multiscale",
                system_prompt=(
                    "你是面向局部重绘的核心缺陷审查员。图像0是原图，图像1是"
                    "程序生成的上下文视图，图像2是程序生成并可能无损放大的细节"
                    "视图。只报告肉眼明显且局部重绘能实质改善的 major/critical "
                    "问题，忽略世界知识、审美和轻微瑕疵。所有自然语言使用简体中文。"
                ),
                user_prompt=(
                    f"区域类型：{content['content_kind']}；语义单元："
                    f"{content['semantic_unit_kind']}；区域说明："
                    f"{content['description']}。\n{rule}\n"
                    "裁剪只用于放大观察，不能把裁剪产生的断头词、半行、孤立数字"
                    "或边缘残片当作原图缺陷。有问题时必须分别确认其在图像0和图像1"
                    "中都成立，并设置 confirmed_in_original=true、"
                    "confirmed_in_context=true；否则必须返回无问题。"
                    "同一区域的全部问题写入 issues，但只返回一个可供 edit 模型"
                    "一次重绘的原图 bbox。所有 bbox 使用图像0整数 [0,1000] 坐标。"
                ),
                images=(image, context_crop, detail_crop),
                response_schema=_V41_INSPECTION_SCHEMA,
                metadata={
                    "workflow": "adversarial_consensus",
                    "coverage_mode": "harness_v41",
                    "content_region_id": content["region_id"],
                    "semantic_group_id": content["semantic_group_id"],
                    "semantic_unit_kind": content["semantic_unit_kind"],
                    "content_kind": content["content_kind"],
                    "inspection_role": inspection_role,
                    "content_bbox_norm": content_bbox.as_list(),
                    "detail_bbox_norm": detail_bbox.as_list(),
                    "context_bbox_norm": context_bbox.as_list(),
                    "detail_crop_size": detail_crop.metadata.get(
                        "review_crop_size"
                    ),
                    "zoom_scale": detail_crop.metadata.get(
                        "zoom_scale",
                        1,
                    ),
                    "context_crop_size": context_crop.metadata.get(
                        "review_crop_size"
                    ),
                },
            )
        )
        parsed = dict(parse_response(response))
        confirmed = (
            parsed.get("defect_present") is not True
            or (
                parsed.get("confirmed_in_original") is True
                and parsed.get("confirmed_in_context") is True
            )
        )
        if not confirmed:
            parsed.update(
                {
                    "defect_present": False,
                    "materiality": "none",
                    "confidence": min(
                        float(parsed.get("confidence", 0.0)),
                        0.5,
                    ),
                    "grounding_status": "invalid",
                    "issues": [],
                    "rationale": (
                        "候选问题未同时通过原图与上下文视图确认，程序门控丢弃。"
                    ),
                }
            )
        parsed["crop_boundary_complete"] = True
        parsed["revised_content_bbox_norm"] = content_bbox.as_list()
        return parsed, detail_crop


class HarnessedEditGroundedConsensusCritic(
    EditGroundedAdversarialConsensusCritic
):
    """V4 consensus with evidence routing and safe terminal semantics."""

    async def critique(
        self,
        image: ImageAsset,
        context: CriticContext | None = None,
    ):
        result = await super().critique(image, context)
        coverage_summary = getattr(
            self.text_auditor,
            "last_summary",
            {},
        )
        coverage_status = str(
            coverage_summary.get("coverage_status", "failed")
        )
        result.metadata.update(
            {
                "coverage_status": coverage_status,
                "coverage_ledger": copy.deepcopy(
                    coverage_summary.get("coverage_ledger", [])
                ),
                "semantic_zones": copy.deepcopy(
                    coverage_summary.get("semantic_zones", [])
                ),
                "findings_non_exhaustive": (
                    coverage_status != "ok"
                ),
                "harness_inventory_health": copy.deepcopy(
                    coverage_summary.get(
                        "harness_inventory_health",
                        {},
                    )
                ),
            }
        )
        if coverage_status != "ok":
            result.metadata["quality_verdict"] = result.verdict.value
            result.metadata["harness_abstention_reason"] = str(
                coverage_summary.get(
                    "harness_abstention_reason",
                    "内容覆盖账本未全部核销。",
                )
            )
            result.verdict = Verdict.UNCERTAIN
            result.confidence = min(result.confidence, 0.5)
        return result

    async def _propose(
        self,
        image: ImageAsset,
        context: CriticContext,
        role: CriticRole,
    ) -> tuple[CriticRole, list[Finding], dict[str, Any]]:
        role, findings, summary = await super()._propose(
            image,
            context,
            role,
        )
        if role.name == "text_coverage":
            return role, findings, summary
        rerouted: list[Finding] = []
        raw_findings = summary.get("findings", [])
        if isinstance(raw_findings, list):
            for raw in raw_findings:
                if (
                    not isinstance(raw, Mapping)
                    or not str(raw.get("artifact_type", "")).startswith(
                        "text_"
                    )
                    or not candidate_is_core(raw)
                ):
                    continue
                try:
                    finding = Finding.from_mapping(
                        raw,
                        default_region_id=f"{image.id}:full",
                        source="role:rerouted_text_evidence",
                    )
                except (TypeError, ValueError):
                    continue
                finding.metadata.update(
                    {
                        "edit_target": str(
                            raw.get("edit_target", "")
                        ).strip(),
                        "edit_region_rationale": str(
                            raw.get("edit_region_rationale", "")
                        ).strip(),
                        "original_bbox_norm": finding.bbox.as_list(),
                        "grounding_status": "pending",
                        "defect_present": raw.get("defect_present"),
                        "materiality": str(
                            raw.get("materiality", "")
                        ).strip().lower(),
                        "evidence_basis": str(
                            raw.get("evidence_basis", "")
                        ).strip().lower(),
                        "harness_rerouted": True,
                        "harness_source_role": role.name,
                    }
                )
                rerouted.append(finding)
        summary = dict(summary)
        summary["harness_rerouted_finding_count"] = len(rerouted)
        return role, [*findings, *rerouted], summary

    async def _judge(
        self,
        *,
        image: ImageAsset,
        context: CriticContext,
        role_summaries: list[dict[str, Any]],
        candidates: Mapping[str, Finding],
        challenges: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, Any]:
        parsed = await super()._judge(
            image=image,
            context=context,
            role_summaries=role_summaries,
            candidates=candidates,
            challenges=challenges,
        )
        accepted = parsed.get("accepted_candidate_ids", [])
        terminal_evidence_without_candidate = (
            not candidates
            and any(
                str(summary.get("verdict", "uncertain"))
                in {"artifact_present", "uncertain"}
                for summary in role_summaries
            )
        )
        if not accepted and terminal_evidence_without_candidate:
            parsed["verdict"] = "uncertain"
            parsed["confidence"] = min(
                float(parsed.get("confidence", 0.5)),
                0.5,
            )
            parsed["harness_abstention_reason"] = (
                "至少一个角色报告有问题或覆盖不确定，但没有形成可复核候选；"
                "V4 禁止将该矛盾状态降为无问题。"
            )
        return parsed


class V41EditGroundedConsensusCritic(
    HarnessedEditGroundedConsensusCritic
):
    """V4.1 terminal semantics with explicit per-channel coverage."""

    async def critique(
        self,
        image: ImageAsset,
        context: CriticContext | None = None,
    ):
        result = await super().critique(image, context)
        summary = getattr(self.text_auditor, "last_summary", {})
        result.metadata.update(
            {
                "pipeline_version": "v4.1",
                "coverage_channels": copy.deepcopy(
                    summary.get("coverage_channels", {})
                ),
                "text_coverage": copy.deepcopy(
                    summary.get("text_coverage", {})
                ),
                "artifact_coverage": copy.deepcopy(
                    summary.get("artifact_coverage", {})
                ),
            }
        )
        return result
