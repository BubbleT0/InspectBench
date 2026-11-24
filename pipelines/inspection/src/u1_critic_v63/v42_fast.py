from __future__ import annotations

import asyncio
import copy
import json
import time
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
    VerificationStatus,
)
from critic_workflows.utils import parse_response

from .adaptive_coverage import (
    _COVERAGE_AUDIT_SCHEMA,
    _INVENTORY_SCHEMA,
    _area,
    _intersection,
    _issue_is_core,
    _normalize_regions,
)
from .adaptive_image import overview_image
from .edit_consensus import _EDIT_CHALLENGE_SCHEMA, is_near_full_frame
from .quality_policy import candidate_is_core
from .v4_harness import (
    _V41_INSPECTION_SCHEMA,
    V41AdaptiveContentCoverageAuditor,
    V41EditGroundedConsensusCritic,
    build_multiscale_review_bboxes,
)


_V42_BATCH_REGION_SCHEMA = copy.deepcopy(_V41_INSPECTION_SCHEMA)
_V42_BATCH_REGION_SCHEMA["required"] = [
    "batch_region_id",
    *list(_V42_BATCH_REGION_SCHEMA["required"]),
]
_V42_BATCH_REGION_SCHEMA["properties"] = {
    "batch_region_id": {"type": "string"},
    **dict(_V42_BATCH_REGION_SCHEMA["properties"]),
}
_V42_BATCH_INSPECTION_SCHEMA: Mapping[str, Any] = {
    "type": "object",
    "required": ["region_results", "rationale"],
    "properties": {
        "region_results": {
            "type": "array",
            "items": _V42_BATCH_REGION_SCHEMA,
        },
        "rationale": {"type": "string"},
    },
}

_V42_BATCH_CHALLENGE_ITEM_SCHEMA = copy.deepcopy(_EDIT_CHALLENGE_SCHEMA)
_V42_BATCH_CHALLENGE_ITEM_SCHEMA["required"] = [
    "candidate_id",
    *list(_V42_BATCH_CHALLENGE_ITEM_SCHEMA["required"]),
]
_V42_BATCH_CHALLENGE_ITEM_SCHEMA["properties"] = {
    "candidate_id": {"type": "string"},
    **dict(_V42_BATCH_CHALLENGE_ITEM_SCHEMA["properties"]),
}
_V42_BATCH_CHALLENGE_SCHEMA: Mapping[str, Any] = {
    "type": "object",
    "required": ["reviews"],
    "properties": {
        "reviews": {
            "type": "array",
            "items": _V42_BATCH_CHALLENGE_ITEM_SCHEMA,
        }
    },
}

_V42_EDIT_TARGET_DECOMPOSITION_SCHEMA: Mapping[str, Any] = {
    "type": "object",
    "required": ["edit_regions", "rationale"],
    "properties": {
        "edit_regions": {
            "type": "array",
            "items": {
                "type": "object",
                "required": [
                    "bbox_norm",
                    "edit_target",
                    "rationale",
                ],
                "properties": {
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
                    "edit_target": {"type": "string"},
                    "rationale": {"type": "string"},
                },
            },
        },
        "rationale": {"type": "string"},
    },
}

_V42_COMBINED_INVENTORY_SCHEMA: Mapping[str, Any] = {
    "type": "object",
    "required": [
        "text_regions",
        "artifact_regions",
        "text_confirmed_empty",
        "artifact_confirmed_empty",
        "rationale",
    ],
    "properties": {
        "text_regions": copy.deepcopy(_INVENTORY_SCHEMA["properties"]["regions"]),
        "artifact_regions": copy.deepcopy(
            _INVENTORY_SCHEMA["properties"]["regions"]
        ),
        "text_confirmed_empty": {"type": "boolean"},
        "artifact_confirmed_empty": {"type": "boolean"},
        "rationale": {"type": "string"},
    },
}


def _chunks(values: Sequence[Any], size: int) -> list[list[Any]]:
    return [list(values[index : index + size]) for index in range(0, len(values), size)]


class V42AdaptiveContentCoverageAuditor(V41AdaptiveContentCoverageAuditor):
    """V4.1 coverage semantics with multi-region inspection requests."""

    coverage_mode_name = "harness_v42"
    coerce_targeted_recovery_channel = True
    targeted_recovery_crop_local = True

    def __init__(
        self,
        model: Any,
        image_tool: Any,
        *,
        text_batch_size: int = 2,
        text_crosscheck_batch_size: int | None = None,
        artifact_batch_size: int = 4,
        combined_inventory: bool = False,
        combined_coverage_audit: bool = False,
        eager_inventory: bool = False,
        text_crosscheck: bool = False,
        decompose_broad_text: bool = False,
        overlap_coverage_inspection: bool = False,
        review_original_max_side: int = 0,
        inventory_overview_max_side: int = 0,
        **kwargs: Any,
    ):
        super().__init__(model, image_tool, **kwargs)
        if text_batch_size <= 0 or artifact_batch_size <= 0:
            raise ValueError("V4.2 inspection batch sizes must be positive")
        if text_crosscheck_batch_size is None:
            text_crosscheck_batch_size = text_batch_size
        if text_crosscheck_batch_size <= 0:
            raise ValueError(
                "V4.2 text crosscheck batch size must be positive"
            )
        self.text_batch_size = text_batch_size
        self.text_crosscheck_batch_size = text_crosscheck_batch_size
        self.artifact_batch_size = artifact_batch_size
        self.combined_inventory = combined_inventory
        self.combined_coverage_audit = combined_coverage_audit
        self.eager_inventory = eager_inventory
        self.text_crosscheck = text_crosscheck
        self.decompose_broad_text = decompose_broad_text
        self.overlap_coverage_inspection = overlap_coverage_inspection
        if review_original_max_side < 0:
            raise ValueError("V4.2 overview max side must be non-negative")
        self.review_original_max_side = review_original_max_side
        if inventory_overview_max_side < 0:
            raise ValueError(
                "V4.2 inventory overview max side must be non-negative"
            )
        self.inventory_overview_max_side = inventory_overview_max_side
        self._review_original_task: asyncio.Task | None = None
        self._inventory_overview_task: asyncio.Task | None = None
        self._combined_inventory_task: asyncio.Task | None = None
        self._eager_inventory_tasks: dict[
            str, asyncio.Task[list[dict[str, Any]]]
        ] | None = None
        self._recovery_prefetch_started = False
        self._prefetched_recovery_tasks: dict[
            tuple[str, str], asyncio.Task[tuple[list[dict[str, Any]], bool, dict[str, Any]]]
        ] = {}
        self._prefetched_recovery_inspection_task: asyncio.Task[
            dict[tuple[Any, ...], tuple[Finding | None, int, bool]]
        ] | None = None

    async def audit(
        self,
        image: ImageAsset,
        context: CriticContext,
    ) -> tuple[list[Finding], dict[str, Any]]:
        self._recovery_prefetch_started = False
        self._prefetched_recovery_tasks = {}
        self._prefetched_recovery_inspection_task = None
        if not self.eager_inventory:
            findings, summary = await super().audit(image, context)
        else:
            # Zone enumeration and the two content inventories all inspect
            # the same original pixels. Start them together, then let the
            # inherited harness reconcile regions spatially.
            self._semantic_zones = []
            tasks = {
                kind: asyncio.create_task(
                    self._inventory_impl(image, kind)
                )
                for kind in ("text", "entity")
            }
            self._eager_inventory_tasks = tasks
            try:
                findings, summary = await super().audit(image, context)
            finally:
                for task in tasks.values():
                    if not task.done():
                        task.cancel()
                await asyncio.gather(
                    *tasks.values(),
                    return_exceptions=True,
                )
                self._eager_inventory_tasks = None
        if self.decompose_broad_text and findings:
            findings = await self._decompose_broad_text_findings(
                image,
                findings,
            )
            summary = dict(summary)
            summary["suspicious_region_count"] = len(findings)
            summary["v42_edit_target_decomposition"] = True
            self.last_summary = copy.deepcopy(summary)
        return findings, summary

    async def _request_zone_recovery(
        self,
        image: ImageAsset,
        zone_id: str,
        content_kind: str,
    ) -> tuple[list[dict[str, Any]], bool, dict[str, Any]]:
        return await super()._recover_zone_requirement(
            image,
            zone_id,
            content_kind,
        )

    async def _recover_zone_requirement(
        self,
        image: ImageAsset,
        zone_id: str,
        content_kind: str,
    ) -> tuple[list[dict[str, Any]], bool, dict[str, Any]]:
        key = (str(zone_id), str(content_kind))
        task = self._prefetched_recovery_tasks.pop(key, None)
        if task is not None:
            return await task
        return await self._request_zone_recovery(
            image,
            zone_id,
            content_kind,
        )

    def _prefetch_missing_zone_recoveries(
        self,
        image: ImageAsset,
        regions: Sequence[Mapping[str, Any]],
    ) -> None:
        if self._recovery_prefetch_started or not self._semantic_zones:
            return
        self._recovery_prefetch_started = True
        checked_outcomes = [
            {
                "region_id": str(region.get("region_id", "")),
                "content_kind": str(region.get("content_kind", "")),
                "status": "checked",
            }
            for region in regions
        ]
        provisional = self._build_coverage_ledger(
            self._semantic_zones,
            regions,
            checked_outcomes,
        )
        for item in provisional:
            if item.get("state") not in {"pending", "failed"}:
                continue
            key = (
                str(item.get("zone_id", "")),
                str(item.get("content_kind", "")),
            )
            if not all(key) or key in self._prefetched_recovery_tasks:
                continue
            self._prefetched_recovery_tasks[key] = asyncio.create_task(
                self._request_zone_recovery(image, *key)
            )
        if self._prefetched_recovery_tasks:
            snapshot = list(self._prefetched_recovery_tasks.items())
            self._prefetched_recovery_inspection_task = asyncio.create_task(
                self._inspect_prefetched_recoveries(image, snapshot)
            )

    @staticmethod
    def _region_inspection_cache_key(
        region: Mapping[str, Any],
    ) -> tuple[Any, ...]:
        return (
            str(region.get("zone_id", "")),
            str(region.get("region_id", "")),
            str(region.get("content_kind", "")),
            *tuple(region.get("bbox_norm", [])),
        )

    async def _inspect_prefetched_recoveries(
        self,
        image: ImageAsset,
        tasks: Sequence[
            tuple[
                tuple[str, str],
                asyncio.Task[
                    tuple[list[dict[str, Any]], bool, dict[str, Any]]
                ],
            ]
        ],
    ) -> dict[tuple[Any, ...], tuple[Finding | None, int, bool]]:
        recovery_values = await asyncio.gather(*(task for _, task in tasks))
        regions = [
            region
            for (recovered_regions, _, _) in recovery_values
            for region in recovered_regions
        ]
        if not regions:
            return {}
        inspected = await self._inspect_regions_core(image, regions)
        return {
            self._region_inspection_cache_key(region): value
            for region, value in zip(regions, inspected)
        }

    async def _decompose_broad_text_findings(
        self,
        image: ImageAsset,
        findings: Sequence[Finding],
    ) -> list[Finding]:
        # This is a repair-target normalizer, not another inspection pass.
        # Small cards and labels already have a usable edit region; asking the
        # model to localize every one only adds latency. Trigger only when a
        # confirmed text-only finding is broad enough to plausibly contain
        # several spatially independent repaint targets.
        minimum_decomposition_area = 40_000

        def text_only(finding: Finding) -> bool:
            raw_issues = finding.metadata.get("issues", [])
            issues = raw_issues if isinstance(raw_issues, list) else []
            artifact_types = [
                str(issue.get("artifact_type", ""))
                for issue in issues
                if isinstance(issue, Mapping)
            ]
            if not artifact_types:
                artifact_types = [finding.artifact_type]
            return bool(artifact_types) and all(
                value.startswith("text_") for value in artifact_types
            )

        candidate_indexes = {
            index
            for index, finding in enumerate(findings)
            if finding.source == "role:text_coverage"
            and text_only(finding)
            and _area(finding.bbox) >= minimum_decomposition_area
            and (
                finding.bbox.x2 - finding.bbox.x1 >= 250
                or finding.bbox.y2 - finding.bbox.y1 >= 400
            )
        }
        # When a broad entity inspection encloses a more precise text
        # inspection of the same pixels, localize the precise text candidate
        # only. The deterministic final assembler will remove the redundant
        # outer container after the split children exist.
        for outer_index in tuple(candidate_indexes):
            outer = findings[outer_index]
            for inner_index in candidate_indexes:
                if inner_index == outer_index:
                    continue
                inner = findings[inner_index]
                if _area(inner.bbox) >= _area(outer.bbox):
                    continue
                intersection = _intersection(outer.bbox, inner.bbox)
                if intersection / _area(inner.bbox) >= 0.9:
                    candidate_indexes.discard(outer_index)
                    break
        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def decompose(
            index: int,
            finding: Finding,
        ) -> list[Finding]:
            if index not in candidate_indexes:
                return [finding]
            async with semaphore:
                return await self._decompose_one_text_finding(
                    image,
                    finding,
                )

        groups = await asyncio.gather(
            *(decompose(index, finding) for index, finding in enumerate(findings))
        )
        return [finding for group in groups for finding in group]

    async def _decompose_one_text_finding(
        self,
        image: ImageAsset,
        finding: Finding,
    ) -> list[Finding]:
        crop = await self.image_tool.crop(
            image,
            Region(
                id=f"edit-target-decompose:{finding.region_id}",
                bbox=finding.bbox,
                label="text_edit_target",
            ),
        )
        response = await self.model.complete(
            ModelRequest(
                stage="content.edit_target.decompose",
                system_prompt=(
                    "你是局部重绘目标拆分员。已有文字缺陷已经过原图和上下文"
                    "确认，你只负责定位，不重新判断缺陷、不新增问题。图像0是"
                    "待拆分区域，图像1是原图上下文。所有自然语言使用简体中文。"
                ),
                user_prompt=(
                    "按后续 edit 模型的一次局部重绘来划分：若缺陷位于不同设施、"
                    "节点、卡片、图标或空间分离的标签，必须逐个输出独立区域；"
                    "若确实是一个语义连续、适合整体重绘的文字块，则只输出一个。"
                    "禁止返回一个大框套住多个可独立重绘的小框，也不要按单个字或"
                    "词拆碎。bbox 使用图像0裁剪自身整数 [0,1000] 坐标。必须覆盖"
                    "已有缺陷涉及的全部可编辑目标，不得枚举图像0之外内容。\n"
                    f"已有编辑目标：{finding.metadata.get('edit_target', '')}\n"
                    f"已确认问题：{finding.observation}\n"
                    f"异常说明：{finding.abnormality}"
                ),
                images=(crop, image),
                response_schema=_V42_EDIT_TARGET_DECOMPOSITION_SCHEMA,
                metadata={
                    "workflow": "adversarial_consensus",
                    "coverage_mode": "harness_v42",
                    "source_region_id": finding.region_id,
                    "bbox_coordinate_frame": "target_crop_bbox1000",
                },
            )
        )
        parsed = parse_response(response)
        raw_regions = parsed.get("edit_regions", [])
        if not isinstance(raw_regions, list) or not raw_regions:
            return [finding]
        mapped: list[tuple[BBox, str, str]] = []
        for raw in raw_regions:
            if not isinstance(raw, Mapping):
                continue
            try:
                local = BBox.from_value(raw.get("bbox_norm"))
                parent = finding.bbox
                full = BBox(
                    round(parent.x1 + local.x1 * (parent.x2 - parent.x1) / 1000),
                    round(parent.y1 + local.y1 * (parent.y2 - parent.y1) / 1000),
                    round(parent.x1 + local.x2 * (parent.x2 - parent.x1) / 1000),
                    round(parent.y1 + local.y2 * (parent.y2 - parent.y1) / 1000),
                )
            except (TypeError, ValueError):
                continue
            target = str(raw.get("edit_target", "")).strip()
            rationale = str(raw.get("rationale", "")).strip()
            if not target or not rationale or is_near_full_frame(full):
                continue
            mapped.append((full, target, rationale))
        if not mapped:
            return [finding]
        for index, (left, _, _) in enumerate(mapped):
            for right, _, _ in mapped[index + 1 :]:
                intersection = _intersection(left, right)
                smaller = min(_area(left), _area(right))
                if smaller and intersection / smaller >= 0.9:
                    return [finding]
        output: list[Finding] = []
        for index, (bbox, target, rationale) in enumerate(mapped, 1):
            value = copy.deepcopy(finding)
            value.region_id = f"{finding.region_id}:split{index:03d}"
            value.bbox = bbox
            value.observation = f"{target}：{finding.observation}"
            value.metadata.update(
                {
                    "edit_target": target,
                    "edit_region_rationale": rationale,
                    "original_bbox_norm": bbox.as_list(),
                    "content_bbox_norm": bbox.as_list(),
                    "parent_content_bbox_norm": finding.bbox.as_list(),
                    "semantic_group_id": value.region_id,
                    "semantic_unit_kind": "independent_label",
                    "text_entity_atomic": True,
                    "v42_edit_target_decomposed": True,
                    "v42_decomposition_count": len(mapped),
                }
            )
            output.append(value)
        return output

    async def _review_original(self, image: ImageAsset) -> ImageAsset:
        if self.review_original_max_side <= 0:
            return image
        if self._review_original_task is None:
            self._review_original_task = asyncio.create_task(
                overview_image(
                    image,
                    max_side=self.review_original_max_side,
                )
            )
        return await self._review_original_task

    async def _inventory_overview(self, image: ImageAsset) -> ImageAsset:
        if self.inventory_overview_max_side <= 0:
            return image
        if self._inventory_overview_task is None:
            self._inventory_overview_task = asyncio.create_task(
                overview_image(
                    image,
                    max_side=self.inventory_overview_max_side,
                )
            )
        return await self._inventory_overview_task

    async def _zone_inventory(
        self,
        image: ImageAsset,
        *,
        blind: bool,
    ) -> list[dict[str, Any]]:
        return await super()._zone_inventory(
            await self._inventory_overview(image),
            blind=blind,
        )

    async def _inventory(
        self,
        image: ImageAsset,
        kind: str,
    ) -> list[dict[str, Any]]:
        if self._eager_inventory_tasks is not None:
            task = self._eager_inventory_tasks.get(kind)
            if task is not None:
                return copy.deepcopy(await task)
        return await self._inventory_impl(image, kind)

    async def _inventory_impl(
        self,
        image: ImageAsset,
        kind: str,
    ) -> list[dict[str, Any]]:
        if not self.combined_inventory:
            inventory_image = (
                image
                if kind == "text"
                else await self._inventory_overview(image)
            )
            return await super()._inventory(inventory_image, kind)
        if self._combined_inventory_task is None:
            self._combined_inventory_task = asyncio.create_task(
                self._request_combined_inventory(image)
            )
        values = await self._combined_inventory_task
        return copy.deepcopy(values[kind])

    async def _request_combined_inventory(
        self,
        image: ImageAsset,
    ) -> dict[str, list[dict[str, Any]]]:
        response = await self.model.complete(
            ModelRequest(
                stage="content.inventory.combined",
                system_prompt=(
                    "你是内容覆盖清单员，只依据原图像素同时建立文字与非文字区域"
                    "清单，不做缺陷判断。不得读取文件名、目录标签、参考答案或其他"
                    "输入元数据，不得固定网格。所有自然语言使用简体中文。"
                ),
                user_prompt=(
                    "text_regions 必须枚举所有被设计为文字内容的完整语义编辑单元，"
                    "乱码和不可读文字也必须列出；主标题、同一卡片、表格或榜单完整"
                    "一行、连续正文面板分别作为完整单元，禁止按词或字符拆碎。"
                    "散布在不同设施、节点、人物、图标或卡片旁的标签即使属于同一"
                    "大场景，也必须逐个列为 independent_label，禁止用整张场景或"
                    "整块仪表板代替这些空间独立标签。"
                    "artifact_regions 必须枚举所有主要独立实体和语义完整的非文字"
                    "版面区块，包括人物、动物、物体、建筑、图表和插图，正常实体也"
                    "必须列出。两份清单分别核对，不得用一份替代另一份。所有 bbox"
                    "使用原图整数 [0,1000] 坐标。只有原图确实没有对应内容时，才能"
                    "设置相应 confirmed_empty=true。"
                ),
                images=(image,),
                response_schema=_V42_COMBINED_INVENTORY_SCHEMA,
                metadata={
                    "workflow": "adversarial_consensus",
                    "coverage_mode": "harness_v42",
                    "combined_inventory": True,
                },
            )
        )
        parsed = parse_response(response)
        text_raw = parsed.get("text_regions", [])
        artifact_raw = parsed.get("artifact_regions", [])
        text_regions = [
            region
            for region in _normalize_regions(
                text_raw,
                source="content.inventory.text",
            )
            if region["content_kind"] == "text"
        ]
        artifact_regions = [
            region
            for region in _normalize_regions(
                artifact_raw,
                source="content.inventory.entity",
            )
            if region["content_kind"] != "text"
        ]
        for kind, raw, regions, confirmed_key in (
            ("text", text_raw, text_regions, "text_confirmed_empty"),
            (
                "entity",
                artifact_raw,
                artifact_regions,
                "artifact_confirmed_empty",
            ),
        ):
            raw_count = len(raw) if isinstance(raw, list) else 0
            confirmed_empty = parsed.get(confirmed_key) is True
            status = "ok" if regions or confirmed_empty else "failed"
            self._inventory_health[kind] = {
                "status": status,
                "attempts": 1,
                "raw_count": raw_count,
                "valid_count": len(regions),
                "confirmed_empty": confirmed_empty,
                "validation_errors": (
                    []
                    if status == "ok"
                    else ["合并清单未返回有效区域，也未确认该通道为空。"]
                ),
            }
        return {"text": text_regions, "entity": artifact_regions}

    async def _coverage_audit(
        self,
        image: ImageAsset,
        regions: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        if not self.combined_coverage_audit:
            return await super()._coverage_audit(image, regions)
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
                stage="content.coverage.audit.combined",
                system_prompt=(
                    "你是独立覆盖终审员，只核对内容清单漏项，不判断图像缺陷、"
                    "审美或外部事实。文字与非文字通道必须分别完整核对。所有"
                    "自然语言使用简体中文。"
                ),
                user_prompt=(
                    "对照原图与已有清单，补充明显漏掉的独立文字语义单元、主要"
                    "实体或语义完整非文字区块；不得重复已有区域。标题、标签、"
                    "图例、角落信息、密集小字和主要实体均不得遗漏。没有漏项时"
                    "返回空数组。已有清单：\n"
                    + json.dumps(inventory, ensure_ascii=False)
                ),
                images=(image,),
                response_schema=_COVERAGE_AUDIT_SCHEMA,
                metadata={
                    "workflow": "adversarial_consensus",
                    "coverage_mode": "harness_v42",
                    "combined_coverage_audit": True,
                },
            )
        )
        parsed = parse_response(response)
        return _normalize_regions(
            parsed.get("missing_regions", []),
            source="content.coverage.audit.combined",
        )

    async def _inspect_regions(
        self,
        image: ImageAsset,
        regions: Sequence[Mapping[str, Any]],
    ) -> list[tuple[Finding | None, int, bool]]:
        initial_call = not self._recovery_prefetch_started
        self._prefetch_missing_zone_recoveries(image, regions)
        if (
            not initial_call
            and self._prefetched_recovery_inspection_task is not None
        ):
            cached = await self._prefetched_recovery_inspection_task
            output: list[tuple[Finding | None, int, bool] | None] = [
                cached.get(self._region_inspection_cache_key(region))
                for region in regions
            ]
            missing_indexes = [
                index for index, value in enumerate(output) if value is None
            ]
            if missing_indexes:
                fresh = await self._inspect_regions_core(
                    image,
                    [regions[index] for index in missing_indexes],
                )
                for index, value in zip(missing_indexes, fresh):
                    output[index] = value
            return [value for value in output if value is not None]
        return await self._inspect_regions_core(image, regions)

    async def _inspect_regions_during_coverage_audit(
        self,
        image: ImageAsset,
        regions: Sequence[Mapping[str, Any]],
    ) -> list[tuple[Finding | None, int, bool]]:
        # Do not start semantic-zone recovery from the preliminary inventory;
        # the independent coverage audit may still supply those regions.
        return await self._inspect_regions_core(image, regions)

    def _prepare_completed_inventory_regions(
        self,
        image: ImageAsset,
        regions: Sequence[Mapping[str, Any]],
    ) -> None:
        self._prefetch_missing_zone_recoveries(image, regions)

    async def _inspect_additional_regions_after_coverage_audit(
        self,
        image: ImageAsset,
        regions: Sequence[Mapping[str, Any]],
    ) -> list[tuple[Finding | None, int, bool]]:
        return await self._inspect_regions_core(image, regions)

    async def _inspect_regions_core(
        self,
        image: ImageAsset,
        regions: Sequence[Mapping[str, Any]],
    ) -> list[tuple[Finding | None, int, bool]]:
        indexed = list(enumerate(regions))
        text = [item for item in indexed if item[1].get("content_kind") == "text"]
        artifact = [item for item in indexed if item[1].get("content_kind") != "text"]
        text_batch_size = self._inspection_batch_size(
            "text",
            len(text),
            configured=self.text_batch_size,
        )
        artifact_batch_size = self._inspection_batch_size(
            "entity",
            len(artifact),
            configured=self.artifact_batch_size,
        )
        text_crosscheck_batch_size = self._inspection_batch_size(
            "text_crosscheck",
            len(text),
            configured=self.text_crosscheck_batch_size,
        )
        primary_batches = [
            *_chunks(text, text_batch_size),
            *_chunks(artifact, artifact_batch_size),
        ]
        crosscheck_batches = (
            _chunks(text, text_crosscheck_batch_size)
            if self.text_crosscheck
            else []
        )
        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def call_batch(batch, inspection_pass):
            async with semaphore:
                values = await self._inspect_batch(
                    image,
                    batch,
                    inspection_pass=inspection_pass,
                )
            missing = [
                item
                for item, value in zip(batch, values)
                if value is None
            ]
            if not missing or len(batch) <= 1:
                return values
            if len(missing) == len(batch):
                midpoint = self._failed_batch_split_index(len(missing))
                retry_batches = [missing[:midpoint], missing[midpoint:]]
            else:
                retry_batches = [missing]
            retry_batches = [item for item in retry_batches if item]
            retry_outputs = await asyncio.gather(
                *(
                    call_batch(retry_batch, inspection_pass)
                    for retry_batch in retry_batches
                )
            )
            recovered = {
                indexed_item[0]: value
                for retry_batch, retry_values in zip(
                    retry_batches,
                    retry_outputs,
                )
                for indexed_item, value in zip(retry_batch, retry_values)
            }
            return [
                recovered.get(indexed_item[0]) if value is None else value
                for indexed_item, value in zip(batch, values)
            ]

        primary_outputs, crosscheck_outputs = await asyncio.gather(
            asyncio.gather(
                *(call_batch(batch, "primary") for batch in primary_batches)
            ),
            asyncio.gather(
                *(
                    call_batch(batch, "text_crosscheck")
                    for batch in crosscheck_batches
                )
            ),
        )
        primary_by_index: dict[
            int, tuple[Finding | None, int, bool] | None
        ] = {}
        for batch, values in zip(primary_batches, primary_outputs):
            for (index, _), value in zip(batch, values):
                primary_by_index[index] = value
        crosscheck_by_index: dict[
            int, tuple[Finding | None, int, bool] | None
        ] = {}
        for batch, values in zip(crosscheck_batches, crosscheck_outputs):
            for (index, _), value in zip(batch, values):
                crosscheck_by_index[index] = value

        output: dict[int, tuple[Finding | None, int, bool]] = {}
        missing: list[tuple[int, Mapping[str, Any]]] = []
        for index, content in indexed:
            value = primary_by_index.get(index)
            if self.text_crosscheck and content.get("content_kind") == "text":
                value = self._merge_crosschecked_inspection(
                    value,
                    crosscheck_by_index.get(index),
                )
            if value is None:
                missing.append((index, content))
            else:
                output[index] = value

        async def fallback(index: int, content: Mapping[str, Any]):
            async with semaphore:
                value = await super(
                    V42AdaptiveContentCoverageAuditor,
                    self,
                )._inspect_region(image, content)
            return index, value

        if missing:
            fallback_values = await asyncio.gather(
                *(fallback(index, content) for index, content in missing)
            )
            output.update(fallback_values)
        return [output[index] for index in range(len(regions))]

    def _inspection_batch_size(
        self,
        channel: str,
        region_count: int,
        *,
        configured: int,
    ) -> int:
        """Resolve a channel batch size; profiles may override safely."""

        del channel, region_count
        return configured

    def _failed_batch_split_index(self, batch_size: int) -> int:
        """Choose the split point after a batch response loses every item."""

        return max(1, batch_size // 2)

    def _include_inventory_description(
        self,
        content: Mapping[str, Any],
        inspection_pass: str,
    ) -> bool:
        del content
        return inspection_pass != "text_crosscheck"

    def _artifact_inspection_rule(
        self,
        contents: Sequence[Mapping[str, Any]],
    ) -> str:
        del contents
        return (
            "对每个实体或版面区分别检查明显渲染、结构、拓扑、关系和物理"
            "错误；文字由独立文字通道完整检查，本通道不得重复报告文字问题。"
            "正常透视、遮挡、景深、风格化和轻微 AI 瑕疵不报告。"
        )

    async def _inspect_batch(
        self,
        image: ImageAsset,
        batch: Sequence[tuple[int, Mapping[str, Any]]],
        *,
        inspection_pass: str = "primary",
    ) -> list[tuple[Finding | None, int, bool] | None]:
        view_prepare_started = time.monotonic()

        async def build_views(
            local_index: int,
            content: Mapping[str, Any],
        ) -> dict[str, Any]:
            content_bbox = BBox.from_value(content["bbox_norm"])
            zone = next(
                (
                    item
                    for item in self._semantic_zones
                    if str(item.get("zone_id", ""))
                    == str(content.get("zone_id", ""))
                ),
                None,
            )
            try:
                zone_bbox = BBox.from_value(zone.get("bbox_norm")) if zone else None
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
            return {
                "batch_region_id": f"br{local_index:03d}",
                "image_id": image.id,
                "content": content,
                "content_bbox": content_bbox,
                "detail_bbox": detail_bbox,
                "context_bbox": context_bbox,
                "detail_crop": detail_crop,
                "context_crop": context_crop,
            }

        overview_task = asyncio.create_task(self._review_original(image))
        views = await asyncio.gather(
            *(
                build_views(local_index, content)
                for local_index, (_, content) in enumerate(batch)
            )
        )
        images: list[ImageAsset] = [await overview_task]
        descriptors: list[dict[str, Any]] = []
        for view in views:
            context_index = len(images)
            images.append(view["context_crop"])
            detail_index = len(images)
            images.append(view["detail_crop"])
            content = view["content"]
            descriptor = {
                    "batch_region_id": view["batch_region_id"],
                    "content_kind": content["content_kind"],
                    "semantic_unit_kind": content["semantic_unit_kind"],
                    "content_bbox_norm": view["content_bbox"].as_list(),
                    "context_bbox_norm": view["context_bbox"].as_list(),
                    "detail_bbox_norm": view["detail_bbox"].as_list(),
                    "context_image_index": context_index,
                    "detail_image_index": detail_index,
                }
            # The second text pass must be perceptually independent. Feeding
            # the inventory's transcription back to the model anchors it on
            # what the text is *supposed* to say and can make visibly malformed
            # glyphs look correct. Keep only geometry and image bindings.
            if self._include_inventory_description(
                content,
                inspection_pass,
            ):
                descriptor["description"] = content["description"]
            descriptors.append(descriptor)
        channel = (
            "text"
            if all(view["content"].get("content_kind") == "text" for view in views)
            else "artifact"
        )
        view_prepare_seconds = time.monotonic() - view_prepare_started
        if channel == "text":
            if inspection_pass == "text_crosscheck":
                inspection_rule = (
                    "这是独立的第二遍文字漏判复核，不参考其他检查结论。对每个"
                    "语义单元从左到右、从上到下逐字扫视，尤其检查主标题、角落"
                    "标签、密集小字中的明显乱码、伪文字、断裂、粘连、扭曲、"
                    "缺损和不可辨认字符。能根据上下文猜出文字不等于视觉完整；"
                    "即使仍可读，明显错位、笔画畸变、异常重叠或结构破坏也必须"
                    "报告。清单刻意不提供文字转写，禁止自行假定标准字形或正确"
                    "文案，只比较直接可见的字形结构、排印关系和像素证据。不要"
                    "检查年份、片名或外部事实。"
                )
            else:
                inspection_rule = (
                    "对每个文字语义单元逐字检查明显乱码、伪文字、断裂、粘连、"
                    "扭曲、缺损和不可辨认字符；同时只检查图内可直接验证的格式或"
                    "内容矛盾，不得依赖正确文案、片名、年份或语言补全。"
                )
        else:
            inspection_rule = self._artifact_inspection_rule(
                [view["content"] for view in views]
            )
        response = await self.model.complete(
            ModelRequest(
                stage=(
                    "content.inspect.multiscale_batch_crosscheck"
                    if inspection_pass == "text_crosscheck"
                    else "content.inspect.multiscale_batch"
                ),
                system_prompt=(
                    "你是面向局部重绘的多区域核心缺陷审查员。图像0始终是原图；"
                    "其余图像按清单分别对应上下文视图和可能无损放大的细节视图。"
                    "每个区域必须独立完整检查，只报告肉眼明显且局部重绘能实质改善"
                    "的 major/critical 问题，忽略世界知识、审美和轻微瑕疵。所有"
                    "自然语言使用简体中文。"
                ),
                user_prompt=(
                    inspection_rule
                    + "\n区域与图像编号清单：\n"
                    + json.dumps(descriptors, ensure_ascii=False)
                    + "\n必须为清单中的每个 batch_region_id 恰好返回一个结果，"
                    "不得遗漏、重复或新增 ID。裁剪只用于观察，候选必须在图像0和"
                    "对应上下文图中同时成立；否则返回无问题。同一区域全部问题写入"
                    "issues，但只给一个适合一次局部重绘的原图 bbox。所有 bbox 均"
                    "使用图像0整数 [0,1000] 坐标。"
                ),
                images=tuple(images),
                response_schema=_V42_BATCH_INSPECTION_SCHEMA,
                metadata={
                    "workflow": "adversarial_consensus",
                    "coverage_mode": "harness_v42",
                    "review_original_max_side": self.review_original_max_side,
                    "coverage_channel": channel,
                    "batch_size": len(batch),
                    "batch_region_ids": [
                        item["batch_region_id"] for item in descriptors
                    ],
                    "inspection_pass": inspection_pass,
                    "view_prepare_seconds": round(view_prepare_seconds, 4),
                },
            )
        )
        parsed = parse_response(response)
        raw_results = parsed.get("region_results", [])
        by_id: dict[str, list[Mapping[str, Any]]] = {}
        if isinstance(raw_results, list):
            for raw in raw_results:
                if not isinstance(raw, Mapping):
                    continue
                by_id.setdefault(str(raw.get("batch_region_id", "")), []).append(raw)
        output: list[tuple[Finding | None, int, bool] | None] = []
        for view in views:
            values = by_id.get(view["batch_region_id"], [])
            if len(values) != 1:
                output.append(None)
                continue
            output.append(self._finalize_batch_inspection(view, values[0]))
        return output

    @staticmethod
    def _merge_crosschecked_inspection(
        primary: tuple[Finding | None, int, bool] | None,
        crosscheck: tuple[Finding | None, int, bool] | None,
    ) -> tuple[Finding | None, int, bool] | None:
        if primary is None:
            return crosscheck
        if crosscheck is None:
            return primary
        primary_finding, primary_retries, primary_failed = primary
        cross_finding, cross_retries, cross_failed = crosscheck
        if primary_finding is None and cross_finding is None:
            return (
                None,
                max(primary_retries, cross_retries),
                primary_failed and cross_failed,
            )
        if primary_finding is None:
            merged = copy.deepcopy(cross_finding)
        elif cross_finding is None:
            merged = copy.deepcopy(primary_finding)
        else:
            merged = copy.deepcopy(primary_finding)
            merged.confidence = max(
                primary_finding.confidence,
                cross_finding.confidence,
            )
            merged.observation = "；".join(
                dict.fromkeys(
                    value
                    for value in (
                        primary_finding.observation,
                        cross_finding.observation,
                    )
                    if value
                )
            )
            merged.abnormality = "；".join(
                dict.fromkeys(
                    value
                    for value in (
                        primary_finding.abnormality,
                        cross_finding.abnormality,
                    )
                    if value
                )
            )
            merged_issues: list[dict[str, Any]] = []
            seen_issues: set[tuple[str, str, str]] = set()
            for finding in (primary_finding, cross_finding):
                raw_issues = finding.metadata.get("issues", [])
                if not isinstance(raw_issues, list):
                    continue
                for issue in raw_issues:
                    if not isinstance(issue, Mapping):
                        continue
                    key = (
                        str(issue.get("artifact_type", "")),
                        str(issue.get("observation", "")),
                        str(issue.get("abnormality", "")),
                    )
                    if key in seen_issues:
                        continue
                    seen_issues.add(key)
                    merged_issues.append(copy.deepcopy(dict(issue)))
            merged.metadata["issues"] = merged_issues
            merged.metadata["inspection_rationale"] = "；".join(
                dict.fromkeys(
                    str(finding.metadata.get("inspection_rationale", ""))
                    for finding in (primary_finding, cross_finding)
                    if str(
                        finding.metadata.get("inspection_rationale", "")
                    ).strip()
                )
            )
        if merged is not None:
            merged.metadata["v42_text_crosscheck"] = True
        return (
            merged,
            max(primary_retries, cross_retries),
            primary_failed and cross_failed,
        )

    def _finalize_batch_inspection(
        self,
        view: Mapping[str, Any],
        raw: Mapping[str, Any],
    ) -> tuple[Finding | None, int, bool]:
        content = view["content"]
        content_bbox: BBox = view["content_bbox"]
        detail_bbox: BBox = view["detail_bbox"]
        detail_crop: ImageAsset = view["detail_crop"]
        parsed = dict(raw)
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
                    "confidence": min(float(parsed.get("confidence", 0.0)), 0.5),
                    "grounding_status": "invalid",
                    "issues": [],
                }
            )
        if (
            not candidate_is_core(parsed)
            or parsed.get("grounding_status") != "valid"
        ):
            return None, 0, False
        accepted: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()
        raw_issues = parsed.get("issues", [])
        if isinstance(raw_issues, list):
            for issue in raw_issues:
                if not isinstance(issue, Mapping) or not _issue_is_core(issue):
                    continue
                key = (
                    str(issue.get("artifact_type", "")),
                    str(issue.get("observation", "")).strip(),
                    str(issue.get("abnormality", "")).strip(),
                )
                if key in seen:
                    continue
                seen.add(key)
                accepted.append(copy.deepcopy(dict(issue)))
        if not accepted:
            return None, 0, False
        has_text_issue = any(
            str(issue.get("artifact_type", "")).startswith("text_")
            for issue in accepted
        )
        if content.get("content_kind") == "text" or has_text_issue:
            edit_bbox = content_bbox
        else:
            try:
                edit_bbox = BBox.from_value(parsed.get("edit_bbox_full_norm"))
            except (TypeError, ValueError):
                return None, 0, False
            if _area(edit_bbox) <= 0:
                return None, 0, False
            if _intersection(edit_bbox, detail_bbox) / _area(edit_bbox) < 0.8:
                return None, 0, False
        confidence = max(
            min(float(parsed.get("confidence", 0.0)), float(issue["confidence"]))
            for issue in accepted
        )
        if confidence <= 0.6:
            return None, 0, False
        observation = "；".join(
            str(issue.get("observation", "")).strip()
            for issue in accepted
            if str(issue.get("observation", "")).strip()
        )
        abnormality = "；".join(
            str(issue.get("abnormality", "")).strip()
            for issue in accepted
            if str(issue.get("abnormality", "")).strip()
        )
        finding = Finding(
            region_id=str(content["semantic_group_id"]),
            bbox=edit_bbox,
            artifact_type=str(accepted[0]["artifact_type"]),
            observation=observation,
            abnormality=abnormality,
            confidence=confidence,
            evidence_ids=[
                str(view["image_id"]),
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
                "materiality": str(parsed.get("materiality", "major")),
                "evidence_basis": "visible_only",
                "text_entity_atomic": content.get("content_kind") == "text",
                "content_kind": content["content_kind"],
                "content_description": content["description"],
                "semantic_group_id": content["semantic_group_id"],
                "semantic_unit_kind": content["semantic_unit_kind"],
                "designed_as_text": content["designed_as_text"],
                "initial_content_bbox_norm": content_bbox.as_list(),
                "content_bbox_norm": content_bbox.as_list(),
                "review_bbox_norm": detail_bbox.as_list(),
                "inventory_sources": list(content.get("inventory_sources", [])),
                "boundary_retry_count": 0,
                "original_crop_size": detail_crop.metadata.get("original_crop_size"),
                "review_crop_size": detail_crop.metadata.get("review_crop_size"),
                "zoom_scale": detail_crop.metadata.get("zoom_scale", 1),
                "issues": accepted,
                "inspection_roles": (
                    ["text_visual", "text_fidelity"]
                    if content.get("content_kind") == "text"
                    else ["artifact", "embedded_text"]
                ),
                "inspection_rationale": str(parsed.get("rationale", "")),
                "v42_batched_inspection": True,
            },
        )
        return finding, 0, False


class V42EditGroundedConsensusCritic(V41EditGroundedConsensusCritic):
    """V4.1 terminal semantics with batched Skeptic reviews."""

    def __init__(
        self,
        *args: Any,
        challenge_batch_size: int = 4,
        challenge_concurrency: int = 4,
        **kwargs: Any,
    ):
        self.pipeline_version = str(
            kwargs.pop("pipeline_version", "v4.2-experimental")
        )
        self.trust_preverified = bool(kwargs.pop("trust_preverified", False))
        self.deterministic_judge = bool(kwargs.pop("deterministic_judge", False))
        self.coverage_only = bool(kwargs.pop("coverage_only", False))
        self.single_artifact_proposer = bool(
            kwargs.pop("single_artifact_proposer", False)
        )
        self.peer_scout_consensus = bool(
            kwargs.pop("peer_scout_consensus", False)
        )
        self.review_original_max_side = int(
            kwargs.pop("review_original_max_side", 0)
        )
        if self.review_original_max_side < 0:
            raise ValueError("V4.2 overview max side must be non-negative")
        self._review_original_task: asyncio.Task | None = None
        super().__init__(*args, **kwargs)
        if challenge_batch_size <= 0:
            raise ValueError("V4.2 challenge batch size must be positive")
        if challenge_concurrency <= 0:
            raise ValueError("V4.2 challenge concurrency must be positive")
        self.challenge_batch_size = challenge_batch_size
        self.challenge_concurrency = challenge_concurrency
        self._last_challenge_batch_count = 0
        self._last_challenge_wave_count = 0
        self._last_preverified_count = 0
        self._last_skeptic_request_candidate_count = 0
        self._last_peer_scout_consensus_count = 0

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
        if not (
            self.single_artifact_proposer
            and role.name == "core_artifact"
        ):
            return role, findings, summary
        existing = {
            (finding.artifact_type, tuple(finding.bbox.as_list()))
            for finding in findings
        }
        review_candidates: list[Finding] = []
        raw_findings = summary.get("findings", [])
        if isinstance(raw_findings, list):
            for raw in raw_findings:
                if not isinstance(raw, Mapping):
                    continue
                artifact_type = str(raw.get("artifact_type", ""))
                if artifact_type.startswith("text_"):
                    continue
                try:
                    confidence = float(raw.get("confidence", 0.0))
                except (TypeError, ValueError):
                    continue
                if not (
                    raw.get("defect_present") is True
                    and str(raw.get("materiality", "")).lower() == "minor"
                    and str(raw.get("evidence_basis", "")).lower()
                    == "visible_only"
                    and confidence > 0.6
                    and str(raw.get("edit_target", "")).strip()
                ):
                    continue
                try:
                    finding = Finding.from_mapping(
                        raw,
                        default_region_id=f"{image.id}:full",
                        source="role:core_artifact_review",
                    )
                except (TypeError, ValueError):
                    continue
                if is_near_full_frame(finding.bbox):
                    continue
                key = (finding.artifact_type, tuple(finding.bbox.as_list()))
                if key in existing:
                    continue
                finding.metadata.update(
                    {
                        "edit_target": str(raw.get("edit_target", "")).strip(),
                        "edit_region_rationale": str(
                            raw.get("edit_region_rationale", "")
                        ).strip(),
                        "original_bbox_norm": finding.bbox.as_list(),
                        "grounding_status": "pending",
                        "defect_present": True,
                        "materiality": "minor",
                        "evidence_basis": "visible_only",
                        "v42_minor_review_candidate": True,
                    }
                )
                review_candidates.append(finding)
                existing.add(key)
        summary = dict(summary)
        summary["v42_minor_review_candidate_count"] = len(
            review_candidates
        )
        # A 27B response can contradict itself by returning the top-level
        # artifact_present verdict while every structured finding explicitly
        # says defect_present=false. With no positive or reviewable candidate,
        # this is a serialization inconsistency rather than unresolved visual
        # evidence. Missing or mixed evidence still follows the conservative
        # uncertain path.
        explicit_findings = (
            [raw for raw in raw_findings if isinstance(raw, Mapping)]
            if isinstance(raw_findings, list)
            else []
        )
        if (
            not findings
            and not review_candidates
            and explicit_findings
            and len(explicit_findings) == len(raw_findings)
            and all(
                raw.get("defect_present") is False
                for raw in explicit_findings
            )
        ):
            summary["verdict"] = "no_artifact"
            summary["v42_summary_reconciled_from_explicit_negatives"] = True
        return role, [*findings, *review_candidates], summary

    async def critique(
        self,
        image: ImageAsset,
        context: CriticContext | None = None,
    ):
        result = await super().critique(image, context)
        result.metadata["pipeline_version"] = self.pipeline_version
        result.metadata["v42_challenge_batch_size"] = self.challenge_batch_size
        result.metadata["v42_challenge_concurrency"] = (
            self.challenge_concurrency
        )
        result.metadata["v42_challenge_batch_count"] = (
            self._last_challenge_batch_count
        )
        result.metadata["v42_challenge_wave_count"] = (
            self._last_challenge_wave_count
        )
        result.metadata["v42_trust_preverified"] = self.trust_preverified
        result.metadata["v42_deterministic_judge"] = self.deterministic_judge
        result.metadata["v42_coverage_only"] = self.coverage_only
        result.metadata["v42_single_artifact_proposer"] = (
            self.single_artifact_proposer
        )
        result.metadata["v42_review_original_max_side"] = (
            self.review_original_max_side
        )
        if self.pipeline_version.startswith("v6.3"):
            result.metadata["v63_preverified_finding_count"] = (
                self._last_preverified_count
            )
            result.metadata["v63_skeptic_candidate_count"] = (
                self._last_skeptic_request_candidate_count
            )
            result.metadata["v63_peer_scout_consensus_count"] = (
                self._last_peer_scout_consensus_count
            )
            result.metadata["v63_deterministic_judge"] = (
                self.deterministic_judge
            )
            result.metadata["v63_online_consolidation_required"] = True
            triage_metrics = getattr(
                self.text_auditor,
                "v63_triage_metrics",
                None,
            )
            if isinstance(triage_metrics, Mapping):
                result.metadata["v63_triage_metrics"] = copy.deepcopy(
                    dict(triage_metrics)
                )
        adaptive_metrics = getattr(
            self.text_auditor,
            "v62_batch_metrics",
            None,
        )
        if isinstance(adaptive_metrics, Mapping):
            result.metadata["v62_adaptive_batching"] = copy.deepcopy(
                dict(adaptive_metrics)
            )
        return result

    async def _challenge_candidates(
        self,
        image: ImageAsset,
        context: CriticContext,
        candidates: Mapping[str, Finding],
        *,
        semaphore: asyncio.Semaphore,
    ) -> list[tuple[str, VerificationStatus, dict[str, Any]]]:
        preverified: dict[
            str,
            tuple[str, VerificationStatus, dict[str, Any]],
        ] = {}
        entries: list[tuple[str, Finding]] = []
        peer_supported = self._peer_supported_scout_candidates(candidates)
        self._last_peer_scout_consensus_count = len(peer_supported)
        for candidate_id, finding in candidates.items():
            if candidate_id in peer_supported:
                preverified[candidate_id] = self._apply_challenge_review(
                    candidate_id,
                    finding,
                    {
                        "status": "supported",
                        "confidence": finding.confidence,
                        "counterargument": (
                            "两个独立 Artifact 专员在同一区域发现相互印证的问题。"
                        ),
                        "missing_evidence": "",
                        "defect_present": True,
                        "materiality": finding.metadata.get(
                            "materiality",
                            "major",
                        ),
                        "evidence_basis": "visible_only",
                        "grounding_status": "valid",
                        "revised_bbox_norm": None,
                        "grounding_rationale": (
                            "双专员空间证据重叠，保留至在线整编合并。"
                        ),
                        "v63_peer_scout_consensus": True,
                    },
                )
            elif (
                self.trust_preverified
                and finding.source == "role:text_coverage"
                and finding.metadata.get("v42_batched_inspection") is True
            ):
                preverified[candidate_id] = self._apply_challenge_review(
                    candidate_id,
                    finding,
                    {
                        "status": "supported",
                        "confidence": finding.confidence,
                        "counterargument": "多尺度检查已在原图与上下文中确认。",
                        "missing_evidence": "",
                        "defect_present": True,
                        "materiality": finding.metadata.get(
                            "materiality",
                            "major",
                        ),
                        "evidence_basis": "visible_only",
                        "grounding_status": "valid",
                        "revised_bbox_norm": None,
                        "grounding_rationale": (
                            "V4.2 多尺度检查已确认该区域适合局部重绘。"
                        ),
                        "v42_preverified_fast_path": True,
                    },
                )
            else:
                entries.append((candidate_id, finding))
        batches = _chunks(entries, self.challenge_batch_size)
        self._last_preverified_count = len(preverified)
        self._last_skeptic_request_candidate_count = len(entries)
        self._last_challenge_batch_count = len(batches)
        self._last_challenge_wave_count = (
            (
                len(batches) + self.challenge_concurrency - 1
            )
            // self.challenge_concurrency
            if batches
            else 0
        )
        challenge_semaphore = asyncio.Semaphore(
            self.challenge_concurrency
        )

        async def challenge_batch(batch):
            async with challenge_semaphore:
                parsed = await self._request_challenge_batch(
                    image,
                    context,
                    batch,
                )
            output: list[tuple[str, VerificationStatus, dict[str, Any]]] = []
            missing: list[tuple[str, Finding]] = []
            for candidate_id, finding in batch:
                values = parsed.get(candidate_id, [])
                if len(values) != 1:
                    missing.append((candidate_id, finding))
                    continue
                output.append(
                    self._apply_challenge_review(
                        candidate_id,
                        finding,
                        values[0],
                    )
                )
            for candidate_id, finding in missing:
                async with challenge_semaphore:
                    output.append(
                        await self._challenge(
                            image,
                            context,
                            candidate_id,
                            finding,
                        )
                    )
            return output

        grouped = await asyncio.gather(
            *(challenge_batch(batch) for batch in batches)
        )
        by_id: dict[
            str,
            tuple[str, VerificationStatus, dict[str, Any]],
        ] = dict(preverified)
        for values in grouped:
            for value in values:
                by_id[value[0]] = value
        return [by_id[candidate_id] for candidate_id in candidates]

    def _peer_supported_scout_candidates(
        self,
        candidates: Mapping[str, Finding],
    ) -> set[str]:
        """Return cross-specialist candidates with substantial spatial overlap."""

        if not self.peer_scout_consensus:
            return set()
        scout_sources = {
            "role:artifact_structure_scout",
            "role:artifact_rendering_scout",
        }
        values = [
            (candidate_id, finding)
            for candidate_id, finding in candidates.items()
            if finding.source in scout_sources
        ]
        supported: set[str] = set()

        def issue_family(finding: Finding) -> str:
            if finding.artifact_type == "rendering_artifact":
                return "appearance"
            if finding.artifact_type in {
                "structural_artifact",
                "topology_artifact",
                "relational_physical_artifact",
            }:
                return "geometry"
            return finding.artifact_type

        for index, (left_id, left) in enumerate(values):
            for right_id, right in values[index + 1 :]:
                if left.source == right.source:
                    continue
                if issue_family(left) != issue_family(right):
                    continue
                intersection = _intersection(left.bbox, right.bbox)
                smaller_area = min(_area(left.bbox), _area(right.bbox))
                if smaller_area > 0 and intersection / smaller_area >= 0.35:
                    supported.update((left_id, right_id))
        return supported

    async def _judge(
        self,
        *,
        image: ImageAsset,
        context: CriticContext,
        role_summaries: list[dict[str, Any]],
        candidates: Mapping[str, Finding],
        challenges: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, Any]:
        if not self.deterministic_judge:
            return await super()._judge(
                image=image,
                context=context,
                role_summaries=role_summaries,
                candidates=candidates,
                challenges=challenges,
            )
        accepted = [
            candidate_id
            for candidate_id, finding in candidates.items()
            if (
                finding.verification is VerificationStatus.SUPPORTED
                and finding.metadata.get("grounding_status") == "valid"
                and candidate_is_core(
                    {
                        "defect_present": finding.metadata.get(
                            "review_defect_present"
                        ),
                        "materiality": finding.metadata.get(
                            "review_materiality"
                        ),
                        "evidence_basis": finding.metadata.get(
                            "review_evidence_basis"
                        ),
                        "confidence": finding.confidence,
                    }
                )
            )
        ]
        contradictory_empty = not candidates and any(
            str(summary.get("verdict", "uncertain"))
            in {"artifact_present", "uncertain"}
            for summary in role_summaries
        )
        return {
            "verdict": (
                "artifact_present"
                if accepted
                else "uncertain"
                if contradictory_empty
                else "no_artifact"
            ),
            "confidence": (
                max((candidates[item].confidence for item in accepted), default=0.9)
                if accepted
                else 0.5
                if contradictory_empty
                else 0.9
            ),
            "accepted_candidate_ids": accepted,
            "uncertain_candidate_ids": [],
            "rationale": (
                "候选已经通过结构化 Skeptic 与代码核心门槛，按确定性规则裁决。"
            ),
            "v42_deterministic_judge": True,
        }

    async def _request_challenge_batch(
        self,
        image: ImageAsset,
        context: CriticContext,
        batch: Sequence[tuple[str, Finding]],
    ) -> dict[str, list[Mapping[str, Any]]]:
        if self.review_original_max_side > 0:
            if self._review_original_task is None:
                self._review_original_task = asyncio.create_task(
                    overview_image(
                        image,
                        max_side=self.review_original_max_side,
                    )
                )
            overview_task = self._review_original_task
        else:
            overview_task = asyncio.create_task(asyncio.sleep(0, result=image))
        crops = await asyncio.gather(
            *(
                self.image_tool.crop(
                    image,
                    Region(
                        id=f"{candidate_id}:verify",
                        bbox=finding.bbox,
                        label=finding.artifact_type,
                    ),
                    padding=0.08,
                )
                for candidate_id, finding in batch
            )
        )
        packet = [
            {
                "candidate_id": candidate_id,
                "crop_image_index": index + 1,
                "finding": finding.as_dict(),
            }
            for index, (candidate_id, finding) in enumerate(batch)
        ]
        response = await self.skeptic_model.complete(
            ModelRequest(
                stage="consensus.challenge_batch",
                system_prompt=(
                    "你是严格的多候选视觉问题与编辑区域复核员。图像0是整图，"
                    "其余图像按清单对应候选裁剪。每个候选必须独立复核：先寻找"
                    "遮挡、透视、模糊、风格化、反射或重叠等正常解释，再判断问题"
                    "是否为直接可见的 major/critical 核心错误，并检查 bbox 是否可"
                    "直接交给编辑模型。不得使用外部世界知识。所有自然语言使用简体中文。"
                ),
                user_prompt=(
                    "候选与裁剪编号：\n"
                    + json.dumps(packet, ensure_ascii=False)
                    + "\n必须为每个 candidate_id 恰好返回一条 review，不得遗漏、"
                    "重复或新增 ID。有效 bbox 应覆盖完整编辑目标和少量上下文，禁止"
                    "整图框、无目标大框和大框套多个独立小框。原框不合适时返回"
                    "needs_revision 及 revised_bbox_norm。轻微、正常、合理或仅属"
                    "风格化时必须反驳。"
                ),
                images=(
                    await overview_task,
                    *crops,
                    *context.reference_images,
                ),
                response_schema=_V42_BATCH_CHALLENGE_SCHEMA,
                metadata={
                    "workflow": "adversarial_consensus",
                    "coverage_mode": "harness_v42",
                    "candidate_count": len(batch),
                    "candidate_ids": [candidate_id for candidate_id, _ in batch],
                    "review_original_max_side": self.review_original_max_side,
                },
            )
        )
        parsed = parse_response(response)
        reviews = parsed.get("reviews", [])
        output: dict[str, list[Mapping[str, Any]]] = {}
        if isinstance(reviews, list):
            for raw in reviews:
                if not isinstance(raw, Mapping):
                    continue
                output.setdefault(str(raw.get("candidate_id", "")), []).append(raw)
        return output
