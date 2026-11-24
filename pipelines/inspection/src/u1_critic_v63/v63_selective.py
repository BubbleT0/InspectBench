from __future__ import annotations

import copy
import json
from collections.abc import Mapping, Sequence
from typing import Any

from critic_workflows import CriticContext, ImageAsset, ModelRequest
from critic_workflows.utils import parse_response

from .adaptive_coverage import _INVENTORY_SCHEMA
from .v62_adaptive import V62AdaptiveContentCoverageAuditor


_V63_TRIAGE_REGION_SCHEMA = copy.deepcopy(
    _INVENTORY_SCHEMA["properties"]["regions"]["items"]
)
_V63_TRIAGE_REGION_SCHEMA["required"] = [
    *_V63_TRIAGE_REGION_SCHEMA["required"],
    "review_required",
    "triage_confidence",
    "triage_reason",
]
_V63_TRIAGE_REGION_SCHEMA["properties"] = {
    **dict(_V63_TRIAGE_REGION_SCHEMA["properties"]),
    "review_required": {"type": "boolean"},
    "triage_confidence": {
        "type": "number",
        "minimum": 0,
        "maximum": 1,
    },
    "triage_reason": {"type": "string"},
}
_V63_TRIAGE_INVENTORY_SCHEMA = copy.deepcopy(_INVENTORY_SCHEMA)
_V63_TRIAGE_INVENTORY_SCHEMA["properties"]["regions"]["items"] = (
    _V63_TRIAGE_REGION_SCHEMA
)
_V63_COMPACT_TRIAGE_INVENTORY_SCHEMA = copy.deepcopy(
    _V63_TRIAGE_INVENTORY_SCHEMA
)
_V63_COMPACT_TRIAGE_REGION_SCHEMA = (
    _V63_COMPACT_TRIAGE_INVENTORY_SCHEMA["properties"]["regions"]["items"]
)
_V63_COMPACT_TRIAGE_REGION_SCHEMA["required"] = [
    name
    for name in _V63_COMPACT_TRIAGE_REGION_SCHEMA["required"]
    if name not in {"description", "triage_reason"}
]
_V63_COMPACT_TRIAGE_REGION_SCHEMA["properties"].pop("description", None)
_V63_COMPACT_TRIAGE_REGION_SCHEMA["properties"].pop("triage_reason", None)


class V63SelectiveCoverageAuditor(V62AdaptiveContentCoverageAuditor):
    """V6.3 coverage that locally reviews only suspicious/uncertain regions."""

    coverage_mode_name = "harness_v63_selective"
    inventory_response_schema = _V63_TRIAGE_INVENTORY_SCHEMA
    minimum_clear_negative_confidence = 0.8

    def __init__(
        self,
        *args: Any,
        entity_safe_review: bool = False,
        dense_path_latency: bool = False,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.entity_safe_review = bool(entity_safe_review)
        self.dense_path_latency = bool(dense_path_latency)
        self._v63_triage_by_kind: dict[str, dict[str, dict[str, Any]]] = {}
        self.v63_triage_metrics: dict[str, int] = {
            "total_regions": 0,
            "reviewed_regions": 0,
            "skipped_regions": 0,
            "forced_entity_regions": 0,
        }

    def _inventory_schema_for_request(self) -> Mapping[str, Any]:
        return (
            _V63_COMPACT_TRIAGE_INVENTORY_SCHEMA
            if self.dense_path_latency
            else self.inventory_response_schema
        )

    def _hydrate_compact_inventory_region(
        self,
        raw: Mapping[str, Any],
        kind: str,
    ) -> dict[str, Any]:
        region = copy.deepcopy(dict(raw))
        if not str(region.get("description", "")).strip():
            semantic_group = str(
                region.get("semantic_group_id")
                or region.get("region_id")
                or "未命名"
            ).strip()
            unit_kind = str(
                region.get("semantic_unit_kind") or "other"
            ).strip()
            label = "文字" if kind == "text" else "实体"
            region["description"] = (
                f"{label}语义单元 {semantic_group}（{unit_kind}）"
            )
        if not str(region.get("triage_reason", "")).strip():
            region["triage_reason"] = (
                "模型要求对该区域执行局部复核。"
                if region.get("review_required") is True
                else "模型高置信判定该区域无需局部复核。"
            )
        return region

    async def audit(
        self,
        image: ImageAsset,
        context: CriticContext,
    ) -> tuple[list[Any], dict[str, Any]]:
        self._v63_triage_by_kind = {}
        self.v63_triage_metrics = {
            "total_regions": 0,
            "reviewed_regions": 0,
            "skipped_regions": 0,
            "forced_entity_regions": 0,
        }
        findings, summary = await super().audit(image, context)
        summary = dict(summary)
        summary["v63_selective_review"] = True
        summary["v63_entity_safe_review"] = self.entity_safe_review
        summary["v63_triage_metrics"] = copy.deepcopy(
            self.v63_triage_metrics
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
        text_mode = kind == "text"
        system_prompt = (
            "你是高召回内容清单与局部复核分流员，只依据图片像素建立完整区域"
            "清单，同时判断每个区域是否需要进一步放大复核。不得读取文件名、"
            "目录标签、参考答案或其他输入元数据。所有 bbox 使用原图整数 "
            "[0,1000] 坐标，所有自然语言使用简体中文。"
        )
        inventory_task = (
            "枚举所有被设计为文字内容的完整语义编辑单元；乱码、伪文字和当前"
            "不可读文字也必须列出。不要按行、词或字符拆碎，空间独立标签必须"
            "分开。"
            if text_mode
            else
            "枚举所有主要独立实体和语义完整的非文字版面区块，包括人物、动物、"
            "物体、建筑、图表和插图；正常实体也必须列出，不得用整图占位。"
        )
        triage_task = (
            " 对每个区域同时设置 review_required：只要看到任何疑似 major/critical "
            "文字错误或 Artifact，或区域过小、复杂、遮挡而无法从整图确信正常，"
            "必须为 true；只有直接可见证据足以高置信确认没有明显错误时才为 false。"
            "triage_confidence 表示对该分流决定的置信度。"
            + (
                "无需输出逐区域描述或分流理由，程序会补充稳定说明。"
                if self.dense_path_latency
                else "triage_reason 简述直接可见依据。"
            )
            + "不得因为没有外部正确文案而跳过疑似乱码或字形扭曲。"
        )
        empty_task = (
            " 若完整检查后确认不存在该类内容，返回 regions=[] 且 "
            "confirmed_empty=true；存在任何该类内容时为 false。"
        )
        zone_task = ""
        if self._semantic_zones:
            zone_task = (
                "\n以下语义 zone 必须逐区覆盖并填写 zone_id：\n"
                + json.dumps(self._semantic_zones, ensure_ascii=False)
            )
        feedback = ""
        if validation_errors:
            feedback = (
                "\n前一轮未通过程序校验："
                + "；".join(validation_errors)
                + "。请重新提交完整清单和合法 bbox。"
            )
        response = await self.model.complete(
            ModelRequest(
                stage=stage,
                system_prompt=system_prompt,
                user_prompt=(
                    inventory_task
                    + triage_task
                    + empty_task
                    + zone_task
                    + feedback
                ),
                images=(image,),
                response_schema=self._inventory_schema_for_request(),
                metadata={
                    "workflow": "adversarial_consensus",
                    "coverage_mode": self.coverage_mode_name,
                    "inventory_kind": kind,
                    "harness_semantic_attempt": attempt,
                    "v63_selective_review": True,
                    "v63_compact_inventory": self.dense_path_latency,
                },
            )
        )
        parsed = dict(parse_response(response))
        if self.dense_path_latency:
            raw_regions = parsed.get("regions", [])
            if isinstance(raw_regions, list):
                parsed["regions"] = [
                    self._hydrate_compact_inventory_region(raw, kind)
                    for raw in raw_regions
                    if isinstance(raw, Mapping)
                ]
        triage = self._v63_triage_by_kind.setdefault(kind, {})
        raw_regions = parsed.get("regions", [])
        if isinstance(raw_regions, list):
            for raw in raw_regions:
                if not isinstance(raw, Mapping):
                    continue
                region_id = str(raw.get("region_id", "")).strip()
                if not region_id:
                    continue
                try:
                    confidence = float(raw.get("triage_confidence", 0.0))
                except (TypeError, ValueError):
                    confidence = 0.0
                triage[region_id] = {
                    "review_required": raw.get("review_required") is True,
                    "confidence": confidence,
                    "reason": str(raw.get("triage_reason", "")).strip(),
                }
        return parsed

    async def _inventory_impl(
        self,
        image: ImageAsset,
        kind: str,
    ) -> list[dict[str, Any]]:
        regions = await super()._inventory_impl(image, kind)
        triage = self._v63_triage_by_kind.get(kind, {})
        for region in regions:
            region["v63_inventory_kind"] = kind
            decision = triage.get(str(region.get("region_id", "")))
            if decision is None:
                region["v63_review_required"] = True
                region["v63_triage_confidence"] = 0.0
                region["v63_triage_reason"] = "缺少可靠分流结果，保守进入局部复核。"
                continue
            confidence = float(decision["confidence"])
            region["v63_review_required"] = bool(
                decision["review_required"]
                or confidence < self.minimum_clear_negative_confidence
            )
            region["v63_triage_confidence"] = confidence
            region["v63_triage_reason"] = str(decision["reason"])
        return regions

    def _requires_local_review(
        self,
        region: Mapping[str, Any],
    ) -> tuple[bool, bool]:
        triage_required = bool(
            region.get("v63_review_required") is not False
            or float(region.get("v63_triage_confidence", 0.0))
            < self.minimum_clear_negative_confidence
        )
        force_entity = bool(
            self.entity_safe_review
            and not triage_required
            and region.get("v63_inventory_kind") == "entity"
            and region.get("content_kind") == "entity"
        )
        return triage_required or force_entity, force_entity

    def _is_entity_safe_region(self, content: Mapping[str, Any]) -> bool:
        return bool(
            self.entity_safe_review
            and content.get("v63_inventory_kind") == "entity"
            and content.get("content_kind") == "entity"
        )

    def _include_inventory_description(
        self,
        content: Mapping[str, Any],
        inspection_pass: str,
    ) -> bool:
        if self._is_entity_safe_region(content):
            return False
        return super()._include_inventory_description(
            content,
            inspection_pass,
        )

    def _artifact_inspection_rule(
        self,
        contents: Sequence[Mapping[str, Any]],
    ) -> str:
        if not any(self._is_entity_safe_region(item) for item in contents):
            return super()._artifact_inspection_rule(contents)
        return (
            "这是独立的局部 Artifact 反漏判复核，不采信内容清单或其他阶段的"
            "正常结论。对每个实体必须先逐项核验再决定阴性：一、清点手指、"
            "肢体、器官、零件和重复纹理等部件数量及形态；二、沿轮廓追踪部件"
            "之间的连接、附着、连续性、穿插和遮挡顺序；三、检查持物、抓握、"
            "接触、支撑、受力和人物交互关系；四、检查边界断裂、融合、溶解、"
            "材质失败和局部渲染残留。人物、动物、手、脸、肢体与复杂交互必须"
            "放大逐一核对。只有上述检查都能用直接像素证据确认正常时才能返回"
            "无问题；不得因为整体看起来自然、清单描述声称正常或问题位于小图"
            "中而跳过。文字由独立文字通道检查，本通道不报告文字问题；正常"
            "透视、景深、风格化和轻微瑕疵仍不报告。"
        )

    def _inspection_batch_size(
        self,
        channel: str,
        region_count: int,
        *,
        configured: int,
    ) -> int:
        if (
            self.dense_path_latency
            and channel in {"text", "text_crosscheck", "entity"}
            and region_count >= 12
        ):
            return max(configured, 4)
        return configured

    async def _inspect_regions_core(
        self,
        image: ImageAsset,
        regions: Sequence[Mapping[str, Any]],
    ) -> list[tuple[Any, int, bool]]:
        routing = [self._requires_local_review(region) for region in regions]
        selected_indexes = [
            index for index, (required, _) in enumerate(routing) if required
        ]
        selected = [regions[index] for index in selected_indexes]
        reviewed = (
            await super()._inspect_regions_core(image, selected)
            if selected
            else []
        )
        output: list[tuple[Any, int, bool]] = [
            (None, 0, False) for _ in regions
        ]
        for index, value in zip(selected_indexes, reviewed):
            output[index] = value
        self.v63_triage_metrics["total_regions"] += len(regions)
        self.v63_triage_metrics["reviewed_regions"] += len(selected)
        self.v63_triage_metrics["skipped_regions"] += (
            len(regions) - len(selected)
        )
        self.v63_triage_metrics["forced_entity_regions"] += sum(
            forced for _, forced in routing
        )
        return output
