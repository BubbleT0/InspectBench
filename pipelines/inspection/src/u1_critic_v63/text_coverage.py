from __future__ import annotations

import asyncio
import copy
import math
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


_BBOX_SCHEMA: Mapping[str, Any] = {
    "type": "array",
    "items": {"type": "integer", "minimum": 0, "maximum": 1000},
    "minItems": 4,
    "maxItems": 4,
}

_TEXT_BLOCK_SCHEMA: Mapping[str, Any] = {
    "type": "object",
    "required": [
        "region_id",
        "bbox_norm",
        "text_role",
        "visible_text",
        "intended_readable",
        "status",
        "suspicion_kind",
        "suspicion_reason",
        "confidence",
    ],
    "properties": {
        "region_id": {"type": "string"},
        "bbox_norm": _BBOX_SCHEMA,
        "text_role": {
            "type": "string",
            "enum": ["title", "label", "body", "caption", "other"],
        },
        "visible_text": {"type": "string"},
        "intended_readable": {"type": "boolean"},
        "status": {
            "type": "string",
            "enum": ["clear", "suspicious"],
        },
        "suspicion_kind": {
            "type": "string",
            "enum": [
                "visual_integrity",
                "direct_visible_format",
                "none",
            ],
        },
        "suspicion_reason": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
}

_TEXT_COVERAGE_SCHEMA: Mapping[str, Any] = {
    "type": "object",
    "required": ["blocks", "coverage_rationale"],
    "properties": {
        "blocks": {
            "type": "array",
            "items": _TEXT_BLOCK_SCHEMA,
        },
        "coverage_rationale": {"type": "string"},
    },
}


def coverage_tiles(
    grid: int = 3,
    overlap: float = 0.12,
) -> list[Region]:
    if grid <= 0:
        raise ValueError("grid must be positive")
    if overlap < 0.0 or overlap >= 1.0:
        raise ValueError("overlap must be in [0, 1)")
    extent = 1000.0 / (grid - (grid - 1) * overlap)
    step = extent * (1.0 - overlap)
    output: list[Region] = []
    for row in range(grid):
        for column in range(grid):
            x1 = round(min(column * step, 1000.0 - extent))
            y1 = round(min(row * step, 1000.0 - extent))
            x2 = 1000 if column == grid - 1 else round(x1 + extent)
            y2 = 1000 if row == grid - 1 else round(y1 + extent)
            output.append(
                Region(
                    id=f"text-tile-r{row}-c{column}",
                    bbox=BBox(x1, y1, x2, y2),
                    label="text_coverage_tile",
                    metadata={"row": row, "column": column},
                )
            )
    return output


def map_tile_bbox(tile: BBox, local: BBox) -> BBox:
    width = tile.x2 - tile.x1
    height = tile.y2 - tile.y1
    return BBox(
        math.floor(tile.x1 + local.x1 * width / 1000),
        math.floor(tile.y1 + local.y1 * height / 1000),
        math.ceil(tile.x1 + local.x2 * width / 1000),
        math.ceil(tile.y1 + local.y2 * height / 1000),
    )


def _area(box: BBox) -> float:
    return (box.x2 - box.x1) * (box.y2 - box.y1)


def _intersection(left: BBox, right: BBox) -> float:
    width = max(0.0, min(left.x2, right.x2) - max(left.x1, right.x1))
    height = max(0.0, min(left.y2, right.y2) - max(left.y1, right.y1))
    return width * height


def _same_text_entity(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> bool:
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


def _merge_blocks(
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
    sources = list(output.get("coverage_sources", []))
    sources.extend(right.get("coverage_sources", []))
    output["coverage_sources"] = list(dict.fromkeys(str(x) for x in sources))
    if right.get("status") == "suspicious":
        for key in (
            "status",
            "suspicion_kind",
            "suspicion_reason",
            "confidence",
        ):
            if key in right:
                output[key] = copy.deepcopy(right[key])
    elif float(right.get("confidence", 0.0)) > float(
        output.get("confidence", 0.0)
    ):
        output["confidence"] = right["confidence"]
    return output


def deduplicate_text_blocks(
    blocks: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for raw in blocks:
        try:
            BBox.from_value(raw["bbox_norm"])
        except (KeyError, TypeError, ValueError):
            continue
        for index, current in enumerate(output):
            if _same_text_entity(current, raw):
                output[index] = _merge_blocks(current, raw)
                break
        else:
            output.append(copy.deepcopy(dict(raw)))
    return output


def suppress_aggregate_text_blocks(
    blocks: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for index, candidate in enumerate(blocks):
        candidate_box = BBox.from_value(candidate["bbox_norm"])
        contained: list[Mapping[str, Any]] = []
        for other_index, other in enumerate(blocks):
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
        has_separate_children = any(
            not _same_text_entity(left, right)
            for left_index, left in enumerate(contained)
            for right in contained[left_index + 1:]
        )
        if len(contained) >= 2 and has_separate_children:
            continue
        output.append(copy.deepcopy(dict(candidate)))
    return output


class TextCoverageAuditor:
    def __init__(self, model, image_tool):
        self.model = model
        self.image_tool = image_tool

    async def audit(
        self,
        image: ImageAsset,
        context: CriticContext,
    ) -> tuple[list[Finding], dict[str, Any]]:
        full = await self._inspect(
            image,
            context=context,
            stage="text.coverage.full",
            source_id="full",
            metadata={"coverage_scope": "full"},
        )
        tiles = coverage_tiles()

        async def inspect_tile(region: Region):
            crop = await self.image_tool.crop(image, region)
            parsed = await self._inspect(
                crop,
                context=context,
                stage="text.coverage.tile",
                source_id=region.id,
                metadata={
                    "coverage_scope": "tile",
                    "tile_row": region.metadata["row"],
                    "tile_column": region.metadata["column"],
                    "tile_bbox_norm": region.bbox.as_list(),
                },
            )
            for block in parsed:
                local = BBox.from_value(block["bbox_norm"])
                block["bbox_norm"] = map_tile_bbox(
                    region.bbox,
                    local,
                ).as_list()
            return parsed

        tile_groups = await asyncio.gather(
            *(inspect_tile(region) for region in tiles)
        )
        all_blocks = [*full]
        for group in tile_groups:
            all_blocks.extend(group)
        blocks = suppress_aggregate_text_blocks(
            deduplicate_text_blocks(all_blocks)
        )
        suspicious = [
            block
            for block in blocks
            if block.get("intended_readable") is True
            and block.get("status") == "suspicious"
            and block.get("suspicion_kind") != "none"
        ]
        findings = [
            self._finding(image, block, index)
            for index, block in enumerate(suspicious)
        ]
        return findings, {
            "verdict": (
                "artifact_present" if findings else "no_artifact"
            ),
            "confidence": max(
                (finding.confidence for finding in findings),
                default=0.9,
            ),
            "covered_text_block_count": len(blocks),
            "suspicious_text_block_count": len(suspicious),
            "coverage_blocks": blocks,
        }

    async def _inspect(
        self,
        image: ImageAsset,
        *,
        context: CriticContext,
        stage: str,
        source_id: str,
        metadata: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        scope = (
            "整张图片" if stage == "text.coverage.full" else "当前分块"
        )
        response = await self.model.complete(
            ModelRequest(
                stage=stage,
                system_prompt=(
                    "你是文字覆盖审查员，只依据直接可见像素工作。你的任务不是评价"
                    "美观，而是枚举所有意图可读的独立文字实体，并标记明显文字错误。"
                    "不得使用当前日期、历史事实、排名、价格、身份或其他外部世界知识。"
                    "书法、字体和排版只要清晰可辨且可合理解释为风格设计，就视为正常。"
                    "所有自然语言字段使用简体中文。"
                ),
                user_prompt=(
                    f"检查范围：{scope}。\n"
                    "逐一列出其中所有空间独立、意图可读的标题、标签、正文、说明和"
                    "其他文字块。空间分离的标签必须分别输出 bbox，不能因为属于同一"
                    "场景而合成大框。清晰正常的文字也要列出并标记 clear，以证明已覆盖。"
                    "只有乱码、伪文字、缺字、多字、明显错字、明显笔画断裂/粘连/扭曲，"
                    "或无需外部知识即可确认的格式错误和图内矛盾，才标记 suspicious。"
                    "轻微锯齿、软边、压缩、光晕和正常风格化不得标记 suspicious。"
                    "bbox_norm 使用当前输入图像的整数 [0,1000] 归一化坐标。"
                ),
                images=(image, *context.reference_images),
                response_schema=_TEXT_COVERAGE_SCHEMA,
                metadata={
                    "workflow": "adversarial_consensus",
                    **dict(metadata),
                },
            )
        )
        parsed = parse_response(response)
        raw_blocks = parsed.get("blocks", [])
        output: list[dict[str, Any]] = []
        if not isinstance(raw_blocks, list):
            return output
        for index, raw in enumerate(raw_blocks):
            if not isinstance(raw, Mapping):
                continue
            try:
                bbox = BBox.from_value(raw.get("bbox_norm"))
                confidence = float(raw.get("confidence", 0.0))
            except (TypeError, ValueError):
                continue
            block = copy.deepcopy(dict(raw))
            block["bbox_norm"] = bbox.as_list()
            block["confidence"] = max(0.0, min(1.0, confidence))
            block["coverage_sources"] = [
                f"{source_id}:b{index:03d}"
            ]
            output.append(block)
        return output

    @staticmethod
    def _finding(
        image: ImageAsset,
        block: Mapping[str, Any],
        index: int,
    ) -> Finding:
        visible_text = str(block.get("visible_text", "")).strip()
        target = visible_text or f"文字实体 {index + 1}"
        suspicion_kind = str(block.get("suspicion_kind", ""))
        artifact_type = (
            "text_content_fidelity"
            if suspicion_kind == "direct_visible_format"
            else "text_visual_integrity"
        )
        reason = str(block.get("suspicion_reason", "")).strip()
        return Finding(
            region_id=f"text-{index:03d}",
            bbox=BBox.from_value(block["bbox_norm"]),
            artifact_type=artifact_type,
            observation=f"独立文字块“{target}”存在可见异常。",
            abnormality=reason,
            confidence=float(block.get("confidence", 0.0)),
            evidence_ids=[
                image.id,
                *[
                    str(item)
                    for item in block.get("coverage_sources", [])
                ],
            ],
            source="role:text_coverage",
            metadata={
                "edit_target": f"独立文字块“{target}”",
                "edit_region_rationale": (
                    "覆盖完整文字实体和少量周边背景，供编辑模型重绘字形，"
                    "不包含空间分离的其他文字块。"
                ),
                "original_bbox_norm": list(block["bbox_norm"]),
                "grounding_status": "pending",
                "defect_present": True,
                "materiality": "major",
                "evidence_basis": "visible_only",
                "text_entity_atomic": True,
                "text_role": block.get("text_role", "other"),
                "visible_text": visible_text,
                "coverage_sources": list(
                    block.get("coverage_sources", [])
                ),
            },
        )
