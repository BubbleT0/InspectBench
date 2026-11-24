from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .translate_existing import localize_value


WORKFLOW_ORDER = (
    "specialist_cascade",
    "evidence_seeking_agent",
    "adversarial_consensus",
)

WORKFLOW_INTROS = {
    "specialist_cascade": {
        "title": "专员级联管线",
        "code": "Specialist Cascade",
        "bias": "高召回候选 + 独立复核",
        "description": "六个窄领域专员并行检查整图，候选问题按边界框裁剪后再由独立复核器确认。",
        "stages": [
            ["specialist.regions", "准备候选区域"],
            ["specialist.detect", "六类专员并行提出问题"],
            ["specialist.proposals", "汇总达到阈值的候选"],
            ["specialist.verify", "逐条裁剪并独立验证"],
            ["derived verdict", "按“证据支持”或“不确定”聚合"],
        ],
    },
    "evidence_seeking_agent": {
        "title": "主动取证管线",
        "code": "Evidence Seeking Agent",
        "bias": "主动裁剪 + 动态证据",
        "description": "规划器判断证据是否充分；必要时申请局部裁剪，循环观察后给出原子问题项并复核。",
        "stages": [
            ["evidence.plan", "决定继续检查、结束取证或拒绝判断"],
            ["evidence.action", "执行局部裁剪并积累证据"],
            ["evidence.force_finish", "达到次数上限时强制收束"],
            ["evidence.verify", "逐条验证最终问题项"],
            ["derived verdict", "根据验证状态聚合"],
        ],
    },
    "adversarial_consensus": {
        "title": "Adversarial Consensus",
        "code": "Adversarial Consensus",
        "bias": "核心 Artifact + 文字全覆盖 + 真实裁剪",
        "description": "三个 Artifact 关注面检查明显错误；文字通道通过整图清单与重叠分块覆盖每个文字实体，Skeptic 使用真实裁剪复核，最终只保留适合局部重绘的核心问题。",
        "stages": [
            ["consensus.propose", "三个 Artifact 角色提出核心问题"],
            ["text.coverage.full", "建立整图文字清单"],
            ["text.coverage.tile", "重叠分块检查独立文字实体"],
            ["consensus.proposals", "分配稳定候选编号"],
            ["consensus.challenge", "使用真实裁剪逐条复核"],
            ["consensus.challenges", "汇总核心问题复核结果"],
            ["consensus.judge", "受约束裁决者给出最终结论"],
            ["consensus.consolidate", "整编为独立局部重绘区域"],
        ],
    },
}

ADAPTIVE_CONSENSUS_INTRO = {
    "title": "Adversarial Consensus V3",
    "code": "Adversarial Consensus V3",
    "bias": "核心 Artifact + 自适应语义区块 + 裁剪边界复核",
    "description": (
        "先建立完整标题、卡片、榜单行、独立标签和主要实体清单，"
        "再对语义区块无损放大检查；裁剪不完整时自动扩框重试，"
        "最终只保留适合局部重绘的核心问题。"
    ),
    "stages": [
        ["consensus.propose", "三个 Artifact 角色提出核心问题"],
        ["content.inventory.text", "建立文字语义编辑单元清单"],
        ["content.inventory.entity", "建立主要实体清单"],
        ["content.coverage.audit", "终审内容覆盖并补充漏项"],
        ["content.inspect", "无损放大并独立检查语义区块"],
        ["consensus.challenge", "使用真实裁剪逐条复核"],
        ["consensus.judge", "受约束裁决者给出最终结论"],
        ["consensus.consolidate", "整编为独立局部重绘区域"],
    ],
}

HARNESS_CONSENSUS_INTRO = {
    **ADAPTIVE_CONSENSUS_INTRO,
    "title": "Adversarial Consensus V4 Harness",
    "code": "Adversarial Consensus V4 Harness",
    "bias": "语义契约 + 定向重试 + 证据重路由 + 安全弃权",
    "description": (
        "在 V3 自适应内容覆盖外增加可靠性 harness：两次独立语义版面"
        "观察先盲审后对账，逐 zone 核销文字、实体和版面通道；对失败区"
        "定向恢复，独立复核文字负结论，并在覆盖不完整时保留已发现问题"
        "但把总结果标记为不确定。"
    ),
    "stages": [
        ["content.zones.primary", "建立主语义版面区清单"],
        ["content.coverage.blind_inventory", "独立盲审语义版面覆盖"],
        ["content.inventory.text", "逐区建立文字语义单元清单"],
        ["content.inventory.entity", "逐区建立实体与版面清单"],
        ["content.coverage.audit", "已有清单漏项复核"],
        ["content.inventory.zone_recovery", "定向恢复失败的区域通道"],
        ["content.inspect", "无损放大并独立检查语义区块"],
        ["consensus.challenge", "使用真实裁剪逐条复核"],
        ["consensus.judge", "受约束裁决者给出质量结论"],
        ["consensus.consolidate", "整编为独立局部重绘区域"],
    ],
}

STAGE_ZH = {
    "specialist.regions": "候选区域准备",
    "specialist.detect": "专员检测",
    "specialist.proposals": "候选问题汇总",
    "specialist.verify": "候选问题复核",
    "evidence.plan": "取证规划",
    "evidence.action": "取证动作",
    "evidence.limit": "证据数量上限",
    "evidence.invalid_action": "无效取证动作",
    "evidence.force_finish": "强制收束",
    "evidence.verify": "证据结论复核",
    "consensus.propose": "多角色提出候选",
    "text.coverage.full": "整图文字清单",
    "text.coverage.tile": "分块文字覆盖",
    "content.inventory.text": "文字语义单元清单",
    "content.inventory.text.harness_retry": "文字清单语义修复重试",
    "content.inventory.entity": "主要实体清单",
    "content.inventory.entity.harness_retry": "实体清单语义修复重试",
    "content.zones.primary": "主语义版面区清单",
    "content.zones.primary.harness_retry": "主版面区清单修复重试",
    "content.coverage.blind_inventory": "独立盲审版面覆盖",
    "content.coverage.blind_inventory.harness_retry": "盲审版面清单修复重试",
    "content.inventory.zone_recovery": "失败区域通道定向恢复",
    "content.coverage.audit": "内容覆盖终审",
    "content.coverage.audit.text": "文字覆盖独立终审",
    "content.coverage.audit.artifact": "非文字覆盖独立终审",
    "content.inspect": "语义区块独立检查",
    "content.inspect.multiscale": "多尺度语义区块独立检查",
    "consensus.proposals": "候选意见汇总",
    "consensus.challenge": "真实裁剪复核",
    "consensus.challenges": "质疑意见汇总",
    "consensus.judge": "最终裁决",
    "consensus.consolidate": "最终结果整编",
    "consensus.consolidate_repair": "整编结果修正",
    "consensus.consolidate_global_repair": "分层整编全图终审",
    "translation.batch": "批量中文翻译",
    "derived verdict": "程序聚合结论",
}

KEY_ZH = {
    "action": "动作",
    "reason": "原因",
    "present": "是否存在",
    "confidence": "置信度",
    "observation": "观察到的现象",
    "abnormality": "异常原因",
    "bbox_norm": "归一化坐标",
    "label": "区域标签",
    "verdict": "结论",
    "findings": "问题列表",
    "status": "验证状态",
    "counterargument": "反方论据",
    "missing_evidence": "缺失证据",
    "accepted_candidate_ids": "接受的候选 ID",
    "uncertain_candidate_ids": "不确定的候选 ID",
    "rationale": "裁决理由",
    "artifact_type": "问题分类",
    "region_id": "区域 ID",
    "evidence_ids": "证据 ID",
    "message": "过程说明",
    "edit_regions": "编辑区域",
    "edit_target": "编辑目标",
    "edit_region_rationale": "编辑区域设计",
    "source_finding_ids": "来源问题 ID",
    "issues": "问题列表",
    "grounding_status": "区域定位状态",
    "grounding_rationale": "区域定位说明",
    "discarded_finding_ids": "丢弃的问题 ID",
    "discard_reasons": "丢弃原因",
    "materiality": "问题重要性",
    "defect_present": "是否存在真实缺陷",
    "evidence_basis": "证据基础",
    "blocks": "文字块",
    "text_role": "文字用途",
    "visible_text": "可见文字",
    "intended_readable": "是否意图可读",
    "suspicion_kind": "可疑类型",
    "suspicion_reason": "可疑原因",
    "coverage_rationale": "覆盖说明",
    "semantic_group_id": "语义组 ID",
    "semantic_unit_kind": "语义单元类型",
    "designed_as_text": "是否被设计为文字内容",
    "boundary_complete": "清单边界是否完整",
    "crop_boundary_complete": "裁剪边界是否完整",
    "revised_content_bbox_norm": "修正后的完整语义单元坐标",
    "edit_bbox_full_norm": "原图编辑区域坐标",
    "original_crop_size": "原始裁剪尺寸",
    "review_crop_size": "检查裁剪尺寸",
    "zoom_scale": "无损放大倍数",
    "boundary_retry_count": "边界扩框重试次数",
}

VALUE_ZH = {
    "artifact_present": "发现质量问题",
    "no_artifact": "未发现质量问题",
    "uncertain": "不确定",
    "supported": "证据支持",
    "contradicted": "证据反驳",
    "proposed": "待复核",
    "inspect": "检查局部区域",
    "finish": "结束取证",
    "abstain": "证据不足，拒绝判断",
    "rendering_artifact": "局部渲染错误",
    "structural_artifact": "实体结构错误",
    "topology_artifact": "拓扑连接错误",
    "relational_physical_artifact": "关系与物理错误",
    "text_visual_integrity": "文字视觉完整性错误",
    "text_content_fidelity": "文字内容保真度错误",
    "ok": "成功",
    "error": "失败",
    "major": "明显问题",
    "critical": "严重问题",
    "minor": "轻微瑕疵",
    "none": "无问题",
    "visible_only": "仅直接可见证据",
    "external_knowledge": "外部世界知识",
    "aesthetic_preference": "审美偏好",
    "clear": "清晰正常",
    "suspicious": "存在明显可疑文字",
}


def adapt_report_bbox(
    value: Any,
    *,
    coordinate_space: str | None,
) -> list[int] | None:
    """Convert a valid report bbox to canonical [0,1000] without editing files."""

    if (
        not isinstance(value, list)
        or len(value) != 4
        or any(isinstance(item, bool) for item in value)
    ):
        return None
    try:
        numeric = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    if any(not math.isfinite(item) for item in numeric):
        return None
    if coordinate_space == "bbox_1000":
        if any(
            item < 0
            or item > 1000
            or not item.is_integer()
            for item in numeric
        ):
            return None
        output = [int(item) for item in numeric]
    elif all(0 <= item <= 1 for item in numeric):
        output = [round(item * 1000) for item in numeric]
    elif all(
        0 <= item <= 1000 and item.is_integer()
        for item in numeric
    ):
        output = [int(item) for item in numeric]
    else:
        return None
    if output[0] >= output[2] or output[1] >= output[3]:
        return None
    return output


def _adapt_report_bboxes(
    value: Any,
    *,
    coordinate_space: str | None,
) -> Any:
    if isinstance(value, dict):
        output: dict[str, Any] = {}
        for key, item in value.items():
            if (
                "bbox" in str(key).lower()
                and isinstance(item, list)
                and len(item) == 4
            ):
                adapted = adapt_report_bbox(
                    item,
                    coordinate_space=coordinate_space,
                )
                output[key] = adapted if adapted is not None else item
            else:
                output[key] = _adapt_report_bboxes(
                    item,
                    coordinate_space=coordinate_space,
                )
        return output
    if isinstance(value, list):
        return [
            _adapt_report_bboxes(
                item,
                coordinate_space=coordinate_space,
            )
            for item in value
        ]
    return value


def _display_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            KEY_ZH.get(str(key), str(key)): _display_value(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_display_value(item) for item in value]
    if isinstance(value, str):
        return VALUE_ZH.get(value, value)
    return value


def _reference_label_zh(value: str) -> str:
    replacements = {
        "D1_Artifact": "D1 非文字图像错误",
        "A1_RenderingArtifact": "A1 局部渲染错误",
        "A2_StructuralArtifact": "A2 实体结构错误",
        "A3_TopologyArtifact": "A3 拓扑连接错误",
        "A4_Relational&Physical Artifact": "A4 关系与物理错误",
        "D2_TextCorrectness": "D2 文字正确性",
        "A1_Text_Visual_Integrity": "A1 文字视觉完整性",
        "1.glyph_unrecognizable": "1 字形不可辨识",
        "2.glyph_malformed": "2 字形畸变",
        "3.character_fusion": "3 字符粘连",
        "4.character_overlap": "4 字符重叠",
        "5.spacing_error": "5 字间距错误",
        "6.alignment_error": "6 对齐错误",
    }
    return " / ".join(replacements.get(part, part) for part in value.split("/"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _latest_coverage_traces(
    values: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    latest: dict[tuple[Any, ...], dict[str, Any]] = {}
    for trace in values:
        stage = trace.get("stage")
        if stage not in {"text.coverage.full", "text.coverage.tile"}:
            continue
        if trace.get("status") != "ok":
            continue
        request = trace.get("request")
        metadata = (
            request.get("metadata", {})
            if isinstance(request, dict)
            else {}
        )
        if stage == "text.coverage.full":
            key = (stage,)
        else:
            row = metadata.get("tile_row")
            column = metadata.get("tile_column")
            key = (
                (stage, row, column)
                if row is not None and column is not None
                else (stage, trace.get("sequence"))
            )
        latest[key] = trace
    return list(latest.values())


def load_report_data(
    output_root: Path,
    *,
    include_unit_traces: bool = True,
) -> dict[str, Any]:
    manifest = json.loads((output_root / "manifest.json").read_text("utf-8"))
    run_config_path = output_root / "run_config.json"
    run_config = (
        json.loads(run_config_path.read_text("utf-8"))
        if run_config_path.exists()
        else {}
    )
    configured_workflows = (
        run_config.get("workflows")
    )
    coverage_mode = str(run_config.get("coverage_mode", "grid"))
    bbox_coordinate_space = run_config.get("bbox_coordinate_space")
    workflow_intros = dict(WORKFLOW_INTROS)
    if coverage_mode in {"adaptive", "harness", "harness_v41"}:
        workflow_intros["adversarial_consensus"] = (
            HARNESS_CONSENSUS_INTRO
            if coverage_mode in {"harness", "harness_v41"}
            else ADAPTIVE_CONSENSUS_INTRO
        )
    workflow_order = tuple(
        workflow
        for workflow in (
            configured_workflows
            if isinstance(configured_workflows, list)
            else WORKFLOW_ORDER
        )
        if workflow in WORKFLOW_ORDER
    ) or WORKFLOW_ORDER
    comparison_enabled = len(workflow_order) > 1
    translation_path = output_root / "translations.json"
    translations = (
        json.loads(translation_path.read_text("utf-8")).get("translations", {})
        if translation_path.exists()
        else {}
    )
    results = {
        (row["sample_id"], row["workflow"]): row
        for row in _adapt_report_bboxes(
            _jsonl(output_root / "results.jsonl"),
            coordinate_space=bbox_coordinate_space,
        )
        if row.get("status") == "ok"
    }
    consolidated_results = {} if coverage_mode == "harness_v41" else {
        (row["sample_id"], row["workflow"]): row
        for row in _adapt_report_bboxes(
            _jsonl(output_root / "consolidated_results.jsonl"),
            coordinate_space=bbox_coordinate_space,
        )
        if row.get("status") == "ok"
    }
    consolidation_errors: dict[
        tuple[str, str], list[dict[str, Any]]
    ] = defaultdict(list)
    for row in _jsonl(output_root / "consolidation_errors.jsonl"):
        consolidation_errors[(row["sample_id"], row["workflow"])].append(row)
    errors: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in _jsonl(output_root / "errors.jsonl"):
        errors[(row["sample_id"], row["workflow"])].append(row)
    traces: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in _adapt_report_bboxes(
        _jsonl(output_root / "traces.jsonl"),
        coordinate_space=bbox_coordinate_space,
    ):
        traces[(row["sample_id"], row["workflow"])].append(row)
    records = []
    verdict_counts: dict[str, Counter] = {
        workflow: Counter() for workflow in workflow_order
    }
    success_units = 0
    error_units = 0
    disagreement = 0
    coverage_status_counts: Counter = Counter()
    coverage_channel_counts: dict[str, Counter] = {
        "text": Counter(),
        "artifact": Counter(),
    }
    for image in manifest["images"]:
        units: dict[str, Any] = {}
        verdicts: list[str] = []
        for workflow in workflow_order:
            key = (image["image_id"], workflow)
            result_row = results.get(key)
            consolidated_row = consolidated_results.get(key)
            consolidated_original = (
                consolidated_row.get("consolidated_result")
                if consolidated_row
                else (
                    result_row.get("consolidated_result")
                    if result_row
                    else None
                )
            )
            error_history = errors.get(key, [])
            consolidation_history = consolidation_errors.get(key, [])
            current_consolidation_errors = (
                [] if consolidated_original else consolidation_history
            )
            error_row = None if result_row else (error_history[-1] if error_history else None)
            requires_consolidation = workflow == "adversarial_consensus"
            unit_succeeded = bool(
                result_row
                and (
                    not requires_consolidation
                    or consolidated_original
                )
            )
            unit_failed = bool(
                error_row
                or (
                    result_row
                    and requires_consolidation
                    and not consolidated_original
                )
            )
            if unit_succeeded:
                success_units += 1
                verdict = result_row["result"]["verdict"]
                verdict_counts[workflow][verdict] += 1
                verdicts.append(verdict)
                metadata = result_row["result"].get("metadata", {})
                if isinstance(metadata, dict):
                    coverage_status = str(
                        metadata.get("coverage_status", "")
                    )
                    if coverage_status:
                        coverage_status_counts[coverage_status] += 1
                    channels = metadata.get("coverage_channels", {})
                    if isinstance(channels, dict):
                        for channel in ("text", "artifact"):
                            channel_value = channels.get(channel, {})
                            if isinstance(channel_value, dict):
                                status = str(channel_value.get("status", ""))
                                if status:
                                    coverage_channel_counts[channel][status] += 1
            elif unit_failed:
                error_units += 1
                verdict_counts[workflow]["error"] += 1
            else:
                verdict_counts[workflow]["pending"] += 1
            result_original = result_row["result"] if result_row else None
            result_localized = (
                localize_value(result_original, translations)
                if result_original
                else None
            )
            consolidated_localized = (
                localize_value(consolidated_original, translations)
                if consolidated_original
                else None
            )
            localized_traces = []
            for trace in traces.get(key, []) if include_unit_traces else []:
                item = {
                    key: trace.get(key)
                    for key in (
                        "sequence",
                        "sample_id",
                        "workflow",
                        "stage",
                        "attempt",
                        "status",
                        "latency_seconds",
                        "error",
                        "usage",
                    )
                    if key in trace
                }
                item["stage_zh"] = STAGE_ZH.get(
                    str(item.get("stage", "")),
                    str(item.get("stage", "")),
                )
                localized_parsed = localize_value(
                    trace.get("parsed"),
                    translations,
                )
                item["parsed_display"] = _display_value(localized_parsed)
                request = trace.get("request")
                request_metadata = (
                    request.get("metadata", {})
                    if isinstance(request, dict)
                    else {}
                )
                item["request_metadata_display"] = _display_value(
                    request_metadata
                )
                localized_traces.append(item)
            units[workflow] = {
                "image_url": f"/images/{image['image_id']}",
                "result": result_localized,
                "result_original": result_original,
                "consolidated_result": consolidated_localized,
                "consolidated_result_original": consolidated_original,
                "consolidation_errors": current_consolidation_errors,
                "historical_consolidation_errors": (
                    consolidation_history if consolidated_original else []
                ),
                "error": error_row,
                "historical_errors": error_history,
                "traces": localized_traces,
            }
        if len(verdicts) > 1 and len(set(verdicts)) > 1:
            disagreement += 1
        records.append(
            {
                "sample_id": image["image_id"],
                "reference_label": image.get("reference_label", ""),
                "reference_label_zh": _reference_label_zh(
                    image.get("reference_label", "")
                ),
                "image_url": f"/images/{image['image_id']}",
                "disagreement": len(verdicts) > 1 and len(set(verdicts)) > 1,
                "units": units,
            }
        )
    stage_latency: dict[str, list[float]] = defaultdict(list)
    covered_text_blocks = 0
    suspicious_text_blocks = 0
    inventory_region_keys: set[tuple[str, str, str]] = set()
    inspected_region_keys: set[tuple[str, str, str]] = set()
    enlarged_crops = 0
    for values in traces.values():
        for trace in values:
            if trace.get("status") == "ok":
                stage_latency[trace["stage"]].append(
                    float(trace.get("latency_seconds", 0))
                )
            stage = trace.get("stage")
            parsed = trace.get("parsed")
            if trace.get("status") == "ok" and stage in {
                "content.inventory.text",
                "content.inventory.entity",
                "content.coverage.audit",
                "content.coverage.audit.text",
                "content.coverage.audit.artifact",
            }:
                field = (
                    "missing_regions"
                    if str(stage).startswith("content.coverage.audit")
                    else "regions"
                )
                regions = (
                    parsed.get(field, [])
                    if isinstance(parsed, dict)
                    else []
                )
                if isinstance(regions, list):
                    for index, region in enumerate(regions):
                        if not isinstance(region, dict):
                            continue
                        group_id = str(
                            region.get("semantic_group_id")
                            or region.get("region_id")
                            or index
                        )
                        inventory_region_keys.add(
                            (
                                str(trace.get("sample_id", "")),
                                str(trace.get("workflow", "")),
                                group_id,
                            )
                        )
            if trace.get("status") == "ok" and stage in {
                "content.inspect",
                "content.inspect.multiscale",
            }:
                request = trace.get("request")
                metadata = (
                    request.get("metadata", {})
                    if isinstance(request, dict)
                    else {}
                )
                group_id = str(
                    metadata.get("semantic_group_id")
                    or metadata.get("content_region_id")
                    or trace.get("sequence")
                )
                inspected_region_keys.add(
                    (
                        str(trace.get("sample_id", "")),
                        str(trace.get("workflow", "")),
                        group_id,
                    )
                )
                try:
                    if int(metadata.get("zoom_scale", 1)) > 1:
                        enlarged_crops += 1
                except (TypeError, ValueError):
                    pass
        for trace in _latest_coverage_traces(values):
            parsed = trace.get("parsed")
            blocks = (
                parsed.get("blocks", [])
                if isinstance(parsed, dict)
                else []
            )
            if isinstance(blocks, list):
                covered_text_blocks += len(blocks)
                suspicious_text_blocks += sum(
                    1
                    for block in blocks
                    if isinstance(block, dict)
                    and block.get("status") == "suspicious"
                )
    final_edit_regions = 0
    discarded_findings = 0
    for key in set(results) | set(consolidated_results):
        external = consolidated_results.get(key)
        result_row = results.get(key)
        consolidated = (
            external.get("consolidated_result")
            if external
            else (
                result_row.get("consolidated_result")
                if result_row
                else None
            )
        )
        if not isinstance(consolidated, dict):
            continue
        final_edit_regions += len(consolidated.get("edit_regions", []))
        discarded_findings += len(
            consolidated.get("discarded_finding_ids", [])
        )
    return {
        "coverage_mode": coverage_mode,
        "bbox_coordinate_space": "bbox_1000",
        "workflow_order": list(workflow_order),
        "workflow_intros": workflow_intros,
        "stage_zh": STAGE_ZH,
        "report_title": (
            (
                (
                    "Adversarial Consensus V4.1 Harness 评估"
                    if coverage_mode == "harness_v41"
                    else "Adversarial Consensus V4 Harness 评估"
                    if coverage_mode == "harness"
                    else "Adversarial Consensus V3 自适应内容评估"
                )
                if coverage_mode in {"adaptive", "harness", "harness_v41"}
                else "Adversarial Consensus 图像质量评估"
            )
            if workflow_order == ("adversarial_consensus",)
            else "三管线图像质量评估"
        ),
        "workflow_heading": (
            "Adversarial Consensus 工作流程"
            if workflow_order == ("adversarial_consensus",)
            else "三套管线如何工作"
        ),
        "comparison_enabled": comparison_enabled,
        "layout_class": "single" if len(workflow_order) == 1 else "",
        "records": records,
        "stats": {
            "images": len(records),
            "success_units": success_units,
            "error_units": error_units,
            "disagreement_images": disagreement,
            "covered_text_blocks": covered_text_blocks,
            "suspicious_text_blocks": suspicious_text_blocks,
            "inventory_regions": len(inventory_region_keys),
            "inspected_regions": len(inspected_region_keys),
            "enlarged_crops": enlarged_crops,
            "final_edit_regions": final_edit_regions,
            "discarded_findings": discarded_findings,
            "coverage_status_counts": dict(coverage_status_counts),
            "coverage_channel_counts": {
                key: dict(value)
                for key, value in coverage_channel_counts.items()
            },
            "verdict_counts": {
                key: dict(value) for key, value in verdict_counts.items()
            },
            "stage_latency": {
                key: {
                    "calls": len(values),
                    "average_seconds": round(sum(values) / len(values), 3),
                }
                for key, values in stage_latency.items()
                if values
            },
        },
    }


def _script_json(value: Any) -> str:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def build_report(
    output_root: Path,
    *,
    include_unit_traces: bool = True,
) -> Path:
    data = load_report_data(
        output_root,
        include_unit_traces=include_unit_traces,
    )
    output = output_root / "index.html"
    output.write_text(HTML.replace("__DATA__", _script_json(data)), encoding="utf-8")
    return output


HTML = r"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>三管线图像质量评估</title>
<style>
:root{--bg:#eef0ec;--paper:#fff;--ink:#17251d;--muted:#65736a;--line:#d9dfda;--blue:#2563a9;--orange:#c56625;--green:#16734a;--red:#b63732}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}
header{padding:30px clamp(18px,5vw,74px);background:linear-gradient(125deg,#183a2c,#243852);color:#fff}h1{margin:0;font:700 clamp(27px,4vw,44px)/1.1 Georgia,serif}.sub{margin-top:8px;color:#d8e4dc}
main{max-width:1800px;margin:auto;padding:18px}.stats,.flow-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px}.stat,.flow,.toolbar,.sample{background:var(--paper);border:1px solid var(--line);border-radius:14px}.stat{padding:13px}.stat b{display:block;font-size:25px}.stat span{color:var(--muted)}
h2{margin:27px 0 11px}.flow{padding:15px}.flow h3{margin:0}.bias{color:var(--muted);margin:2px 0 9px}.flow-desc{min-height:44px}.steps{display:flex;flex-wrap:wrap;align-items:center;gap:5px;margin-top:12px}.step{border:0;border-radius:7px;padding:6px 8px;background:#e9f0ec;cursor:pointer;text-align:left}.step:hover{background:#d7e9df}.step b{display:block;font:11px ui-monospace,monospace}.step small{color:var(--muted)}.arrow{color:#8a9790}
.legend{margin-top:10px;color:var(--muted)}.toolbar{display:flex;gap:8px;flex-wrap:wrap;padding:10px;position:sticky;top:7px;z-index:8;box-shadow:0 8px 24px #20382b18}select{border:1px solid var(--line);border-radius:8px;background:#fff;padding:8px}
.samples{display:grid;gap:16px;margin-top:14px}.sample{overflow:hidden}.sample-head{display:flex;justify-content:space-between;padding:12px 15px;border-bottom:1px solid var(--line)}.sid{font:700 13px ui-monospace,monospace}.ref{color:var(--muted)}.disagree{color:var(--orange);font-weight:700}
.overlay{position:absolute;pointer-events:none}.box{position:absolute;border:3px solid var(--c)}.box span{position:absolute;top:-15px;left:-15px;width:30px;height:30px;border-radius:50%;display:grid;place-items:center;background:var(--c);color:#fff;font-size:14px;font-weight:800;box-shadow:0 2px 8px #0007}
.unit-grid{display:grid;grid-template-columns:repeat(3,minmax(320px,1fr));gap:1px;background:var(--line)}.unit-grid.single{grid-template-columns:minmax(320px,1fr)}.unit{background:#fff;min-width:0}.unit-content{padding:16px}.unit h3{font-size:17px;margin:0 0 5px}.pipeline-code{font:11px ui-monospace,monospace;color:var(--muted)}.pipeline-visual{background:#111;height:clamp(560px,70vh,900px);position:relative}.pipeline-visual img{width:100%;height:clamp(560px,70vh,900px);object-fit:contain;display:block}.verdict{display:inline-block;border-radius:99px;padding:2px 7px;font-size:11px}.artifact_present{background:#ffd9d4;color:#962d29}.no_artifact{background:#dcf2e4;color:#12633d}.uncertain{background:#fff0c7;color:#765200}.error{background:#ffe0df;color:#8d211e}.coverage-warning{margin:10px 0;padding:10px 12px;border:1px solid #e0b74c;border-radius:8px;background:#fff4cd;color:#6e4e00;font-weight:700}
.region-list{display:grid;grid-template-columns:repeat(auto-fit,minmax(380px,1fr));gap:12px;margin-top:14px}.edit-region{border:1px solid var(--line);border-left:5px solid var(--c);border-radius:10px;padding:12px;background:#fafbf8}.edit-region h4{margin:0 0 6px;font-size:16px}.region-number{display:inline-grid;place-items:center;width:26px;height:26px;border-radius:50%;background:var(--c);color:#fff;margin-right:6px}.issue{margin-top:9px;padding:9px;border-radius:8px;background:#fff;border:1px solid var(--line)}.issue b{color:#244d38}.finding{border-left:4px solid var(--c);padding:7px 8px;margin:8px 0;background:#faf9f5}.finding .meta,.edit-region .meta{color:var(--muted);font-size:11px}.empty{color:var(--muted);font-style:italic}.timeline details{margin:6px 0;border:1px solid var(--line);border-radius:8px;padding:5px 7px}.timeline details.highlight{outline:3px solid #ffcf70}.timeline summary{cursor:pointer}.stage{font:11px ui-monospace,monospace}.raw{white-space:pre-wrap;overflow:auto;max-height:280px;font:11px ui-monospace,monospace;background:#f4f5f2;padding:8px}.hidden{display:none!important}
@media(max-width:1150px){.unit-grid{grid-template-columns:1fr}.pipeline-visual,.pipeline-visual img{height:clamp(520px,68vh,820px)}}@media(max-width:650px){main{padding:9px}.pipeline-visual,.pipeline-visual img{height:420px}.region-list{grid-template-columns:1fr}}
</style></head><body>
<header><h1 id=reportTitle>图像质量评估</h1><div class=sub>每套管线独立视图 · 中文过程轨迹 · 最终结论 · 前端坐标定位</div></header>
<main><section class=stats id=stats></section>
<h2 id=workflowHeading>工作流程</h2><section class=flow-grid id=flowGrid></section>
<div class=legend>图例：候选提出阶段负责高召回发现问题；证据包括原图和局部裁剪；复核阶段逐条检查问题；最终裁决只能选择已有候选；“不确定”表示证据不足。程序聚合结论不是模型调用。</div>
<h2>逐图结果与真实轨迹</h2>
<div class=toolbar><select id=labelFilter><option value="">全部参考分类</option></select><select id=verdictFilter><option value="">全部结论</option><option value=artifact_present>发现质量问题</option><option value=no_artifact>未发现质量问题</option><option value=uncertain>不确定</option><option value=error>失败</option></select><select id=disagreementFilter><option value="">全部样本</option><option value=yes>仅三管线分歧</option></select><select id=taxonomyFilter><option value="">全部问题分类</option></select><select id=statusFilter><option value="">全部状态</option><option value=ok>仅成功</option><option value=error>包含失败</option></select><button id=clearStage>清除阶段高亮</button></div>
<section class=samples id=samples></section></main>
<script>const DATA=__DATA__;
const COLORS={specialist_cascade:"#d55b40",evidence_seeking_agent:"#2873bd",adversarial_consensus:"#16845c"};
const NAMES={specialist_cascade:"专员级联管线",evidence_seeking_agent:"主动取证管线",adversarial_consensus:"Adversarial Consensus"};
const CODES={specialist_cascade:"Specialist Cascade",evidence_seeking_agent:"Evidence Seeking Agent",adversarial_consensus:"Adversarial Consensus"};
const VERDICT={artifact_present:"发现质量问题",no_artifact:"未发现质量问题",uncertain:"不确定",error:"失败",pending:"等待运行"};
const VERIFY={supported:"证据支持",contradicted:"证据反驳",uncertain:"不确定",proposed:"待复核"};
const TAXONOMY={rendering_artifact:"局部渲染错误",structural_artifact:"实体结构错误",topology_artifact:"拓扑连接错误",relational_physical_artifact:"关系与物理错误",text_visual_integrity:"文字视觉完整性错误",text_content_fidelity:"文字内容保真度错误"};
const esc=s=>String(s??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
reportTitle.textContent=DATA.report_title;
document.title=DATA.report_title;workflowHeading.textContent=DATA.workflow_heading;
if(!DATA.comparison_enabled)disagreementFilter.classList.add("hidden");
const statRows=[["图片",DATA.stats.images],["完整成功",DATA.stats.success_units],["链路失败",DATA.stats.error_units]];if(["adaptive","harness","harness_v41"].includes(DATA.coverage_mode))statRows.push(["语义内容单元",DATA.stats.inventory_regions],["已检查语义单元",DATA.stats.inspected_regions],["无损放大裁剪",DATA.stats.enlarged_crops],["最终编辑区域",DATA.stats.final_edit_regions]);if(DATA.coverage_mode==="harness_v41"){const cc=DATA.stats.coverage_channel_counts||{};statRows.push(["文字覆盖完整",cc.text?.ok||0],["Artifact覆盖完整",cc.artifact?.ok||0],["覆盖不完整",(DATA.stats.coverage_status_counts?.failed||0)+(DATA.stats.coverage_status_counts?.degraded||0)])}else if(DATA.stats.covered_text_blocks)statRows.push(["文字块覆盖数量",DATA.stats.covered_text_blocks],["可疑文字块",DATA.stats.suspicious_text_blocks],["最终编辑区域",DATA.stats.final_edit_regions]);if(DATA.comparison_enabled)statRows.push(["结论分歧图片",DATA.stats.disagreement_images]);stats.innerHTML=statRows.map(([k,v])=>`<div class=stat><b>${v}</b><span>${k}</span></div>`).join("");
flowGrid.innerHTML=DATA.workflow_order.map(w=>{const f=DATA.workflow_intros[w];return `<article class=flow><h3>${esc(f.title)}</h3><div class=bias>${esc(f.bias)}</div><div class=flow-desc>${esc(f.description)}</div><div class=steps>${f.stages.map((s,i)=>`${i?'<span class=arrow>→</span>':''}<button class=step data-stage="${esc(s[0])}"><b>${esc(s[1])}</b></button>`).join("")}</div></article>`}).join("");
const labelRows=[...new Map(DATA.records.map(r=>[r.reference_label,r.reference_label_zh])).entries()].sort((a,b)=>a[1].localeCompare(b[1]));labelRows.forEach(([raw,zh])=>labelFilter.insertAdjacentHTML("beforeend",`<option value="${esc(raw)}">${esc(zh)}</option>`));
function displayedIssues(u){return u.consolidated_result?.edit_regions?.flatMap(r=>r.issues||[])??(u.result?.findings||[])}
const tax=[...new Set(DATA.records.flatMap(r=>Object.values(r.units).flatMap(u=>displayedIssues(u).map(f=>f.artifact_type))))].sort();tax.forEach(x=>taxonomyFilter.insertAdjacentHTML("beforeend",`<option value="${esc(x)}">${esc(TAXONOMY[x]||x)}</option>`));
function allFindings(r){return Object.entries(r.units).flatMap(([w,u])=>displayedIssues(u).map(f=>({...f,workflow:w})))}
function fit(img){const st=img.parentElement,ov=st.querySelector(".overlay"),sw=st.clientWidth,sh=st.clientHeight,ir=img.naturalWidth/img.naturalHeight,sr=sw/sh;let w,h,l,t;if(ir>sr){w=sw;h=sw/ir;l=0;t=(sh-h)/2}else{h=sh;w=sh*ir;t=0;l=(sw-w)/2}Object.assign(ov.style,{left:l+"px",top:t+"px",width:w+"px",height:h+"px"})}
function bboxStyle(v){if(!Array.isArray(v)||v.length!==4||!v.every(Number.isFinite))return null;const [a,b,c,d]=v;if(a<0||b<0||c>1000||d>1000||a>=c||b>=d)return null;return `left:${a/10}%;top:${b/10}%;width:${(c-a)/10}%;height:${(d-b)/10}%`}
function boxes(w,u){const cr=u.consolidated_result;if(cr)return (cr.edit_regions||[]).map((r,i)=>{const style=bboxStyle(r.bbox_norm);return style?`<div class=box style="--c:${COLORS[w]};${style}"><span>${i+1}</span></div>`:""}).join("");if(!DATA.comparison_enabled&&!(u.consolidation_errors||[]).length)return "";return [...(u.result?.findings||[]),...(u.result?.uncertain_findings||[])].map((f,i)=>{const style=bboxStyle(f.bbox_norm);return style?`<div class=box style="--c:${COLORS[w]};${style}"><span>${i+1}</span></div>`:""}).join("")}
function trace(t){const params=Object.keys(t.request_metadata_display||{}).length?`<div><b>检查参数</b></div><pre class=raw>${esc(JSON.stringify(t.request_metadata_display,null,2))}</pre>`:"";const err=t.error?`<div><b>错误</b></div><pre class=raw>${esc(typeof t.error==="string"?t.error:JSON.stringify(t.error,null,2))}</pre>`:"";return `<details data-trace-stage="${esc(t.stage)}"><summary><b>${esc(t.stage_zh||t.stage)}</b> · ${t.status==="ok"?"成功":"失败"} · ${esc(t.latency_seconds)} 秒${t.attempt>1?` · 第 ${t.attempt} 次尝试`:""}</summary>${params}<div><b>中文解析结果</b></div><pre class=raw>${esc(JSON.stringify(t.parsed_display??{},null,2))}</pre>${err}</details>`}
function traceBundle(u){return (u.traces||[]).length?`<details class=trace-bundle><summary>查看模型调用轨迹（${u.traces.length} 次）</summary><div class=timeline>${u.traces.map(trace).join("")}</div></details>`:""}
function finding(f,w){const m=f.metadata||{};return `<div class=finding style="--c:${COLORS[w]}"><b>${esc(TAXONOMY[f.artifact_type]||f.artifact_type)}</b> · ${esc(VERIFY[f.verification]||f.verification)} · 置信度 ${esc(f.confidence)}<div><b>观察：</b>${esc(f.observation)}</div><div><b>异常：</b>${esc(f.abnormality)}</div>${m.edit_target?`<div><b>编辑目标：</b>${esc(m.edit_target)}</div>`:""}${m.edit_region_rationale?`<div><b>编辑区域设计：</b>${esc(m.edit_region_rationale)}</div>`:""}${m.grounding_rationale?`<div><b>区域复核：</b>${esc(m.grounding_rationale)}</div>`:""}<div class=meta>坐标 [${f.bbox_norm.join(", ")}] · 区域 ${esc(f.region_id)}${m.grounding_status?` · Grounding ${esc(m.grounding_status)}`:""}</div></div>`}
function issueCard(x){return `<div class=issue><b>${esc(TAXONOMY[x.artifact_type]||x.artifact_type)}</b> · ${esc(x.materiality||"")} · 置信度 ${esc(x.confidence)}<div><b>观察：</b>${esc(x.observation)}</div><div><b>异常：</b>${esc(x.abnormality)}</div></div>`}
function editRegion(r,w,i){return `<article class=edit-region style="--c:${COLORS[w]}"><h4><span class=region-number>${i+1}</span>编辑区域 ${i+1} · ${esc(r.edit_target)}</h4><div>${esc(VERIFY[r.verification]||r.verification)} · 综合置信度 ${esc(r.confidence)}</div><div><b>区域设计：</b>${esc(r.edit_region_rationale)}</div><div class=meta>坐标 [${r.bbox_norm.join(", ")}] · 来源 ${esc((r.source_finding_ids||[]).join(", "))}</div>${(r.issues||[]).map(issueCard).join("")}</article>`}
function history(u){const hs=[...(u.historical_errors||[]),...(u.historical_consolidation_errors||[])];return hs.length?`<details><summary>历史失败（已恢复）· ${hs.length} 次</summary><div>历史运行曾出现坐标、结构化输出或整编校验错误，之后已经重跑成功。</div><details><summary>查看英文错误原文</summary><pre class=raw>${esc(JSON.stringify(hs,null,2))}</pre></details></details>`:""}
function algorithmTrace(r){return (r.trace||[]).map(e=>`<div class=finding style="--c:#88958d"><b>${esc(DATA.stage_zh?.[e.stage]||e.stage)}</b><div>${esc(e.message)}</div></div>`).join("")}
function unit(w,u,image){const visual=`<div class=pipeline-visual><img loading=lazy decoding=async src="${esc(image)}" onload="fit(this)"><div class=overlay>${boxes(w,u)}</div></div>`;if(u.error)return `<section class=unit>${visual}<div class=unit-content><h3>${esc(NAMES[w])}</h3><span class="verdict error">失败</span><div>该管线运行失败，具体英文错误见折叠项。</div>${history(u)}${traceBundle(u)}</div></section>`;if(!u.result)return `<section class=unit>${visual}<div class=unit-content><h3>${esc(NAMES[w])}</h3><span class="verdict uncertain">等待运行</span></div></section>`;const cr=u.consolidated_result;if(!DATA.comparison_enabled&&!cr){const es=u.consolidation_errors||[];const raw=(u.result.findings||[]).map(f=>finding(f,w)).join("")||'<div class=empty>整编前没有通过复核的问题</div>';return `<section class=unit>${visual}<div class=unit-content><h3>${esc(NAMES[w])}</h3><span class="verdict ${es.length?"error":"uncertain"}">${es.length?"整编失败":"等待最终整编"}</span>${es.length?`<div class=coverage-warning>critic 已完成，但最终整编未通过；下面保留展示整编前问题和 bbox。</div>${raw}<details><summary>查看整编失败原因</summary><pre class=raw>${esc(JSON.stringify(es,null,2))}</pre></details>`:""}${traceBundle(u)}</div></section>`}const r=cr||u.result;const partial=Boolean(r.findings_non_exhaustive||u.result?.metadata?.findings_non_exhaustive||u.result?.metadata?.coverage_status&&u.result.metadata.coverage_status!=="ok");const coverageWarning=partial?`<div class=coverage-warning>覆盖不完整：以下已发现问题不是穷尽结果，请检查失败的语义区域和内容通道。</div>`:"";const details=cr?`<div class=region-list>${(cr.edit_regions||[]).map((x,i)=>editRegion(x,w,i)).join("")||'<div class=empty>没有需要重绘的编辑区域</div>'}</div><div><b>整编说明：</b>${esc(cr.rationale)}</div>`:`<div>${(r.findings||[]).map(f=>finding(f,w)).join("")||'<div class=empty>没有通过复核的问题</div>'}</div>`;return `<section class=unit>${visual}<div class=unit-content><h3>${esc(NAMES[w])}</h3><span class="verdict ${esc(r.verdict)}">${esc(VERDICT[r.verdict]||r.verdict)}</span> · 置信度 ${esc(r.confidence)}${coverageWarning}${details}${history(u)}<details><summary>查看中文工作流内部轨迹</summary>${algorithmTrace(u.result)}</details><details><summary>查看整编前 Judge 结果</summary><pre class=raw>${esc(JSON.stringify(u.result_original,null,2))}</pre></details>${traceBundle(u)}</div></section>`}
function card(r){const comparison=DATA.comparison_enabled?(r.disagreement?'<div class=disagree>三管线结论分歧</div>':'<div>三管线结论一致</div>'):"";return `<article class=sample><div class=sample-head><div><div class=sid>${esc(r.sample_id)}</div><div class=ref>人工参考：${esc(r.reference_label_zh)}</div></div>${comparison}</div><div class="unit-grid ${DATA.layout_class}">${DATA.workflow_order.map(w=>unit(w,r.units[w],r.units[w].image_url)).join("")}</div></article>`}
function render(){const rows=DATA.records.filter(r=>(!labelFilter.value||r.reference_label===labelFilter.value)&&(!disagreementFilter.value||r.disagreement)&&(!taxonomyFilter.value||allFindings(r).some(f=>f.artifact_type===taxonomyFilter.value))&&(!verdictFilter.value||Object.values(r.units).some(u=>(u.error?"error":u.consolidated_result?.verdict||u.result?.verdict)===verdictFilter.value))&&(!statusFilter.value||(statusFilter.value==="ok"&&Object.values(r.units).every(u=>u.result&&(DATA.comparison_enabled||u.consolidated_result)))||(statusFilter.value==="error"&&Object.values(r.units).some(u=>u.error||(u.consolidation_errors||[]).length))));samples.innerHTML=rows.map(card).join("");if(stageFilter)highlight(stageFilter)}
[labelFilter,verdictFilter,disagreementFilter,taxonomyFilter,statusFilter].forEach(x=>x.onchange=render);
let stageFilter="";function highlight(stage){stageFilter=stage;document.querySelectorAll("[data-trace-stage]").forEach(x=>x.classList.toggle("highlight",x.dataset.traceStage===stage));document.querySelectorAll(".step").forEach(x=>x.style.outline=x.dataset.stage===stage?"3px solid #ffbd45":"")}
flowGrid.onclick=e=>{const b=e.target.closest("[data-stage]");if(b)highlight(b.dataset.stage)};clearStage.onclick=()=>highlight("");render();
addEventListener("resize",()=>document.querySelectorAll(".pipeline-visual img").forEach(x=>x.complete&&fit(x)));
</script></body></html>"""


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--omit-model-traces",
        action="store_true",
        help=(
            "Keep full traces.jsonl on disk but omit per-call model traces "
            "from the HTML payload."
        ),
    )
    args = parser.parse_args(argv)
    print(
        build_report(
            args.output_root,
            include_unit_traces=not args.omit_model_traces,
        )
    )


if __name__ == "__main__":
    main()
