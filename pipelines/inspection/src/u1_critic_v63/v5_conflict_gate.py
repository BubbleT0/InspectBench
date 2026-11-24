from __future__ import annotations

import asyncio
import copy
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from critic_workflows import (
    BBox,
    Finding,
    ModelRequest,
    Region,
    VerificationStatus,
)
from critic_workflows.utils import parse_response

from .v4_harness import (
    _V41_INSPECTION_SCHEMA,
    V41AdaptiveContentCoverageAuditor,
    V41EditGroundedConsensusCritic,
    build_multiscale_review_bboxes,
)
from .quality_policy import candidate_is_core


@dataclass(frozen=True)
class LocalInspectionEvidence:
    image_id: str
    region_id: str
    inspection_role: str
    content_kind: str
    content_bbox: BBox
    context_bbox: BBox
    detail_bbox: BBox
    crop_boundary_complete: bool
    defect_present: bool
    materiality: str
    evidence_basis: str
    confidence: float
    grounding_status: str
    confirmed_in_original: bool
    confirmed_in_context: bool
    issues: tuple[dict, ...]
    rationale: str


@dataclass(frozen=True)
class LocalNegativeMatch:
    evidence: LocalInspectionEvidence
    candidate_coverage: float


def candidate_channel(finding: Finding) -> str:
    if finding.artifact_type == "text_visual_integrity":
        return "text_visual"
    if finding.artifact_type == "text_content_fidelity":
        return "text_fidelity"
    return "artifact"


def _compatible_inspection_roles(channel: str) -> frozenset[str]:
    if channel == "text_visual":
        return frozenset({"text_visual", "embedded_text"})
    if channel == "text_fidelity":
        return frozenset({"text_fidelity", "embedded_text"})
    return frozenset({"artifact"})


def _area(bbox: BBox) -> int:
    return (bbox.x2 - bbox.x1) * (bbox.y2 - bbox.y1)


def _candidate_coverage(candidate: BBox, local: BBox) -> float:
    width = max(0, min(candidate.x2, local.x2) - max(candidate.x1, local.x1))
    height = max(0, min(candidate.y2, local.y2) - max(candidate.y1, local.y1))
    return (width * height) / _area(candidate)


def _center_inside(candidate: BBox, local: BBox) -> bool:
    center_x = (candidate.x1 + candidate.x2) / 2
    center_y = (candidate.y1 + candidate.y2) / 2
    return (
        local.x1 <= center_x <= local.x2
        and local.y1 <= center_y <= local.y2
    )


def find_strong_local_negatives(
    finding: Finding,
    evidence: Sequence[LocalInspectionEvidence],
    *,
    min_confidence: float = 0.8,
    min_candidate_coverage: float = 0.7,
) -> list[LocalNegativeMatch]:
    roles = _compatible_inspection_roles(candidate_channel(finding))
    matches: list[LocalNegativeMatch] = []
    for item in evidence:
        if (
            item.inspection_role not in roles
            or item.crop_boundary_complete is not True
            or item.defect_present is not False
            or item.materiality != "none"
            or item.evidence_basis != "visible_only"
            or item.confidence < min_confidence
            or item.grounding_status != "valid"
            or not _center_inside(finding.bbox, item.content_bbox)
        ):
            continue
        coverage = _candidate_coverage(finding.bbox, item.content_bbox)
        if coverage < min_candidate_coverage:
            continue
        matches.append(
            LocalNegativeMatch(
                evidence=item,
                candidate_coverage=coverage,
            )
        )
    return sorted(
        matches,
        key=lambda match: (
            -match.candidate_coverage,
            _area(match.evidence.content_bbox),
            match.evidence.region_id,
        ),
    )


class V5AdaptiveContentCoverageAuditor(V41AdaptiveContentCoverageAuditor):
    """V4.1 coverage plus auditable per-image local inspection evidence."""

    coverage_mode_name = "harness_v5"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._local_evidence_by_image: dict[
            str, list[LocalInspectionEvidence]
        ] = {}
        self._summary_by_image: dict[str, dict[str, Any]] = {}

    def _inspection_rule(self, inspection_role: str) -> str:
        if inspection_role == "embedded_text":
            return (
                "检查实体内所有被设计为可读内容的文字视觉完整性和文字内容。"
                "报告明显乱码、伪文字、断裂、粘连、扭曲，以及仅依据本图即可"
                "直接可验证的明显文字内容错误；不得依赖外部知识、预期文案或"
                "语言猜测。没有文字或没有明显错误时返回无问题。"
            )
        return super()._inspection_rule(inspection_role)

    def _review_bboxes(
        self,
        content: Mapping[str, Any],
        content_bbox: BBox,
    ) -> tuple[BBox, BBox]:
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
        return build_multiscale_review_bboxes(content_bbox, zone_bbox)

    async def _request_inspection(
        self,
        image,
        content: Mapping[str, Any],
        content_bbox: BBox,
        review_bbox: BBox,
        *,
        inspection_role: str,
        boundary_retry_count: int,
    ):
        detail_bbox, context_bbox = self._review_bboxes(
            content,
            content_bbox,
        )
        parsed, detail_crop = await super()._request_inspection(
            image,
            content,
            content_bbox,
            review_bbox,
            inspection_role=inspection_role,
            boundary_retry_count=boundary_retry_count,
        )
        raw_issues = parsed.get("issues", [])
        issues = tuple(
            copy.deepcopy(dict(issue))
            for issue in raw_issues
            if isinstance(issue, Mapping)
        ) if isinstance(raw_issues, list) else ()
        record = LocalInspectionEvidence(
            image_id=image.id,
            region_id=str(content.get("region_id", "")),
            inspection_role=inspection_role,
            content_kind=str(content.get("content_kind", "")),
            content_bbox=content_bbox,
            context_bbox=context_bbox,
            detail_bbox=detail_bbox,
            crop_boundary_complete=(
                parsed.get("crop_boundary_complete") is True
            ),
            defect_present=(parsed.get("defect_present") is True),
            materiality=str(parsed.get("materiality", "")).lower(),
            evidence_basis=str(
                parsed.get("evidence_basis", "")
            ).lower(),
            confidence=float(parsed.get("confidence", 0.0)),
            grounding_status=str(
                parsed.get("grounding_status", "")
            ).lower(),
            confirmed_in_original=(
                parsed.get("confirmed_in_original") is True
            ),
            confirmed_in_context=(
                parsed.get("confirmed_in_context") is True
            ),
            issues=issues,
            rationale=str(parsed.get("rationale", "")).strip(),
        )
        self._local_evidence_by_image.setdefault(image.id, []).append(
            record
        )
        return parsed, detail_crop

    async def audit(self, image, context):
        self._local_evidence_by_image[image.id] = []
        findings, summary = await super().audit(image, context)
        summary = copy.deepcopy(dict(summary))
        self._summary_by_image[image.id] = summary
        self.last_summary = copy.deepcopy(summary)
        return findings, summary

    def get_local_inspection_evidence(
        self,
        image_id: str,
    ) -> tuple[LocalInspectionEvidence, ...]:
        return tuple(self._local_evidence_by_image.get(image_id, ()))

    def get_summary(self, image_id: str) -> dict[str, Any]:
        return copy.deepcopy(self._summary_by_image.get(image_id, {}))


def _evidence_summary(match: LocalNegativeMatch) -> dict[str, Any]:
    item = match.evidence
    return {
        "region_id": item.region_id,
        "inspection_role": item.inspection_role,
        "content_kind": item.content_kind,
        "content_bbox_norm": item.content_bbox.as_list(),
        "context_bbox_norm": item.context_bbox.as_list(),
        "detail_bbox_norm": item.detail_bbox.as_list(),
        "confidence": item.confidence,
        "rationale": item.rationale,
        "candidate_coverage": round(match.candidate_coverage, 6),
    }


def _spatially_corresponds(left: BBox, right: BBox) -> bool:
    width = max(0, min(left.x2, right.x2) - max(left.x1, right.x1))
    height = max(0, min(left.y2, right.y2) - max(left.y1, right.y1))
    intersection = width * height
    return intersection / min(_area(left), _area(right)) >= 0.5


def _issue_channel(issue: Mapping[str, Any]) -> str:
    artifact_type = str(issue.get("artifact_type", ""))
    if artifact_type == "text_visual_integrity":
        return "text_visual"
    if artifact_type == "text_content_fidelity":
        return "text_fidelity"
    return "artifact"


def summarize_local_conflict_gates(
    challenges: Mapping[str, Mapping[str, Any]],
) -> dict[str, int]:
    decisions = [
        str(record.get("local_conflict_gate", {}).get("decision", ""))
        for record in challenges.values()
        if isinstance(record, Mapping)
        and isinstance(record.get("local_conflict_gate"), Mapping)
    ]
    return {
        "conflict_candidate_count": len(decisions),
        "blocked_candidate_count": decisions.count(
            "blocked_by_local_negative"
        ),
        "restored_candidate_count": decisions.count(
            "restored_by_neutral_review"
        ),
    }


class V5EditGroundedConsensusCritic(V41EditGroundedConsensusCritic):
    """V4.1 consensus with local-negative precedence and neutral review."""

    async def critique(self, image, context=None):
        result = await super().critique(image, context)
        challenges: Mapping[str, Mapping[str, Any]] = {}
        for event in result.trace:
            if event.stage == "consensus.challenges":
                challenges = event.data
                break
        result.metadata["pipeline_version"] = "v5"
        result.metadata["v5_local_conflict_gate"] = (
            summarize_local_conflict_gates(challenges)
        )
        get_evidence = getattr(
            self.text_auditor,
            "get_local_inspection_evidence",
            None,
        )
        result.metadata["v5_local_inspection_evidence_count"] = len(
            get_evidence(image.id) if callable(get_evidence) else ()
        )
        return result

    async def _challenge(
        self,
        image,
        context,
        candidate_id: str,
        finding: Finding,
    ):
        get_evidence = getattr(
            self.text_auditor,
            "get_local_inspection_evidence",
            None,
        )
        local_evidence = (
            get_evidence(image.id) if callable(get_evidence) else ()
        )
        matches = find_strong_local_negatives(finding, local_evidence)
        if not matches:
            return await super()._challenge(
                image,
                context,
                candidate_id,
                finding,
            )
        return await self._neutral_conflict_review(
            image,
            candidate_id,
            finding,
            matches,
        )

    async def _neutral_conflict_review(
        self,
        image,
        candidate_id: str,
        finding: Finding,
        matches: Sequence[LocalNegativeMatch],
    ):
        primary = matches[0]
        local = primary.evidence
        context_crop, detail_crop = await asyncio.gather(
            self.image_tool.crop(
                image,
                Region(
                    id=f"{candidate_id}:v5-conflict-context",
                    bbox=local.context_bbox,
                    label="local_conflict_context",
                ),
            ),
            self.image_tool.crop(
                image,
                Region(
                    id=f"{candidate_id}:v5-conflict-detail",
                    bbox=local.detail_bbox,
                    label="local_conflict_detail",
                ),
            ),
        )
        channel = candidate_channel(finding)
        channel_rule = (
            "逐字检查明显文字视觉损坏，以及仅依据本图内部即可直接验证的明显"
            "文字内容错误。不得依据外部知识、预期文案或语言猜测补全。"
            if channel in {"text_visual", "text_fidelity"}
            else
            "检查明显的渲染、结构、拓扑、关系或物理错误。"
        )
        response = await self.skeptic_model.complete(
            ModelRequest(
                stage="consensus.local_conflict_review",
                system_prompt=(
                    "你是中立的核心视觉缺陷复核员。图像0是原图，图像1是程序"
                    "生成的上下文视图，图像2是程序生成并可能无损放大的细节视图。"
                    "你没有收到任何其他审查员的判断，必须仅依据像素从零检查。"
                    "只报告肉眼明显、值得局部重绘的 major/critical 问题；忽略"
                    "审美偏好、正常风格化、遮挡、透视、景深、磨损、压缩和轻微"
                    "瑕疵。所有自然语言使用简体中文。"
                ),
                user_prompt=(
                    f"检查通道：{channel}。{channel_rule}\n"
                    "裁剪仅用于放大观察，不得把裁剪边缘产生的断头、半行或残片"
                    "当作原图问题。阳性问题必须在图像0和图像1中同时成立，并返回"
                    "可直接交给 edit 模型的一处原图整数 [0,1000] bbox；否则返回"
                    "无问题。同一区域的全部核心问题放入 issues。"
                ),
                images=(image, context_crop, detail_crop),
                response_schema=_V41_INSPECTION_SCHEMA,
                metadata={
                    "workflow": "adversarial_consensus",
                    "coverage_mode": "harness_v5",
                    "candidate_id": candidate_id,
                    "local_region_id": local.region_id,
                    "inspection_role": "neutral_conflict_review",
                },
            )
        )
        parsed = dict(parse_response(response))
        decision = self._apply_neutral_conflict_decision(
            candidate_id,
            finding,
            parsed,
            matches,
        )
        return candidate_id, decision[0], decision[1]

    def _apply_neutral_conflict_decision(
        self,
        candidate_id: str,
        finding: Finding,
        parsed: Mapping[str, Any],
        matches: Sequence[LocalNegativeMatch],
    ) -> tuple[VerificationStatus, dict[str, Any]]:
        local_summaries = [_evidence_summary(match) for match in matches]
        raw_issues = parsed.get("issues", [])
        issues = [
            copy.deepcopy(dict(issue))
            for issue in raw_issues
            if isinstance(issue, Mapping)
            and _issue_channel(issue) == candidate_channel(finding)
        ] if isinstance(raw_issues, list) else []
        try:
            review_bbox = BBox.from_value(parsed.get("edit_bbox_full_norm"))
        except (TypeError, ValueError):
            review_bbox = None
        try:
            confidence = float(parsed.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        restored = (
            candidate_is_core(parsed)
            and confidence >= 0.8
            and str(parsed.get("grounding_status", "")).lower() == "valid"
            and parsed.get("crop_boundary_complete") is True
            and parsed.get("confirmed_in_original") is True
            and parsed.get("confirmed_in_context") is True
            and bool(issues)
            and review_bbox is not None
            and _spatially_corresponds(finding.bbox, review_bbox)
        )
        gate = {
            "candidate_id": candidate_id,
            "strong_local_negatives": local_summaries,
            "decision": (
                "restored_by_neutral_review"
                if restored
                else "blocked_by_local_negative"
            ),
            "neutral_rationale": str(
                parsed.get("rationale", "")
            ).strip(),
        }
        if not restored:
            return VerificationStatus.CONTRADICTED, {
                "status": "contradicted",
                "confidence": max(
                    (match.evidence.confidence for match in matches),
                    default=0.8,
                ),
                "counterargument": (
                    "同区域高置信度局部检查判定正常，中立复核未能独立确认"
                    "可局部重绘的明显核心缺陷。"
                ),
                "missing_evidence": "缺少通过中立复核的直接可见阳性证据。",
                "defect_present": False,
                "materiality": "none",
                "evidence_basis": "visible_only",
                "grounding_status": "invalid",
                "revised_bbox_norm": None,
                "grounding_rationale": "局部阴性证据优先，候选被门控拦截。",
                "effective_grounding_status": "invalid",
                "effective_bbox_norm": finding.bbox.as_list(),
                "local_conflict_gate": gate,
            }

        original_claim = copy.deepcopy(finding.as_dict())
        issue_confidences = [
            float(issue.get("confidence", confidence)) for issue in issues
        ]
        finding.bbox = review_bbox
        finding.artifact_type = str(issues[0]["artifact_type"])
        finding.observation = "；".join(
            str(issue.get("observation", "")).strip()
            for issue in issues
            if str(issue.get("observation", "")).strip()
        )
        finding.abnormality = "；".join(
            str(issue.get("abnormality", "")).strip()
            for issue in issues
            if str(issue.get("abnormality", "")).strip()
        )
        finding.confidence = min(confidence, max(issue_confidences))
        finding.metadata.update(
            {
                "v5_original_full_image_claim": original_claim,
                "v5_local_negative_matches": local_summaries,
                "v5_neutral_review": copy.deepcopy(dict(parsed)),
                "edit_target": finding.observation,
                "edit_region_rationale": str(
                    parsed.get("rationale", "")
                ).strip(),
                "revised_bbox_norm": review_bbox.as_list(),
                "grounding_status": "valid",
                "grounding_rationale": str(
                    parsed.get("rationale", "")
                ).strip(),
                "review_defect_present": True,
                "review_materiality": str(
                    parsed.get("materiality", "")
                ).lower(),
                "review_evidence_basis": "visible_only",
            }
        )
        return VerificationStatus.SUPPORTED, {
            "status": "supported",
            "confidence": finding.confidence,
            "counterargument": str(parsed.get("rationale", "")).strip(),
            "missing_evidence": "无",
            "defect_present": True,
            "materiality": str(parsed.get("materiality", "")).lower(),
            "evidence_basis": "visible_only",
            "grounding_status": "valid",
            "revised_bbox_norm": review_bbox.as_list(),
            "grounding_rationale": str(parsed.get("rationale", "")).strip(),
            "effective_grounding_status": "valid",
            "effective_bbox_norm": review_bbox.as_list(),
            "local_conflict_gate": gate,
        }
