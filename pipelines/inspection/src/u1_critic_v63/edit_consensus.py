from __future__ import annotations

from typing import Any, Mapping

from critic_workflows import (
    AdversarialConsensusCritic,
    BBox,
    CriticContext,
    CriticRole,
    Finding,
    ImageAsset,
    ModelRequest,
    Region,
    VerificationStatus,
)
from critic_workflows.utils import parse_response, serialize_json

from .quality_policy import candidate_is_core


_BBOX_SCHEMA: Mapping[str, Any] = {
    "type": "array",
    "items": {"type": "integer", "minimum": 0, "maximum": 1000},
    "minItems": 4,
    "maxItems": 4,
}

_EDIT_FINDING_SCHEMA: Mapping[str, Any] = {
    "type": "object",
    "required": [
        "region_id",
        "bbox_norm",
        "artifact_type",
        "observation",
        "abnormality",
        "confidence",
        "defect_present",
        "materiality",
        "evidence_basis",
        "evidence_ids",
        "edit_target",
        "edit_region_rationale",
    ],
    "properties": {
        "region_id": {"type": "string"},
        "bbox_norm": _BBOX_SCHEMA,
        "artifact_type": {"type": "string"},
        "observation": {"type": "string"},
        "abnormality": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
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
        "evidence_ids": {"type": "array", "items": {"type": "string"}},
        "edit_target": {"type": "string"},
        "edit_region_rationale": {"type": "string"},
    },
}

_EDIT_PROPOSAL_SCHEMA: Mapping[str, Any] = {
    "type": "object",
    "required": ["verdict", "confidence", "findings"],
    "properties": {
        "verdict": {
            "type": "string",
            "enum": ["artifact_present", "no_artifact", "uncertain"],
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "findings": {
            "type": "array",
            "items": _EDIT_FINDING_SCHEMA,
        },
    },
}

_EDIT_CHALLENGE_SCHEMA: Mapping[str, Any] = {
    "type": "object",
    "required": [
        "status",
        "confidence",
        "counterargument",
        "missing_evidence",
        "defect_present",
        "materiality",
        "evidence_basis",
        "grounding_status",
        "revised_bbox_norm",
        "grounding_rationale",
    ],
    "properties": {
        "status": {
            "type": "string",
            "enum": ["supported", "contradicted", "uncertain"],
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "counterargument": {"type": "string"},
        "missing_evidence": {"type": "string"},
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
        "grounding_status": {
            "type": "string",
            "enum": ["valid", "needs_revision", "invalid"],
        },
        "revised_bbox_norm": {
            "anyOf": [_BBOX_SCHEMA, {"type": "null"}],
        },
        "grounding_rationale": {"type": "string"},
    },
}

_EDIT_JUDGE_SCHEMA: Mapping[str, Any] = {
    "type": "object",
    "required": [
        "verdict",
        "confidence",
        "accepted_candidate_ids",
        "uncertain_candidate_ids",
        "rationale",
    ],
    "properties": {
        "verdict": {
            "type": "string",
            "enum": ["artifact_present", "no_artifact", "uncertain"],
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "accepted_candidate_ids": {
            "type": "array",
            "items": {"type": "string"},
        },
        "uncertain_candidate_ids": {
            "type": "array",
            "items": {"type": "string"},
        },
        "rationale": {"type": "string"},
    },
}


def is_near_full_frame(
    bbox: BBox,
    edge_tolerance: int = 20,
) -> bool:
    return (
        bbox.x1 < edge_tolerance
        and bbox.y1 < edge_tolerance
        and bbox.x2 > 1000 - edge_tolerance
        and bbox.y2 > 1000 - edge_tolerance
    )


class EditGroundedAdversarialConsensusCritic(AdversarialConsensusCritic):
    """Adversarial consensus whose bboxes are repaint-ready edit regions."""

    def __init__(
        self,
        proposer_model,
        *,
        skeptic_model,
        judge_model,
        image_tool,
        text_auditor=None,
        role_models=None,
        config=None,
    ):
        super().__init__(
            proposer_model,
            skeptic_model=skeptic_model,
            judge_model=judge_model,
            role_models=role_models,
            config=config,
        )
        self.image_tool = image_tool
        self.text_auditor = text_auditor

    async def _propose(
        self,
        image: ImageAsset,
        context: CriticContext,
        role: CriticRole,
    ) -> tuple[CriticRole, list[Finding], dict[str, Any]]:
        if role.name == "text_coverage":
            if self.text_auditor is None:
                raise RuntimeError("text_coverage role requires a text auditor")
            findings, summary = await self.text_auditor.audit(
                image,
                context,
            )
            return role, findings, summary
        model = self.role_models.get(role.name, self.proposer_model)
        response = await model.complete(
            ModelRequest(
                stage="consensus.propose",
                system_prompt=(
                    f"你是 {role.name} 视觉质量评估角色，专注范围是：{role.focus}"
                    "只依据图中直接可见证据，所有自然语言字段使用简体中文。"
                    "只报告肉眼明显、值得局部重绘的 major 或 critical 生成错误。"
                    "不要报告审美偏好、正常风格化、景深、雾气、光晕、轻微软边、"
                    "弱阴影/反射、压缩痕迹或其他不影响使用的轻微瑕疵。不得使用"
                    "当前日期、历史事实、排名、价格、身份等外部世界知识。"
                ),
                user_prompt=(
                    f"生成提示词：{context.generation_prompt or '未提供'}\n"
                    f"可用问题分类：{list(context.taxonomy) or '开放分类'}\n\n"
                    "独立检查图片，返回结论和最多 5 个原子问题。每个问题必须先确定"
                    "一个可供图片编辑模型局部重绘的独立 edit_target，再给出 bbox_norm。"
                    "bbox 应覆盖完整问题实体、缺陷像素和重绘所需的少量上下文，允许适度"
                    "冗余，不追求贴边最紧；同时排除无关实体和大面积无关场景。不能用整图"
                    "框代替定位，空间分离的问题必须拆开。每条 finding 必须包含 "
                    "region_id、bbox_norm、artifact_type、observation、abnormality、"
                    "confidence、evidence_ids、edit_target 和 edit_region_rationale。"
                    "同时返回 defect_present、materiality 和 evidence_basis。"
                    "若观察实际表示正常、合理、仅属风格化或无明显缺陷，就不要把它"
                    "放进 findings。"
                ),
                images=(image, *context.reference_images),
                response_schema=_EDIT_PROPOSAL_SCHEMA,
                metadata={
                    "workflow": "adversarial_consensus",
                    "role": role.name,
                },
            )
        )
        parsed = parse_response(response)
        findings: list[Finding] = []
        invalid_finding_count = 0
        policy_rejected_finding_count = 0
        raw_findings = parsed.get("findings", [])
        if isinstance(raw_findings, list):
            for raw in raw_findings:
                if not isinstance(raw, Mapping):
                    invalid_finding_count += 1
                    continue
                if (
                    str(raw.get("artifact_type", "")).startswith("text_")
                    or not candidate_is_core(raw)
                ):
                    policy_rejected_finding_count += 1
                    continue
                try:
                    finding = Finding.from_mapping(
                        raw,
                        default_region_id=f"{image.id}:full",
                        source=f"role:{role.name}",
                    )
                except (TypeError, ValueError):
                    invalid_finding_count += 1
                    continue
                finding.metadata.update(
                    {
                        "edit_target": str(raw.get("edit_target", "")).strip(),
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
                    }
                )
                findings.append(finding)
        summary = dict(parsed)
        summary["invalid_finding_count"] = invalid_finding_count
        summary["policy_rejected_finding_count"] = (
            policy_rejected_finding_count
        )
        return role, findings, summary

    async def _challenge(
        self,
        image: ImageAsset,
        context: CriticContext,
        candidate_id: str,
        finding: Finding,
    ) -> tuple[str, VerificationStatus, dict[str, Any]]:
        crop = await self.image_tool.crop(
            image,
            Region(
                id=f"{candidate_id}:verify",
                bbox=finding.bbox,
                label=finding.artifact_type,
            ),
            padding=0.08,
        )
        response = await self.skeptic_model.complete(
            ModelRequest(
                stage="consensus.challenge",
                system_prompt=(
                    "你是严格的视觉问题与编辑区域复核员。先寻找遮挡、透视、模糊、"
                    "风格化、反射或物体重叠等正常解释，再判断问题是否成立。随后检查 bbox "
                    "能否直接交给图片编辑模型局部重绘。图像0是整图上下文，图像1是候选"
                    "区域的真实裁剪图。只接受肉眼明显、值得局部重绘的 major/critical "
                    "问题；不得接受美观偏好、正常风格化、景深、雾气、光晕、轻微软边、"
                    "弱阴影/反射、压缩痕迹或其他轻微瑕疵。不得使用外部世界知识。"
                    "所有自然语言字段使用简体中文。"
                ),
                user_prompt=(
                    f"候选编号：{candidate_id}\n"
                    f"生成提示词：{context.generation_prompt or '未提供'}\n"
                    f"候选问题：\n{serialize_json(finding.as_dict())}\n\n"
                    "分别判断问题证据和编辑区域。有效编辑区域应覆盖完整问题实体、缺陷像素"
                    "及重绘所需的少量上下文，可适度冗余，但应排除无关实体和大面积无关场景。"
                    "不要为了更紧而裁掉实体连接、纹理过渡或必要背景。若原 bbox 不适合，"
                    "返回 needs_revision 和修正后的 revised_bbox_norm；修正可以缩小无关"
                    "区域，也可以补足必要上下文。不能返回整图框。"
                    "同时返回 defect_present、materiality 和 evidence_basis。若问题"
                    "只是轻微、正常、合理、风格化或无明显缺陷，必须返回 "
                    "defect_present=false 或 materiality=minor/none。"
                ),
                images=(image, crop, *context.reference_images),
                response_schema=_EDIT_CHALLENGE_SCHEMA,
                metadata={
                    "workflow": "adversarial_consensus",
                    "candidate_id": candidate_id,
                },
            )
        )
        parsed = dict(parse_response(response))
        return self._apply_challenge_review(
            candidate_id,
            finding,
            parsed,
        )

    def _apply_challenge_review(
        self,
        candidate_id: str,
        finding: Finding,
        parsed: Mapping[str, Any],
    ) -> tuple[str, VerificationStatus, dict[str, Any]]:
        parsed = dict(parsed)
        raw_status = str(parsed.get("status", "uncertain")).lower()
        raw_grounding = str(
            parsed.get("grounding_status", "invalid")
        ).lower()
        original_is_full = is_near_full_frame(finding.bbox)
        revised_bbox = self._valid_revised_bbox(
            parsed.get("revised_bbox_norm")
        )
        if revised_bbox is not None:
            finding.bbox = revised_bbox
            finding.metadata["revised_bbox_norm"] = revised_bbox.as_list()

        if raw_grounding == "invalid":
            grounding_status = "invalid"
        elif revised_bbox is not None:
            grounding_status = "valid"
        elif original_is_full:
            grounding_status = "invalid"
        elif raw_grounding == "valid":
            grounding_status = "valid"
        else:
            grounding_status = "invalid"

        grounding_rationale = str(
            parsed.get("grounding_rationale", "")
        ).strip()
        finding.metadata.update(
            {
                "grounding_status": grounding_status,
                "grounding_rationale": grounding_rationale,
                "review_defect_present": parsed.get("defect_present"),
                "review_materiality": str(
                    parsed.get("materiality", "")
                ).strip().lower(),
                "review_evidence_basis": str(
                    parsed.get("evidence_basis", "")
                ).strip().lower(),
            }
        )
        confidence = float(parsed.get("confidence", 0.0))
        if (
            raw_status == VerificationStatus.SUPPORTED.value
            and candidate_is_core(parsed)
            and grounding_status == "valid"
        ):
            status = VerificationStatus.SUPPORTED
        elif raw_status == VerificationStatus.CONTRADICTED.value:
            status = VerificationStatus.CONTRADICTED
        elif not candidate_is_core(parsed):
            status = VerificationStatus.CONTRADICTED
        else:
            status = VerificationStatus.UNCERTAIN
        if finding.source == "role:text_coverage":
            finding.confidence = confidence
        else:
            finding.confidence = min(
                finding.confidence,
                confidence or finding.confidence,
            )
        parsed["effective_grounding_status"] = grounding_status
        parsed["effective_bbox_norm"] = finding.bbox.as_list()
        return candidate_id, status, parsed

    async def _judge(
        self,
        *,
        image: ImageAsset,
        context: CriticContext,
        role_summaries: list[dict[str, Any]],
        candidates: Mapping[str, Finding],
        challenges: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, Any]:
        packet = {
            "role_summaries": role_summaries,
            "candidates": {
                candidate_id: finding.as_dict()
                for candidate_id, finding in candidates.items()
            },
            "skeptic_reviews": challenges,
        }
        response = await self.judge_model.complete(
            ModelRequest(
                stage="consensus.judge",
                system_prompt=(
                    "你是最终裁决者。依据可见证据、质疑意见和编辑区域复核结果解决分歧。"
                    "只能选择已有 candidate ID，不能创建新问题。所有自然语言字段使用"
                    "简体中文。"
                ),
                user_prompt=(
                    f"生成提示词：{context.generation_prompt or '未提供'}\n"
                    f"复核包：\n{serialize_json(packet)}\n\n"
                    "只有视觉问题得到支持且 grounding_status=valid 的候选才能进入 "
                    "accepted_candidate_ids。编辑区域无效、仍为整图或问题已被反驳的候选"
                    "不得接受。返回最终结论、置信度、接受候选、不确定候选和简短理由。"
                ),
                images=(image, *context.reference_images),
                response_schema=_EDIT_JUDGE_SCHEMA,
                metadata={
                    "workflow": "adversarial_consensus",
                    "candidate_count": len(candidates),
                },
            )
        )
        parsed = dict(parse_response(response))
        raw_accepted = self._valid_ids(
            parsed.get("accepted_candidate_ids", []),
            candidates,
        )
        raw_uncertain = self._valid_ids(
            parsed.get("uncertain_candidate_ids", []),
            candidates,
        )

        def edit_ready(candidate_id: str) -> bool:
            finding = candidates[candidate_id]
            review = {
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
            return (
                finding.verification is VerificationStatus.SUPPORTED
                and candidate_is_core(review)
                and finding.metadata.get("grounding_status") == "valid"
                and not is_near_full_frame(finding.bbox)
            )

        accepted = [
            candidate_id
            for candidate_id in raw_accepted
            if edit_ready(candidate_id)
        ]
        uncertain: list[str] = []
        rejected = [
            candidate_id
            for candidate_id in raw_accepted
            if candidate_id not in accepted
        ]
        if accepted:
            parsed["verdict"] = "artifact_present"
        else:
            parsed["verdict"] = "no_artifact"
        parsed["accepted_candidate_ids"] = accepted
        parsed["uncertain_candidate_ids"] = uncertain
        parsed["grounding_rejected_candidate_ids"] = rejected
        return parsed

    @staticmethod
    def _valid_revised_bbox(value: Any) -> BBox | None:
        if value is None:
            return None
        try:
            bbox = BBox.from_value(value)
        except (TypeError, ValueError):
            return None
        return None if is_near_full_frame(bbox) else bbox
