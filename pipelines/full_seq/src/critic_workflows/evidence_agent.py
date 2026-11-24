from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .adapters import PassthroughImageTool
from .interfaces import ImageTool, ModelClient
from .types import (
    BBox,
    CriticContext,
    CriticResult,
    Finding,
    ImageAsset,
    ModelRequest,
    Region,
    TraceEvent,
    Verdict,
    VerificationStatus,
)
from .utils import parse_response, safe_verdict, serialize_json


@dataclass(frozen=True)
class EvidenceSeekingConfig:
    max_steps: int = 6
    crop_padding: float = 0.12
    verifier_threshold: float = 0.55
    max_evidence_images: int = 8


_ACTION_SCHEMA: Mapping[str, Any] = {
    "type": "object",
    "required": ["action", "reason"],
    "properties": {
        "action": {"type": "string", "enum": ["inspect", "finish", "abstain"]},
        "reason": {"type": "string"},
        "bbox_norm": {
            "type": "array",
            "items": {"type": "integer", "minimum": 0, "maximum": 1000},
            "minItems": 4,
            "maxItems": 4,
        },
        "label": {"type": "string"},
        "verdict": {
            "type": "string",
            "enum": ["artifact_present", "no_artifact", "uncertain"],
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "findings": {"type": "array", "items": {"type": "object"}},
    },
}

_VERIFY_SCHEMA: Mapping[str, Any] = {
    "type": "object",
    "required": ["status", "confidence", "reason"],
    "properties": {
        "status": {
            "type": "string",
            "enum": ["supported", "contradicted", "uncertain"],
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "reason": {"type": "string"},
    },
}


class EvidenceSeekingAgentCritic:
    """ARM-Thinker/V*-style dynamic critic.

    The planner decides whether the current views are sufficient. It can request
    normalized crops, observe them, and continue until it emits a structured
    conclusion. A verifier then checks each final claim.
    """

    def __init__(
        self,
        planner_model: ModelClient,
        *,
        verifier_model: ModelClient | None = None,
        image_tool: ImageTool | None = None,
        config: EvidenceSeekingConfig | None = None,
    ):
        self.planner_model = planner_model
        self.verifier_model = verifier_model or planner_model
        self.image_tool = image_tool or PassthroughImageTool()
        self.config = config or EvidenceSeekingConfig()

    async def critique(
        self,
        image: ImageAsset,
        context: CriticContext | None = None,
    ) -> CriticResult:
        context = context or CriticContext()
        evidence: list[ImageAsset] = [image]
        trace: list[TraceEvent] = []
        conclusion: dict[str, Any] | None = None

        for step in range(1, self.config.max_steps + 1):
            parsed = await self._next_action(
                image=image,
                context=context,
                evidence=evidence,
                trace=trace,
                step=step,
                force_finish=False,
            )
            action = str(parsed.get("action", "abstain")).lower()
            trace.append(
                TraceEvent(
                    stage="evidence.action",
                    message=f"Step {step}: {action}.",
                    data=parsed,
                )
            )

            if action == "inspect":
                if len(evidence) >= self.config.max_evidence_images:
                    trace.append(
                        TraceEvent(
                            stage="evidence.limit",
                            message="Evidence image limit reached; forcing a conclusion.",
                        )
                    )
                    break
                try:
                    bbox = BBox.from_value(parsed["bbox_norm"])
                except (KeyError, TypeError, ValueError) as exc:
                    trace.append(
                        TraceEvent(
                            stage="evidence.invalid_action",
                            message=f"Invalid crop request: {exc}",
                            data=parsed,
                        )
                    )
                    conclusion = {
                        "action": "abstain",
                        "verdict": Verdict.UNCERTAIN.value,
                        "confidence": 0.3,
                        "findings": [],
                        "reason": "Planner emitted an invalid crop request.",
                    }
                    break
                region = Region(
                    id=f"{image.id}:agent:{step}",
                    bbox=bbox,
                    label=str(parsed.get("label", "requested_region")),
                    metadata={"planner_reason": str(parsed.get("reason", ""))},
                )
                crop = await self.image_tool.crop(
                    image,
                    region,
                    padding=self.config.crop_padding,
                )
                evidence.append(crop)
                continue

            conclusion = parsed
            break

        if conclusion is None:
            conclusion = await self._next_action(
                image=image,
                context=context,
                evidence=evidence,
                trace=trace,
                step=self.config.max_steps + 1,
                force_finish=True,
            )
            trace.append(
                TraceEvent(
                    stage="evidence.force_finish",
                    message="Planner was forced to conclude after reaching a limit.",
                    data=conclusion,
                )
            )

        proposed = self._parse_findings(conclusion, image)
        verified: list[Finding] = []
        uncertain: list[Finding] = []
        for finding in proposed:
            status = await self._verify(image, context, evidence, finding)
            trace.append(
                TraceEvent(
                    stage="evidence.verify",
                    message=f"Verifier marked finding as {status.value}.",
                    data={"finding": finding.as_dict()},
                )
            )
            if status is VerificationStatus.SUPPORTED:
                verified.append(finding)
            elif status is VerificationStatus.UNCERTAIN:
                uncertain.append(finding)

        if verified:
            verdict = Verdict.ARTIFACT_PRESENT
            confidence = max(finding.confidence for finding in verified)
        elif uncertain:
            verdict = Verdict.UNCERTAIN
            confidence = max(finding.confidence for finding in uncertain)
        else:
            raw_verdict = safe_verdict(conclusion.get("verdict"))
            verdict = (
                Verdict.NO_ARTIFACT
                if raw_verdict is Verdict.NO_ARTIFACT
                else Verdict.UNCERTAIN
            )
            confidence = float(conclusion.get("confidence", 0.5))

        return CriticResult(
            verdict=verdict,
            confidence=confidence,
            findings=verified,
            uncertain_findings=uncertain,
            trace=trace,
            metadata={
                "workflow": "evidence_seeking_agent",
                "evidence_ids": [item.id for item in evidence],
                "planner_reason": str(conclusion.get("reason", "")),
            },
        )

    async def _next_action(
        self,
        *,
        image: ImageAsset,
        context: CriticContext,
        evidence: list[ImageAsset],
        trace: list[TraceEvent],
        step: int,
        force_finish: bool,
    ) -> dict[str, Any]:
        evidence_index = [
            {
                "image_index": index,
                "evidence_id": item.id,
                "parent_id": item.parent_id,
                "bbox_norm": item.bbox.as_list() if item.bbox else None,
            }
            for index, item in enumerate(evidence)
        ]
        prior_actions = [
            {"stage": event.stage, "message": event.message, "data": dict(event.data)}
            for event in trace[-6:]
        ]
        instruction = (
            "You must finish now. Do not request another crop."
            if force_finish
            else (
                "If the evidence is insufficient, request exactly one crop using "
                "integer [0,1000] bbox coordinates and action=inspect. Otherwise "
                "finish or abstain."
            )
        )
        response = await self.planner_model.complete(
            ModelRequest(
                stage="evidence.force_finish" if force_finish else "evidence.plan",
                system_prompt=(
                    "You are an evidence-seeking visual critic. Inspect the image for "
                    "visual artifacts. Allocate extra visual attention only when needed. "
                    "Do not treat ambiguity, style, blur, perspective, or occlusion as a "
                    "confirmed artifact without direct evidence."
                ),
                user_prompt=(
                    f"Step: {step}\n"
                    f"Generation prompt: {context.generation_prompt or 'not provided'}\n"
                    f"Requested taxonomy: {list(context.taxonomy) or 'open set'}\n"
                    f"Evidence index:\n{serialize_json(evidence_index)}\n"
                    f"Recent actions:\n{serialize_json(prior_actions)}\n\n"
                    f"{instruction}\n"
                    "When finishing, return verdict, confidence, and atomic findings. "
                    "Every finding must include region_id, bbox_norm, artifact_type, "
                    "observation, abnormality, confidence, and evidence_ids. Return JSON only."
                ),
                images=tuple(evidence),
                response_schema=_ACTION_SCHEMA,
                metadata={
                    "workflow": "evidence_seeking_agent",
                    "step": step,
                    "force_finish": force_finish,
                    "root_image_id": image.id,
                },
            )
        )
        return parse_response(response)

    @staticmethod
    def _parse_findings(
        conclusion: Mapping[str, Any],
        image: ImageAsset,
    ) -> list[Finding]:
        values = conclusion.get("findings", [])
        if not isinstance(values, list):
            return []
        findings: list[Finding] = []
        for value in values:
            if not isinstance(value, Mapping):
                continue
            findings.append(
                Finding.from_mapping(
                    value,
                    default_region_id=f"{image.id}:full",
                    source="evidence_seeking_agent",
                )
            )
        return findings

    async def _verify(
        self,
        image: ImageAsset,
        context: CriticContext,
        evidence: list[ImageAsset],
        finding: Finding,
    ) -> VerificationStatus:
        response = await self.verifier_model.complete(
            ModelRequest(
                stage="evidence.verify",
                system_prompt=(
                    "You are an independent visual claim verifier. Evaluate only whether "
                    "the proposed observation and abnormality are supported by the supplied "
                    "pixels. Prefer uncertain over an unsupported confident judgment."
                ),
                user_prompt=(
                    f"Generation prompt: {context.generation_prompt or 'not provided'}\n"
                    f"Claim:\n{serialize_json(finding.as_dict())}\n\n"
                    "Return supported, contradicted, or uncertain with confidence and a "
                    "short evidence-based reason."
                ),
                images=tuple(evidence) + context.reference_images,
                response_schema=_VERIFY_SCHEMA,
                metadata={
                    "workflow": "evidence_seeking_agent",
                    "region_id": finding.region_id,
                },
            )
        )
        parsed = parse_response(response)
        raw_status = str(parsed.get("status", "uncertain")).lower()
        confidence = float(parsed.get("confidence", 0.0))
        if (
            raw_status == VerificationStatus.SUPPORTED.value
            and confidence >= self.config.verifier_threshold
        ):
            status = VerificationStatus.SUPPORTED
        elif raw_status == VerificationStatus.CONTRADICTED.value:
            status = VerificationStatus.CONTRADICTED
        else:
            status = VerificationStatus.UNCERTAIN
        finding.verification = status
        finding.confidence = confidence
        finding.metadata["verifier_reason"] = str(parsed.get("reason", ""))
        return status
