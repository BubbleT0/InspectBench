from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Mapping

from .interfaces import ModelClient
from .types import (
    CriticContext,
    CriticResult,
    Finding,
    ImageAsset,
    ModelRequest,
    TraceEvent,
    Verdict,
    VerificationStatus,
)
from .utils import parse_response, safe_verdict, serialize_json


@dataclass(frozen=True)
class CriticRole:
    name: str
    focus: str


DEFAULT_ROLES: tuple[CriticRole, ...] = (
    CriticRole(
        name="anatomy_morphology",
        focus=(
            "Human and animal anatomy, object morphology, duplicated or missing parts, "
            "broken boundaries, and local geometric plausibility."
        ),
    ),
    CriticRole(
        name="physics_relations",
        focus=(
            "Object interactions, contact and support, occlusion order, scale, perspective, "
            "shadows, reflections, and physical or spatial consistency."
        ),
    ),
    CriticRole(
        name="rendering_text",
        focus=(
            "Text legibility, repeated textures, local rendering discontinuities, color "
            "leakage, over-smoothing, and other generation-specific visual artifacts."
        ),
    ),
)


@dataclass(frozen=True)
class AdversarialConsensusConfig:
    roles: tuple[CriticRole, ...] = DEFAULT_ROLES
    max_findings_per_role: int = 5
    max_concurrency: int = 4
    skeptic_threshold: float = 0.55


_PROPOSAL_SCHEMA: Mapping[str, Any] = {
    "type": "object",
    "required": ["verdict", "confidence", "findings"],
    "properties": {
        "verdict": {
            "type": "string",
            "enum": ["artifact_present", "no_artifact", "uncertain"],
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "findings": {"type": "array", "items": {"type": "object"}},
    },
}

_CHALLENGE_SCHEMA: Mapping[str, Any] = {
    "type": "object",
    "required": ["status", "confidence", "counterargument"],
    "properties": {
        "status": {
            "type": "string",
            "enum": ["supported", "contradicted", "uncertain"],
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "counterargument": {"type": "string"},
        "missing_evidence": {"type": "string"},
    },
}

_JUDGE_SCHEMA: Mapping[str, Any] = {
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
        "accepted_candidate_ids": {"type": "array", "items": {"type": "string"}},
        "uncertain_candidate_ids": {"type": "array", "items": {"type": "string"}},
        "rationale": {"type": "string"},
    },
}


class AdversarialConsensusCritic:
    """Precision-oriented proposer–skeptic–judge workflow.

    Independent roles first inspect the full image from different perspectives.
    A skeptic challenges every atomic claim, then a final judge may only select
    from those candidate IDs. This prevents the aggregation stage from inventing
    a new ungrounded diagnosis.
    """

    def __init__(
        self,
        proposer_model: ModelClient,
        *,
        skeptic_model: ModelClient,
        judge_model: ModelClient,
        role_models: Mapping[str, ModelClient] | None = None,
        config: AdversarialConsensusConfig | None = None,
    ):
        self.proposer_model = proposer_model
        self.skeptic_model = skeptic_model
        self.judge_model = judge_model
        self.role_models = dict(role_models or {})
        self.config = config or AdversarialConsensusConfig()

    async def critique(
        self,
        image: ImageAsset,
        context: CriticContext | None = None,
    ) -> CriticResult:
        context = context or CriticContext()
        trace: list[TraceEvent] = []
        semaphore = asyncio.Semaphore(self.config.max_concurrency)

        async def propose(role: CriticRole) -> tuple[CriticRole, list[Finding], dict[str, Any]]:
            async with semaphore:
                return await self._propose(image, context, role)

        proposal_results = await asyncio.gather(
            *(propose(role) for role in self.config.roles)
        )
        candidates: dict[str, Finding] = {}
        role_summaries: list[dict[str, Any]] = []
        next_candidate = 0
        for role, findings, summary in proposal_results:
            candidate_ids: list[str] = []
            for finding in findings[: self.config.max_findings_per_role]:
                candidate_id = f"c{next_candidate}"
                next_candidate += 1
                finding.metadata["candidate_id"] = candidate_id
                finding.metadata["role"] = role.name
                candidates[candidate_id] = finding
                candidate_ids.append(candidate_id)
            role_summaries.append(
                {
                    "role": role.name,
                    "verdict": summary.get("verdict", "uncertain"),
                    "confidence": summary.get("confidence", 0.5),
                    "candidate_ids": candidate_ids,
                }
            )
        trace.append(
            TraceEvent(
                stage="consensus.proposals",
                message=f"{len(self.config.roles)} roles proposed {len(candidates)} claims.",
                data={"roles": role_summaries},
            )
        )

        async def challenge(
            candidate_id: str,
            finding: Finding,
        ) -> tuple[str, VerificationStatus, dict[str, Any]]:
            async with semaphore:
                return await self._challenge(
                    image,
                    context,
                    candidate_id,
                    finding,
                )

        challenged = await asyncio.gather(
            *(challenge(candidate_id, finding) for candidate_id, finding in candidates.items())
        )
        challenge_records: dict[str, dict[str, Any]] = {}
        for candidate_id, status, record in challenged:
            finding = candidates[candidate_id]
            finding.verification = status
            finding.metadata["skeptic"] = record
            challenge_records[candidate_id] = record
        trace.append(
            TraceEvent(
                stage="consensus.challenges",
                message=f"Skeptic reviewed {len(challenge_records)} claims.",
                data=challenge_records,
            )
        )

        judge_payload = await self._judge(
            image=image,
            context=context,
            role_summaries=role_summaries,
            candidates=candidates,
            challenges=challenge_records,
        )
        accepted_ids = self._valid_ids(
            judge_payload.get("accepted_candidate_ids", []),
            candidates,
        )
        uncertain_ids = self._valid_ids(
            judge_payload.get("uncertain_candidate_ids", []),
            candidates,
        )
        accepted = self._deduplicate([candidates[item] for item in accepted_ids])
        uncertain = self._deduplicate(
            [
                candidates[item]
                for item in uncertain_ids
                if item not in set(accepted_ids)
            ]
        )

        for finding in accepted:
            finding.verification = VerificationStatus.SUPPORTED
        for finding in uncertain:
            finding.verification = VerificationStatus.UNCERTAIN

        raw_verdict = safe_verdict(judge_payload.get("verdict"))
        if accepted:
            verdict = Verdict.ARTIFACT_PRESENT
        elif uncertain:
            verdict = Verdict.UNCERTAIN
        elif raw_verdict is Verdict.NO_ARTIFACT:
            verdict = Verdict.NO_ARTIFACT
        else:
            verdict = Verdict.UNCERTAIN
        confidence = float(judge_payload.get("confidence", 0.5))

        trace.append(
            TraceEvent(
                stage="consensus.judge",
                message=f"Judge returned {verdict.value}.",
                data=judge_payload,
            )
        )
        return CriticResult(
            verdict=verdict,
            confidence=confidence,
            findings=accepted,
            uncertain_findings=uncertain,
            trace=trace,
            metadata={
                "workflow": "adversarial_consensus",
                "roles": [role.name for role in self.config.roles],
                "judge_rationale": str(judge_payload.get("rationale", "")),
            },
        )

    async def _propose(
        self,
        image: ImageAsset,
        context: CriticContext,
        role: CriticRole,
    ) -> tuple[CriticRole, list[Finding], dict[str, Any]]:
        model = self.role_models.get(role.name, self.proposer_model)
        response = await model.complete(
            ModelRequest(
                stage="consensus.propose",
                system_prompt=(
                    f"You are the {role.name} critic. Your exclusive focus is: "
                    f"{role.focus} Be independent, evidence-bound, and concise."
                ),
                user_prompt=(
                    f"Generation prompt: {context.generation_prompt or 'not provided'}\n"
                    f"Optional taxonomy: {list(context.taxonomy) or 'open set'}\n\n"
                    "Inspect the image independently. Return a verdict and at most "
                    f"{self.config.max_findings_per_role} atomic findings. Each finding "
                    "must include region_id, bbox_norm, artifact_type, observation, "
                    "abnormality, confidence, and evidence_ids. Return JSON only."
                ),
                images=(image, *context.reference_images),
                response_schema=_PROPOSAL_SCHEMA,
                metadata={
                    "workflow": "adversarial_consensus",
                    "role": role.name,
                },
            )
        )
        parsed = parse_response(response)
        raw_findings = parsed.get("findings", [])
        findings: list[Finding] = []
        if isinstance(raw_findings, list):
            for raw in raw_findings:
                if not isinstance(raw, Mapping):
                    continue
                findings.append(
                    Finding.from_mapping(
                        raw,
                        default_region_id=f"{image.id}:full",
                        source=f"role:{role.name}",
                    )
                )
        return role, findings, parsed

    async def _challenge(
        self,
        image: ImageAsset,
        context: CriticContext,
        candidate_id: str,
        finding: Finding,
    ) -> tuple[str, VerificationStatus, dict[str, Any]]:
        response = await self.skeptic_model.complete(
            ModelRequest(
                stage="consensus.challenge",
                system_prompt=(
                    "You are a skeptical visual reviewer. Search for ordinary explanations "
                    "such as occlusion, perspective, blur, stylization, reflections, or "
                    "overlapping objects. Do not reject direct pixel evidence merely because "
                    "the artifact is subtle."
                ),
                user_prompt=(
                    f"Candidate id: {candidate_id}\n"
                    f"Generation prompt: {context.generation_prompt or 'not provided'}\n"
                    f"Claim:\n{serialize_json(finding.as_dict())}\n\n"
                    "Return whether the claim is supported, contradicted, or uncertain, "
                    "plus the strongest counterargument and any missing evidence."
                ),
                images=(image, *context.reference_images),
                response_schema=_CHALLENGE_SCHEMA,
                metadata={
                    "workflow": "adversarial_consensus",
                    "candidate_id": candidate_id,
                },
            )
        )
        parsed = parse_response(response)
        raw_status = str(parsed.get("status", "uncertain")).lower()
        confidence = float(parsed.get("confidence", 0.0))
        if (
            raw_status == VerificationStatus.SUPPORTED.value
            and confidence >= self.config.skeptic_threshold
        ):
            status = VerificationStatus.SUPPORTED
        elif raw_status == VerificationStatus.CONTRADICTED.value:
            status = VerificationStatus.CONTRADICTED
        else:
            status = VerificationStatus.UNCERTAIN
        finding.confidence = min(finding.confidence, confidence or finding.confidence)
        return candidate_id, status, dict(parsed)

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
                    "You are the final calibrated judge. Resolve disagreements using visible "
                    "evidence and skeptic reviews. You may only select existing candidate IDs; "
                    "you must not invent a new finding."
                ),
                user_prompt=(
                    f"Generation prompt: {context.generation_prompt or 'not provided'}\n"
                    f"Review packet:\n{serialize_json(packet)}\n\n"
                    "Return accepted candidate IDs, uncertain candidate IDs, the final "
                    "verdict, confidence, and a short rationale. A contradicted candidate "
                    "should not be accepted without an explicit evidence-based reason."
                ),
                images=(image, *context.reference_images),
                response_schema=_JUDGE_SCHEMA,
                metadata={
                    "workflow": "adversarial_consensus",
                    "candidate_count": len(candidates),
                },
            )
        )
        return parse_response(response)

    @staticmethod
    def _valid_ids(values: Any, candidates: Mapping[str, Finding]) -> list[str]:
        if not isinstance(values, list):
            return []
        seen: set[str] = set()
        output: list[str] = []
        for value in values:
            candidate_id = str(value)
            if candidate_id in candidates and candidate_id not in seen:
                seen.add(candidate_id)
                output.append(candidate_id)
        return output

    @staticmethod
    def _deduplicate(findings: list[Finding]) -> list[Finding]:
        seen: set[tuple[Any, ...]] = set()
        output: list[Finding] = []
        for finding in findings:
            key = (
                finding.artifact_type,
                *(round(value, 2) for value in finding.bbox.as_list()),
            )
            if key in seen:
                continue
            seen.add(key)
            output.append(finding)
        return output

