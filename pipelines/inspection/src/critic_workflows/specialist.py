from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Mapping

from .adapters import (
    FullImageRegionProposer,
    NullExampleRetriever,
    PassthroughImageTool,
)
from .interfaces import ExampleRetriever, ImageTool, ModelClient, RegionProposer
from .types import (
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
from .utils import parse_response, serialize_json


@dataclass(frozen=True)
class SpecialistDefinition:
    name: str
    artifact_type: str
    instruction: str
    region_labels: tuple[str, ...] = ()


DEFAULT_SPECIALISTS: tuple[SpecialistDefinition, ...] = (
    SpecialistDefinition(
        name="human_anatomy",
        artifact_type="human_anatomy",
        instruction=(
            "Inspect visible human anatomy for malformed, missing, duplicated, fused, "
            "or implausibly connected body parts. Treat occlusion and perspective as "
            "alternative explanations rather than automatic artifacts."
        ),
    ),
    SpecialistDefinition(
        name="object_morphology",
        artifact_type="object_morphology",
        instruction=(
            "Inspect objects for malformed geometry, duplicated or missing components, "
            "broken boundaries, and locally impossible structure."
        ),
    ),
    SpecialistDefinition(
        name="text_rendering",
        artifact_type="text_rendering",
        instruction=(
            "Inspect visible text for illegible glyphs, inconsistent character shapes, "
            "missing strokes, repeated fragments, or impossible layout."
        ),
    ),
    SpecialistDefinition(
        name="physical_consistency",
        artifact_type="physical_consistency",
        instruction=(
            "Inspect contact, support, occlusion, reflection, shadow, scale, and spatial "
            "relations for contradictions with the visible scene."
        ),
    ),
)


@dataclass(frozen=True)
class SpecialistCascadeConfig:
    specialists: tuple[SpecialistDefinition, ...] = DEFAULT_SPECIALISTS
    examples_per_call: int = 4
    max_regions: int = 10
    max_concurrency: int = 6
    proposal_threshold: float = 0.55
    verify_candidates: bool = True
    verifier_threshold: float = 0.55
    crop_padding: float = 0.15


_DETECTION_SCHEMA: Mapping[str, Any] = {
    "type": "object",
    "required": ["present", "confidence", "observation", "abnormality"],
    "properties": {
        "present": {"type": "boolean"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "observation": {"type": "string"},
        "abnormality": {"type": "string"},
        "bbox_norm": {
            "type": "array",
            "items": {"type": "integer", "minimum": 0, "maximum": 1000},
            "minItems": 4,
            "maxItems": 4,
        },
    },
}

_VERIFICATION_SCHEMA: Mapping[str, Any] = {
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


class SpecialistCascadeCritic:
    """ArtifactLens-style fixed scaffold.

    The workflow favors high recall: regions are crossed with narrow specialists,
    optional in-context examples are attached, and candidate findings can be
    checked by a second verifier model before aggregation.
    """

    def __init__(
        self,
        model: ModelClient,
        *,
        region_proposer: RegionProposer | None = None,
        retriever: ExampleRetriever | None = None,
        image_tool: ImageTool | None = None,
        verifier_model: ModelClient | None = None,
        config: SpecialistCascadeConfig | None = None,
    ):
        self.model = model
        self.region_proposer = region_proposer or FullImageRegionProposer()
        self.retriever = retriever or NullExampleRetriever()
        self.image_tool = image_tool or PassthroughImageTool()
        self.verifier_model = verifier_model or model
        self.config = config or SpecialistCascadeConfig()

    async def critique(
        self,
        image: ImageAsset,
        context: CriticContext | None = None,
    ) -> CriticResult:
        context = context or CriticContext()
        labels = tuple(spec.artifact_type for spec in self.config.specialists)
        regions = await self.region_proposer.propose(image, context, labels)
        regions = sorted(regions, key=lambda item: item.score, reverse=True)[
            : self.config.max_regions
        ]
        trace = [
            TraceEvent(
                stage="specialist.regions",
                message=f"Prepared {len(regions)} candidate regions.",
                data={"region_ids": [region.id for region in regions]},
            )
        ]

        semaphore = asyncio.Semaphore(self.config.max_concurrency)

        async def run_one(
            specialist: SpecialistDefinition,
            region: Region,
        ) -> Finding | None:
            if not self._applies(specialist, region):
                return None
            async with semaphore:
                return await self._detect(image, context, specialist, region)

        tasks = [
            asyncio.create_task(run_one(specialist, region))
            for specialist in self.config.specialists
            for region in regions
        ]
        raw_candidates = await asyncio.gather(*tasks)
        candidates = [candidate for candidate in raw_candidates if candidate is not None]
        trace.append(
            TraceEvent(
                stage="specialist.proposals",
                message=f"Specialists proposed {len(candidates)} findings.",
                data={"findings": [candidate.as_dict() for candidate in candidates]},
            )
        )

        if not self.config.verify_candidates:
            verdict = (
                Verdict.ARTIFACT_PRESENT if candidates else Verdict.NO_ARTIFACT
            )
            confidence = (
                max((candidate.confidence for candidate in candidates), default=0.75)
                if candidates
                else 0.75
            )
            return CriticResult(
                verdict=verdict,
                confidence=confidence,
                findings=candidates,
                trace=trace,
                metadata={"workflow": "specialist_cascade", "verified": False},
            )

        verified: list[Finding] = []
        uncertain: list[Finding] = []
        for candidate in candidates:
            status = await self._verify(image, context, candidate)
            trace.append(
                TraceEvent(
                    stage="specialist.verify",
                    message=f"Verifier marked finding as {status.value}.",
                    data={"finding": candidate.as_dict()},
                )
            )
            if status is VerificationStatus.SUPPORTED:
                verified.append(candidate)
            elif status is VerificationStatus.UNCERTAIN:
                uncertain.append(candidate)

        if verified:
            verdict = Verdict.ARTIFACT_PRESENT
            confidence = max(finding.confidence for finding in verified)
        elif uncertain:
            verdict = Verdict.UNCERTAIN
            confidence = max(finding.confidence for finding in uncertain)
        else:
            verdict = Verdict.NO_ARTIFACT
            confidence = 0.75

        return CriticResult(
            verdict=verdict,
            confidence=confidence,
            findings=verified,
            uncertain_findings=uncertain,
            trace=trace,
            metadata={
                "workflow": "specialist_cascade",
                "regions": len(regions),
                "specialists": len(self.config.specialists),
                "verified": True,
            },
        )

    @staticmethod
    def _applies(specialist: SpecialistDefinition, region: Region) -> bool:
        if not specialist.region_labels or region.label == "full_image":
            return True
        return region.label in specialist.region_labels

    async def _detect(
        self,
        image: ImageAsset,
        context: CriticContext,
        specialist: SpecialistDefinition,
        region: Region,
    ) -> Finding | None:
        crop = region.image or await self.image_tool.crop(
            image,
            region,
            padding=self.config.crop_padding,
        )
        examples = await self.retriever.retrieve(
            crop,
            specialist.artifact_type,
            k=self.config.examples_per_call,
        )
        example_summary = [
            {
                "image_index": index + 2,
                "label": example.label,
                "artifact_type": example.artifact_type,
                "note": example.note,
            }
            for index, example in enumerate(examples)
        ]
        user_prompt = (
            f"Target artifact type: {specialist.artifact_type}\n"
            f"Specialist instruction: {specialist.instruction}\n"
            f"Region id: {region.id}\n"
            f"Region bbox (integer [0,1000]): {region.bbox.as_list()}\n"
            f"Generation prompt: {context.generation_prompt or 'not provided'}\n"
            f"In-context examples: {serialize_json(example_summary)}\n\n"
            "Image 0 is the full image. Image 1 is the region crop. Remaining images "
            "are labeled demonstrations in the same order as the summary. Prioritize "
            "recall, but distinguish genuine artifacts from occlusion, perspective, "
            "motion blur, artistic style, and ordinary ambiguity. Return JSON only."
        )
        response = await self.model.complete(
            ModelRequest(
                stage="specialist.detect",
                system_prompt=(
                    "You are a narrow visual artifact specialist. Report only directly "
                    "visible evidence and do not infer invisible structure."
                ),
                user_prompt=user_prompt,
                images=(image, crop, *(example.image for example in examples)),
                response_schema=_DETECTION_SCHEMA,
                metadata={
                    "workflow": "specialist_cascade",
                    "specialist": specialist.name,
                    "artifact_type": specialist.artifact_type,
                    "specialist_instruction": specialist.instruction,
                    "region_id": region.id,
                    "region_bbox": region.bbox.as_list(),
                    "generation_prompt": context.generation_prompt or "",
                },
            )
        )
        parsed = parse_response(response)
        present = bool(parsed.get("present", False))
        confidence = float(parsed.get("confidence", 0.0))
        if not present or confidence < self.config.proposal_threshold:
            return None
        payload = dict(parsed)
        payload["artifact_type"] = specialist.artifact_type
        payload["region_id"] = region.id
        payload["evidence_ids"] = [image.id, crop.id]
        return Finding.from_mapping(
            payload,
            default_region_id=region.id,
            default_bbox=region.bbox,
            source=f"specialist:{specialist.name}",
        )

    async def _verify(
        self,
        image: ImageAsset,
        context: CriticContext,
        finding: Finding,
    ) -> VerificationStatus:
        region = Region(
            id=finding.region_id,
            bbox=finding.bbox,
            label=finding.artifact_type,
        )
        crop = await self.image_tool.crop(
            image,
            region,
            padding=self.config.crop_padding,
        )
        response = await self.verifier_model.complete(
            ModelRequest(
                stage="specialist.verify",
                system_prompt=(
                    "You are an evidence-focused verifier. Challenge the proposed "
                    "artifact and reject it when occlusion, perspective, blur, or style "
                    "is an equally plausible explanation."
                ),
                user_prompt=(
                    f"Generation prompt: {context.generation_prompt or 'not provided'}\n"
                    f"Candidate finding:\n{serialize_json(finding.as_dict())}\n\n"
                    "Image 0 is the full image and image 1 is the claimed region. "
                    "Return supported, contradicted, or uncertain as JSON."
                ),
                images=(image, crop, *context.reference_images),
                response_schema=_VERIFICATION_SCHEMA,
                metadata={
                    "workflow": "specialist_cascade",
                    "finding_source": finding.source,
                    "region_id": finding.region_id,
                },
            )
        )
        parsed = parse_response(response)
        raw_status = str(parsed.get("status", "uncertain")).lower()
        confidence = float(parsed.get("confidence", 0.0))
        if raw_status == VerificationStatus.SUPPORTED.value:
            status = (
                VerificationStatus.SUPPORTED
                if confidence >= self.config.verifier_threshold
                else VerificationStatus.UNCERTAIN
            )
        elif raw_status == VerificationStatus.CONTRADICTED.value:
            status = VerificationStatus.CONTRADICTED
        else:
            status = VerificationStatus.UNCERTAIN
        finding.verification = status
        finding.confidence = confidence
        finding.metadata["verifier_reason"] = str(parsed.get("reason", ""))
        return status
