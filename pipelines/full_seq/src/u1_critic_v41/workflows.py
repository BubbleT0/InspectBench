from __future__ import annotations

from critic_workflows import (
    AdversarialConsensusConfig,
    CriticContext,
    CriticRole,
    EvidenceSeekingAgentCritic,
    EvidenceSeekingConfig,
    FullImageRegionProposer,
    SpecialistCascadeConfig,
    SpecialistCascadeCritic,
    SpecialistDefinition,
)

from .edit_consensus import EditGroundedAdversarialConsensusCritic
from .adaptive_coverage import AdaptiveContentCoverageAuditor
from .adaptive_image import LosslessZoomImageTool
from .text_coverage import TextCoverageAuditor
from .v4_harness import (
    HarnessedAdaptiveContentCoverageAuditor,
    HarnessedEditGroundedConsensusCritic,
    V41AdaptiveContentCoverageAuditor,
    V41EditGroundedConsensusCritic,
)


TAXONOMY = (
    "rendering_artifact",
    "structural_artifact",
    "topology_artifact",
    "relational_physical_artifact",
    "text_visual_integrity",
    "text_content_fidelity",
)

COMMON_RULE = (
    "Use direct visible evidence and return an integer bbox normalized to "
    "[0,1000], in [x1,y1,x2,y2] order. "
    "Do not confuse occlusion, perspective, blur, reflection, or intentional "
    "style with a generation defect. Abstain when evidence is insufficient. "
)

SPECIALISTS = (
    SpecialistDefinition(
        "rendering",
        "rendering_artifact",
        COMMON_RULE
        + "Check local melting, broken pixels or boundaries, repeated texture, "
        "noise, material discontinuity, and inconsistent local light.",
    ),
    SpecialistDefinition(
        "structure",
        "structural_artifact",
        COMMON_RULE
        + "Check malformed anatomy or objects, duplicated or missing parts, "
        "implausible count, proportion, and component geometry.",
    ),
    SpecialistDefinition(
        "topology",
        "topology_artifact",
        COMMON_RULE
        + "Check impossible fusion, intersection, disconnection, attachment, "
        "continuity, or occlusion ordering.",
    ),
    SpecialistDefinition(
        "relations_physics",
        "relational_physical_artifact",
        COMMON_RULE
        + "Check interaction, support, gravity, scale, spatial relation, shadow, "
        "reflection, and physical consistency.",
    ),
    SpecialistDefinition(
        "text_visual",
        "text_visual_integrity",
        COMMON_RULE
        + "Only when visible text exists, check glyph breakage, fusion, distortion, "
        "missing or extra strokes, and illegibility. Otherwise report absent.",
    ),
    SpecialistDefinition(
        "text_fidelity",
        "text_content_fidelity",
        COMMON_RULE
        + "Only when visible text itself provides a clearly checkable intended "
        "content, check obvious misspelling, missing/extra characters, or gibberish. "
        "No reference text is supplied: never invent target wording.",
    ),
)

ROLES = (
    CriticRole(
        "anatomy_structure_topology",
        "Human/animal anatomy, object morphology, component count, attachment, "
        "intersection, continuity, and occlusion order.",
    ),
    CriticRole(
        "relations_physics",
        "Spatial and physical relations, interaction, support, gravity, scale, "
        "perspective, shadow, and reflection.",
    ),
    CriticRole(
        "rendering",
        "Rendering discontinuities, broken boundaries, repeated textures, "
        "material failures, and other obvious local rendering defects.",
    ),
    CriticRole(
        "text_coverage",
        "Inventory every intended-readable text entity and flag only obvious "
        "visible text corruption or directly checkable internal/format errors.",
    ),
)


def critic_context() -> CriticContext:
    return CriticContext(taxonomy=TAXONOMY)


def build_critics(model, image_tool, *, coverage_mode="grid"):
    if coverage_mode not in {
        "grid",
        "adaptive",
        "harness",
        "harness_v41",
    }:
        raise ValueError(
            "coverage_mode must be 'grid', 'adaptive', 'harness', or "
            "'harness_v41'"
        )
    consensus_image_tool = (
        LosslessZoomImageTool(image_tool)
        if coverage_mode in {"adaptive", "harness", "harness_v41"}
        else image_tool
    )
    if coverage_mode == "harness_v41":
        coverage_auditor = V41AdaptiveContentCoverageAuditor(
            model,
            consensus_image_tool,
        )
    elif coverage_mode == "harness":
        coverage_auditor = HarnessedAdaptiveContentCoverageAuditor(
            model,
            consensus_image_tool,
        )
    elif coverage_mode == "adaptive":
        coverage_auditor = AdaptiveContentCoverageAuditor(
            model,
            consensus_image_tool,
        )
    else:
        coverage_auditor = TextCoverageAuditor(
            model,
            consensus_image_tool,
        )
    if coverage_mode == "harness_v41":
        consensus_class = V41EditGroundedConsensusCritic
    elif coverage_mode == "harness":
        consensus_class = HarnessedEditGroundedConsensusCritic
    else:
        consensus_class = EditGroundedAdversarialConsensusCritic
    return {
        "specialist_cascade": SpecialistCascadeCritic(
            model=model,
            verifier_model=model,
            region_proposer=FullImageRegionProposer(),
            image_tool=image_tool,
            config=SpecialistCascadeConfig(
                specialists=SPECIALISTS,
                max_regions=1,
                max_concurrency=6,
                proposal_threshold=0.45,
                verify_candidates=True,
                verifier_threshold=0.55,
                crop_padding=0.15,
            ),
        ),
        "evidence_seeking_agent": EvidenceSeekingAgentCritic(
            planner_model=model,
            verifier_model=model,
            image_tool=image_tool,
            config=EvidenceSeekingConfig(
                max_steps=4,
                crop_padding=0.12,
                verifier_threshold=0.55,
                max_evidence_images=5,
            ),
        ),
        "adversarial_consensus": consensus_class(
            proposer_model=model,
            skeptic_model=model,
            judge_model=model,
            image_tool=consensus_image_tool,
            text_auditor=coverage_auditor,
            config=AdversarialConsensusConfig(
                roles=ROLES,
                max_findings_per_role=64,
                max_concurrency=4,
                skeptic_threshold=0.55,
            ),
        ),
    }
