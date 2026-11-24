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
from .v42_fast import (
    V42AdaptiveContentCoverageAuditor,
    V42EditGroundedConsensusCritic,
)
from .v61_streaming import (
    V61AdaptiveContentCoverageAuditor,
    V61EditGroundedConsensusCritic,
)
from .v62_adaptive import V62AdaptiveContentCoverageAuditor
from .v63_selective import V63SelectiveCoverageAuditor
from .v5_conflict_gate import (
    V5AdaptiveContentCoverageAuditor,
    V5EditGroundedConsensusCritic,
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

V42_CORE_ARTIFACT_ROLE = CriticRole(
    "core_artifact",
    "All directly visible major/critical non-text rendering, structural, "
    "topology, relational, and physical errors. Inspect the full image once "
    "and keep spatially independent repaint targets separate.",
)

V63_ARTIFACT_SCOUT_ROLE = CriticRole(
    "artifact_scout",
    "Independently scan the full image for directly visible major/critical "
    "non-text defects. Explicitly count anatomy and object components; trace "
    "attachment, continuity, intersection, occlusion, holding, contact, "
    "support, and physical relations; inspect broken boundaries, fusion, "
    "material failure, and rendering residue. Keep spatially independent "
    "repaint targets separate and ignore aesthetics or minor imperfections.",
)

V63_ARTIFACT_STRUCTURE_SCOUT_ROLE = CriticRole(
    "artifact_structure_scout",
    "Independently inspect the full image for every directly visible "
    "major/critical non-text Artifact. First audit geometry: count anatomy and "
    "object components and trace morphology, attachment, continuity, "
    "intersection, occlusion, holding, contact, support, and physical "
    "relations. Then make a mandatory second pass for broken boundaries, "
    "fusion, melting, material or texture discontinuity, and rendering "
    "residue. Keep independent repaint targets separate and ignore aesthetics "
    "or minor imperfections.",
)

V63_ARTIFACT_RENDERING_SCOUT_ROLE = CriticRole(
    "artifact_rendering_scout",
    "Independently inspect the full image for every directly visible "
    "major/critical non-text Artifact. First audit appearance: check broken "
    "boundaries, fusion, melting, repeated or discontinuous texture, residue, "
    "transparency, and local material failures. Then make a mandatory second "
    "pass for anatomy and component count, morphology, topology, attachment, "
    "interaction, support, and physical relations. Keep independent repaint "
    "targets separate and ignore aesthetics or minor imperfections.",
)


def critic_context() -> CriticContext:
    return CriticContext(taxonomy=TAXONOMY)


def build_critics(
    model,
    image_tool,
    *,
    coverage_mode="grid",
    v42_text_batch_size=2,
    v42_text_crosscheck_batch_size=None,
    v42_artifact_batch_size=4,
    v42_inspection_concurrency=12,
    v42_challenge_batch_size=4,
    v42_challenge_concurrency=4,
    v42_trust_preverified=False,
    v42_deterministic_judge=False,
    v42_combined_inventory=False,
    v42_combined_coverage_audit=False,
    v42_eager_inventory=False,
    v42_text_crosscheck=False,
    v42_decompose_broad_text=False,
    v42_coverage_only=False,
    v42_single_artifact_proposer=False,
    v42_review_original_max_side=0,
    v42_inventory_overview_max_side=0,
    v42_pipeline_version="v4.2-experimental",
    v61_streaming=False,
    v62_adaptive=False,
    v63_selective_review=False,
    v63_entity_safe_review=False,
    v63_artifact_scout=False,
    v63_dual_artifact_scout=False,
    v63_dense_path_latency=False,
):
    if coverage_mode not in {
        "grid",
        "adaptive",
        "harness",
        "harness_v41",
        "harness_v42",
        "harness_v5",
    }:
        raise ValueError(
            "coverage_mode must be 'grid', 'adaptive', 'harness', or "
            "'harness_v41', 'harness_v42', or 'harness_v5'"
        )
    consensus_image_tool = (
        LosslessZoomImageTool(image_tool)
        if coverage_mode in {
            "adaptive",
            "harness",
            "harness_v41",
            "harness_v42",
            "harness_v5",
        }
        else image_tool
    )
    if coverage_mode == "harness_v5":
        coverage_auditor = V5AdaptiveContentCoverageAuditor(
            model,
            consensus_image_tool,
        )
    elif coverage_mode == "harness_v42":
        coverage_auditor_class = (
            V61AdaptiveContentCoverageAuditor
            if v61_streaming
            else V63SelectiveCoverageAuditor
            if v63_selective_review
            else V62AdaptiveContentCoverageAuditor
            if v62_adaptive
            else V42AdaptiveContentCoverageAuditor
        )
        v63_options = (
            {
                "entity_safe_review": v63_entity_safe_review,
                "dense_path_latency": v63_dense_path_latency,
            }
            if coverage_auditor_class is V63SelectiveCoverageAuditor
            else {}
        )
        coverage_auditor = coverage_auditor_class(
            model,
            consensus_image_tool,
            text_batch_size=v42_text_batch_size,
            text_crosscheck_batch_size=v42_text_crosscheck_batch_size,
            artifact_batch_size=v42_artifact_batch_size,
            max_concurrency=v42_inspection_concurrency,
            combined_inventory=v42_combined_inventory,
            combined_coverage_audit=v42_combined_coverage_audit,
            eager_inventory=v42_eager_inventory,
            text_crosscheck=v42_text_crosscheck,
            decompose_broad_text=v42_decompose_broad_text,
            overlap_coverage_inspection=(
                v42_pipeline_version
                in {
                    "v4.1-flash",
                    "v4.2.3-latency-experiment",
                    "v6.2",
                    "v6.3",
                    "v6.3-i",
                }
            ),
            review_original_max_side=v42_review_original_max_side,
            inventory_overview_max_side=(
                v42_inventory_overview_max_side
            ),
            **v63_options,
        )
    elif coverage_mode == "harness_v41":
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
    if coverage_mode == "harness_v5":
        consensus_class = V5EditGroundedConsensusCritic
    elif coverage_mode == "harness_v42":
        consensus_class = (
            V61EditGroundedConsensusCritic
            if v61_streaming
            else V42EditGroundedConsensusCritic
        )
    elif coverage_mode == "harness_v41":
        consensus_class = V41EditGroundedConsensusCritic
    elif coverage_mode == "harness":
        consensus_class = HarnessedEditGroundedConsensusCritic
    else:
        consensus_class = EditGroundedAdversarialConsensusCritic
    consensus_kwargs = (
        {
            "challenge_batch_size": v42_challenge_batch_size,
            "challenge_concurrency": v42_challenge_concurrency,
            "trust_preverified": v42_trust_preverified,
            "deterministic_judge": v42_deterministic_judge,
            "coverage_only": v42_coverage_only,
            "single_artifact_proposer": (
                v42_single_artifact_proposer or v63_artifact_scout
            ),
            "peer_scout_consensus": v63_dual_artifact_scout,
            "review_original_max_side": v42_review_original_max_side,
            "pipeline_version": v42_pipeline_version,
        }
        if coverage_mode == "harness_v42"
        else {}
    )
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
                roles=(
                    V63_ARTIFACT_STRUCTURE_SCOUT_ROLE,
                    V63_ARTIFACT_RENDERING_SCOUT_ROLE,
                    ROLES[-1],
                )
                if (
                    coverage_mode == "harness_v42"
                    and v63_dual_artifact_scout
                )
                else (V63_ARTIFACT_SCOUT_ROLE, ROLES[-1])
                if (
                    coverage_mode == "harness_v42"
                    and v63_artifact_scout
                )
                else (ROLES[-1],)
                if coverage_mode == "harness_v42" and v42_coverage_only
                else (V42_CORE_ARTIFACT_ROLE, ROLES[-1])
                if (
                    coverage_mode == "harness_v42"
                    and v42_single_artifact_proposer
                )
                else ROLES,
                max_findings_per_role=64,
                max_concurrency=4,
                skeptic_threshold=0.55,
            ),
            **consensus_kwargs,
        ),
    }
