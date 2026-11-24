from .adapters import (
    CallableModelClient,
    FullImageRegionProposer,
    GridRegionProposer,
    NullExampleRetriever,
    PassthroughImageTool,
    PillowImageTool,
    StaticExampleRetriever,
)
from .consensus import (
    AdversarialConsensusConfig,
    AdversarialConsensusCritic,
    CriticRole,
    DEFAULT_ROLES,
)
from .evidence_agent import EvidenceSeekingAgentCritic, EvidenceSeekingConfig
from .interfaces import Critic, ExampleRetriever, ImageTool, ModelClient, RegionProposer
from .specialist import (
    DEFAULT_SPECIALISTS,
    SpecialistCascadeConfig,
    SpecialistCascadeCritic,
    SpecialistDefinition,
)
from .types import (
    BBox,
    CriticContext,
    CriticResult,
    Finding,
    ImageAsset,
    ModelRequest,
    ModelResponse,
    Region,
    RetrievedExample,
    TraceEvent,
    Verdict,
    VerificationStatus,
)

__all__ = [
    "AdversarialConsensusConfig",
    "AdversarialConsensusCritic",
    "BBox",
    "CallableModelClient",
    "Critic",
    "CriticContext",
    "CriticRole",
    "CriticResult",
    "DEFAULT_ROLES",
    "DEFAULT_SPECIALISTS",
    "EvidenceSeekingAgentCritic",
    "EvidenceSeekingConfig",
    "ExampleRetriever",
    "Finding",
    "FullImageRegionProposer",
    "GridRegionProposer",
    "ImageAsset",
    "ImageTool",
    "ModelClient",
    "ModelRequest",
    "ModelResponse",
    "NullExampleRetriever",
    "PassthroughImageTool",
    "PillowImageTool",
    "Region",
    "RegionProposer",
    "RetrievedExample",
    "StaticExampleRetriever",
    "SpecialistCascadeConfig",
    "SpecialistCascadeCritic",
    "SpecialistDefinition",
    "TraceEvent",
    "Verdict",
    "VerificationStatus",
]

try:
    from .dspy_specialist import (
        ArtifactSpecialistSignature,
        BinaryMetrics,
        DSPyArtifactSpecialist,
        DSPyComparison,
        DSPySpecialistCompiler,
        DSPySpecialistModelClient,
        DSPySpecialistRecord,
        artifact_detection_metric,
        compare_specialist_programs,
        evaluate_specialist_program,
        image_asset_to_dspy,
    )
except ModuleNotFoundError as exc:
    if exc.name != "dspy":
        raise
else:
    __all__ += [
        "ArtifactSpecialistSignature",
        "BinaryMetrics",
        "DSPyArtifactSpecialist",
        "DSPyComparison",
        "DSPySpecialistCompiler",
        "DSPySpecialistModelClient",
        "DSPySpecialistRecord",
        "artifact_detection_metric",
        "compare_specialist_programs",
        "evaluate_specialist_program",
        "image_asset_to_dspy",
    ]
