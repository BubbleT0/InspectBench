from __future__ import annotations

import asyncio
import unittest

from critic_workflows import (
    AdversarialConsensusCritic,
    CriticContext,
    EvidenceSeekingAgentCritic,
    ImageAsset,
    ModelRequest,
    ModelResponse,
    PassthroughImageTool,
    SpecialistCascadeCritic,
    Verdict,
    VerificationStatus,
)


class FakeModel:
    def __init__(self) -> None:
        self.calls: list[ModelRequest] = []

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.calls.append(request)
        stage = request.stage

        if stage == "specialist.detect":
            artifact_type = request.metadata["artifact_type"]
            if artifact_type == "human_anatomy":
                return ModelResponse(
                    parsed={
                        "present": True,
                        "confidence": 0.88,
                        "observation": "A hand has six distinct finger-like structures.",
                        "abnormality": "The visible anatomy is implausible.",
                        "bbox_norm": [550, 350, 780, 700],
                    }
                )
            return ModelResponse(
                parsed={
                    "present": False,
                    "confidence": 0.9,
                    "observation": "",
                    "abnormality": "",
                }
            )

        if stage in {"specialist.verify", "evidence.verify"}:
            return ModelResponse(
                parsed={
                    "status": "supported",
                    "confidence": 0.91,
                    "reason": "The claimed structure is directly visible.",
                }
            )

        if stage == "evidence.plan":
            step = int(request.metadata["step"])
            if step == 1:
                return ModelResponse(
                    parsed={
                        "action": "inspect",
                        "reason": "The hand is too small in the global image.",
                        "bbox_norm": [500, 300, 820, 750],
                        "label": "hand",
                    }
                )
            return ModelResponse(
                parsed={
                    "action": "finish",
                    "reason": "The zoomed evidence is sufficient.",
                    "verdict": "artifact_present",
                    "confidence": 0.87,
                    "findings": [
                        {
                            "region_id": "hand",
                            "bbox_norm": [550, 350, 780, 700],
                            "artifact_type": "human_anatomy",
                            "observation": "Six finger-like structures are visible.",
                            "abnormality": "The hand anatomy is implausible.",
                            "confidence": 0.87,
                            "evidence_ids": ["sample:agent:1:crop"],
                        }
                    ],
                }
            )

        if stage == "evidence.force_finish":
            return ModelResponse(
                parsed={
                    "action": "abstain",
                    "reason": "Insufficient evidence.",
                    "verdict": "uncertain",
                    "confidence": 0.3,
                    "findings": [],
                }
            )

        if stage == "consensus.propose":
            if request.metadata["role"] == "anatomy_morphology":
                return ModelResponse(
                    parsed={
                        "verdict": "artifact_present",
                        "confidence": 0.84,
                        "findings": [
                            {
                                "region_id": "hand",
                                "bbox_norm": [550, 350, 780, 700],
                                "artifact_type": "human_anatomy",
                                "observation": "Six finger-like structures are visible.",
                                "abnormality": "The anatomy is implausible.",
                                "confidence": 0.84,
                                "evidence_ids": ["sample"],
                            }
                        ],
                    }
                )
            return ModelResponse(
                parsed={
                    "verdict": "no_artifact",
                    "confidence": 0.75,
                    "findings": [],
                }
            )

        if stage == "consensus.challenge":
            return ModelResponse(
                parsed={
                    "status": "supported",
                    "confidence": 0.8,
                    "counterargument": "Overlapping fingers are possible but not supported.",
                    "missing_evidence": "",
                }
            )

        if stage == "consensus.judge":
            return ModelResponse(
                parsed={
                    "verdict": "artifact_present",
                    "confidence": 0.82,
                    "accepted_candidate_ids": ["c0"],
                    "uncertain_candidate_ids": [],
                    "rationale": "Candidate c0 is supported by visible evidence.",
                }
            )

        raise AssertionError(f"Unexpected stage: {stage}")


class WorkflowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.image = ImageAsset(id="sample", path="placeholder.png")
        self.context = CriticContext(
            generation_prompt="A person holding a cup.",
            taxonomy=("human_anatomy", "object_morphology"),
        )

    def test_specialist_cascade(self) -> None:
        model = FakeModel()
        critic = SpecialistCascadeCritic(
            model=model,
            verifier_model=model,
            image_tool=PassthroughImageTool(),
        )
        result = asyncio.run(critic.critique(self.image, self.context))

        self.assertEqual(result.verdict, Verdict.ARTIFACT_PRESENT)
        self.assertEqual(len(result.findings), 1)
        self.assertEqual(
            result.findings[0].verification,
            VerificationStatus.SUPPORTED,
        )
        self.assertIn("specialist.detect", [call.stage for call in model.calls])

    def test_evidence_seeking_agent(self) -> None:
        model = FakeModel()
        critic = EvidenceSeekingAgentCritic(
            planner_model=model,
            verifier_model=model,
            image_tool=PassthroughImageTool(),
        )
        result = asyncio.run(critic.critique(self.image, self.context))

        self.assertEqual(result.verdict, Verdict.ARTIFACT_PRESENT)
        self.assertEqual(len(result.findings), 1)
        self.assertEqual(len(result.metadata["evidence_ids"]), 2)
        stages = [call.stage for call in model.calls]
        self.assertEqual(stages.count("evidence.plan"), 2)
        self.assertIn("evidence.verify", stages)

    def test_adversarial_consensus(self) -> None:
        model = FakeModel()
        critic = AdversarialConsensusCritic(
            proposer_model=model,
            skeptic_model=model,
            judge_model=model,
        )
        result = asyncio.run(critic.critique(self.image, self.context))

        self.assertEqual(result.verdict, Verdict.ARTIFACT_PRESENT)
        self.assertEqual(len(result.findings), 1)
        self.assertEqual(
            result.findings[0].verification,
            VerificationStatus.SUPPORTED,
        )
        stages = [call.stage for call in model.calls]
        self.assertEqual(stages.count("consensus.propose"), 3)
        self.assertEqual(stages.count("consensus.challenge"), 1)
        self.assertEqual(stages.count("consensus.judge"), 1)


if __name__ == "__main__":
    unittest.main()

