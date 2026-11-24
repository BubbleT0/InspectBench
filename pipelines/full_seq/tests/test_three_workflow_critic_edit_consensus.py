from __future__ import annotations

import unittest

from critic_workflows import (
    AdversarialConsensusConfig,
    BBox,
    CriticContext,
    CriticRole,
    ImageAsset,
    ModelResponse,
    Region,
    VerificationStatus,
)
from u1_critic_v41.edit_consensus import (
    EditGroundedAdversarialConsensusCritic,
    is_near_full_frame,
)


class QueueModel:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []

    async def complete(self, request):
        self.requests.append(request)
        return ModelResponse(parsed=self.responses.pop(0))


class RecordingImageTool:
    def __init__(self):
        self.calls = []

    async def crop(self, image, region: Region, *, padding=0.0):
        self.calls.append((image, region, padding))
        return ImageAsset(
            id=f"{region.id}:crop",
            data=b"real-crop",
            mime_type="image/png",
            parent_id=image.id,
            bbox=region.bbox,
        )


class EditGroundingPolicyTests(unittest.TestCase):
    def test_rejects_only_four_edge_full_frame_not_large_subjects(self) -> None:
        self.assertTrue(is_near_full_frame(BBox(0, 0, 1000, 1000)))
        self.assertFalse(is_near_full_frame(BBox(20, 40, 980, 960)))
        self.assertFalse(is_near_full_frame(BBox(30, 80, 970, 930)))


class EditGroundedConsensusTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.image = ImageAsset(id="img_a", data=b"image", mime_type="image/jpeg")
        self.context = CriticContext(taxonomy=("structural_artifact",))
        self.role = CriticRole("structure", "检查实体结构")

    def critic(self, proposer, skeptic, image_tool=None):
        return EditGroundedAdversarialConsensusCritic(
            proposer_model=proposer,
            skeptic_model=skeptic,
            judge_model=QueueModel(),
            image_tool=image_tool or RecordingImageTool(),
            config=AdversarialConsensusConfig(
                roles=(self.role,),
                max_findings_per_role=2,
                skeptic_threshold=0.55,
            ),
        )

    async def propose_full_frame(self, critic):
        _, findings, _ = await critic._propose(
            self.image,
            self.context,
            self.role,
        )
        self.assertEqual(len(findings), 1)
        return findings[0]

    async def test_near_full_frame_cannot_be_supported_without_repair(self) -> None:
        proposer = QueueModel(
            {
                "verdict": "artifact_present",
                "confidence": 0.9,
                "findings": [
                    {
                        "region_id": "hand",
                        "bbox_norm": [0, 0, 1000, 1000],
                        "artifact_type": "structural_artifact",
                        "observation": "人物左手有六根手指。",
                        "abnormality": "手指数量异常。",
                        "confidence": 0.9,
                        "defect_present": True,
                        "materiality": "major",
                        "evidence_basis": "visible_only",
                        "evidence_ids": ["img_a"],
                        "edit_target": "人物左手及手腕连接处",
                        "edit_region_rationale": "重绘手掌并保留手腕连接上下文。",
                    }
                ],
            }
        )
        skeptic = QueueModel(
            {
                "status": "supported",
                "confidence": 0.9,
                "counterargument": "没有合理遮挡可解释。",
                "missing_evidence": "",
                "defect_present": True,
                "materiality": "major",
                "evidence_basis": "visible_only",
                "grounding_status": "valid",
                "revised_bbox_norm": None,
                "grounding_rationale": "原框是整图，不能直接用于局部重绘。",
            }
        )
        critic = self.critic(proposer, skeptic)
        finding = await self.propose_full_frame(critic)

        _, status, _ = await critic._challenge(
            self.image,
            self.context,
            "c0",
            finding,
        )

        self.assertEqual(status, VerificationStatus.UNCERTAIN)
        self.assertEqual(finding.metadata["grounding_status"], "invalid")
        self.assertEqual(
            finding.metadata["edit_target"],
            "人物左手及手腕连接处",
        )

    async def test_invalid_proposal_bbox_does_not_abort_other_candidates(self) -> None:
        proposer = QueueModel(
            {
                "verdict": "artifact_present",
                "confidence": 0.8,
                "findings": [
                    {
                        "region_id": "invalid",
                        "bbox_norm": [20, 480, 420, 320],
                        "artifact_type": "structural_artifact",
                        "observation": "无效坐标候选。",
                        "abnormality": "纵坐标顺序错误。",
                        "confidence": 0.7,
                        "defect_present": True,
                        "materiality": "major",
                        "evidence_basis": "visible_only",
                        "evidence_ids": ["img_a"],
                        "edit_target": "无效区域",
                        "edit_region_rationale": "该候选应被丢弃。",
                    },
                    {
                        "region_id": "valid",
                        "bbox_norm": [100, 200, 500, 700],
                        "artifact_type": "structural_artifact",
                        "observation": "人物手部结构异常。",
                        "abnormality": "手指数量不合理。",
                        "confidence": 0.8,
                        "defect_present": True,
                        "materiality": "major",
                        "evidence_basis": "visible_only",
                        "evidence_ids": ["img_a"],
                        "edit_target": "人物手部和手腕",
                        "edit_region_rationale": "覆盖完整手部和连接区域。",
                    },
                ],
            }
        )
        critic = self.critic(proposer, QueueModel())

        _, findings, summary = await critic._propose(
            self.image,
            self.context,
            self.role,
        )

        self.assertEqual([finding.region_id for finding in findings], ["valid"])
        self.assertEqual(summary["invalid_finding_count"], 1)

    async def test_skeptic_replaces_full_frame_with_edit_ready_region(self) -> None:
        proposer = QueueModel(
            {
                "verdict": "artifact_present",
                "confidence": 0.9,
                "findings": [
                    {
                        "region_id": "hand",
                        "bbox_norm": [0, 0, 1000, 1000],
                        "artifact_type": "structural_artifact",
                        "observation": "人物左手有六根手指。",
                        "abnormality": "手指数量异常。",
                        "confidence": 0.9,
                        "defect_present": True,
                        "materiality": "major",
                        "evidence_basis": "visible_only",
                        "evidence_ids": ["img_a"],
                        "edit_target": "人物左手及手腕连接处",
                        "edit_region_rationale": "重绘手掌并保留手腕连接上下文。",
                    }
                ],
            }
        )
        skeptic = QueueModel(
            {
                "status": "supported",
                "confidence": 0.9,
                "counterargument": "没有合理遮挡可解释。",
                "missing_evidence": "",
                "defect_present": True,
                "materiality": "major",
                "evidence_basis": "visible_only",
                "grounding_status": "needs_revision",
                "revised_bbox_norm": [120, 220, 480, 710],
                "grounding_rationale": "修正区域覆盖整只手和手腕连接，并保留少量背景。",
            }
        )
        critic = self.critic(proposer, skeptic)
        finding = await self.propose_full_frame(critic)

        _, status, _ = await critic._challenge(
            self.image,
            self.context,
            "c0",
            finding,
        )

        self.assertEqual(status, VerificationStatus.SUPPORTED)
        self.assertEqual(finding.bbox.as_list(), [120, 220, 480, 710])
        self.assertEqual(finding.metadata["grounding_status"], "valid")
        self.assertEqual(
            finding.metadata["original_bbox_norm"],
            [0, 0, 1000, 1000],
        )
        self.assertEqual(
            finding.metadata["revised_bbox_norm"],
            [120, 220, 480, 710],
        )

    async def test_propose_keeps_only_visible_core_non_text_defects(self) -> None:
        def raw(
            region_id: str,
            *,
            confidence: float = 0.9,
            defect_present: bool = True,
            materiality: str = "major",
            evidence_basis: str = "visible_only",
        ):
            return {
                "region_id": region_id,
                "bbox_norm": [100, 200, 500, 700],
                "artifact_type": "structural_artifact",
                "observation": "局部实体结构可见。",
                "abnormality": "结构存在明显错误。",
                "confidence": confidence,
                "defect_present": defect_present,
                "materiality": materiality,
                "evidence_basis": evidence_basis,
                "evidence_ids": ["img_a"],
                "edit_target": "独立实体",
                "edit_region_rationale": "覆盖完整实体和少量上下文。",
            }

        proposer = QueueModel(
            {
                "verdict": "artifact_present",
                "confidence": 0.9,
                "findings": [
                    raw("core"),
                    raw("normal", defect_present=False, materiality="none"),
                    raw("minor", materiality="minor"),
                    raw("world", evidence_basis="external_knowledge"),
                    raw("low", confidence=0.6),
                    {
                        **raw("text-from-artifact-role"),
                        "artifact_type": "text_visual_integrity",
                    },
                ],
            }
        )

        _, findings, summary = await self.critic(
            proposer,
            QueueModel(),
        )._propose(self.image, self.context, self.role)

        self.assertEqual(
            [finding.region_id for finding in findings],
            ["core"],
        )
        self.assertEqual(summary["policy_rejected_finding_count"], 5)
        self.assertEqual(findings[0].metadata["materiality"], "major")

    async def test_text_crop_review_can_raise_low_initial_confidence(self) -> None:
        image_tool = RecordingImageTool()
        skeptic = QueueModel(
            {
                "status": "supported",
                "confidence": 0.9,
                "counterargument": "裁剪图中笔画断裂清晰可见。",
                "missing_evidence": "",
                "defect_present": True,
                "materiality": "major",
                "evidence_basis": "visible_only",
                "grounding_status": "valid",
                "revised_bbox_norm": None,
                "grounding_rationale": "原框覆盖完整文字实体。",
            }
        )
        critic = self.critic(QueueModel(), skeptic, image_tool)
        finding = self._finding(
            BBox(200, 300, 500, 500),
            "pending",
            artifact_type="text_visual_integrity",
            confidence=0.55,
        )
        finding.source = "role:text_coverage"

        _, status, record = await critic._challenge(
            self.image,
            self.context,
            "c0",
            finding,
        )

        self.assertEqual(status, VerificationStatus.SUPPORTED)
        self.assertEqual(finding.confidence, 0.9)
        self.assertEqual(len(image_tool.calls), 1)
        self.assertEqual(len(skeptic.requests[0].images), 2)
        self.assertEqual(skeptic.requests[0].images[0].id, "img_a")
        self.assertEqual(
            skeptic.requests[0].images[1].parent_id,
            "img_a",
        )
        self.assertEqual(
            skeptic.requests[0].images[1].data,
            b"real-crop",
        )
        self.assertEqual(record["materiality"], "major")

    async def test_crop_review_at_threshold_is_not_a_final_issue(self) -> None:
        skeptic = QueueModel(
            {
                "status": "supported",
                "confidence": 0.6,
                "counterargument": "只有轻微边缘噪点。",
                "missing_evidence": "",
                "defect_present": True,
                "materiality": "major",
                "evidence_basis": "visible_only",
                "grounding_status": "valid",
                "revised_bbox_norm": None,
                "grounding_rationale": "区域有效但证据置信度不足。",
            }
        )
        finding = self._finding(
            BBox(200, 300, 500, 500),
            "pending",
            artifact_type="text_visual_integrity",
            confidence=0.55,
        )
        finding.source = "role:text_coverage"

        _, status, _ = await self.critic(
            QueueModel(),
            skeptic,
        )._challenge(
            self.image,
            self.context,
            "c0",
            finding,
        )

        self.assertEqual(status, VerificationStatus.CONTRADICTED)

    async def test_judge_cannot_accept_grounding_invalid_candidate(self) -> None:
        judge = QueueModel(
            {
                "verdict": "artifact_present",
                "confidence": 0.85,
                "accepted_candidate_ids": ["c0", "c1"],
                "uncertain_candidate_ids": [],
                "rationale": "两个候选问题均有视觉证据。",
            }
        )
        critic = EditGroundedAdversarialConsensusCritic(
            proposer_model=QueueModel(),
            skeptic_model=QueueModel(),
            judge_model=judge,
            image_tool=RecordingImageTool(),
        )
        valid = self._finding(BBox(100, 200, 500, 700), "valid")
        invalid = self._finding(BBox.full_image(), "invalid")

        result = await critic._judge(
            image=self.image,
            context=self.context,
            role_summaries=[],
            candidates={"c0": valid, "c1": invalid},
            challenges={},
        )

        self.assertEqual(result["accepted_candidate_ids"], ["c0"])
        self.assertEqual(result["uncertain_candidate_ids"], [])
        self.assertEqual(result["verdict"], "artifact_present")
        self.assertEqual(
            result["grounding_rejected_candidate_ids"],
            ["c1"],
        )

    async def test_judge_cannot_accept_minor_or_external_candidate(self) -> None:
        judge = QueueModel(
            {
                "verdict": "artifact_present",
                "confidence": 0.9,
                "accepted_candidate_ids": ["c0", "c1", "c2"],
                "uncertain_candidate_ids": ["c1"],
                "rationale": "模型尝试接受全部候选。",
            }
        )
        critic = EditGroundedAdversarialConsensusCritic(
            proposer_model=QueueModel(),
            skeptic_model=QueueModel(),
            judge_model=judge,
            image_tool=RecordingImageTool(),
        )
        core = self._finding(BBox(100, 100, 200, 200), "valid")
        minor = self._finding(BBox(300, 100, 400, 200), "valid")
        external = self._finding(BBox(500, 100, 600, 200), "valid")
        minor.metadata["review_materiality"] = "minor"
        external.metadata["review_evidence_basis"] = "external_knowledge"

        result = await critic._judge(
            image=self.image,
            context=self.context,
            role_summaries=[],
            candidates={"c0": core, "c1": minor, "c2": external},
            challenges={},
        )

        self.assertEqual(result["accepted_candidate_ids"], ["c0"])
        self.assertEqual(result["uncertain_candidate_ids"], [])
        self.assertEqual(
            result["grounding_rejected_candidate_ids"],
            ["c1", "c2"],
        )

    @staticmethod
    def _finding(
        bbox: BBox,
        grounding_status: str,
        *,
        artifact_type: str = "structural_artifact",
        confidence: float = 0.8,
    ):
        from critic_workflows import Finding

        return Finding(
            region_id="region",
            bbox=bbox,
            artifact_type=artifact_type,
            observation="结构异常。",
            abnormality="局部形态不合理。",
            confidence=confidence,
            evidence_ids=["img_a"],
            verification=VerificationStatus.SUPPORTED,
            metadata={
                "edit_target": "独立实体",
                "grounding_status": grounding_status,
                "defect_present": True,
                "materiality": "major",
                "evidence_basis": "visible_only",
                "review_defect_present": True,
                "review_materiality": "major",
                "review_evidence_basis": "visible_only",
            },
        )


if __name__ == "__main__":
    unittest.main()
