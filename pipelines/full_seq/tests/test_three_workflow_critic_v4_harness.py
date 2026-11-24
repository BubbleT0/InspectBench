import unittest
from unittest.mock import AsyncMock, patch

from critic_workflows import (
    BBox,
    CriticContext,
    CriticResult,
    CriticRole,
    Finding,
    ImageAsset,
    ModelResponse,
    Verdict,
)

from u1_critic_v41.adaptive_image import (
    LosslessZoomImageTool,
)
from u1_critic_v41.consolidation import (
    validate_consolidated_result,
)
from u1_critic_v41.edit_consensus import (
    EditGroundedAdversarialConsensusCritic,
)
from u1_critic_v41.v4_harness import (
    HarnessedAdaptiveContentCoverageAuditor,
    HarnessedEditGroundedConsensusCritic,
    apply_coverage_gate,
    build_multiscale_review_bboxes,
    build_channel_coverage_ledger,
    build_coverage_ledger,
    reconcile_semantic_zones,
    summarize_coverage_channels,
    V41AdaptiveContentCoverageAuditor,
)


class PassthroughImageTool:
    async def crop(self, image, region, padding=0.0):
        del padding
        return ImageAsset(
            id=f"{image.id}:{region.id}",
            data=image.data,
            parent_id=image.id,
            bbox=region.bbox,
            metadata={
                "original_crop_size": [100, 100],
                "review_crop_size": [800, 800],
                "zoom_scale": 8,
            },
        )


def region(
    region_id,
    bbox,
    *,
    kind="text",
    unit="complete_title",
    zone_id="main",
):
    return {
        "region_id": region_id,
        "zone_id": zone_id,
        "semantic_group_id": region_id,
        "semantic_unit_kind": unit,
        "bbox_norm": bbox,
        "content_kind": kind,
        "description": region_id,
        "designed_as_text": kind == "text",
        "boundary_complete": True,
    }


class InventoryRetryModel:
    def __init__(self):
        self.requests = []
        self.text_attempts = 0

    async def complete(self, request):
        self.requests.append(request)
        if request.stage.startswith("content.zones.primary"):
            return ModelResponse(
                parsed={
                    "zones": [
                        {
                            "zone_id": "main",
                            "bbox_norm": [0, 0, 1000, 1000],
                            "description": "主要内容区",
                            "expected_content_kinds": [
                                "text",
                                "entity",
                            ],
                        }
                    ],
                    "rationale": "主版面清单。",
                }
            )
        if request.stage.startswith(
            "content.coverage.blind_inventory"
        ):
            return ModelResponse(
                parsed={
                    "zones": [
                        {
                            "zone_id": "blind-main",
                            "bbox_norm": [0, 0, 1000, 1000],
                            "description": "独立观察主要内容区",
                            "expected_content_kinds": [
                                "text",
                                "entity",
                            ],
                        }
                    ],
                    "rationale": "盲审版面清单。",
                }
            )
        if request.stage == "content.inventory.text":
            self.text_attempts += 1
            return ModelResponse(
                parsed={
                    "regions": [
                        region("broken", [1000, 0, 1000, 1000]),
                    ],
                    "inventory_rationale": "坐标退化。",
                }
            )
        if request.stage == "content.inventory.text.harness_retry":
            self.text_attempts += 1
            return ModelResponse(
                parsed={
                    "regions": [
                        region("title", [100, 100, 800, 300]),
                    ],
                    "inventory_rationale": "修正坐标。",
                }
            )
        if request.stage == "content.inventory.entity":
            return ModelResponse(
                parsed={
                    "regions": [
                        region(
                            "subject",
                            [200, 350, 700, 900],
                            kind="entity",
                            unit="entity",
                        ),
                    ],
                    "inventory_rationale": "实体清单。",
                }
            )
        if request.stage == "content.coverage.audit":
            return ModelResponse(
                parsed={
                    "missing_regions": [],
                    "coverage_complete": True,
                    "rationale": "覆盖完整。",
                }
            )
        return ModelResponse(
            parsed={
                "crop_boundary_complete": True,
                "revised_content_bbox_norm": [100, 100, 800, 300],
                "defect_present": False,
                "materiality": "none",
                "evidence_basis": "visible_only",
                "confidence": 0.95,
                "grounding_status": "valid",
                "edit_bbox_full_norm": [100, 100, 800, 300],
                "issues": [],
                "rationale": "未见核心问题。",
            }
        )


class QueueModel:
    def __init__(self, parsed):
        self.parsed = parsed
        self.requests = []

    async def complete(self, request):
        self.requests.append(request)
        return ModelResponse(parsed=self.parsed)


class BlindZoneModel:
    def __init__(self):
        self.requests = []

    async def complete(self, request):
        self.requests.append(request)
        return ModelResponse(
            parsed={
                "zones": [
                    {
                        "zone_id": "blind-left",
                        "bbox_norm": [0, 0, 500, 1000],
                        "description": "左侧文字与主体区",
                        "expected_content_kinds": ["text", "entity"],
                    }
                ],
                "rationale": "独立盲审完成。",
            }
        )


class MissingEntityRecoveryModel(InventoryRetryModel):
    async def complete(self, request):
        if request.stage.startswith("content.inventory.entity"):
            self.requests.append(request)
            return ModelResponse(
                parsed={
                    "regions": [],
                    "inventory_rationale": "实体通道异常为空。",
                }
            )
        if request.stage == "content.inventory.zone_recovery":
            self.requests.append(request)
            return ModelResponse(
                parsed={
                    "regions": [
                        region(
                            "recovered-subject",
                            [200, 350, 700, 900],
                            kind="entity",
                            unit="entity",
                        )
                    ],
                    "confirmed_empty": False,
                    "rationale": "局部恢复找到主体。",
                }
            )
        return await super().complete(request)


class V4HarnessTests(unittest.IsolatedAsyncioTestCase):
    def test_v41_builds_deterministic_detail_and_context_views(self):
        detail, context = build_multiscale_review_bboxes(
            BBox(200, 200, 400, 400),
            BBox(180, 100, 800, 900),
        )

        self.assertEqual(detail.as_list(), [184, 184, 416, 416])
        self.assertEqual(context.as_list(), [130, 130, 470, 470])

    def test_v41_reports_text_and_artifact_coverage_separately(self):
        channels = summarize_coverage_channels(
            [
                {
                    "zone_id": "main",
                    "content_kind": "text",
                    "state": "checked",
                    "region_ids": ["title"],
                },
                {
                    "zone_id": "main",
                    "content_kind": "entity",
                    "state": "failed",
                    "region_ids": ["person"],
                },
            ],
            {
                "text": {"status": "ok"},
                "entity": {"status": "failed"},
                "zone_primary": {"status": "ok"},
                "zone_blind": {"status": "ok"},
            },
        )

        self.assertEqual(channels["text"]["status"], "ok")
        self.assertEqual(channels["artifact"]["status"], "failed")

    def test_v41_entity_and_layout_share_one_artifact_requirement(self):
        ledger = build_channel_coverage_ledger(
            [
                {
                    "zone_id": "photo",
                    "bbox_norm": [0, 0, 500, 500],
                    "description": "拍立得区域",
                    "expected_content_kinds": ["entity", "layout"],
                }
            ],
            [region("person", [50, 50, 450, 450], kind="entity")],
            [
                {
                    "region_id": "person",
                    "content_kind": "entity",
                    "status": "checked",
                }
            ],
        )

        self.assertEqual(len(ledger), 1)
        self.assertEqual(ledger[0]["content_kind"], "artifact")
        self.assertEqual(ledger[0]["state"], "checked")

    async def test_v41_rejects_crop_only_candidate_and_sends_three_views(self):
        model = QueueModel(
            {
                "crop_boundary_complete": False,
                "revised_content_bbox_norm": [100, 100, 500, 300],
                "defect_present": True,
                "materiality": "major",
                "evidence_basis": "visible_only",
                "confidence": 0.9,
                "grounding_status": "valid",
                "edit_bbox_full_norm": [100, 100, 500, 300],
                "issues": [
                    {
                        "artifact_type": "text_visual_integrity",
                        "observation": "裁剪边缘看似缺字。",
                        "abnormality": "只在细节图中成立。",
                        "confidence": 0.9,
                        "evidence_kind": "crop_boundary_artifact",
                    }
                ],
                "confirmed_in_original": False,
                "confirmed_in_context": False,
                "rationale": "无法在原图确认。",
            }
        )
        auditor = V41AdaptiveContentCoverageAuditor(
            model,
            PassthroughImageTool(),
        )
        parsed, _ = await auditor._request_inspection(
            ImageAsset(id="anonymous", data=b"pixels"),
            region("title", [100, 100, 500, 300]),
            BBox(100, 100, 500, 300),
            BBox(80, 80, 520, 320),
            inspection_role="text_visual",
            boundary_retry_count=0,
        )

        self.assertEqual(len(model.requests[0].images), 3)
        self.assertEqual(
            model.requests[0].stage,
            "content.inspect.multiscale",
        )
        self.assertFalse(parsed["defect_present"])
        self.assertEqual(parsed["issues"], [])
        self.assertTrue(parsed["crop_boundary_complete"])

    async def test_v41_accepts_explicitly_confirmed_empty_inventory(self):
        class EmptyInventoryModel:
            async def complete(self, request):
                return ModelResponse(
                    parsed={
                        "regions": [],
                        "confirmed_empty": True,
                        "inventory_rationale": "原图没有文字。",
                    }
                )

        auditor = V41AdaptiveContentCoverageAuditor(
            EmptyInventoryModel(),
            PassthroughImageTool(),
        )
        regions = await auditor._inventory(
            ImageAsset(id="anonymous", data=b"pixels"),
            "text",
        )

        self.assertEqual(regions, [])
        self.assertEqual(auditor._inventory_health["text"]["status"], "ok")
        self.assertTrue(
            auditor._inventory_health["text"]["confirmed_empty"]
        )

    def test_reconciled_zones_union_independent_content_expectations(self):
        zones = reconcile_semantic_zones(
            [
                {
                    "zone_id": "primary-left",
                    "bbox_norm": [0, 0, 500, 1000],
                    "description": "左侧信息区",
                    "expected_content_kinds": ["text"],
                }
            ],
            [
                {
                    "zone_id": "blind-left",
                    "bbox_norm": [0, 0, 520, 1000],
                    "description": "左侧主体区",
                    "expected_content_kinds": ["entity"],
                }
            ],
        )

        self.assertEqual(len(zones), 1)
        self.assertEqual(
            zones[0]["expected_content_kinds"],
            ["entity", "text"],
        )
        self.assertEqual(
            zones[0]["coverage_sources"],
            ["primary", "blind"],
        )

    def test_coverage_ledger_requires_every_expected_zone_channel(self):
        zones = [
            {
                "zone_id": "left",
                "bbox_norm": [0, 0, 500, 1000],
                "description": "左侧信息区",
                "expected_content_kinds": ["text", "entity"],
            }
        ]
        regions = [
            region("title", [50, 50, 450, 250], kind="text"),
        ]
        outcomes = [
            {
                "region_id": "title",
                "content_kind": "text",
                "status": "checked",
            }
        ]

        ledger = build_coverage_ledger(zones, regions, outcomes)

        self.assertEqual(
            {
                (item["content_kind"], item["state"])
                for item in ledger
            },
            {("text", "checked"), ("entity", "pending")},
        )

    def test_failed_coverage_forces_uncertain_even_with_partial_findings(self):
        summary = apply_coverage_gate(
            {
                "verdict": "artifact_present",
                "confidence": 0.93,
                "suspicious_region_count": 11,
            },
            ledger=[
                {
                    "zone_id": "main",
                    "content_kind": "text",
                    "state": "checked",
                },
                {
                    "zone_id": "main",
                    "content_kind": "entity",
                    "state": "failed",
                },
            ],
            inventory_health={
                "text": {"status": "ok"},
                "entity": {"status": "failed"},
            },
        )

        self.assertEqual(summary["quality_verdict"], "artifact_present")
        self.assertEqual(summary["coverage_status"], "failed")
        self.assertEqual(summary["verdict"], "uncertain")
        self.assertEqual(summary["suspicious_region_count"], 11)
        self.assertTrue(summary["findings_non_exhaustive"])

    async def test_blind_zone_inventory_cannot_see_primary_inventory(self):
        model = BlindZoneModel()
        auditor = HarnessedAdaptiveContentCoverageAuditor(
            model,
            LosslessZoomImageTool(PassthroughImageTool()),
        )
        auditor._primary_zones = [
            {
                "zone_id": "secret",
                "description": "SECRET_PRIMARY_REGION",
                "bbox_norm": [0, 0, 1000, 1000],
                "expected_content_kinds": ["text"],
            }
        ]

        zones = await auditor._zone_inventory(
            ImageAsset(id="anonymous", data=b"pixels"),
            blind=True,
        )

        self.assertEqual(len(zones), 1)
        request = model.requests[0]
        request_text = (
            request.system_prompt
            + request.user_prompt
            + str(dict(request.metadata))
        )
        self.assertNotIn("SECRET_PRIMARY_REGION", request_text)
        self.assertEqual(request.stage, "content.coverage.blind_inventory")

    async def test_failed_channel_runs_targeted_zone_recovery_but_stays_partial(
        self,
    ):
        model = MissingEntityRecoveryModel()
        auditor = HarnessedAdaptiveContentCoverageAuditor(
            model,
            LosslessZoomImageTool(PassthroughImageTool()),
        )

        _, summary = await auditor.audit(
            ImageAsset(id="anonymous", data=b"pixels"),
            CriticContext(),
        )

        self.assertIn(
            "content.inventory.zone_recovery",
            [request.stage for request in model.requests],
        )
        self.assertIn(
            "recovered-subject",
            {
                item["region_id"]
                for item in summary["content_regions"]
            },
        )
        self.assertEqual(summary["coverage_status"], "failed")
        self.assertEqual(summary["verdict"], "uncertain")
        self.assertTrue(summary["findings_non_exhaustive"])

    async def test_final_consensus_preserves_findings_but_marks_partial_uncertain(
        self,
    ):
        model = QueueModel({})
        text_auditor = type(
            "Auditor",
            (),
            {
                "last_summary": {
                    "coverage_status": "failed",
                    "coverage_ledger": [
                        {
                            "zone_id": "main",
                            "content_kind": "text",
                            "state": "failed",
                        }
                    ],
                    "semantic_zones": [],
                    "harness_inventory_health": {
                        "text": {"status": "failed"}
                    },
                    "harness_abstention_reason": "覆盖不完整。",
                }
            },
        )()
        critic = HarnessedEditGroundedConsensusCritic(
            proposer_model=model,
            skeptic_model=model,
            judge_model=model,
            image_tool=PassthroughImageTool(),
            text_auditor=text_auditor,
        )
        finding = Finding(
            region_id="known",
            bbox=BBox(100, 100, 300, 300),
            artifact_type="text_visual_integrity",
            observation="已确认乱码。",
            abnormality="字符无法辨认。",
            confidence=0.9,
        )
        parent_result = CriticResult(
            verdict=Verdict.ARTIFACT_PRESENT,
            confidence=0.9,
            findings=[finding],
        )

        with patch.object(
            EditGroundedAdversarialConsensusCritic,
            "critique",
            new=AsyncMock(return_value=parent_result),
        ):
            result = await critic.critique(
                ImageAsset(id="anonymous", data=b"pixels"),
                CriticContext(),
            )

        self.assertEqual(result.verdict, Verdict.UNCERTAIN)
        self.assertEqual(result.findings, [finding])
        self.assertTrue(result.metadata["findings_non_exhaustive"])
        self.assertEqual(
            result.metadata["quality_verdict"],
            "artifact_present",
        )

    def test_nested_container_around_multiple_edit_regions_is_rejected(self):
        def edit_region(region_id, bbox, source_id):
            return {
                "edit_region_id": region_id,
                "bbox_norm": bbox,
                "edit_target": "需要局部重绘的文字区域",
                "edit_region_rationale": "该区域对应一次独立局部重绘。",
                "source_finding_ids": [source_id],
                "issues": [
                    {
                        "artifact_type": "text_visual_integrity",
                        "observation": "文字明显扭曲，无法正常辨认。",
                        "abnormality": "字符笔画断裂并发生粘连。",
                        "confidence": 0.9,
                        "materiality": "major",
                        "source_finding_ids": [source_id],
                    }
                ],
                "confidence": 0.9,
                "verification": "supported",
            }

        sources = [
            {
                "source_finding_id": source_id,
                "bbox_norm": bbox,
                "artifact_type": "text_visual_integrity",
                "metadata": {"content_kind": "text"},
            }
            for source_id, bbox in (
                ("f000", [200, 100, 800, 800]),
                ("f001", [300, 200, 450, 300]),
                ("f002", [550, 400, 700, 500]),
            )
        ]
        value = {
            "verdict": "artifact_present",
            "confidence": 0.9,
            "edit_regions": [
                edit_region("outer", [200, 100, 800, 800], "f000"),
                edit_region("inner-a", [300, 200, 450, 300], "f001"),
                edit_region("inner-b", [550, 400, 700, 500], "f002"),
            ],
            "discarded_finding_ids": [],
            "discard_reasons": {},
            "rationale": "整编完成，所有问题均已映射到编辑区域。",
        }

        errors = validate_consolidated_result(
            value,
            sources,
            reject_nested_containers=True,
        )

        self.assertTrue(
            any("redundant nested edit container" in error for error in errors)
        )

    async def test_degenerate_inventory_triggers_semantic_retry(self):
        model = InventoryRetryModel()
        auditor = HarnessedAdaptiveContentCoverageAuditor(
            model,
            LosslessZoomImageTool(PassthroughImageTool()),
            max_concurrency=4,
        )

        _, summary = await auditor.audit(
            ImageAsset(id="anonymous", data=b"pixels"),
            CriticContext(),
        )

        self.assertEqual(model.text_attempts, 2)
        self.assertEqual(summary["harness_coverage_status"], "ok")
        self.assertEqual(summary["harness_inventory_retries"], 1)
        self.assertGreater(summary["inspected_region_count"], 0)

    def test_text_regions_receive_independent_negative_verification(self):
        auditor = HarnessedAdaptiveContentCoverageAuditor(
            QueueModel({}),
            LosslessZoomImageTool(PassthroughImageTool()),
        )

        self.assertEqual(
            auditor._inspection_roles({"content_kind": "text"}),
            (
                "text_visual",
                "text_visual_negative_verification",
                "text_fidelity",
            ),
        )

    async def test_cross_role_text_evidence_is_rerouted_not_discarded(self):
        proposal = {
            "verdict": "artifact_present",
            "confidence": 0.94,
            "findings": [
                {
                    "region_id": "text-block",
                    "bbox_norm": [100, 200, 400, 300],
                    "artifact_type": "text_visual_integrity",
                    "observation": "正文为明显乱码。",
                    "abnormality": "字符结构破碎且无法辨认。",
                    "confidence": 0.94,
                    "defect_present": True,
                    "materiality": "major",
                    "evidence_basis": "visible_only",
                    "evidence_ids": ["text-block"],
                    "edit_target": "正文文字块",
                    "edit_region_rationale": "覆盖完整正文语义单元。",
                }
            ],
        }
        model = QueueModel(proposal)
        critic = HarnessedEditGroundedConsensusCritic(
            proposer_model=model,
            skeptic_model=model,
            judge_model=model,
            image_tool=PassthroughImageTool(),
        )

        _, findings, summary = await critic._propose(
            ImageAsset(id="anonymous", data=b"pixels"),
            CriticContext(),
            CriticRole("rendering", "rendering defects"),
        )

        self.assertEqual(len(findings), 1)
        self.assertTrue(findings[0].metadata["harness_rerouted"])
        self.assertEqual(
            findings[0].source,
            "role:rerouted_text_evidence",
        )
        self.assertEqual(summary["harness_rerouted_finding_count"], 1)

    async def test_contradictory_empty_terminal_state_abstains(self):
        judge = QueueModel(
            {
                "verdict": "no_artifact",
                "confidence": 0.9,
                "accepted_candidate_ids": [],
                "uncertain_candidate_ids": [],
                "rationale": "没有候选。",
            }
        )
        critic = HarnessedEditGroundedConsensusCritic(
            proposer_model=judge,
            skeptic_model=judge,
            judge_model=judge,
            image_tool=PassthroughImageTool(),
        )

        result = await critic._judge(
            image=ImageAsset(id="anonymous", data=b"pixels"),
            context=CriticContext(),
            role_summaries=[
                {
                    "role": "text_coverage",
                    "verdict": "artifact_present",
                    "confidence": 0.9,
                    "candidate_ids": [],
                }
            ],
            candidates={},
            challenges={},
        )

        self.assertEqual(result["verdict"], "uncertain")
        self.assertIn("harness_abstention_reason", result)


if __name__ == "__main__":
    unittest.main()
