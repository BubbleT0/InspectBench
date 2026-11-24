from __future__ import annotations

import unittest

from critic_workflows import (
    CriticContext,
    ImageAsset,
    ModelResponse,
    PassthroughImageTool,
)

from u1_critic_v41.adaptive_coverage import (
    AdaptiveContentCoverageAuditor,
    _normalize_regions,
    deduplicate_content_regions,
    suppress_aggregate_content_regions,
)
from u1_critic_v41.adaptive_image import (
    LosslessZoomImageTool,
)


def content_region(
    region_id,
    bbox_norm,
    *,
    kind="text",
    description="文字块",
    semantic_group_id=None,
    semantic_unit_kind=None,
    boundary_complete=True,
    designed_as_text=None,
):
    return {
        "region_id": region_id,
        "semantic_group_id": semantic_group_id or region_id,
        "semantic_unit_kind": semantic_unit_kind or (
            "independent_label" if kind == "text" else "entity"
        ),
        "bbox_norm": bbox_norm,
        "content_kind": kind,
        "description": description,
        "designed_as_text": (
            kind == "text"
            if designed_as_text is None
            else designed_as_text
        ),
        "boundary_complete": boundary_complete,
        "inventory_sources": [region_id],
    }


class AdaptiveCoverageGeometryTests(unittest.TestCase):
    def test_merges_duplicate_entity_but_keeps_spatially_separate_labels(
        self,
    ) -> None:
        regions = [
            content_region(
                "title-full",
                [100, 100, 400, 200],
                description="主标题",
            ),
            content_region(
                "title-repeat",
                [110, 100, 390, 200],
                description="主标题",
            ),
            content_region(
                "label-right",
                [600, 100, 700, 200],
                description="右侧标签",
            ),
        ]

        output = deduplicate_content_regions(regions)

        self.assertEqual(len(output), 2)
        self.assertEqual(
            output[0]["bbox_norm"],
            [100, 100, 400, 200],
        )
        self.assertEqual(
            output[0]["inventory_sources"],
            ["title-full", "title-repeat"],
        )
        self.assertEqual(output[1]["region_id"], "label-right")

    def test_row_group_does_not_merge_spatially_separate_entities(
        self,
    ) -> None:
        regions = [
            content_region(
                "card-left",
                [100, 200, 200, 500],
                kind="entity",
                description="左侧人物卡片",
                semantic_group_id="row-1",
            ),
            content_region(
                "card-right",
                [700, 200, 800, 500],
                kind="entity",
                description="右侧人物卡片",
                semantic_group_id="row-1",
            ),
        ]

        output = deduplicate_content_regions(regions)

        self.assertEqual(
            [item["region_id"] for item in output],
            ["card-left", "card-right"],
        )

    def test_zero_area_inventory_box_is_rejected(self) -> None:
        regions = _normalize_regions(
            [
                content_region(
                    "invalid-whole-image",
                    [1000, 0, 1000, 1000],
                    description="错误的整图文字框",
                ),
                content_region(
                    "valid-title",
                    [100, 100, 800, 200],
                    description="完整标题",
                    semantic_unit_kind="complete_title",
                ),
            ],
            source="test",
        )

        self.assertEqual(
            [item["region_id"] for item in regions],
            ["valid-title"],
        )

    def test_removes_broad_container_over_independent_children(self) -> None:
        regions = [
            content_region(
                "whole-panel",
                [100, 100, 900, 900],
                kind="layout",
                description="整张信息面板",
            ),
            content_region(
                "label-left",
                [200, 200, 300, 300],
                description="左标签",
            ),
            content_region(
                "label-right",
                [700, 700, 800, 800],
                description="右标签",
            ),
        ]

        output = suppress_aggregate_content_regions(regions)

        self.assertEqual(
            [item["region_id"] for item in output],
            ["label-left", "label-right"],
        )

    def test_merges_separate_lines_in_one_semantic_card_unit(self) -> None:
        regions = [
            content_region(
                "card-title",
                [100, 100, 500, 200],
                description="卡片标题",
                semantic_group_id="card-a",
                semantic_unit_kind="card",
            ),
            content_region(
                "card-body",
                [100, 300, 500, 700],
                description="卡片正文",
                semantic_group_id="card-a",
                semantic_unit_kind="card",
            ),
        ]

        output = deduplicate_content_regions(regions)

        self.assertEqual(len(output), 1)
        self.assertEqual(output[0]["semantic_group_id"], "card-a")
        self.assertEqual(
            output[0]["bbox_norm"],
            [100, 100, 500, 700],
        )
        self.assertEqual(
            output[0]["inventory_sources"],
            ["card-title", "card-body"],
        )

    def test_one_detail_does_not_delete_complete_entity(self) -> None:
        regions = [
            content_region(
                "person",
                [100, 100, 800, 900],
                kind="entity",
                description="完整人物",
            ),
            content_region(
                "person-hand",
                [200, 500, 300, 650],
                kind="entity",
                description="人物手部",
            ),
        ]

        output = suppress_aggregate_content_regions(regions)

        self.assertEqual(len(output), 2)

    def test_complete_semantic_text_unit_supersedes_line_fragments(
        self,
    ) -> None:
        regions = [
            content_region(
                "complete-card",
                [100, 100, 900, 800],
                description="完整信息卡片文字区",
                semantic_group_id="card-complete",
                semantic_unit_kind="card",
            ),
            content_region(
                "title-line",
                [200, 200, 700, 300],
                description="卡片标题碎片",
                semantic_group_id="title-fragment",
                semantic_unit_kind="other",
            ),
            content_region(
                "body-line",
                [200, 400, 800, 500],
                description="卡片正文碎片",
                semantic_group_id="body-fragment",
                semantic_unit_kind="other",
            ),
        ]

        output = suppress_aggregate_content_regions(regions)

        self.assertEqual(
            [item["region_id"] for item in output],
            ["complete-card"],
        )


class AdaptiveCoverageModel:
    def __init__(self) -> None:
        self.requests = []

    async def complete(self, request):
        self.requests.append(request)
        if request.stage == "content.inventory.text":
            return ModelResponse(
                parsed={
                    "regions": [
                        content_region(
                            "external-title-fragment",
                            [50, 50, 250, 120],
                            description="英文片名片段",
                        ),
                        content_region(
                            "direct-format",
                            [100, 200, 300, 270],
                            description="时间文字",
                        ),
                        content_region(
                            "stylized-clear",
                            [100, 320, 350, 400],
                            description="清晰艺术字",
                        ),
                        content_region(
                            "glyph-corrupt",
                            [200, 500, 400, 600],
                            description="扭曲标题",
                        ),
                    ],
                    "inventory_rationale": "已枚举全部可读文字块。",
                }
            )
        if request.stage == "content.inventory.entity":
            return ModelResponse(
                parsed={
                    "regions": [
                        content_region(
                            "main-person",
                            [550, 100, 900, 800],
                            kind="entity",
                            description="主要人物",
                        )
                    ],
                    "inventory_rationale": "已枚举主要独立实体。",
                }
            )
        if request.stage == "content.coverage.audit":
            return ModelResponse(
                parsed={
                    "missing_regions": [
                        content_region(
                            "missed-label",
                            [720, 820, 860, 880],
                            description="遗漏的小标签",
                        )
                    ],
                    "coverage_complete": False,
                    "rationale": "补充一个独立标签。",
                }
            )
        region_id = request.metadata["content_region_id"]
        inspection_role = request.metadata.get("inspection_role")
        return ModelResponse(
            parsed=self.inspection(region_id, inspection_role)
        )

    @staticmethod
    def inspection(region_id, inspection_role=None):
        base = {
            "crop_boundary_complete": True,
            "revised_content_bbox_norm": [0, 0, 1000, 1000],
            "defect_present": False,
            "materiality": "none",
            "evidence_basis": "visible_only",
            "confidence": 0.95,
            "grounding_status": "valid",
            "edit_bbox_full_norm": [0, 0, 1000, 1000],
            "issues": [],
            "rationale": "未见明显核心错误。",
        }
        if (
            region_id == "external-title-fragment"
            and inspection_role == "text_fidelity"
        ):
            return {
                **base,
                "defect_present": True,
                "materiality": "major",
                "confidence": 0.92,
                "issues": [
                    {
                        "artifact_type": "text_content_fidelity",
                        "observation": "标题看起来缺少首字母。",
                        "abnormality": "根据已知片名应补一个字母。",
                        "confidence": 0.92,
                        "evidence_kind": "external_completion",
                    }
                ],
            }
        if (
            region_id == "direct-format"
            and inspection_role == "text_fidelity"
        ):
            return {
                **base,
                "defect_present": True,
                "materiality": "major",
                "confidence": 0.9,
                "edit_bbox_full_norm": [150, 210, 200, 240],
                "issues": [
                    {
                        "artifact_type": "text_content_fidelity",
                        "observation": "时间写成 23::45。",
                        "abnormality": "同一时间块内出现连续双冒号。",
                        "confidence": 0.9,
                        "evidence_kind": "direct_format",
                    }
                ],
            }
        if (
            region_id == "glyph-corrupt"
            and inspection_role == "text_visual"
        ):
            return {
                **base,
                "defect_present": True,
                "materiality": "major",
                "confidence": 0.88,
                "issues": [
                    {
                        "artifact_type": "text_visual_integrity",
                        "observation": "标题中多个字形明显扭曲粘连。",
                        "abnormality": "字符在原图像素中无法辨认。",
                        "confidence": 0.88,
                        "evidence_kind": "visible_corruption",
                    },
                    {
                        "artifact_type": "rendering_artifact",
                        "observation": "标题底板边缘同时发生断裂。",
                        "abnormality": "边缘出现不连续缺口。",
                        "confidence": 0.82,
                        "evidence_kind": "visible_corruption",
                    },
                ],
            }
        if (
            region_id == "missed-label"
            and inspection_role == "text_visual"
        ):
            return {
                **base,
                "defect_present": True,
                "materiality": "minor",
                "confidence": 0.59,
                "evidence_basis": "aesthetic_preference",
                "issues": [
                    {
                        "artifact_type": "text_visual_integrity",
                        "observation": "字号略小。",
                        "abnormality": "可以更美观。",
                        "confidence": 0.59,
                        "evidence_kind": "aesthetic_preference",
                    }
                ],
            }
        return base


class AdaptiveContentCoverageAuditorTests(
    unittest.IsolatedAsyncioTestCase
):
    async def test_inventories_all_content_and_inspects_every_region(self) -> None:
        model = AdaptiveCoverageModel()
        auditor = AdaptiveContentCoverageAuditor(
            model,
            LosslessZoomImageTool(PassthroughImageTool()),
            max_concurrency=4,
        )

        findings, summary = await auditor.audit(
            ImageAsset(id="anonymous", data=b"pixels"),
            CriticContext(),
        )

        stages = [request.stage for request in model.requests]
        self.assertEqual(stages.count("content.inventory.text"), 1)
        self.assertEqual(stages.count("content.inventory.entity"), 1)
        self.assertEqual(stages.count("content.coverage.audit"), 1)
        self.assertEqual(stages.count("content.inspect"), 12)
        self.assertNotIn("text.coverage.tile", stages)
        self.assertEqual(summary["inventory_region_count"], 6)
        self.assertEqual(summary["inspected_region_count"], 6)
        self.assertEqual(len(findings), 2)
        self.assertEqual(
            {finding.region_id for finding in findings},
            {"direct-format", "glyph-corrupt"},
        )
        self.assertTrue(
            all(finding.source == "role:text_coverage" for finding in findings)
        )

    async def test_rejects_external_completion_and_preserves_multi_issue_region(
        self,
    ) -> None:
        model = AdaptiveCoverageModel()
        auditor = AdaptiveContentCoverageAuditor(
            model,
            LosslessZoomImageTool(PassthroughImageTool()),
        )

        findings, _ = await auditor.audit(
            ImageAsset(id="anonymous", data=b"pixels"),
            CriticContext(),
        )

        self.assertNotIn(
            "external-title-fragment",
            {finding.region_id for finding in findings},
        )
        glyph = next(
            finding
            for finding in findings
            if finding.region_id == "glyph-corrupt"
        )
        self.assertEqual(len(glyph.metadata["issues"]), 2)
        self.assertIn("字符在原图像素中无法辨认", glyph.abnormality)
        self.assertIn("边缘出现不连续缺口", glyph.abnormality)

    async def test_text_finding_uses_complete_semantic_unit_bbox(
        self,
    ) -> None:
        model = AdaptiveCoverageModel()
        auditor = AdaptiveContentCoverageAuditor(
            model,
            LosslessZoomImageTool(PassthroughImageTool()),
        )

        findings, _ = await auditor.audit(
            ImageAsset(id="anonymous", data=b"pixels"),
            CriticContext(),
        )

        direct = next(
            finding
            for finding in findings
            if finding.region_id == "direct-format"
        )
        self.assertEqual(
            direct.bbox.as_list(),
            [100, 200, 300, 270],
        )
        self.assertEqual(
            direct.metadata["content_bbox_norm"],
            [100, 200, 300, 270],
        )
        self.assertEqual(
            direct.metadata["review_bbox_norm"],
            [84, 194, 316, 276],
        )

    async def test_unreadable_text_flag_cannot_remove_text_semantics(
        self,
    ) -> None:
        model = AdaptiveCoverageModel()
        original_complete = model.complete

        async def complete(request):
            response = await original_complete(request)
            if request.stage != "content.inventory.text":
                return response
            parsed = dict(response.parsed)
            parsed["regions"] = [
                {
                    **region,
                    "designed_as_text": False,
                }
                for region in parsed["regions"]
            ]
            return ModelResponse(parsed=parsed)

        model.complete = complete
        auditor = AdaptiveContentCoverageAuditor(
            model,
            LosslessZoomImageTool(PassthroughImageTool()),
        )

        findings, summary = await auditor.audit(
            ImageAsset(id="anonymous", data=b"pixels"),
            CriticContext(),
        )

        self.assertIn(
            "glyph-corrupt",
            {finding.region_id for finding in findings},
        )
        glyph_region = next(
            region
            for region in summary["content_regions"]
            if region["region_id"] == "glyph-corrupt"
        )
        self.assertTrue(glyph_region["designed_as_text"])

    async def test_independent_text_inspectors_merge_into_one_edit_region(
        self,
    ) -> None:
        model = AdaptiveCoverageModel()
        auditor = AdaptiveContentCoverageAuditor(
            model,
            LosslessZoomImageTool(PassthroughImageTool()),
        )

        findings, _ = await auditor.audit(
            ImageAsset(id="anonymous", data=b"pixels"),
            CriticContext(),
        )

        glyph = next(
            finding
            for finding in findings
            if finding.region_id == "glyph-corrupt"
        )
        glyph_requests = [
            request
            for request in model.requests
            if (
                request.stage == "content.inspect"
                and request.metadata["content_region_id"]
                == "glyph-corrupt"
            )
        ]
        self.assertEqual(
            {
                request.metadata.get("inspection_role")
                for request in glyph_requests
            },
            {"text_visual", "text_fidelity"},
        )
        self.assertEqual(len(glyph_requests), 2)
        self.assertEqual(
            glyph.bbox.as_list(),
            [200, 500, 400, 600],
        )
        self.assertEqual(len(glyph.metadata["issues"]), 2)


class BoundaryRetryModel:
    def __init__(self) -> None:
        self.requests = []
        self.inspect_count = 0

    async def complete(self, request):
        self.requests.append(request)
        if request.stage == "content.inventory.text":
            return ModelResponse(
                parsed={
                    "regions": [
                        content_region(
                            "complete-title",
                            [300, 100, 500, 200],
                            description="完整主标题",
                            semantic_unit_kind="complete_title",
                        )
                    ],
                    "inventory_rationale": "完整标题。",
                }
            )
        if request.stage == "content.inventory.entity":
            return ModelResponse(
                parsed={
                    "regions": [],
                    "inventory_rationale": "无主要实体。",
                }
            )
        if request.stage == "content.coverage.audit":
            return ModelResponse(
                parsed={
                    "missing_regions": [],
                    "coverage_complete": True,
                    "rationale": "无漏项。",
                }
            )
        self.inspect_count += 1
        if request.metadata.get("boundary_retry_count") == 0:
            return ModelResponse(
                parsed={
                    "crop_boundary_complete": False,
                    "revised_content_bbox_norm": [100, 80, 900, 240],
                    "defect_present": False,
                    "materiality": "none",
                    "evidence_basis": "visible_only",
                    "confidence": 0.95,
                    "grounding_status": "invalid",
                    "edit_bbox_full_norm": [0, 0, 1000, 1000],
                    "issues": [],
                    "rationale": "当前裁剪切断标题，需扩到完整标题。",
                }
            )
        if request.metadata.get("inspection_role") == "text_fidelity":
            return ModelResponse(
                parsed={
                    "crop_boundary_complete": True,
                    "revised_content_bbox_norm": [100, 80, 900, 240],
                    "defect_present": False,
                    "materiality": "none",
                    "evidence_basis": "visible_only",
                    "confidence": 0.95,
                    "grounding_status": "valid",
                    "edit_bbox_full_norm": [100, 80, 900, 240],
                    "issues": [],
                    "rationale": "未发现内容保真错误。",
                }
            )
        return ModelResponse(
            parsed={
                "crop_boundary_complete": True,
                "revised_content_bbox_norm": [100, 80, 900, 240],
                "defect_present": True,
                "materiality": "major",
                "evidence_basis": "visible_only",
                "confidence": 0.9,
                "grounding_status": "valid",
                "edit_bbox_full_norm": [100, 80, 900, 240],
                "issues": [
                    {
                        "artifact_type": "text_visual_integrity",
                        "observation": "完整标题中的多个字符明显粘连。",
                        "abnormality": "字形在原图中也无法辨认。",
                        "confidence": 0.9,
                        "evidence_kind": "visible_corruption",
                    }
                ],
                "rationale": "扩框后标题完整，确认原图文字损坏。",
            }
        )


class EntityTextFallbackModel:
    def __init__(self) -> None:
        self.requests = []

    async def complete(self, request):
        self.requests.append(request)
        if request.stage == "content.inventory.text":
            return ModelResponse(
                parsed={
                    "regions": [],
                    "inventory_rationale": "文字清单误漏。",
                }
            )
        if request.stage == "content.inventory.entity":
            return ModelResponse(
                parsed={
                    "regions": [
                        content_region(
                            "player-card",
                            [200, 200, 400, 600],
                            kind="entity",
                            description="含人物和多行文字的信息卡片",
                        )
                    ],
                    "inventory_rationale": "枚举独立卡片。",
                }
            )
        if request.stage == "content.coverage.audit":
            return ModelResponse(
                parsed={
                    "missing_regions": [],
                    "coverage_complete": True,
                    "rationale": "未补充文字区域。",
                }
            )
        base = {
            "crop_boundary_complete": True,
            "revised_content_bbox_norm": [200, 200, 400, 600],
            "defect_present": False,
            "materiality": "none",
            "evidence_basis": "visible_only",
            "confidence": 0.95,
            "grounding_status": "valid",
            "edit_bbox_full_norm": [240, 420, 360, 570],
            "issues": [],
            "rationale": "未见核心问题。",
        }
        if request.metadata.get("inspection_role") != "embedded_text":
            return ModelResponse(parsed=base)
        return ModelResponse(
            parsed={
                **base,
                "defect_present": True,
                "materiality": "major",
                "confidence": 0.94,
                "issues": [
                    {
                        "artifact_type": "text_content_fidelity",
                        "observation": "卡片内多行文字均为明显乱码。",
                        "abnormality": "字符笔画断裂粘连且无法辨认。",
                        "confidence": 0.94,
                        "evidence_kind": "visible_corruption",
                    }
                ],
                "rationale": "实体清单兜底发现卡片内的明显文字损坏。",
            }
        )


class BoundaryCompletenessTests(unittest.IsolatedAsyncioTestCase):
    async def test_incomplete_crop_expands_and_retries_before_finding(self) -> None:
        model = BoundaryRetryModel()
        auditor = AdaptiveContentCoverageAuditor(
            model,
            LosslessZoomImageTool(PassthroughImageTool()),
        )

        findings, summary = await auditor.audit(
            ImageAsset(id="anonymous", data=b"pixels"),
            CriticContext(),
        )

        inspections = [
            request
            for request in model.requests
            if request.stage == "content.inspect"
        ]
        self.assertEqual(len(inspections), 4)
        self.assertEqual(len(findings), 1)
        self.assertEqual(
            findings[0].bbox.as_list(),
            [100, 80, 900, 240],
        )
        self.assertEqual(findings[0].metadata["boundary_retry_count"], 1)
        self.assertEqual(
            findings[0].metadata["review_bbox_norm"],
            [36, 67, 964, 253],
        )
        self.assertEqual(summary["boundary_retry_count"], 1)
        self.assertEqual(summary["boundary_failure_count"], 0)

    async def test_entity_inventory_provides_independent_text_fallback(
        self,
    ) -> None:
        model = EntityTextFallbackModel()
        auditor = AdaptiveContentCoverageAuditor(
            model,
            LosslessZoomImageTool(PassthroughImageTool()),
        )

        findings, _ = await auditor.audit(
            ImageAsset(id="anonymous", data=b"pixels"),
            CriticContext(),
        )

        inspections = [
            request
            for request in model.requests
            if request.stage == "content.inspect"
        ]
        self.assertEqual(
            {
                request.metadata.get("inspection_role")
                for request in inspections
            },
            {"artifact", "embedded_text"},
        )
        self.assertEqual(len(findings), 1)
        self.assertEqual(
            findings[0].bbox.as_list(),
            [200, 200, 400, 600],
        )
        self.assertEqual(
            findings[0].metadata["semantic_group_id"],
            "player-card",
        )

    async def test_requests_never_include_input_metadata_labels(self) -> None:
        model = AdaptiveCoverageModel()
        auditor = AdaptiveContentCoverageAuditor(
            model,
            LosslessZoomImageTool(PassthroughImageTool()),
        )
        context = CriticContext(
            metadata={
                "source_path": "/secret/category/error/example.png",
                "reference_label": "answer-label",
                "filename": "example.png",
            }
        )

        await auditor.audit(
            ImageAsset(
                id="anonymous",
                data=b"pixels",
                metadata={"source_path": "/secret/category/error/example.png"},
            ),
            context,
        )

        for request in model.requests:
            serialized = " ".join(
                [
                    request.system_prompt,
                    request.user_prompt,
                    str(dict(request.metadata)),
                ]
            )
            self.assertNotIn("/secret/", serialized)
            self.assertNotIn("answer-label", serialized)
            self.assertNotIn("example.png", serialized)


if __name__ == "__main__":
    unittest.main()
