from __future__ import annotations

import unittest

from critic_workflows import (
    BBox,
    CriticContext,
    ImageAsset,
    ModelResponse,
    PassthroughImageTool,
)
from u1_critic_v41.text_coverage import (
    TextCoverageAuditor,
    coverage_tiles,
    deduplicate_text_blocks,
    map_tile_bbox,
    suppress_aggregate_text_blocks,
)


class TextCoverageGeometryTests(unittest.TestCase):
    def test_three_by_three_tiles_cover_every_pixel_with_exact_overlap(
        self,
    ) -> None:
        tiles = coverage_tiles()

        self.assertEqual(len(tiles), 9)
        self.assertEqual(tiles[0].bbox.x1, 0)
        self.assertEqual(tiles[0].bbox.y1, 0)
        self.assertEqual(tiles[0].bbox.x2, 362)
        self.assertEqual(tiles[-1].bbox.x1, 638)
        self.assertEqual(tiles[-1].bbox.x2, 1000)
        self.assertEqual(tiles[-1].bbox.y2, 1000)
        first = tiles[0].bbox
        second = tiles[1].bbox
        overlap = first.x2 - second.x1
        self.assertAlmostEqual(
            overlap / (first.x2 - first.x1),
            0.12,
            places=2,
        )

    def test_maps_local_tile_bbox_back_to_full_image(self) -> None:
        tile = BBox(250, 0, 750, 500)
        local = BBox(250, 500, 750, 1000)

        mapped = map_tile_bbox(tile, local)

        self.assertEqual(mapped.as_list(), [375, 250, 625, 500])

    def test_deduplicates_only_overlapping_text_entities(self) -> None:
        blocks = [
            {
                "region_id": "full-title",
                "bbox_norm": [100, 100, 400, 200],
                "visible_text": "标题",
                "status": "clear",
                "confidence": 0.9,
            },
            {
                "region_id": "tile-title",
                "bbox_norm": [110, 100, 390, 200],
                "visible_text": "标题",
                "status": "suspicious",
                "confidence": 0.7,
                "suspicion_kind": "visual_integrity",
                "suspicion_reason": "笔画明显断裂。",
            },
            {
                "region_id": "separate-label",
                "bbox_norm": [500, 100, 600, 200],
                "visible_text": "标签",
                "status": "suspicious",
                "confidence": 0.8,
                "suspicion_kind": "visual_integrity",
                "suspicion_reason": "字符无法辨认。",
            },
        ]

        merged = deduplicate_text_blocks(blocks)

        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[0]["status"], "suspicious")
        self.assertEqual(
            merged[0]["bbox_norm"],
            [100, 100, 400, 200],
        )
        self.assertEqual(
            merged[1]["region_id"],
            "separate-label",
        )

    def test_separate_tile_entities_supersede_one_aggregate_full_box(
        self,
    ) -> None:
        blocks = [
            {
                "region_id": "aggregate",
                "bbox_norm": [100, 100, 900, 900],
                "status": "suspicious",
                "intended_readable": True,
            },
            {
                "region_id": "label-left",
                "bbox_norm": [200, 200, 300, 300],
                "status": "suspicious",
                "intended_readable": True,
            },
            {
                "region_id": "label-right",
                "bbox_norm": [700, 700, 800, 800],
                "status": "suspicious",
                "intended_readable": True,
            },
        ]

        atomic = suppress_aggregate_text_blocks(blocks)

        self.assertEqual(
            [block["region_id"] for block in atomic],
            ["label-left", "label-right"],
        )

    def test_one_detail_crop_does_not_delete_its_complete_text_entity(
        self,
    ) -> None:
        blocks = [
            {
                "region_id": "complete-title",
                "bbox_norm": [100, 100, 800, 300],
                "status": "suspicious",
                "intended_readable": True,
            },
            {
                "region_id": "title-detail",
                "bbox_norm": [200, 120, 300, 280],
                "status": "suspicious",
                "intended_readable": True,
            },
        ]

        atomic = suppress_aggregate_text_blocks(blocks)

        self.assertEqual(len(atomic), 2)


class CoverageModel:
    def __init__(self) -> None:
        self.requests = []

    async def complete(self, request):
        self.requests.append(request)
        if request.stage == "text.coverage.full":
            return ModelResponse(
                parsed={
                    "blocks": [
                        self.block(
                            "title",
                            [100, 100, 400, 200],
                            status="clear",
                            visible_text="清晰标题",
                        ),
                        self.block(
                            "body",
                            [100, 300, 300, 400],
                            status="clear",
                            visible_text="清晰正文",
                        ),
                        self.block(
                            "label-a-full",
                            [500, 500, 590, 560],
                            status="suspicious",
                            visible_text="疑似标签甲",
                        ),
                    ],
                    "coverage_rationale": "整图文字清单。",
                }
            )
        row = request.metadata["tile_row"]
        column = request.metadata["tile_column"]
        blocks = []
        if (row, column) == (1, 1):
            blocks = [
                self.block(
                    "label-a-tile",
                    [500, 500, 750, 660],
                    status="suspicious",
                    visible_text="疑似标签甲",
                )
            ]
        if (row, column) == (1, 2):
            blocks = [
                self.block(
                    "label-b-tile",
                    [180, 500, 420, 660],
                    status="suspicious",
                    visible_text="疑似标签乙",
                )
            ]
        return ModelResponse(
            parsed={
                "blocks": blocks,
                "coverage_rationale": "分块文字清单。",
            }
        )

    @staticmethod
    def block(
        region_id,
        bbox_norm,
        *,
        status,
        visible_text,
    ):
        suspicious = status == "suspicious"
        return {
            "region_id": region_id,
            "bbox_norm": bbox_norm,
            "text_role": "label",
            "visible_text": visible_text,
            "intended_readable": True,
            "status": status,
            "suspicion_kind": (
                "visual_integrity" if suspicious else "none"
            ),
            "suspicion_reason": (
                "字符明显扭曲，无法辨认。" if suspicious else ""
            ),
            "confidence": 0.55 if suspicious else 0.95,
        }


class TextCoverageAuditorTests(unittest.IsolatedAsyncioTestCase):
    async def test_inventories_full_image_and_tiles_without_merging_labels(
        self,
    ) -> None:
        model = CoverageModel()
        auditor = TextCoverageAuditor(
            model,
            PassthroughImageTool(),
        )

        findings, summary = await auditor.audit(
            ImageAsset(id="anonymous", data=b"pixels"),
            CriticContext(),
        )

        stage_counts = {
            stage: len(
                [request for request in model.requests if request.stage == stage]
            )
            for stage in ("text.coverage.full", "text.coverage.tile")
        }
        self.assertEqual(stage_counts["text.coverage.full"], 1)
        self.assertEqual(stage_counts["text.coverage.tile"], 9)
        self.assertEqual(summary["covered_text_block_count"], 4)
        self.assertEqual(summary["suspicious_text_block_count"], 2)
        self.assertEqual(len(findings), 2)
        self.assertTrue(
            all(finding.source == "role:text_coverage" for finding in findings)
        )
        self.assertTrue(
            all(
                finding.metadata["text_entity_atomic"]
                for finding in findings
            )
        )
        self.assertTrue(
            all(finding.confidence == 0.55 for finding in findings)
        )
        self.assertLess(findings[0].bbox.x2, findings[1].bbox.x1)


if __name__ == "__main__":
    unittest.main()
