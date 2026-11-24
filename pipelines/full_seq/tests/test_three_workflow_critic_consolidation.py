from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from critic_workflows import ImageAsset, ModelResponse
from u1_critic_v41.consolidation import (
    ConsolidationValidationError,
    ResultConsolidator,
    is_duplicate_edit_region,
    partition_source_findings,
    source_findings,
    spatial_hints,
    validate_consolidated_result,
)
from u1_critic_v41.consolidate_existing import (
    _tagged,
    completed_samples,
    consolidate_existing,
    parse_args,
    reject_nested_containers,
)


def finding(
    bbox,
    artifact_type,
    edit_target,
    verification="supported",
):
    return {
        "region_id": edit_target,
        "bbox_norm": bbox,
        "artifact_type": artifact_type,
        "observation": f"{edit_target}存在可见问题。",
        "abnormality": f"{edit_target}需要局部重绘。",
        "confidence": 0.8,
        "evidence_ids": ["img_a"],
        "verification": verification,
        "metadata": {
            "edit_target": edit_target,
            "edit_region_rationale": "覆盖完整实体和必要上下文。",
            "grounding_status": "valid",
            "grounding_rationale": "该区域可独立重绘。",
            "defect_present": True,
            "materiality": "major",
            "evidence_basis": "visible_only",
        },
    }


class SourceFindingTests(unittest.TestCase):
    def test_assigns_stable_ids_and_reports_literal_spatial_relations(self) -> None:
        result = {
            "findings": [
                finding(
                    [0, 0, 500, 500],
                    "text_visual_integrity",
                    "标题文字块",
                ),
                finding(
                    [0, 0, 500, 500],
                    "text_content_fidelity",
                    "标题文字块",
                ),
            ],
            "uncertain_findings": [
                finding(
                    [250, 0, 750, 500],
                    "structural_artifact",
                    "标题旁人物",
                    "uncertain",
                )
            ],
        }

        sources = source_findings(result)
        hints = spatial_hints(sources)

        self.assertEqual(
            [item["source_finding_id"] for item in sources],
            ["f000", "f001", "f002"],
        )
        self.assertEqual(
            [item["source_status"] for item in sources],
            ["accepted", "accepted", "uncertain"],
        )
        self.assertEqual(
            hints,
            [
                {
                    "left_id": "f000",
                    "right_id": "f001",
                    "iou": 1.0,
                    "left_covered": 1.0,
                    "right_covered": 1.0,
                },
                {
                    "left_id": "f000",
                    "right_id": "f002",
                    "iou": 0.3333,
                    "left_covered": 0.5,
                    "right_covered": 0.5,
                },
                {
                    "left_id": "f001",
                    "right_id": "f002",
                    "iou": 0.3333,
                    "left_covered": 0.5,
                    "right_covered": 0.5,
                },
            ],
        )

    def test_partition_keeps_semantic_group_bounded(self) -> None:
        values = []
        for index in range(23):
            item = finding(
                [index, 10, index + 10, 20],
                "text_visual_integrity",
                f"文字区域{index // 3}",
            )
            item["source_finding_id"] = f"f{index:03d}"
            item["metadata"]["semantic_group_id"] = f"g{index // 3}"
            values.append(item)

        batches = partition_source_findings(values, max_batch_size=10)

        self.assertEqual([len(batch) for batch in batches], [10, 10, 3])
        self.assertEqual(
            {item["source_finding_id"] for batch in batches for item in batch},
            {f"f{index:03d}" for index in range(23)},
        )


def issue(source_ids, artifact_type="structural_artifact"):
    return {
        "artifact_type": artifact_type,
        "observation": "局部存在清晰可见的问题。",
        "abnormality": "该问题需要通过局部重绘修复。",
        "confidence": 0.8,
        "materiality": "major",
        "source_finding_ids": source_ids,
    }


def region(region_id, bbox, target, source_ids):
    return {
        "edit_region_id": region_id,
        "bbox_norm": bbox,
        "edit_target": target,
        "edit_region_rationale": "覆盖完整编辑实体和必要上下文。",
        "source_finding_ids": source_ids,
        "issues": [issue(source_ids)],
        "confidence": 0.8,
        "verification": "supported",
    }


def sources(*source_ids, artifact_type="structural_artifact", bboxes=None):
    values = bboxes or [
        [0.1 + index * 0.2, 0.1, 0.2 + index * 0.2, 0.2]
        for index in range(len(source_ids))
    ]
    return [
        {
            "source_finding_id": source_id,
            "bbox_norm": values[index],
            "artifact_type": artifact_type,
            "metadata": {
                "text_entity_atomic": artifact_type.startswith("text_"),
            },
        }
        for index, source_id in enumerate(source_ids)
    ]


class ConsolidationValidationTests(unittest.TestCase):
    def test_rejects_single_nested_child_in_v41_mode(self) -> None:
        value = {
            "verdict": "artifact_present",
            "confidence": 0.9,
            "edit_regions": [
                region(
                    "outer",
                    [100, 100, 900, 900],
                    "完整信息面板",
                    ["f000"],
                ),
                region(
                    "inner",
                    [300, 300, 500, 500],
                    "面板内独立图标",
                    ["f001"],
                ),
            ],
            "discarded_finding_ids": [],
            "discard_reasons": {},
            "rationale": "两个框发生包含。",
        }

        errors = validate_consolidated_result(
            value,
            sources("f000", "f001"),
            reject_nested_containers=True,
        )

        self.assertIn("redundant nested edit container", "\n".join(errors))

    def test_rejects_reason_for_non_discarded_source(self) -> None:
        value = {
            "verdict": "artifact_present",
            "confidence": 0.9,
            "edit_regions": [
                region(
                    "e001",
                    [100, 100, 300, 300],
                    "独立主体",
                    ["f000"],
                )
            ],
            "discarded_finding_ids": [],
            "discard_reasons": {"f000_note": "多余说明。"},
            "rationale": "完成整编。",
        }

        errors = validate_consolidated_result(value, sources("f000"))

        self.assertIn(
            "discard reason references non-discarded id: f000_note",
            errors,
        )
    def test_high_overlap_similar_target_names_are_duplicate(self) -> None:
        left = region(
            "e001",
            [100, 100, 800, 800],
            "右侧作者介绍与书籍说明文字块",
            ["f000"],
        )
        right = region(
            "e002",
            [110, 110, 810, 810],
            "右侧作者介绍与书籍简介文本块",
            ["f001"],
        )

        self.assertTrue(is_duplicate_edit_region(left, right))

    def test_high_overlap_independent_targets_are_not_rejected_by_iou(self) -> None:
        value = {
            "verdict": "artifact_present",
            "confidence": 0.8,
            "edit_regions": [
                region(
                    "e001",
                    [20, 40, 980, 960],
                    "左侧人物主体",
                    ["f000"],
                ),
                region(
                    "e002",
                    [40, 60, 960, 940],
                    "右侧人物主体",
                    ["f001"],
                ),
            ],
            "discarded_finding_ids": [],
            "discard_reasons": {},
            "rationale": "两个实体需要分别局部重绘。",
        }

        errors = validate_consolidated_result(
            value,
            sources("f000", "f001"),
        )

        self.assertEqual(errors, [])
        self.assertFalse(
            is_duplicate_edit_region(
                value["edit_regions"][0],
                value["edit_regions"][1],
            )
        )

    def test_rejects_broken_source_mapping_and_invalid_regions(self) -> None:
        duplicate = region(
            "e001",
            [0, 0, 1000, 1000],
            "标题文字块",
            ["f000", "unknown"],
        )
        duplicate["issues"][0]["source_finding_ids"] = ["f001"]
        value = {
            "verdict": "artifact_present",
            "confidence": 0.8,
            "edit_regions": [
                duplicate,
                region(
                    "e002",
                    [0, 0, 1000, 1000],
                    "标题文字块",
                    ["f000"],
                ),
            ],
            "discarded_finding_ids": ["f000"],
            "discard_reasons": {"f000": "重复候选已丢弃。"},
            "rationale": "完成整编。",
        }

        errors = validate_consolidated_result(
            value,
            sources("f000", "f001", "f002"),
        )
        joined = "\n".join(errors)

        self.assertIn("unknown source finding id: unknown", joined)
        self.assertIn("source finding id appears more than once: f000", joined)
        self.assertIn("source finding id not handled: f002", joined)
        self.assertIn("issue references id outside its edit region: f001", joined)
        self.assertIn("near-full-frame bbox", joined)
        self.assertIn("duplicate edit regions: e001 and e002", joined)

    def test_same_text_entity_from_overlapping_tiles_can_merge(self) -> None:
        source_values = sources(
            "f000",
            "f001",
            artifact_type="text_visual_integrity",
            bboxes=[
                [100, 100, 300, 200],
                [110, 100, 290, 200],
            ],
        )
        value = {
            "verdict": "artifact_present",
            "confidence": 0.9,
            "edit_regions": [
                {
                    **region(
                        "e001",
                        [100, 100, 300, 200],
                        "同一标题文字实体",
                        ["f000", "f001"],
                    ),
                    "issues": [
                        issue(
                            ["f000", "f001"],
                            "text_visual_integrity",
                        )
                    ],
                }
            ],
            "discarded_finding_ids": [],
            "discard_reasons": {},
            "rationale": "重叠来源属于同一文字实体。",
        }

        self.assertEqual(
            validate_consolidated_result(value, source_values),
            [],
        )

    def test_spatially_separate_text_entities_cannot_share_one_region(
        self,
    ) -> None:
        source_values = sources(
            "f000",
            "f001",
            artifact_type="text_visual_integrity",
            bboxes=[
                [100, 100, 200, 200],
                [700, 700, 800, 800],
            ],
        )
        value = {
            "verdict": "artifact_present",
            "confidence": 0.9,
            "edit_regions": [
                {
                    **region(
                        "e001",
                        [100, 100, 800, 800],
                        "两个分离标签",
                        ["f000", "f001"],
                    ),
                    "issues": [
                        issue(
                            ["f000", "f001"],
                            "text_visual_integrity",
                        )
                    ],
                }
            ],
            "discarded_finding_ids": [],
            "discard_reasons": {},
            "rationale": "错误地合并了分离文字。",
        }

        errors = validate_consolidated_result(value, source_values)

        self.assertIn(
            "spatially separate text entities",
            "\n".join(errors),
        )

    def test_semantically_identical_text_findings_can_merge_despite_bbox_drift(
        self,
    ) -> None:
        source_values = sources(
            "f000",
            "f001",
            "f002",
            artifact_type="text_content_fidelity",
            bboxes=[
                [456, 198, 976, 532],
                [518, 555, 956, 900],
                [745, 615, 935, 735],
            ],
        )
        targets = [
            "SP组右上角格子的类型标签ISFJ",
            "SP象限右上角格子当前误标ISFJ",
            "SP区块右上卡片当前标为ISFJ的标题与描述文字",
        ]
        for source, target in zip(source_values, targets, strict=True):
            source["metadata"]["edit_target"] = target
        value = {
            "verdict": "artifact_present",
            "confidence": 0.98,
            "edit_regions": [
                {
                    **region(
                        "sp_isfj_cell",
                        [745, 615, 935, 735],
                        "SP组右上角误标为ISFJ的卡片",
                        ["f000", "f001", "f002"],
                    ),
                    "issues": [
                        issue(
                            ["f000", "f001", "f002"],
                            "text_content_fidelity",
                        )
                    ],
                }
            ],
            "discarded_finding_ids": [],
            "discard_reasons": {},
            "rationale": "三个角色指向同一个卡片，使用准确框统一重绘。",
        }

        self.assertEqual(
            validate_consolidated_result(value, source_values),
            [],
        )

    def test_rejects_non_core_materiality_and_threshold_confidence(
        self,
    ) -> None:
        bad_region = region(
            "e001",
            [100, 100, 300, 300],
            "轻微区域",
            ["f000"],
        )
        bad_region["confidence"] = 0.6
        bad_region["issues"][0]["confidence"] = 0.6
        bad_region["issues"][0]["materiality"] = "minor"
        value = {
            "verdict": "artifact_present",
            "confidence": 0.6,
            "edit_regions": [bad_region],
            "discarded_finding_ids": [],
            "discard_reasons": {},
            "rationale": "不应保留轻微问题。",
        }

        errors = validate_consolidated_result(
            value,
            sources("f000"),
        )
        joined = "\n".join(errors)

        self.assertIn("non-core materiality", joined)
        self.assertIn("confidence must be > 0.6", joined)


class QueueModel:
    def __init__(self, *values):
        self.values = list(values)
        self.requests = []

    async def complete(self, request):
        self.requests.append(request)
        return ModelResponse(parsed=self.values.pop(0))


class ResultConsolidatorTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.image = ImageAsset(id="img_a", data=b"image", mime_type="image/jpeg")
        self.source_result = {
            "verdict": "artifact_present",
            "confidence": 0.8,
            "findings": [
                finding(
                    [100, 100, 500, 500],
                    "text_visual_integrity",
                    "标题文字块",
                ),
                finding(
                    [100, 100, 500, 500],
                    "text_content_fidelity",
                    "标题文字块",
                ),
            ],
            "uncertain_findings": [],
        }

    async def test_v41_materiality_gate_discards_low_confidence_without_model(self):
        low = finding(
            [100, 100, 300, 300],
            "rendering_artifact",
            "轻微纹理区域",
        )
        low["confidence"] = 0.6
        model = QueueModel()

        result = await ResultConsolidator(
            model,
            enforce_materiality_gate=True,
        ).consolidate(
            self.image,
            {
                "verdict": "artifact_present",
                "confidence": 0.6,
                "findings": [low],
            },
        )

        self.assertEqual(result["verdict"], "no_artifact")
        self.assertEqual(result["discarded_finding_ids"], ["f000"])
        self.assertEqual(model.requests, [])

    async def test_large_source_set_is_consolidated_in_bounded_batches(self):
        source_values = [
            finding(
                [20 + index * 30, 100, 40 + index * 30, 180],
                "structural_artifact",
                f"独立实体{index}",
            )
            for index in range(12)
        ]

        def partial(start, stop):
            return {
                "verdict": "artifact_present",
                "confidence": 0.9,
                "edit_regions": [
                    region(
                        f"local-{index}",
                        source_values[index]["bbox_norm"],
                        f"独立实体{index}",
                        [f"f{index:03d}"],
                    )
                    for index in range(start, stop)
                ],
                "discarded_finding_ids": [],
                "discard_reasons": {},
                "rationale": "完成本批语义区域整编。",
            }

        model = QueueModel(partial(0, 10), partial(10, 12))
        result = await ResultConsolidator(
            model,
            max_batch_size=10,
        ).consolidate(
            self.image,
            {
                "verdict": "artifact_present",
                "confidence": 0.9,
                "findings": source_values,
            },
        )

        self.assertEqual(len(model.requests), 2)
        self.assertTrue(
            all(
                request.metadata["source_finding_count"] <= 10
                for request in model.requests
            )
        )
        self.assertEqual(len(result["edit_regions"]), 12)

    async def test_v41_refuses_offline_consolidation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "run_config.json").write_text(
                json.dumps({"pipeline_version": "v4.1"}),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                RuntimeError,
                "forbids offline consolidation",
            ):
                await consolidate_existing(
                    SimpleNamespace(output_root=root)
                )

    @staticmethod
    def valid_output():
        return {
            "verdict": "artifact_present",
            "confidence": 0.8,
            "edit_regions": [
                {
                    **region(
                        "e001",
                        [100, 100, 500, 500],
                        "标题文字块",
                        ["f000", "f001"],
                    ),
                    "issues": [
                        issue(["f000"], "text_visual_integrity"),
                        issue(["f001"], "text_content_fidelity"),
                    ],
                }
            ],
            "discarded_finding_ids": [],
            "discard_reasons": {},
            "rationale": "同一文字块的问题合并为一个重绘区域。",
        }

    async def test_repairs_invalid_source_mapping_once(self) -> None:
        invalid = self.valid_output()
        invalid["discarded_finding_ids"] = ["f001"]
        invalid["discard_reasons"] = {"f001": "重复来源。"}
        model = QueueModel(invalid, self.valid_output())

        result = await ResultConsolidator(model).consolidate(
            self.image,
            self.source_result,
        )

        self.assertEqual(len(result["edit_regions"]), 1)
        self.assertEqual(
            [request.stage for request in model.requests],
            ["consensus.consolidate", "consensus.consolidate_repair"],
        )

    async def test_preserves_region_source_omitted_from_issue_mapping(
        self,
    ) -> None:
        incomplete = self.valid_output()
        incomplete["edit_regions"][0]["issues"] = [
            issue(["f000"], "text_visual_integrity")
        ]
        model = QueueModel(incomplete)

        result = await ResultConsolidator(model).consolidate(
            self.image,
            self.source_result,
        )

        self.assertEqual(len(result["edit_regions"][0]["issues"]), 2)
        self.assertEqual(
            {
                source_id
                for value in result["edit_regions"][0]["issues"]
                for source_id in value["source_finding_ids"]
            },
            {"f000", "f001"},
        )
        self.assertEqual(
            [request.stage for request in model.requests],
            ["consensus.consolidate"],
        )

    async def test_discards_missing_broad_container_when_precise_regions_exist(
        self,
    ) -> None:
        source_result = {
            "verdict": "artifact_present",
            "confidence": 0.95,
            "findings": [
                finding(
                    [0, 950, 1000, 1000],
                    "text_visual_integrity",
                    "底部状态栏整行文字",
                ),
                finding(
                    [10, 960, 250, 995],
                    "text_visual_integrity",
                    "底部左侧文字",
                ),
                finding(
                    [400, 960, 650, 995],
                    "text_visual_integrity",
                    "底部中央文字",
                ),
            ],
            "uncertain_findings": [],
        }
        output = {
            "verdict": "artifact_present",
            "confidence": 0.95,
            "edit_regions": [
                {
                    **region(
                        "bottom_left",
                        [10, 960, 250, 995],
                        "底部左侧文字",
                        ["f001"],
                    ),
                    "issues": [
                        issue(["f001"], "text_visual_integrity")
                    ],
                },
                {
                    **region(
                        "bottom_center",
                        [400, 960, 650, 995],
                        "底部中央文字",
                        ["f002"],
                    ),
                    "issues": [
                        issue(["f002"], "text_visual_integrity")
                    ],
                },
            ],
            "discarded_finding_ids": [],
            "discard_reasons": {},
            "rationale": "底部文字按独立语义区块分别重绘。",
        }
        model = QueueModel(output)

        result = await ResultConsolidator(
            model,
            reject_nested_containers=True,
        ).consolidate(self.image, source_result)

        self.assertEqual(result["discarded_finding_ids"], ["f000"])
        self.assertIn("宽泛重复框", result["discard_reasons"]["f000"])
        self.assertEqual(len(result["edit_regions"]), 2)
        self.assertEqual(
            [request.stage for request in model.requests],
            ["consensus.consolidate"],
        )

    async def test_raises_after_repair_is_still_invalid(self) -> None:
        invalid = self.valid_output()
        invalid["edit_regions"][0]["bbox_norm"] = [0, 0, 1000, 1000]
        model = QueueModel(invalid, invalid)

        with self.assertRaises(ConsolidationValidationError) as caught:
            await ResultConsolidator(model).consolidate(
                self.image,
                self.source_result,
            )

        self.assertIn("near-full-frame bbox", str(caught.exception))
        self.assertEqual(len(model.requests), 2)


class ConsolidationCliTests(unittest.TestCase):
    def test_offline_consolidation_inherits_harness_nested_rule(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "run_config.json").write_text(
                json.dumps({"coverage_mode": "harness"}),
                encoding="utf-8",
            )

            self.assertTrue(reject_nested_containers(root))

    def test_all_requires_explicit_confirmation(self) -> None:
        with self.assertRaises(SystemExit):
            parse_args(["--output-root", "out", "--all"])
        args = parse_args(
            [
                "--output-root",
                "out",
                "--all",
                "--confirm-run-all",
                "YES",
            ]
        )
        self.assertTrue(args.all)

    def test_completed_samples_only_returns_successful_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "consolidated_results.jsonl"
            path.write_text(
                "\n".join(
                    [
                        json.dumps({"sample_id": "a", "status": "ok"}),
                        json.dumps({"sample_id": "b", "status": "error"}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            self.assertEqual(completed_samples(path), {"a"})


class ConsolidationTaskTests(unittest.IsolatedAsyncioTestCase):
    async def test_tagged_preserves_sample_id_for_completed_task(self) -> None:
        async def operation():
            return {"status": "ok"}

        sample_id, value, error = await _tagged("img_a", operation())

        self.assertEqual(sample_id, "img_a")
        self.assertEqual(value, {"status": "ok"})
        self.assertIsNone(error)


if __name__ == "__main__":
    unittest.main()
