from __future__ import annotations

import json
import hashlib
import tempfile
import unittest
from pathlib import Path

from u1_critic_v41.report import (
    adapt_report_bbox,
    build_report,
    load_report_data,
)


class ReportTests(unittest.TestCase):
    def test_v41_does_not_accept_offline_consolidated_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "source.jpg"
            image.write_bytes(b"image")
            (root / "manifest.json").write_text(
                json.dumps(
                    {
                        "images": [
                            {
                                "image_id": "img_a",
                                "source_path": str(image),
                                "reference_label": "",
                                "mime_type": "image/jpeg",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            (root / "run_config.json").write_text(
                json.dumps(
                    {
                        "workflows": ["adversarial_consensus"],
                        "coverage_mode": "harness_v41",
                        "pipeline_version": "v4.1",
                    }
                ),
                encoding="utf-8",
            )
            (root / "results.jsonl").write_text(
                json.dumps(
                    {
                        "sample_id": "img_a",
                        "workflow": "adversarial_consensus",
                        "status": "ok",
                        "result": {
                            "verdict": "artifact_present",
                            "confidence": 0.9,
                            "findings": [],
                            "metadata": {},
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (root / "consolidated_results.jsonl").write_text(
                json.dumps(
                    {
                        "sample_id": "img_a",
                        "workflow": "adversarial_consensus",
                        "status": "ok",
                        "consolidated_result": {
                            "verdict": "artifact_present",
                            "confidence": 0.9,
                            "edit_regions": [],
                            "discarded_finding_ids": [],
                            "discard_reasons": {},
                            "rationale": "离线补结果。",
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            data = load_report_data(root)

        self.assertEqual(data["stats"]["success_units"], 0)
        self.assertIsNone(
            data["records"][0]["units"]["adversarial_consensus"]
            ["consolidated_result"]
        )

    def test_single_pipeline_stats_require_successful_consolidation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "source.jpg"
            image.write_bytes(b"image")
            (root / "manifest.json").write_text(
                json.dumps(
                    {
                        "input_root": str(root),
                        "images": [
                            {
                                "image_id": "img_a",
                                "source_path": str(image),
                                "reference_label": "",
                                "mime_type": "image/jpeg",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (root / "run_config.json").write_text(
                json.dumps(
                    {
                        "workflows": ["adversarial_consensus"],
                        "coverage_mode": "harness",
                    }
                ),
                encoding="utf-8",
            )
            result = {
                "sample_id": "img_a",
                "workflow": "adversarial_consensus",
                "status": "ok",
                "result": {
                    "verdict": "artifact_present",
                    "confidence": 0.9,
                    "findings": [],
                    "uncertain_findings": [],
                    "trace": [],
                    "metadata": {},
                },
            }
            (root / "results.jsonl").write_text(
                json.dumps(result) + "\n",
                encoding="utf-8",
            )
            consolidation_error = {
                "sample_id": "img_a",
                "workflow": "adversarial_consensus",
                "status": "error",
                "error_type": "ConsolidationValidationError",
                "error": "source finding id not handled: f001",
            }
            (root / "consolidation_errors.jsonl").write_text(
                json.dumps(consolidation_error) + "\n",
                encoding="utf-8",
            )

            failed = load_report_data(root)

            self.assertEqual(failed["stats"]["success_units"], 0)
            self.assertEqual(failed["stats"]["error_units"], 1)
            failed_unit = failed["records"][0]["units"][
                "adversarial_consensus"
            ]
            self.assertEqual(len(failed_unit["consolidation_errors"]), 1)

            consolidated = {
                "sample_id": "img_a",
                "workflow": "adversarial_consensus",
                "status": "ok",
                "consolidated_result": {
                    "verdict": "artifact_present",
                    "confidence": 0.9,
                    "edit_regions": [],
                    "discarded_finding_ids": [],
                    "discard_reasons": {},
                    "rationale": "已完成最终整编。",
                },
            }
            (root / "consolidated_results.jsonl").write_text(
                json.dumps(consolidated, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

            recovered = load_report_data(root)

        self.assertEqual(recovered["stats"]["success_units"], 1)
        self.assertEqual(recovered["stats"]["error_units"], 0)
        recovered_unit = recovered["records"][0]["units"][
            "adversarial_consensus"
        ]
        self.assertEqual(recovered_unit["consolidation_errors"], [])
        self.assertEqual(
            len(recovered_unit["historical_consolidation_errors"]),
            1,
        )

    def test_report_bbox_adapter_supports_new_and_legacy_coordinates(self):
        self.assertEqual(
            adapt_report_bbox(
                [100, 200, 700, 800],
                coordinate_space="bbox_1000",
            ),
            [100, 200, 700, 800],
        )
        self.assertEqual(
            adapt_report_bbox(
                [0.1, 0.2, 0.7, 0.8],
                coordinate_space=None,
            ),
            [100, 200, 700, 800],
        )
        self.assertIsNone(
            adapt_report_bbox(
                [100, 200, 100, 800],
                coordinate_space="bbox_1000",
            )
        )

    def test_adaptive_report_describes_semantic_units_and_crop_zoom(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "source.jpg"
            image.write_bytes(b"image")
            (root / "manifest.json").write_text(
                json.dumps(
                    {
                        "input_root": str(root),
                        "images": [
                            {
                                "image_id": "img_a",
                                "source_path": str(image),
                                "reference_label": "",
                                "mime_type": "image/jpeg",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (root / "run_config.json").write_text(
                json.dumps(
                    {
                        "workflows": ["adversarial_consensus"],
                        "coverage_mode": "adaptive",
                    }
                ),
                encoding="utf-8",
            )
            (root / "results.jsonl").write_text(
                json.dumps(
                    {
                        "sample_id": "img_a",
                        "workflow": "adversarial_consensus",
                        "status": "ok",
                        "result": {
                            "verdict": "no_artifact",
                            "confidence": 0.9,
                            "findings": [],
                            "uncertain_findings": [],
                            "trace": [],
                            "metadata": {},
                        },
                        "consolidated_result": {
                            "verdict": "no_artifact",
                            "confidence": 0.9,
                            "edit_regions": [],
                            "discarded_finding_ids": [],
                            "discard_reasons": {},
                            "rationale": "未发现核心问题。",
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            traces = [
                {
                    "sequence": 1,
                    "sample_id": "img_a",
                    "workflow": "adversarial_consensus",
                    "stage": "content.inventory.text",
                    "status": "ok",
                    "latency_seconds": 0.2,
                    "request": {"metadata": {"inventory_kind": "text"}},
                    "parsed": {
                        "regions": [
                            {
                                "semantic_group_id": "card-a",
                                "semantic_unit_kind": "card",
                                "description": "左侧信息卡片完整文字区",
                            }
                        ]
                    },
                },
                {
                    "sequence": 2,
                    "sample_id": "img_a",
                    "workflow": "adversarial_consensus",
                    "stage": "content.inventory.entity",
                    "status": "ok",
                    "latency_seconds": 0.2,
                    "request": {"metadata": {"inventory_kind": "entity"}},
                    "parsed": {
                        "regions": [
                            {
                                "semantic_group_id": "person",
                                "semantic_unit_kind": "entity",
                                "description": "主要人物",
                            }
                        ]
                    },
                },
                {
                    "sequence": 3,
                    "sample_id": "img_a",
                    "workflow": "adversarial_consensus",
                    "stage": "content.coverage.audit",
                    "status": "ok",
                    "latency_seconds": 0.1,
                    "request": {"metadata": {}},
                    "parsed": {
                        "missing_regions": [],
                        "coverage_complete": True,
                    },
                },
                {
                    "sequence": 4,
                    "sample_id": "img_a",
                    "workflow": "adversarial_consensus",
                    "stage": "content.inspect",
                    "status": "ok",
                    "latency_seconds": 0.3,
                    "request": {
                        "metadata": {
                            "semantic_group_id": "card-a",
                            "semantic_unit_kind": "card",
                            "original_crop_size": [320, 200],
                            "review_crop_size": [1280, 800],
                            "zoom_scale": 4,
                            "boundary_retry_count": 0,
                        }
                    },
                    "parsed": {
                        "crop_boundary_complete": True,
                        "defect_present": False,
                        "issues": [],
                        "rationale": "完整卡片未见明显问题。",
                    },
                },
            ]
            (root / "traces.jsonl").write_text(
                "\n".join(
                    json.dumps(row, ensure_ascii=False)
                    for row in traces
                )
                + "\n",
                encoding="utf-8",
            )

            data = load_report_data(root)
            html = build_report(root).read_text("utf-8")

        self.assertEqual(data["coverage_mode"], "adaptive")
        intro = data["workflow_intros"]["adversarial_consensus"]
        self.assertIn("自适应语义区块", intro["bias"])
        trace = data["records"][0]["units"]["adversarial_consensus"][
            "traces"
        ]
        self.assertEqual(trace[0]["stage_zh"], "文字语义单元清单")
        self.assertEqual(trace[1]["stage_zh"], "主要实体清单")
        self.assertEqual(trace[2]["stage_zh"], "内容覆盖终审")
        self.assertEqual(trace[3]["stage_zh"], "语义区块独立检查")
        self.assertEqual(data["stats"]["inventory_regions"], 2)
        self.assertEqual(data["stats"]["inspected_regions"], 1)
        self.assertEqual(data["stats"]["enlarged_crops"], 1)
        self.assertIn("左侧信息卡片完整文字区", html)
        self.assertIn("原始裁剪尺寸", html)
        self.assertIn("检查裁剪尺寸", html)
        self.assertIn("无损放大倍数", html)
        self.assertIn("自适应语义区块", html)

    def test_report_contains_workflow_intros_traces_and_escaped_model_text(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "source.jpg"
            image.write_bytes(b"image")
            (root / "manifest.json").write_text(
                json.dumps(
                    {
                        "input_root": str(root),
                        "images": [
                            {
                                "image_id": "img_a",
                                "source_path": str(image),
                                "reference_label": "D1/A2",
                                "mime_type": "image/jpeg",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            result = {
                "sample_id": "img_a",
                "workflow": "specialist_cascade",
                "status": "ok",
                "result": {
                    "verdict": "artifact_present",
                    "confidence": 0.8,
                    "findings": [
                        {
                            "region_id": "r1",
                            "bbox_norm": [100, 200, 400, 500],
                            "artifact_type": "structural_artifact",
                            "observation": "</script><img src=x onerror=alert(1)>",
                            "abnormality": "broken",
                            "confidence": 0.8,
                            "evidence_ids": ["img_a"],
                            "verification": "supported",
                            "source": "specialist",
                            "metadata": {},
                        }
                    ],
                    "uncertain_findings": [],
                    "trace": [],
                    "metadata": {},
                },
            }
            (root / "results.jsonl").write_text(
                json.dumps(result) + "\n",
                encoding="utf-8",
            )
            (root / "traces.jsonl").write_text(
                json.dumps(
                    {
                        "sequence": 1,
                        "sample_id": "img_a",
                        "workflow": "specialist_cascade",
                        "stage": "specialist.detect",
                        "status": "ok",
                        "latency_seconds": 1.2,
                        "request": {"system_prompt": "s", "user_prompt": "u"},
                        "parsed": {"present": True},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (root / "errors.jsonl").write_text(
                json.dumps(
                    {
                        "sample_id": "img_a",
                        "workflow": "specialist_cascade",
                        "status": "error",
                        "error_type": "ValueError",
                        "error": "old bbox failure",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            translated_observation = "发现明显的结构错误"
            translated_abnormality = "结构不符合正常形态"
            (root / "translations.json").write_text(
                json.dumps(
                    {
                        "translations": {
                            hashlib.sha256(
                                "</script><img src=x onerror=alert(1)>".encode()
                            ).hexdigest(): translated_observation,
                            hashlib.sha256("broken".encode()).hexdigest(): (
                                translated_abnormality
                            ),
                        }
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            data = load_report_data(root)
            html = build_report(root).read_text("utf-8")

        self.assertEqual(
            data["records"][0]["units"]["specialist_cascade"][
                "historical_errors"
            ][0]["error"],
            "old bbox failure",
        )
        specialist = data["records"][0]["units"]["specialist_cascade"]
        self.assertEqual(
            specialist["result"]["findings"][0]["observation"],
            translated_observation,
        )
        self.assertEqual(
            specialist["result_original"]["findings"][0]["observation"],
            "</script><img src=x onerror=alert(1)>",
        )
        self.assertEqual(specialist["image_url"], "/images/img_a")
        self.assertEqual(specialist["traces"][0]["stage_zh"], "专员检测")
        self.assertIn("Specialist Cascade", html)
        self.assertIn("Evidence Seeking Agent", html)
        self.assertIn("Adversarial Consensus", html)
        self.assertIn("specialist.detect", html)
        self.assertIn("evidence.plan", html)
        self.assertIn("consensus.judge", html)
        self.assertNotIn("</script><img src=x onerror=alert(1)>", html)
        self.assertIn("\\u003c/script\\u003e", html)
        self.assertIn("/images/img_a", html)
        self.assertIn("stageFilter", html)
        self.assertIn(translated_observation, html)
        self.assertIn("每套管线独立视图", html)
        self.assertNotIn(
            '<div class=pipeline-code>${esc(CODES[w])}</div>',
            html,
        )
        self.assertNotIn(
            '<div class=stage>${esc(t.stage)}</div>',
            html,
        )
        self.assertNotIn(
            '<small>${esc(s[0])}</small>',
            html,
        )

    def test_single_consensus_report_uses_one_pipeline_and_edit_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "source.jpg"
            image.write_bytes(b"image")
            (root / "manifest.json").write_text(
                json.dumps(
                    {
                        "input_root": str(root),
                        "images": [
                            {
                                "image_id": "img_a",
                                "source_path": str(image),
                                "reference_label": "D1/A2",
                                "mime_type": "image/jpeg",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (root / "run_config.json").write_text(
                json.dumps(
                    {
                        "workflows": ["adversarial_consensus"],
                        "bbox_coordinate_space": "bbox_1000",
                    }
                ),
                encoding="utf-8",
            )
            result = {
                "sample_id": "img_a",
                "workflow": "adversarial_consensus",
                "status": "ok",
                "result": {
                    "verdict": "artifact_present",
                    "confidence": 0.9,
                    "findings": [
                        {
                            "region_id": "hand",
                            "bbox_norm": [100, 200, 500, 700],
                            "artifact_type": "structural_artifact",
                            "observation": "手指数量异常。",
                            "abnormality": "人物左手有六根手指。",
                            "confidence": 0.9,
                            "evidence_ids": ["img_a"],
                            "verification": "supported",
                            "source": "role:structure",
                            "metadata": {
                                "edit_target": "人物左手和手腕连接处",
                                "edit_region_rationale": "覆盖完整手掌并保留手腕上下文。",
                                "grounding_status": "valid",
                                "grounding_rationale": "该区域可独立重绘。",
                            },
                        }
                    ],
                    "uncertain_findings": [],
                    "trace": [],
                    "metadata": {
                        "coverage_status": "failed",
                        "findings_non_exhaustive": True,
                    },
                },
            }
            (root / "results.jsonl").write_text(
                json.dumps(result, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            (root / "consolidated_results.jsonl").write_text(
                json.dumps(
                    {
                        "sample_id": "img_a",
                        "workflow": "adversarial_consensus",
                        "status": "ok",
                        "consolidated_result": {
                            "verdict": "artifact_present",
                            "confidence": 0.9,
                            "edit_regions": [
                                {
                                    "edit_region_id": "e001",
                                    "bbox_norm": [100, 200, 500, 700],
                                    "edit_target": "人物左手编辑区域",
                                    "edit_region_rationale": (
                                        "一次重绘同时修复结构和连接问题。"
                                    ),
                                    "source_finding_ids": ["f000", "f001"],
                                    "issues": [
                                        {
                                            "artifact_type": (
                                                "structural_artifact"
                                            ),
                                            "observation": "手指数量异常。",
                                            "abnormality": "人物左手有六根手指。",
                                            "confidence": 0.9,
                                            "materiality": "major",
                                            "source_finding_ids": ["f000"],
                                        },
                                        {
                                            "artifact_type": (
                                                "topology_artifact"
                                            ),
                                            "observation": "手腕连接异常。",
                                            "abnormality": "手掌和手腕边界断裂。",
                                            "confidence": 0.8,
                                            "materiality": "major",
                                            "source_finding_ids": ["f001"],
                                        },
                                    ],
                                    "confidence": 0.9,
                                    "verification": "supported",
                                },
                                {
                                    "edit_region_id": "e002",
                                    "bbox_norm": [600, 200, 900, 600],
                                    "edit_target": "背景文字块",
                                    "edit_region_rationale": (
                                        "单独重绘背景文字区域。"
                                    ),
                                    "source_finding_ids": ["f002"],
                                    "issues": [
                                        {
                                            "artifact_type": (
                                                "text_visual_integrity"
                                            ),
                                            "observation": "文字字形破损。",
                                            "abnormality": "部分字符无法辨认。",
                                            "confidence": 0.7,
                                            "materiality": "major",
                                            "source_finding_ids": ["f002"],
                                        }
                                    ],
                                    "confidence": 0.7,
                                    "verification": "supported",
                                },
                            ],
                            "discarded_finding_ids": [],
                            "discard_reasons": {},
                            "rationale": "三个问题整编为两个编辑区域。",
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            trace_rows = [
                {
                    "sequence": 1,
                    "sample_id": "img_a",
                    "workflow": "adversarial_consensus",
                    "stage": "text.coverage.full",
                    "status": "ok",
                    "latency_seconds": 0.5,
                    "request": {
                        "metadata": {"coverage_scope": "full"},
                    },
                    "parsed": {
                        "blocks": [
                            {
                                "status": "clear",
                                "visible_text": "清晰标题",
                            }
                        ]
                    },
                },
                {
                    "sequence": 2,
                    "sample_id": "img_a",
                    "workflow": "adversarial_consensus",
                    "stage": "text.coverage.tile",
                    "status": "ok",
                    "latency_seconds": 0.4,
                    "request": {
                        "metadata": {
                            "coverage_scope": "tile",
                            "tile_row": 0,
                            "tile_column": 0,
                        },
                    },
                    "parsed": {
                        "blocks": [
                            {
                                "status": "suspicious",
                                "visible_text": "异常标签",
                            }
                        ]
                    },
                },
                {
                    "sequence": 3,
                    "sample_id": "img_a",
                    "workflow": "adversarial_consensus",
                    "stage": "consensus.challenge",
                    "status": "ok",
                    "latency_seconds": 0.3,
                    "parsed": {"defect_present": True},
                },
                {
                    "sequence": 4,
                    "sample_id": "img_a",
                    "workflow": "adversarial_consensus",
                    "stage": "consensus.consolidate",
                    "status": "ok",
                    "latency_seconds": 1.0,
                    "parsed": {
                        "edit_regions": [
                            {
                                "edit_target": "人物左手编辑区域",
                                "issues": [],
                            }
                        ]
                    },
                },
                {
                    "sequence": 5,
                    "sample_id": "img_a",
                    "workflow": "adversarial_consensus",
                    "stage": "text.coverage.full",
                    "status": "ok",
                    "latency_seconds": 0.6,
                    "request": {
                        "metadata": {"coverage_scope": "full"},
                    },
                    "parsed": {
                        "blocks": [
                            {
                                "status": "clear",
                                "visible_text": "重试后的清晰标题",
                            }
                        ]
                    },
                },
                {
                    "sequence": 6,
                    "sample_id": "img_a",
                    "workflow": "adversarial_consensus",
                    "stage": "text.coverage.tile",
                    "status": "ok",
                    "latency_seconds": 0.5,
                    "request": {
                        "metadata": {
                            "coverage_scope": "tile",
                            "tile_row": 0,
                            "tile_column": 0,
                        },
                    },
                    "parsed": {
                        "blocks": [
                            {
                                "status": "suspicious",
                                "visible_text": "重试后的异常标签",
                            }
                        ]
                    },
                },
            ]
            (root / "traces.jsonl").write_text(
                "\n".join(
                    json.dumps(row, ensure_ascii=False)
                    for row in trace_rows
                )
                + "\n",
                encoding="utf-8",
            )

            data = load_report_data(root)
            html = build_report(root).read_text("utf-8")

        self.assertEqual(data["workflow_order"], ["adversarial_consensus"])
        self.assertEqual(
            data["report_title"],
            "Adversarial Consensus 图像质量评估",
        )
        self.assertEqual(
            data["workflow_heading"],
            "Adversarial Consensus 工作流程",
        )
        self.assertFalse(data["comparison_enabled"])
        self.assertEqual(data["layout_class"], "single")
        self.assertEqual(
            set(data["records"][0]["units"]),
            {"adversarial_consensus"},
        )
        consolidated = data["records"][0]["units"][
            "adversarial_consensus"
        ]["consolidated_result"]
        self.assertEqual(len(consolidated["edit_regions"]), 2)
        self.assertEqual(len(consolidated["edit_regions"][0]["issues"]), 2)
        trace = data["records"][0]["units"]["adversarial_consensus"][
            "traces"
        ]
        self.assertEqual(trace[0]["stage_zh"], "整图文字清单")
        self.assertEqual(trace[1]["stage_zh"], "分块文字覆盖")
        self.assertEqual(trace[2]["stage_zh"], "真实裁剪复核")
        self.assertEqual(trace[3]["stage_zh"], "最终结果整编")
        self.assertIn("编辑区域", trace[3]["parsed_display"])
        self.assertEqual(data["stats"]["covered_text_blocks"], 2)
        self.assertEqual(data["stats"]["suspicious_text_blocks"], 1)
        self.assertEqual(data["stats"]["final_edit_regions"], 2)
        self.assertIn("人物左手和手腕连接处", html)
        self.assertIn("覆盖完整手掌并保留手腕上下文", html)
        self.assertIn("该区域可独立重绘", html)
        self.assertIn("人物左手编辑区域", html)
        self.assertIn("手腕连接异常", html)
        self.assertIn("clamp(560px,70vh,900px)", html)
        self.assertIn("${i+1}</span>", html)
        self.assertIn("编辑区域 ${i+1}", html)
        self.assertIn("整图文字清单", html)
        self.assertIn("分块文字覆盖", html)
        self.assertIn("真实裁剪复核", html)
        self.assertIn("文字块覆盖数量", html)
        self.assertIn(
            "覆盖不完整：以下已发现问题不是穷尽结果",
            html,
        )
        self.assertIn("function traceBundle(u)", html)
        self.assertIn(
            "查看模型调用轨迹（${u.traces.length} 次）",
            html,
        )
        self.assertIn("${traceBundle(u)}", html)


if __name__ == "__main__":
    unittest.main()
