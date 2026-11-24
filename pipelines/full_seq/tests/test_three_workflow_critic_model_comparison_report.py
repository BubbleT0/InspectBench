from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from u1_critic_v41.model_comparison_report import (
    build_model_comparison_report,
)


class ModelComparisonReportTests(unittest.TestCase):
    def _write_run(self, root: Path, image: Path, verdict: str) -> None:
        root.mkdir()
        (root / "manifest.json").write_text(
            json.dumps(
                {
                    "input_root": str(image.parent),
                    "images": [
                        {
                            "image_id": "img_a",
                            "source_path": str(image),
                            "reference_label": ".",
                            "mime_type": "image/png",
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
                    "coverage_mode": "harness_v41",
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
                "verdict": verdict,
                "confidence": 0.9,
                "metadata": {
                    "coverage_status": "ok",
                    "coverage_channels": {
                        "text": {"status": "ok"},
                        "artifact": {"status": "ok"},
                    },
                },
            },
            "consolidated_result": {
                "verdict": verdict,
                "confidence": 0.9,
                "edit_regions": [],
                "rationale": "done",
            },
        }
        (root / "results.jsonl").write_text(
            json.dumps(result) + "\n",
            encoding="utf-8",
        )

    def test_builds_side_by_side_report_for_identical_samples(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            image = base / "image.png"
            image.write_bytes(b"png")
            left = base / "left"
            right = base / "right"
            output = base / "comparison"
            self._write_run(left, image, "artifact_present")
            self._write_run(right, image, "no_artifact")

            report = build_model_comparison_report(
                left,
                "Gemini",
                right,
                "Internal",
                output,
            )
            html = report.read_text("utf-8")

            self.assertIn("Gemini", html)
            self.assertIn("Internal", html)
            self.assertIn("img_a", html)
            self.assertIn('"disagreements":1', html)
            self.assertTrue((output / "manifest.json").exists())

    def test_adds_single_pass_direct_critic_as_third_column(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            image = base / "image.png"
            image.write_bytes(b"png")
            left = base / "left"
            right = base / "right"
            direct = base / "direct"
            output = base / "comparison"
            self._write_run(left, image, "artifact_present")
            self._write_run(right, image, "no_artifact")
            direct.mkdir()
            direct_row = {
                "status": "ok",
                "source": {"image_path": str(image)},
                "critic": {
                    "has_quality_issue": True,
                    "overall_severity": "major",
                    "confidence": 0.8,
                    "summary": "发现明显手部错误",
                    "artifact": {
                        "has_artifact": True,
                        "structural_artifact": {
                            "severity": "major",
                            "issues": [{
                                "description": "手部结构错误",
                                "location": "人物右手",
                                "bbox_2d": [500, 500, 700, 800],
                            }],
                        },
                    },
                    "text_correctness": {"has_text_error": False},
                },
            }
            (direct / "input.critic.jsonl").write_text(
                json.dumps(direct_row) + "\n",
                encoding="utf-8",
            )

            report = build_model_comparison_report(
                left,
                "Internal",
                right,
                "GPT Agent",
                output,
                direct_root=direct,
                direct_label="GPT Direct",
            )
            html = report.read_text("utf-8")

            self.assertIn("GPT Direct", html)
            self.assertIn("单轮直调不提供覆盖证明", html)
            self.assertIn("手部结构错误", html)
            self.assertIn('"success":1', html)


if __name__ == "__main__":
    unittest.main()
