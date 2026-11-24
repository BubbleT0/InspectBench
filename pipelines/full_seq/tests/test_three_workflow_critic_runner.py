from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from critic_workflows import (
    AdversarialConsensusCritic,
    EvidenceSeekingAgentCritic,
    ImageAsset,
    PassthroughImageTool,
    SpecialistCascadeCritic,
)
from u1_critic_v41.run import (
    append_jsonl,
    completed_units,
    consolidate_with_failure_record,
    consolidate_with_retries,
    consolidate_if_needed,
    iter_completed_units,
    parse_args,
    run_evaluation,
)
from u1_critic_v41.adaptive_coverage import (
    AdaptiveContentCoverageAuditor,
)
from u1_critic_v41.consolidation import (
    ConsolidationValidationError,
)
from u1_critic_v41.adaptive_image import (
    LosslessZoomImageTool,
)
from u1_critic_v41.edit_consensus import (
    EditGroundedAdversarialConsensusCritic,
)
from u1_critic_v41.text_coverage import TextCoverageAuditor
from u1_critic_v41.v4_harness import (
    HarnessedAdaptiveContentCoverageAuditor,
    HarnessedEditGroundedConsensusCritic,
    V41AdaptiveContentCoverageAuditor,
    V41EditGroundedConsensusCritic,
)
from u1_critic_v41.workflows import (
    ROLES,
    SPECIALISTS,
    TAXONOMY,
    build_critics,
    critic_context,
)


class NeverCalledModel:
    async def complete(self, request):
        raise AssertionError(f"Unexpected model call: {request.stage}")


class WorkflowRunnerTests(unittest.TestCase):
    def test_builds_the_three_original_workflows_with_six_specialists(self) -> None:
        critics = build_critics(NeverCalledModel(), PassthroughImageTool())

        self.assertEqual(
            set(critics),
            {
                "specialist_cascade",
                "evidence_seeking_agent",
                "adversarial_consensus",
            },
        )
        self.assertIsInstance(critics["specialist_cascade"], SpecialistCascadeCritic)
        self.assertIsInstance(
            critics["evidence_seeking_agent"],
            EvidenceSeekingAgentCritic,
        )
        self.assertIsInstance(
            critics["adversarial_consensus"],
            AdversarialConsensusCritic,
        )
        self.assertIsInstance(
            critics["adversarial_consensus"],
            EditGroundedAdversarialConsensusCritic,
        )
        self.assertEqual(len(SPECIALISTS), 6)
        self.assertEqual(tuple(item.artifact_type for item in SPECIALISTS), TAXONOMY)
        self.assertEqual(critic_context().taxonomy, TAXONOMY)
        self.assertIsNone(critic_context().generation_prompt)
        self.assertEqual(critic_context().metadata, {})
        self.assertEqual(
            [role.name for role in ROLES],
            [
                "anatomy_structure_topology",
                "relations_physics",
                "rendering",
                "text_coverage",
            ],
        )
        self.assertEqual(
            critics[
                "adversarial_consensus"
            ].config.max_findings_per_role,
            64,
        )
        self.assertIsInstance(
            critics["adversarial_consensus"].text_auditor,
            TextCoverageAuditor,
        )

    def test_adaptive_mode_selects_content_auditor_and_zoom_tool(self) -> None:
        critics = build_critics(
            NeverCalledModel(),
            PassthroughImageTool(),
            coverage_mode="adaptive",
        )
        consensus = critics["adversarial_consensus"]

        self.assertIsInstance(
            consensus.text_auditor,
            AdaptiveContentCoverageAuditor,
        )
        self.assertIsInstance(
            consensus.image_tool,
            LosslessZoomImageTool,
        )

    def test_harness_mode_selects_v4_critic_and_content_auditor(self) -> None:
        critics = build_critics(
            NeverCalledModel(),
            PassthroughImageTool(),
            coverage_mode="harness",
        )
        consensus = critics["adversarial_consensus"]

        self.assertIsInstance(
            consensus,
            HarnessedEditGroundedConsensusCritic,
        )
        self.assertIsInstance(
            consensus.text_auditor,
            HarnessedAdaptiveContentCoverageAuditor,
        )

    def test_v41_mode_selects_v41_critic_and_content_auditor(self) -> None:
        critics = build_critics(
            NeverCalledModel(),
            PassthroughImageTool(),
            coverage_mode="harness_v41",
        )
        consensus = critics["adversarial_consensus"]

        self.assertIsInstance(consensus, V41EditGroundedConsensusCritic)
        self.assertIsInstance(
            consensus.text_auditor,
            V41AdaptiveContentCoverageAuditor,
        )

    def test_only_successful_units_are_resumed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.jsonl"
            append_jsonl(
                path,
                {
                    "sample_id": "img_a",
                    "workflow": "specialist_cascade",
                    "status": "ok",
                },
            )
            append_jsonl(
                path,
                {
                    "sample_id": "img_b",
                    "workflow": "evidence_seeking_agent",
                    "status": "error",
                },
            )

            self.assertEqual(
                completed_units(path),
                {("img_a", "specialist_cascade")},
            )
            rows = [
                json.loads(line)
                for line in path.read_text("utf-8").splitlines()
            ]
            self.assertEqual(len(rows), 2)

    def test_unconsolidated_consensus_unit_is_not_resumed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.jsonl"
            append_jsonl(
                path,
                {
                    "sample_id": "img_partial",
                    "workflow": "adversarial_consensus",
                    "status": "ok",
                    "result": {"verdict": "artifact_present"},
                },
            )
            append_jsonl(
                path,
                {
                    "sample_id": "img_complete",
                    "workflow": "adversarial_consensus",
                    "status": "ok",
                    "result": {"verdict": "artifact_present"},
                    "consolidated_result": {"edit_regions": []},
                },
            )

            self.assertEqual(
                completed_units(path),
                {("img_complete", "adversarial_consensus")},
            )

    def test_all_requires_explicit_confirmation(self) -> None:
        with self.assertRaises(SystemExit):
            parse_args(
                [
                    "--input-root",
                    "images",
                    "--output-root",
                    "out",
                    "--all",
                ]
            )
        args = parse_args(
            [
                "--input-root",
                "images",
                "--output-root",
                "out",
                "--all",
                "--confirm-run-all",
                "YES",
            ]
        )
        self.assertTrue(args.all)

    def test_can_select_only_adversarial_consensus(self) -> None:
        args = parse_args(
            [
                "--input-root",
                "images",
                "--output-root",
                "out",
                "--limit-images",
                "1",
                "--workflow",
                "adversarial_consensus",
            ]
        )
        self.assertEqual(args.workflows, ("adversarial_consensus",))
        self.assertEqual(args.coverage_mode, "grid")

    def test_can_select_adaptive_content_coverage(self) -> None:
        args = parse_args(
            [
                "--input-root",
                "images",
                "--output-root",
                "out",
                "--limit-images",
                "1",
                "--workflow",
                "adversarial_consensus",
                "--coverage-mode",
                "adaptive",
            ]
        )

        self.assertEqual(args.coverage_mode, "adaptive")

    def test_can_select_v4_harness_coverage(self) -> None:
        args = parse_args(
            [
                "--input-root",
                "images",
                "--output-root",
                "out",
                "--limit-images",
                "1",
                "--workflow",
                "adversarial_consensus",
                "--coverage-mode",
                "harness",
            ]
        )

        self.assertEqual(args.coverage_mode, "harness")

    def test_can_select_v41_harness_coverage(self) -> None:
        args = parse_args(
            [
                "--input-root",
                "images",
                "--output-root",
                "out",
                "--limit-images",
                "1",
                "--workflow",
                "adversarial_consensus",
                "--coverage-mode",
                "harness_v41",
            ]
        )

        self.assertEqual(args.coverage_mode, "harness_v41")
        self.assertEqual(args.unit_retries, 2)
        self.assertEqual(args.consolidation_retries, 2)

    def test_can_set_bounded_image_level_concurrency(self) -> None:
        args = parse_args(
            [
                "--input-root",
                "images",
                "--output-root",
                "out",
                "--limit-images",
                "1",
                "--max-images-in-flight",
                "4",
            ]
        )

        self.assertEqual(args.max_images_in_flight, 4)

    def test_workflow_selection_defaults_to_all_three(self) -> None:
        args = parse_args(
            [
                "--input-root",
                "images",
                "--output-root",
                "out",
                "--limit-images",
                "1",
            ]
        )
        self.assertEqual(
            args.workflows,
            (
                "specialist_cascade",
                "evidence_seeking_agent",
                "adversarial_consensus",
            ),
        )


class WorkflowPostprocessTests(unittest.IsolatedAsyncioTestCase):
    async def test_pending_images_run_concurrently_with_explicit_bound(
        self,
    ) -> None:
        active = 0
        max_active = 0
        started_two = asyncio.Event()
        release = asyncio.Event()

        async def run_unit(record, workflow):
            nonlocal active, max_active
            active += 1
            max_active = max(max_active, active)
            if active == 2:
                started_two.set()
            await release.wait()
            active -= 1
            return {"sample_id": record, "workflow": workflow}

        async def collect():
            return [
                outcome
                async for _, _, outcome in iter_completed_units(
                    [("a", "w"), ("b", "w"), ("c", "w")],
                    run_unit,
                    max_units_in_flight=2,
                )
            ]

        task = asyncio.create_task(collect())
        await asyncio.wait_for(started_two.wait(), timeout=1)
        self.assertEqual(active, 2)
        release.set()
        outcomes = await task

        self.assertEqual(max_active, 2)
        self.assertEqual(
            {value["sample_id"] for value in outcomes},
            {"a", "b", "c"},
        )

    async def test_run_config_persists_coverage_mode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_root = root / "empty-input"
            output_root = root / "output"
            input_root.mkdir()
            args = parse_args(
                [
                    "--input-root",
                    str(input_root),
                    "--output-root",
                    str(output_root),
                    "--limit-images",
                    "1",
                    "--workflow",
                    "adversarial_consensus",
                    "--coverage-mode",
                    "adaptive",
                ]
            )

            await run_evaluation(args)

            config = json.loads(
                (output_root / "run_config.json").read_text("utf-8")
            )
            self.assertEqual(config["coverage_mode"], "adaptive")
            self.assertEqual(
                config["bbox_coordinate_space"],
                "bbox_1000",
            )

    async def test_only_consensus_is_consolidated_by_default(self) -> None:
        class FakeConsolidator:
            async def consolidate(self, image, result):
                return {"edit_regions": [{"edit_region_id": image.id}]}

        image = ImageAsset(id="img_a", data=b"image")
        consolidator = FakeConsolidator()

        consensus = await consolidate_if_needed(
            "adversarial_consensus",
            image,
            {"findings": []},
            consolidator,
            skip=False,
        )
        specialist = await consolidate_if_needed(
            "specialist_cascade",
            image,
            {"findings": []},
            consolidator,
            skip=False,
        )

        self.assertEqual(
            consensus,
            {"edit_regions": [{"edit_region_id": "img_a"}]},
        )
        self.assertIsNone(specialist)

    async def test_consolidation_cannot_hide_partial_coverage(self) -> None:
        class FakeConsolidator:
            async def consolidate(self, image, result):
                return {
                    "verdict": "artifact_present",
                    "confidence": 0.9,
                    "edit_regions": [],
                }

        consolidated = await consolidate_if_needed(
            "adversarial_consensus",
            ImageAsset(id="img_a", data=b"image"),
            {
                "verdict": "uncertain",
                "findings": [],
                "metadata": {
                    "coverage_status": "failed",
                    "harness_abstention_reason": "实体通道未完成。",
                },
            },
            FakeConsolidator(),
            skip=False,
        )

        self.assertEqual(consolidated["quality_verdict"], "artifact_present")
        self.assertEqual(consolidated["verdict"], "uncertain")
        self.assertEqual(consolidated["coverage_status"], "failed")
        self.assertTrue(consolidated["findings_non_exhaustive"])

    async def test_consolidation_retries_finish_inside_live_unit(self) -> None:
        class FlakyConsolidator:
            def __init__(self):
                self.calls = 0

            async def consolidate(self, image, result):
                self.calls += 1
                if self.calls < 3:
                    raise ConsolidationValidationError(["结构输出未闭环"])
                return {"verdict": "no_artifact", "edit_regions": []}

        consolidator = FlakyConsolidator()
        result = await consolidate_with_retries(
            workflow="adversarial_consensus",
            image=ImageAsset(id="img_a", data=b"image"),
            result={"findings": []},
            consolidator=consolidator,
            skip=False,
            max_retries=2,
        )

        self.assertEqual(consolidator.calls, 3)
        self.assertEqual(result["verdict"], "no_artifact")

    async def test_consolidation_failure_preserves_raw_critic_result(
        self,
    ) -> None:
        class FailingConsolidator:
            async def consolidate(self, image, result):
                raise ConsolidationValidationError(
                    ["大框合并了空间独立文字实体"]
                )

        with tempfile.TemporaryDirectory() as directory:
            error_path = Path(directory) / "consolidation_errors.jsonl"
            consolidated, error = await consolidate_with_failure_record(
                workflow="adversarial_consensus",
                image=ImageAsset(id="img_a", data=b"image"),
                result={"verdict": "artifact_present", "findings": []},
                consolidator=FailingConsolidator(),
                skip=False,
                error_path=error_path,
                sample_id="img_a",
                model="model",
            )
            rows = [
                json.loads(line)
                for line in error_path.read_text("utf-8").splitlines()
            ]

        self.assertIsNone(consolidated)
        self.assertEqual(
            error["error_type"],
            "ConsolidationValidationError",
        )
        self.assertEqual(rows[0]["sample_id"], "img_a")
        self.assertIn("空间独立文字实体", rows[0]["error"])


if __name__ == "__main__":
    unittest.main()
