from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from critic_workflows import (
    BBox,
    Finding,
    ImageAsset,
    PassthroughImageTool,
)
from u1_critic_v63 import run as run_module
from u1_critic_v63.v42_fast import (
    V42EditGroundedConsensusCritic,
)
from u1_critic_v63.v4_harness import (
    V41EditGroundedConsensusCritic,
)
from u1_critic_v63.v62_adaptive import (
    V62AdaptiveContentCoverageAuditor,
)
from u1_critic_v63.v63_selective import (
    V63SelectiveCoverageAuditor,
)
from u1_critic_v63.workflows import build_critics


class NeverCalledModel:
    async def complete(self, request):
        raise AssertionError(f"unexpected model call: {request.stage}")


def finding(
    *,
    batched: bool = True,
    full_frame: bool = False,
    source: str = "role:text_coverage",
) -> Finding:
    value = Finding(
        region_id="region",
        bbox=BBox(0, 0, 1000, 1000) if full_frame else BBox(100, 100, 400, 400),
        artifact_type="structural_artifact",
        observation="结构明显错误",
        abnormality="部件连接断裂",
        confidence=0.9,
        source=source,
        metadata={
            "v42_batched_inspection": batched,
            "defect_present": True,
            "materiality": "major",
            "evidence_basis": "visible_only",
            "grounding_status": "pending",
            "content_kind": "entity",
        },
    )
    return value


class V63ProfileTests(unittest.TestCase):
    def test_v63_enables_coverage_authoritative_profile(self) -> None:
        args = run_module.parse_args(
            [
                "--input-root", "images",
                "--output-root", "out",
                "--limit-images", "1",
                "--coverage-mode", "harness_v42",
                "--v63",
                "--max-tokens", "16000",
                "--consolidation-max-tokens", "16000",
                "--v42-text-batch-size", "2",
                "--v42-text-crosscheck-batch-size", "2",
                "--v42-artifact-batch-size", "2",
                "--v42-challenge-batch-size", "2",
                "--max-images-in-flight", "4",
            ]
        )

        self.assertTrue(run_module.v63_enabled(args))
        self.assertFalse(run_module.v62_enabled(args))
        self.assertTrue(run_module.structured_recovery_enabled(args))
        self.assertTrue(run_module.component_repair_enabled(args))
        self.assertEqual(run_module.pipeline_version_for_args(args), "v6.3")
        self.assertEqual(args.max_images_in_flight, 4)
        self.assertTrue(args.v42_quality_first)
        self.assertTrue(args.v42_coverage_only)
        self.assertTrue(args.v42_trust_preverified)
        self.assertTrue(args.v42_deterministic_judge)
        self.assertTrue(args.v42_text_crosscheck)
        self.assertFalse(args.v42_deterministic_source_consolidation)

    def test_v63_selects_v62_auditor_and_only_coverage_role(self) -> None:
        critic = build_critics(
            NeverCalledModel(),
            PassthroughImageTool(),
            coverage_mode="harness_v42",
            v62_adaptive=True,
            v42_coverage_only=True,
            v42_trust_preverified=True,
            v42_deterministic_judge=True,
            v42_pipeline_version="v6.3",
        )["adversarial_consensus"]

        self.assertIsInstance(critic, V42EditGroundedConsensusCritic)
        self.assertIsInstance(
            critic.text_auditor,
            V62AdaptiveContentCoverageAuditor,
        )
        self.assertEqual(
            [role.name for role in critic.config.roles],
            ["text_coverage"],
        )
        self.assertTrue(critic.trust_preverified)
        self.assertTrue(critic.deterministic_judge)
        self.assertEqual(critic.pipeline_version, "v6.3")

    def test_v63_allows_combined_planning_ablations(self) -> None:
        args = run_module.parse_args(
            [
                "--input-root", "images",
                "--output-root", "out",
                "--limit-images", "1",
                "--coverage-mode", "harness_v42",
                "--v63",
                "--max-tokens", "16000",
                "--consolidation-max-tokens", "16000",
                "--v42-text-batch-size", "2",
                "--v42-text-crosscheck-batch-size", "2",
                "--v42-artifact-batch-size", "2",
                "--v42-challenge-batch-size", "2",
                "--v42-combined-inventory",
                "--v42-combined-coverage-audit",
            ]
        )

        self.assertTrue(args.v42_combined_inventory)
        self.assertTrue(args.v42_combined_coverage_audit)

    def test_v63_selective_review_selects_specialized_auditor(self) -> None:
        critic = build_critics(
            NeverCalledModel(),
            PassthroughImageTool(),
            coverage_mode="harness_v42",
            v62_adaptive=True,
            v63_selective_review=True,
            v42_coverage_only=True,
            v42_pipeline_version="v6.3",
        )["adversarial_consensus"]

        self.assertIsInstance(
            critic.text_auditor,
            V63SelectiveCoverageAuditor,
        )

    def test_v63_entity_safe_review_requires_selective_review(self) -> None:
        with self.assertRaises(SystemExit):
            run_module.parse_args(
                [
                    "--input-root", "images",
                    "--output-root", "out",
                    "--limit-images", "1",
                    "--coverage-mode", "harness_v42",
                    "--v63",
                    "--v63-entity-safe-review",
                ]
            )

        args = run_module.parse_args(
            [
                "--input-root", "images",
                "--output-root", "out",
                "--limit-images", "1",
                "--coverage-mode", "harness_v42",
                "--v63",
                "--v63-selective-review",
                "--v63-entity-safe-review",
                "--max-images-in-flight", "4",
                "--v42-text-batch-size", "2",
                "--v42-text-crosscheck-batch-size", "2",
                "--v42-artifact-batch-size", "2",
                "--v42-challenge-batch-size", "2",
                "--max-tokens", "16000",
                "--consolidation-max-tokens", "16000",
            ]
        )

        self.assertTrue(args.v63_entity_safe_review)
        self.assertEqual(args.max_images_in_flight, 4)

    def test_v63_artifact_scout_selects_one_verified_global_role(self) -> None:
        args = run_module.parse_args(
            [
                "--input-root", "images",
                "--output-root", "out",
                "--limit-images", "1",
                "--coverage-mode", "harness_v42",
                "--v63",
                "--v63-selective-review",
                "--v63-artifact-scout",
                "--max-images-in-flight", "4",
                "--v42-text-batch-size", "2",
                "--v42-text-crosscheck-batch-size", "2",
                "--v42-artifact-batch-size", "2",
                "--v42-challenge-batch-size", "2",
                "--max-tokens", "16000",
                "--consolidation-max-tokens", "16000",
            ]
        )

        self.assertTrue(args.v63_artifact_scout)
        self.assertFalse(args.v42_coverage_only)
        self.assertTrue(args.v42_single_artifact_proposer)
        self.assertEqual(args.max_images_in_flight, 4)

        critic = build_critics(
            NeverCalledModel(),
            PassthroughImageTool(),
            coverage_mode="harness_v42",
            v62_adaptive=True,
            v63_selective_review=True,
            v63_artifact_scout=True,
            v42_trust_preverified=True,
            v42_deterministic_judge=True,
            v42_pipeline_version="v6.3",
        )["adversarial_consensus"]
        self.assertEqual(
            [role.name for role in critic.config.roles],
            ["artifact_scout", "text_coverage"],
        )

    def test_v63_dual_scout_selects_two_specialists(self) -> None:
        args = run_module.parse_args(
            [
                "--input-root", "images",
                "--output-root", "out",
                "--limit-images", "1",
                "--coverage-mode", "harness_v42",
                "--v63",
                "--v63-selective-review",
                "--v63-dual-artifact-scout",
                "--v42-text-batch-size", "2",
                "--v42-text-crosscheck-batch-size", "2",
                "--v42-artifact-batch-size", "2",
                "--v42-challenge-batch-size", "2",
                "--max-tokens", "16000",
                "--consolidation-max-tokens", "16000",
            ]
        )
        self.assertTrue(args.v63_dual_artifact_scout)
        self.assertFalse(args.v42_coverage_only)

        critic = build_critics(
            NeverCalledModel(),
            PassthroughImageTool(),
            coverage_mode="harness_v42",
            v62_adaptive=True,
            v63_selective_review=True,
            v63_dual_artifact_scout=True,
            v42_trust_preverified=True,
            v42_deterministic_judge=True,
            v42_pipeline_version="v6.3",
        )["adversarial_consensus"]
        self.assertEqual(
            [role.name for role in critic.config.roles],
            [
                "artifact_structure_scout",
                "artifact_rendering_scout",
                "text_coverage",
            ],
        )

    def test_v63_dense_path_profile_is_isolated(self) -> None:
        argv = [
            "--input-root", "images",
            "--output-root", "out",
            "--limit-images", "1",
            "--coverage-mode", "harness_v42",
            "--v63",
            "--v63-selective-review",
            "--v63-dual-artifact-scout",
            "--v63-dense-path-latency",
            "--v42-text-batch-size", "2",
            "--v42-text-crosscheck-batch-size", "2",
            "--v42-artifact-batch-size", "2",
            "--v42-challenge-batch-size", "2",
            "--max-tokens", "16000",
            "--consolidation-max-tokens", "16000",
            "--max-images-in-flight", "4",
        ]

        args = run_module.parse_args(argv)

        self.assertTrue(args.v63_dense_path_latency)
        self.assertEqual(
            run_module.pipeline_version_for_args(args),
            "v6.3-i",
        )
        self.assertEqual(args.max_images_in_flight, 4)

        legacy_args = run_module.parse_args(
            [value for value in argv if value != "--v63-dense-path-latency"]
        )
        self.assertFalse(legacy_args.v63_dense_path_latency)
        self.assertEqual(
            run_module.pipeline_version_for_args(legacy_args),
            "v6.3",
        )

        critic = build_critics(
            NeverCalledModel(),
            PassthroughImageTool(),
            coverage_mode="harness_v42",
            v63_selective_review=True,
            v63_dual_artifact_scout=True,
            v63_dense_path_latency=True,
            v42_pipeline_version="v6.3-i",
        )["adversarial_consensus"]
        self.assertTrue(critic.text_auditor.dense_path_latency)
        self.assertEqual(critic.pipeline_version, "v6.3-i")

    def test_v63_dense_path_requires_full_h_profile(self) -> None:
        with self.assertRaises(SystemExit):
            run_module.parse_args(
                [
                    "--input-root", "images",
                    "--output-root", "out",
                    "--limit-images", "1",
                    "--coverage-mode", "harness_v42",
                    "--v63",
                    "--v63-dense-path-latency",
                ]
            )

    def test_v63i_uses_four_region_batches_only_for_dense_channels(self) -> None:
        dense = V63SelectiveCoverageAuditor(
            NeverCalledModel(),
            PassthroughImageTool(),
            text_batch_size=2,
            text_crosscheck_batch_size=2,
            artifact_batch_size=2,
            dense_path_latency=True,
        )
        legacy = V63SelectiveCoverageAuditor(
            NeverCalledModel(),
            PassthroughImageTool(),
            text_batch_size=2,
            text_crosscheck_batch_size=2,
            artifact_batch_size=2,
        )

        self.assertEqual(
            dense._inspection_batch_size("text", 12, configured=2),
            4,
        )
        self.assertEqual(
            dense._inspection_batch_size("entity", 12, configured=2),
            4,
        )
        self.assertEqual(
            dense._inspection_batch_size("text", 11, configured=2),
            2,
        )
        self.assertEqual(
            legacy._inspection_batch_size("text", 30, configured=2),
            2,
        )
        self.assertEqual(dense._failed_batch_split_index(4), 2)

    def test_v63i_compacts_inventory_and_hydrates_local_descriptions(self) -> None:
        dense = V63SelectiveCoverageAuditor(
            NeverCalledModel(),
            PassthroughImageTool(),
            dense_path_latency=True,
        )
        legacy = V63SelectiveCoverageAuditor(
            NeverCalledModel(),
            PassthroughImageTool(),
        )

        dense_item = dense._inventory_schema_for_request()["properties"][
            "regions"
        ]["items"]
        legacy_item = legacy._inventory_schema_for_request()["properties"][
            "regions"
        ]["items"]
        self.assertNotIn("description", dense_item["required"])
        self.assertNotIn("triage_reason", dense_item["required"])
        self.assertNotIn("description", dense_item["properties"])
        self.assertNotIn("triage_reason", dense_item["properties"])
        self.assertIn("description", legacy_item["required"])
        self.assertIn("triage_reason", legacy_item["required"])

        hydrated = dense._hydrate_compact_inventory_region(
            {
                "region_id": "title_1",
                "semantic_group_id": "hero_title",
                "semantic_unit_kind": "complete_title",
                "content_kind": "text",
                "review_required": True,
            },
            "text",
        )
        self.assertEqual(
            hydrated["description"],
            "文字语义单元 hero_title（complete_title）",
        )
        self.assertEqual(
            hydrated["triage_reason"],
            "模型要求对该区域执行局部复核。",
        )


class V63PreverificationTests(unittest.IsolatedAsyncioTestCase):
    async def test_overlapping_dual_scout_findings_skip_skeptic(self) -> None:
        critic = build_critics(
            NeverCalledModel(),
            PassthroughImageTool(),
            coverage_mode="harness_v42",
            v63_dual_artifact_scout=True,
            v42_trust_preverified=True,
            v42_deterministic_judge=True,
            v42_pipeline_version="v6.3",
        )["adversarial_consensus"]
        structure = finding(source="role:artifact_structure_scout")
        rendering = finding(source="role:artifact_rendering_scout")
        rendering.bbox = BBox(150, 150, 450, 450)

        challenged = await critic._challenge_candidates(
            ImageAsset(id="image", data=b"pixels"),
            SimpleNamespace(reference_images=()),
            {"c0": structure, "c1": rendering},
            semaphore=None,
        )

        self.assertEqual(
            [status.value for _, status, _ in challenged],
            ["supported", "supported"],
        )
        self.assertEqual(critic._last_peer_scout_consensus_count, 2)
        self.assertEqual(critic._last_skeptic_request_candidate_count, 0)

    def test_incompatible_dual_scout_findings_do_not_form_consensus(self) -> None:
        critic = build_critics(
            NeverCalledModel(),
            PassthroughImageTool(),
            coverage_mode="harness_v42",
            v63_dual_artifact_scout=True,
            v42_pipeline_version="v6.3",
        )["adversarial_consensus"]
        structure = finding(source="role:artifact_structure_scout")
        rendering = finding(source="role:artifact_rendering_scout")
        rendering.artifact_type = "rendering_artifact"

        supported = critic._peer_supported_scout_candidates(
            {"c0": structure, "c1": rendering}
        )

        self.assertEqual(supported, set())

    async def test_locally_verified_artifact_skips_skeptic(self) -> None:
        critic = build_critics(
            NeverCalledModel(),
            PassthroughImageTool(),
            coverage_mode="harness_v42",
            v42_coverage_only=True,
            v42_trust_preverified=True,
            v42_deterministic_judge=True,
            v42_pipeline_version="v6.3",
        )["adversarial_consensus"]
        candidate = finding()

        challenged = await critic._challenge_candidates(
            ImageAsset(id="image", data=b"pixels"),
            SimpleNamespace(reference_images=()),
            {"c0": candidate},
            semaphore=None,
        )

        self.assertEqual(challenged[0][0], "c0")
        self.assertEqual(challenged[0][1].value, "supported")
        self.assertEqual(critic._last_preverified_count, 1)
        self.assertEqual(critic._last_skeptic_request_candidate_count, 0)

    async def test_full_frame_local_finding_is_not_preverified_as_valid(self) -> None:
        critic = build_critics(
            NeverCalledModel(),
            PassthroughImageTool(),
            coverage_mode="harness_v42",
            v42_coverage_only=True,
            v42_trust_preverified=True,
            v42_deterministic_judge=True,
            v42_pipeline_version="v6.3",
        )["adversarial_consensus"]
        candidate = finding(full_frame=True)

        challenged = await critic._challenge_candidates(
            ImageAsset(id="image", data=b"pixels"),
            SimpleNamespace(reference_images=()),
            {"c0": candidate},
            semaphore=None,
        )

        self.assertNotEqual(challenged[0][1].value, "supported")

    async def test_v63_terminal_metadata_reports_skipped_stages(self) -> None:
        critic = build_critics(
            NeverCalledModel(),
            PassthroughImageTool(),
            coverage_mode="harness_v42",
            v42_coverage_only=True,
            v42_trust_preverified=True,
            v42_deterministic_judge=True,
            v42_pipeline_version="v6.3",
        )["adversarial_consensus"]
        critic._last_preverified_count = 3
        critic._last_skeptic_request_candidate_count = 0
        critic.text_auditor.v63_triage_metrics = {
            "total_regions": 9,
            "reviewed_regions": 4,
            "skipped_regions": 5,
        }
        terminal = SimpleNamespace(metadata={})

        with patch.object(
            V41EditGroundedConsensusCritic,
            "critique",
            new=AsyncMock(return_value=terminal),
        ):
            result = await critic.critique(ImageAsset(id="image", data=b"pixels"))

        self.assertEqual(result.metadata["v63_preverified_finding_count"], 3)
        self.assertEqual(result.metadata["v63_skeptic_candidate_count"], 0)
        self.assertTrue(result.metadata["v63_deterministic_judge"])
        self.assertTrue(result.metadata["v63_online_consolidation_required"])
        self.assertEqual(
            result.metadata["v63_triage_metrics"]["skipped_regions"],
            5,
        )


class V63SelectiveReviewTests(unittest.IsolatedAsyncioTestCase):
    def test_entity_safe_prompt_is_independent_and_explicit(self) -> None:
        safe = V63SelectiveCoverageAuditor(
            NeverCalledModel(),
            PassthroughImageTool(),
            entity_safe_review=True,
        )
        legacy = V63SelectiveCoverageAuditor(
            NeverCalledModel(),
            PassthroughImageTool(),
        )
        entity = {
            "content_kind": "entity",
            "v63_inventory_kind": "entity",
            "description": "伸出的手掌，五指结构正常",
        }

        self.assertFalse(
            safe._include_inventory_description(entity, "primary")
        )
        self.assertTrue(
            legacy._include_inventory_description(entity, "primary")
        )
        safe_rule = safe._artifact_inspection_rule([entity])
        legacy_rule = legacy._artifact_inspection_rule([entity])
        self.assertIn("逐项核验", safe_rule)
        self.assertIn("部件数量", safe_rule)
        self.assertIn("连接", safe_rule)
        self.assertIn("接触", safe_rule)
        self.assertNotIn("逐项核验", legacy_rule)

    async def test_entity_safe_review_forces_only_entity_inventory_regions(
        self,
    ) -> None:
        auditor = V63SelectiveCoverageAuditor(
            NeverCalledModel(),
            PassthroughImageTool(),
            entity_safe_review=True,
        )
        regions = [
            {
                "region_id": "normal_text",
                "content_kind": "text",
                "v63_inventory_kind": "text",
                "v63_review_required": False,
                "v63_triage_confidence": 0.95,
            },
            {
                "region_id": "normal_layout",
                "content_kind": "layout",
                "v63_inventory_kind": "entity",
                "v63_review_required": False,
                "v63_triage_confidence": 0.95,
            },
            {
                "region_id": "normal_entity",
                "content_kind": "entity",
                "v63_inventory_kind": "entity",
                "v63_review_required": False,
                "v63_triage_confidence": 0.95,
            },
            {
                "region_id": "suspicious_entity",
                "content_kind": "entity",
                "v63_inventory_kind": "entity",
                "v63_review_required": True,
                "v63_triage_confidence": 0.9,
            },
        ]
        selected_result = (None, 0, False)

        with patch(
            "u1_critic_v63.v62_adaptive."
            "V62AdaptiveContentCoverageAuditor._inspect_regions_core",
            new=AsyncMock(return_value=[selected_result] * 2),
        ) as inspect:
            output = await auditor._inspect_regions_core(
                ImageAsset(id="image", data=b"pixels"),
                regions,
            )

        self.assertEqual(output, [(None, 0, False)] * 4)
        inspected = inspect.await_args.args[1]
        self.assertEqual(
            [region["region_id"] for region in inspected],
            ["normal_entity", "suspicious_entity"],
        )
        self.assertEqual(
            auditor.v63_triage_metrics["forced_entity_regions"],
            1,
        )
        self.assertEqual(auditor.v63_triage_metrics["skipped_regions"], 2)

    async def test_only_uncertain_or_suspicious_regions_are_locally_reviewed(self) -> None:
        auditor = V63SelectiveCoverageAuditor(
            NeverCalledModel(),
            PassthroughImageTool(),
        )
        regions = [
            {
                "region_id": "normal",
                "content_kind": "entity",
                "v63_review_required": False,
                "v63_triage_confidence": 0.95,
            },
            {
                "region_id": "suspicious",
                "content_kind": "text",
                "v63_review_required": True,
                "v63_triage_confidence": 0.9,
            },
            {
                "region_id": "uncertain",
                "content_kind": "entity",
                "v63_review_required": False,
                "v63_triage_confidence": 0.6,
            },
            {
                "region_id": "recovered",
                "content_kind": "entity",
            },
        ]
        selected_result = (None, 0, False)

        with patch(
            "u1_critic_v63.v62_adaptive."
            "V62AdaptiveContentCoverageAuditor._inspect_regions_core",
            new=AsyncMock(return_value=[selected_result] * 3),
        ) as inspect:
            output = await auditor._inspect_regions_core(
                ImageAsset(id="image", data=b"pixels"),
                regions,
            )

        self.assertEqual(output, [(None, 0, False)] * 4)
        inspected = inspect.await_args.args[1]
        self.assertEqual(
            [region["region_id"] for region in inspected],
            ["suspicious", "uncertain", "recovered"],
        )
        self.assertEqual(auditor.v63_triage_metrics["skipped_regions"], 1)
        self.assertEqual(auditor.v63_triage_metrics["reviewed_regions"], 3)


if __name__ == "__main__":
    unittest.main()
