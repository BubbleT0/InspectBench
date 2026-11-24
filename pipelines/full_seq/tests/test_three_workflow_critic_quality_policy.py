from __future__ import annotations

import unittest

from u1_critic_v41.quality_policy import candidate_is_core


class CoreQualityPolicyTests(unittest.TestCase):
    def test_accepts_only_visible_major_or_critical_defects(self) -> None:
        base = {
            "defect_present": True,
            "materiality": "major",
            "evidence_basis": "visible_only",
            "confidence": 0.81,
        }

        self.assertTrue(candidate_is_core(base))
        for patch in (
            {"defect_present": False},
            {"materiality": "minor"},
            {"materiality": "none"},
            {"evidence_basis": "external_knowledge"},
            {"evidence_basis": "aesthetic_preference"},
            {"confidence": 0.6},
        ):
            with self.subTest(patch=patch):
                self.assertFalse(candidate_is_core({**base, **patch}))

    def test_low_confidence_text_can_reach_crop_verification_only(self) -> None:
        value = {
            "defect_present": True,
            "materiality": "major",
            "evidence_basis": "visible_only",
            "confidence": 0.45,
        }

        self.assertFalse(candidate_is_core(value))
        self.assertTrue(
            candidate_is_core(
                value,
                allow_low_confidence_text=True,
            )
        )

    def test_low_confidence_exception_does_not_bypass_evidence_or_materiality(
        self,
    ) -> None:
        for value in (
            {
                "defect_present": False,
                "materiality": "major",
                "evidence_basis": "visible_only",
                "confidence": 0.4,
            },
            {
                "defect_present": True,
                "materiality": "minor",
                "evidence_basis": "visible_only",
                "confidence": 0.4,
            },
            {
                "defect_present": True,
                "materiality": "major",
                "evidence_basis": "external_knowledge",
                "confidence": 0.4,
            },
        ):
            with self.subTest(value=value):
                self.assertFalse(
                    candidate_is_core(
                        value,
                        allow_low_confidence_text=True,
                    )
                )


if __name__ == "__main__":
    unittest.main()
