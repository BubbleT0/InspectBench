from __future__ import annotations

import hashlib
import unittest

from u1_critic_v41.translate_existing import (
    _waves,
    collect_translatable_texts,
    localize_value,
    text_id,
)


class TranslationTests(unittest.TestCase):
    def test_waves_bound_parallel_batches_without_dropping_order(self) -> None:
        self.assertEqual(
            _waves([["a"], ["b"], ["c"], ["d"], ["e"]], 2),
            [[["a"], ["b"]], [["c"], ["d"]], [["e"]]],
        )

    def test_collects_only_user_facing_natural_language(self) -> None:
        value = {
            "stage": "specialist.detect",
            "artifact_type": "structural_artifact",
            "observation": "A hand has six fingers.",
            "metadata": {
                "verifier_reason": "The extra finger is directly visible.",
                "candidate_id": "c0",
            },
        }

        texts = collect_translatable_texts(value)

        self.assertEqual(
            texts,
            {
                "A hand has six fingers.",
                "The extra finger is directly visible.",
            },
        )

    def test_localization_preserves_structure_and_replaces_known_text(self) -> None:
        source = "A hand has six fingers."
        translations = {text_id(source): "一只手清楚可见六根手指。"}
        value = {
            "observation": source,
            "confidence": 0.9,
            "bbox_norm": [100, 200, 300, 400],
        }

        localized = localize_value(value, translations)

        self.assertEqual(localized["observation"], "一只手清楚可见六根手指。")
        self.assertEqual(localized["confidence"], 0.9)
        self.assertEqual(localized["bbox_norm"], [100, 200, 300, 400])
        self.assertEqual(text_id(source), hashlib.sha256(source.encode()).hexdigest())


if __name__ == "__main__":
    unittest.main()
