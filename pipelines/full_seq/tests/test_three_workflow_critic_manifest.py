from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from u1_critic_v41.input_manifest import (
    discover_images,
    load_images_from_jsonl,
    manifest_rows,
    select_images_from_jsonl,
    to_image_asset,
)


class InputManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        labeled = self.root / "D1_Artifact" / "A2_StructuralArtifact"
        labeled.mkdir(parents=True)
        (labeled / "secret_name.jpg").write_bytes(b"jpeg bytes")
        (labeled / "ignored.txt").write_text("not an image", encoding="utf-8")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_discovery_is_stable_and_keeps_labels_only_in_manifest(self) -> None:
        first = discover_images(self.root)
        second = discover_images(self.root)

        self.assertEqual([item.image_id for item in first], [item.image_id for item in second])
        self.assertEqual(len(first), 1)
        self.assertEqual(
            first[0].reference_label,
            "D1_Artifact/A2_StructuralArtifact",
        )
        self.assertEqual(
            manifest_rows(first)[0]["reference_label"],
            "D1_Artifact/A2_StructuralArtifact",
        )

    def test_image_asset_contains_pixels_but_no_source_identity(self) -> None:
        record = discover_images(self.root)[0]
        asset = to_image_asset(record)

        self.assertIsNone(asset.path)
        self.assertEqual(asset.data, b"jpeg bytes")
        self.assertEqual(asset.metadata, {})
        serialized = repr(asset)
        self.assertNotIn("StructuralArtifact", serialized)
        self.assertNotIn("secret_name.jpg", serialized)

    def test_jsonl_selection_preserves_explicit_order(self) -> None:
        second_dir = self.root / "other"
        second_dir.mkdir()
        (second_dir / "second.png").write_bytes(b"png bytes")
        records = discover_images(self.root)
        selection = self.root / "selection.jsonl"
        selection.write_text(
            "\n".join(
                json.dumps({"image_absolute_path": str(record.source_path)})
                for record in reversed(records)
            )
            + "\n",
            encoding="utf-8",
        )

        selected = select_images_from_jsonl(records, selection)

        self.assertEqual(selected, list(reversed(records)))

    def test_jsonl_direct_load_supports_multiple_roots(self) -> None:
        outside = Path(self.temp.name).parent / f"{self.root.name}_outside.png"
        try:
            outside.write_bytes(b"png bytes")
            inside = discover_images(self.root)[0].source_path
            selection = self.root / "selection.jsonl"
            selection.write_text(
                "\n".join(
                    json.dumps({"image_absolute_path": str(path)})
                    for path in (outside, inside)
                )
                + "\n",
                encoding="utf-8",
            )
            selected = load_images_from_jsonl(selection)
            self.assertEqual([x.source_path for x in selected], [outside, inside])
            self.assertTrue(all(x.reference_label == "" for x in selected))
        finally:
            outside.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
