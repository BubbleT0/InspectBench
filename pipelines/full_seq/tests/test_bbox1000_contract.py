from __future__ import annotations

from io import BytesIO
import math
import unittest

from critic_workflows import BBox, ImageAsset, PillowImageTool, Region
from critic_workflows.evidence_agent import _ACTION_SCHEMA
from critic_workflows.specialist import _DETECTION_SCHEMA
from u1_critic_v41.adaptive_coverage import (
    _BBOX_SCHEMA as ADAPTIVE_BBOX_SCHEMA,
)
from u1_critic_v41.consolidation import (
    _BBOX_SCHEMA as CONSOLIDATION_BBOX_SCHEMA,
)
from u1_critic_v41.edit_consensus import (
    _BBOX_SCHEMA as EDIT_BBOX_SCHEMA,
)
from u1_critic_v41.model_adapter import (
    NORMALIZED_BBOX_SCHEMA,
)
from u1_critic_v41.text_coverage import (
    _BBOX_SCHEMA as TEXT_BBOX_SCHEMA,
)


def png_asset(width: int, height: int) -> ImageAsset:
    from PIL import Image

    buffer = BytesIO()
    Image.new("RGB", (width, height), (12, 34, 56)).save(
        buffer,
        format="PNG",
    )
    return ImageAsset(
        id=f"image-{width}x{height}",
        data=buffer.getvalue(),
        mime_type="image/png",
    )


class BBox1000ContractTests(unittest.IsolatedAsyncioTestCase):
    def test_all_model_bbox_schemas_require_integer_bbox1000(self) -> None:
        schemas = (
            NORMALIZED_BBOX_SCHEMA,
            ADAPTIVE_BBOX_SCHEMA,
            TEXT_BBOX_SCHEMA,
            EDIT_BBOX_SCHEMA,
            CONSOLIDATION_BBOX_SCHEMA,
            _ACTION_SCHEMA["properties"]["bbox_norm"],
            _DETECTION_SCHEMA["properties"]["bbox_norm"],
        )

        for schema in schemas:
            with self.subTest(schema=schema):
                self.assertEqual(
                    schema["items"],
                    {"type": "integer", "minimum": 0, "maximum": 1000},
                )

    def test_full_image_and_serialization_use_integer_bbox1000(self) -> None:
        bbox = BBox.full_image()

        self.assertEqual(bbox.as_list(), [0, 0, 1000, 1000])
        self.assertTrue(all(isinstance(value, int) for value in bbox.as_list()))

    def test_integer_valued_floats_are_normalized_to_integers(self) -> None:
        bbox = BBox.from_value([100.0, 200.0, 700.0, 800.0])

        self.assertEqual(bbox.as_list(), [100, 200, 700, 800])
        self.assertTrue(all(isinstance(value, int) for value in bbox.as_list()))

    def test_invalid_bbox_values_are_rejected_without_rounding(self) -> None:
        invalid_values = (
            [0, 0, 1001, 1000],
            [0, 0, 1.5, 100],
            [0, 0, math.nan, 100],
            [0, 0, math.inf, 100],
            [False, 0, 100, 100],
            [100, 0, 100, 100],
        )

        for value in invalid_values:
            with self.subTest(value=value):
                with self.assertRaises((TypeError, ValueError)):
                    BBox.from_value(value)

    async def test_pillow_crop_converts_bbox1000_only_at_pixel_boundary(
        self,
    ) -> None:
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("Pillow optional dependency is not installed")
        region = Region("crop", BBox(100, 200, 701, 801))

        crop = await PillowImageTool().crop(
            png_asset(101, 103),
            region,
        )

        self.assertEqual(crop.bbox.as_list(), [100, 200, 701, 801])
        with Image.open(BytesIO(crop.data)) as opened:
            self.assertEqual(opened.size, (61, 63))


if __name__ == "__main__":
    unittest.main()
