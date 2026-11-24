from __future__ import annotations

from io import BytesIO
import unittest

from PIL import Image

from critic_workflows import BBox, ImageAsset, PillowImageTool, Region

from u1_critic_v41.adaptive_image import (
    LosslessZoomImageTool,
    expand_bbox,
)


def png_asset(width: int, height: int) -> ImageAsset:
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


class AdaptiveImageTests(unittest.IsolatedAsyncioTestCase):
    def test_expand_bbox_clamps_padding_at_image_edges(self) -> None:
        expanded = expand_bbox(BBox(0, 200, 500, 600), 0.1)

        self.assertEqual(
            expanded.as_list(),
            [0, 160, 550, 640],
        )

    async def test_small_crop_uses_integer_nearest_neighbor_zoom(self) -> None:
        tool = LosslessZoomImageTool(
            PillowImageTool(),
            min_short_side=768,
            max_scale=4,
        )
        region = Region("whole", BBox.full_image())

        crop = await tool.crop(png_asset(100, 200), region)

        self.assertEqual(crop.mime_type, "image/png")
        self.assertEqual(crop.metadata["original_crop_size"], [100, 200])
        self.assertEqual(crop.metadata["review_crop_size"], [400, 800])
        self.assertEqual(crop.metadata["zoom_scale"], 4)
        with Image.open(BytesIO(crop.data)) as opened:
            self.assertEqual(opened.size, (400, 800))

    async def test_zoom_uses_smallest_integer_that_reaches_target(self) -> None:
        tool = LosslessZoomImageTool(
            PillowImageTool(),
            min_short_side=768,
            max_scale=4,
        )

        crop = await tool.crop(
            png_asset(500, 1000),
            Region("whole", BBox.full_image()),
        )

        self.assertEqual(crop.metadata["original_crop_size"], [500, 1000])
        self.assertEqual(crop.metadata["review_crop_size"], [1000, 2000])
        self.assertEqual(crop.metadata["zoom_scale"], 2)

    async def test_large_crop_remains_pixel_identical_in_size(self) -> None:
        tool = LosslessZoomImageTool(
            PillowImageTool(),
            min_short_side=768,
            max_scale=4,
        )

        crop = await tool.crop(
            png_asset(800, 1000),
            Region("whole", BBox.full_image()),
        )

        self.assertEqual(crop.metadata["original_crop_size"], [800, 1000])
        self.assertEqual(crop.metadata["review_crop_size"], [800, 1000])
        self.assertEqual(crop.metadata["zoom_scale"], 1)
        with Image.open(BytesIO(crop.data)) as opened:
            self.assertEqual(opened.size, (800, 1000))


if __name__ == "__main__":
    unittest.main()
