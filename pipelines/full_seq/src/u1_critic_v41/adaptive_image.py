from __future__ import annotations

from io import BytesIO
import math
from typing import Any

from critic_workflows import BBox, ImageAsset, Region


def expand_bbox(bbox: BBox, padding: float) -> BBox:
    if padding < 0.0:
        raise ValueError("padding must be non-negative")
    pad_x = (bbox.x2 - bbox.x1) * padding
    pad_y = (bbox.y2 - bbox.y1) * padding
    return BBox(
        math.floor(max(0.0, bbox.x1 - pad_x)),
        math.floor(max(0.0, bbox.y1 - pad_y)),
        math.ceil(min(1000.0, bbox.x2 + pad_x)),
        math.ceil(min(1000.0, bbox.y2 + pad_y)),
    )


class LosslessZoomImageTool:
    def __init__(
        self,
        base: Any,
        *,
        min_short_side: int = 768,
        max_scale: int = 4,
    ):
        if min_short_side <= 0:
            raise ValueError("min_short_side must be positive")
        if max_scale <= 0:
            raise ValueError("max_scale must be positive")
        self.base = base
        self.min_short_side = min_short_side
        self.max_scale = max_scale

    async def crop(
        self,
        image: ImageAsset,
        region: Region,
        *,
        padding: float = 0.0,
    ) -> ImageAsset:
        cropped = await self.base.crop(
            image,
            region,
            padding=padding,
        )
        try:
            from PIL import Image

            source = (
                BytesIO(cropped.data)
                if cropped.data is not None
                else cropped.path
            )
            if source is None:
                raise ValueError("crop has no image source")
            with Image.open(source) as opened:
                opened.load()
                width, height = opened.size
                short_side = min(width, height)
                scale = min(
                    self.max_scale,
                    max(1, math.ceil(self.min_short_side / short_side)),
                )
                review_size = (width * scale, height * scale)
                review = (
                    opened.copy()
                    if scale == 1
                    else opened.resize(
                        review_size,
                        resample=Image.Resampling.NEAREST,
                    )
                )
                buffer = BytesIO()
                review.save(buffer, format="PNG")
        except (OSError, TypeError, ValueError):
            metadata = {
                **dict(cropped.metadata),
                "original_crop_size": None,
                "review_crop_size": None,
                "zoom_scale": 1,
            }
            return ImageAsset(
                id=cropped.id,
                path=cropped.path,
                data=cropped.data,
                mime_type=cropped.mime_type,
                parent_id=cropped.parent_id,
                bbox=cropped.bbox,
                metadata=metadata,
            )

        return ImageAsset(
            id=cropped.id,
            data=buffer.getvalue(),
            mime_type="image/png",
            parent_id=cropped.parent_id,
            bbox=cropped.bbox,
            metadata={
                **dict(cropped.metadata),
                "original_crop_size": [width, height],
                "review_crop_size": [*review_size],
                "zoom_scale": scale,
            },
        )
