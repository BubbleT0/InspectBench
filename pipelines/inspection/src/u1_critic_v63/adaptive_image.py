from __future__ import annotations

from io import BytesIO
import asyncio
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


async def overview_image(
    image: ImageAsset,
    *,
    max_side: int,
) -> ImageAsset:
    """Return a bounded overview while leaving all detail crops lossless."""

    if max_side <= 0:
        return image

    def resize() -> ImageAsset:
        try:
            from PIL import Image

            source = BytesIO(image.data) if image.data is not None else image.path
            if source is None:
                return image
            with Image.open(source) as opened:
                opened.load()
                width, height = opened.size
                longest = max(width, height)
                if longest <= max_side:
                    return image
                scale = max_side / longest
                review_size = (
                    max(1, round(width * scale)),
                    max(1, round(height * scale)),
                )
                review = opened.resize(
                    review_size,
                    resample=Image.Resampling.LANCZOS,
                )
                buffer = BytesIO()
                review.save(buffer, format="PNG")
        except (OSError, TypeError, ValueError):
            return image
        return ImageAsset(
            id=f"{image.id}:overview-{max_side}",
            data=buffer.getvalue(),
            mime_type="image/png",
            parent_id=image.id,
            bbox=image.bbox,
            metadata={
                **dict(image.metadata),
                "overview_original_size": [width, height],
                "overview_review_size": [*review_size],
                "overview_max_side": max_side,
            },
        )

    return await asyncio.to_thread(resize)


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
        self._crop_tasks: dict[tuple[Any, ...], asyncio.Task[ImageAsset]] = {}
        self._crop_tasks_lock = asyncio.Lock()

    async def crop(
        self,
        image: ImageAsset,
        region: Region,
        *,
        padding: float = 0.0,
    ) -> ImageAsset:
        cache_key = (
            id(image),
            image.id,
            region.id,
            *region.bbox.as_list(),
            padding,
        )
        async with self._crop_tasks_lock:
            task = self._crop_tasks.get(cache_key)
            if task is None:
                task = asyncio.create_task(
                    self._crop_uncached(image, region, padding=padding)
                )
                self._crop_tasks[cache_key] = task
        try:
            return await asyncio.shield(task)
        except BaseException:
            async with self._crop_tasks_lock:
                if self._crop_tasks.get(cache_key) is task:
                    self._crop_tasks.pop(cache_key, None)
            raise

    async def _crop_uncached(
        self,
        image: ImageAsset,
        region: Region,
        *,
        padding: float,
    ) -> ImageAsset:
        cropped = await self.base.crop(
            image,
            region,
            padding=padding,
        )
        return await asyncio.to_thread(self._zoom_sync, cropped)

    def _zoom_sync(self, cropped: ImageAsset) -> ImageAsset:
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
