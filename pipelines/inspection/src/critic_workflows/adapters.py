from __future__ import annotations

import asyncio
import inspect
import math
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from io import BytesIO
from typing import Any

from .types import (
    BBox,
    CriticContext,
    ImageAsset,
    ModelRequest,
    ModelResponse,
    Region,
    RetrievedExample,
)


ModelCallable = Callable[
    [ModelRequest],
    ModelResponse | dict[str, Any] | str | Awaitable[ModelResponse | dict[str, Any] | str],
]


class CallableModelClient:
    """Wrap any sync or async model function behind ``ModelClient``."""

    def __init__(self, fn: ModelCallable):
        self._fn = fn

    async def complete(self, request: ModelRequest) -> ModelResponse:
        value = self._fn(request)
        if inspect.isawaitable(value):
            value = await value
        if isinstance(value, ModelResponse):
            return value
        if isinstance(value, str):
            return ModelResponse(text=value)
        if isinstance(value, dict):
            return ModelResponse(parsed=value)
        raise TypeError(f"Unsupported model response type: {type(value)!r}")


class FullImageRegionProposer:
    async def propose(
        self,
        image: ImageAsset,
        context: CriticContext,
        labels: Sequence[str],
    ) -> list[Region]:
        del context, labels
        return [
            Region(
                id=f"{image.id}:full",
                bbox=BBox.full_image(),
                label="full_image",
                image=image,
            )
        ]


@dataclass(frozen=True)
class GridRegionProposer:
    rows: int = 2
    columns: int = 2
    include_full_image: bool = True

    async def propose(
        self,
        image: ImageAsset,
        context: CriticContext,
        labels: Sequence[str],
    ) -> list[Region]:
        del context, labels
        if self.rows < 1 or self.columns < 1:
            raise ValueError("Grid rows and columns must be positive.")
        regions: list[Region] = []
        if self.include_full_image:
            regions.append(
                Region(
                    id=f"{image.id}:full",
                    bbox=BBox.full_image(),
                    label="full_image",
                    image=image,
                )
            )
        for row in range(self.rows):
            for column in range(self.columns):
                x1 = round(column * 1000 / self.columns)
                y1 = round(row * 1000 / self.rows)
                x2 = round((column + 1) * 1000 / self.columns)
                y2 = round((row + 1) * 1000 / self.rows)
                regions.append(
                    Region(
                        id=f"{image.id}:grid:{row}:{column}",
                        bbox=BBox(x1, y1, x2, y2),
                        label="grid",
                    )
                )
        return regions


class NullExampleRetriever:
    async def retrieve(
        self,
        image: ImageAsset,
        artifact_type: str,
        *,
        k: int,
    ) -> list[RetrievedExample]:
        del image, artifact_type, k
        return []


class StaticExampleRetriever:
    def __init__(self, examples: Sequence[RetrievedExample]):
        self._examples = list(examples)

    async def retrieve(
        self,
        image: ImageAsset,
        artifact_type: str,
        *,
        k: int,
    ) -> list[RetrievedExample]:
        del image
        matches = [
            example
            for example in self._examples
            if example.artifact_type == artifact_type
        ]
        positives = [example for example in matches if example.label]
        negatives = [example for example in matches if not example.label]
        balanced: list[RetrievedExample] = []
        while len(balanced) < k and (positives or negatives):
            if positives:
                balanced.append(positives.pop(0))
            if len(balanced) < k and negatives:
                balanced.append(negatives.pop(0))
        return balanced


class PassthroughImageTool:
    """Testing adapter that records the bbox without changing pixels."""

    async def crop(
        self,
        image: ImageAsset,
        region: Region,
        *,
        padding: float = 0.0,
    ) -> ImageAsset:
        return ImageAsset(
            id=f"{region.id}:crop",
            path=image.path,
            data=image.data,
            mime_type=image.mime_type,
            parent_id=image.id,
            bbox=region.bbox,
            metadata={"padding": padding, **dict(region.metadata)},
        )


class PillowImageTool:
    """Real crop implementation. Install with ``pip install -e .[images]``."""

    def __init__(self, output_format: str = "PNG"):
        self.output_format = output_format.upper()

    async def crop(
        self,
        image: ImageAsset,
        region: Region,
        *,
        padding: float = 0.0,
    ) -> ImageAsset:
        return await asyncio.to_thread(
            self._crop_sync,
            image,
            region,
            padding,
        )

    def _crop_sync(
        self,
        image: ImageAsset,
        region: Region,
        padding: float,
    ) -> ImageAsset:
        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError(
                "PillowImageTool requires Pillow. Install `critic-workflows[images]`."
            ) from exc

        source = BytesIO(image.data) if image.data is not None else image.path
        if source is None:
            raise ValueError("ImageAsset has no readable image source.")

        with Image.open(source) as opened:
            width, height = opened.size
            box = region.bbox
            pad_x = (box.x2 - box.x1) * padding
            pad_y = (box.y2 - box.y1) * padding
            left = max(0.0, box.x1 - pad_x)
            top = max(0.0, box.y1 - pad_y)
            right = min(1000.0, box.x2 + pad_x)
            bottom = min(1000.0, box.y2 + pad_y)
            pixel_box = (
                math.floor(left * width / 1000),
                math.floor(top * height / 1000),
                math.ceil(right * width / 1000),
                math.ceil(bottom * height / 1000),
            )
            cropped = opened.crop(pixel_box)
            buffer = BytesIO()
            cropped.save(buffer, format=self.output_format)

        mime = "image/jpeg" if self.output_format in {"JPG", "JPEG"} else "image/png"
        return ImageAsset(
            id=f"{region.id}:crop",
            data=buffer.getvalue(),
            mime_type=mime,
            parent_id=image.id,
            bbox=region.bbox,
            metadata={"padding": padding, **dict(region.metadata)},
        )
