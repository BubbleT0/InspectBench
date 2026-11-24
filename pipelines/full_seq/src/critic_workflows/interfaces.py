from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

from .types import (
    CriticContext,
    CriticResult,
    ImageAsset,
    ModelRequest,
    ModelResponse,
    Region,
    RetrievedExample,
)


@runtime_checkable
class ModelClient(Protocol):
    """The only interface a user's VLM adapter must implement."""

    async def complete(self, request: ModelRequest) -> ModelResponse:
        ...


@runtime_checkable
class Critic(Protocol):
    async def critique(
        self,
        image: ImageAsset,
        context: CriticContext | None = None,
    ) -> CriticResult:
        ...


@runtime_checkable
class RegionProposer(Protocol):
    async def propose(
        self,
        image: ImageAsset,
        context: CriticContext,
        labels: Sequence[str],
    ) -> list[Region]:
        ...


@runtime_checkable
class ExampleRetriever(Protocol):
    async def retrieve(
        self,
        image: ImageAsset,
        artifact_type: str,
        *,
        k: int,
    ) -> list[RetrievedExample]:
        ...


@runtime_checkable
class ImageTool(Protocol):
    async def crop(
        self,
        image: ImageAsset,
        region: Region,
        *,
        padding: float = 0.0,
    ) -> ImageAsset:
        ...

