from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from critic_workflows import Finding, ImageAsset

from .v42_fast import V42AdaptiveContentCoverageAuditor


class V62AdaptiveContentCoverageAuditor(V42AdaptiveContentCoverageAuditor):
    """V6.0 coverage with atomic, failure-isolated region micro-batches."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.v62_batch_metrics = self._empty_batch_metrics()

    @staticmethod
    def _empty_batch_metrics() -> dict[str, Any]:
        return {
            "requests": 0,
            "requested_regions": 0,
            "request_sizes": {},
            "contract_failures": 0,
            "atomic_rejected_regions": 0,
        }

    async def audit(
        self,
        image: ImageAsset,
        context: Any,
    ) -> tuple[list[Finding], dict[str, Any]]:
        self.v62_batch_metrics = self._empty_batch_metrics()
        findings, summary = await super().audit(image, context)
        summary = dict(summary)
        summary["v62_adaptive_batching"] = dict(self.v62_batch_metrics)
        return findings, summary

    def _failed_batch_split_index(self, batch_size: int) -> int:
        # A failed three-item request is isolated as 2+1. This preserves the
        # useful micro-batch while keeping the singleton recovery independent.
        return max(1, (batch_size + 1) // 2)

    async def _inspect_batch(
        self,
        image: ImageAsset,
        batch: Sequence[tuple[int, Mapping[str, Any]]],
        *,
        inspection_pass: str = "primary",
    ) -> list[tuple[Finding | None, int, bool] | None]:
        size = len(batch)
        metrics = self.v62_batch_metrics
        metrics["requests"] += 1
        metrics["requested_regions"] += size
        request_sizes = metrics["request_sizes"]
        request_sizes[str(size)] = request_sizes.get(str(size), 0) + 1

        values = await super()._inspect_batch(
            image,
            batch,
            inspection_pass=inspection_pass,
        )
        if any(value is None for value in values):
            # Reject the entire response when its ID contract is broken. A
            # partly valid answer can be positionally misbound, so accepting
            # its surviving entries would silently trade recall for latency.
            metrics["contract_failures"] += 1
            metrics["atomic_rejected_regions"] += size
            return [None] * size
        return values
