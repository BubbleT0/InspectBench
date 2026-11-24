from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


class Verdict(str, Enum):
    ARTIFACT_PRESENT = "artifact_present"
    NO_ARTIFACT = "no_artifact"
    UNCERTAIN = "uncertain"


class VerificationStatus(str, Enum):
    PROPOSED = "proposed"
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    UNCERTAIN = "uncertain"


@dataclass(frozen=True)
class BBox:
    """Integer [0, 1000] bounding box in x1, y1, x2, y2 order."""

    x1: int
    y1: int
    x2: int
    y2: int

    def __post_init__(self) -> None:
        raw_values = (self.x1, self.y1, self.x2, self.y2)
        normalized: list[int] = []
        for value in raw_values:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(
                    "BBox values must be finite integer coordinates in "
                    f"[0, 1000], got {raw_values}."
                )
            numeric = float(value)
            if not math.isfinite(numeric) or not numeric.is_integer():
                raise ValueError(
                    "BBox values must be finite integer coordinates in "
                    f"[0, 1000], got {raw_values}."
                )
            normalized.append(int(numeric))
        values = tuple(normalized)
        if any(value < 0 or value > 1000 for value in values):
            raise ValueError(
                f"BBox values must be normalized to [0, 1000], got {values}."
            )
        object.__setattr__(self, "x1", values[0])
        object.__setattr__(self, "y1", values[1])
        object.__setattr__(self, "x2", values[2])
        object.__setattr__(self, "y2", values[3])
        if self.x1 >= self.x2 or self.y1 >= self.y2:
            raise ValueError(f"BBox must have positive area, got {values}.")

    @classmethod
    def full_image(cls) -> "BBox":
        return cls(0, 0, 1000, 1000)

    @classmethod
    def from_value(cls, value: Sequence[float] | "BBox") -> "BBox":
        if isinstance(value, cls):
            return value
        if len(value) != 4:
            raise ValueError(f"Expected four bbox values, got {value}.")
        return cls(*value)

    def as_list(self) -> list[int]:
        return [self.x1, self.y1, self.x2, self.y2]


@dataclass(frozen=True)
class ImageAsset:
    """Provider-neutral image input.

    A model adapter may consume ``path`` or ``data``. Derived crops keep their
    parent and normalized bbox so every finding remains auditable.
    """

    id: str
    path: str | Path | None = None
    data: bytes | None = None
    mime_type: str = "image/png"
    parent_id: str | None = None
    bbox: BBox | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.path is None and self.data is None:
            raise ValueError("ImageAsset requires either path or data.")


@dataclass(frozen=True)
class Region:
    id: str
    bbox: BBox
    label: str = "unknown"
    score: float = 1.0
    image: ImageAsset | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievedExample:
    image: ImageAsset
    label: bool
    artifact_type: str
    note: str = ""


@dataclass(frozen=True)
class CriticContext:
    generation_prompt: str | None = None
    reference_images: tuple[ImageAsset, ...] = ()
    taxonomy: tuple[str, ...] = ()
    user_preferences: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class Finding:
    region_id: str
    bbox: BBox
    artifact_type: str
    observation: str
    abnormality: str
    confidence: float
    evidence_ids: list[str] = field(default_factory=list)
    verification: VerificationStatus = VerificationStatus.PROPOSED
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.confidence = max(0.0, min(1.0, float(self.confidence)))

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        default_region_id: str = "full_image",
        default_bbox: BBox | None = None,
        source: str = "",
    ) -> "Finding":
        bbox_value = value.get("bbox_norm", value.get("bbox"))
        bbox = BBox.from_value(bbox_value) if bbox_value else (default_bbox or BBox.full_image())
        verification_raw = str(value.get("verification", VerificationStatus.PROPOSED.value))
        try:
            verification = VerificationStatus(verification_raw)
        except ValueError:
            verification = VerificationStatus.PROPOSED
        evidence = value.get("evidence_ids", [])
        if isinstance(evidence, str):
            evidence = [evidence]
        return cls(
            region_id=str(value.get("region_id", default_region_id)),
            bbox=bbox,
            artifact_type=str(value.get("artifact_type", "unknown")),
            observation=str(value.get("observation", "")),
            abnormality=str(value.get("abnormality", value.get("reason", ""))),
            confidence=float(value.get("confidence", 0.5)),
            evidence_ids=[str(item) for item in evidence],
            verification=verification,
            source=source or str(value.get("source", "")),
            metadata=dict(value.get("metadata", {})),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "region_id": self.region_id,
            "bbox_norm": self.bbox.as_list(),
            "artifact_type": self.artifact_type,
            "observation": self.observation,
            "abnormality": self.abnormality,
            "confidence": self.confidence,
            "evidence_ids": self.evidence_ids,
            "verification": self.verification.value,
            "source": self.source,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class TraceEvent:
    stage: str
    message: str
    data: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class CriticResult:
    verdict: Verdict
    confidence: float
    findings: list[Finding] = field(default_factory=list)
    uncertain_findings: list[Finding] = field(default_factory=list)
    trace: list[TraceEvent] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.confidence = max(0.0, min(1.0, float(self.confidence)))

    def as_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict.value,
            "confidence": self.confidence,
            "findings": [finding.as_dict() for finding in self.findings],
            "uncertain_findings": [
                finding.as_dict() for finding in self.uncertain_findings
            ],
            "trace": [
                {"stage": event.stage, "message": event.message, "data": dict(event.data)}
                for event in self.trace
            ],
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class ModelRequest:
    stage: str
    system_prompt: str
    user_prompt: str
    images: tuple[ImageAsset, ...] = ()
    response_schema: Mapping[str, Any] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class ModelResponse:
    text: str = ""
    parsed: Mapping[str, Any] | None = None
    raw: Any = None
    usage: Mapping[str, Any] = field(default_factory=dict)
