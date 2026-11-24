"""DSPy optimization layer for the specialist-cascade critic.

DSPy is deliberately limited to ``specialist.detect``. Region proposal,
verification, and aggregation continue to use the provider-neutral workflow so
an experiment compares prompt/program optimization rather than two entirely
different critic systems.
"""

from __future__ import annotations

import base64
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import dspy

from .interfaces import ModelClient
from .types import ImageAsset, ModelRequest, ModelResponse


class ArtifactSpecialistSignature(dspy.Signature):
    """Judge one requested artifact type from visible evidence only.

    Compare the full image with the supplied region crop. Distinguish genuine
    generation artifacts from occlusion, perspective, blur, stylization, and
    ordinary ambiguity. The observation must say what is directly visible; the
    abnormality must explain why that observation violates the requested
    artifact criterion. If evidence is insufficient, set present to false and
    lower confidence instead of inventing hidden structure.
    """

    full_image: dspy.Image = dspy.InputField(
        desc="The complete image, used for global scene context."
    )
    crop_image: dspy.Image = dspy.InputField(
        desc="The candidate region crop to inspect closely."
    )
    artifact_type: str = dspy.InputField(
        desc="The single artifact category being tested."
    )
    specialist_instruction: str = dspy.InputField(
        desc="Operational definition and decision rule for this specialist."
    )
    generation_prompt: str = dspy.InputField(
        desc="Original image-generation prompt, or 'not provided'."
    )
    region_context: str = dspy.InputField(
        desc="Region identifier, integer [0,1000] box, and optional contextual notes."
    )

    present: bool = dspy.OutputField(
        desc="True only when the requested artifact is visibly supported."
    )
    confidence: float = dspy.OutputField(
        desc="Calibrated probability from 0 to 1 that present is correct."
    )
    observation: str = dspy.OutputField(
        desc="Concise description of directly visible evidence."
    )
    abnormality: str = dspy.OutputField(
        desc="Why the observation is anomalous, empty when no artifact is present."
    )


class DSPyArtifactSpecialist(dspy.Module):
    """The DSPy student program compiled independently per artifact type."""

    def __init__(self) -> None:
        super().__init__()
        self.detect = dspy.Predict(ArtifactSpecialistSignature)

    def forward(
        self,
        *,
        full_image: dspy.Image,
        crop_image: dspy.Image,
        artifact_type: str,
        specialist_instruction: str,
        generation_prompt: str,
        region_context: str,
    ) -> dspy.Prediction:
        return self.detect(
            full_image=full_image,
            crop_image=crop_image,
            artifact_type=artifact_type,
            specialist_instruction=specialist_instruction,
            generation_prompt=generation_prompt,
            region_context=region_context,
        )

    async def aforward(
        self,
        *,
        full_image: dspy.Image,
        crop_image: dspy.Image,
        artifact_type: str,
        specialist_instruction: str,
        generation_prompt: str,
        region_context: str,
    ) -> dspy.Prediction:
        return await self.detect.acall(
            full_image=full_image,
            crop_image=crop_image,
            artifact_type=artifact_type,
            specialist_instruction=specialist_instruction,
            generation_prompt=generation_prompt,
            region_context=region_context,
        )


@dataclass(frozen=True)
class DSPySpecialistRecord:
    """One labeled region used for DSPy compilation or held-out evaluation."""

    full_image: ImageAsset
    crop_image: ImageAsset
    artifact_type: str
    specialist_instruction: str
    present: bool
    generation_prompt: str = "not provided"
    region_context: str = ""
    confidence: float = 1.0
    observation: str = ""
    abnormality: str = ""

    def to_example(self) -> dspy.Example:
        return dspy.Example(
            full_image=image_asset_to_dspy(self.full_image),
            crop_image=image_asset_to_dspy(self.crop_image),
            artifact_type=self.artifact_type,
            specialist_instruction=self.specialist_instruction,
            generation_prompt=self.generation_prompt or "not provided",
            region_context=self.region_context,
            present=self.present,
            confidence=_clamp(self.confidence),
            observation=self.observation,
            abnormality=self.abnormality,
        ).with_inputs(
            "full_image",
            "crop_image",
            "artifact_type",
            "specialist_instruction",
            "generation_prompt",
            "region_context",
        )


@dataclass(frozen=True)
class BinaryMetrics:
    samples: int
    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int

    @property
    def accuracy(self) -> float:
        return (
            (self.true_positive + self.true_negative) / self.samples
            if self.samples
            else 0.0
        )

    @property
    def precision(self) -> float:
        denominator = self.true_positive + self.false_positive
        return self.true_positive / denominator if denominator else 0.0

    @property
    def recall(self) -> float:
        denominator = self.true_positive + self.false_negative
        return self.true_positive / denominator if denominator else 0.0

    @property
    def f1(self) -> float:
        denominator = self.precision + self.recall
        return 2 * self.precision * self.recall / denominator if denominator else 0.0

    def as_dict(self) -> dict[str, float | int]:
        return {
            "samples": self.samples,
            "true_positive": self.true_positive,
            "false_positive": self.false_positive,
            "true_negative": self.true_negative,
            "false_negative": self.false_negative,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
        }


@dataclass(frozen=True)
class DSPyComparison:
    baseline: BinaryMetrics
    compiled: BinaryMetrics

    def as_dict(self) -> dict[str, Any]:
        return {
            "baseline": self.baseline.as_dict(),
            "compiled": self.compiled.as_dict(),
            "delta": {
                "accuracy": self.compiled.accuracy - self.baseline.accuracy,
                "precision": self.compiled.precision - self.baseline.precision,
                "recall": self.compiled.recall - self.baseline.recall,
                "f1": self.compiled.f1 - self.baseline.f1,
            },
        }


def image_asset_to_dspy(image: ImageAsset) -> dspy.Image:
    """Convert the provider-neutral image without changing its pixels."""

    if image.data is not None:
        encoded = base64.b64encode(image.data).decode("ascii")
        return dspy.Image(f"data:{image.mime_type};base64,{encoded}")
    if image.path is None:
        raise ValueError(f"ImageAsset {image.id!r} has neither data nor path.")
    return dspy.Image(str(Path(image.path).expanduser().resolve()))


def artifact_detection_metric(
    example: dspy.Example,
    prediction: dspy.Prediction,
    trace: Any = None,
) -> float:
    """Decomposable compile-time metric; held-out reporting still uses F1."""

    del trace
    return float(_coerce_bool(prediction.present) is _coerce_bool(example.present))


class DSPySpecialistCompiler:
    """Compile one few-shot DSPy program for each artifact category."""

    def __init__(
        self,
        lm: dspy.BaseLM,
        *,
        max_bootstrapped_demos: int = 4,
        max_labeled_demos: int = 8,
        max_rounds: int = 1,
        metric: Any = artifact_detection_metric,
    ) -> None:
        self.lm = lm
        self.max_bootstrapped_demos = max_bootstrapped_demos
        self.max_labeled_demos = max_labeled_demos
        self.max_rounds = max_rounds
        self.metric = metric

    def compile(
        self,
        records: Iterable[DSPySpecialistRecord],
    ) -> dict[str, DSPyArtifactSpecialist]:
        grouped: dict[str, list[dspy.Example]] = defaultdict(list)
        for record in records:
            grouped[record.artifact_type].append(record.to_example())
        if not grouped:
            raise ValueError("At least one DSPySpecialistRecord is required.")

        compiled: dict[str, DSPyArtifactSpecialist] = {}
        for artifact_type, trainset in grouped.items():
            student = DSPyArtifactSpecialist()
            optimizer = dspy.BootstrapFewShot(
                metric=self.metric,
                max_bootstrapped_demos=self.max_bootstrapped_demos,
                max_labeled_demos=self.max_labeled_demos,
                max_rounds=self.max_rounds,
            )
            with dspy.context(lm=self.lm):
                program = optimizer.compile(student, trainset=trainset)
            program.set_lm(self.lm)
            compiled[artifact_type] = program
        return compiled


class DSPySpecialistModelClient(ModelClient):
    """Route specialist detection to DSPy and all other stages to a fallback."""

    def __init__(
        self,
        programs: Mapping[str, DSPyArtifactSpecialist],
        fallback: ModelClient,
        *,
        lm: dspy.BaseLM | None = None,
    ) -> None:
        if not programs:
            raise ValueError("At least one compiled DSPy program is required.")
        self.programs = dict(programs)
        self.fallback = fallback
        if lm is not None:
            for program in self.programs.values():
                program.set_lm(lm)

    async def complete(self, request: ModelRequest) -> ModelResponse:
        if request.stage != "specialist.detect":
            return await self.fallback.complete(request)

        artifact_type = str(request.metadata.get("artifact_type", ""))
        program = self.programs.get(artifact_type)
        if program is None:
            return await self.fallback.complete(request)
        if len(request.images) < 2:
            raise ValueError("DSPy specialist detection requires full image and crop.")

        region_context = (
            f"region_id={request.metadata.get('region_id', 'unknown')}; "
            f"bbox_norm={request.metadata.get('region_bbox', 'unknown')}"
        )
        prediction = await program.acall(
            full_image=image_asset_to_dspy(request.images[0]),
            crop_image=image_asset_to_dspy(request.images[1]),
            artifact_type=artifact_type,
            specialist_instruction=str(
                request.metadata.get("specialist_instruction", "")
            ),
            generation_prompt=str(
                request.metadata.get("generation_prompt", "") or "not provided"
            ),
            region_context=region_context,
        )
        parsed = {
            "present": _coerce_bool(prediction.present),
            "confidence": _clamp(prediction.confidence),
            "observation": str(prediction.observation),
            "abnormality": str(prediction.abnormality),
        }
        return ModelResponse(parsed=parsed, raw=prediction)


def evaluate_specialist_program(
    program: DSPyArtifactSpecialist,
    records: Sequence[DSPySpecialistRecord],
) -> BinaryMetrics:
    """Evaluate a DSPy program on a held-out set without optimizer leakage."""

    outcomes: list[tuple[bool, bool]] = []
    for record in records:
        prediction = program(**_record_inputs(record))
        outcomes.append((record.present, _coerce_bool(prediction.present)))
    return _binary_metrics(outcomes)


def compare_specialist_programs(
    baseline: DSPyArtifactSpecialist,
    compiled: DSPyArtifactSpecialist,
    records: Sequence[DSPySpecialistRecord],
) -> DSPyComparison:
    """Report the exact before/after comparison requested for a DSPy trial."""

    return DSPyComparison(
        baseline=evaluate_specialist_program(baseline, records),
        compiled=evaluate_specialist_program(compiled, records),
    )


def _record_inputs(record: DSPySpecialistRecord) -> dict[str, Any]:
    return {
        "full_image": image_asset_to_dspy(record.full_image),
        "crop_image": image_asset_to_dspy(record.crop_image),
        "artifact_type": record.artifact_type,
        "specialist_instruction": record.specialist_instruction,
        "generation_prompt": record.generation_prompt or "not provided",
        "region_context": record.region_context,
    }


def _binary_metrics(outcomes: Sequence[tuple[bool, bool]]) -> BinaryMetrics:
    tp = sum(expected and predicted for expected, predicted in outcomes)
    fp = sum(not expected and predicted for expected, predicted in outcomes)
    tn = sum(not expected and not predicted for expected, predicted in outcomes)
    fn = sum(expected and not predicted for expected, predicted in outcomes)
    return BinaryMetrics(
        samples=len(outcomes),
        true_positive=tp,
        false_positive=fp,
        true_negative=tn,
        false_negative=fn,
    )


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1", "present"}:
            return True
        if normalized in {"false", "no", "0", "absent"}:
            return False
    return bool(value)


def _clamp(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = 0.0
    return max(0.0, min(1.0, numeric))
