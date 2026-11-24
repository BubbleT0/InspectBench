from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image


FAMILIES = (
    "D1_Artifact/A1_RenderingArtifact",
    "D1_Artifact/A2_StructuralArtifact",
    "D1_Artifact/A3_TopologyArtifact",
    "D1_Artifact/A4_RelationalAndPhysicalArtifact",
    "D2_TextCorrectness/A1_TextVisualIntegrity",
    "D2_TextCorrectness/A2_TextContentFidelity",
)


class DataValidationError(ValueError):
    pass


@dataclass(frozen=True)
class GTBox:
    bbox_xywh: tuple[float, float, float, float]
    family: str
    detail: str = ""


@dataclass(frozen=True)
class Sample:
    sample_id: str
    image_path: str
    image_width: int
    image_height: int
    prompt: str
    no_error: bool
    boxes: tuple[GTBox, ...]
    source_file: str
    source_line: int


@dataclass(frozen=True)
class PredictionBox:
    bbox_1000_xywh: tuple[float, float, float, float]
    family: str | None = None
    description: str = ""

    def absolute_xywh(self, width: int, height: int) -> tuple[float, float, float, float]:
        x, y, w, h = self.bbox_1000_xywh
        return x * width / 1000, y * height / 1000, w * width / 1000, h * height / 1000


@dataclass
class PredictionRecord:
    sample_id: str
    model: str
    status: str
    boxes: list[PredictionBox]
    no_error: bool
    latency_seconds: float = 0.0
    attempts: int = 0
    parse_warnings: list[str] | None = None
    error: str | None = None
    usage: dict | None = None
    sample_attempts: int = 1

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["boxes"] = [asdict(box) for box in self.boxes]
        return payload

    @classmethod
    def from_dict(cls, payload: dict) -> "PredictionRecord":
        return cls(
            sample_id=payload["sample_id"],
            model=payload["model"],
            status=payload["status"],
            boxes=[PredictionBox(tuple(b["bbox_1000_xywh"]), b.get("family"), b.get("description", "")) for b in payload.get("boxes", [])],
            no_error=bool(payload.get("no_error", not payload.get("boxes"))),
            latency_seconds=float(payload.get("latency_seconds", 0)),
            attempts=int(payload.get("attempts", 0)),
            parse_warnings=list(payload.get("parse_warnings") or []),
            error=payload.get("error"),
            usage=payload.get("usage"),
            sample_attempts=int(payload.get("sample_attempts", 1)),
        )


def _validate_bbox(value: object, width: int, height: int, where: str) -> tuple[float, float, float, float]:
    if not isinstance(value, list) or len(value) != 4:
        raise DataValidationError(f"{where}: bbox_xywh must contain four numbers")
    try:
        x, y, w, h = (float(v) for v in value)
    except (TypeError, ValueError) as exc:
        raise DataValidationError(f"{where}: bbox_xywh contains a non-number") from exc
    if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > width + 0.01 or y + h > height + 0.01:
        raise DataValidationError(f"{where}: bbox_xywh is outside the {width}x{height} image")
    return x, y, w, h


def load_annotations(paths: Iterable[str | Path], check_images: bool = False) -> list[Sample]:
    samples: list[Sample] = []
    seen: dict[str, str] = {}
    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise DataValidationError(f"annotation file not found: {path}")
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                where = f"{path}:{line_no}"
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise DataValidationError(f"{where}: invalid JSON: {exc}") from exc
                sample_id = str(row.get("sample_id", "")).strip()
                if not sample_id:
                    raise DataValidationError(f"{where}: missing sample_id")
                if sample_id in seen:
                    raise DataValidationError(f"{where}: duplicate sample_id {sample_id!r}; first seen at {seen[sample_id]}")
                seen[sample_id] = where
                image_path = Path(str(row.get("image_absolute_path", ""))).expanduser()
                if not image_path.is_file():
                    raise DataValidationError(f"{where}: image not found: {image_path}")
                try:
                    width, height = int(row["image_width"]), int(row["image_height"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise DataValidationError(f"{where}: invalid image dimensions") from exc
                if width <= 0 or height <= 0:
                    raise DataValidationError(f"{where}: image dimensions must be positive")
                if check_images:
                    with Image.open(image_path) as image:
                        if image.size != (width, height):
                            raise DataValidationError(f"{where}: metadata dimensions {(width, height)} != actual {image.size}")
                annotation = row.get("annotation") or {}
                no_error = annotation.get("no_error")
                if not isinstance(no_error, bool):
                    raise DataValidationError(f"{where}: annotation.no_error must be boolean")
                raw_boxes = annotation.get("boxes")
                if not isinstance(raw_boxes, list):
                    raise DataValidationError(f"{where}: annotation.boxes must be a list")
                if no_error != (len(raw_boxes) == 0):
                    raise DataValidationError(f"{where}: no_error is inconsistent with boxes")
                boxes: list[GTBox] = []
                for idx, box in enumerate(raw_boxes):
                    box_where = f"{where}:box[{idx}]"
                    bbox = _validate_bbox(box.get("bbox_xywh"), width, height, box_where)
                    error_type = box.get("error_type") or {}
                    family = str(error_type.get("code", ""))
                    if family not in FAMILIES:
                        raise DataValidationError(f"{box_where}: unsupported primary family {family!r}")
                    boxes.append(GTBox(bbox, family, str(box.get("detail", ""))))
                samples.append(Sample(sample_id, str(image_path.resolve()), width, height, str(row.get("prompt") or ""), no_error, tuple(boxes), str(path), line_no))
    if not samples:
        raise DataValidationError("no annotation records were loaded")
    return samples


def load_predictions(path: str | Path) -> dict[str, PredictionRecord]:
    records: dict[str, PredictionRecord] = {}
    file_path = Path(path)
    if not file_path.exists():
        return records
    with file_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            payload = json.loads(line)
            record = PredictionRecord.from_dict(payload)
            if record.sample_id in records:
                raise DataValidationError(f"{file_path}:{line_no}: duplicate prediction for {record.sample_id}")
            records[record.sample_id] = record
    return records
