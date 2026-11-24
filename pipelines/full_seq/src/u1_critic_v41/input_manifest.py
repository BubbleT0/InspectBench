from __future__ import annotations

import hashlib
import json
import mimetypes
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from critic_workflows import ImageAsset


SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def image_mime_type(path: Path) -> str:
    """Keep historical filenames while identifying the actual encoded image."""
    with path.open("rb") as handle:
        header = handle.read(12)
    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if header.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if header.startswith(b"RIFF") and header[8:12] == b"WEBP":
        return "image/webp"
    return mimetypes.guess_type(path.name)[0] or "application/octet-stream"


@dataclass(frozen=True)
class ImageRecord:
    image_id: str
    source_path: Path
    reference_label: str
    mime_type: str


def discover_images(root: Path) -> list[ImageRecord]:
    resolved = root.resolve(strict=True)
    records: list[ImageRecord] = []
    for path in sorted(item for item in resolved.rglob("*") if item.is_file()):
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        relative = path.relative_to(resolved)
        digest = hashlib.sha256(relative.as_posix().encode("utf-8")).hexdigest()[:12]
        mime_type = image_mime_type(path)
        records.append(
            ImageRecord(
                image_id=f"img_{digest}",
                source_path=path,
                reference_label=relative.parent.as_posix(),
                mime_type=mime_type,
            )
        )
    return sorted(records, key=lambda item: item.image_id)


def select_images_from_jsonl(
    records: Sequence[ImageRecord],
    jsonl_path: Path,
    *,
    path_field: str = "image_absolute_path",
) -> list[ImageRecord]:
    """Select discovered images in JSONL order using an explicit path field."""

    source_records = {
        record.source_path.resolve(strict=True): record for record in records
    }
    selected: list[ImageRecord] = []
    seen: set[Path] = set()
    with jsonl_path.resolve(strict=True).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            raw_path = row.get(path_field) if isinstance(row, dict) else None
            if not isinstance(raw_path, str) or not raw_path.strip():
                raise ValueError(
                    f"JSONL line {line_number} has no non-empty {path_field!r}"
                )
            source_path = Path(raw_path).resolve(strict=True)
            if source_path in seen:
                raise ValueError(
                    f"JSONL line {line_number} repeats image path: {source_path}"
                )
            record = source_records.get(source_path)
            if record is None:
                raise ValueError(
                    f"JSONL line {line_number} image is outside --input-root "
                    f"or unsupported: {source_path}"
                )
            seen.add(source_path)
            selected.append(record)
    if not selected:
        raise ValueError("JSONL image selection is empty")
    return selected


def load_images_from_jsonl(
    jsonl_path: Path,
    *,
    path_field: str = "image_absolute_path",
) -> list[ImageRecord]:
    """Load an absolute-path allowlist directly, without recursively scanning AFS.

    The original delivery first discovers every image below ``--input-root`` and
    then filters that list.  A benchmark assembled from multiple storage roots
    would otherwise require scanning their common parent.  IDs are derived from
    resolved paths, while source paths and filenames remain local-only metadata.
    """

    selected: list[ImageRecord] = []
    seen_paths: set[Path] = set()
    seen_ids: set[str] = set()
    with jsonl_path.resolve(strict=True).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            raw_path = row.get(path_field) if isinstance(row, dict) else None
            if not isinstance(raw_path, str) or not raw_path.strip():
                raise ValueError(
                    f"JSONL line {line_number} has no non-empty {path_field!r}"
                )
            source_path = Path(raw_path).resolve(strict=True)
            if source_path in seen_paths:
                raise ValueError(
                    f"JSONL line {line_number} repeats image path: {source_path}"
                )
            if source_path.suffix.lower() not in SUPPORTED_SUFFIXES:
                raise ValueError(
                    f"JSONL line {line_number} uses an unsupported image type: "
                    f"{source_path}"
                )
            digest = hashlib.sha256(str(source_path).encode("utf-8")).hexdigest()[:12]
            image_id = f"img_{digest}"
            if image_id in seen_ids:
                raise ValueError(f"JSONL line {line_number} has a colliding image ID")
            seen_paths.add(source_path)
            seen_ids.add(image_id)
            selected.append(
                ImageRecord(
                    image_id=image_id,
                    source_path=source_path,
                    reference_label="",
                    mime_type=image_mime_type(source_path),
                )
            )
    if not selected:
        raise ValueError("JSONL image selection is empty")
    return selected


def to_image_asset(record: ImageRecord) -> ImageAsset:
    return ImageAsset(
        id=record.image_id,
        data=record.source_path.read_bytes(),
        mime_type=record.mime_type,
        metadata={},
    )


def manifest_rows(records: Sequence[ImageRecord]) -> list[dict[str, str]]:
    return [
        {
            **{
                key: str(value) if isinstance(value, Path) else value
                for key, value in asdict(record).items()
            }
        }
        for record in records
    ]
