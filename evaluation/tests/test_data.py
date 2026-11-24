import json
from pathlib import Path

import pytest
from PIL import Image

from vlm_judge_bench.data import DataValidationError, load_annotations


def _row(image: Path, sample_id="s1", no_error=True, boxes=None):
    return {
        "sample_id": sample_id,
        "image_absolute_path": str(image),
        "image_width": 20,
        "image_height": 10,
        "prompt": "",
        "annotation": {"no_error": no_error, "boxes": boxes or []},
    }


def _write(path: Path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def test_load_and_validate_image(tmp_path):
    image = tmp_path / "a.png"
    Image.new("RGB", (20, 10)).save(image)
    data = tmp_path / "a.jsonl"
    _write(data, [_row(image)])
    samples = load_annotations([data], check_images=True)
    assert samples[0].sample_id == "s1"
    assert samples[0].no_error


def test_duplicate_ids_across_files_fail(tmp_path):
    image = tmp_path / "a.png"
    Image.new("RGB", (20, 10)).save(image)
    first, second = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    _write(first, [_row(image)])
    _write(second, [_row(image)])
    with pytest.raises(DataValidationError, match="duplicate sample_id"):
        load_annotations([first, second])


def test_invalid_no_error_and_bbox_fail(tmp_path):
    image = tmp_path / "a.png"
    Image.new("RGB", (20, 10)).save(image)
    box = {"bbox_xywh": [18, 2, 5, 5], "error_type": {"code": "D1_Artifact/A1_RenderingArtifact"}}
    data = tmp_path / "a.jsonl"
    _write(data, [_row(image, no_error=False, boxes=[box])])
    with pytest.raises(DataValidationError, match="outside"):
        load_annotations([data])
