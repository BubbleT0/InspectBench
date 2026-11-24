#!/usr/bin/env python3
"""Read-only integrity audit for the fixed R1/R2/R3 development datasets."""

from __future__ import annotations

import json
import os
import statistics
from collections import Counter
from pathlib import Path

from PIL import Image


ROOT = Path(
    "/path/to/storage/multimodal_ckpt/research/autotrain/experiments/"
    "critic_singlemodel_r123_8gpu_20260824/data/dev_4000_800"
)
FORBIDDEN = (
    "generation_prompt",
    "reference generation instruction",
    "source_name",
    "source_model",
    "base_prediction",
    "筛选管线",
)
ALLOWED_FAMILIES = {
    "D1_Artifact/A1_RenderingArtifact",
    "D1_Artifact/A2_StructuralArtifact",
    "D1_Artifact/A3_TopologyArtifact",
    "D1_Artifact/A4_RelationalAndPhysicalArtifact",
    "D2_TextCorrectness/A1_TextVisualIntegrity",
}


def audit_split(name: str, expected_rows: int) -> dict:
    path = ROOT / f"{name}.jsonl"
    counts = Counter()
    images = set()
    pixels = []
    target_chars = []
    prompt = None
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            row = json.loads(line)
            assert set(row) == {"messages", "images"}, (name, index, row.keys())
            assert len(row["messages"]) == 2 and len(row["images"]) == 1
            user, assistant = row["messages"]
            assert user["role"] == "user" and assistant["role"] == "assistant"
            prompt = prompt or user["content"]
            assert user["content"] == prompt, (name, index, "prompt drift")
            lower = user["content"].lower()
            assert all(marker not in lower for marker in FORBIDDEN), (name, index)
            target = json.loads(assistant["content"])
            assert set(target) == {"no_error", "boxes"}
            label = "clean" if target["no_error"] else "error"
            counts[label] += 1
            assert label == ("clean" if index % 2 == 0 else "error"), (name, index, label)
            assert bool(target["no_error"]) == (target["boxes"] == [])
            for box in target["boxes"]:
                assert set(box) == {"bbox_1000_xyxy", "family", "description"}
                x1, y1, x2, y2 = box["bbox_1000_xyxy"]
                assert 0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000
                assert box["family"] in ALLOWED_FAMILIES
                assert isinstance(box["description"], str) and box["description"].strip()
            image_path = row["images"][0]
            assert os.path.isfile(image_path), (name, index, image_path)
            assert image_path not in images, (name, index, "duplicate image")
            images.add(image_path)
            with Image.open(image_path) as image:
                width, height = image.size
                assert width > 0 and height > 0
                pixels.append(width * height)
            target_chars.append(len(assistant["content"]))
    assert index + 1 == expected_rows, (name, index + 1, expected_rows)
    assert counts["clean"] == counts["error"] == expected_rows // 2, counts
    return {
        "rows": expected_rows,
        "counts": dict(counts),
        "images": images,
        "pixel_max": max(pixels),
        "pixel_p50": int(statistics.median(pixels)),
        "target_chars_max": max(target_chars),
        "target_chars_mean": round(statistics.mean(target_chars), 2),
        "prompt_sha256": __import__("hashlib").sha256(prompt.encode()).hexdigest(),
    }


def main() -> None:
    train = audit_split("train", 4000)
    validation = audit_split("validation", 800)
    overlap = train["images"] & validation["images"]
    assert not overlap, f"train/validation image overlap: {len(overlap)}"
    train.pop("images")
    validation.pop("images")
    assert train["prompt_sha256"] == validation["prompt_sha256"]
    print(json.dumps({"train": train, "validation": validation, "image_overlap": 0}, indent=2))
    print("dataset preflight: PASS")


if __name__ == "__main__":
    main()
