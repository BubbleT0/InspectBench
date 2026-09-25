#!/usr/bin/env python3
"""Adapt the anonymized InspectBench annotation format for bbox_scorer.py.

Reads annotations_subset_*.jsonl (scrubbed format: category instead of
error_type.code, image_path relative) and emits scorer-compatible JSONL to
stdout or --output. Identity-safe: no external paths, no original ids.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

CATEGORY_TO_FAMILY = {
    "artifact/rendering": "D1_Artifact/A1_RenderingArtifact",
    "artifact/structural": "D1_Artifact/A2_StructuralArtifact",
    "artifact/topology": "D1_Artifact/A3_TopologyArtifact",
    "artifact/relational_physical": "D1_Artifact/A4_RelationalAndPhysicalArtifact",
    "text/visual_integrity": "D2_TextCorrectness/A1_TextVisualIntegrity",
    "text/content_fidelity": "D2_TextCorrectness/A2_TextContentFidelity",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", nargs="+", required=True)
    parser.add_argument("--image-root", default=".", help="root dir containing images/")
    parser.add_argument("--output")
    args = parser.parse_args()

    rows_out = []
    for path in args.annotations:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            rows_out.append({
                "sample_id": row["sample_id"],
                "image_absolute_path": str(Path(args.image_root) / row["image_path"]),
                "image_width": row["image_width"],
                "image_height": row["image_height"],
                "prompt": "",
                "annotation": {
                    "no_error": row["no_error"],
                    "boxes": [
                        {
                            "bbox_xywh": b["bbox_xywh"],
                            "error_type": {"code": CATEGORY_TO_FAMILY[b["category"]]},
                            "detail": b.get("note", ""),
                        }
                        for b in row["boxes"]
                    ],
                },
            })
    text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows_out)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text, encoding="utf-8")
        print(f"wrote {len(rows_out)} rows to {args.output}")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
