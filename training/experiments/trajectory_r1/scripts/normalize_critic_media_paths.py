#!/usr/bin/env python3
"""Copy OpenAI JSONL shards while making local media paths root-relative.

The validator rejects absolute media paths.  This script deliberately writes a
new immutable stage, preserving the structured-QC output byte-for-byte except
for the image/video/audio path values and the annotation path in the new meta.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


MEDIA_KEYS = ("image", "images", "video", "videos", "audio", "audios")


def normalize_one(value: str, root: Path) -> tuple[str, bool]:
    if not isinstance(value, str) or not value.startswith("/"):
        return value, False
    path = Path(value)
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"absolute media path is outside root {root}: {value}") from exc
    return relative.as_posix(), True


def normalize_media(obj: dict, root: Path) -> int:
    changed = 0
    for key in MEDIA_KEYS:
        if key not in obj:
            continue
        value = obj[key]
        if isinstance(value, str):
            obj[key], did_change = normalize_one(value, root)
            changed += int(did_change)
        elif isinstance(value, list):
            new_values = []
            for item in value:
                new_item, did_change = normalize_one(item, root)
                new_values.append(new_item)
                changed += int(did_change)
            obj[key] = new_values
    return changed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-meta", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--root", default="/path/to/storage")
    args = parser.parse_args()

    input_meta_path = Path(args.input_meta)
    output_dir = Path(args.output_dir)
    shard_dir = output_dir / "critic"
    root = Path(args.root).resolve()
    shard_dir.mkdir(parents=True, exist_ok=True)

    with input_meta_path.open(encoding="utf-8") as f:
        input_meta = json.load(f)
    if not isinstance(input_meta, dict):
        raise TypeError("input meta must be a dataset-name keyed object")

    output_meta: dict[str, dict] = {}
    total_rows = 0
    total_media = 0
    existing_media = 0
    shard_summaries = []

    for name, config in sorted(input_meta.items()):
        source = Path(config["annotation"])
        destination = shard_dir / source.name
        rows = 0
        media = 0
        exists = 0
        digest = hashlib.sha256()
        with source.open(encoding="utf-8") as src, destination.open("w", encoding="utf-8") as dst:
            for line_number, line in enumerate(src, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                media += normalize_media(row, root)
                for key in MEDIA_KEYS:
                    values = row.get(key, [])
                    if isinstance(values, str):
                        values = [values]
                    for value in values:
                        if isinstance(value, str) and not value.startswith("/"):
                            exists += int((root / value).is_file())
                serialized = json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                dst.write(serialized)
                digest.update(serialized.encode("utf-8"))
                rows += 1

        expected = int(config.get("length", rows))
        if rows != expected:
            raise ValueError(f"{name}: expected {expected} rows, got {rows}")
        if media != exists:
            raise FileNotFoundError(f"{name}: only {exists}/{media} normalized media files exist")

        new_config = dict(config)
        new_config["annotation"] = str(destination)
        new_config["root"] = str(root)
        output_meta[name] = new_config
        total_rows += rows
        total_media += media
        existing_media += exists
        shard_summaries.append({
            "name": name,
            "rows": rows,
            "normalized_media": media,
            "existing_media": exists,
            "output": str(destination),
            "sha256": digest.hexdigest(),
        })

    meta_output = output_dir / "meta_middle_format.json"
    with meta_output.open("w", encoding="utf-8") as f:
        json.dump(output_meta, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")

    summary = {
        "input_meta": str(input_meta_path),
        "output_meta": str(meta_output),
        "root": str(root),
        "datasets": len(output_meta),
        "rows": total_rows,
        "normalized_media": total_media,
        "existing_media": existing_media,
        "shards": shard_summaries,
    }
    with (output_dir / "normalization_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
