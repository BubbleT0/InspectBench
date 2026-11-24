#!/usr/bin/env python3
"""Export the frozen benchmark with public IDs and lossless metadata removal.

This is a maintainer utility. The private ID map is read but never copied into
the output. Image pixels and annotation coordinates must remain unchanged.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import struct

from PIL import Image

CATEGORIES = {
    "D1_Artifact/A1_RenderingArtifact": "artifact/rendering",
    "D1_Artifact/A2_StructuralArtifact": "artifact/structural",
    "D1_Artifact/A3_TopologyArtifact": "artifact/topology",
    "D1_Artifact/A4_RelationalAndPhysicalArtifact": "artifact/relational_physical",
    "D2_TextCorrectness/A1_TextVisualIntegrity": "text/visual_integrity",
}
SOURCES = {"A": "GPT-image-2", "B": "Public-4", "C": "SenseNova-U1.5",
           "D": "SenseNova-U1.5", "E": "Natural"}


def jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def strip_metadata(data, image_format):
    if image_format == "PNG":
        assert data[:8] == b"\x89PNG\r\n\x1a\n"
        out = bytearray(data[:8]); pos = 8
        while pos < len(data):
            size = struct.unpack(">I", data[pos:pos + 4])[0]
            kind = data[pos + 4:pos + 8]
            end = pos + size + 12
            if kind not in {b"tEXt", b"zTXt", b"iTXt", b"eXIf", b"tIME"}:
                out.extend(data[pos:end])
            pos = end
            if kind == b"IEND":
                break
        return bytes(out)
    if image_format in {"JPEG", "MPO"}:
        assert data[:2] == b"\xff\xd8"
        out = bytearray(data[:2]); pos = 2
        while pos < len(data):
            start = pos
            assert data[pos] == 0xff
            while data[pos] == 0xff:
                pos += 1
            marker = data[pos]; pos += 1
            if marker == 0xd9:
                out.extend(data[start:pos]); break
            if marker == 0x01 or 0xd0 <= marker <= 0xd7:
                out.extend(data[start:pos]); continue
            size = struct.unpack(">H", data[pos:pos + 2])[0]
            end = pos + size
            # Preserve JFIF, ICC and Adobe color-transform markers.
            is_mpf = marker == 0xe2 and data[pos + 2:pos + 6] == b"MPF\x00"
            if not is_mpf and marker not in {0xe1, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8,
                                            0xe9, 0xea, 0xeb, 0xec, 0xed, 0xef, 0xfe}:
                out.extend(data[start:end])
            pos = end
            if marker == 0xda:
                entropy_start = pos
                while True:
                    scan = data.index(b"\xff", pos)
                    next_byte = scan + 1
                    while data[next_byte] == 0xff:
                        next_byte += 1
                    if data[next_byte] == 0x00 or 0xd0 <= data[next_byte] <= 0xd7:
                        pos = next_byte + 1
                        continue
                    out.extend(data[entropy_start:scan])
                    pos = scan
                    break
        return bytes(out)
    raise ValueError("Unsupported image format: " + str(image_format))


def export_image(task):
    public_id, original, mapping, output = task
    src = Path(original["image_absolute_path"])
    suffix = src.suffix.lower()
    before = src.read_bytes()
    with Image.open(io.BytesIO(before)) as im:
        if im.getexif().get(274, 1) not in (0, 1):
            raise ValueError(public_id + ": nontrivial EXIF orientation requires explicit handling")
        size = im.size
        image_format = im.format
        pixel_hash = hashlib.sha256(im.convert("RGBA").tobytes()).hexdigest()
    assert size == (original["image_width"], original["image_height"]), public_id
    after = strip_metadata(before, image_format)
    with Image.open(io.BytesIO(after)) as im:
        assert im.size == size
        assert hashlib.sha256(im.convert("RGBA").tobytes()).hexdigest() == pixel_hash
        assert not dict(im.getexif())
        assert not any(k in im.info for k in ("comment", "comments", "exif", "xmp"))
    rel = f"images/{public_id}{suffix}"
    (output / rel).write_bytes(after)
    boxes = []
    for b in original["annotation"]["boxes"]:
        code = b["error_type"]["code"]
        assert code in CATEGORIES, code
        boxes.append({"bbox_xywh": b["bbox_xywh"], "category": CATEGORIES[code],
                      "error_type": {"code": code}, "detail": str(b.get("detail") or "")})
    no_error = not boxes
    assert bool(original["annotation"]["no_error"]) == no_error
    row = {"sample_id": public_id, "image_path": rel,
           "image_width": size[0], "image_height": size[1],
           "subset": mapping["subset"], "source_group": SOURCES[mapping["subset"]],
           "annotation": {"no_error": no_error, "boxes": boxes}}
    fingerprint = {"sample_id": public_id, "image_path": rel,
                   "sha256": hashlib.sha256(after).hexdigest(),
                   "source_sha256": hashlib.sha256(before).hexdigest(),
                   "decoded_rgba_sha256": pixel_hash, "format": image_format, "bytes": len(after)}
    return row, fingerprint


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--annotations", nargs="+", type=Path, required=True)
    ap.add_argument("--private-id-map", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    output = args.output
    (output / "images").mkdir(parents=True, exist_ok=True)
    originals = {}
    for path in args.annotations:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                assert row["sample_id"] not in originals
                originals[row["sample_id"]] = row
    mapping = json.loads(args.private_id_map.read_text())
    assert len(originals) == len(mapping) == 1039
    tasks = [(pid, originals[m["orig_sample_id"]], m, output) for pid, m in sorted(mapping.items())]
    rows, fingerprints = [], []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, (row, fingerprint) in enumerate(pool.map(export_image, tasks), 1):
            rows.append(row); fingerprints.append(fingerprint)
            if i % 100 == 0:
                print(f"Verified identical decoded pixels: {i}/1039", flush=True)
    assert sum(len(r["annotation"]["boxes"]) for r in rows) == 1983
    assert sum(r["annotation"]["no_error"] for r in rows) == 321
    jsonl(output / "annotations/annotations.jsonl", rows)
    # ImageFolder metadata is consumable without custom dataset-loading code.
    metadata = []
    for r in rows:
        boxes = r["annotation"]["boxes"]
        metadata.append({"file_name": Path(r["image_path"]).name,
                         "sample_id": r["sample_id"], "subset": r["subset"],
                         "source_group": r["source_group"],
                         "width": r["image_width"], "height": r["image_height"],
                         "no_error": r["annotation"]["no_error"],
                         "objects": {"bbox_xywh": [b["bbox_xywh"] for b in boxes],
                                     "category": [b["category"] for b in boxes],
                                     "description": [b["detail"] for b in boxes]}})
    jsonl(output / "images/metadata.jsonl", metadata)
    jsonl(output / "annotations/inputs_image_only.jsonl", [
        {k: r[k] for k in ("sample_id", "image_path", "image_width", "image_height")} for r in rows])
    jsonl(output / "image_checksums.jsonl", fingerprints)
    stats = defaultdict(lambda: {"images": 0, "flawed": 0, "clean": 0, "regions": 0})
    for r in rows:
        s = stats[r["source_group"]]
        s["images"] += 1; s["regions"] += len(r["annotation"]["boxes"])
        s["clean" if r["annotation"]["no_error"] else "flawed"] += 1
    summary = {"images": 1039, "flawed": 718, "clean": 321, "regions": 1983,
               "decoded_pixels_preserved": 1039, "sources": dict(stats),
               "image_bytes": sum(r["bytes"] for r in fingerprints),
               "metadata_policy": "Lossless removal of EXIF, comments and identifying text; color information preserved."}
    (output / "dataset_statistics.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
