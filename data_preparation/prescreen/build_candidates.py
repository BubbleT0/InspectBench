#!/usr/bin/env python3
import argparse
import hashlib
import json
import pathlib
import re


HERE = pathlib.Path(__file__).resolve().parent
RAW = pathlib.Path("/path/to/storage/rawdata_data/critic")


def read_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def arti_profile(explanation):
    text = (explanation or "").lower()
    topology = r"\b(diagram|flowchart|arrow|circuit|wire|line art|pathway|icon|symbol)\b"
    physical = r"\b(float(?:ing)?|suspend(?:ed)?|gravity|shadow|reflection|mirror|perspective|interpenetrat|passing through|support|impossible pose|relative scale)\b"
    rendering = r"\b(blurr|smear|smudg|texture|color corruption|discolor|pixelat|detail loss|poorly rendered|unfinished render)\b"
    if re.search(topology, text):
        return "D1-A3", "explanation contains an explicit connection/diagram cue"
    if re.search(physical, text):
        return "D1-A4", "explanation contains a relational/physical cue"
    if re.search(rendering, text):
        return "D1-A1", "explanation contains a local rendering cue"
    return "D1-A2", "default artifact route: entity structure"


def salart_profile(artifact_type):
    mapping = {
        "local_render_structure_anomaly": "D1-A1",
        "anatomy_anomaly": "D1-A2",
        "count_anomaly": "D1-A2",
        "topology_anomaly": "D1-A2",
        "plausibility_anomaly": "D1-A4",
    }
    return mapping.get(artifact_type, "D1-A2")


def xaigd_profile(labels, prompt):
    label_set = set(labels or [])
    prompt_text = (prompt or "").lower()
    if label_set & {"cognitive-level-physics", "cognitive-level-commonsense"}:
        return "D1-A4", "physics/commonsense coarse label"
    if "low-level-symbol" in label_set:
        text_cue = re.search(r"\b(text|word|letter|number|caption|title|sign|signage|poster|logo|label|written|says|reads)\b", prompt_text)
        diagram_cue = re.search(r"\b(diagram|flowchart|chart|arrow|circuit|schematic|icon|symbol|map)\b", prompt_text)
        if text_cue:
            return "D2-A", "symbol label plus textual prompt cue"
        if diagram_cue:
            return "D1-A3", "symbol label plus explicit graphic-structure cue"
    if "high-level-semantics" in label_set:
        return "D1-A2", "high-level semantics coarse label"
    if label_set & {"low-level-texture", "low-level-color"}:
        return "D1-A1", "texture/color coarse label"
    if "low-level-symbol" in label_set:
        return "D1-A3", "symbol coarse label without a text cue"
    return "D1-A2", "edge/shape default route: entity structure"


def build_artibench():
    root = RAW / "ArtiBench"
    for row in read_jsonl(root / "metadata.jsonl"):
        if not row.get("has_artifacts"):
            continue
        profile, reason = arti_profile(row.get("explanation"))
        yield {
            "id": f"ArtiBench::{row['id']}",
            "dataset": "ArtiBench",
            "image_path": str((root / row["file_name"]).resolve()),
            "profile": profile,
            "metadata": {
                "source_role": "artifact",
                "source_id": row["id"],
                "bboxes": row.get("bboxes", []),
                "explanation": row.get("explanation", ""),
                "routing_reason": reason,
            },
        }


def build_salart():
    root = RAW / "SalArt-VQA"
    for row in read_jsonl(root / "data/test.jsonl"):
        if row.get("image_role") != "artifact":
            continue
        artifact_type = row.get("artifact_type")
        q2_answer = row.get("q2_answer")
        q3_answer = row.get("q3_answer")
        q4_answer = row.get("q4_answer")
        yield {
            "id": f"SalArt-VQA::{row['row_id']}",
            "dataset": "SalArt-VQA",
            "image_path": str((root / row["image_path"]).resolve()),
            "profile": salart_profile(artifact_type),
            "metadata": {
                "source_role": "artifact",
                "artifact_type": artifact_type,
                "q2_selected_region": (row.get("q2_options") or {}).get(q2_answer),
                "q3_selected_box": (row.get("q3_boxes") or {}).get(q3_answer),
                "q4_selected_description": (row.get("q4_options") or {}).get(q4_answer),
                "routing_note": "ordinary-entity topology_anomaly is intentionally routed to D1-A2",
            },
        }


def build_xaigd():
    for split in ("test", "train"):
        manifest = RAW / f"X-AIGD/labeled_{split}_manifest.jsonl"
        for row in read_jsonl(manifest):
            labels = row.get("labels") or []
            if not labels:
                continue
            profile, reason = xaigd_profile(labels, row.get("prompt"))
            yield {
                "id": f"X-AIGD::{split}::{row['uid']}::{row['generator']}",
                "dataset": "X-AIGD",
                "image_path": str(pathlib.Path(row["image_path"]).resolve()),
                "profile": profile,
                "metadata": {
                    "source_role": "labeled_artifact",
                    "source_split": split,
                    "uid": row["uid"],
                    "generator": row["generator"],
                    "prompt": row.get("prompt", ""),
                    "coarse_labels": sorted(set(labels)),
                    "polygon_count": row.get("polygon_count", 0),
                    "routing_reason": reason,
                },
            }


def build_textpecker():
    root = RAW / "TextPecker-1.5M/decoded_test"
    seen = set()
    image_suffixes = {".png", ".jpg", ".jpeg", ".webp"}
    for path in sorted(root.glob("*/*")):
        if not path.is_file() or path.suffix.lower() not in image_suffixes:
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        family = path.parent.name.rsplit("_", 1)[0]
        yield {
            "id": f"TextPecker-1.5M::{digest[:20]}",
            "dataset": "TextPecker-1.5M",
            "image_path": str(path.resolve()),
            "profile": "D2-A",
            "metadata": {
                "source_role": "existing_text_visual_artifact",
                "decoded_family": family,
                "source_directory": path.parent.name,
                "sha256": digest,
                "target_text_available": False,
            },
        }


def main():
    parser = argparse.ArgumentParser(description="Build the VLM candidate manifest from unpacked datasets")
    parser.add_argument("--output", type=pathlib.Path, default=HERE / "candidates.jsonl")
    parser.add_argument("--pilot-output", type=pathlib.Path, default=HERE / "pilot_candidates.jsonl")
    parser.add_argument("--pilot-per-pool", type=int, default=20)
    args = parser.parse_args()
    builders = [build_artibench, build_salart, build_xaigd, build_textpecker]
    counts = {}
    pilot_rows = {}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for builder in builders:
            for row in builder():
                image_path = pathlib.Path(row["image_path"])
                if not image_path.is_file():
                    raise FileNotFoundError(f"manifest source image is missing: {image_path}")
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                key = f"{row['dataset']}::{row['profile']}"
                counts[key] = counts.get(key, 0) + 1
                pool = pilot_rows.setdefault(key, [])
                if len(pool) < args.pilot_per_pool:
                    pool.append(row)
    args.pilot_output.parent.mkdir(parents=True, exist_ok=True)
    with args.pilot_output.open("w", encoding="utf-8") as handle:
        for index in range(args.pilot_per_pool):
            for key in sorted(pilot_rows):
                pool = pilot_rows[key]
                if index < len(pool):
                    handle.write(json.dumps(pool[index], ensure_ascii=False) + "\n")
    pilot_total = sum(len(rows) for rows in pilot_rows.values())
    print(json.dumps({
        "output": str(args.output.resolve()),
        "total": sum(counts.values()),
        "pilot_output": str(args.pilot_output.resolve()),
        "pilot_total": pilot_total,
        "counts": counts,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
