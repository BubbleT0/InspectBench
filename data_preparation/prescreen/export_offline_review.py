#!/usr/bin/env python3
import argparse
import concurrent.futures
import hashlib
import json
import os
import pathlib
import re
import shutil
from datetime import datetime, timezone

from dashboard import run_chain, slim_record


HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_RUN = pathlib.Path("/path/to/storage/rawdata_data/critic/classify/continuation01")


def safe_name(value):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("_") or "unknown"


def read_latest_records(run_dir):
    latest = {}
    for linked_run in run_chain(run_dir):
        decisions_path = linked_run / "decisions.jsonl"
        if not decisions_path.exists():
            continue
        with decisions_path.open(encoding="utf-8", errors="replace") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = record.get("item_id") or record.get("call_id")
                latest[key] = {
                    "record": record,
                    "source_run": str(linked_run),
                    "source_log": str(decisions_path),
                    "source_line": line_number,
                }
    return sorted(latest.values(), key=lambda item: item["record"].get("completed_at", ""), reverse=True)


def copy_with_hash(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with source.open("rb") as reader, temporary.open("wb") as writer:
        while True:
            block = reader.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
            writer.write(block)
    shutil.copystat(source, temporary)
    os.replace(temporary, destination)
    return digest.hexdigest(), source.stat().st_size


def prepare_copy(item, output):
    record = item["record"]
    source = pathlib.Path(record["image_path"]).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    dataset = safe_name(record.get("dataset"))
    key = hashlib.sha256(str(record.get("item_id") or source).encode("utf-8")).hexdigest()[:24]
    suffix = source.suffix.lower() or ".img"
    relative = pathlib.Path("images") / dataset / f"{key}{suffix}"
    digest, size = copy_with_hash(source, output / relative)
    return record.get("item_id"), str(relative).replace(os.sep, "/"), digest, size


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Export a standalone offline review bundle")
    parser.add_argument("--run-dir", type=pathlib.Path, default=DEFAULT_RUN)
    parser.add_argument("--output", type=pathlib.Path, default=ROOT / "critic_benchmark_offline_review")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=250)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    records_dir = output / "records"
    data_dir = output / "data"
    records_dir.mkdir(exist_ok=True)
    data_dir.mkdir(exist_ok=True)

    items = read_latest_records(args.run_dir.resolve())
    print(f"Exporting {len(items)} records and images to {output}", flush=True)
    copy_results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, min(args.workers, 32))) as executor:
        futures = {executor.submit(prepare_copy, item, output): item for item in items}
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            item_id, relative, digest, size = future.result()
            copy_results[item_id] = {"offline_image": relative, "image_sha256": digest, "image_bytes": size}
            if index % 250 == 0 or index == len(items):
                print(f"Copied {index}/{len(items)} images", flush=True)

    catalog = []
    manifest_path = output / "export_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for item in items:
            record = item["record"]
            item_id = record.get("item_id")
            copied = copy_results[item_id]
            record_key = hashlib.sha256(str(item_id).encode("utf-8")).hexdigest()[:24]
            record_relative = pathlib.Path("records") / safe_name(record.get("dataset")) / f"{record_key}.json"
            provenance = {
                "source_run": item["source_run"],
                "source_log": item["source_log"],
                "source_line": item["source_line"],
                "original_image_path": record.get("image_path"),
                **copied,
            }
            write_json(output / record_relative, {"offline_provenance": provenance, "original_decision_record": record})
            slim = slim_record(record)
            slim.update({
                "offline_image": copied["offline_image"],
                "original_image_path": record.get("image_path"),
                "copied_image_sha256": copied["image_sha256"],
                "copied_image_bytes": copied["image_bytes"],
                "raw_record_path": str(record_relative).replace(os.sep, "/"),
                "source_run": item["source_run"],
                "source_log": item["source_log"],
                "source_line": item["source_line"],
            })
            catalog.append(slim)
            manifest.write(json.dumps({
                "item_id": item_id,
                "dataset": record.get("dataset"),
                "status": record.get("status"),
                "profile": record.get("profile"),
                "primary_leaf": (record.get("decision") or {}).get("primary_leaf"),
                **provenance,
                "raw_record_path": str(record_relative).replace(os.sep, "/"),
            }, ensure_ascii=False) + "\n")

    chunk_files = []
    for index in range(0, len(catalog), args.chunk_size):
        chunk = catalog[index:index + args.chunk_size]
        name = f"records_{index // args.chunk_size:03d}.js"
        (data_dir / name).write_text(
            "window.OFFLINE_RECORDS.push(..." + json.dumps(chunk, ensure_ascii=False, separators=(",", ":")) + ");\n",
            encoding="utf-8",
        )
        chunk_files.append(f"data/{name}")

    state = json.loads((args.run_dir / "state.json").read_text(encoding="utf-8"))
    plan = json.loads((HERE / "plan.json").read_text(encoding="utf-8"))
    config = {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "source_run": str(args.run_dir.resolve()),
        "run_chain": [str(path) for path in run_chain(args.run_dir.resolve())],
        "record_count": len(catalog),
        "accepted_count": sum(row["accepted"] for row in catalog),
        "image_bytes": sum(row["copied_image_bytes"] for row in catalog),
        "state": state,
        "plan": plan,
    }
    (data_dir / "bootstrap.js").write_text(
        "window.OFFLINE_RECORDS=[];window.OFFLINE_CONFIG=" + json.dumps(config, ensure_ascii=False, separators=(",", ":")) + ";\n",
        encoding="utf-8",
    )
    scripts = "\n".join(f'<script src="{path}"></script>' for path in chunk_files)
    template = (HERE / "offline_template/index.html").read_text(encoding="utf-8")
    (output / "index.html").write_text(template.replace("<!-- DATA_CHUNKS -->", scripts), encoding="utf-8")
    shutil.copy2(HERE / "offline_template/app.js", output / "app.js")
    shutil.copy2(HERE / "offline_template/style.css", output / "style.css")
    shutil.copy2(HERE / "offline_template/README_OFFLINE.md", output / "README_OFFLINE.md")
    shutil.copy2(HERE / "offline_template/serve_offline.py", output / "serve_offline.py")
    shutil.copy2(ROOT / "summary.md", output / "summary.md")
    shutil.copy2(ROOT / "taxonomy.md", output / "taxonomy.md")
    shutil.copy2(HERE / "plan.json", output / "plan.json")
    write_json(output / "review_annotations_template.json", {
        "format": "critic-benchmark-manual-review-v1",
        "exported_at": None,
        "reviewer": "",
        "annotations": {},
    })
    print(json.dumps({
        "output": str(output),
        "records": len(catalog),
        "accepted": config["accepted_count"],
        "images_gib": round(config["image_bytes"] / 1024 ** 3, 3),
        "chunks": len(chunk_files),
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
