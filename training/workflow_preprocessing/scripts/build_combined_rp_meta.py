#!/usr/bin/env python3
"""Build a non-destructive combined Swift meta with date-aware repeat times."""

import argparse
import json
import os
from collections import Counter
from pathlib import Path


BATCHES = {
    "0715": Path(
        "/path/to/storage/multimodal_data/research/agen/0715_structured_qc_v1/"
        "agent_data_pipeline_main_v1/data_swift_format/meta.json"
    ),
    "0716_0717": Path(
        "/path/to/storage/multimodal_data/research/agen/0716_0717_structured_qc_v2/"
        "agent_data_pipeline_main_v3/data_swift_format/meta.json"
    ),
    "0721": Path(
        "/path/to/storage/multimodal_data/research/agen/0721_structured_qc_v1/"
        "agent_data_pipeline_main_v1/data_swift_format/meta.json"
    ),
}
DEFAULT_OUTPUT = Path(
    "/path/to/storage/multimodal_data/research/agen/"
    "0715_0716_0717_0721_combined_rp_v1"
)


def repeat_time(batch, dataset):
    is_core_agen = dataset == "agen_agen_success"
    if batch == "0721":
        return 4 if is_core_agen else 3
    return 2 if is_core_agen else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output
    if output.exists():
        raise SystemExit(f"refusing existing output: {output}")
    data_dir = output / "data"
    data_dir.mkdir(parents=True)

    combined_meta = {}
    repeat_rows = []
    by_batch = Counter()
    effective_by_batch = Counter()
    by_agent = Counter()
    effective_by_agent = Counter()
    sources = {}
    for batch, meta_path in BATCHES.items():
        source_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        for dataset, item in sorted(source_meta.items()):
            source = Path(item["annotation"])
            if not source.is_file():
                raise RuntimeError(f"missing source annotation: {source}")
            shard_name = f"{batch}_{dataset}"
            destination = data_dir / f"{shard_name}.jsonl"
            os.symlink(source, destination)
            length = int(item["length"])
            repeat = repeat_time(batch, dataset)
            config = dict(item)
            config.update({
                "annotation": str(destination),
                "expect_name": shard_name,
                "length": length,
                "repeat_time": repeat,
                "comment": f"{item.get('comment', '')}; batch={batch}; rp={repeat}",
            })
            combined_meta[shard_name] = config
            repeat_rows.append((shard_name, destination, length * repeat, length, repeat))
            sources[shard_name] = str(source)
            agent = dataset.removeprefix("agen_").removesuffix("_success")
            by_batch[batch] += length
            effective_by_batch[batch] += length * repeat
            by_agent[agent] += length
            effective_by_agent[agent] += length * repeat

    meta_path = output / "meta.json"
    meta_path.write_text(json.dumps(combined_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    repeat_map = output / "repeat_map.tsv"
    with repeat_map.open("w", encoding="utf-8") as handle:
        handle.write("shard_name\tjsonl_path\tN\torig_length\trepeat_time\n")
        for row in repeat_rows:
            handle.write("\t".join(map(str, row)) + "\n")
    summary = {
        "meta": str(meta_path),
        "repeat_map": str(repeat_map),
        "policy": {
            "0715": {"default": 1, "agen": 2},
            "0716_0717": {"default": 1, "agen": 2},
            "0721": {"default": 3, "agen": 4},
        },
        "datasets": len(combined_meta),
        "original_records": sum(by_batch.values()),
        "effective_records": sum(effective_by_batch.values()),
        "records_by_batch": dict(by_batch),
        "effective_records_by_batch": dict(effective_by_batch),
        "records_by_agent": dict(by_agent),
        "effective_records_by_agent": dict(effective_by_agent),
        "source_annotations": sources,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
