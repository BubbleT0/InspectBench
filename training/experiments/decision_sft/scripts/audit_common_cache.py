#!/usr/bin/env python3
import json
from pathlib import Path

from datasets import load_from_disk


BASE = Path(
    "/path/to/storage/multimodal_ckpt/research/autotrain/experiments/"
    "critic_singlemodel_r123_8gpu_20260824"
)


def main():
    result = {}
    for split, expected in (("train", 4000), ("validation", 800)):
        with (BASE / "data/dev_4000_800" / f"{split}.jsonl").open(encoding="utf-8") as handle:
            source = [json.loads(line) for line in handle if line.strip()]
        dataset = load_from_disk(str(BASE / "cache_full_common_v1" / split / "train"))
        assert len(source) == len(dataset) == expected
        for index in range(expected):
            assert source[index]["messages"] == dataset[index]["messages"], (split, index, "messages")
            cached_images = [item["path"] if isinstance(item, dict) else item for item in dataset[index]["images"]]
            assert source[index]["images"] == cached_images, (split, index, "images")
        result[split] = {
            "rows": len(dataset),
            "order_exact": True,
            "columns": dataset.column_names,
        }
    print(json.dumps(result, indent=2))
    print("common cache exact match: PASS")


if __name__ == "__main__":
    main()
