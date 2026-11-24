#!/usr/bin/env python3
"""Exercise critic_decision_v1 through the real Qwen VLM template encoder."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


BASE = Path(__file__).resolve().parent
ENTRY = BASE / "critic_weighted_entry.py"
DATA = Path(
    "/path/to/storage/multimodal_ckpt/research/autotrain/experiments/"
    "critic_singlemodel_r123_8gpu_20260824/data/dev_4000_800/train.jsonl"
)
MODEL = (
    "/path/to/storage/multimodal_ckpt/research/autotrain/output/"
    "Qwen3.6-27B_TP2_CP4_PP4_VPP2_EP1_ETP1_mbs1_131072_agen_0722_"
    "agentv4_final_rp_3epv_0-20260722-214144_checkpoint-145"
)


def load_entry():
    spec = importlib.util.spec_from_file_location("critic_weighted_entry", ENTRY)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main() -> None:
    module = load_entry()
    module._register_loss_scale()

    from swift.model import get_processor
    from swift.template import TemplateInputs, get_template

    processor = get_processor(MODEL)
    template = get_template(
        processor,
        max_length=8192,
        max_pixels=200704,
        truncation_strategy="delete",
        loss_scale="critic_decision_v1",
        enable_thinking=False,
        add_non_thinking_prefix=False,
        response_prefix="",
    )
    template.mode = "train"

    rows = []
    with DATA.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            target = json.loads(row["messages"][-1]["content"])
            rows.append((row, bool(target["no_error"])))
            if len(rows) == 2:
                break
    assert [flag for _, flag in rows] == [True, False], "dataset must alternate clean/error"

    for index, (row, is_clean) in enumerate(rows):
        encoded = template.encode(TemplateInputs.from_dict(row))
        scales = encoded["loss_scale"]
        labels = encoded["labels"]
        assert scales is not None and len(scales) == len(labels) == len(encoded["input_ids"])
        supervised = [float(s) for s, label in zip(scales, labels) if label != -100 and float(s) != 0.0]
        assert supervised, f"row {index} has no supervised weighted tokens"
        assert all(value > 0 for value in supervised), supervised[:20]
        factor = module.CLEAN_CLASS_SENTINEL if is_clean else 1.0
        decoded_values = {value / factor for value in supervised}
        assert 10.0 in decoded_values, f"decision span weight missing: {decoded_values}"
        if is_clean:
            assert decoded_values <= {1.0, 10.0}, decoded_values
            assert max(supervised) >= module.CLEAN_CLASS_SENTINEL
        else:
            assert 0.5 in decoded_values and 0.25 in decoded_values, decoded_values
            assert max(supervised) < module.CLEAN_CLASS_SENTINEL
        print(
            json.dumps(
                {
                    "row": index,
                    "class": "clean" if is_clean else "error",
                    "tokens": len(labels),
                    "supervised_tokens": len(supervised),
                    "decoded_weights": sorted(decoded_values),
                },
                ensure_ascii=False,
            )
        )
    print("real template loss-scale integration: PASS")


if __name__ == "__main__":
    main()
