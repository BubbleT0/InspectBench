# Public training datasets

The main checkpoint chain is Workflow-SFT followed by Decision-SFT. Both recovered
input datasets are public; the InspectBench evaluation benchmark remains private.

| Dataset | Records | Image coverage |
|---|---|---|
| [Workflow-SFT](https://huggingface.co/datasets/BubbleT0/InspectBench-Workflow-SFT-Data) | 8,834 conversations in 27 shards; 12,312 after saved repetition factors | Text SFT inputs; original linked workflow assets/tool environments are not bundled |
| [Decision-SFT](https://huggingface.co/datasets/BubbleT0/InspectBench-Decision-SFT-Data) | All 4,000 train + 800 validation conversations/labels | 3,000 train + 600 validation images present; 1,000 + 200 missing |

`dataset_releases.json` pins the verified revisions and records the file counts,
sizes and completeness statistics. Every uploaded payload file was checked against
Hub object hashes, and public access was checked without authentication. Dataset
cards and manifests explain the release transformations and limitations.

## Coverage and transformations

Workflow-SFT preserves all saved text conversations, roles and turn order.
Credential-shaped values were redacted and internal account identifiers anonymized
in 1,374 conversations. `record_manifest.jsonl` records per-row transformations
and source/export hashes. This is complete saved text-input coverage, not a
byte-identical copy or a full environment for rerunning the recorded workflows.

Decision-SFT preserves all messages and labels unchanged. The 1,200 unavailable
images all belong to clean Wikimedia examples, with sample IDs and source-relative
paths recorded in `missing_images.jsonl`. No substitutes were inserted. PNG/JPEG
metadata was removed; 16 WebP and 3 GIF images were exported to PNG. Primary-frame
pixels and dimensions were verified for all 3,600 included images. Two sources
had multiple frames; only the primary frame is included.

The Decision-SFT `full` config preserves all 4,800 rows. Missing-image rows have
`images: []` and `image_available: false` and are **not valid text-only training
examples**. The `available` config selects only the 3,000/600 usable rows. This
changes the source/label mixture and cannot reproduce the historical full-set
training result. The original checkpoint weights remain available in [MODELS.md](MODELS.md).

## Prepare model-input files

Install `huggingface_hub`, then download pinned versions:

```python
import json
from huggingface_hub import snapshot_download

releases = json.load(open("training/dataset_releases.json"))
for key, release in releases.items():
    snapshot_download(release["repo_id"], repo_type="dataset",
                      revision=release["commit"], local_dir=f"data/{key}")
```

```bash
python scripts/prepare_training_data.py data/workflow_sft \
  --output-dir .local/workflow-inputs --apply-repeats
python scripts/prepare_training_data.py data/decision_sft \
  --output-dir .local/decision-inputs --available-only
```

The helper strips release metadata, emits `messages` (and Decision-SFT `images`),
and resolves image paths against the local download. It refuses Decision-SFT
preparation without explicit `--available-only` because original inputs are missing.
Workflow repetition reproduces saved mixture counts, not historical shuffling or
token packing. Use the archived training settings and maintain the image download
while training. Each dataset also includes this helper and `SHA256SUMS.txt`.

Raw inputs for the other four fine-tuned checkpoints, optimizer states and training
caches are not part of these datasets. No new blanket license is asserted for
third-party training content; consult the dataset cards and original source terms.
