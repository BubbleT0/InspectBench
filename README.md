# InspectBench

A benchmark for evaluating vision-language models as **image anomaly critics**:
given an AI-generated (or natural) image, the model must (1) decide whether the
image contains visual anomalies, and (2) localize every anomaly with a tight
bounding box — while staying restrained on clean images.

This repository contains the full benchmark data (1,039 images with human
ground-truth boxes) and a standalone, dependency-free scorer.

## Why this benchmark

Most anomaly-detection benchmarks evaluate only image-level yes/no judgments,
or use a single strict IoU threshold that collapses all localization nuance
into one bit. InspectBench instead measures a continuous localization quality
per ground-truth box, explicitly handles the common "one big box covering
several adjacent anomalies" behavior, penalizes duplicate and background boxes,
and heavily weights restraint on clean images. The result is a single main
score that cannot be gamed by either over- or under-triggering.

## Data

| Subset | Images | Positive (anomalous) | Clean | GT boxes |
| --- | ---: | ---: | ---: | ---: |
| A | 208 | 97 | 111 | 266 |
| B | 499 | 499 | 0 | 1,126 |
| C | 141 | 122 | 19 | 591 |
| D | 81 | 0 | 81 | 0 |
| E | 110 | 0 | 110 | 0 |
| **Total** | **1,039** | **718** | **321** | **1,983** |

- Subsets A–C mix AI-generated imagery with annotated rendering, structural,
  topology, relational/physical and text anomalies; D and E are clean-only
  subsets for measuring false-positive restraint.
- Annotation format (one JSON object per line, see `annotations/`):

```json
{
  "sample_id": "IMG_0001",
  "image_path": "images/IMG_0001.png",
  "image_width": 2048,
  "image_height": 3072,
  "subset": "A",
  "no_error": false,
  "boxes": [
    {"bbox_xywh": [1363.71, 1879.611, 96.0, 96.0],
     "category": "artifact/structural",
     "note": "human-language description of the anomaly"}
  ]
}
```

- `bbox_xywh` is `[x, y, w, h]` in absolute pixels, top-left origin.
- Categories: `artifact/rendering`, `artifact/structural`, `artifact/topology`,
  `artifact/relational_physical`, `text/visual_integrity`,
  `text/content_fidelity`. Categories are diagnostic only — they never enter
  the main score.

## Task protocol

The model receives the full image and a fixed instruction, and returns:

```json
{
  "no_error": false,
  "boxes": [
    {"bbox_1000_xywh": [100.0, 200.0, 150.0, 120.0],
     "family": "artifact/structural",
     "description": "visible evidence of the anomaly"}
  ]
}
```

- `bbox_1000_xywh` is `[x, y, w, h]` normalized to a 0–1000 grid on both axes.
- `family` and `description` are optional and diagnostic only.
- Clean images should return `no_error: true` with an empty `boxes` list.

## Scoring

The scorer is a single file with **no third-party dependencies**
(Python >= 3.9 standard library only): `scorer/bbox_scorer.py`.
Full metric definitions and rationale: `scorer/README_SCORER.md`.

### Quick start

```bash
# 1. merge the anonymized annotation subsets into scorer input format
python scorer/prepare_annotations.py \
    --annotations annotations/annotations_subset_*.jsonl \
    --image-root . \
    --output gt_merged.jsonl

# 2. score a predictions file (same sample_ids as the annotations)
python scorer/bbox_scorer.py \
    --annotations gt_merged.jsonl \
    --predictions your_predictions.jsonl \
    --output metrics.json \
    --per-sample-out per_sample.json \
    --bootstrap 2000
```

### The main score in one paragraph

Each ground-truth (GT) box elects exactly one owning prediction (highest
coverage, tie-broken by base quality, then output order). A prediction that
owns K >= 2 GTs is re-scored against their geometric union with a sqrt(K)-widened
area tolerance, so tight merges are rewarded while blob boxes are not. Each GT
receives a continuous quality in [0, 1] combining coverage, tightness (Dice)
and a smooth area-ratio penalty; unmatched predictions earn zero credit. A
positive image's score is the harmonic mean of soft precision and soft recall;
a clean image scores exp(-N) for N predicted boxes. The headline is
`100 * (0.7*sqrt(S_pos) + 0.3*sqrt(S_clean))^2`, where S_pos is the
sqrt(#GT)-weighted mean over positive images and S_clean the mean over clean
images.

Key details (exact formulas, tie-breaking, failure handling) are documented in
`scorer/README_SCORER.md` — including the area-tolerance curve
`tau(f) = 1.25 + 0.35*exp(-f/0.01)`, the ownership tie-break order, and the
policy that failed samples are scored as empty predictions (positive images
get 0, clean images get 1 — always read together with the failure count).

## Repository layout

```
annotations/            5 anonymized JSONL subset files (1,039 rows total)
scorer/
  bbox_scorer.py        standalone scorer (stdlib only)
  prepare_annotations.py  converts the anonymized format to scorer input
  README_SCORER.md      complete metric documentation
```

The 1,039 benchmark images (`images/IMG_0001..IMG_1039`, ~2.9 GB) are **not
stored in the git tree**. Download them from the repository's
[Releases](../../releases) page: grab `InspectBench_images_part1.zip` and
`InspectBench_images_part2.zip` and extract both archives into the repository
root — they will create the `images/` directory with all files. Image
metadata (EXIF, JPEG comments) has been stripped for anonymization.

## License / usage

Released for research and double-blind review. The benchmark data is provided
as-is for evaluation purposes.
