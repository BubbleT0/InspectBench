# InspectBench

InspectBench evaluates **image-only defect coverage and false-positive restraint**. A critic receives an image without its generation prompt and returns localized defect regions, or an empty set for a clean image.

The frozen paper benchmark contains **1,039 images, 718 flawed images, 321 clean controls and 1,983 reference regions**. This repository collects evaluation and inference code, the available inspection-pipeline implementations, recorded training configurations and data-preparation scripts, and all 17 saved prediction sets. The pixel-preserving benchmark images and annotations are available on [Hugging Face](https://huggingface.co/datasets/BubbleT0/InspectBench). The original GitHub v1.0 image assets remain available in [Releases](https://github.com/BubbleT0/InspectBench/releases/tag/v1.0).

The model repositories are public; the benchmark dataset currently requires an authorized Hugging Face account.

## Contents

| Directory | Contents |
|---|---|
| `benchmark/` | Final annotations, image-only input manifest, source statistics and image checksums |
| `annotations/` | Original v1.0 subset files, retained for compatibility |
| `scorer/` | Original published standalone scorer and annotation converter |
| `evaluation/` | Direct model runner, adapters, prompts, metrics and tests |
| `pipelines/` | Saved Full-Seq and inspection implementations, plus six recovered run configurations |
| `training/` | Six checkpoint argument snapshots, training/preprocessing scripts, split manifests and dependency records |
| `reproducibility/` | 17 predictions, reference scores, independently checked scorer and offline analyses |
| `data_preparation/` | Final data selection, prescreening and offline review tools |
| `paper_analysis/` | Paper table/figure generation code and frozen aggregate inputs |
| `scripts/` | Dataset download, local-manifest preparation, export and integrity checks |
| `docs/` | Scoring specification, taxonomy, provenance and limitations |

## Dataset

| Source | Images | Flawed | Clean | Regions |
|---|---:|---:|---:|---:|
| GPT-image-2 | 208 | 97 | 111 | 266 |
| Public-4 | 499 | 499 | 0 | 1,126 |
| SenseNova-U1.5 | 222 | 122 | 100 | 591 |
| Natural | 110 | 0 | 110 | 0 |
| Total | 1,039 | 718 | 321 | 1,983 |

The dataset is an evaluation split. Clean controls are not content-matched to flawed images. Ground-truth boxes measure recovery of the retained, cross-reviewed annotations; they do not certify exhaustive defect annotation.

```bash
python -m pip install -r requirements-reproduction.txt
python scripts/download_dataset.py \
  --repo-id BubbleT0/InspectBench --local-dir data/InspectBench
python scripts/verify_release.py --dataset-root data/InspectBench --hash-images
python scripts/prepare_local_dataset.py \
  --dataset-root data/InspectBench --output-dir .local/benchmark
```

Public IDs `IMG_0001` through `IMG_1039` match the existing GitHub release. Ground-truth `bbox_xywh` coordinates are absolute image pixels. Prediction `bbox_1000_xywh` coordinates use a 0-1000 grid. Image dimensions and decoded pixels are preserved; only identifying metadata and auxiliary MPO data are removed. See `docs/DATASET.md` for schema and version details.

## Score saved or new predictions

The original scorer needs only Python 3.9 or later:

```bash
python scorer/bbox_scorer.py \
  --annotations benchmark/annotations.jsonl \
  --predictions reproducibility/inputs/predictions/config_01.jsonl \
  --output metrics.json --bootstrap 2000
```

Run the independent specification tests and reproduce all offline analyses:

```bash
python -m unittest discover -s reproducibility/scorer -p 'test_scorer.py'
python scripts/reproduce_paper_scores.py
python reproducibility/analysis/analyze_existing.py
```

`reproducibility/inputs/configurations.json` maps files to the 17 paper configurations and unrounded reference scores. The analysis verifies score reproduction and computes ownership sensitivity, conventional matching, source strata, parameter sensitivity and paired bootstrap intervals. It invokes no model APIs. See `reproducibility/README.md` for statistical scope.

## Run models and inspect training records

Direct inference requires Python 3.11 or later:

```bash
python -m pip install -e evaluation
export INSPECTBENCH_API_KEY=YOUR_KEY
python -m vlm_judge_bench run \
  --annotations .local/benchmark/annotations.jsonl \
  --models-config evaluation/configs/models.example.yaml \
  --models local_critic --output-dir .local/runs/direct
```

Set the endpoint and served model name in the example configuration before running it. Credentials are read from environment variables. `pipelines/README.md` describes the saved procedures and provides `pipelines/run_paper_pipeline.py` to launch all six paper pairings with validated archived settings. `training/README.md` connects the six checkpoint snapshots to their data preparation and training files.

Recorded training arguments are provided as evidence of the actual checkpoint settings. Training scripts retain framework and hardware assumptions; private infrastructure locations have been replaced with `/path/to/...` placeholders. The six original paper checkpoint weights are available in the [model release index](training/MODELS.md), with pinned Hugging Face revisions and verified shard hashes. Raw training conversations/images are not included. It does not claim that copying a configuration recreates the historical training environment or proprietary model endpoints.

## Interpretation and usage

Report region recall and clean FPR alongside the composite score. The 0.1 cutoff is an area-penalized quality threshold, not IoU. Containment-first ownership can reduce credit after adding a broad box, and changing the clean/flawed weight can change rankings. Historical pipeline timings use 1,054-image batches and should not be treated as synchronized costs for the final 1,039-image evaluation.

The existing release is provided as-is for research and evaluation. Third-party image rights remain with their respective owners; no blanket permissive license for source images is asserted. Source and release notes are in `docs/DATASET.md`.
