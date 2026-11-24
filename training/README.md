# Training artifacts

This directory connects the six paper checkpoints to recovered source files and arguments saved with the actual checkpoints. No training was rerun to prepare this release.

| Paper checkpoint | Key | Saved step | Epochs | Sequence length | LR | Global batch | TP / PP / CP / DP |
|---|---|---:|---:|---:|---:|---:|---|
| Workflow-SFT | workflow_sft | 145 | 3 | 131,072 | 2e-5 | 64 | 2 / 4 / 4 / 2 |
| Trajectory-SFT-R1 | trajectory_r1 | 740 | 1 | 8,192 | 1e-6 | 8 | 2 / 4 / 1 / 2 |
| Trajectory-SFT-R2 | trajectory_r2 | 1001 | 1 | 8,192 | 1e-6 | 8 | 2 / 4 / 1 / 2 |
| Decision-SFT | decision_sft | 136 | 1 | 8,192 | 1e-6 | 8 | 2 / 4 / 1 / 2 |
| Decision-SFT-Balanced+ | decision_balanced | 444 | 1 | 8,192 | 1e-6 | 8 | 2 / 4 / 1 / 2 |
| Decision-SFT-Pool+ | decision_pool | 688 | 1 | 8,192 | 1e-6 | 8 | 2 / 4 / 1 / 2 |

These are recovered values from `checkpoints/<key>/args.json`, not values inferred from directory names. All have micro-batch size 1. `checkpoints.json` provides a compact machine-readable index. Checkpoint directories also contain model/processor configuration and the chat template, with full inference weights and tokenizer files linked in [MODELS.md](MODELS.md).

## Source map

- `workflow_preprocessing/`: successful-session filtering, structured quality checks, conversation conversion, repetition and token-counting tools.
- `experiments/workflow_sft/`: original launch/train shell scripts, experiment manifest, mixture metadata and repeat map. Workflow-SFT uses 8,834 original conversations and a 12,312-record repeated mixture.
- `experiments/trajectory_r1/`: repeat-1 data preparation, stratified validation, training and validation scripts.
- `experiments/trajectory_r2/`: four-source trajectory construction, repeat-2 packing and training scripts.
- `experiments/decision_sft/`: balanced 4,000/800 data construction, packing, data audits, default/weighted-loss entry points and training scripts. The paper's checkpoint uses `r1_formal_packed` and default loss; the archived scripts also contain variants not evaluated in the paper.
- `experiments/expanded_decision/`: expanded dataset construction and training scripts. Paper Balanced+ corresponds to `a_balanced`; Pool+ corresponds to `c_all_unique`. Other archived variants are not paper results.
- `data_summaries/`: recovered mixture statistics, neutral prompt, repeat maps and conversion metadata.
- `manifests/`: per-record split membership, source/shard and hashes, with no raw conversations or image paths. `statistics.json` contains actual record counts.

## Reuse and dependencies

The scripts are historical training artifacts, not a turnkey installer. Infrastructure-specific paths are replaced with `/path/to/...`, cluster/container identifiers are placeholders, and authentication material is removed. Before running a script, configure its experiment, dataset, model, environment and dependency roots; supply the required source data and checkpoints separately. Preserve the recorded settings if claiming to reproduce a paper configuration.

The saved framework version is `4.2.0.dev0`. The Decision/expanded training dependency lock records ms-swift commit `b8241e85781b8f8c8f3f4b1019082657aef14603`, Megatron-LM commit `31e3277683dbf06cff4732cbc0033573437d1097`, and mcore-bridge commit `c527134caa372e85e8dbf75127d4df02b088bd44`. A commit identifier alone does not establish that a historical fork or all local modifications are publicly retrievable. Framework availability, GPU topology and data access remain required for rerunning training.

Full inference weights are uploaded separately to Hugging Face; see [MODELS.md](MODELS.md). Raw training conversations, source images, optimizer states and caches are not included in this release. A hashed split manifest is not a replacement for those inputs. `image_identity_sha256` hashes the resolved source identity and is not an image-content hash; `file_sha256` is included only where independently cached content fingerprints are available.

## Label and evaluation scope

Decision-SFT's Wikimedia/Pinterest negatives are assigned clean labels during construction after discarding prescreener boxes. They have not received an independent human label audit. The benchmark's cross-reviewed controls use a separate annotation process. The saved Decision-SFT exact/perceptual separation audit covers its 4,000 training and 800 validation examples versus the 1,039 benchmark images; it does not establish separation for all other checkpoints or pretraining data.
