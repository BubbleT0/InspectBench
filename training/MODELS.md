# Paper model weights

The six original fine-tuned paper checkpoints are uploaded as separate Hugging Face model repositories. Each contains 12 original safetensors shards (about 54.7 GB), a shard index, tokenizer, chat template, model/processor configuration and sanitized training arguments. SHA-256 values for all remote weight shards were checked against the local originals. The exported tensors were also checked against the shard index, shapes, byte offsets and total size. No training, quantization or weight conversion was performed.

| Paper checkpoint | Hugging Face repository | Verified revision | Visibility |
|---|---|---|---|
| Workflow-SFT | [BubbleT0/InspectBench-Workflow-SFT](https://huggingface.co/BubbleT0/InspectBench-Workflow-SFT) | `2cbd9cb0d1f77973f1123df6f94c399555bad8eb` | public |
| Trajectory-SFT-R1 | [BubbleT0/InspectBench-Trajectory-SFT-R1](https://huggingface.co/BubbleT0/InspectBench-Trajectory-SFT-R1) | `bdac629496cd3e3776e40c95dfe89a5fa5ef7718` | public |
| Trajectory-SFT-R2 | [BubbleT0/InspectBench-Trajectory-SFT-R2](https://huggingface.co/BubbleT0/InspectBench-Trajectory-SFT-R2) | `3f29c7b5a13998978f5d8af4ed3fc9bc7a058c19` | public |
| Decision-SFT | [BubbleT0/InspectBench-Decision-SFT](https://huggingface.co/BubbleT0/InspectBench-Decision-SFT) | `e735d2dd4fba0e3f44a35532d749bb2f4d6d8c33` | public |
| Decision-SFT-Balanced+ | [BubbleT0/InspectBench-Decision-SFT-BalancedPlus](https://huggingface.co/BubbleT0/InspectBench-Decision-SFT-BalancedPlus) | `98148ee8d98c6fcc9bf441ea0047a29c1ed29b44` | public |
| Decision-SFT-Pool+ | [BubbleT0/InspectBench-Decision-SFT-PoolPlus](https://huggingface.co/BubbleT0/InspectBench-Decision-SFT-PoolPlus) | `f3fbe37b382faadd3db0fab74bfb0cea01f80433` | public |

`model_releases.json` records machine-readable versions and validation counts. `SHA256SUMS.txt` in each model repository provides file hashes. The six model repositories are public. The InspectBench dataset remains private and requires an authorized account to download.

```python
import json
from huggingface_hub import snapshot_download

releases = json.load(open("training/model_releases.json"))
release = next(r for r in releases if r["key"] == "decision_sft")
model_dir = snapshot_download(release["repo_id"], revision=release["commit"])
```

The checkpoint configuration declares `Qwen3_5ForConditionalGeneration` / `qwen3_5`, BF16 parameters, and Transformers 5.2.0. The paper names the backbone Qwen3.6-27B; the export preserves the actual saved architecture/configuration. Use a serving stack that supports this architecture. Full GPU loading/inference was not rerun as part of publication.

Serve the chosen checkpoint using an OpenAI-compatible vision-language endpoint. Use the Direct evaluator in `evaluation/` or the [paper pipeline launcher](../pipelines/README.md) with the matching checkpoint/procedure pairing. The launcher validates the saved harness settings and does not verify the weights behind an arbitrary endpoint.

The weights above are the six fine-tuned checkpoints introduced/evaluated in this work. Third-party base-model and proprietary API baselines are identified in the evaluation records; their weights are not duplicated here. Model weights live on Hugging Face rather than in GitHub or the code ZIP. Recovered Workflow-SFT and Decision-SFT training data are released separately; see [DATASETS.md](DATASETS.md) for coverage and missing-image details. Optimizer states are outside this release.
