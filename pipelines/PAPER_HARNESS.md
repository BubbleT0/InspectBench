# Paper pipeline harness

All six multi-stage pairings use `adversarial_consensus`, with the checkpoint and settings selected by `run_paper_pipeline.py`. The launcher's six presets reproduce the archived parameter values, including derived flags; they use the available source snapshots and do not promise bitwise regeneration of past model responses.

| Component | Full-Seq snapshot | Inspection snapshot |
|---|---|---|
| Runner, concurrency, retries, trace/output files | `full_seq/src/u1_critic_v41/run.py` | `inspection/src/u1_critic_v63/run.py` |
| Model endpoint, messages and prompts | `full_seq/src/u1_critic_v41/model_adapter.py` | `inspection/src/u1_critic_v63/model_adapter.py` |
| Workflow assembly | `full_seq/src/u1_critic_v41/workflows.py` | `inspection/src/u1_critic_v63/workflows.py` |
| Coverage ledger and review | `full_seq/src/u1_critic_v41/v4_harness.py` | `inspection/src/u1_critic_v63/v4_harness.py` |
| Scheduled review | — | `inspection/src/u1_critic_v63/v42_fast.py` |
| Conservative conflict gating | — | `inspection/src/u1_critic_v63/v5_conflict_gate.py` |
| Selective review and dual scouts | — | `inspection/src/u1_critic_v63/v63_selective.py` |
| Adaptive batching/recovery | — | `inspection/src/u1_critic_v63/v62_adaptive.py` |
| Edit-region consensus | `full_seq/src/u1_critic_v41/edit_consensus.py` | `inspection/src/u1_critic_v63/edit_consensus.py` |
| Final consolidation | `full_seq/src/u1_critic_v41/consolidation.py` | `inspection/src/u1_critic_v63/consolidation.py` |
| Image tools and common critic components | `full_seq/src/critic_workflows/` | `inspection/src/critic_workflows/` |

The packages share a `critic_workflows` namespace. The paper launcher selects one package per process and avoids installing both into the same namespace. Code metadata and the benchmark's ground-truth annotations are not added to model input. The launcher takes the image-only manifest created by `scripts/prepare_local_dataset.py`.

Full-Seq detects MIME type from image headers when a historical filename extension differs from its actual encoding, matching the inspection package's handling. Image bytes and region coordinates remain unchanged.

The launcher has a regression check covering all six pairings, especially that Conservative and Full-Sched do not inherit Selective CLI defaults. The recorded paper scores remain reproducible from the 17 saved prediction sets without rerunning model inference.
