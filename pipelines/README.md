# Saved inspection pipelines

The paper compares Direct, Full-Seq, Conservative, Full-Sched and Selective. `full_seq/` contains the saved v4.1 implementation; `inspection/` contains the later implementation with Conservative/v5, scheduled review and v6.3 selective-review code. Each package includes its own `critic_workflows` module, so install/use the snapshots in separate environments to avoid namespace collisions.

| Paper configuration | Historical mode | Recovered settings |
|---|---|---|
| Workflow Full-Seq | harness_v41 / v4.1 | configurations/workflow_full_seq.json |
| Workflow Conservative | harness_v5 | configurations/workflow_conservative.json |
| Workflow Full-Sched | harness_v42 / v4.1-flash | configurations/workflow_full_sched.json |
| Workflow Selective | v6.3 selective review | configurations/workflow_selective.json |
| Trajectory-R2 Selective | v6.3 selective review | configurations/trajectory_r2_selective.json |
| Decision Selective | v6.3 selective review | configurations/decision_selective.json |

The recovered settings include requests/image concurrency, token limits, retries, crop/review batching and routing flags. They describe archived runs, often on the earlier 1,054-image batch; paper scoring retains the final 1,039-image subset. The code snapshots are available implementations, not immutable per-request source archives for every run.

The packages require Python 3.10 or later and Pillow. For example:

```bash
python -m pip install -e pipelines/inspection
python -m u1_critic_v63.run --help
```

Use `--input-root` pointing to the downloaded dataset's `images/`, and pass the local `inputs_image_only.jsonl` produced by `scripts/prepare_local_dataset.py`. Supply your model endpoint and API-key environment variable. Source-specific example scripts are retained inside each package. Configure the procedure flags from the corresponding settings file; the default CLI mode alone does not reconstruct every historical setting.

No API calls or training are necessary for the fixed-prediction analyses in `reproducibility/`. No matched-compute or causal component advantage is implied by the archived cost measurements.
