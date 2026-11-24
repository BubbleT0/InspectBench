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

## Run a paper configuration

`run_paper_pipeline.py` selects the source snapshot and explicitly restores every recorded setting for the chosen pairing. It validates both CLI settings and derived routing settings before making model requests. This matters because the inspection package's standalone `main()` adds Selective defaults; use this launcher for Conservative and Full-Sched as well as Selective.

After downloading the benchmark and running `scripts/prepare_local_dataset.py`, start an OpenAI-compatible endpoint with the corresponding paper checkpoint, then run from the repository root:

```bash
python -m pip install 'Pillow>=10.0'
python pipelines/run_paper_pipeline.py \
  --configuration workflow_full_sched \
  --input-root data/InspectBench/images \
  --input-jsonl .local/benchmark/inputs_image_only.jsonl \
  --output-root .local/runs/workflow_full_sched \
  --base-url http://127.0.0.1:8000 \
  --model inspectbench-critic --all
```

Use any of the six configuration names from the table, without the directory or `.json` suffix. Add `--dry-run` to inspect the resolved settings without model requests. Omit `--all` to process one image, or use `--limit-images N`. Credentials are read from `CRITIC_API_KEY`, or the variable named by `--api-key-env`. The endpoint must serve the checkpoint specified by the chosen pairing; the launcher cannot identify weights behind an arbitrary endpoint.

The launcher writes `paper_configuration.json`, followed by the original harness outputs: `manifest.json`, `run_config.json`, `results.jsonl`, `errors.jsonl` and `traces.jsonl`. Final regions are `consolidated_result.edit_regions` in successful result rows. Coordinates are **xyxy in [0,1000]**. Internal image IDs are local hashes; join `manifest.json` source paths to the prepared input manifest to recover public `IMG_....` IDs. Keep failure records when evaluating new runs.

The full code for the coverage ledger, crop/zoom tools, prompt construction, verification, adjudication, consolidation, batching, retry and recovery logic is included in the two `src/` trees. [`PAPER_HARNESS.md`](PAPER_HARNESS.md) maps these components to source files.

The packages require Python 3.10 or later and Pillow. For example:

```bash
python -m pip install -e pipelines/inspection
python -m u1_critic_v63.run --help
```

Use `--input-root` pointing to the downloaded dataset's `images/`, and pass the local `inputs_image_only.jsonl` produced by `scripts/prepare_local_dataset.py`. Supply your model endpoint and API-key environment variable. Source-specific example scripts are retained inside each package. Configure the procedure flags from the corresponding settings file; the default CLI mode alone does not reconstruct every historical setting.

No API calls or training are necessary for the fixed-prediction analyses in `reproducibility/`. No matched-compute or causal component advantage is implied by the archived cost measurements.
