# Statistical promotion control

RQ3 uses 20 independent calibration blocks. Each block contains 19 calibration sequences and five test sequences drawn from the same held-out Beauty pool. All sequences use 20 decisions, 50 candidates and top-5 outputs. The source construction and its fixed 100-to-20-decision projection are shared between calibration and test roles. Sampling does not discard draws based on model outcomes.

After preparing the Beauty experiment, create the independent blocks:

```bash
python scripts/prepare_controls.py \
  --experiment outputs/beauty \
  --output outputs/rq3/preparation
```

Run the frozen Qwen2.5 recommender and screening once per block:

```bash
python scripts/run_controls.py \
  --units outputs/rq3/preparation/units \
  --model models/Qwen/Qwen2.5-7B-Instruct \
  --output outputs/rq3/runs \
  --budget-seconds 3600
```

The command resumes completed stages and exact ranking queries. Exit code 2 means the invocation budget was reached; rerun the same command. `--block 0` selects a single block. A run directory is locked to one worker and cannot be reused with a different model/runtime or method configuration.

Recompute calibration and final selection across the alpha and eta values without further model calls:

```bash
python scripts/replay_controls.py \
  --units outputs/rq3/preparation/units \
  --runs outputs/rq3/runs \
  --output outputs/rq3/analysis
```

The replay verifies the saved unit, run and stage hashes. It writes `traces.json`, `controls.json` and `plot_values.json`. Archived traces can subsequently be replayed with `--traces outputs/rq3/analysis/traces.json` instead of `--units` and `--runs`.

The alpha comparison fixes eta at 0.10; the eta comparison fixes alpha at 0.10. Their default point is computed once and shared. Confidence intervals use 10,000 whole-block bootstrap resamples, retaining each block's five test sequences together. Recommendation metrics and mean promotion are conditional on return. If no sequence returns, these metrics and their intervals are undefined. The joint event of returning a violating sequence is false for a sequence that is infeasible; its conditional violation rate remains undefined.

Rates, Recall and NDCG are percentages in the plotting export. `mean_gmax` remains on the unit interval. The model inputs, rather than manuscript table values, determine every estimate. Bootstrap intervals describe variation among the sampled blocks; a degenerate zero-event interval is not a zero-risk guarantee.
