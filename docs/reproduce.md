# Reproduction

Run these commands from the repository root after completing the README setup. Generated data, plans and outputs stay outside the source directories. No command requires the original compute cluster or scheduler.

## RQ1: overall effectiveness

Prepare the chronological sequences and fixed instruction/history attacks:

```bash
python scripts/prepare_experiment.py sequences \
  --data-dir data/prepared --dataset amazon-2018-all-beauty \
  --output outputs/beauty
```

Use `steam` or `movielens-1m` for the other datasets and a distinct output directory. The default sequence mode preserves the original T100-to-T20 source projection for Beauty/Steam and uses native T20 for MovieLens. `--sequence-mode native` is available for a separately identified experiment, not an identical replay of the projected cohort.

Create and run a single comparison:

```bash
python scripts/prepare_experiment.py plan \
  --experiment outputs/beauty --family instruction_injection \
  --model models/Qwen/Qwen2.5-7B-Instruct \
  --guard-model models/meta-llama/Llama-Prompt-Guard-2-86M

CREST_PLAN=outputs/beauty/plans/Qwen--Qwen2.5-7B-Instruct/instruction_injection/plan.json
CREST_RUN=outputs/beauty/runs/injection
python scripts/run_rq1.py run --plan "$CREST_PLAN" --output "$CREST_RUN" \
  --invocation-budget-seconds 3600
```

A budget-limited invocation returns status `partial` and exit code 2. Repeat the same command to resume; completed stages and exact model queries are reused. A completed invocation writes `report.json` under `CREST_RUN/cells/<unit-id>/`. Failures remain recorded rather than silently dropped.

For history manipulation, change `--family` to `interaction_history_manipulation`, update the plan path and use a separate run directory. The plan builder constructs the train-only RETURN graph automatically.

Rewriting needs generated payloads before evaluation:

```bash
python scripts/prepare_experiment.py rewrite-prepare \
  --experiment outputs/beauty --model models/Qwen/Qwen2.5-7B-Instruct
python scripts/prepare_experiment.py rewrite-run \
  --experiment outputs/beauty --model models/Qwen/Qwen2.5-7B-Instruct \
  --device cuda:0 --budget-seconds 3600
python scripts/prepare_experiment.py plan \
  --experiment outputs/beauty --family deceptive_text_rewriting \
  --model models/Qwen/Qwen2.5-7B-Instruct \
  --guard-model models/meta-llama/Llama-Prompt-Guard-2-86M
```

Resume `rewrite-run` until generation completes, then run its plan with `run_rq1.py`. Missing payloads stop plan creation; no surrogate descriptions or successful-attempt filtering are used.

Generate tables from a completed cell:

```bash
python scripts/report_results.py --input "$CREST_RUN" \
  --output-dir outputs/tables/beauty-injection --bootstrap
```

This produces LaTeX, JSON and CSV from per-sequence outputs. The CSV is a generated artifact, not a bundled input. The optional intervals enumerate all 3,125 resamples of five seed groups; they describe stability conditional on the fixed calibration.

## Additional backbones

Prepare a new plan from the same experiment directory using the Llama-3.1 or Qwen3.5 model path. This reuses the candidate sets, users, targets and generated attacks, while recalibrating the new backbone. Use the separate Qwen3.5 environment described in the README.

For rewriting with either additional backbone, pass `--continuation-model models/Qwen/Qwen2.5-7B-Instruct` to keep RewriteDetection's auxiliary continuation model fixed. Payload generation always runs in the Qwen2.5 environment.

## RQ2: attack robustness

Start with the completed Beauty injection run above. Each single-cell run contains one directory under `cells`; resolve that path for subsequent commands:

```bash
CREST_CELL=$(find "$CREST_RUN/cells" -mindepth 1 -maxdepth 1 -type d)
python scripts/run_robustness.py --plan "$CREST_PLAN" --primary-cell "$CREST_CELL" \
  --mode intensity --source-descriptors outputs/beauty/sequences/descriptors.json \
  --source-attacks outputs/beauty/attack-settings.json --output outputs/rq2-intensity
python scripts/run_robustness.py --plan "$CREST_PLAN" --primary-cell "$CREST_CELL" \
  --mode recurrence --source-attacks outputs/beauty/attack-settings.json \
  --output outputs/rq2-recurrence
python scripts/run_robustness.py --plan "$CREST_PLAN" --primary-cell "$CREST_CELL" \
  --mode timing --source-attacks outputs/beauty/attack-settings.json \
  --output outputs/rq2-timing
```

The primary calibration and screening parameters stay fixed. Intensity varies the edited-record budget. Recurrence holds three identities and their payloads fixed while varying their affected decisions. Timing moves the same ten affected decisions to early, delayed or dispersed schedules; dispersed timing reuses the 50% recurrence inputs. Recurrence uses a fixed shared-identity cohort and is not assumed identical to RQ1 at its 50% point.

```bash
python scripts/plot_results.py rq2 \
  --intensity outputs/rq2-intensity/plot-values.json \
  --recurrence outputs/rq2-recurrence/plot-values.json \
  --output outputs/figures/rq2_four_panel.pdf
```

Pass `--layout dual` for two plots with left/right metric axes; Gmax uses each method's usual line style and NDCG uses dashed lines. The four-plot layout is the main-paper version. Timing conditions contain the same per-sequence `report.json` format used by `report_results.py`.

## RQ3: statistical promotion control

The independent-block protocol and complete commands are in [Statistical control](statistical-control.md). It uses 20 blocks with 19 calibration and five test sequences each. After generation, the alpha/eta comparisons only rerun calibration and final selection on saved rankings.

```bash
python scripts/plot_results.py rq3 --values outputs/rq3/analysis/plot_values.json \
  --output outputs/figures/rq3_three_panel.pdf
```

The figure uses computed whole-block bootstrap intervals. Infeasible settings have undefined returned-output utility and are not plotted as zero utility. Return and violation statistics remain available in the generated analysis report.

## RQ4: component ablation and timing

```bash
python scripts/ablation.py --plan "$CREST_PLAN" --run-dir "$CREST_RUN" \
  --output outputs/rq4-ablation
python scripts/benchmark_finalstage.py --plan "$CREST_PLAN" --run-dir "$CREST_RUN" \
  --output outputs/rq4-time-beauty --test-index 0 --repetitions 3
```

Ablation reuses saved rankings. Without Screening receives its own calibration from unfiltered rankings; Screening-Only omits final selection. Run the timing command separately for a completed Steam plan. The timer covers generation from prepared inputs and final selection, not screening or the full defense pipeline. Its output includes the measured generation and selection components, model/runtime metadata and cache policy.

After timing both datasets, generate the LaTeX timing table:

```bash
python scripts/report_timing.py --beauty outputs/rq4-time-beauty/summary.json \
  --steam outputs/rq4-time-steam/summary.json --output outputs/tables/rq4-time.tex
```

## Reproducibility scope

The commands reproduce the implemented protocols and compute results from model outputs. Exact numerical reproduction also depends on the original input artifacts, model/runtime and hardware. The repository contains no hardcoded paper tables, precomputed rankings, manually revised aggregates or illustrative sensitivity results. Saved outputs cannot be replaced by aggregate values when calculating paired comparisons or case studies.
