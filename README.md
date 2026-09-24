# CREST

CREST screens recommendation records by their temporal promotional influence and constrains repeated item exposure through calibrated final selection.

## Installation

Use Python 3.11 and a CUDA-capable GPU for model inference.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[plot,download]'
python -m pip install -r requirements/main.txt
python scripts/check_environment.py --profile main --require-cuda
```

For Qwen3.5, use a separate environment with `requirements/qwen35.txt` and run the environment check with `--profile qwen35`.

## Models and datasets

Log in with a Hugging Face account that has access to the Meta models, then download the models and data:

```bash
huggingface-cli login
python scripts/download_models.py --models qwen25 guard --output-dir models
python scripts/download_datasets.py --raw-dir data/raw
python scripts/prepare_datasets.py --raw-dir data/raw --output-dir data/prepared
```

Newer Hugging Face Hub versions use `hf auth login`. Additional backbones can be downloaded with `--models llama31` or `--models qwen35`.

Datasets: Amazon Reviews 2018 **All_Beauty**, Steam version 2, and MovieLens-1M. Sources and preprocessing settings are in `configs/datasets.json`.

## Experiments

Default settings are in `configs/paper.json`: 50 candidates, top-5 outputs, 20 decisions, 50 calibration sequences, and ten test sequences across five seeds. Screening uses `beta=0.05`, `penalty=0.05`, `beam_width=1`, and `max_depth=1`; `alpha=eta=0.10`.

Run commands from the repository root. Repeat a budget-limited command to resume; exit code 2 indicates a partial run. Use a new output directory when changing settings.

### RQ1: overall effectiveness

Prepare Beauty sequences, create an instruction-injection plan, and run the comparison:

```bash
python scripts/prepare_experiment.py sequences \
  --data-dir data/prepared --dataset amazon-2018-all-beauty \
  --output outputs/beauty
python scripts/prepare_experiment.py plan \
  --experiment outputs/beauty --family instruction_injection \
  --model models/Qwen/Qwen2.5-7B-Instruct \
  --guard-model models/meta-llama/Llama-Prompt-Guard-2-86M

CREST_PLAN=outputs/beauty/plans/Qwen--Qwen2.5-7B-Instruct/instruction_injection/plan.json
CREST_RUN=outputs/beauty/runs/injection
python scripts/run_rq1.py run --plan "$CREST_PLAN" --output "$CREST_RUN" \
  --invocation-budget-seconds 3600
python scripts/report_results.py --input "$CREST_RUN" \
  --output-dir outputs/tables/beauty-injection --bootstrap
```

Use `--dataset steam` or `--dataset movielens-1m` with a separate output directory for the other datasets. For history manipulation, set `--family interaction_history_manipulation` and update the plan and run paths.

For deceptive text rewriting, generate the payloads in the Qwen2.5 environment first:

```bash
python scripts/prepare_experiment.py rewrite-prepare \
  --experiment outputs/beauty --model models/Qwen/Qwen2.5-7B-Instruct
python scripts/prepare_experiment.py rewrite-run \
  --experiment outputs/beauty --model models/Qwen/Qwen2.5-7B-Instruct \
  --device cuda:0 --budget-seconds 3600
```

Once generation completes, create and run a plan with `--family deceptive_text_rewriting`.

For additional backbones, create a new plan from the same experiment directory using the Llama-3.1 or Qwen3.5 model path. For rewriting comparisons, also pass `--continuation-model models/Qwen/Qwen2.5-7B-Instruct` to keep the auxiliary model fixed.

### RQ2: attack robustness

Use the completed Beauty instruction-injection run:

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
python scripts/plot_results.py rq2 \
  --intensity outputs/rq2-intensity/plot-values.json \
  --recurrence outputs/rq2-recurrence/plot-values.json \
  --output outputs/figures/rq2_four_panel.pdf
```

Calibration stays fixed. Intensity varies the edited-record budget; recurrence varies the affected decisions while fixing three record identities and their payloads. Timing compares early, delayed, and dispersed schedules. Add `--layout dual` for the two-plot layout.

### RQ3: statistical promotion control

Use 20 independent blocks, each with 19 calibration and five test sequences:

```bash
python scripts/prepare_controls.py --experiment outputs/beauty \
  --output outputs/rq3/preparation
python scripts/run_controls.py --units outputs/rq3/preparation/units \
  --model models/Qwen/Qwen2.5-7B-Instruct --output outputs/rq3/runs \
  --budget-seconds 3600
python scripts/replay_controls.py --units outputs/rq3/preparation/units \
  --runs outputs/rq3/runs --output outputs/rq3/analysis
python scripts/plot_results.py rq3 --values outputs/rq3/analysis/plot_values.json \
  --output outputs/figures/rq3_three_panel.pdf
```

The alpha and eta comparisons reuse saved rankings. Confidence intervals resample complete calibration blocks. Recommendation metrics are computed over returned sequences.

### RQ4: ablation and inference timing

```bash
python scripts/ablation.py --plan "$CREST_PLAN" --run-dir "$CREST_RUN" \
  --output outputs/rq4-ablation
python scripts/benchmark_finalstage.py --plan "$CREST_PLAN" --run-dir "$CREST_RUN" \
  --output outputs/rq4-time-beauty --test-index 0 --repetitions 3
```

Ablation reuses saved rankings and recalibrates Without Screening separately. Timing measures generation from prepared inputs and final selection, excluding screening, detector execution, calibration, and model loading. Repeat timing with the completed Steam plan and run directory, setting `--output outputs/rq4-time-steam`, then generate the table:

```bash
python scripts/report_timing.py --beauty outputs/rq4-time-beauty/summary.json \
  --steam outputs/rq4-time-steam/summary.json --output outputs/tables/rq4-time.tex
```

Reports contain LaTeX, JSON, and CSV outputs. Recall@5, NDCG@5, and FPR are percentages; Gmax uses the 0–1 scale. RQ1 metrics are averaged within seeds and then across seeds.

## Code structure

- `src/crest/`: screening, calibration, final selection, attacks, and baselines.
- `configs/`: dataset and experiment settings.
- `scripts/`: data preparation, experiments, reporting, and plotting.
- `requirements/`: model inference dependencies.

## References

- [Spotlighting](https://arxiv.org/abs/2403.14720): caret datamarking of record contents.
- [Prompt Guard 2](https://huggingface.co/meta-llama/Llama-Prompt-Guard-2-86M): record-level detection with the 86M classifier and Meta's [label mapping](https://github.com/meta-llama/llama-cookbook/blob/3c106f3e6ee79d6df51ae706bc7ef2d734ec3ded/getting-started/responsible_ai/prompt_guard/inference.py).
- [TextSimu and RewriteDetection](https://arxiv.org/abs/2409.11690): item-text rewriting with Qwen2.5 and record-level detection.
- [RETURN](https://arxiv.org/abs/2504.02458), [source implementation](https://github.com/Biglemon-Ning/RETURN/tree/f56bc959890a39a8eae83097d041e33026c2495e): collaborative-history support scoring with deletion-only record filtering.
- [Transformers](https://github.com/huggingface/transformers): model inference and generation.

Models and datasets retain their publishers' licenses and access terms.
