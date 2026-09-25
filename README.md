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

## Quick start

Run an instruction-injection comparison on Amazon Beauty:

```bash
python scripts/prepare_experiment.py sequences \
  --data-dir data/prepared --dataset amazon-2018-all-beauty --output outputs/beauty
python scripts/prepare_experiment.py plan \
  --experiment outputs/beauty --family instruction_injection \
  --model models/Qwen/Qwen2.5-7B-Instruct \
  --guard-model models/meta-llama/Llama-Prompt-Guard-2-86M

CREST_PLAN=outputs/beauty/plans/Qwen--Qwen2.5-7B-Instruct/instruction_injection/plan.json
CREST_RUN=outputs/beauty/runs/injection
python scripts/run_rq1.py run --plan "$CREST_PLAN" --output "$CREST_RUN" \
  --invocation-budget-seconds 3600 && \
python scripts/report_results.py --input "$CREST_RUN" --output-dir outputs/tables/beauty
```

Repeat the run command if it reaches the time budget (exit code 2); reporting runs after completion. Settings are in `configs/paper.json`. Use `--help` for script options.

## Scripts

| Task | Script |
| --- | --- |
| Sequence and attack preparation | `scripts/prepare_experiment.py` |
| Baseline comparison | `scripts/run_rq1.py` |
| Attack robustness | `scripts/run_robustness.py` |
| Statistical control | `scripts/prepare_controls.py`, `scripts/run_controls.py`, `scripts/replay_controls.py` |
| Component ablation | `scripts/ablation.py` |
| Final-stage timing | `scripts/benchmark_finalstage.py`, `scripts/report_timing.py` |
| Tables and plots | `scripts/report_results.py`, `scripts/plot_results.py` |

## References

Baseline and attack implementations build on [Spotlighting](https://arxiv.org/abs/2403.14720), [Prompt Guard 2](https://huggingface.co/meta-llama/Llama-Prompt-Guard-2-86M), [TextSimu and RewriteDetection](https://arxiv.org/abs/2409.11690), and [RETURN](https://github.com/Biglemon-Ning/RETURN/tree/f56bc959890a39a8eae83097d041e33026c2495e). Model inference uses [Transformers](https://github.com/huggingface/transformers).

Models and datasets retain their publishers' licenses and access terms.
