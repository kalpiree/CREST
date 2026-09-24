# CREST

CREST screens recommendation records by their temporal promotional influence and constrains repeated item exposure through calibrated final selection. This repository contains the method, record-based baseline adaptations, attack construction, and experiment tools.

## Setup

Use Python 3.11 on Linux for model inference. The core method also runs on macOS.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[plot,download]'
python scripts/check_environment.py
python scripts/smoke_test.py
```

The smoke test uses synthetic rankings to check the complete screening, calibration, and selection path without downloading models or datasets.

For Qwen2.5 and Llama-3.1 inference, install the pinned CUDA 12.1 environment:

```bash
python -m pip install -r requirements/main.txt
python scripts/check_environment.py --profile main --require-cuda
```

Qwen3.5 uses a separate environment with `requirements/qwen35.txt`. Do not install both inference requirement files in the same environment. Its ranker uses text-only inputs with thinking disabled.

## Models and data

Authenticate with Hugging Face using an account with access to the Meta checkpoints, then download the pinned models:

```bash
hf auth login
python scripts/download_models.py --models qwen25 guard --output-dir models
python scripts/download_models.py --models llama31 --output-dir models
```

If the installed Hub version provides the older CLI, use `huggingface-cli login`. Models are checked against their immutable revisions and upstream file hashes. Credentials are read from the standard Hugging Face login or `HF_TOKEN`; they are never written into experiment configurations.

```bash
python scripts/download_datasets.py --raw-dir data/raw
python scripts/prepare_datasets.py --raw-dir data/raw --output-dir data/prepared
```

The datasets are Amazon Reviews 2018 **All_Beauty**, Steam version 2, and MovieLens-1M. Dataset licenses and access terms remain those of the original providers. Existing archives can be placed at the paths in `configs/datasets.json` instead of downloaded again. No datasets, model weights, or experiment results are bundled.

## Experiments

The main configuration uses 50 candidates, top-5 recommendations, 20 decisions, 50 calibration sequences, and ten test sequences grouped under seeds 11, 23, 37, 53, and 71. Screening uses `beta=0.05`, `penalty=0.05`, `beam_width=1`, and `max_depth=1`; calibration and promotion tolerances are both 0.10.

See [Reproduction](docs/reproduce.md) for preparation, execution, reporting, and figure commands, and [Method and protocols](docs/method.md) for the mapping between the paper and implementation.

Each run records its configuration, model identity, source hashes, input hashes, and per-sequence outputs. Interrupted runs resume validated completed stages. Changing code, data, or settings requires a new run directory.

## Results

Tables and figures are generated from saved rankings. Recall@5, NDCG@5, and FPR are reported as percentages; Gmax remains on the 0–1 scale. FPR is computed over distinct unmodified record identities within each sequence and then averaged. Method comparisons use the same returned sequences with valid clean references; return counts include all attempts.

Model generation, screening, calibration, and final selection are timed separately. A measurement from prepared inputs describes the final stage and does not measure the full defense pipeline.

## Repository

- `src/crest/`: record representation, screening, calibration, flow selection, attacks, baselines, and inference.
- `configs/`: dataset sources and experiment settings.
- `scripts/`: preparation, execution, analysis, and plotting commands.
- `requirements/`: separate inference environments.

Baseline adaptations and source references are documented in [Third-party methods](THIRD_PARTY.md).
