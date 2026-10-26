# Reproduction guide

Three levels, from cheapest to most expensive. All commands are run from the repository root.

## 0. Environment

Python 3.13 was used (WSL2 Ubuntu, NVIDIA driver with CUDA 12.8).

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.lock --extra-index-url https://download.pytorch.org/whl/cu128
```

Level 1 needs only `numpy`, `scikit-learn` and `matplotlib` (no GPU, no API keys).

```bash
.venv/bin/python -m pytest -q tests
```

## 1. Regenerate every table and figure from the saved predictions (no GPU, no API)

`results/predictions/*.jsonl` contain one line per test example (gold label, predicted label, probability per class,
latency, status); `results/metrics/*.json` contain the configuration and provenance of each run.

```bash
.venv/bin/python -m dmstudy.recompute      # metrics from predictions
.venv/bin/python -m dmstudy.analysis       # paired bootstrap contrasts, selective prediction, figures, LaTeX tables
.venv/bin/python -m dmstudy.robustness     # Ottavo-R by phenomenon
.venv/bin/python -m dmstudy.extras         # error analysis, per-class and per-seed tables, reliability, quality-latency, costs
.venv/bin/python -m dmstudy.audit          # audit of the earlier exploratory study
.venv/bin/python -m dmstudy.status         # measured compute plan
```

Outputs: `reports/contrasts.csv`, `reports/selective.csv`, `reports/robustness.csv`, `results/runs_index.csv`,
`paper/figures/*.pdf`, `paper/generated/*.tex`. To rebuild the PDF with [tectonic](https://tectonic-typesetting.github.io/):

```bash
cd paper && tectonic -X compile main.tex
```

## 2. Re-run one local experiment

Splits and few-shot subsets are frozen in `data/manifests/` (seeds 17, 42, 123; budgets nested 5 ⊂ 10 ⊂ 20 ⊂ 50).

```bash
# TF-IDF + logistic regression, 5 examples per class, seed 42 (CPU, seconds)
.venv/bin/python -m dmstudy.classic --method tfidf --task ottavo --budget 5 --seed 42 --also ottavo_adv ottavo_r
# SetFit (GPU recommended, ~5-30 min)
.venv/bin/python -m dmstudy.classic --method setfit --task ottavo --budget 20 --seed 42 --also ottavo_adv ottavo_r
# recipe selection by 3-fold CV inside the budget, then the final run with the selected recipe
.venv/bin/python -m dmstudy.train_hf --task ottavo --model qwen0.6b --budget 5 --seed 17 --cv 3 --lrs 5e-5 2e-4 --checkpoints 5 10 20
.venv/bin/python -m dmstudy.train_hf --task ottavo --model qwen0.6b --budget 5 --seed 42 --lr 2e-4 --epochs 10 --also ottavo_adv ottavo_r
```

Each run writes a new `results/metrics/<run_id>.json` and `results/predictions/<run_id>.jsonl`; the analysis uses the most
recent run for each configuration.

## 3. Re-run the full GPU and API experiments

| Part | Command | Hardware / access | Time (measured, see `reports/compute_plan.csv`) | Cost |
|---|---|---|---|---|
| Local queue (TF-IDF, SetFit, mmBERT, Qwen3-0.6B; 3 seeds; 3 datasets) | `.venv/bin/python -m dmstudy.queue --plan local` | 6 GB GPU | many hours (resumable) | electricity |
| Colab queue (Qwen3-8B/Qwen3.5-9B zero-shot, Qwen3-8B QLoRA, Qwen3-1.7B) | open `notebooks/colab_runner.ipynb` in Colab with a T4, *Run all* | 16 GB GPU | several sessions (resumable via Google Drive) | free tier |
| Decision APIs | `scripts/run_api_plan.sh openai` (and `elephant`, `mammoth`), `scripts/run_api_official.sh`, `scripts/run_api_r.sh` | `OPENAI_API_KEY`, `BASETEN_API_KEY` in the environment | ~1 h | OpenAI ≈ 8 USD at Oct-2026 beta prices; Baseten was promotional |

API outputs can change over time (beta endpoints, `latest` aliases): re-running them measures the current endpoint, not the
one evaluated in the paper. The paper's numbers are reproducible from the saved predictions (level 1).

Not distributed: the cache of raw API request/response bodies (`results/cache/`, contains message texts and full responses)
and model weights (downloaded from Hugging Face at run time).
