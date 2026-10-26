# When Are Decision Models Worth It?

**An empirical comparison of zero-shot decision models with small task-specific classifiers.**
Alessio Pomaro, 2026. Preprint, not peer reviewed.

> Paper: [doi:10.5281/zenodo.23652645](https://doi.org/10.5281/zenodo.23652645) ·
> Archived code and predictions: [doi:10.5281/zenodo.23652764](https://doi.org/10.5281/zenodo.23652764) · How to cite: `CITATION.cff`
> This repository contains the code, frozen protocol, data manifests, synthetic datasets and **per-example predictions** needed
> to reproduce every number in the paper.

## Question

New "decision models" (OpenAI Decisions, Base Labs' Elephant-1 and Mammoth-1) return a probability for each option of a closed
question without generating text, and need no task-specific training. Can a small open model trained on the task, possibly run
locally, do the same job? We compare three decision APIs with TF-IDF, SetFit, mmBERT and Qwen3 0.6B/1.7B/8B classifiers across
label budgets (5 and 20 examples per class, full training set), category descriptions, a lexical stress test and a phenomenon-level
robustness set, on Banking77 (English), MASSIVE it-IT (Italian) and a synthetic Italian customer-service task (Ottavo Store).

## Main results (macro-F1, historical test samples)

| | Banking77 | MASSIVE-it | Ottavo (synthetic) | Ottavo lexical stress |
|---|---:|---:|---:|---:|
| Best decision model, zero-shot, label names only (no task labels) | 0.934 (Mammoth-1) | 0.858 (Mammoth-1) | 0.975 (Mammoth-1) | 0.968 (Mammoth-1) |
| Best decision model, zero-shot, with descriptions¹ | 0.925 (Mammoth-1) | 0.864 (Elephant-1) | 0.991 (Mammoth-1) | 0.998 (Elephant-1) |
| Best compact classifier, 5 examples/class | 0.717 (SetFit) | 0.615 (SetFit) | 0.766 (SetFit) | 0.718 (SetFit) |
| Best compact classifier, 20 examples/class | 0.846 (Qwen3-0.6B) | 0.688 (SetFit) | 0.907 (SetFit) | 0.928 (Qwen3-8B) |
| Best compact classifier, full training set | 0.910 (Qwen3-0.6B) | 0.803 (mmBERT) | 0.975 (Qwen3-8B) | 0.959 (Qwen3-8B) |
| Same, exploratory recipe (sensitivity, 4 seeds) | 0.936 (Qwen3-0.6B) | – | – | – |

¹ On Banking77 and MASSIVE-it the descriptions were written from labelled training examples, so this row uses task labels there.

Supervised values are means over three seeds. Paired bootstrap intervals and all contrasts are in `reports/contrasts.csv`.
In short: with label names only, decision models are 21–24 points ahead of the best compact classifier trained on 5 examples per class.
With the full training data and the protocol recipes the gap is 2.4 points on Banking77, 5.5 on MASSIVE-it and not resolved on Ottavo.
On Banking77 the exploratory recipe raises Qwen3-0.6B to 0.936, and its difference from Mammoth-1 is no longer resolved: with many labels
the result depends on the training recipe. The effect of category descriptions varies with model and dataset. Elephant-1 is much closer
to its own base model, Qwen3.5-9B, used zero-shot, than to an older generalist.

## Main limitations

No real customer messages and no human gold labels; the Ottavo data are synthetic (written and validated by Claude models); public
benchmarks may be contaminated; endpoints were in beta in October 2026; test samples of ~1,000 messages do not resolve differences of
about one point. See the paper's Limitations section and `reports/limitations.md`.

## Repository layout

| Path | Content |
|---|---|
| `dmstudy/` | all experiment and analysis code (API runner, trainers, zero-shot scorer, statistics, figures) |
| `protocol.md`, `reports/contrasts_primary.md` | frozen protocol and pre-declared contrasts |
| `data/manifests/` | frozen splits, few-shot subsets, API subsets, option permutations (by example ID) |
| `data/descriptions/`, `data/*/labels.json` | category descriptions (condition D) |
| `data/ottavo*/` | synthetic Ottavo datasets with generation prompts (`data/ottavo/generation/`) |
| `data/banking77*`, `data/massive_it*` | derived copies of the public datasets (CC BY 4.0) |
| `results/metrics/`, `results/predictions/` | one JSON (config, provenance, metrics) and one JSONL (per-example predictions) per run |
| `results/runs_index.csv` | all runs; `used=True` marks the run used for each configuration |
| `reports/` | audit, contrasts, robustness, selective prediction, compute plan, claims-evidence map |
| `paper/` | LaTeX sources, generated tables and figures |
| `notebooks/` | Colab notebooks that run the same modules on a T4 |

## Install and reproduce

See [`README_reproduction.md`](README_reproduction.md). Minimal example (no GPU, no API key), regenerating all tables and figures from
the saved predictions:

```bash
python3 -m venv .venv && .venv/bin/pip install numpy scikit-learn matplotlib
.venv/bin/python -m dmstudy.analysis && .venv/bin/python -m dmstudy.robustness && .venv/bin/python -m dmstudy.extras
```

## Hardware and cost

Supervised models were trained on a free Google Colab Tesla T4 (16 GB) and an RTX 4050 laptop GPU (6 GB); TF-IDF on CPU. OpenAI API
calls cost about 8 USD at October-2026 beta prices; Baseten endpoints were free during a promotional period (no production price).

## AI assistance

The experiments were designed and run, and the paper drafted, by an AI coding agent (Claude Code) under the author's direction; the
synthetic data were written and validated by Claude models. Details in the paper.

## Citation and licences

Please cite the paper (see `CITATION.cff`, which GitHub shows as *Cite this repository*). Code: MIT. Paper, documentation, synthetic
data and predictions: CC BY 4.0. Third-party datasets keep their licences: see `LICENSES.md` and `THIRD_PARTY_NOTICES.md`.
