# v1.0.0 — Decision model study

First public release accompanying the preprint *When Are Decision Models Worth It? An Empirical Comparison with Small
Task-Specific Classifiers* (Alessio Pomaro, 2026).

## Contents
- `dmstudy/`: API runner for OpenAI Decisions and Baseten System One (Elephant-1, Mammoth-1), supervised trainers (TF-IDF,
  SetFit, mmBERT, Qwen3 LoRA/QLoRA) with recipe selection by cross-validation, generalist zero-shot scorer, statistics and figures.
- Frozen protocol (`protocol.md`), pre-declared contrasts, split manifests with example IDs.
- Synthetic datasets: Ottavo Store (2,629 messages), lexical stress set (601), Ottavo-R robustness set (312), with generation prompts.
- Per-example predictions and metrics for 442 runs used in the paper (470 including superseded runs, flagged in `results/runs_index.csv`).
- Colab notebooks that run the same modules on a T4.

## Not included
Raw API request/response caches; model weights; real customer data (none were used).

## Reproduce the paper's tables and figures
`python -m dmstudy.analysis && python -m dmstudy.robustness && python -m dmstudy.extras` (no GPU, no API keys).
