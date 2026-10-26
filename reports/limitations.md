# Limitations (summary; the paper's Section 7 is the reference)

1. No real customer messages and no human gold labels (E06 blocked). Ottavo, its lexical stress set and Ottavo-R are synthetic,
   written and validated by Claude models; scores near 0.99 there are upper bounds with a ceiling effect.
2. Possible contamination of Banking77 and MASSIVE for all models; Base Labs report training on public decision tasks.
3. Endpoints in beta / `latest` aliases (October 2026); one cross-day re-run of OpenAI differed by 0.012 macro-F1.
4. Test samples of ~1,000 messages (601 and 312 for the Ottavo stress sets) do not resolve differences of about one point.
5. Hyperparameters selected on seed 17 and frozen; small grid; selected duration at the grid edge for mmBERT and Qwen3-1.7B at
   5/class; the full-data recipe proved suboptimal for Qwen3-0.6B on Banking77 (exploratory recipe +0.026,
   difference from Mammoth-1 no longer resolved) and was not compared elsewhere.
6. Decoder classifiers use a fresh linear head that does not read descriptions; an LM-head variant was not tested.
7. Hardware heterogeneity (Colab T4, RTX 4050 laptop); library versions pinned only after the first Colab jobs.
8. Generalist zero-shot controls only on the 24-class Ottavo task (single-letter scoring).
9. Not tested: Jev, self-hosted decision models, distillation, real-traffic and labelling costs.
- Descriptions (D) for Banking77 and MASSIVE-it were written by agents that read labelled training examples, so D is not a
  zero-label condition there; the zero-label budget comparison (N, family P2N) was added post hoc after an external review.
