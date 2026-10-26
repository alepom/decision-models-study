# Pre-declared primary contrasts

Declared on 2026-10-24, before the new evaluations. Every contrast is a paired difference in macro-F1 on the same test
examples, with a 95% paired-bootstrap interval (2,000 replicates). For supervised systems the contrast is computed per
seed and summarised by the mean over seeds (with min-max across seeds); seed variability is reported separately.
Within each family, Holm adjustment is applied to the bootstrap p-values; intervals are reported unadjusted.
Everything not listed here is exploratory and labelled as such.

| ID | Family | Contrast | Datasets |
|---|---|---|---|
| P1 | descriptions (E03) | decision model with D minus same model with N, zero-shot, canonical order (OpenAI, Elephant-1, Mammoth-1) | banking77, massive_it, ottavo |
| P2 | label budget (E02/E05) | best decision model with D, zero-shot, minus each supervised system at k = 5, 20, full (corrected recipe) | banking77, massive_it, ottavo |
| P2N | label budget, zero labels (added 26/10/2026 after an external review, post hoc) | best decision model with N, zero-shot, minus each supervised system at k = 5, 20, full (corrected recipe); D on Banking77 and MASSIVE-it was written from labelled training data | banking77, massive_it, ottavo |
| P3 | progenitor (E04) | Elephant-1 (D) minus Qwen3.5-9B zero-shot (D), and Qwen3.5-9B minus Qwen3-8B zero-shot | ottavo, ottavo_adv |
| P4 | adaptation (E02) | Qwen3-8B fine-tuned (k = 5, 20, full) minus Qwen3-8B zero-shot (D) | ottavo, ottavo_adv |
| P5 | recipe (E02) | corrected recipe minus original recipe at k = 5 and 20 | Qwen3-0.6B and Qwen3-8B on ottavo |
| P6 | lexical stress (E07) | score on ottavo_adv minus score on ottavo for each system trained on ottavo | ottavo vs ottavo_adv (unpaired: different examples; bootstrap per set) |
| S1 | recipe sensitivity, full data (added 25/10/2026, commit ee3feacc, before the runs) | Qwen3-0.6B exploratory recipe (4 training seeds) minus protocol recipe; Mammoth-1 (D and N) minus the exploratory recipe | banking77, banking77_official |

If a system in a contrast cannot be run (resource, context limit, access), the contrast is reported as blocked, not dropped.
