#!/bin/bash
# Decision model zero-shot (D) sul set di robustezza Ottavo-R.
cd "$(dirname "$0")/.."
eval "$(grep BASETEN_API_KEY ~/.bashrc)"; export BASETEN_API_KEY
for P in openai elephant mammoth; do
  (echo "== $(date '+%F %T') ottavo_r D" >> logs/api_$P.log; .venv/bin/python -m dmstudy.decisions --provider $P --task ottavo_r --labels D >> logs/api_$P.log 2>&1 && echo "== R COMPLETATO" >> logs/api_$P.log || echo "!! FALLITO: ottavo_r" >> logs/api_$P.log) &
done
wait
