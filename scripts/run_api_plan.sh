#!/bin/bash
# Piano dei run API (E03/E05/E07/E09) in ordine di priorità. Riprendibile: le chiamate già fatte sono in cache.
#   scripts/run_api_plan.sh openai|elephant|mammoth
# Log: logs/api_<provider>.log
set -u
cd "$(dirname "$0")/.."
P=$1
PY=.venv/bin/python
eval "$(grep BASETEN_API_KEY ~/.bashrc)"; export BASETEN_API_KEY
LOG=logs/api_$P.log
run() { echo "== $(date '+%F %T') $*" >> $LOG; $PY -m dmstudy.decisions --provider $P "$@" >> $LOG 2>&1 || echo "!! FALLITO: $*" >> $LOG; }

# 1. zero-shot N e D sul test completo (contrasto P1) + stress lessicale
for t in ottavo banking77 massive_it; do for l in D N; do run --task $t --labels $l; done; done
run --task ottavo_adv --labels D
run --task ottavo_adv --labels N
# 2. stabilità dell'endpoint: ripetizione sul campione API
for t in ottavo banking77 massive_it; do run --task $t --labels D --subset api --repeat 1; done
# 3. sensibilità all'ordine delle opzioni (permutazioni 1-3) sul campione API
for t in ottavo banking77 massive_it; do for p in 1 2 3; do run --task $t --labels D --subset api --perm $p; done; done
# 4. esempi nel prompt con D fissato, stessi esempi di E02 (seed 42), sul campione API
for t in ottavo banking77 massive_it ottavo_adv; do for s in 5 20 r10; do run --task $t --labels D --subset api --shots $s; done; done
echo "== $(date '+%F %T') PIANO COMPLETATO" >> $LOG
