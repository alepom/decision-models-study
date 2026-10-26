#!/bin/bash
# Test ufficiali completi (analisi distinta). OpenAI solo D per budget; Baseten N e D (gratuito fino al 27/10/2026).
#   scripts/run_api_official.sh openai|elephant|mammoth
set -u
cd "$(dirname "$0")/.."
P=$1
PY=.venv/bin/python
eval "$(grep BASETEN_API_KEY ~/.bashrc)"; export BASETEN_API_KEY
LOG=logs/api_$P.log
# aspetta la fine del piano principale dello stesso provider (stesse quote di richieste)
until grep -q "PIANO COMPLETATO" $LOG 2>/dev/null; do sleep 30; done
run() { echo "== $(date '+%F %T') $*" >> $LOG; $PY -m dmstudy.decisions --provider $P "$@" >> $LOG 2>&1 || echo "!! FALLITO: $*" >> $LOG; }
MODES="D N"; [ "$P" = openai ] && MODES="D"
for t in banking77_official massive_it_official; do for l in $MODES; do run --task $t --labels $l; done; done
echo "== $(date '+%F %T') UFFICIALI COMPLETATI" >> $LOG
