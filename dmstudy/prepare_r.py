"""E07: set di robustezza Ottavo-R (sintetico). Due fasi:

  python -m dmstudy.prepare_r blind   -> data/ottavo_r/raw/blind.jsonl (id + testo, mescolato, senza etichette)
  python -m dmstudy.prepare_r build   -> data/ottavo_r/test.jsonl (solo casi confermati da entrambi i validatori)

Regola di conferma: per i fenomeni a etichetta singola entrambi i validatori devono dare l'etichetta del generatore;
per multi_intent l'etichetta di ciascun validatore deve stare fra le due intenzioni dichiarate; per insufficient_info
e out_of_scope entrambi devono rispondere rispettivamente "ambiguous" e "out_of_scope". Le coppie typo_pair restano
solo se entrambi i membri sono confermati. Split dev/test 50/50 per fenomeno (le coppie restano insieme):
dev serve solo a scegliere le soglie di rinvio a revisione, test per la valutazione.
"""
import hashlib
import json
import random
import sys
from collections import defaultdict

from .common import DATA, read_jsonl, write_jsonl

RAW = DATA / "ottavo_r" / "raw"


def load():
    rows = []
    for f in sorted(RAW.glob("gen_part*.jsonl")):
        for r in read_jsonl(f):
            r["rid"] = hashlib.sha1(r["text"].encode()).hexdigest()[:10]
            r["source"] = f.stem
            rows.append(r)
    return rows


def blind():
    rows = load()
    random.Random(7).shuffle(rows)
    write_jsonl(RAW / "blind.jsonl", [{"id": r["rid"], "text": r["text"]} for r in rows])
    print(len(rows), "messaggi")


def build():
    rows = load()
    votes = [{v["id"]: v["label"] for v in read_jsonl(f)} for f in sorted(RAW.glob("val_*.jsonl"))]
    assert len(votes) >= 2, "servono due validatori"

    def ok(r):
        v = [vv.get(r["rid"]) for vv in votes]
        if r["phenomenon"] == "multi_intent":
            return all(x in r["gold_labels"] for x in v)
        return all(x == r["gold_label"] for x in v)
    kept = {r["rid"]: r for r in rows if ok(r)}
    # coppie: entrambi i membri o nessuno
    pairs = defaultdict(list)
    for r in rows:
        if r["phenomenon"] == "typo_pair":
            pairs[r["pair_id"]].append(r)
    for pid, members in pairs.items():
        if not all(m["rid"] in kept for m in members) or len(members) != 2:
            for m in members:
                kept.pop(m["rid"], None)
    rng = random.Random(42)
    out = []
    by_ph = defaultdict(list)
    for r in kept.values():
        by_ph[r["phenomenon"]].append(r)
    for ph, items in sorted(by_ph.items()):
        if ph == "typo_pair":
            pids = sorted({r["pair_id"] for r in items})
            rng.shuffle(pids)
            dev = set(pids[:len(pids) // 2])
            for r in items:
                r["split_role"] = "dev" if r["pair_id"] in dev else "test"
        else:
            items.sort(key=lambda r: r["rid"])
            rng.shuffle(items)
            for i, r in enumerate(items):
                r["split_role"] = "dev" if i < len(items) // 2 else "test"
        out += items
    out.sort(key=lambda r: (r["split_role"], r["phenomenon"], r["rid"]))
    final = [{"text": r["text"], "label": r["gold_label"], "phenomenon": r["phenomenon"], "split_role": r["split_role"],
              **{k: r[k] for k in ("gold_labels", "candidates", "pair_id", "variant") if k in r}} for r in out]
    write_jsonl(DATA / "ottavo_r" / "test.jsonl", final)
    stats = defaultdict(lambda: [0, 0])
    for r in rows:
        stats[r["phenomenon"]][0] += 1
    for r in out:
        stats[r["phenomenon"]][1] += 1
    agree = [sum(vv.get(r["rid"]) == r["gold_label"] or (r["phenomenon"] == "multi_intent" and vv.get(r["rid"]) in r["gold_labels"])
                 for r in rows) / len(rows) for vv in votes]
    json.dump({"generated_vs_kept": stats, "validator_agreement": agree, "n_final": len(final)},
              open(DATA / "ottavo_r" / "build_report.json", "w"), indent=1)
    print(json.dumps({"generated_vs_kept": stats, "validator_agreement": agree, "n_final": len(final)}))


if __name__ == "__main__":
    {"blind": blind, "build": build}[sys.argv[1]]()
