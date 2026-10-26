"""E07: analisi del set di robustezza Ottavo-R per fenomeno. Usa solo le predizioni salvate.

  python -m dmstudy.robustness   -> reports/robustness.csv, paper/generated/table_robustness.tex

- closed-set (negation, intent_correction, lexical_distractor, colloquial_short, typo_pair): accuratezza sullo split test;
- multi_intent: predizione fra le due intenzioni dichiarate (any) e uguale alla primaria della regola di priorità;
- typo_pair: coerenza fra versione pulita e perturbata della stessa coppia;
- insufficient_info / out_of_scope: un classificatore a 24 classi non può astenersi; si valuta il rinvio a revisione
  con soglia sulla probabilità massima scelta sullo split dev (massimo di TPR(rinvio | ambiguo o fuori tassonomia)
  - FPR(rinvio | casi in tassonomia)), applicata allo split test.
"""
import csv

import numpy as np

from .analysis import SYSNAME, load_runs, pick
from .common import ROOT, test_set

CLOSED = ["negation", "intent_correction", "lexical_distractor", "colloquial_short", "typo_pair"]


def analyse(run_ids):
    meta = {r["id"]: r for r in test_set("ottavo_r")}
    from .analysis import preds
    res = []
    for rid in run_ids:
        P = preds(rid)
        rows = [(meta[i], p) for i, p in P.items()]
        out = {}
        for ph in CLOSED:
            sel = [(m, p) for m, p in rows if m["phenomenon"] == ph and m["split_role"] == "test"]
            if sel:
                out[f"acc_{ph}"] = np.mean([p["predicted_label"] == m["label"] for m, p in sel])
        mi = [(m, p) for m, p in rows if m["phenomenon"] == "multi_intent" and m["split_role"] == "test"]
        if mi:
            out["multi_any"] = np.mean([p["predicted_label"] in m["gold_labels"] for m, p in mi])
            out["multi_primary"] = np.mean([p["predicted_label"] == m["label"] for m, p in mi])
        pairs = {}
        for m, p in rows:
            if m["phenomenon"] == "typo_pair" and m["split_role"] == "test":
                pairs.setdefault(m["pair_id"], {})[m["variant"]] = p["predicted_label"]
        full = [v for v in pairs.values() if len(v) == 2]
        if full:
            out["typo_consistency"] = np.mean([v["clean"] == v["perturbed"] for v in full])
        # rinvio a revisione
        conf = lambda p: max(p["probs"]) if p.get("probs") else 0.0
        neg = lambda m: m["label"] in ("ambiguous", "out_of_scope")
        dev = [(conf(p), neg(m)) for m, p in rows if m["split_role"] == "dev"]
        tst = [(conf(p), neg(m), m["label"]) for m, p in rows if m["split_role"] == "test"]
        if dev and any(n for _, n in dev) and any(not n for _, n in dev):
            best, bt = -2, 0.0
            for t in sorted({c for c, _ in dev}) + [1.01]:
                tpr = np.mean([c < t for c, n in dev if n]); fpr = np.mean([c < t for c, n in dev if not n])
                if tpr - fpr > best:
                    best, bt = tpr - fpr, t
            out["threshold"] = bt
            out["review_ambiguous"] = np.mean([c < bt for c, n, l in tst if l == "ambiguous"])
            out["review_oos"] = np.mean([c < bt for c, n, l in tst if l == "out_of_scope"])
            out["review_inscope"] = np.mean([c < bt for c, n, l in tst if not n])
        res.append(out)
    return {k: float(np.mean([r[k] for r in res if k in r])) for k in set().union(*res)} if res else {}


def main():
    runs = load_runs()
    rows = []
    api = dict(family="decision", perm=0, shots="0", subset="full", repeat=0)
    for s in ["mammoth", "elephant", "openai"]:
        r = pick(runs, task="ottavo_r", system=s, labels="D", **api)
        if r:
            rows.append(dict(system=SYSNAME[s], setting="zero-shot, D", runs=len(r), **analyse([x["run_id"] for x in r])))
    for s in ["qwen3-8b", "qwen3.5-9b"]:
        r = pick(runs, task="ottavo_r", system=s, labels="D", perm=0, subset="full")
        if r:
            rows.append(dict(system=SYSNAME[s], setting="zero-shot, D", runs=len(r), **analyse([x["run_id"] for x in r])))
    for s in ["tfidf", "setfit", "mmbert", "qwen0.6b", "qwen1.7b", "qwen8b"]:
        for b in ["5", "20", "full"]:
            r = pick(runs, task="ottavo_r", system=s, budget=b, recipe="corrected")
            if r:
                rows.append(dict(system=SYSNAME[s], setting=f"trained on Ottavo, k={b}", runs=len(r),
                                 **analyse([x["run_id"] for x in r])))
    if not rows:
        print("nessun run su ottavo_r"); return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    keys = keys[:3] + sorted(keys[3:])  # ordine stabile fra esecuzioni
    with open(ROOT / "reports" / "robustness.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    cols = [("acc_negation", "Neg."), ("acc_intent_correction", "Corr."), ("acc_lexical_distractor", "Distr."),
            ("acc_colloquial_short", "Short"), ("acc_typo_pair", "Typos"), ("typo_consistency", "Cons."),
            ("multi_any", "Multi"), ("review_ambiguous", "Amb.$\\to$rev."), ("review_oos", "OOS$\\to$rev."),
            ("review_inscope", "In-scope$\\to$rev.")]
    lines = [r"\begin{tabular}{ll" + "r" * len(cols) + "}", r"\toprule",
             "System & Setting & " + " & ".join(c for _, c in cols) + r" \\", r"\midrule"]
    for r in [r for r in rows if "k=5" not in r["setting"] and "k=20" not in r["setting"]]:  # tabella del paper: zero-shot e full
        lines.append(f"{r['system']} & {r['setting'].replace('trained on Ottavo, k=full', 'all Ottavo data')} & " + " & ".join(
            f"{r[k]:.2f}" if k in r else "--" for k, _ in cols) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (ROOT / "paper" / "generated").mkdir(parents=True, exist_ok=True)
    (ROOT / "paper" / "generated" / "table_robustness.tex").write_text("\n".join(lines))
    print(f"{len(rows)} righe")


if __name__ == "__main__":
    main()
