"""E08-E10: statistica, calibrazione, rischio-copertura, tabelle e figure del paper, tutto dalle predizioni salvate.

  python -m dmstudy.analysis

Scrive results/runs_index.csv, reports/contrasts.csv, reports/selective.csv, paper/generated/*.tex, paper/figures/*.pdf.
Nessuna inferenza, nessuna API: riproducibile da chiunque abbia results/predictions.
"""
import csv
import json
from collections import defaultdict

import numpy as np

from .common import RESULTS, ROOT, labels, read_jsonl
from .metrics import ece

B = 2000
BOOT_SEED = 20261024
GEN = ROOT / "paper" / "generated"
FIG = ROOT / "paper" / "figures"
SYSNAME = {"openai": "OpenAI Decisions", "elephant": "Elephant-1", "mammoth": "Mammoth-1", "tfidf": "TF-IDF + LR",
           "setfit": "SetFit (mE5-base)", "mmbert": "mmBERT-base", "qwen0.6b": "Qwen3-0.6B LoRA",
           "qwen1.7b": "Qwen3-1.7B LoRA", "qwen8b": "Qwen3-8B QLoRA", "qwen3-8b": "Qwen3-8B zero-shot",
           "qwen3.5-9b": "Qwen3.5-9B zero-shot"}
DATASETS = {"banking77": "Banking77", "massive_it": "MASSIVE-it", "ottavo": "Ottavo", "ottavo_adv": "Ottavo stress",
            "banking77_official": "Banking77 (full test)", "massive_it_official": "MASSIVE-it (full test)"}


# ---------- caricamento
def load_runs():
    runs = []
    for p in sorted((RESULTS / "metrics").glob("*.json")):
        d = json.load(open(p))
        if "run_id" not in d or not (RESULTS / "predictions" / f"{d['run_id']}.jsonl").exists():
            continue
        c = d["config"]
        kind = c.get("kind")
        if kind == "decision_api":
            key = dict(family="decision", system=c["provider"], labels=c["labels"], perm=c["perm"], shots=c["shots"],
                       subset=c["subset"], repeat=c["repeat"], budget="0", seed=None, recipe=None)
        elif kind == "zeroshot_lm":
            key = dict(family="generalist", system=c["model"], labels=c["labels"], perm=c["perm"], shots="0",
                       subset=c["subset"], repeat=0, budget="0", seed=None, recipe=None)
        elif kind == "supervised":
            key = dict(family="supervised", system=c["model"], labels=None, perm=0, shots=None, subset="full", repeat=0,
                       budget=str(c["budget"]), seed=c["seed"], recipe=c.get("recipe", "corrected"),
                       train_seed=c.get("train_seed"))
        else:
            continue
        runs.append(dict(run_id=d["run_id"], task=c["task"], metrics=d["metrics"], config=c, **key))
    return runs


_pred_cache = {}


def preds(run_id):
    if run_id not in _pred_cache:
        _pred_cache[run_id] = {r["example_id"]: r for r in read_jsonl(RESULTS / "predictions" / f"{run_id}.jsonl")}
    return _pred_cache[run_id]


# ---------- macro-F1 veloce per il bootstrap
def encode(task, ids, run_ids):
    lab = labels(task)
    idx = {l: i for i, l in enumerate(lab)}
    first = preds(run_ids[0])
    g = np.array([idx[first[i]["gold_label"]] for i in ids])
    P = []
    for rid in run_ids:
        pr = preds(rid)
        P.append(np.array([idx.get(pr[i]["predicted_label"], -1) if pr[i]["predicted_label"] else -1 for i in ids]))
    return g, P, len(lab)


def macro_f1_idx(g, p, k, w=None):
    w = np.ones(len(g)) if w is None else w
    present = np.bincount(g, weights=w, minlength=k) > 0
    tp = np.bincount(g[p == g], weights=w[p == g], minlength=k)
    fp = np.bincount(p[(p != g) & (p >= 0)], weights=w[(p != g) & (p >= 0)], minlength=k)
    fn = np.bincount(g[p != g], weights=w[p != g], minlength=k)
    denom = 2 * tp + fp + fn
    f1 = np.where(denom > 0, 2 * tp / np.maximum(denom, 1e-12), 0.0)
    return f1[present].mean()


def paired_bootstrap(task, ids, runs_a, runs_b):
    """Differenza di macro-F1 (media sui run di A meno media sui run di B) con IC 95% appaiato sugli esempi."""
    g, Pa, k = encode(task, ids, runs_a)
    _, Pb, _ = encode(task, ids, runs_b)
    point = np.mean([macro_f1_idx(g, p, k) for p in Pa]) - np.mean([macro_f1_idx(g, p, k) for p in Pb])
    rng = np.random.default_rng(BOOT_SEED)
    n = len(ids)
    diffs = np.empty(B)
    for b in range(B):
        w = np.bincount(rng.integers(0, n, n), minlength=n).astype(float)  # pesi multinomiali = ricampionamento
        diffs[b] = np.mean([macro_f1_idx(g, p, k, w) for p in Pa]) - np.mean([macro_f1_idx(g, p, k, w) for p in Pb])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p_two = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return point, lo, hi, min(1.0, p_two)


def bootstrap_ci(task, ids, run_ids):
    g, P, k = encode(task, ids, run_ids)
    rng = np.random.default_rng(BOOT_SEED)
    n = len(ids)
    vals = []
    for b in range(B):
        w = np.bincount(rng.integers(0, n, n), minlength=n).astype(float)
        vals.append(np.mean([macro_f1_idx(g, p, k, w) for p in P]))
    return np.percentile(vals, [2.5, 97.5])


# ---------- selezione dei run
def pick(runs, **kw):
    out = [r for r in runs if all(r.get(k) == v for k, v in kw.items())]
    # in caso di run duplicati con la stessa configurazione, il più recente
    by = {}
    for r in out:
        by[config_key(r)] = r
    return list(by.values())


def config_key(r):
    """Una configurazione = stesso task, sistema, condizioni e seed; fra run ripetuti vale il più recente."""
    return (r["task"], r["system"], r.get("seed"), r.get("labels"), r.get("perm"), r.get("shots"), r.get("subset"),
            r.get("repeat"), r.get("budget"), r.get("recipe"), r.get("train_seed"))


def used_runs(runs):
    """run_id dei run effettivamente usati (i run_id contengono il timestamp: in ordine, l'ultimo vince)."""
    last = {}
    for r in sorted(runs, key=lambda r: r["run_id"]):
        last[config_key(r)] = r["run_id"]
    return set(last.values())


def near_duplicate_ids(task, thr=0.9):
    """ID del test con similarità coseno (char 3-5-gram) >= thr con un esempio di train."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import linear_kernel
    from .common import test_set, train_pool
    tr, te = train_pool(task), test_set(task)
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True).fit([r["text"] for r in tr + te])
    A, Bm = vec.transform([r["text"] for r in te]), vec.transform([r["text"] for r in tr])
    out = set()
    for i in range(0, len(te), 500):
        best = linear_kernel(A[i:i + 500], Bm).max(1)
        out |= {te[i + j]["id"] for j in np.where(best >= thr)[0]}
    return out


def mean_f1(rs):
    return float(np.mean([r["metrics"]["macro_f1"] for r in rs])) if rs else None


def ids_of(task, rs):
    return list(preds(rs[0]["run_id"]).keys())


def holm(pvals):
    order = np.argsort(pvals)
    adj = np.empty(len(pvals))
    run = 0
    for rank, i in enumerate(order):
        run = max(run, (len(pvals) - rank) * pvals[i])
        adj[i] = min(1.0, run)
    return adj


# ---------- contrasti primari
def contrasts(runs):
    rows = []

    def add(cid, task, a_desc, b_desc, ra, rb):
        if not ra or not rb:
            rows.append(dict(id=cid, task=task, a=a_desc, b=b_desc, status="blocked/missing"))
            return
        ids = sorted(set(preds(ra[0]["run_id"])) & set(preds(rb[0]["run_id"])))
        d, lo, hi, p = paired_bootstrap(task, ids, [r["run_id"] for r in ra], [r["run_id"] for r in rb])
        rows.append(dict(id=cid, task=task, a=a_desc, b=b_desc, n=len(ids), a_runs=len(ra), b_runs=len(rb),
                         a_f1=round(mean_f1(ra), 4), b_f1=round(mean_f1(rb), 4), diff=round(d, 5), ci_lo=round(lo, 5),
                         ci_hi=round(hi, 5), p_boot=round(p, 4), status="ok"))

    api = dict(family="decision", perm=0, shots="0", subset="full", repeat=0)
    for t in ["banking77", "massive_it", "ottavo", "banking77_official", "massive_it_official"]:
        for s in ["openai", "elephant", "mammoth"]:
            add("P1", t, f"{s} D", f"{s} N", pick(runs, task=t, system=s, labels="D", **api),
                pick(runs, task=t, system=s, labels="N", **api))
    for t in ["banking77", "massive_it", "ottavo"]:
        dm = {s: pick(runs, task=t, system=s, labels="D", **api) for s in ["openai", "elephant", "mammoth"]}
        dm = {s: r for s, r in dm.items() if r}
        if not dm:
            continue
        best = max(dm, key=lambda s: mean_f1(dm[s]))
        for sup in ["tfidf", "setfit", "mmbert", "qwen0.6b", "qwen1.7b", "qwen8b"]:
            for b in ["5", "20", "full"]:
                rs = pick(runs, task=t, family="supervised", system=sup, budget=b, recipe="corrected")
                if rs:
                    add("P2", t, f"{best} D zero-shot", f"{sup} k={b} ({len(rs)} seeds)", dm[best], rs)
        # P2N (aggiunta 26/10/2026 dopo revisione esterna): le descrizioni D di Banking77 e MASSIVE-it sono state scritte
        # leggendo il training etichettato, quindi il confronto a zero etichette usa i soli nomi (N)
        dmn = {s: pick(runs, task=t, system=s, labels="N", **api) for s in ["openai", "elephant", "mammoth"]}
        dmn = {s: r for s, r in dmn.items() if r}
        bestn = max(dmn, key=lambda s: mean_f1(dmn[s]))
        for sup in ["tfidf", "setfit", "mmbert", "qwen0.6b", "qwen1.7b", "qwen8b"]:
            for b in ["5", "20", "full"]:
                rs = pick(runs, task=t, family="supervised", system=sup, budget=b, recipe="corrected")
                if rs:
                    add("P2N", t, f"{bestn} N zero-shot", f"{sup} k={b} ({len(rs)} seeds)", dmn[bestn], rs)
    for t in ["ottavo", "ottavo_adv"]:
        el = pick(runs, task=t, system="elephant", labels="D", **api)
        q35 = pick(runs, task=t, system="qwen3.5-9b", labels="D", perm=0, subset="full")
        q3 = pick(runs, task=t, system="qwen3-8b", labels="D", perm=0, subset="full")
        add("P3", t, "Elephant-1 D", "Qwen3.5-9B zero-shot D", el, q35)
        add("P3", t, "Qwen3.5-9B zero-shot D", "Qwen3-8B zero-shot D", q35, q3)
        for b in ["5", "20", "full"]:
            add("P4", t, f"Qwen3-8B QLoRA k={b}", "Qwen3-8B zero-shot D",
                pick(runs, task=t, system="qwen8b", budget=b, recipe="corrected"), q3)
    for sup in ["qwen0.6b", "qwen8b"]:
        for b in ["5", "20"]:
            orig = pick(runs, task="ottavo", system=sup, budget=b, recipe="original")
            seeds = {r["seed"] for r in orig}
            corr = [r for r in pick(runs, task="ottavo", system=sup, budget=b, recipe="corrected") if r["seed"] in seeds]
            add("P5", "ottavo", f"{sup} k={b} corrected", f"{sup} k={b} original", corr, orig)
    # S1: sensibilità dichiarata prima del run, ricetta esplorativa su Banking77 con tutti i dati (4 seed di training)
    for t in ["banking77", "banking77_official"]:
        orig = pick(runs, task=t, system="qwen0.6b", budget="full", recipe="original")
        corr = pick(runs, task=t, system="qwen0.6b", budget="full", recipe="corrected")
        add("S1", t, f"qwen0.6b k=full original ({len(orig)} train seeds)", f"qwen0.6b k=full corrected ({len(corr)} seeds)",
            orig, corr)
        for lab in ["D", "N"]:
            add("S1", t, f"mammoth {lab} zero-shot", f"qwen0.6b k=full original ({len(orig)} train seeds)",
                pick(runs, task=t, system="mammoth", labels=lab, **api), orig)
    fam = defaultdict(list)
    for i, r in enumerate(rows):
        if r["status"] == "ok":
            fam[r["id"]].append(i)
    for f, idx in fam.items():
        adj = holm(np.array([rows[i]["p_boot"] for i in idx]))
        for i, a in zip(idx, adj):
            rows[i]["p_holm"] = round(float(a), 4)
    return rows


# ---------- rischio-copertura con soglie scelte in cross-fitting (metà test sceglie, l'altra valuta, e viceversa)
def crossfit_accept(run_id, target_risk):
    """Soglia scelta su una metà del test per avere errore <= target_risk fra i casi accettati, applicata all'altra metà,
    poi scambio. Le risposte senza probabilità (rifiuti) restano nel denominatore e vanno sempre in revisione.
    Ritorna (id, accettato, errato, metà) per messaggio; ogni messaggio è valutato una sola volta."""
    rows = sorted(preds(run_id).values(), key=lambda r: r["example_id"])
    if not any(r.get("probs") for r in rows):
        return None
    conf = np.array([max(r["probs"]) if r.get("probs") else -1.0 for r in rows])
    err = np.array([r["predicted_label"] != r["gold_label"] for r in rows])
    rng = np.random.default_rng(BOOT_SEED)
    half = rng.permutation(len(rows)) < len(rows) // 2
    auto = np.zeros(len(rows), bool)
    for sel, ev in ((half, ~half), (~half, half)):
        cands = np.unique(conf[sel][conf[sel] >= 0])
        ok = [t for t in cands if err[sel][conf[sel] >= t].mean() <= target_risk]
        t = min(ok) if ok else 1.01
        auto[ev] = conf[ev] >= t
    return [r["example_id"] for r in rows], auto, err, half


def selective(run_ids, task, target_risk=0.02):
    """Copertura ed errore fra gli accettati con soglie in cross-fitting (media delle due metà di valutazione)."""
    out = []
    for rid in run_ids:
        res = crossfit_accept(rid, target_risk)
        if res is None:
            continue
        _, auto, err, half = res
        acc_auto, err_auto, caught = [], [], []
        for ev in (~half, half):
            a, e = auto[ev], err[ev]
            acc_auto.append(a.mean())
            err_auto.append(e[a].mean() if a.any() else 0.0)
            caught.append((e & ~a).sum() / max(1, e.sum()))
        out.append(dict(auto=np.mean(acc_auto), review=1 - np.mean(acc_auto), err_auto=np.mean(err_auto),
                        errors_caught=np.mean(caught), n=len(auto), base_error=float(err.mean())))
    if not out:
        return None
    return {k: float(np.mean([o[k] for o in out])) for k in out[0]}


def risk_coverage_curve(run_id):
    rows = [r for r in preds(run_id).values() if r.get("probs")]
    conf = np.array([max(r["probs"]) for r in rows])
    err = np.array([r["predicted_label"] != r["gold_label"] for r in rows], float)
    o = np.argsort(-conf)
    cov = np.arange(1, len(o) + 1) / len(o)
    risk = np.cumsum(err[o]) / np.arange(1, len(o) + 1)
    return cov, risk


# ---------- uscita
def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(rows)


def main():
    runs = load_runs()
    used = used_runs(runs)
    idx_rows = [dict(used=r["run_id"] in used, run_id=r["run_id"], task=r["task"], family=r["family"], system=r["system"], labels=r["labels"],
                     perm=r["perm"], shots=r["shots"], subset=r["subset"], repeat=r["repeat"], budget=r["budget"],
                     seed=r["seed"], recipe=r["recipe"], **{k: r["metrics"].get(k) for k in
                     ("n", "coverage", "macro_f1", "accuracy", "ece15", "brier", "latency_p50_ms", "cost_usd", "refusals")})
                for r in runs]
    write_csv(RESULTS / "runs_index.csv", idx_rows)
    runs = [r for r in runs if r["run_id"] in used]  # i run superati restano su disco ma non entrano nelle analisi
    con = contrasts(runs)
    write_csv(ROOT / "reports" / "contrasts.csv", con)
    # sensibilità: P2 e P2N nel regime full escludendo i quasi-duplicati train/test (Banking77, MASSIVE-it)
    sens = []
    for task in ["banking77", "massive_it"]:
        nd = near_duplicate_ids(task)
        for c in [c for c in con if c["id"] in ("P2", "P2N") and c["task"] == task and c["status"] == "ok" and "k=full" in c["b"]]:
            sup = c["b"].split()[0]
            dm, lab = c["a"].split()[:2]
            ra = pick(runs, task=task, system=dm, labels=lab, family="decision", perm=0, shots="0", subset="full", repeat=0)
            rb = pick(runs, task=task, system=sup, budget="full", recipe="corrected")
            ids = sorted(set(preds(ra[0]["run_id"])) - nd)
            d, lo, hi, p = paired_bootstrap(task, ids, [x["run_id"] for x in ra], [x["run_id"] for x in rb])
            sens.append(dict(id=c["id"], task=task, a=c["a"], b=c["b"], n_all=c["n"], n_without_near_dups=len(ids), excluded=len(nd),
                             diff_all=c["diff"], diff=round(d, 4), ci_lo=round(lo, 4), ci_hi=round(hi, 4)))
    write_csv(ROOT / "reports" / "contrasts_sensitivity_neardup.csv", sens)
    sel = []
    for r in runs:
        if r["subset"] == "full" and r["perm"] == 0 and r["repeat"] == 0 and (r["shots"] in ("0", None)):
            for risk in (0.02, 0.05):
                s = selective([r["run_id"]], r["task"], risk)
                if s:
                    sel.append(dict(run_id=r["run_id"], task=r["task"], system=r["system"], budget=r["budget"],
                                    seed=r["seed"], labels=r["labels"], recipe=r["recipe"], target_risk=risk,
                                    **{k: round(v, 4) for k, v in s.items()}))
    write_csv(ROOT / "reports" / "selective.csv", sel)
    print(f"{len(runs)} run, {sum(c['status'] == 'ok' for c in con)} contrasti calcolati, {len(sel)} analisi selettive")
    from . import figures
    figures.main(runs)


if __name__ == "__main__":
    main()
