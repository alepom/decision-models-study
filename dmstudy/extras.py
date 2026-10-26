"""Analisi aggiuntive del paper, tutte dalle predizioni salvate (nessuna inferenza):

- analisi degli errori: coppie di classi più confuse, errori condivisi fra decision model e classificatore compatto;
- F1 per classe con supporto (CSV completo) e sintesi per l'appendice;
- variabilità fra seed (media, deviazione standard, min, max);
- diagrammi di affidabilità (calibrazione);
- qualità contro latenza, con costo misurato;
- scenari di costo con ipotesi esplicite;
- accettazione automatica: copertura ed errore osservato sulle metà di valutazione, con intervallo di Wilson.

  python -m dmstudy.extras
"""
import csv
from collections import Counter, defaultdict

import numpy as np

from .analysis import BOOT_SEED, DATASETS, crossfit_accept, SYSNAME, load_runs, mean_f1, pick, preds, used_runs
from .common import ROOT, labels, manifest
from .figures import API, COL, setup

GEN = ROOT / "paper" / "generated"
FIG = ROOT / "paper" / "figures"
REP = ROOT / "reports"
SUP = ["tfidf", "setfit", "mmbert", "qwen0.6b", "qwen1.7b", "qwen8b"]
DM = ["mammoth", "elephant", "openai"]


def tex(s):
    return str(s).replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")


def brk(s):
    """nome di classe con interruzioni di riga consentite dopo i trattini bassi"""
    return tex(s).replace(r"\_", r"\_\allowbreak{}")


def best_dm(runs, task):
    cand = {s: pick(runs, task=task, system=s, labels="D", **API) for s in DM}
    cand = {s: r for s, r in cand.items() if r}
    s = max(cand, key=lambda k: mean_f1(cand[k]))
    return s, cand[s]


def best_sup(runs, task, budget="full"):
    cand = {s: pick(runs, task=task, system=s, budget=budget, recipe="corrected") for s in SUP}
    cand = {s: r for s, r in cand.items() if r}
    s = max(cand, key=lambda k: mean_f1(cand[k]))
    return s, cand[s]


# ---------- analisi degli errori
def error_analysis(runs):
    rows_tex, rows_csv, overlap = [], [], []
    for task in ["banking77", "massive_it", "ottavo"]:
        sd, rd = best_dm(runs, task)
        ss, rs = best_sup(runs, task)
        pd_ = preds(rd[0]["run_id"])
        conf_d = Counter((r["gold_label"], r["predicted_label"]) for r in pd_.values()
                         if r["predicted_label"] and r["predicted_label"] != r["gold_label"])
        conf_s = Counter()
        err_s_any = set()
        for r in rs:  # errori sommati sui seed, poi divisi per il numero di seed
            for i, p in preds(r["run_id"]).items():
                if p["predicted_label"] != p["gold_label"]:
                    conf_s[(p["gold_label"], p["predicted_label"])] += 1 / len(rs)
                    err_s_any.add(i)
        err_d = {i for i, p in pd_.items() if p["predicted_label"] != p["gold_label"]}
        # sovrapposizione calcolata seed per seed e poi mediata (confronto simmetrico: un run contro un run)
        per_seed = []
        for r in rs:
            es = {i for i, p in preds(r["run_id"]).items() if p["predicted_label"] != p["gold_label"]}
            per_seed.append((len(es), len(err_d & es), len(err_d - es), len(es - err_d)))
        m = np.mean(per_seed, 0)
        overlap.append(dict(task=task, dm=sd, sup=ss, dm_errors=len(err_d), sup_errors=round(m[0], 1),
                            both=round(m[1], 1), only_dm=round(m[2], 1), only_sup=round(m[3], 1), n=len(pd_)))
        for who, conf in ((sd, conf_d), (ss, conf_s)):
            for (g, p), c in conf.most_common(5):
                rows_csv.append(dict(task=task, system=who, gold=g, predicted=p, errors=round(c, 2)))
        # tabella LaTeX: le 3 coppie più confuse per ciascuno dei due sistemi, una per riga
        for name, top, fmt in ((SYSNAME[sd] + " (zero-shot, D)", conf_d.most_common(3), "{:.0f}"),
                               (SYSNAME[ss] + " (all data)", conf_s.most_common(3), "{:.1f}")):
            for k, ((g, pr), c) in enumerate(top):
                rows_tex.append((DATASETS[task] if not rows_tex or rows_tex[-1][0] == "" and False else "", name if k == 0 else "",
                                 f"{brk(g)} $\\to$ {brk(pr)}", fmt.format(c), task))
    with open(REP / "error_pairs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_csv[0])); w.writeheader(); w.writerows(rows_csv)
    with open(REP / "error_overlap.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(overlap[0])); w.writeheader(); w.writerows(overlap)
    lines = [r"\begin{tabular}{ll>{\raggedright\arraybackslash}p{7.7cm}r}", r"\toprule",
             r"Data & System & Confusion (gold $\to$ predicted) & Count \\", r"\midrule"]
    last = None
    for r in rows_tex:
        if last and r[4] != last:
            lines.append(r"\midrule")
        lines.append(" & ".join((DATASETS[r[4]] if r[4] != last else "",) + r[1:4]) + r" \\")
        last = r[4]
    lines += [r"\bottomrule", r"\end{tabular}"]
    (GEN / "table_errors.tex").write_text("\n".join(lines))
    lines = [r"\begin{tabular}{llrrrrrr}", r"\toprule",
             r"Data & Decision model / compact classifier & $n$ & DM errors & Compact errors & Both & Only DM & Only compact \\", r"\midrule"]
    for o in overlap:
        lines.append(f"{DATASETS[o['task']]} & {SYSNAME[o['dm']]} / {SYSNAME[o['sup']]} & {o['n']} & {o['dm_errors']} & "
                     f"{o['sup_errors']:.1f} & {o['both']:.1f} & {o['only_dm']:.1f} & {o['only_sup']:.1f}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (GEN / "table_error_overlap.tex").write_text("\n".join(lines))
    return overlap


# ---------- F1 per classe e variabilità fra seed
def per_class_and_seeds(runs):
    from sklearn.metrics import f1_score
    out = []
    summ = []
    for task in ["banking77", "massive_it", "ottavo"]:
        lab = labels(task)
        systems = [(s, pick(runs, task=task, system=s, labels="D", **API)) for s in DM] + \
                  [(s, pick(runs, task=task, system=s, budget="full", recipe="corrected")) for s in SUP]
        for s, rs in systems:
            if not rs:
                continue
            per = defaultdict(list)
            sup = Counter()
            for r in rs:
                pr = list(preds(r["run_id"]).values())
                g = [p["gold_label"] for p in pr]
                y = [p["predicted_label"] or "__none__" for p in pr]
                present = sorted(set(g))
                f = f1_score(g, y, labels=present, average=None, zero_division=0)
                for l, v in zip(present, f):
                    per[l].append(v)
                if not sup:
                    sup = Counter(g)
            for l in sorted(per):
                out.append(dict(task=task, system=s, label=l, support=sup[l], f1_mean=round(float(np.mean(per[l])), 4)))
            vals = np.array([np.mean(v) for v in per.values()])
            worst = sorted(per, key=lambda l: np.mean(per[l]))[:3]
            summ.append(dict(task=task, system=s, median=float(np.median(vals)), p10=float(np.percentile(vals, 10)),
                             min=float(vals.min()), below_05=int((vals < 0.5).sum()), classes=len(vals),
                             worst=", ".join(worst)))
    with open(REP / "per_class_f1.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    lines = [r"\begin{tabular}{llrrrr>{\raggedright\arraybackslash}p{4.9cm}}", r"\toprule",
             r"Data & System & Median & 10th pct. & Min & $<0.5$ & Three weakest classes \\", r"\midrule"]
    last = None
    for s in summ:
        if last and s["task"] != last:
            lines.append(r"\midrule")
        last = s["task"]
        lines.append(f"{DATASETS[s['task']]} & {SYSNAME[s['system']]} & {s['median']:.3f} & {s['p10']:.3f} & {s['min']:.3f} & "
                     f"{s['below_05']}/{s['classes']} & {brk(s['worst'])}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (GEN / "table_per_class.tex").write_text("\n".join(lines))
    # variabilità fra seed
    seeds = []
    lines = [r"\begin{tabular}{lll" + "r" * 4 + "}", r"\toprule",
             r"Data & System & Examples/class & Mean & SD & Min & Max \\", r"\midrule"]
    last = None
    for task in ["banking77", "massive_it", "ottavo"]:
        for s in SUP:
            for b in ["5", "20", "full"]:
                rs = pick(runs, task=task, system=s, budget=b, recipe="corrected")
                if len(rs) < 2:
                    continue
                v = np.array([r["metrics"]["macro_f1"] for r in rs])
                seeds.append(dict(task=task, system=s, budget=b, n=len(v), mean=v.mean(), sd=v.std(ddof=1), min=v.min(), max=v.max()))
                if last and task != last:
                    lines.append(r"\midrule")
                last = task
                lines.append(f"{DATASETS[task]} & {SYSNAME[s]} & {b if b != 'full' else 'all'} & {v.mean():.3f} & {v.std(ddof=1):.3f} & "
                             f"{v.min():.3f} & {v.max():.3f}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (GEN / "table_seeds.tex").write_text("\n".join(lines))
    with open(REP / "seed_variability.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(seeds[0])); w.writeheader(); w.writerows(seeds)
    return summ, seeds


# ---------- diagrammi di affidabilità
def reliability(plt, runs):
    tasks = ["banking77", "massive_it", "ottavo"]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))
    edges = np.linspace(0, 1, 11)
    for ax, task in zip(axes, tasks):
        ax.plot([0, 1], [0, 1], color="#999", lw=0.8, ls=":")
        series = [(s, pick(runs, task=task, system=s, labels="D", **API)) for s in DM]
        ss, rs = best_sup(runs, task)
        series.append((ss, [x for x in rs if x["seed"] == 42] or rs[:1]))
        for s, r in series:
            if not r:
                continue
            pr = [p for p in preds(r[0]["run_id"]).values() if p.get("probs")]
            conf = np.array([max(p["probs"]) for p in pr])
            ok = np.array([p["predicted_label"] == p["gold_label"] for p in pr])
            xs, ys = [], []
            for lo, hi in zip(edges[:-1], edges[1:]):
                m = (conf > lo) & (conf <= hi)
                if m.sum() >= 10:  # bin con almeno 10 esempi
                    xs.append(conf[m].mean()); ys.append(ok[m].mean())
            ax.plot(xs, ys, "-o", ms=2.5, lw=1.1, color=COL[s],
                    label=SYSNAME[s] + (" (all data, seed 42)" if s in SUP else ""))
        ax.set_title(DATASETS[task]); ax.set_xlabel("mean confidence in bin"); ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    axes[0].set_ylabel("accuracy in bin")
    h, l = [], []
    for ax in axes:
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in l:
                h.append(hh); l.append(ll)
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.12), fontsize=6.5)
    fig.text(0.995, -0.16, "10 equal-width bins; bins with fewer than 10 messages omitted. Dotted: perfect calibration.",
             ha="right", fontsize=6, color="#666")
    fig.tight_layout()
    fig.savefig(FIG / "reliability.pdf", bbox_inches="tight"); plt.close(fig)


# ---------- qualità contro latenza
def quality_latency(plt, runs):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=False)
    for ax, task in zip(axes, ["banking77", "ottavo"]):
        pts = []
        for s in DM:
            r = pick(runs, task=task, system=s, labels="D", **API)
            if r:
                m = r[0]["metrics"]
                cost = 1000 * m.get("cost_usd", 0) / m["n"] if s == "openai" else None
                pts.append((s, m["latency_p50_ms"], m["macro_f1"], "API", cost))
        for s in SUP:
            rs = pick(runs, task=task, system=s, budget="full", recipe="corrected")
            lat = [r["metrics"].get("latency_p50_ms") for r in rs if r["metrics"].get("latency_p50_ms")]
            if rs and lat:
                pts.append((s, float(np.mean(lat)), mean_f1(rs), "local", None))
        for s in ["qwen3-8b", "qwen3.5-9b"]:
            r = pick(runs, task=task, system=s, labels="D", perm=0, subset="full")
            if r:
                pts.append((s, r[0]["metrics"]["latency_p50_ms"], r[0]["metrics"]["macro_f1"], "local", None))
        # posizioni delle etichette scelte a mano dove i punti sono vicini (offset in punti tipografici)
        OFF = {("ottavo", "mammoth"): (-62, 4), ("ottavo", "openai"): (5, 2), ("ottavo", "elephant"): (5, -9),
               ("ottavo", "setfit"): (-8, 6), ("ottavo", "qwen1.7b"): (5, -9), ("ottavo", "qwen8b"): (-66, 3),
               ("banking77", "openai"): (-20, 8), ("banking77", "mammoth"): (-40, 5)}
        NAME = {"qwen8b": "Qwen3-8B QLoRA", "qwen1.7b": "Qwen3-1.7B LoRA", "qwen0.6b": "Qwen3-0.6B LoRA",
                "qwen3-8b": "Qwen3-8B zero-shot", "qwen3.5-9b": "Qwen3.5-9B zero-shot", "setfit": "SetFit"}
        for s, lat, f1, kind, cost in pts:
            ax.scatter(lat, f1, color=COL[s], marker="o" if kind == "API" else "s", s=28, zorder=3,
                       edgecolor="white", linewidth=0.6)
            lab = NAME.get(s, SYSNAME[s])
            if cost is not None:
                lab += f" ({cost:.2f} USD/1k)"
            dx, dy = OFF.get((task, s), (5, -3))
            ax.annotate(lab, (lat, f1), textcoords="offset points", xytext=(dx, dy), fontsize=5.5, color="#333")
        ax.set_xscale("log"); ax.set_xlabel("median latency per message, ms (log scale)")
        ax.set_title(DATASETS[task])
    axes[0].set_ylabel("macro-F1")
    fig.text(0.01, -0.05, "Circles: decision APIs (end to end, network included; OpenAI cost from billed input tokens; Baseten free "
             "promotional access). Squares: local models on a Colab T4 (supervised: all data, mean of 3 seeds; generalists: zero-shot). "
             "Latencies are not comparable across API and local hardware.", fontsize=5.5, color="#666", wrap=True)
    fig.tight_layout()
    fig.savefig(FIG / "quality_latency.pdf", bbox_inches="tight"); plt.close(fig)


# ---------- scenari di costo
def cost_scenarios(runs):
    """Costi misurati + ipotesi esplicite; nessuna soglia universale."""
    hf_thr = {}
    for s in ["mmbert", "qwen0.6b", "qwen8b"]:
        rs = pick(runs, task="ottavo", system=s, budget="full", recipe="corrected")
        lat = [r["metrics"].get("latency_p50_ms") for r in rs if r["metrics"].get("latency_p50_ms")]
        train = [r["metrics"].get("train_s") for r in rs if r["metrics"].get("train_s")]
        if lat:
            hf_thr[s] = (float(np.mean(lat)), float(np.mean(train)) / 60)
    oa = {}
    for task in ["ottavo", "banking77"]:
        r = pick(runs, task=task, system="openai", labels="D", **API)
        oa[task] = 1000 * r[0]["metrics"]["cost_usd"] / r[0]["metrics"]["n"]
    labels_hour = 120  # IPOTESI: messaggi etichettati per ora da una persona (con doppia annotazione il tempo raddoppia)
    lines = [r"\begin{tabular}{lrrr}", r"\toprule",
             r"Item & Ottavo-like (24 classes) & Banking77-like (77 classes) & Basis \\", r"\midrule",
             f"OpenAI Decisions, zero-shot, USD per 1M messages & {oa['ottavo'] * 1000:.0f} & {oa['banking77'] * 1000:.0f} & measured (beta price) \\\\",
             r"Elephant-1 / Mammoth-1, USD per 1M messages & n/a & n/a & promotional access, no price \\",
             f"Labels for 20 per class & {20 * 24:,} & {20 * 77:,} & protocol \\\\",
             f"Labels for the full training sets used here & {1669:,} & {9993:,} & protocol \\\\",
             f"Person-hours to label 20 per class (one annotator) & {20 * 24 / labels_hour:.0f} & {20 * 77 / labels_hour:.0f} & assumption: {labels_hour} labels/hour \\\\",
             f"Person-hours to label the full sets (one annotator) & {1669 / labels_hour:.0f} & {9993 / labels_hour:.0f} & assumption: {labels_hour} labels/hour \\\\"]
    for s, (lat, tmin) in hf_thr.items():
        lines.append(f"{SYSNAME[s]}: training on all Ottavo data, T4 minutes & {tmin:.0f} & -- & measured \\\\")
        lines.append(f"{SYSNAME[s]}: inference, T4 GPU-hours per 1M messages & {1e6 * lat / 1000 / 3600:.1f} & -- & upper bound from batch-1 latency \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (GEN / "table_costs.tex").write_text("\n".join(lines).replace(",", "{,}"))
    return oa, hf_thr


# ---------- accettazione automatica, conteggi verificabili
def selective_counts(run_ids, target, reps=2000):
    """Accettati ed errori sommati sulle due metà di valutazione e sui seed; intervallo dell'errore osservato con bootstrap
    sui messaggi (tutte le predizioni dei seed per lo stesso messaggio restano insieme)."""
    res = [crossfit_accept(r, target) for r in run_ids]
    ids = res[0][0]
    assert all(r[0] == ids for r in res), "run con messaggi diversi"
    A = np.stack([r[1] for r in res]).astype(float)        # seed x messaggi
    E = np.stack([r[1] & r[2] for r in res]).astype(float)  # errori fra gli accettati
    a_m, e_m = A.sum(0), E.sum(0)
    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, len(ids), size=(reps, len(ids)))
    acc_b, err_b = a_m[idx].sum(1), e_m[idx].sum(1)
    ratio = err_b[acc_b > 0] / acc_b[acc_b > 0]
    lo, hi = (np.percentile(ratio, [2.5, 97.5]) if len(ratio) else (float("nan"), float("nan")))
    return len(ids), len(run_ids), int(A.sum()), int(E.sum()), float(lo), float(hi)


def selective_table(runs):
    lines = [r"\begin{tabular}{llrrrrrl}", r"\toprule",
             r"Data & System & Target & $n$ & Runs & Coverage & Err./acc. & Error [95\% CI] \\", r"\midrule"]
    out = []
    for i, task in enumerate(["banking77", "massive_it", "ottavo"]):
        if i:
            lines.append(r"\midrule")
        systems = [(SYSNAME[s], pick(runs, task=task, system=s, labels="D", **API)) for s in DM]
        ss, rs = best_sup(runs, task)
        systems.append((SYSNAME[ss].replace(" LoRA", "").replace(" QLoRA", "") + ", all data", rs))
        if task == "banking77":
            orig = pick(runs, task=task, system="qwen0.6b", budget="full", recipe="original")
            systems.append(("Qwen3-0.6B, expl. recipe", orig))
        for target in (0.02, 0.05):
            for name, rr in systems:
                if not rr:
                    continue
                n, k, acc, bad, lo, hi = selective_counts([r["run_id"] for r in rr], target)
                cov = [crossfit_accept(r["run_id"], target)[1].mean() for r in rr]
                out.append(dict(task=task, system=name, target=target, n=n, runs=k, accepted=acc, errors=bad,
                                coverage=acc / (n * k), coverage_min=min(cov), coverage_max=max(cov),
                                observed_error=bad / acc if acc else None, ci_lo=lo, ci_hi=hi))
                c = f"{100 * acc / (n * k):.0f}\\%" + (f" ({100 * min(cov):.0f}--{100 * max(cov):.0f})" if k > 1 else "")
                obs = f"{100 * bad / acc:.1f}\\% [{100 * lo:.1f}, {100 * hi:.1f}]" if acc else "--"
                first = not any(o["task"] == task for o in out[:-1])
                lines.append(f"{DATASETS[task] if first else ''} & {tex(name)} & {100 * target:.0f}\\% & {n} & {k} & {c} & "
                             f"{bad} / {acc} & {obs} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (GEN / "table_selective.tex").write_text("\n".join(lines))
    with open(REP / "selective_counts.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    return out


def main():
    allr = load_runs()
    used = used_runs(allr)
    runs = [r for r in allr if r["run_id"] in used]
    plt = setup()
    ov = error_analysis(runs)
    summ, seeds = per_class_and_seeds(runs)
    reliability(plt, runs)
    quality_latency(plt, runs)
    oa, thr = cost_scenarios(runs)
    selective_table(runs)
    for o in ov:
        print(o)
    print("seed sd max:", max(s["sd"] for s in seeds))
    print("openai USD/1k:", oa, "throughput/train:", thr)


if __name__ == "__main__":
    main()
