"""Figure (PDF vettoriali) e tabelle LaTeX del paper, generate dai run salvati. Chiamato da dmstudy.analysis."""
import csv
from collections import defaultdict

import numpy as np

from .common import ROOT, labels, manifest
from .analysis import (DATASETS, FIG, GEN, SYSNAME, bootstrap_ci, ids_of, mean_f1, pick, preds, risk_coverage_curve)

# palette categoriale validata (ordine fisso, colorblind-safe sulle coppie adiacenti)
COL = {"mammoth": "#2a78d6", "elephant": "#eb6834", "openai": "#1baf7a", "qwen0.6b": "#eda100", "setfit": "#e87ba4",
       "tfidf": "#008300", "qwen1.7b": "#6250d6", "qwen8b": "#e34948", "mmbert": "#7a7a7a",
       "qwen3-8b": "#444444", "qwen3.5-9b": "#9a6b2f"}
API = dict(family="decision", perm=0, shots="0", subset="full", repeat=0)


def setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "legend.fontsize": 7,
                         "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
                         "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.6})
    FIG.mkdir(parents=True, exist_ok=True)
    GEN.mkdir(parents=True, exist_ok=True)
    return plt


def per_class(task, budget):
    if budget != "full":
        return int(budget)
    m = manifest(task)
    return m["train_size"] / len(m["label_order"])


def fig_learning(plt, runs):
    tasks = ["banking77", "massive_it", "ottavo", "ottavo_adv"]
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.3), sharey=False)
    for ax, t in zip(axes, tasks):
        for s in ["mammoth", "elephant", "openai"]:
            r = pick(runs, task=t, system=s, labels="N", **API)  # N: nessuna informazione dal training etichettato
            if r:
                ax.axhline(mean_f1(r), color=COL[s], ls="--", lw=1.1, label=f"{SYSNAME[s]} (0 ex., names)")
        for s in ["tfidf", "setfit", "mmbert", "qwen0.6b", "qwen1.7b", "qwen8b"]:
            xs, ys, lo, hi = [], [], [], []
            for b in ["5", "10", "20", "50", "full"]:
                rs = pick(runs, task=t, family="supervised", system=s, budget=b, recipe="corrected")
                if rs:
                    v = [r["metrics"]["macro_f1"] for r in rs]
                    xs.append(per_class(t, b)); ys.append(np.mean(v)); lo.append(min(v)); hi.append(max(v))
            if xs:
                ax.plot(xs, ys, "-o", ms=3, lw=1.3, color=COL[s], label=SYSNAME[s])
                ax.fill_between(xs, lo, hi, color=COL[s], alpha=0.15, lw=0)
        ax.set_xscale("log")
        full = per_class(t, "full")
        ax.set_xticks([5, 20, full])
        ax.set_xticklabels(["5", "20", f"all\n(~{full:.0f})"])
        ax.minorticks_off()
        ax.set_xlim(4, full * 1.35)
        ax.set_title(DATASETS[t])
    axes[0].set_ylabel("macro-F1")
    fig.supxlabel("labelled examples per class (log scale; 'all' = full training set, average per class)", fontsize=8, y=0.06)
    h, l = axes[2].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.17))
    fig.text(0.995, -0.21, "Solid lines: mean over 3 seeds; bands: min-max. Dashed: decision models, zero-shot, label names only.",
             ha="right", fontsize=6, color="#666")
    fig.tight_layout()
    fig.savefig(FIG / "learning_curves.pdf", bbox_inches="tight"); plt.close(fig)


def fig_descriptions(plt, con):
    rows = [c for c in con if c["id"] == "P1" and c["status"] == "ok"]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(6.2, 0.27 * len(rows) + 0.7))
    vals = [float(c[k]) for c in rows for k in ("a_f1", "b_f1")]
    lo, hi = min(vals) - 0.01, max(vals) + 0.01
    for i, c in enumerate(rows):
        sname = c["a"].split()[0]
        ax.plot([float(c["b_f1"]), float(c["a_f1"])], [i, i], color="#bbb", lw=2, zorder=1)
        ax.scatter([float(c["b_f1"])], [i], color="white", edgecolor=COL[sname], s=24, zorder=2, linewidths=1.3)
        ax.scatter([float(c["a_f1"])], [i], color=COL[sname], s=24, zorder=3)
        ax.text(1.02, i, f"{float(c['diff']):+.3f}  [{float(c['ci_lo']):+.3f}, {float(c['ci_hi']):+.3f}]",
                va="center", fontsize=6.5, transform=ax.get_yaxis_transform(), family="monospace")
    ax.set_xlim(lo, hi)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{DATASETS[c['task']]} · {SYSNAME[c['a'].split()[0]]}" for c in rows])
    ax.set_xlabel("macro-F1   (hollow: label names only;  filled: names + descriptions)")
    ax.text(1.02, -1.1, "D − N  [95% CI]", fontsize=6.5, transform=ax.get_yaxis_transform(), weight="bold")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(FIG / "descriptions_effect.pdf", bbox_inches="tight"); plt.close(fig)


def fig_stress(plt, runs):
    fig, ax = plt.subplots(figsize=(3.6, 2.9))
    items = []
    for s in ["mammoth", "elephant", "openai"]:
        a, b = pick(runs, task="ottavo", system=s, labels="D", **API), pick(runs, task="ottavo_adv", system=s, labels="D", **API)
        if a and b:
            items.append((s, mean_f1(a), mean_f1(b)))
    for s in ["tfidf", "setfit", "mmbert", "qwen0.6b", "qwen1.7b", "qwen8b"]:
        a = pick(runs, task="ottavo", system=s, budget="full", recipe="corrected")
        b = pick(runs, task="ottavo_adv", system=s, budget="full", recipe="corrected")
        if a and b:
            items.append((s, mean_f1(a), mean_f1(b)))
    for s, a, b in items:
        ax.plot([0, 1], [a, b], "-o", ms=3, color=COL[s])
    ymin = min(min(a, b) for _, a, b in items) - 0.01
    ax.set_ylim(ymin, 1.003)
    gap = 0.045 * (1.003 - ymin)
    lab = sorted(((b, s) for s, a, b in items), reverse=True)
    last = 10
    for b, s in lab:  # etichette dall'alto verso il basso, distanziate
        y = min(b, last - gap); last = y
        ax.text(1.05, y, SYSNAME[s], fontsize=6, va="center", color=COL[s])
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Ottavo\n(normal test)", "Ottavo\n(lexical stress)"])
    ax.set_xlim(-0.1, 2.0); ax.set_ylabel("macro-F1")
    ax.set_title("Same training data, lexically adversarial test", pad=10)
    fig.tight_layout(); fig.savefig(FIG / "stress.pdf", bbox_inches="tight"); plt.close(fig)


def fig_risk(plt, runs):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5))
    for ax, t in zip(axes, ["banking77", "ottavo"]):
        for s in ["mammoth", "elephant", "openai"]:
            r = pick(runs, task=t, system=s, labels="D", **API)
            if r:
                cov, risk = risk_coverage_curve(r[0]["run_id"])
                ax.plot(cov, risk, color=COL[s], lw=1.2, label=SYSNAME[s])
        for s in ["setfit", "qwen0.6b", "qwen8b"]:
            r = [x for x in pick(runs, task=t, system=s, budget="full", recipe="corrected") if x["seed"] == 42]
            if r:
                cov, risk = risk_coverage_curve(r[0]["run_id"])
                ax.plot(cov, risk, color=COL[s], lw=1.2, ls="-.", label=SYSNAME[s] + " (all data, seed 42)")
        ax.set_title(DATASETS[t]); ax.set_xlabel("coverage (most confident first)"); ax.set_ylabel("error rate on covered")
        ax.set_xlim(0.02, 1)
    axes[0].set_ylim(0, 0.2)
    axes[1].set_ylim(0, 0.1)
    axes[1].legend(frameon=False, loc="upper left", fontsize=6)
    fig.text(0.01, -0.03, "Curves start at 2% coverage; y-axes are truncated (0.20 and 0.10).", fontsize=6, color="#666")
    fig.tight_layout(); fig.savefig(FIG / "risk_coverage.pdf", bbox_inches="tight"); plt.close(fig)


def fig_progenitor(plt, runs):
    from matplotlib.patches import Patch
    rows = []
    for t in ["ottavo", "ottavo_adv"]:
        for name, rs in [("Qwen3-8B\nzero-shot", pick(runs, task=t, system="qwen3-8b", labels="D", perm=0, subset="full")),
                         ("Qwen3.5-9B\nzero-shot", pick(runs, task=t, system="qwen3.5-9b", labels="D", perm=0, subset="full")),
                         ("Qwen3-8B QLoRA\nall data", pick(runs, task=t, system="qwen8b", budget="full", recipe="corrected")),
                         ("Elephant-1\n(from Qwen3.5-9B)", pick(runs, task=t, system="elephant", labels="D", **API))]:
            if rs:
                lo, hi = bootstrap_ci(t, ids_of(t, rs), [r["run_id"] for r in rs])
                rows.append((t, name, mean_f1(rs), lo, hi))
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    names = list(dict.fromkeys(r[1] for r in rows))
    colors = ["#2a78d6", "#eb6834"]
    for j, t in enumerate(["ottavo", "ottavo_adv"]):
        for i, n in enumerate(names):
            r = [x for x in rows if x[0] == t and x[1] == n]
            if r:
                _, _, v, lo, hi = r[0]
                x = i + (j - 0.5) * 0.36
                ax.bar(x, v, 0.34, color=colors[j], alpha=0.9)
                ax.errorbar(x, v, yerr=[[v - lo], [hi - v]], color="#333", lw=0.8, capsize=2)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, fontsize=6.5)
    ax.set_ylim(0.85, 1.0); ax.set_ylabel("macro-F1 (axis starts at 0.85)")
    ax.legend(handles=[Patch(color=colors[0], label="Ottavo"), Patch(color=colors[1], label="Ottavo lexical stress")],
              frameon=False, loc="upper left", fontsize=6.5, ncol=2)
    fig.tight_layout(); fig.savefig(FIG / "progenitor.pdf", bbox_inches="tight"); plt.close(fig)


def tex_table_main(runs):
    """Tabella principale: macro-F1 per dataset (media sui seed; [min, max] per i supervisionati)."""
    tasks = ["banking77", "massive_it", "ottavo", "ottavo_adv"]
    lines = [r"\begin{tabular}{llrrrr}", r"\toprule",
             r"System & Labels used & Banking77 & MASSIVE-it & Ottavo & Ottavo stress \\", r"\midrule"]
    for s in ["mammoth", "elephant", "openai"]:
        for lab in ["N", "D"]:
            cells = []
            for t in tasks:
                r = pick(runs, task=t, system=s, labels=lab, **API)
                cells.append(f"{mean_f1(r):.3f}" if r else "--")
            lines.append(f"{SYSNAME[s]} ({lab}) & " + ("0" if lab == "N" else r"0$^\dagger$") + " & " + " & ".join(cells) + r" \\")
    for s in ["qwen3-8b", "qwen3.5-9b"]:
        cells = []
        for t in tasks:
            r = pick(runs, task=t, system=s, labels="D", perm=0, subset="full")
            cells.append(f"{mean_f1(r):.3f}" if r else "--")
        lines.append(f"{SYSNAME[s]} (D) & 0 & " + " & ".join(cells) + r" \\")
    lines.append(r"\midrule")
    for s in ["tfidf", "setfit", "mmbert", "qwen0.6b", "qwen1.7b", "qwen8b"]:
        for b in ["5", "20", "full"]:
            cells, any_ = [], False
            for t in tasks:
                rs = pick(runs, task=t, family="supervised", system=s, budget=b, recipe="corrected")
                if rs:
                    any_ = True
                    v = [r["metrics"]["macro_f1"] for r in rs]
                    cells.append(f"{np.mean(v):.3f}" + (r"\textsuperscript{" + str(len(v)) + "}" if len(v) != 3 else ""))
                else:
                    cells.append("--")
            if any_:
                lines.append(f"{SYSNAME[s]} & {b if b != 'full' else 'full'}{'/class' if b != 'full' else ''} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (GEN / "table_main.tex").write_text("\n".join(lines))


def fmt_p(p):
    """2.000 repliche bootstrap: risoluzione 0,0005, quindi i valori nulli si riportano come < 0.001."""
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def tex_contrasts(con):
    order = ("P1", "P3", "P4", "P5", "S1")
    rows = sorted([c for c in con if c["status"] == "ok" and c["id"] in order], key=lambda c: order.index(c["id"]))  # stabile
    fam = {"P1": "descriptions, D $-$ N", "P3": "progenitor", "P4": "adaptation of Qwen3-8B", "P5": "recipe (prot.: protocol recipe; expl.: exploratory recipe)",
           "S1": "recipe sensitivity, all data"}

    def esc(x):
        for k, v in [("qwen0.6b", "Qwen3-0.6B"), ("qwen8b", "Qwen3-8B"), ("openai", "OpenAI"), ("elephant", "Elephant-1"),
                     ("mammoth", "Mammoth-1"), (" (4 train seeds)", ""), (" (3 seeds)", ""), ("k=full", "all data"),
                     ("original", "expl."), ("corrected", "prot."), ("all data expl.", "expl."),
                     ("all data prot.", "prot."), (" zero-shot", "")]:
            x = x.replace(k, v)
        return x.replace("_", r"\_")
    lines = [r"\begin{tabular}{lllrll}", r"\toprule", r"Data & A & B & $\Delta$ macro-F1 & 95\% CI & $p_{\mathrm{Holm}}$ \\"]
    last = None
    for c in rows:
        if c["id"] != last:
            lines += [r"\midrule", rf"\multicolumn{{6}}{{l}}{{\textit{{{c['id']}: {fam[c['id']]}}}}} \\"]
            last = c["id"]
        lines.append(f"{DATASETS[c['task']]} & {esc(c['a'])} & {esc(c['b'])} & {float(c['diff']):+.3f} & "
                     f"[{float(c['ci_lo']):+.3f}, {float(c['ci_hi']):+.3f}] & {fmt_p(float(c.get('p_holm', 1)))}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (GEN / "table_contrasts.tex").write_text("\n".join(lines))


def main(runs):
    plt = setup()
    con = list(csv.DictReader(open(ROOT / "reports" / "contrasts.csv")))
    for f in (fig_learning, fig_stress, fig_risk, fig_progenitor):
        f(plt, runs)
    fig_descriptions(plt, con)
    tex_table_main(runs)
    tex_contrasts(con)
    print("figure e tabelle scritte in", FIG.relative_to(ROOT), "e", GEN.relative_to(ROOT))
