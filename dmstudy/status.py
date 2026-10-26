"""Stato degli esperimenti e piano di calcolo misurato: reports/experiment_status.csv e reports/compute_plan.csv.

  python -m dmstudy.status
"""
import csv
import json
from collections import defaultdict

import numpy as np

from .common import RESULTS, ROOT
from .queue import plan


def measured_seconds():
    """Durate reali dai marker dei job completati (locale e, se scaricati, Colab)."""
    by = defaultdict(list)
    for p in (RESULTS / "done").glob("*.json"):
        d = json.load(open(p))
        if "seconds" in d:
            parts = p.stem.split("-")
            group = "cv" if parts[0] == "cv" else ("zero-shot LM" if parts[0] in ("zs", "probe") else parts[1] if len(parts) > 1 else parts[0])
            by[group].append(d["seconds"])
    return by


def vram():
    out = defaultdict(list)
    for p in (RESULTS / "metrics").glob("sup-*.json"):
        d = json.load(open(p))
        v = d["metrics"].get("peak_vram_gb")
        if v:
            out[(d["config"]["model"], d["config"].get("gpu", ""))].append(v)
    return {k: max(v) for k, v in out.items()}


def main():
    sec = measured_seconds()
    vr = vram()
    budget = json.load(open(RESULTS / "budget_openai.json"))["usd"] if (RESULTS / "budget_openai.json").exists() else 0
    rows = []
    for plan_name, hw in (("local", "RTX 4050 Laptop 6 GB"), ("colab", "Colab Tesla T4 16 GB")):
        jobs = plan(plan_name)
        groups = defaultdict(lambda: [0, 0])
        for key, _ in jobs:
            parts = key.split("-")
            g = "cv" if parts[0] == "cv" else ("zero-shot LM" if parts[0] in ("zs", "probe") else parts[1])
            groups[g][0] += 1
            groups[g][1] += (RESULTS / "done" / f"{key}.json").exists()
        for g, (n, done) in groups.items():
            s = sec.get(g, [])
            mean = float(np.mean(s)) if s else None
            mem = [v for (m, gpu), v in vr.items() if m == g]
            rows.append(dict(activity=g, plan=plan_name, hardware=hw, runs_planned=n, runs_done=done,
                             feasibility="verified" if done else "planned",
                             peak_vram_gb=max(mem) if mem else "", mean_seconds_per_run=round(mean, 1) if mean else "",
                             est_remaining_hours=round(mean * (n - done) / 3600, 1) if mean else "",
                             api_cost_usd=0, note="GPU time only; Colab waits and human time excluded"))
    rows.append(dict(activity="decision APIs (OpenAI)", plan="api", hardware="OpenAI Decisions", runs_planned="",
                     runs_done=len(list((RESULTS / "metrics").glob("api-openai-*.json"))), feasibility="verified",
                     api_cost_usd=round(budget, 2), note="cap 10 USD; few-shot-in-prompt on 300-example subset"))
    rows.append(dict(activity="decision APIs (Baseten)", plan="api", hardware="Baseten System One",
                     runs_done=len(list((RESULTS / "metrics").glob("api-elephant-*.json"))) +
                     len(list((RESULTS / "metrics").glob("api-mammoth-*.json"))), feasibility="verified",
                     api_cost_usd=0, note="promotional free access until 2026-10-27; not a production price"))
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(ROOT / "reports" / "compute_plan.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    for r in rows:
        print(r["activity"], r.get("plan"), f"{r.get('runs_done')}/{r.get('runs_planned')}", r.get("mean_seconds_per_run"), r.get("est_remaining_hours"))


if __name__ == "__main__":
    main()
