"""Metriche dalle predizioni per esempio.

Convenzioni (dichiarate nel paper):
- macro-F1 sulle classi presenti nel gold del test (sklearn, zero_division=0);
- un esempio senza predizione (errore API, rifiuto) conta come errore nel punteggio "operativo";
  il punteggio "sui casi riusciti" lo esclude ed è riportato separatamente insieme alla copertura;
- confidenza = probabilità massima della distribuzione sulle classi (non il campo `confidence` delle API);
- ECE con 15 bin di uguale larghezza su (0,1]; Brier multiclasse = media_i sum_k (p_ik - y_ik)^2.
"""
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

NONE = "__none__"


def macro_f1(gold, pred):
    return float(f1_score(gold, pred, average="macro", labels=sorted(set(gold)), zero_division=0))


def ece(correct, conf, bins=15):
    correct, conf = np.asarray(correct, float), np.asarray(conf, float)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def brier(probs, gold_idx):
    p = np.asarray(probs, float)
    y = np.zeros_like(p)
    y[np.arange(len(gold_idx)), gold_idx] = 1.0
    return float(((p - y) ** 2).sum(1).mean())


def evaluate(rows, label_order):
    """rows: dict con gold_label, predicted_label (o None), probs (lista nell'ordine label_order o None)."""
    gold = [r["gold_label"] for r in rows]
    pred = [r["predicted_label"] or NONE for r in rows]
    ok = [r["predicted_label"] is not None for r in rows]
    out = {
        "n": len(rows),
        "coverage": float(np.mean(ok)),
        "macro_f1": macro_f1(gold, pred),
        "accuracy": float(accuracy_score(gold, pred)),
    }
    if not all(ok):
        g2 = [g for g, o in zip(gold, ok) if o]
        p2 = [p for p, o in zip(pred, ok) if o]
        out["macro_f1_answered"] = macro_f1(g2, p2) if g2 else None
    out["refusals"] = sum(r.get("status") == "refusal" for r in rows)
    # calibrazione sui soli casi con una distribuzione di probabilità (n_calibrated lo dichiara)
    idx = {l: i for i, l in enumerate(label_order)}
    # gold fuori tassonomia (ambiguous / out_of_scope del set di robustezza) esclusi dalla calibrazione
    with_p = [r for r in rows if r.get("probs") is not None and r["gold_label"] in idx]
    if with_p:
        P = np.asarray([r["probs"] for r in with_p], float)
        gi = [idx[r["gold_label"]] for r in with_p]
        conf = P.max(1)
        correct = [r["predicted_label"] == r["gold_label"] for r in with_p]
        out["n_calibrated"] = len(with_p)
        out["ece15"] = ece(correct, conf)
        out["brier"] = brier(P, gi)
        out["mean_confidence"] = float(conf.mean())
    lat = [r["latency_ms"] for r in rows if r.get("latency_ms") is not None]
    if lat:
        out["latency_p50_ms"] = float(np.percentile(lat, 50))
        out["latency_p95_ms"] = float(np.percentile(lat, 95))
    return out
