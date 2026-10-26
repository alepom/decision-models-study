"""Baseline supervisionate leggere: TF-IDF + regressione logistica e SetFit.

  python -m dmstudy.classic --method tfidf  --task ottavo --budget 5 --seed 42 [--also ottavo_adv]
  python -m dmstudy.classic --method setfit --task ottavo --budget 5 --seed 42 [--also ottavo_adv]

TF-IDF: C in {1, 10, 100} scelto con CV stratificata a 3 fold dentro il budget (k-shot) o su full_val (full).
SetFit: ricetta congelata dai default del paper SetFit (Tunstall et al., 2022), non tarata su questi dati:
encoder intfloat/multilingual-e5-base con prefisso "query: ", max 256 token, 20 iterazioni di coppie, 1 epoca, batch 16,
testa logistica; nel regime full il fine-tuning contrastivo è limitato a 2.000 step (dichiarato nel paper).
"""
import argparse
import json
import time

import numpy as np

from .common import labels, manifest, new_run_id, save_run, test_set, train_subset
from .metrics import evaluate, macro_f1
from .train_hf import stratified_folds

E5 = "intfloat/multilingual-e5-base"


def tfidf(C):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline, make_union
    return make_pipeline(
        make_union(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True),
                   TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True)),
        LogisticRegression(C=C, max_iter=3000))


def select_C(train, val_pairs):
    scores = {}
    for C in (1, 10, 100):
        f = []
        for tr, va in val_pairs:
            clf = tfidf(C).fit([r["text"] for r in tr], [r["label"] for r in tr])
            f.append(macro_f1([r["label"] for r in va], list(clf.predict([r["text"] for r in va]))))
        scores[C] = float(np.mean(f))
    return max(scores, key=scores.get), scores


def run_tfidf(train, budget, task, seed):
    if budget == "full":
        val_ids = set(manifest(task)["full_val"][str(seed)])
        pairs = [([r for r in train if r["id"] not in val_ids], [r for r in train if r["id"] in val_ids])]
    else:
        folds = stratified_folds(train, 3, seed)
        pairs = [([train[i] for j in range(3) if j != f for i in folds[j]], [train[i] for i in folds[f]]) for f in range(3)]
    C, grid = select_C(train, pairs)
    t0 = time.perf_counter()
    clf = tfidf(C).fit([r["text"] for r in train], [r["label"] for r in train])
    train_s = time.perf_counter() - t0

    def predict(rows, lab):
        P = np.zeros((len(rows), len(lab)))
        cols = [lab.index(c) for c in clf.classes_]
        lat = []
        for i, r in enumerate(rows):
            t = time.perf_counter()
            P[i, cols] = clf.predict_proba([r["text"]])[0]
            lat.append((time.perf_counter() - t) * 1000)
        return P, lat
    return predict, train_s, {"by": "cv3" if budget != "full" else "full_val", "C": C, "grid": grid}


def run_setfit(train, budget, task, seed):
    from datasets import Dataset
    from setfit import SetFitModel, Trainer, TrainingArguments
    import torch
    model = SetFitModel.from_pretrained(E5)
    model.model_body.max_seq_length = 256  # stessa lunghezza massima degli altri modelli del confronto
    ds = Dataset.from_dict({"text": ["query: " + r["text"] for r in train], "label": [r["label"] for r in train]})
    args = TrainingArguments(batch_size=16, num_epochs=1, num_iterations=20, seed=seed,
                             max_steps=2000 if budget == "full" else -1, report_to="none")
    t0 = time.perf_counter()
    Trainer(model=model, args=args, train_dataset=ds).train()
    train_s = time.perf_counter() - t0

    def predict(rows, lab):
        texts = ["query: " + r["text"] for r in rows]
        P = np.zeros((len(rows), len(lab)))
        cols = [lab.index(c) for c in model.labels]
        P[:, cols] = np.asarray(model.predict_proba(texts, batch_size=64))
        lat = []
        for t in texts[:200]:  # latenza batch 1 su 200 richieste
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            s = time.perf_counter(); model.predict_proba([t])
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            lat.append((time.perf_counter() - s) * 1000)
        return P, lat + [None] * (len(rows) - len(lat))
    return predict, train_s, {"by": "frozen_recipe", "encoder": E5, "num_iterations": 20, "epochs": 1,
                              "max_steps": 2000 if budget == "full" else None, "max_seq_length": 256}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["tfidf", "setfit"], required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--budget", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--also", nargs="*", default=[])
    a = ap.parse_args()
    lab = labels(a.task)
    budget = a.budget if a.budget == "full" else int(a.budget)
    train = train_subset(a.task, budget, a.seed)
    predict, train_s, sel = (run_tfidf if a.method == "tfidf" else run_setfit)(train, budget, a.task, a.seed)
    for t in [a.task, *a.also]:
        rows = test_set(t)
        P, lat = predict(rows, lab)
        out = [dict(example_id=r["id"], split=t, gold_label=r["label"], predicted_label=lab[int(P[i].argmax())],
                    probs=[float(x) for x in P[i]], latency_ms=lat[i], status="ok") for i, r in enumerate(rows)]
        m = evaluate(out, lab)
        m.update(train_s=round(train_s, 1), selection=sel)
        cfg = dict(kind="supervised", task=t, trained_on_task=a.task, model=a.method, budget=a.budget, seed=a.seed,
                   recipe="corrected", train_n=len(train))
        rid = new_run_id(f"sup-{a.method}-{t}-k{a.budget}-seed{a.seed}-corrected")
        save_run(rid, cfg, out, m)
        print(json.dumps({"run_id": rid, "macro_f1": m["macro_f1"], "selection": {k: v for k, v in sel.items() if k != "grid"}}))


if __name__ == "__main__":
    main()
