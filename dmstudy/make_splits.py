"""Congela gli split del protocollo (E01) in data/manifests/<task>.json.

- test: i test storici restano quelli già usati (ID posizionali), con hash;
- fewshot[seed][k]: ID del train, k esempi per classe, ANNIDATI (5 ⊂ 10 ⊂ 20 ⊂ 50) e identici fra metodi;
  se una classe ha meno di k esempi li prende tutti, senza duplicare, e lo registra in `shortfall`;
- full_val[seed]: 10% stratificato del train completo, per selezionare la ricetta nel regime full;
- api_subset[task]: campione stratificato di 300 messaggi del test (seed 42) per le prove API costose
  (esempi nel prompt, permutazioni dell'ordine delle classi);
- permutations[p]: ordini delle classi per la sensibilità all'ordine (p = 0 è l'ordine canonico).

  python -m dmstudy.make_splits
"""
import json
import random
from collections import defaultdict

from .common import BUDGETS, MANIFESTS, SEEDS, labels, sha256_file, test_set, train_pool, DATA

API_SUBSET = 300


def by_class(rows):
    d = defaultdict(list)
    for r in rows:
        d[r["label"]].append(r["id"])
    return d


def stratified(rows, n, seed):
    rng = random.Random(seed)
    groups = by_class(rows)
    total = len(rows)
    quota = {l: max(1, round(n * len(ids) / total)) for l, ids in groups.items()}
    while sum(quota.values()) > n:  # aggiusta l'arrotondamento togliendo dalle classi più grandi
        quota[max(quota, key=lambda l: (quota[l], l))] -= 1
    out = []
    for l in sorted(groups):
        ids = groups[l][:]
        rng.shuffle(ids)
        out += ids[:quota[l]]
    return sorted(out)


def main():
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    for base, tests in [("banking77", ["banking77", "banking77_official"]), ("massive_it", ["massive_it", "massive_it_official"]),
                        ("ottavo", ["ottavo", "ottavo_adv"] + (["ottavo_r"] if (DATA / "ottavo_r" / "test.jsonl").exists() else []))]:
        pool = train_pool(base)
        groups = by_class(pool)
        m = {"task": base, "label_order": labels(base), "seeds": SEEDS, "budgets": BUDGETS,
             "train_file_sha256": sha256_file(DATA / base / "train_full.jsonl"),
             "train_size": len(pool), "fewshot": {}, "shortfall": {}, "full_val": {},
             "test": {}, "api_subset": {}, "permutations": {}}
        for seed in SEEDS:
            rng = random.Random(seed)
            order = {l: rng.sample(ids, len(ids)) for l, ids in sorted(groups.items())}
            m["fewshot"][str(seed)] = {str(k): sorted(i for l in sorted(order) for i in order[l][:k]) for k in BUDGETS}
            m["shortfall"][str(seed)] = {str(k): {l: len(ids) for l, ids in groups.items() if len(ids) < k} for k in BUDGETS}
            val = stratified(pool, round(0.1 * len(pool)), seed)
            m["full_val"][str(seed)] = val
        for t in tests:
            rows = test_set(t)
            m["test"][t] = {"file_sha256": sha256_file(DATA / t / "test.jsonl"), "n": len(rows),
                            "ids": [r["id"] for r in rows]}
            m["api_subset"][t] = stratified(rows, API_SUBSET, 42)
        lab = labels(base)
        m["permutations"]["0"] = lab
        for p in (1, 2, 3):
            m["permutations"][str(p)] = random.Random(1000 + p).sample(lab, len(lab))
        json.dump(m, open(MANIFESTS / f"{base}.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
        sizes = {k: len(v) for k, v in m["fewshot"]["42"].items()}
        print(f"{base}: train={len(pool)} fewshot(seed42)={sizes} val={len(m['full_val']['42'])} "
              f"api_subset={ {t: len(v) for t, v in m['api_subset'].items()} } "
              f"shortfall50={len(m['shortfall']['42']['50'])} classi")


if __name__ == "__main__":
    main()
