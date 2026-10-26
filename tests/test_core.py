"""Controlli mirati sui punti dove un errore falserebbe i risultati (mapping delle etichette, probabilità, split)."""
import json

import numpy as np
import pytest

from dmstudy import common, metrics
from dmstudy.decisions import build_body, parse


def test_fewshot_subsets_nested_and_balanced():
    for task in ["banking77", "massive_it", "ottavo"]:
        m = common.manifest(task)
        pool = {r["id"]: r["label"] for r in common.train_pool(task)}
        for seed in map(str, common.SEEDS):
            prev = set()
            for k in common.BUDGETS:
                ids = set(m["fewshot"][seed][str(k)])
                assert prev <= ids, f"{task} seed {seed}: {k} non contiene il budget precedente"
                counts = {}
                for i in ids:
                    counts[pool[i]] = counts.get(pool[i], 0) + 1
                short = m["shortfall"][seed][str(k)]
                for lab, c in counts.items():
                    assert c == min(k, short.get(lab, k)), (task, seed, k, lab, c)
                prev = ids


def test_test_ids_stable_and_disjoint_from_train():
    for task in ["banking77", "massive_it", "ottavo", "ottavo_adv"]:
        m = common.manifest(task)
        assert m["test"][task]["ids"] == [r["id"] for r in common.test_set(task)]
        assert m["test"][task]["file_sha256"] == common.sha256_file(common.DATA / task / "test.jsonl")


def test_descriptions_cover_all_labels_in_order():
    for task in ["banking77", "massive_it", "ottavo"]:
        for mode in "ND":
            assert list(common.descriptions(task, mode)) == common.labels(task)


def test_ece_and_brier_reference_values():
    # perfettamente calibrato: confidenza 0.75 e 3 corretti su 4
    assert metrics.ece([1, 1, 1, 0], [0.75] * 4) == pytest.approx(0.0)
    # sempre sicuro e sempre sbagliato
    assert metrics.ece([0, 0], [1.0, 1.0]) == pytest.approx(1.0)
    assert metrics.brier([[1, 0], [0, 1]], [0, 1]) == pytest.approx(0.0)
    assert metrics.brier([[0.5, 0.5]], [0]) == pytest.approx(0.5)


def test_failed_calls_count_as_errors():
    rows = [dict(gold_label="a", predicted_label="a", probs=None), dict(gold_label="b", predicted_label=None, probs=None)]
    out = metrics.evaluate(rows, ["a", "b"])
    assert out["coverage"] == 0.5 and out["accuracy"] == 0.5
    assert out["macro_f1_answered"] == pytest.approx(1.0)


def test_permuted_options_keep_label_mapping():
    lab = common.labels("ottavo")
    perm = common.manifest("ottavo")["permutations"]["2"]
    desc = common.descriptions("ottavo", "D")
    body = build_body("openai", perm, desc, "ciao", [])
    assert [c["value"] for c in body["questions"][0]["choices"]] == perm
    # una risposta OpenAI con probabilità nell'ordine presentato
    fake = {"answers": [{"type": "choice", "choice": perm[0],
                         "probabilities": [{"value": l, "probability": 1.0 if i == 0 else 0.0} for i, l in enumerate(perm)]}]}
    choice, probs, _, status = parse("openai", fake)
    assert status == "ok" and choice == perm[0] and probs[perm[0]] == 1.0
    # il vettore salvato è nell'ordine canonico: la massa deve stare sull'indice canonico di perm[0]
    vec = [probs.get(l, 0.0) for l in lab]
    assert int(np.argmax(vec)) == lab.index(perm[0])


def test_baseten_parse_dict_probabilities():
    fake = {"answers": {"intent": {"type": "choice", "choice": "b", "probabilities": {"a": 0.2, "b": 0.8}, "confidence": 0.6}}}
    choice, probs, conf, status = parse("elephant", fake)
    assert (choice, probs["b"], conf, status) == ("b", 0.8, 0.6, "ok")
