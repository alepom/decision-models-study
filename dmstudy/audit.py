"""E00: audit dello studio originale. Produce reports/audit_initial.md e reports/audit_values.csv.

  python -m dmstudy.audit [--export-provenance]

- riconcilia i valori dichiarati nel report storico con i JSON storici (results/*.json, solo metriche aggregate:
  le predizioni per esempio non erano state salvate) e con le nuove riesecuzioni a configurazione identica;
- cerca duplicati esatti e quasi duplicati fra train e test e dentro i test;
- ricostruisce la provenienza di Ottavo (prompt e modelli dei generatori) e il caso escluso 602 -> 601.
"""
import argparse
import csv
import glob
import json
import re
from pathlib import Path

import numpy as np

from .common import DATA, RESULTS, ROOT, read_jsonl, test_set, train_pool

TRANSCRIPTS = Path.home() / ".claude/projects/-home-alepom-progetti-lab-test-AP-classificatori/d1fea9b4-435b-421a-9137-8a40df42cf07/subagents"
GEN_DIR = DATA / "ottavo" / "generation"


def norm(t):
    return re.sub(r"\W+", " ", t.lower()).strip()


def export_provenance():
    """Salva i prompt dei generatori di Ottavo (senza ID interni) in data/ottavo/generation/."""
    GEN_DIR.mkdir(parents=True, exist_ok=True)
    roles = []
    for f in sorted(TRANSCRIPTS.glob("agent-*.jsonl")):
        meta = json.load(open(str(f).replace(".jsonl", ".meta.json")))
        desc = meta.get("description", "")
        if not any(k in desc for k in ("synthetic data", "Adversarial", "validator", "descriptions")):
            continue
        models, msgs = set(), []
        for line in open(f):
            d = json.loads(line)
            m = d.get("message", {})
            if m.get("model"):
                models.add(m["model"])
            if d.get("type") == "user" and isinstance(m.get("content"), str):
                msgs.append(m["content"])
        name = re.sub(r"\W+", "_", desc.lower()).strip("_")
        (GEN_DIR / f"{name}.md").write_text(
            f"# {desc}\n\nModel(s): {', '.join(sorted(models))}\n\n" +
            "\n\n---\n\n".join(f"## Message {i + 1} to the agent\n\n{c}" for i, c in enumerate(msgs)))
        roles.append({"role": desc, "models": sorted(models), "messages": len(msgs), "file": f"{name}.md"})
    json.dump(roles, open(GEN_DIR / "roles.json", "w"), indent=1)
    return roles


def excluded_case():
    raw = DATA / "ottavo" / "adv_raw"
    from .common import sha256_text  # noqa: F401
    import hashlib
    gen = {}
    for f in sorted(raw.glob("gen_*.jsonl")):
        for line in open(f):
            if line.strip():
                r = json.loads(line)
                gen[hashlib.sha1(line.encode()).hexdigest()[:8]] = (r, f.stem)
    votes = [{json.loads(l)["id"]: json.loads(l)["label"] for l in open(f) if l.strip()} for f in sorted(raw.glob("val_*.jsonl"))]
    out = []
    for i, (r, src) in gen.items():
        v = [vv.get(i) for vv in votes]
        if any(x != r["label"] for x in v):
            out.append({"generator_file": src, "text": r["text"], "generator_label": r["label"],
                        "validator_labels": v, "pair": r.get("pair")})
    agree = [float(np.mean([vv.get(i) == r["label"] for i, (r, _) in gen.items()])) for vv in votes]
    return out, len(gen), agree


def near_duplicates(task_train, task_test, thr=0.9):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import linear_kernel
    tr, te = train_pool(task_train), test_set(task_test)
    exact = len({norm(r["text"]) for r in te} & {norm(r["text"]) for r in tr})
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True).fit([r["text"] for r in tr + te])
    A, B = vec.transform([r["text"] for r in te]), vec.transform([r["text"] for r in tr])
    best = np.zeros(len(te))
    for i in range(0, len(te), 500):
        best[i:i + 500] = linear_kernel(A[i:i + 500], B).max(1)
    examples = [(te[i]["text"], float(best[i])) for i in np.argsort(-best)[:3]]
    return {"exact": exact, "near_ge_0.9": int((best >= thr).sum()), "near_ge_0.8": int((best >= 0.8).sum()),
            "n_test": len(te), "max_sim_examples": examples}


# valori centrali del report storico (artifact v7) e configurazione corrispondente
HISTORICAL = [
    ("banking77", "qwen0.6b", "full-shot", "Banking77 Qwen3-0.6B full", 0.940),
    ("banking77", "mammoth-1", "0-shot", "Banking77 Mammoth zero-shot", 0.935),
    ("banking77", "elephant-1", "0-shot", "Banking77 Elephant zero-shot", 0.927),
    ("banking77", "openai-decisions", "0-shot", "Banking77 OpenAI zero-shot (nomi)", 0.765),
    ("massive_it", "mammoth-1", "0-shot", "MASSIVE-it Mammoth zero-shot", 0.858),
    ("massive_it", "openai-decisions", "0-shot", "MASSIVE-it OpenAI zero-shot (nomi)", 0.736),
    ("ottavo", "openai-decisions", "0-shot", "Ottavo OpenAI zero-shot (descrizioni)", 0.990),
    ("ottavo", "elephant-1", "0-shot", "Ottavo Elephant zero-shot", 0.987),
    ("ottavo", "qwen8b-zeroshot", "0-shot", "Ottavo Qwen3-8B zero-shot", 0.942),
    ("ottavo", "qwen8b", "full-shot", "Ottavo Qwen3-8B QLoRA full", 0.978),
    ("ottavo_adv", "qwen8b-zeroshot", "0-shot", "Stress Qwen3-8B zero-shot", 0.905),
    ("ottavo_adv", "qwen8b", "full-shot", "Stress Qwen3-8B QLoRA full", 0.967),
    ("ottavo_adv", "elephant-1", "0-shot", "Stress Elephant zero-shot", 0.998),
    ("ottavo", "qwen8b", "5-shot", "Ottavo Qwen3-8B 5 per classe", 0.217),
]
# stessa configurazione rieseguita nel nuovo protocollo (zero-shot API: stesso prompt, ordine canonico)
RERUN = {("banking77", "mammoth-1"): ("mammoth", "N"), ("banking77", "elephant-1"): ("elephant", "N"),
         ("banking77", "openai-decisions"): ("openai", "N"), ("massive_it", "mammoth-1"): ("mammoth", "N"),
         ("massive_it", "openai-decisions"): ("openai", "N"), ("ottavo", "openai-decisions"): ("openai", "D"),
         ("ottavo", "elephant-1"): ("elephant", "D"), ("ottavo_adv", "elephant-1"): ("elephant", "D")}


def new_api_value(task, provider, mode):
    for f in glob.glob(str(RESULTS / "metrics" / f"api-{provider}-{task}-{mode}-p0-s0-full-r0-*.json")):
        return json.load(open(f))["metrics"]["macro_f1"]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export-provenance", action="store_true")
    a = ap.parse_args()
    roles = export_provenance() if a.export_provenance else json.load(open(GEN_DIR / "roles.json"))

    rows = []
    for task, system, variant, claim, reported in HISTORICAL:
        hist = json.load(open(RESULTS / "historical" / f"{task}__{system}__{variant}.json"))["macro_f1"]
        new = None
        if (task, system) in RERUN:
            new = new_api_value(task, *RERUN[(task, system)])
        status = "conferma JSON storico" if abs(hist - reported) < 0.0006 else "DIFFORME dal JSON storico"
        if new is not None:
            status += f"; riesecuzione identica {'entro 0,01' if abs(new - hist) <= 0.01 else 'DIVERSA'}"
        else:
            status += "; predizioni non salvate: storico non verificabile per esempio"
        rows.append(dict(claim=claim, reported=reported, historical_json=round(hist, 4),
                         rerun_same_config=None if new is None else round(new, 4), status=status))
    (ROOT / "reports").mkdir(exist_ok=True)
    with open(ROOT / "reports" / "audit_values.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    dups = {t: near_duplicates(b, t) for t, b in [("banking77", "banking77"), ("massive_it", "massive_it"),
                                                   ("ottavo", "ottavo"), ("ottavo_adv", "ottavo")]}
    excl, n_gen, agree = excluded_case()
    n_tests = {t: len(test_set(t)) for t in ["banking77", "massive_it", "ottavo", "ottavo_adv"]}
    hist_files = sorted(p.name for p in (RESULTS / "historical").glob("*.json"))

    md = ["# Audit of the original study (E00)", "",
          "Generated by `python -m dmstudy.audit`. Baseline: git tag `baseline-artifact-v7`.", "",
          "## 1. Evidence available", "",
          f"- Historical results: {len(hist_files)} JSON files in `results/` with aggregate metrics only "
          "(macro-F1, accuracy, ECE, latency). **Per-example predictions were not saved** in the original study, so "
          "historical numbers cannot be recomputed from predictions, bootstrapped or re-calibrated. Every analysis in the "
          "paper uses new runs with saved predictions; historical values are kept as a separate, unverified layer.",
          "- Historical ECE used 10 bins and, for the Baseten and OpenAI APIs, the API `confidence` field when present, "
          "not the probability of the predicted class. This is a methodological defect; new runs use the renormalised "
          "probability distribution and 15 bins.",
          "- No git repository existed: the baseline was committed on 2026-10-24 before any change.", "",
          "## 2. Central values: report vs historical JSON vs identical re-run", "",
          "| Claim | Report | Historical JSON | Re-run, same config | Status |", "|---|---:|---:|---:|---|"]
    for r in rows:
        md.append(f"| {r['claim']} | {r['reported']:.3f} | {r['historical_json']:.4f} | "
                  f"{'' if r['rerun_same_config'] is None else f'{r['rerun_same_config']:.4f}'} | {r['status']} |")
    md += ["", "## 3. Test set sizes", "",
           f"Tests: {n_tests}. Sum = {sum(n_tests.values())}. The original artifact cover stated '~4,500 test messages': "
           "this is an error in the original text (the declared tests sum to 3,557); no other test set was used. "
           "The paper reports 3,557 for the historical samples, plus the official full tests as a separate analysis.",
           "", "## 4. Duplicates between train and test", "",
           "Character 3-5-gram TF-IDF cosine similarity of each test message with its nearest training message.", "",
           "| Test | n | exact duplicates (normalised) | cos >= 0.9 | cos >= 0.8 |", "|---|---:|---:|---:|---:|"]
    for t, d in dups.items():
        md.append(f"| {t} | {d['n_test']} | {d['exact']} | {d['near_ge_0.9']} | {d['near_ge_0.8']} |")
    md += ["", "Near duplicates are reported, not removed: in short intent utterances (e.g. MASSIVE) high surface similarity is "
           "often legitimate. A sensitivity analysis excluding cos >= 0.9 is part of the statistical analysis.",
           "", "## 5. Provenance of Ottavo Store", "",
           "Generation prompts and the messages sent to each agent are exported to `data/ottavo/generation/`.", "",
           "| Role | Model | Messages |", "|---|---|---:|"]
    for r in roles:
        md.append(f"| {r['role']} | {', '.join(r['models'])} | {r['messages']} |")
    md += ["", "- The six training/normal-test generators were separate Claude Sonnet sub-agents, each assigned four classes "
           "(confusable classes were grouped together within one agent). They shared the same prompt template and the same "
           "label specification, so they are **not statistically independent generators**: train and normal test come from "
           "the same generators, split at random per class (40 test, up to 70 train).",
           "- The first data-group agent initially received a prompt with an unreplaced placeholder; it stopped without "
           "writing data and was re-instructed with its four classes (see the later message in its file).",
           "- The lexical-stress set was written by three further Sonnet agents (15 confusable pairs, 20 messages per side) "
           "and labelled blind by two validators (Claude Opus and Claude Sonnet). Agreement is between models, not human gold.",
           "", f"Generated stress messages: {n_gen}; validator agreement with generator labels: "
           f"{', '.join(f'{x:.1%}' for x in agree)}. Excluded cases (any validator disagreeing): {len(excl)}.", ""]
    for e in excl:
        md.append(f"- Excluded: \"{e['text']}\" — generator label `{e['generator_label']}`, validator labels "
                  f"{e['validator_labels']}, pair {e['pair']} ({e['generator_file']}).")
    md += ["", "## 6. Claims in the original report that must change", "",
           "- '~4,500 test messages' → 3,557 (historical tests).",
           "- '0.940 vs 0.935: the small model wins' → 'higher observed score'; the difference is within test-sample noise "
           "until shown otherwise (paired bootstrap).",
           "- 'Descriptions matter more than the model' (inferred from Banking77 vs Ottavo) → must be measured on the same "
           "messages (E03); the new N vs D runs do this.",
           "- 'Elephant-1 and Qwen3-8B differ only in training' → false: different base model (Qwen3.5-9B vs Qwen3-8B), "
           "hybrid attention, block-bidirectional mask, Brier-term objective (E04).",
           "- '5-shot collapse' of Qwen3 was produced by one untuned recipe (34 epochs, OneCycle, seed 0); it is a property "
           "of that recipe until the corrected recipe shows otherwise (E02).",
           "- 'Robustness' of the adversarial set → 'lexical stress set', same task and generators; not general robustness.",
           "- Costs: 'Elephant/Mammoth free' was promotional access, not a production price; 'cents of electricity' had no "
           "measurement behind it. ECE values of API systems were computed on the `confidence` field (see §1).",
           "- 'Distillation removes the labelling cost' was a proposal, never tested.", ""]
    (ROOT / "reports" / "audit_initial.md").write_text("\n".join(md))
    print("\n".join(md[:40]))


if __name__ == "__main__":
    main()
