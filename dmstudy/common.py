"""Infrastruttura comune: dati con ID stabili, manifest degli split, provenienza dei run, salvataggio predizioni.

Ogni run scrive:
  results/predictions/<run_id>.jsonl   una riga per esempio (gold, predizione, probabilità per classe, latenza, stato)
  results/metrics/<run_id>.json        metriche + configurazione completa + provenienza
"""
import datetime as dt
import hashlib
import json
import os
import platform
import subprocess
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MANIFESTS = DATA / "manifests"
RESULTS = Path(os.environ.get("CLS_RESULTS", ROOT / "results"))  # CLS_RESULTS: cartella alternativa per le prove
TASKS = ["banking77", "massive_it", "ottavo", "ottavo_adv", "ottavo_r", "banking77_official", "massive_it_official"]
# ottavo_r: set di robustezza E07 (fenomeni specifici, casi ambigui e fuori tassonomia), sintetico, aggiunto il 24/10/2026
# ottavo_adv è la condizione "stress lessicale" dello stesso task Ottavo: stesso train, altro test.
# *_official: test ufficiali completi (versione MTEB), analisi distinta dai campioni storici da ~1.000.
TRAIN_OF = {"ottavo_adv": "ottavo", "ottavo_r": "ottavo", "banking77_official": "banking77", "massive_it_official": "massive_it"}
SEEDS = [17, 42, 123]
BUDGETS = [5, 10, 20, 50]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def read_jsonl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def labels(task):
    """Ordine canonico delle classi = ordine di labels.json (congelato)."""
    return list(json.load(open(DATA / TRAIN_OF.get(task, task) / "labels.json", encoding="utf-8")))


def descriptions(task, mode):
    """mode N: nome della classe leggibile; mode D: descrizione con regole di inclusione e confini."""
    base = TRAIN_OF.get(task, task)
    if mode == "N":
        return {l: l.replace("_", " ").strip() for l in labels(task)}
    if base == "ottavo":  # le descrizioni di Ottavo sono la specifica usata per generare i dati
        return json.load(open(DATA / "ottavo" / "labels.json", encoding="utf-8"))
    return json.load(open(DATA / "descriptions" / f"{base}_D.json", encoding="utf-8"))


def test_set(task):
    """Test storico congelato, con ID posizionali stabili."""
    rows = read_jsonl(DATA / task / "test.jsonl")
    return [{"id": f"{task}-test-{i:04d}", **r} for i, r in enumerate(rows)]


def train_pool(task):
    base = TRAIN_OF.get(task, task)
    rows = read_jsonl(DATA / base / "train_full.jsonl")
    return [{"id": f"{base}-train-{i:05d}", **r} for i, r in enumerate(rows)]


def manifest(task):
    return json.load(open(MANIFESTS / f"{TRAIN_OF.get(task, task)}.json", encoding="utf-8"))


def train_subset(task, budget, seed):
    """budget: 5/10/20/50 (esempi per classe, annidati per seed) oppure 'full'."""
    pool = {r["id"]: r for r in train_pool(task)}
    if budget == "full":
        return list(pool.values())
    ids = manifest(task)["fewshot"][str(seed)][str(budget)]
    return [pool[i] for i in ids]


def git_commit():
    if os.environ.get("DM_GIT_COMMIT"):  # su Colab il codice arriva da un pacchetto del commit indicato
        return os.environ["DM_GIT_COMMIT"]
    try:
        sha = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"], capture_output=True, text=True).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except OSError:
        return "unknown"


def new_run_id(prefix):
    return f"{prefix}-{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:6]}"


def provenance(task):
    files = [DATA / task / "test.jsonl", DATA / TRAIN_OF.get(task, task) / "train_full.jsonl"]
    return {
        "git_commit": git_commit(),
        "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "host": platform.node(),
        "python": platform.python_version(),
        "data_sha256": {str(f.relative_to(ROOT)): sha256_file(f) for f in files},
    }


def save_run(run_id, config, rows, metrics):
    """rows: lista di dict per esempio; config: tutto ciò che identifica il run."""
    write_jsonl(RESULTS / "predictions" / f"{run_id}.jsonl", rows)
    out = {"run_id": run_id, "config": config, "provenance": provenance(config["task"]), "metrics": metrics}
    p = RESULTS / "metrics" / f"{run_id}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(p, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    return p
