"""Coda di job riprendibile (GPU locale o Colab): esegue in sequenza, salta i job già completati.

  python -m dmstudy.queue --plan local      # RTX 4050
  python -m dmstudy.queue --plan colab      # T4 (lanciata dal notebook)
  python -m dmstudy.queue --plan local --list

Ogni job completato lascia results/done/<chiave>.json (comando, durata, esito): rilanciare riprende da dove era rimasta.
La selezione della ricetta (CV sul seed 17) precede i run finali, che leggono results/selection/*.json.
Prima di ogni job controlla lo spazio disco libero (minimo 50 GB).
"""
import argparse
import json
import shutil
import subprocess
import sys
import time

from .common import RESULTS, ROOT

SEEDS = [17, 42, 123]
LRS = {"mmbert": [2e-5, 5e-5], "qwen0.6b": [5e-5, 2e-4], "qwen1.7b": [5e-5, 2e-4], "qwen8b": [1e-4]}
# durate valutate (epoche) per budget: annidate grazie al learning rate costante
CKPTS = {"5": [5, 10, 20], "20": [3, 6, 12]}
FULL_EPOCHS = {"banking77": 4, "massive_it": 4, "ottavo": 6}
FULL_LR = {"mmbert": 5e-5, "qwen0.6b": 2e-4, "qwen1.7b": 2e-4, "qwen8b": 1e-4}
ALSO = {"ottavo": ["ottavo_adv", "ottavo_r"], "banking77": ["banking77_official"], "massive_it": ["massive_it_official"]}


def py(*args):
    return [sys.executable, "-m", *args]


def hf_jobs(task, model, seeds=SEEDS, budgets=("5", "20", "full"), original=False, original_seeds=SEEDS):
    jobs = []
    also = ["--also", *ALSO[task]]
    for b in budgets:
        if b == "full":
            continue
        jobs.append((f"cv-{task}-{model}-k{b}", py("dmstudy.train_hf", "--task", task, "--model", model, "--budget", b,
                                                    "--seed", "17", "--cv", "3", "--lrs", *map(str, LRS[model]),
                                                    "--checkpoints", *map(str, CKPTS[b]))))
    for s in seeds:
        for b in budgets:
            key = f"{task}-{model}-k{b}-seed{s}"
            if b == "full":
                jobs.append((key, py("dmstudy.train_hf", "--task", task, "--model", model, "--budget", "full", "--seed", str(s),
                                     "--lr", str(FULL_LR[model]), "--epochs", str(FULL_EPOCHS[task]), *also)))
            else:
                jobs.append((key, ["@selected", task, model, b, str(s), *also]))
            if original and b != "full" and s in original_seeds:
                jobs.append((key + "-original", py("dmstudy.train_hf", "--task", task, "--model", model, "--budget", b,
                                                   "--seed", str(s), "--recipe", "original", *also)))
    return jobs


def classic_jobs(task, budgets=("5", "20", "full"), seeds=SEEDS):
    also = ["--also", *ALSO[task]]
    return [(f"{task}-{m}-k{b}-seed{s}", py("dmstudy.classic", "--method", m, "--task", task, "--budget", b, "--seed", str(s), *also))
            for m in ("tfidf", "setfit") for s in seeds for b in budgets]


def plan(name):
    if name == "local":  # portatile: solo TF-IDF (CPU); i modelli su GPU girano su Colab (T4)
        return [j for task in ["ottavo", "banking77", "massive_it"] for j in classic_jobs(task) if "-tfidf-" in j[0]]
    if name == "colab":
        zs = lambda key, *a: (key, py("dmstudy.zeroshot_lm", *a))
        jobs = [
            # E04: progenitore di Elephant (prova di fattibilità, poi test completi) e controllo Qwen3-8B
            zs("probe-qwen3.5-9b", "--model", "qwen3.5-9b", "--task", "ottavo", "--probe", "5"),
            zs("zs-qwen3.5-9b-ottavo-D", "--model", "qwen3.5-9b", "--task", "ottavo", "--labels", "D"),
            zs("zs-qwen3.5-9b-ottavo_adv-D", "--model", "qwen3.5-9b", "--task", "ottavo_adv", "--labels", "D"),
            zs("zs-qwen3-8b-ottavo-D", "--model", "qwen3-8b", "--task", "ottavo", "--labels", "D"),
            zs("zs-qwen3-8b-ottavo_adv-D", "--model", "qwen3-8b", "--task", "ottavo_adv", "--labels", "D"),
            zs("zs-qwen3-8b-ottavo-N", "--model", "qwen3-8b", "--task", "ottavo", "--labels", "N"),
            zs("zs-qwen3.5-9b-ottavo-N", "--model", "qwen3.5-9b", "--task", "ottavo", "--labels", "N"),
            zs("zs-qwen3-8b-ottavo_r-D", "--model", "qwen3-8b", "--task", "ottavo_r", "--labels", "D"),
            zs("zs-qwen3.5-9b-ottavo_r-D", "--model", "qwen3.5-9b", "--task", "ottavo_r", "--labels", "D"),
        ]
        jobs += [zs(f"zs-qwen3-8b-ottavo-D-perm{p}", "--model", "qwen3-8b", "--task", "ottavo", "--labels", "D",
                    "--perm", str(p), "--subset", "api") for p in (1, 2, 3)]
        # Ottavo: tutte le famiglie supervisionate; poi i dataset pubblici
        jobs += [j for j in classic_jobs("ottavo") if "-setfit-" in j[0]]
        jobs += hf_jobs("ottavo", "mmbert") + hf_jobs("ottavo", "qwen0.6b", original=True)
        jobs += [j for j in hf_jobs("ottavo", "qwen8b", original=True, original_seeds=(42,)) if j not in jobs]
        jobs += hf_jobs("ottavo", "qwen1.7b")
        for task in ["banking77", "massive_it"]:
            jobs += [j for j in classic_jobs(task) if "-setfit-" in j[0]]
            jobs += hf_jobs(task, "mmbert") + hf_jobs(task, "qwen0.6b")
        seen, out = set(), []
        for k, c in jobs:  # niente duplicati, ordine di priorità preservato
            if k not in seen:
                seen.add(k); out.append((k, c))
        return out
    if name == "colab_sensitivity":  # E02: lo 0,940 storico di Qwen3-0.6B su Banking77 dipende dalla ricetta?
        base = ["--task", "banking77", "--model", "qwen0.6b", "--budget", "full", "--recipe", "original", "--also", "banking77_official"]
        return [(f"banking77-qwen0.6b-kfull-original-trainseed{t}",
                 py("dmstudy.train_hf", *base, "--seed", str(t if t else 42), "--train-seed", str(t)))
                for t in (0, 17, 42, 123)]
    raise SystemExit(name)


def resolve(cmd):
    if cmd[0] != "@selected":
        return cmd
    _, task, model, b, s, *also = cmd
    path = RESULTS / "selection" / f"{task}__{model}__k{b}__seed17.json"
    if not path.exists():
        return None  # la selezione della ricetta non è riuscita: il job viene saltato, non blocca la coda
    sel = json.load(open(path))
    return py("dmstudy.train_hf", "--task", task, "--model", model, "--budget", b, "--seed", s,
              "--lr", str(sel["best_lr"]), "--epochs", str(sel["best_epochs"]), *also)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    done_dir = RESULTS / "done"
    done_dir.mkdir(parents=True, exist_ok=True)
    jobs = plan(a.plan)
    todo = [(k, c) for k, c in jobs if not (done_dir / f"{k}.json").exists()]
    print(f"{len(jobs)} job, {len(jobs) - len(todo)} già completati, {len(todo)} da fare", flush=True)
    if a.list:
        for k, _ in todo:
            print(" ", k)
        return
    import os
    me = os.getpid()
    while any(p.isdigit() and int(p) != me and any(m in open(f"/proc/{p}/cmdline", "rb").read().decode(errors="ignore")
                                                    for m in ("dmstudy.train_hf", "dmstudy.classic", "dmstudy.zeroshot_lm"))
              for p in os.listdir("/proc") if os.path.exists(f"/proc/{p}/cmdline")):
        print("attendo la fine di un job ancora in esecuzione...", flush=True)
        time.sleep(30)
    for i, (key, cmd) in enumerate(todo, 1):
        free = shutil.disk_usage(ROOT).free / 1e9
        if free < 50:
            raise SystemExit(f"spazio libero {free:.0f} GB < 50 GB: mi fermo")
        cmd = resolve(cmd)
        if cmd is None:
            print(f"[{i}/{len(todo)}] {key}: SALTATO (manca la selezione della ricetta)", flush=True)
            continue
        print(f"[{i}/{len(todo)}] {key}: {' '.join(cmd[2:])}", flush=True)
        t0 = time.time()
        log = RESULTS / "logs" / f"{key}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with open(log, "w") as f:
            code = subprocess.call(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
        if code != 0 and "dmstudy.train_hf" in cmd and "NaN" in log.read_text(errors="ignore"):
            print("    NaN in fp16: ripeto in fp32", flush=True)
            with open(log, "a") as f:
                code = subprocess.call(cmd + ["--precision", "fp32"], cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
        if code != 0:  # mostra la causa direttamente nell'output della coda
            print("".join(log.read_text(errors="ignore").splitlines(keepends=True)[-12:]), flush=True)
        dur = time.time() - t0
        print(f"    {'ok' if code == 0 else 'ERRORE'} in {dur / 60:.1f} min", flush=True)
        if code == 0:
            json.dump({"cmd": cmd, "seconds": round(dur, 1), "finished": time.strftime("%F %T")},
                      open(done_dir / f"{key}.json", "w"))


if __name__ == "__main__":
    main()
