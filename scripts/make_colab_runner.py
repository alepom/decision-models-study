"""Genera notebooks/colab_runner.ipynb: esegue `python -m dmstudy.queue --plan colab` sugli stessi moduli del repository.

Il codice e i dati necessari sono incorporati (zip base64) dal commit corrente; i risultati vanno su Google Drive
(riprendibili) e alla fine viene scaricato uno zip. Uso: python scripts/make_colab_runner.py
"""
import base64
import io
import json
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
import sys
PLAN = sys.argv[1] if len(sys.argv) > 1 else "colab"
OUT = ROOT / "notebooks" / ("colab_runner.ipynb" if PLAN == "colab" else f"{PLAN}.ipynb")
FILES = [*[str(p.relative_to(ROOT)) for p in (ROOT / "results" / "done").glob("ottavo-setfit-*.json")], "protocol.md", *[str(p.relative_to(ROOT)) for p in (ROOT / "dmstudy").glob("*.py")],
         *[str(p.relative_to(ROOT)) for p in (ROOT / "data" / "manifests").glob("*.json")],
         *[str(p.relative_to(ROOT)) for p in (ROOT / "data" / "descriptions").glob("*.json")]]
for t in ["ottavo", "ottavo_adv", "ottavo_r", "banking77", "massive_it", "banking77_official", "massive_it_official"]:
    FILES += [f"data/{t}/{f}" for f in ["labels.json", "test.jsonl", "train_full.jsonl"] if (ROOT / "data" / t / f).exists()]


def cell(kind, src):
    c = {"cell_type": kind, "metadata": {}, "source": src.strip("\n").splitlines(keepends=True)}
    if kind == "code":
        c.update(execution_count=None, outputs=[])
    return c


def main():
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for f in FILES:  # i marker dei job già fatti in locale non devono creare una cartella results/ reale
            z.write(ROOT / f, f.replace("results/done/", "results_local_done/"))
    b64 = base64.b64encode(buf.getvalue()).decode()
    cells = [
        cell("markdown", f"""
# Decision model study: GPU jobs on Colab

Runs the Colab part of the experiment plan (`python -m dmstudy.queue --plan colab`) with the same code as the repository
(commit `{commit[:12]}`). Steps: **Runtime → Change runtime type → T4 GPU**, then **Runtime → Run all**,
accept the Google Drive request. Results are saved to `MyDrive/test_classificatori/v2_results` after every job:
if the session stops, open the notebook again and *Run all*; finished jobs are skipped.
At the end a `dm_results_v2.zip` is downloaded.
"""),
        cell("code", """
import subprocess
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True, text=True).stdout or "NO GPU")
"""),
        cell("code", """
!pip uninstall -y -q torchao  # versione preinstallata incompatibile con peft; non serve allo studio\n!pip install -q transformers==5.20.2 peft==0.21.2 accelerate==1.15.0 bitsandbytes==0.50.2 scikit-learn==1.9.1 setfit==1.2.0 sentence-transformers==6.2.0 datasets==5.1.0
"""),
        cell("code", f"""
import base64, io, zipfile, os
os.makedirs("/content/dm", exist_ok=True)
zipfile.ZipFile(io.BytesIO(base64.b64decode("{b64}"))).extractall("/content/dm")
os.environ["DM_GIT_COMMIT"] = "{commit}"
print(len(os.listdir("/content/dm/dmstudy")), "moduli")
"""),
        cell("code", """
from google.colab import drive
drive.mount("/content/drive")
R = "/content/drive/MyDrive/test_classificatori/v2_results"
os.makedirs(R, exist_ok=True)
if not os.path.islink("/content/dm/results"):
    os.symlink(R, "/content/dm/results")
import shutil, glob
os.makedirs(R + "/done", exist_ok=True)
for f in glob.glob("/content/dm/results_local_done/*.json"):
    if not os.path.exists(R + "/done/" + os.path.basename(f)):
        shutil.copy(f, R + "/done/")
print("risultati su", R, "| job completati:", len(os.listdir(R + "/done")) if os.path.exists(R + "/done") else 0)
"""),
        cell("code", f"""
import subprocess, sys
p = subprocess.Popen([sys.executable, "-m", "dmstudy.queue", "--plan", "{PLAN}"], cwd="/content/dm",
                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=dict(os.environ))
for line in p.stdout:
    print(line, end="", flush=True)
print("FINITO" if p.wait() == 0 else "INTERROTTO")
"""),
        cell("code", """
import shutil
from google.colab import files
shutil.make_archive("/content/dm_results_v2", "zip", R)
files.download("/content/dm_results_v2.zip")
"""),
    ]
    nb = {"cells": cells, "metadata": {"accelerator": "GPU", "kernelspec": {"name": "python3", "display_name": "Python 3"}},
          "nbformat": 4, "nbformat_minor": 0}
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(nb, indent=1))
    print(f"{OUT.relative_to(ROOT)}: {OUT.stat().st_size / 1e6:.1f} MB, commit {commit[:12]}, {len(FILES)} file")


if __name__ == "__main__":
    main()
