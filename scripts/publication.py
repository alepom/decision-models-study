"""Strumenti di pubblicazione: identificatori, citazioni, distribuzione GitHub curata, pacchetti Zenodo, controlli.

  python scripts/publication.py set --paper-doi 10.5281/zenodo.N --software-doi 10.5281/zenodo.M [--paper-record URL]
                                    [--software-record URL] [--repo-url URL] [--tag v1.0.0] [--status reserved|published]
  python scripts/publication.py render     # CITATION.cff, BibTeX, metadati del paper, README della distribuzione
  python scripts/publication.py export     # copia curata in publication/github_export/ (lista di inclusione esplicita)
  python scripts/publication.py package    # zip in publication/packages/ + SHA256SUMS.txt + publication_manifest.json
  python scripts/publication.py check      # segreti, placeholder, file grandi nei materiali pubblici

Fonte unica degli identificatori: publication/links_and_identifiers.json. I DOI riservati restano "reserved" finché
l'autore non pubblica; nessun comando qui pubblica qualcosa.
"""
import argparse
import datetime as dt
import fnmatch
import hashlib
import json
import re
import shutil
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PUB = ROOT / "publication"
IDS = PUB / "links_and_identifiers.json"
EXPORT = PUB / "github_export"
PKG = PUB / "packages"

TITLE = "When Are Decision Models Worth It? An Empirical Comparison with Small Task-Specific Classifiers"
SOFTWARE_TITLE = "dmstudy: code, protocol and per-example predictions for the decision-model study"
AUTHORS = [{"family-names": "Pomaro", "given-names": "Alessio", "affiliation": "Independent Researcher",
            "email": "info@alessiopomaro.it", "orcid": "https://orcid.org/0009-0008-6975-996X"}]  # confermato dall'autore il 24/10/2026; affiliazione ed email il 26/10/2026

# distribuzione pubblica: solo ciò che è elencato qui (pattern relativi alla radice)
INCLUDE = ["publication/huggingface/README.md",  # scheda del dataset (i dati sono già in data/)
           "dmstudy/*.py", "tests/*.py", "scripts/run_api_plan.sh", "scripts/run_api_official.sh", "scripts/run_api_r.sh",
           "scripts/make_colab_runner.py", "scripts/publication.py", "notebooks/colab_runner.ipynb", "protocol.md",
           "data/manifests/*.json", "data/descriptions/*.json", "data/*/labels.json", "data/*/test.jsonl",
           "data/*/train_full.jsonl", "data/ottavo/generation/*", "data/ottavo/CANARY.txt", "data/ottavo/adv_raw/*.jsonl", "data/ottavo_r/raw/*.jsonl",
           "data/ottavo_r/build_report.json", "results/metrics/*.json", "results/predictions/*.jsonl",
           "results/selection/*.json", "results/feasibility/*.json", "results/runs_index.csv", "results/budget_openai.json",
           "results/historical/*.json", "reports/*.md", "reports/*.csv", "paper/main.tex", "paper/references.bib",
           "paper/sections/*.tex", "paper/generated/*.tex", "paper/figures/*.pdf"]
EXCLUDE = ["results/cache/*", "*.env", "logs/*", ".venv/*", "**/__pycache__/*"]
FROM_PUBLICATION = {"README.md": "github/README.md", "README_reproduction.md": "github/README_reproduction.md",
                    "CHANGELOG.md": "github/CHANGELOG.md", "CITATION.cff": "github/CITATION.cff", "LICENSE": "github/LICENSE",
                    "LICENSES.md": "github/LICENSES.md", "THIRD_PARTY_NOTICES.md": "github/THIRD_PARTY_NOTICES.md",
                    ".gitignore": "github/.gitignore", ".env.example": "github/.env.example",
                    "requirements.lock": "github/requirements.lock", "RELEASE_NOTES_v1.0.0.md": "github/RELEASE_NOTES_v1.0.0.md"}
SECRET_PATTERNS = [r"sk-[A-Za-z0-9_-]{20,}", r"Api-Key [A-Za-z0-9]{16,}", r"BASETEN_API_KEY=[A-Za-z0-9]", r"OPENAI_API_KEY" + r"=sk"]  # spezzato per non segnalare questa riga
PLACEHOLDER_PATTERNS = [r"pending\]", r"zenodo\.N\b", r"zenodo\.M\b", r"\bTBD\b", r"XXXX", r"<DOI>", r"To be written from the final results"]


def load_ids():
    if IDS.exists():
        return json.load(open(IDS))
    return {"paper": {"doi": None, "record_url": None, "status": "not_created", "version": "1.0.0"},
            "software": {"doi": None, "record_url": None, "status": "not_created", "version": "1.0.0"},
            "github": {"repo_url": None, "tag": "v1.0.0", "commit": None, "release_url": None},
            "updated": None}


def save_ids(d):
    d["updated"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    PUB.mkdir(exist_ok=True)
    json.dump(d, open(IDS, "w"), indent=1)


def cmd_set(a):
    d = load_ids()
    for obj, doi, rec in (("paper", a.paper_doi, a.paper_record), ("software", a.software_doi, a.software_record)):
        if doi:
            if not re.fullmatch(r"10\.\d{4,9}/\S+", doi):
                raise SystemExit(f"DOI non valido: {doi}")
            d[obj]["doi"] = doi
            d[obj]["status"] = a.status
        if rec:
            d[obj]["record_url"] = rec
    if a.repo_url:
        d["github"]["repo_url"] = a.repo_url
    if a.tag:
        d["github"]["tag"] = a.tag
    if a.commit:
        d["github"]["commit"] = a.commit
    if d["paper"]["doi"] and d["paper"]["doi"] == d["software"]["doi"]:
        raise SystemExit("paper e software devono avere DOI diversi")
    save_ids(d)
    print(json.dumps(d, indent=1))


def cmd_render(a):
    d = load_ids()
    pdoi, sdoi, repo = d["paper"]["doi"], d["software"]["doi"], d["github"]["repo_url"]
    today = dt.date.today().isoformat()
    # CITATION.cff del software, con preferred-citation al paper
    cff = ["cff-version: 1.2.0", "message: \"If you use this software, please cite the paper (preferred citation).\"",
           f"title: \"{SOFTWARE_TITLE}\"", "type: software", f"version: \"{d['software']['version']}\"",
           f"date-released: \"{today}\"", "license: MIT", "authors:"]
    cff += [f"  - family-names: \"{x['family-names']}\"\n    given-names: \"{x['given-names']}\"\n"
            f"    affiliation: \"{x['affiliation']}\"\n    email: \"{x['email']}\"\n    orcid: \"{x['orcid']}\"" for x in AUTHORS]
    if repo:
        cff.append(f"repository-code: \"{repo}\"")
    if sdoi:
        cff.append(f"doi: \"{sdoi}\"")
    cff += ["preferred-citation:", "  type: report", f"  title: \"{TITLE}\"", "  authors:"]
    cff += [f"    - family-names: \"{x['family-names']}\"\n      given-names: \"{x['given-names']}\"\n"
            f"      orcid: \"{x['orcid']}\"" for x in AUTHORS]
    cff += [f"  year: {dt.date.today().year}", "  publisher:", "    name: \"Zenodo\""]
    if pdoi:
        cff.append(f"  doi: \"{pdoi}\"")
    (PUB / "github").mkdir(parents=True, exist_ok=True)
    (PUB / "github" / "CITATION.cff").write_text("\n".join(cff) + "\n")
    names = " and ".join(f"{x['family-names']}, {x['given-names']}" for x in AUTHORS)
    (PUB / "metadata").mkdir(parents=True, exist_ok=True)
    (PUB / "metadata" / "paper_citation.bib").write_text(
        f"@techreport{{pomaro{dt.date.today().year}decision,\n  title = {{{TITLE}}},\n  author = {{{names}}},\n"
        f"  year = {{{dt.date.today().year}}},\n  institution = {{Zenodo}},\n  type = {{Preprint (not peer reviewed)}},\n"
        + (f"  doi = {{{pdoi}}},\n" if pdoi else "") + "}\n")
    (PUB / "metadata" / "software_citation.bib").write_text(
        f"@software{{pomaro{dt.date.today().year}dmstudy,\n  title = {{{SOFTWARE_TITLE}}},\n  author = {{{names}}},\n"
        f"  year = {{{dt.date.today().year}}},\n  version = {{{d['software']['version']}}},\n  publisher = {{Zenodo}},\n"
        + (f"  doi = {{{sdoi}}},\n" if sdoi else "") + (f"  url = {{{repo}}},\n" if repo else "") + "}\n")
    # metadati del paper
    gen = ROOT / "paper" / "generated"
    avail = ["Code, protocol, split manifests, category descriptions, synthetic data and per-example predictions are released "
             "under the licences listed in the repository (MIT for code; CC BY 4.0 for the paper, documentation, synthetic data and "
             "predictions; the derived copies of Banking77 and MASSIVE keep their CC BY 4.0 licences)."]
    if repo:
        avail.append(f"Repository: \\url{{{repo}}} (tag \\texttt{{{d['github']['tag']}}}).")
    if sdoi:
        avail.append(f"Archived software and materials: \\href{{https://doi.org/{sdoi}}}{{doi:{sdoi}}}.")
    avail.append("Cached API request/response bodies are not redistributed; the public data needed to recompute every reported "
                 "metric are the per-example predictions in \\texttt{results/predictions}.")
    (gen / "availability.tex").write_text(" ".join(avail) + "\n")
    meta = (gen / "meta.tex").read_text()
    line = f"Preprint, version {d['paper']['version']}" + (f" --- doi:{pdoi}" if pdoi else "") + " --- \\today"
    meta = re.sub(r"\\newcommand\{\\PaperVersionLine\}\{.*\}", lambda m: "\\newcommand{\\PaperVersionLine}{" + line + "}", meta)
    (gen / "meta.tex").write_text(meta)
    print("CITATION.cff, BibTeX, availability e meta del paper aggiornati")


def included_files():
    files = set()
    for pat in INCLUDE:
        files |= {p for p in ROOT.glob(pat) if p.is_file()}
    rel = sorted(str(p.relative_to(ROOT)) for p in files)
    return [f for f in rel if not any(fnmatch.fnmatch(f, x) for x in EXCLUDE)]


def cmd_export(a):
    keep = PUB / ".github_export_git"  # la storia git del repository pubblicato sopravvive alla rigenerazione
    if (EXPORT / ".git").exists():
        shutil.move(EXPORT / ".git", keep)
    if EXPORT.exists():
        shutil.rmtree(EXPORT)
    files = included_files()
    for f in files:
        (EXPORT / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / f, EXPORT / f)
    for dst, src in FROM_PUBLICATION.items():
        if (PUB / src).exists():
            shutil.copy2(PUB / src, EXPORT / dst)
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    (EXPORT / "EXPORT_SOURCE.json").write_text(json.dumps(
        {"source_commit": commit, "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
         "files": len(files) + len(FROM_PUBLICATION), "include": INCLUDE, "exclude": EXCLUDE}, indent=1))
    if keep.exists():
        shutil.move(keep, EXPORT / ".git")
    print(f"{len(files)} file del laboratorio + {len(FROM_PUBLICATION)} file di distribuzione in {EXPORT.relative_to(ROOT)}")


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def zipdir(out, base, paths):
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(paths):
            z.write(p, p.relative_to(base))


def cmd_package(a):
    if not EXPORT.exists():
        raise SystemExit("prima: python scripts/publication.py export")
    PKG.mkdir(parents=True, exist_ok=True)
    allf = [p for p in EXPORT.rglob("*") if p.is_file() and ".git" not in p.relative_to(EXPORT).parts[:1]]
    paper = [p for p in allf if str(p.relative_to(EXPORT)).startswith("paper/")]
    results = [p for p in allf if str(p.relative_to(EXPORT)).startswith(("results/", "reports/"))]
    zipdir(PKG / "paper_sources.zip", EXPORT, paper)
    zipdir(PKG / "software_release.zip", EXPORT, allf)
    zipdir(PKG / "research_results.zip", EXPORT, results)
    pdf = ROOT / "paper" / "main.pdf"
    if pdf.exists():
        shutil.copy2(pdf, PKG / "decision_models_study.pdf")
    arts = sorted(p for p in PKG.iterdir() if p.name not in ("SHA256SUMS.txt",))
    (PKG / "SHA256SUMS.txt").write_text("".join(f"{sha256(p)}  {p.name}\n" for p in arts))
    d = load_ids()
    dest = {"decision_models_study.pdf": ("Zenodo paper record", "CC BY 4.0"),
            "paper_sources.zip": ("Zenodo paper record", "CC BY 4.0 (text, figures); MIT (build scripts)"),
            "software_release.zip": ("Zenodo software record + GitHub release asset", "MIT (code); CC BY 4.0 (docs, synthetic data); third-party data under their own licences"),
            "research_results.zip": ("Zenodo software record (results supplement)", "CC BY 4.0")}
    manifest = {"created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                "source_commit": json.load(open(EXPORT / "EXPORT_SOURCE.json"))["source_commit"],
                "identifiers": d,
                "files": [{"file": p.name, "bytes": p.stat().st_size, "sha256": sha256(p),
                           "destination": dest.get(p.name, ("", ""))[0], "license": dest.get(p.name, ("", ""))[1]} for p in arts]}
    json.dump(manifest, open(PUB / "publication_manifest.json", "w"), indent=1)
    for f in manifest["files"]:
        print(f"{f['file']:32s} {f['bytes'] / 1e6:7.1f} MB  {f['sha256'][:16]}…  -> {f['destination']}")


def cmd_check(a):
    roots = [EXPORT] if EXPORT.exists() else [ROOT]
    problems = []
    for base in roots:
        for p in base.rglob("*"):
            if not p.is_file() or ".venv" in p.parts or "cache" in p.parts:
                continue
            if p.stat().st_size > 50e6:
                problems.append(f"file grande: {p.relative_to(base)}")
            if p.suffix in (".json", ".jsonl", ".md", ".py", ".tex", ".cff", ".txt", ".csv", ".sh", ".ipynb", ".bib", ""):
                text = p.read_text(errors="ignore")
                for pat in SECRET_PATTERNS:
                    if re.search(pat, text):
                        problems.append(f"possibile segreto ({pat}) in {p.relative_to(base)}")
                if p.suffix in (".md", ".tex", ".cff", ".bib") and "publication" not in p.parts:
                    for pat in PLACEHOLDER_PATTERNS:
                        if re.search(pat, text):
                            problems.append(f"placeholder ({pat}) in {p.relative_to(base)}")
    print("\n".join(problems) if problems else "nessun problema trovato")
    return problems


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("set")
    for x in ("--paper-doi", "--software-doi", "--paper-record", "--software-record", "--repo-url", "--tag", "--commit"):
        s.add_argument(x)
    s.add_argument("--status", choices=["reserved", "published"], default="reserved")
    for c in ("render", "export", "package", "check"):
        sub.add_parser(c)
    a = ap.parse_args()
    {"set": cmd_set, "render": cmd_render, "export": cmd_export, "package": cmd_package, "check": cmd_check}[a.cmd](a)


if __name__ == "__main__":
    main()
