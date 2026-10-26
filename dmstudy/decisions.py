"""Decision model via API (OpenAI Decisions, Baseten System One) con predizioni per esempio.

  python -m dmstudy.decisions --provider openai --task ottavo --labels D [--perm 1] [--shots 5|20|r10]
                              [--subset api] [--repeat 1] [--dry-run]

--labels N|D   nomi delle classi oppure descrizioni (E03)
--perm p       ordine delle classi del manifest (0 = canonico)
--shots        0 = zero-shot; 5/20 = k esempi per classe del seed --shot-seed nel prompt (gli stessi di E02);
               r10 = i 10 esempi più simili (TF-IDF) estratti dal pool a 20 per classe: stesso budget di etichette, prompt corto
--subset       full = test completo; api = campione stratificato congelato (300) per le prove costose
--repeat       indice di ripetizione (stabilità dell'endpoint): cambia solo la chiave di cache

Le risposte sono in cache (results/cache), con chiave = provider + modello + corpo della richiesta + repeat:
una chiamata identica non viene mai ripagata. Le probabilità sono salvate nell'ordine canonico delle classi,
qualunque sia l'ordine presentato. Il campo `confidence` delle API è salvato ma NON usato come probabilità.
"""
import argparse
import asyncio
import hashlib
import json
import os
import time

import httpx

from .common import (RESULTS, descriptions, labels, manifest, new_run_id, read_jsonl, save_run, test_set,
                     train_subset)
from .metrics import evaluate

PROVIDERS = {
    "openai": dict(url="https://api.openai.com/v1/decisions", model="gpt-6-luna", key="OPENAI_API_KEY",
                   auth="Bearer", concurrency=16, usd_per_m_input=0.10),
    "elephant": dict(url="https://inference.baseten.co/v1/systemone", model="decision-models/baseten-elephant-9b-latest",
                     key="BASETEN_API_KEY", auth="Api-Key", concurrency=4, usd_per_m_input=None),
    "mammoth": dict(url="https://inference.baseten.co/v1/systemone", model="decision-models/baseten-mammoth-27b-latest",
                    key="BASETEN_API_KEY", auth="Api-Key", concurrency=4, usd_per_m_input=None),
}
INSTRUCTION = "Classify the user's message into the single most appropriate intent."
BUDGET_FILE = RESULTS / "budget_openai.json"
CAP_USD = float(os.environ.get("OPENAI_CAP_USD", "10"))


def examples_block(examples):
    return "\n\nLabeled examples:\n" + "\n".join(
        f"- {json.dumps(e['text'], ensure_ascii=False)} -> {e['label']}" for e in examples)


def retriever(pool, k):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import linear_kernel
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True).fit([e["text"] for e in pool])
    X = vec.transform([e["text"] for e in pool])

    def top(text):
        sims = linear_kernel(vec.transform([text]), X)[0]
        return [pool[i] for i in sims.argsort()[::-1][:k]]
    return top


def build_body(provider, order, desc, text, examples):
    instr = INSTRUCTION + (examples_block(examples) if examples else "")
    if provider == "openai":
        return {"model": PROVIDERS[provider]["model"], "input": text, "questions": [{
            "name": "intent", "type": "choice", "instructions": instr,
            "choices": [{"value": l, "description": desc[l]} for l in order]}]}
    return {"model": PROVIDERS[provider]["model"], "state": {"message": text}, "questions": {
        "intent": {"type": "choice", "instructions": instr, "criteria": {l: desc[l] for l in order}}}}


def parse(provider, body):
    """-> (choice o None, {label: prob} o None, confidence API o None, stato)"""
    if provider == "openai":
        a = body["answers"][0]
        if a.get("type") != "choice":
            return None, None, None, a.get("type", "unknown")
        return a["choice"], {p["value"]: p["probability"] for p in a.get("probabilities", [])}, a.get("confidence"), "ok"
    a = body["answers"]["intent"]
    if a.get("type", "choice") != "choice" or a.get("choice") is None:
        return None, None, None, a.get("type", "unknown")
    probs = a.get("probabilities") or {}
    if isinstance(probs, list):
        probs = {p["value"]: p["probability"] for p in probs}
    return a["choice"], probs, a.get("confidence"), "ok"


def cache_path(key):
    return RESULTS / "cache" / key[:2] / f"{key}.json"


async def call(client, sem, provider, body, repeat):
    key = hashlib.sha256(json.dumps([provider, body, repeat], sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    cp = cache_path(key)
    if cp.exists():
        c = json.load(open(cp, encoding="utf-8"))
        return c["status_code"], c["body"], c["latency_ms"], c["attempts"], True
    cfg = PROVIDERS[provider]
    async with sem:
        attempts = 0
        while True:
            attempts += 1
            t0 = time.perf_counter()
            try:
                r = await client.post(cfg["url"], json=body)
                code, text = r.status_code, r.text
            except httpx.TransportError as e:
                code, text = -1, str(e)
            lat = (time.perf_counter() - t0) * 1000
            retry_after = None
            if code in (-1, 429, 500, 502, 503, 504) and attempts < 12:
                try:
                    retry_after = float(r.headers.get("retry-after")) if code > 0 else None
                except (TypeError, ValueError):
                    pass
                await asyncio.sleep(retry_after or min(60, 2 ** attempts))
                continue
            break
    try:
        payload = json.loads(text)
    except ValueError:
        payload = {"raw": text[:2000]}
    if code in (200, 422):  # 422 = richiesta non valida (es. oltre il contesto): deterministica, si può mettere in cache
        cp.parent.mkdir(parents=True, exist_ok=True)
        json.dump({"status_code": code, "body": payload, "latency_ms": lat, "attempts": attempts},
                  open(cp, "w", encoding="utf-8"), ensure_ascii=False)
    return code, payload, lat, attempts, False


def spent():
    return json.load(open(BUDGET_FILE))["usd"] if BUDGET_FILE.exists() else 0.0


def add_spent(usd):
    BUDGET_FILE.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"usd": round(spent() + usd, 6), "updated": time.strftime("%Y-%m-%d %H:%M:%S")}, open(BUDGET_FILE, "w"))


async def run(args):
    cfg = PROVIDERS[args.provider]
    task, lab = args.task, labels(args.task)
    m = manifest(task)
    order = m["permutations"][str(args.perm)]
    desc = descriptions(task, args.labels)
    tests = test_set(task)
    if args.subset == "api":
        keep = set(m["api_subset"][task])
        tests = [r for r in tests if r["id"] in keep]
    if args.limit:
        tests = tests[:args.limit]

    shots = args.shots
    top = None
    if shots == "0":
        pool = []
    elif shots.startswith("r"):  # recupero dal pool a 20 per classe: stesso budget di etichette di --shots 20
        pool = train_subset(task, 20, args.shot_seed)
        top = retriever(pool, int(shots[1:]))
    else:
        pool = train_subset(task, int(shots), args.shot_seed)

    bodies = [build_body(args.provider, order, desc, r["text"], top(r["text"]) if top else pool) for r in tests]
    config = dict(kind="decision_api", task=task, provider=args.provider, requested_model=cfg["model"],
                  labels=args.labels, perm=args.perm, shots=shots, shot_seed=args.shot_seed if shots != "0" else None,
                  subset=args.subset, repeat=args.repeat, n=len(tests), instruction=INSTRUCTION,
                  label_order_presented=order, prompt_sha256=hashlib.sha256(json.dumps(bodies[0], sort_keys=True).encode()).hexdigest())

    headers = {"Authorization": f"{cfg['auth']} {os.environ[cfg['key']]}", "Content-Type": "application/json"}
    async with httpx.AsyncClient(headers=headers, timeout=180) as client:
        sem = asyncio.Semaphore(cfg["concurrency"])
        # stima del costo e controllo del contesto su una prima chiamata
        code, body, *_ = await call(client, sem, args.provider, bodies[0], args.repeat)
        if code == 422:
            print(f"NON FATTIBILE (422): {json.dumps(body)[:300]}")
            json.dump({"config": config, "status": "infeasible", "reason": body}, open(
                RESULTS / "metrics" / f"infeasible-{args.provider}-{task}-{args.labels}-s{shots}.json", "w"), indent=1)
            return
        if code != 200:
            raise SystemExit(f"prima chiamata fallita: {code} {json.dumps(body)[:300]}")
        if cfg["usd_per_m_input"]:
            est = body["usage"]["input_tokens"] * len(bodies) / 1e6 * cfg["usd_per_m_input"]
            print(f"stima costo: {est:.3f} $  (spesi finora {spent():.3f} $, tetto {CAP_USD} $)")
            if spent() + est > CAP_USD and not args.force:
                raise SystemExit("oltre il tetto di budget: rilancia con --force solo se autorizzato")
        if args.dry_run:
            return
        t0 = time.perf_counter()
        outs = await asyncio.gather(*(call(client, sem, args.provider, b, args.repeat) for b in bodies))
        wall = time.perf_counter() - t0

    rows, tokens, returned = [], 0, set()
    for r, (code, body, lat, attempts, cached) in zip(tests, outs):
        choice = probs = conf = None
        status = f"http_{code}"
        if code == 200:
            choice, probs, conf, status = parse(args.provider, body)
            tokens += (body.get("usage") or {}).get("input_tokens", 0) if not cached else 0
            returned.add(body.get("model"))
        vec, psum = None, None
        if probs:
            psum = sum(probs.get(l, 0.0) for l in lab)
            vec = [probs.get(l, 0.0) / psum for l in lab] if psum > 0 else None
        rows.append(dict(example_id=r["id"], split=task, gold_label=r["label"], predicted_label=choice,
                         probs=vec, prob_sum_raw=psum, api_confidence=conf, latency_ms=lat if not cached else None,
                         attempts=attempts, status=status, cached=cached))
    cost = tokens / 1e6 * cfg["usd_per_m_input"] if cfg["usd_per_m_input"] else 0.0
    if cost:
        add_spent(cost)
    config["returned_models"] = sorted(x for x in returned if x)
    metrics = evaluate(rows, lab)
    metrics.update(input_tokens_billed=tokens, cost_usd=round(cost, 4), wall_s=round(wall, 1),
                   errors=sum(r["status"] != "ok" for r in rows), cached=sum(r["cached"] for r in rows))
    run_id = new_run_id(f"api-{args.provider}-{task}-{args.labels}-p{args.perm}-s{shots}-{args.subset}-r{args.repeat}")
    save_run(run_id, config, rows, metrics)
    print(json.dumps({"run_id": run_id, **{k: metrics[k] for k in ("n", "coverage", "macro_f1", "accuracy", "cost_usd", "errors")},
                      "ece15": metrics.get("ece15")}, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=list(PROVIDERS), required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--labels", choices=["N", "D"], default="D")
    ap.add_argument("--perm", type=int, default=0)
    ap.add_argument("--shots", default="0")
    ap.add_argument("--shot-seed", type=int, default=42)
    ap.add_argument("--subset", choices=["full", "api"], default="full")
    ap.add_argument("--repeat", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
