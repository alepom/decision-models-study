"""LLM generalista usato come classificatore zero-shot (E03/E04): nessun addestramento sul task.

  python -m dmstudy.zeroshot_lm --model qwen3-8b --task ottavo --labels D [--perm 1] [--subset api]
  python -m dmstudy.zeroshot_lm --model qwen3.5-9b --task ottavo --labels D --probe 5   # prova di fattibilità

Le opzioni sono etichettate A, B, C, ... nell'ordine della permutazione; si legge la distribuzione del prossimo
token sulle sole lettere valide (distribuzione condizionata alle opzioni offerte) e la si rimappa all'ordine
canonico delle classi. Limite: lettere singole, quindi al massimo 26 classi (solo Ottavo).
Su GPU il modello è caricato a 4 bit NF4 (bitsandbytes), calcolo in bf16 se supportato, altrimenti fp16.
"""
import argparse
import json
import string
import time

import numpy as np

from .common import descriptions, labels, manifest, new_run_id, save_run, test_set
from .metrics import evaluate

MODELS = {"qwen3-8b": "Qwen/Qwen3-8B", "qwen3.5-9b": "Qwen/Qwen3.5-9B", "qwen3-0.6b": "Qwen/Qwen3-0.6B"}  # 0.6b: solo test del codice
INSTRUCTION = ("Classify the customer's message into exactly one of the categories below, based on the customer's main intent. "
               "Answer with the letter of the category only.")


def load(name):
    import torch
    from transformers import AutoTokenizer, BitsAndBytesConfig
    mid = MODELS[name]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    bf16 = dev == "cuda" and torch.cuda.is_bf16_supported(including_emulation=False)
    dtype = torch.bfloat16 if bf16 else (torch.float16 if dev == "cuda" else torch.float32)
    kw = dict(dtype=dtype)
    if dev == "cuda":
        kw.update(device_map={"": 0}, quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype))
    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(mid)
    try:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(mid, **kw)
    except (ValueError, KeyError):  # checkpoint multimodale (Qwen3.5): classe dedicata, usata solo come LM testuale
        from transformers import AutoModelForMultimodalLM
        model = AutoModelForMultimodalLM.from_pretrained(mid, **kw)
    model.eval()
    info = dict(model_id=mid, model_class=type(model).__name__, dtype=str(dtype), quantization="nf4-4bit" if dev == "cuda" else None,
                load_s=round(time.perf_counter() - t0, 1), device=torch.cuda.get_device_name() if dev == "cuda" else "cpu")
    return tok, model, dev, info


def prompt(tok, order, desc, text):
    letters = string.ascii_uppercase[:len(order)]
    cats = "\n".join(f"{l}) {c}: {desc[c]}" for l, c in zip(letters, order))
    msgs = [{"role": "user", "content": f"{INSTRUCTION}\n\nCategories:\n{cats}\n\nCustomer message:\n{text}"}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS), required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--labels", choices=["N", "D"], default="D")
    ap.add_argument("--perm", type=int, default=0)
    ap.add_argument("--subset", choices=["full", "api"], default="full")
    ap.add_argument("--probe", type=int, default=0, help="solo fattibilità: N esempi, stampa memoria e tempi")
    a = ap.parse_args()
    import torch

    lab = labels(a.task)
    if len(lab) > 26:
        raise SystemExit("più di 26 classi: lo scoring a lettera singola non è applicabile")
    m = manifest(a.task)
    order = m["permutations"][str(a.perm)]
    desc = descriptions(a.task, a.labels)
    rows = test_set(a.task)
    if a.subset == "api":
        keep = set(m["api_subset"][a.task])
        rows = [r for r in rows if r["id"] in keep]
    if a.probe:
        rows = rows[:a.probe]

    tok, model, dev, info = load(a.model)
    letter_ids = []
    for l in string.ascii_uppercase[:len(order)]:
        ids = tok.encode(l, add_special_tokens=False)
        assert len(ids) == 1, f"la lettera {l} non è un token singolo"
        letter_ids.append(ids[0])
    letter_ids = torch.tensor(letter_ids, device=dev)
    canon = [order.index(c) for c in lab]  # posizione, nell'ordine presentato, di ciascuna classe canonica
    sync = torch.cuda.synchronize if dev == "cuda" else (lambda: None)

    out, n_tokens = [], []
    with torch.no_grad():
        for r in rows:
            ids = tok(prompt(tok, order, desc, r["text"]), return_tensors="pt", add_special_tokens=False).input_ids.to(dev)
            n_tokens.append(ids.shape[1])
            sync(); t = time.perf_counter()
            logits = model(input_ids=ids, use_cache=False, logits_to_keep=1).logits[0, -1].float()
            p_presented = logits[letter_ids].softmax(-1).cpu().numpy()
            sync(); lat = (time.perf_counter() - t) * 1000
            p = p_presented[canon]
            mass = float(logits.softmax(-1)[letter_ids].sum())  # massa del prossimo token sulle lettere valide
            out.append(dict(example_id=r["id"], split=a.task, gold_label=r["label"], predicted_label=lab[int(p.argmax())],
                            probs=[float(x) for x in p], latency_ms=lat, status="ok", letter_mass=mass))
    info.update(peak_vram_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2) if dev == "cuda" else None,
                mean_prompt_tokens=float(np.mean(n_tokens)))
    if a.probe:
        from .common import RESULTS
        rep = {"probe": a.probe, **info, "latency_ms": [round(o["latency_ms"]) for o in out],
               "correct": sum(o["gold_label"] == o["predicted_label"] for o in out),
               "letter_mass": [round(o["letter_mass"], 3) for o in out]}
        (RESULTS / "feasibility").mkdir(parents=True, exist_ok=True)
        json.dump(rep, open(RESULTS / "feasibility" / f"zeroshot_{a.model}.json", "w"), indent=1)
        print(json.dumps(rep))
        return
    met = evaluate(out, lab)
    met["mean_letter_mass"] = float(np.mean([o["letter_mass"] for o in out]))
    cfg = dict(kind="zeroshot_lm", task=a.task, model=a.model, labels=a.labels, perm=a.perm, subset=a.subset,
               instruction=INSTRUCTION, label_order_presented=order, **info)
    rid = new_run_id(f"zs-{a.model}-{a.task}-{a.labels}-p{a.perm}-{a.subset}")
    save_run(rid, cfg, out, met)
    print(json.dumps({"run_id": rid, "macro_f1": met["macro_f1"], "ece15": met.get("ece15"), **info}))


if __name__ == "__main__":
    main()
