"""Classificatori con testa di classificazione (mmBERT, Qwen3 LoRA/QLoRA): training, selezione della ricetta, predizioni.

  # run finale con ricetta data
  python -m dmstudy.train_hf --task ottavo --model qwen0.6b --budget 5 --seed 42 --lr 2e-4 --epochs 8 [--also ottavo_adv]
  # regime full: sceglie l'epoca sulla validation interna (full_val[seed]) e predice con lo stato migliore
  python -m dmstudy.train_hf --task ottavo --model qwen0.6b --budget full --seed 42 --lr 2e-4 --epochs 4
  # ricetta storica dello studio originale (riferimento per E02)
  python -m dmstudy.train_hf --task ottavo --model qwen0.6b --budget 5 --seed 42 --recipe original
  # cross-validation interna al budget (selezione della ricetta): stampa e salva macro-F1 per lr x epoca
  python -m dmstudy.train_hf --task ottavo --model qwen0.6b --budget 5 --seed 17 --cv 3 --lrs 5e-5 2e-4 --checkpoints 3 8 20

Ricetta "corrected": AdamW, warm-up lineare per 10 step poi learning rate COSTANTE, così la valutazione a epoche
diverse dello stesso training equivale a training di durata diversa (durate annidate). Ricetta "original":
OneCycle, epoche = clamp(ceil(4000/n), 3, 40), seed 0, come nello studio iniziale.
"""
import argparse
import copy
import json
import math
import os
import random
import time

import numpy as np

from .common import RESULTS, labels, new_run_id, save_run, test_set, train_pool, train_subset, manifest
from .metrics import evaluate, macro_f1

MODELS = {
    "mmbert": dict(id="jhu-clsp/mmBERT-base", lora=False, bs=32, lr_original=5e-5),
    "qwen0.6b": dict(id="Qwen/Qwen3-0.6B", lora=True, bs=16, lr_original=2e-4),
    # sulla T4 (niente bf16, pesi in fp32) il batch da 16 non entra: micro-batch 4 x accumulo 4 + gradient checkpointing
    "qwen1.7b": dict(id="Qwen/Qwen3-1.7B", lora=True, bs=16, lr_original=2e-4, nobf16_micro_bs=4, nobf16_grad_ckpt=True),
    "qwen8b": dict(id="Qwen/Qwen3-8B", lora=True, bs=8, lr_original=1e-4, bits4=True, grad_ckpt=True),
}
MAX_LEN = 256


def set_seed(seed):
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


class Classifier:
    def __init__(self, name, lab, seed):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        cfg = MODELS[name]
        self.cfg, self.lab, self.l2i = cfg, lab, {l: i for i, l in enumerate(lab)}
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        bf16 = self.dev == "cuda" and torch.cuda.is_bf16_supported(including_emulation=False) and not os.environ.get("CLS_FORCE_FP16")
        self.amp = None if self.dev == "cpu" or os.environ.get("CLS_PRECISION") == "fp32" else (torch.bfloat16 if bf16 else torch.float16)
        # pesi: bf16 dove supportato per i LoRA; su T4 fp32 se entrano, fp16 per l'8B (comunque a 4 bit)
        # micro-batch: il batch effettivo (bs) resta invariato grazie all'accumulo dei gradienti
        self.micro_bs = int(os.environ.get("CLS_MICRO_BS") or (cfg.get("nobf16_micro_bs") if self.dev == "cuda" and not bf16 else 0)
                            or cfg["bs"])
        grad_ckpt = cfg.get("grad_ckpt") or (cfg.get("nobf16_grad_ckpt") and self.dev == "cuda" and not bf16)
        w = torch.float32
        if cfg["lora"] and bf16:
            w = torch.bfloat16
        if cfg.get("bits4") and self.dev == "cuda":
            w = torch.bfloat16 if bf16 else torch.float16
        set_seed(seed)
        t0 = time.perf_counter()
        self.tok = AutoTokenizer.from_pretrained(cfg["id"])
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        kw = {}
        if cfg.get("bits4") and self.dev == "cuda":
            from transformers import BitsAndBytesConfig
            kw = dict(device_map={"": 0}, quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=w))
        model = AutoModelForSequenceClassification.from_pretrained(cfg["id"], num_labels=len(lab), dtype=w, **kw)
        model.config.pad_token_id = self.tok.pad_token_id
        if kw:
            from peft import prepare_model_for_kbit_training
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        elif grad_ckpt:
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()
        if cfg["lora"]:
            from peft import LoraConfig, get_peft_model
            model = get_peft_model(model, LoraConfig(
                task_type="SEQ_CLS", r=16, lora_alpha=32, lora_dropout=0.05,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
            for p in model.parameters():
                if p.requires_grad:
                    p.data = p.data.float()
        if not kw:
            model.to(self.dev)
        self.model = model
        self.load_s = time.perf_counter() - t0
        self.dtype = str(w)

    def head_params(self):
        return {n: p for n, p in self.model.named_parameters() if p.requires_grad and ("score" in n or "classifier" in n)}

    def batch(self, rows):
        enc = self.tok([r["text"] for r in rows], padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
        return {k: v.to(self.dev) for k, v in enc.items()}

    def logits(self, rows):
        import torch
        with torch.autocast(self.dev, dtype=self.amp or torch.float32, enabled=self.amp is not None):
            return self.model(**self.batch(rows)).logits.float()

    def predict(self, rows, bs=None, timing=False):
        import torch
        self.model.eval()
        sync = torch.cuda.synchronize if self.dev == "cuda" else (lambda: None)
        probs, lat = [], []
        bs = bs or max(8, self.micro_bs * 2)
        with torch.no_grad():
            if timing:
                for r in rows[:10]:
                    self.logits([r])
                for r in rows:  # batch 1: latenza per richiesta singola
                    sync(); t = time.perf_counter()
                    probs.append(self.logits([r]).softmax(-1)[0].cpu().numpy())
                    sync(); lat.append((time.perf_counter() - t) * 1000)
            else:
                for i in range(0, len(rows), bs):
                    probs += list(self.logits(rows[i:i + bs]).softmax(-1).cpu().numpy())
        self.model.train()
        return np.asarray(probs), (lat or None)

    def fit(self, train, lr, epochs, recipe, seed, eval_fn=None, eval_at=None):
        """eval_fn(epoch) chiamata alle epoche in eval_at; ritorna la storia per epoca."""
        import torch
        set_seed(seed)
        bs = self.cfg["bs"]
        steps_per_epoch = math.ceil(len(train) / bs)
        params = [p for p in self.model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
        total = steps_per_epoch * epochs
        if recipe == "original":
            sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=total, pct_start=0.1)
        else:
            warm = min(10, total)
            sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm))
        scaler = torch.amp.GradScaler(self.dev, enabled=self.amp == torch.float16)
        head0 = {n: p.detach().clone() for n, p in self.head_params().items()}
        g = torch.Generator().manual_seed(seed)
        hist = []
        self.model.train()
        for ep in range(1, epochs + 1):
            order = torch.randperm(len(train), generator=g).tolist()
            tot = 0.0
            for i in range(0, len(order), bs):
                rows = [train[j] for j in order[i:i + bs]]
                loss_val = 0.0
                for k in range(0, len(rows), self.micro_bs):  # accumulo: media sul batch effettivo
                    part = rows[k:k + self.micro_bs]
                    y = torch.tensor([self.l2i[r["label"]] for r in part], device=self.dev)
                    loss = torch.nn.functional.cross_entropy(self.logits(part), y) * len(part) / len(rows)
                    scaler.scale(loss).backward()
                    loss_val += loss.item()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(opt); scaler.update(); sched.step(); opt.zero_grad()
                tot += loss_val * len(rows)
            h = {"epoch": ep, "train_loss": tot / len(train)}
            if not math.isfinite(h["train_loss"]):
                raise RuntimeError("NaN loss")
            if eval_fn and ep in (eval_at or []):
                h.update(eval_fn(ep))
            hist.append(h)
            print(json.dumps(h), flush=True)
        delta = sum(float((p.detach() - head0[n]).abs().sum()) for n, p in self.head_params().items())
        return hist, {"head_param_change_l1": delta, "head_params": list(head0)}


def stratified_folds(rows, k, seed):
    rng = random.Random(seed)
    by = {}
    for i, r in enumerate(rows):
        by.setdefault(r["label"], []).append(i)
    folds = [[] for _ in range(k)]
    for l in sorted(by):
        idx = by[l][:]
        rng.shuffle(idx)
        for j, i in enumerate(idx):
            folds[j % k].append(i)
    return folds


def rows_out(model_probs, tests, lab, lat):
    out = []
    for i, r in enumerate(tests):
        p = model_probs[i]
        out.append(dict(example_id=r["id"], split=r["split"], gold_label=r["label"], predicted_label=lab[int(p.argmax())],
                        probs=[float(x) for x in p], latency_ms=lat[i] if lat else None, status="ok"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--model", choices=list(MODELS), required=True)
    ap.add_argument("--budget", required=True, help="5|10|20|50|full")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--recipe", choices=["corrected", "original"], default="corrected")
    ap.add_argument("--lr", type=float)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--also", nargs="*", default=[])
    ap.add_argument("--cv", type=int, default=0)
    ap.add_argument("--lrs", type=float, nargs="*")
    ap.add_argument("--checkpoints", type=int, nargs="*")
    ap.add_argument("--no-timing", action="store_true")
    ap.add_argument("--train-seed", type=int, default=None, help="solo --recipe original: seed di training (default 0 come nello studio iniziale)")
    ap.add_argument("--precision", choices=["auto", "fp32"], default="auto", help="fp32: niente mixed precision (fallback per NaN)")
    args = ap.parse_args()
    if args.precision == "fp32":
        os.environ["CLS_PRECISION"] = "fp32"

    import torch
    task, lab = args.task, labels(args.task)
    budget = args.budget if args.budget == "full" else int(args.budget)
    train = train_subset(task, budget, args.seed)
    cfg = MODELS[args.model]
    base = dict(task=task, model=args.model, model_id=cfg["id"], budget=args.budget, seed=args.seed, recipe=args.recipe,
                max_len=MAX_LEN, batch_size=cfg["bs"], train_n=len(train),
                torch=torch.__version__, gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu")

    # --- selezione della ricetta: CV interna al budget
    if args.cv:
        folds = stratified_folds(train, args.cv, args.seed)
        res = {}
        for lr in args.lrs:
            for f in range(args.cv):
                va = [train[i] for i in folds[f]]
                tr = [train[i] for j in range(args.cv) if j != f for i in folds[j]]
                clf = Classifier(args.model, lab, args.seed)
                ev = lambda ep: {"val_macro_f1": macro_f1([r["label"] for r in va],
                                                          [lab[int(p.argmax())] for p in clf.predict(va)[0]])}
                hist, _ = clf.fit(tr, lr, max(args.checkpoints), "corrected", args.seed, ev, args.checkpoints)
                for h in hist:
                    if "val_macro_f1" in h:
                        res.setdefault(f"{lr}|{h['epoch']}", []).append(h["val_macro_f1"])
                del clf; torch.cuda.empty_cache()
        summary = {k: float(np.mean(v)) for k, v in res.items()}
        best = max(summary, key=summary.get)
        out = dict(base, kind="cv_selection", folds=args.cv, grid=summary, best_lr=float(best.split("|")[0]),
                   best_epochs=int(best.split("|")[1]), trainings=len(args.lrs) * args.cv)
        p = RESULTS / "selection" / f"{task}__{args.model}__k{args.budget}__seed{args.seed}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        json.dump(out, open(p, "w"), indent=1)
        print(json.dumps(out))
        return

    # --- run finale
    tests = [(t, [dict(r, split=t) for r in test_set(t)]) for t in [task, *args.also]]
    if args.recipe == "original":
        lr, epochs, seed_train = cfg["lr_original"], min(40, max(3, math.ceil(4000 / len(train)))), 0
        if args.train_seed is not None:  # analisi di sensibilità: stessa ricetta, seed di training diverso
            seed_train = args.train_seed
    else:
        lr, epochs, seed_train = args.lr, args.epochs, args.seed
    clf = Classifier(args.model, lab, seed_train)
    selection = None
    if budget == "full" and args.recipe == "corrected":
        val_ids = set(manifest(task)["full_val"][str(args.seed)])
        val = [r for r in train if r["id"] in val_ids]
        tr = [r for r in train if r["id"] not in val_ids]
        best = {"f1": -1}

        def ev(ep):
            f1 = macro_f1([r["label"] for r in val], [lab[int(p.argmax())] for p in clf.predict(val)[0]])
            if f1 > best["f1"]:
                best.update(f1=f1, epoch=ep, state={n: p.detach().cpu().clone() for n, p in clf.model.named_parameters()
                                                     if p.requires_grad})
            return {"val_macro_f1": f1}
        t0 = time.perf_counter()
        hist, head = clf.fit(tr, lr, epochs, "corrected", seed_train, ev, list(range(1, epochs + 1)))
        train_s = time.perf_counter() - t0
        with torch.no_grad():  # ripristina lo stato dell'epoca migliore sulla validation
            for n, p in clf.model.named_parameters():
                if n in best["state"]:
                    p.copy_(best["state"][n].to(p.device))
        selection = {"by": "full_val", "best_epoch": best["epoch"], "val_macro_f1": best["f1"], "trained_on": len(tr)}
    else:
        t0 = time.perf_counter()
        hist, head = clf.fit(train, lr, epochs, args.recipe, seed_train)
        train_s = time.perf_counter() - t0

    for t, rows in tests:
        probs, lat = clf.predict(rows, timing=not args.no_timing)
        out = rows_out(probs, rows, lab, lat)
        m = evaluate(out, lab)
        m.update(train_s=round(train_s, 1), load_s=round(clf.load_s, 1), selection=selection,
                 peak_vram_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2) if torch.cuda.is_available() else None)
        config = dict(base, kind="supervised", task=t, trained_on_task=task, lr=lr, epochs=epochs, train_seed=seed_train,
                      micro_batch=clf.micro_bs, grad_accumulation=math.ceil(cfg["bs"] / clf.micro_bs),
                      dtype=clf.dtype, amp=str(clf.amp), history=hist, head=head)
        run_id = new_run_id(f"sup-{args.model}-{t}-k{args.budget}-seed{args.seed}-{args.recipe}")
        save_run(run_id, config, out, m)
        print(json.dumps({"run_id": run_id, "macro_f1": m["macro_f1"], "ece15": m.get("ece15")}))


if __name__ == "__main__":
    main()
