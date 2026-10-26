"""Ricalcola le metriche di tutti i run dalle predizioni salvate (nessuna nuova inferenza).

  python -m dmstudy.recompute
"""
import json

from .common import RESULTS, labels, read_jsonl
from .metrics import evaluate

KEEP = ("input_tokens_billed", "cost_usd", "wall_s", "errors", "cached", "train_s", "selection", "peak_vram_gb",
        "throughput_per_s", "load_s")


def main():
    n = 0
    for p in sorted((RESULTS / "metrics").glob("*.json")):
        d = json.load(open(p, encoding="utf-8"))
        pred = RESULTS / "predictions" / f"{d.get('run_id')}.jsonl"
        if "run_id" not in d or not pred.exists():
            continue
        m = evaluate(read_jsonl(pred), labels(d["config"]["task"]))
        m.update({k: d["metrics"][k] for k in KEEP if k in d["metrics"]})
        d["metrics"] = m
        json.dump(d, open(p, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
        n += 1
    print(f"ricalcolati {n} run")


if __name__ == "__main__":
    main()
