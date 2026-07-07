import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.bench import eviction_study, sink_stress_test, save_json, OUT_DIR
from src.train import load_model

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=256)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, stoi, itos = load_model(device=device)
    prompt = torch.tensor([[stoi["\n"]]], device=device)
    study = eviction_study(model, prompt, max_new_tokens=args.tokens)
    sink = sink_stress_test(gen_tokens=args.tokens, budget=32)
    study["sink_stress_test"] = sink
    print(json.dumps(study, indent=2))
    save_json(study, "eviction.json")
    print(f"\nsynthetic attention-sink test (budget 32): FIFO hit {sink['fifo_hit']}  LRU hit {sink['lru_hit']}")

    budgets = sorted({r["budget"] for r in study["rows"]})
    policies = ["fifo", "lru", "none"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for pol in policies:
        hs = [next(r["hit_rate"] for r in study["rows"] if r["budget"] == b and r["policy"] == pol) for b in budgets]
        kls = [next(r["kl_vs_full"] for r in study["rows"] if r["budget"] == b and r["policy"] == pol) for b in budgets]
        ax1.plot(budgets, hs, marker="o", label=pol)
        ax2.plot(budgets, kls, marker="o", label=pol)
    ax1.set_xlabel("cache budget (positions)")
    ax1.set_ylabel("attention mass retained (hit rate)")
    ax1.set_title("Eviction policy hit rate")
    ax1.legend()
    ax2.set_xlabel("cache budget (positions)")
    ax2.set_ylabel("KL(full || bounded) next-token")
    ax2.set_title("Quality cost of eviction")
    ax2.legend()
    fig.tight_layout()
    plot_path = os.path.join(OUT_DIR, "eviction.png")
    fig.savefig(plot_path, dpi=120)
    print(f"saved {plot_path}")
