"""Benchmarks: cached vs uncached throughput, and the eviction-policy study."""
import json
import os
import time

import torch

from .generate import (
    generate_cached,
    generate_uncached,
    reference_run,
    bounded_run,
    simulate_hit_rate,
)

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def _time_gen(fn, model, prompt, n, runs):
    device = prompt.device
    fn(model, prompt, 8)  # warmup
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(runs):
        fn(model, prompt, n)
    _sync(device)
    return (time.perf_counter() - t0) / runs


def throughput_sweep(model, prompt, lengths=(64, 128, 256), runs=5):
    """Sweep the number of generated tokens and report cached vs uncached
    tokens/sec at each length. KV caching wins once the quadratic recompute
    cost of the uncached path dominates per-step launch overhead."""
    rows = []
    for n in lengths:
        unc = _time_gen(generate_uncached, model, prompt, n, runs)
        cac = _time_gen(generate_cached, model, prompt, n, runs)
        rows.append(
            {
                "new_tokens": n,
                "uncached_tok_s": round(n / unc, 1),
                "cached_tok_s": round(n / cac, 1),
                "speedup": round(unc / cac, 2),
            }
        )
    return rows


def kl(p_list, q_list):
    tot = 0.0
    for p, q in zip(p_list, q_list):
        tot += torch.sum(p * (torch.log(p + 1e-12) - torch.log(q + 1e-12))).item()
    return tot / len(p_list)


def eviction_study(model, prompt, max_new_tokens=256, budgets=(32, 64, 128), policies=("none", "fifo", "lru")):
    seq, ref_dists, attn_rows = reference_run(model, prompt, max_new_tokens)
    prompt_len = prompt.shape[1]
    rows = []
    for budget in budgets:
        for policy in policies:
            if policy == "none":
                hit = 1.0
            else:
                hit = simulate_hit_rate(attn_rows, budget, policy)
            b_dists, ms = bounded_run(model, seq, prompt_len, budget, policy)
            divergence = kl(ref_dists, b_dists)
            rows.append(
                {
                    "budget": budget,
                    "policy": policy,
                    "hit_rate": round(hit, 4),
                    "kl_vs_full": round(divergence, 5),
                    "ms_per_token": round(ms, 4),
                }
            )
    return {"prompt_len": prompt_len, "gen_tokens": max_new_tokens, "rows": rows}


def sink_stress_test(gen_tokens=256, budget=32, sink_mass=0.3, n_local=3, seed=0):
    """Synthetic stress test: a persistent attention sink at position 0 plus
    local recency. This is where LRU should beat FIFO, because FIFO evicts the
    sink once it falls outside the budget window while LRU keeps it (the sink is
    re-accessed every step). Real char-LM attention is nearly all local, so the
    two policies tie there; this isolates the case that separates them."""
    rows = []
    for t in range(gen_tokens):
        n = t + 1
        row = [0.0] * n
        if n == 1:
            row[0] = 1.0
        else:
            row[0] = sink_mass  # the sink
            local = min(n_local, n - 1)
            for j in range(local):
                row[n - 1 - j] += (1.0 - sink_mass) / local
        s = sum(row)
        rows.append([x / s for x in row])
    return {
        "budget": budget,
        "sink_mass": sink_mass,
        "fifo_hit": round(simulate_hit_rate(rows, budget, "fifo"), 4),
        "lru_hit": round(simulate_hit_rate(rows, budget, "lru"), 4),
    }


def save_json(obj, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    return path
