"""Autoregressive generation: cached, uncached, and bounded-cache variants."""
import time

import torch

from .kv_cache import EvictionSimulator


@torch.no_grad()
def generate_uncached(model, idx, max_new_tokens):
    """Recompute the full forward pass every step (no KV cache)."""
    model.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.cfg.block_size :]
        logits, _ = model(idx_cond)
        nxt = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        idx = torch.cat([idx, nxt], dim=1)
    return idx


@torch.no_grad()
def generate_cached(model, idx, max_new_tokens):
    """Decode with a growing per-layer KV cache (from scratch)."""
    model.eval()
    device = idx.device
    total = idx.shape[1] + max_new_tokens
    assert total <= model.cfg.block_size, f"context {total} exceeds block_size {model.cfg.block_size}"
    pos_ids = torch.arange(total, device=device)  # precompute to avoid per-step H2D copies
    caches = [None] * model.cfg.n_layer
    # prefill: feed the prompt one token at a time to populate the cache
    for t in range(idx.shape[1]):
        logits, caches, _ = model.step(idx[:, t : t + 1], pos_ids[t : t + 1], caches)
    out = idx
    for t in range(idx.shape[1], total):
        nxt = logits.argmax(dim=-1, keepdim=True)
        out = torch.cat([out, nxt], dim=1)
        logits, caches, _ = model.step(nxt, pos_ids[t : t + 1], caches)
    return out


def _avg_attn_row(attns):
    # attns: list over layers of (B, nh, Tcur); average over layers, heads, batch0
    stacked = torch.stack([a[0].mean(dim=0) for a in attns])  # (n_layer, Tcur)
    return stacked.mean(dim=0)  # (Tcur,)


@torch.no_grad()
def reference_run(model, idx, max_new_tokens):
    """Full-cache greedy run. Returns the produced sequence, the per-step next
    token distribution, and the per-step averaged attention row."""
    model.eval()
    device = idx.device
    caches = [None] * model.cfg.n_layer
    for t in range(idx.shape[1]):
        col = idx[:, t : t + 1]
        logits, caches, attns = model.step(col, torch.tensor([t], device=device), caches, want_attn=True)
    seq = idx.clone()
    dists, attn_rows = [], []
    for t in range(idx.shape[1], idx.shape[1] + max_new_tokens):
        p = torch.softmax(logits[0], dim=-1)
        dists.append(p)
        attn_rows.append(_avg_attn_row(attns).cpu().tolist())
        nxt = logits.argmax(dim=-1, keepdim=True)
        seq = torch.cat([seq, nxt], dim=1)
        logits, caches, attns = model.step(nxt, torch.tensor([t], device=device), caches, want_attn=True)
    return seq, dists, attn_rows


def _drop_column(cache, j):
    k, v = cache
    k = torch.cat([k[:, :, :j], k[:, :, j + 1 :]], dim=2)
    v = torch.cat([v[:, :, :j], v[:, :, j + 1 :]], dim=2)
    return (k, v)


@torch.no_grad()
def bounded_run(model, seq, prompt_len, budget, policy):
    """Replay a fixed token sequence with a real cache capped at `budget`
    positions using the given eviction policy. Returns per-step next-token
    distributions and mean decode latency (ms/token)."""
    model.eval()
    device = seq.device
    caches = [None] * model.cfg.n_layer
    resident, last_access = [], {}

    def maybe_evict(step):
        nonlocal caches, resident
        if policy == "none" or len(resident) <= budget:
            return
        if policy == "fifo":
            j = 0
        else:  # lru
            j = min(range(len(resident)), key=lambda i: last_access[resident[i]])
        victim = resident.pop(j)
        del last_access[victim]
        caches = [_drop_column(c, j) for c in caches]

    for t in range(prompt_len):
        col = seq[:, t : t + 1]
        logits, caches, attns = model.step(col, torch.tensor([t], device=device), caches, want_attn=True)
        resident.append(t)
        last_access[t] = t
        maybe_evict(t)

    dists, times = [], []
    steps = seq.shape[1] - prompt_len
    for i in range(steps):
        t = prompt_len + i
        dists.append(torch.softmax(logits[0], dim=-1))
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        col = seq[:, t : t + 1]
        logits, caches, attns = model.step(col, torch.tensor([t], device=device), caches, want_attn=True)
        resident.append(t)
        last_access[t] = t
        row = attns[-1][0].mean(dim=0).cpu().tolist()  # last layer, mean heads
        if row:
            top = max(range(len(row)), key=lambda k: row[k])
            if top < len(resident):
                last_access[resident[top]] = t
        maybe_evict(t)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    ms_per_token = sum(times) / len(times) if times else 0.0
    return dists, ms_per_token


def simulate_hit_rate(attn_rows, budget, policy):
    sim = EvictionSimulator(budget, policy)
    rates = [sim.step(row) for row in attn_rows]
    return sum(rates) / len(rates) if rates else 0.0
