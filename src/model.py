"""A small decoder-only transformer written from scratch.

The attention module supports three modes:
  - full sequence forward (training / prefill)
  - single-step incremental decode with a per-layer KV cache
  - optionally returning the query token's attention distribution over past
    positions, which the eviction study consumes.
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 65
    block_size: int = 256
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 256
    dropout: float = 0.1


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        # Full parallel forward with a causal mask (training / prefill).
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)

    def step(self, x, past_k, past_v, want_attn=False):
        # Single new token. x: (B, 1, C). past_k/past_v: (B, nh, Tpast, hd) or None.
        B, T, C = x.shape
        assert T == 1
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, 1, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, 1, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, 1, self.n_head, self.head_dim).transpose(1, 2)
        if past_k is not None:
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        # Attend the single query to all cached keys (no mask needed: all are past).
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = F.softmax(att, dim=-1)  # (B, nh, 1, Tcur)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, 1, C)
        y = self.proj(y)
        attn = att.squeeze(2) if want_attn else None  # (B, nh, Tcur)
        return y, k, v, attn


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
            nn.GELU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

    def step(self, x, past_k, past_v, want_attn=False):
        y, k, v, attn = self.attn.step(self.ln1(x), past_k, past_v, want_attn)
        x = x + y
        x = x + self.mlp(self.ln2(x))
        return x, k, v, attn


class TinyGPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # weight tying
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def step(self, idx_col, pos, caches, want_attn=False):
        """Decode one token. caches is a list of (k, v) per layer, or None.

        Returns logits (B, vocab), updated caches, and per-layer attention
        distributions (list of (B, nh, Tcur)) when want_attn is set.
        """
        x = self.tok_emb(idx_col) + self.pos_emb(pos)
        new_caches, attns = [], []
        for blk, cache in zip(self.blocks, caches):
            pk, pv = (None, None) if cache is None else cache
            x, k, v, attn = blk.step(x, pk, pv, want_attn)
            new_caches.append((k, v))
            if want_attn:
                attns.append(attn)
        x = self.ln_f(x)
        logits = self.head(x[:, -1, :])
        return logits, new_caches, (attns if want_attn else None)
