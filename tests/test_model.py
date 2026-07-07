import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.model import GPTConfig, TinyGPT
from src.generate import generate_cached, generate_uncached


def _tiny_model():
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=32, block_size=64, n_layer=2, n_head=2, n_embd=32, dropout=0.0)
    return TinyGPT(cfg).eval()


def test_forward_shapes():
    model = _tiny_model()
    idx = torch.randint(0, 32, (3, 16))
    logits, loss = model(idx, targets=idx)
    assert logits.shape == (3, 16, 32)
    assert loss.item() > 0


def test_cached_matches_uncached():
    # A correct KV cache must produce identical greedy output to recomputing
    # the full context each step.
    model = _tiny_model()
    prompt = torch.randint(0, 32, (1, 5))
    a = generate_uncached(model, prompt.clone(), 40)
    b = generate_cached(model, prompt.clone(), 40)
    assert torch.equal(a, b), "cached and uncached generation diverged"


def test_cache_grows():
    model = _tiny_model()
    device = torch.device("cpu")
    caches = [None] * model.cfg.n_layer
    col = torch.zeros(1, 1, dtype=torch.long)
    for t in range(5):
        _, caches, _ = model.step(col, torch.tensor([t]), caches)
    k, v = caches[0]
    assert k.shape[2] == 5 and v.shape[2] == 5
