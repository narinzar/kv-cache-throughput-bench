"""Short training loop for the TinyGPT model on Tiny Shakespeare."""
import os

import torch
from tqdm import tqdm

from .data import load, get_batch
from .model import GPTConfig, TinyGPT

CKPT_PATH = os.path.join(os.path.dirname(__file__), "..", "outputs", "tinygpt.pt")


def train(steps=2000, batch_size=48, block_size=512, lr=3e-4, eval_interval=250, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok, train_data, val_data = load()
    cfg = GPTConfig(vocab_size=tok.vocab_size, block_size=block_size)
    model = TinyGPT(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params/1e6:.2f}M  device: {device}")

    @torch.no_grad()
    def estimate_loss(data, iters=50):
        model.eval()
        losses = torch.zeros(iters)
        for i in range(iters):
            x, y = get_batch(data, block_size, batch_size, device)
            _, loss = model(x, y)
            losses[i] = loss.item()
        model.train()
        return losses.mean().item()

    for step in tqdm(range(steps), desc="train"):
        x, y = get_batch(train_data, block_size, batch_size, device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % eval_interval == 0 or step == steps - 1:
            vl = estimate_loss(val_data)
            tqdm.write(f"step {step}: train {loss.item():.3f}  val {vl:.3f}")

    os.makedirs(os.path.dirname(os.path.abspath(CKPT_PATH)), exist_ok=True)
    torch.save({"model": model.state_dict(), "cfg": cfg.__dict__, "stoi": tok.stoi}, CKPT_PATH)
    final_val = estimate_loss(val_data, iters=100)
    print(f"saved {CKPT_PATH}  final val loss {final_val:.3f}  val ppl {torch.tensor(final_val).exp():.2f}")
    return CKPT_PATH, final_val


def load_model(ckpt_path=CKPT_PATH, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(ckpt_path, map_location=device)
    cfg = GPTConfig(**ck["cfg"])
    model = TinyGPT(cfg).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    itos = {i: c for c, i in ck["stoi"].items()}
    return model, ck["stoi"], itos
