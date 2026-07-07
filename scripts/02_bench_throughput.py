import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.bench import throughput_sweep, save_json
from src.model import GPTConfig, TinyGPT
from src.train import load_model

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=5)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) the trained small model (4 layers, 256 dim)
    model, stoi, itos = load_model(device=device)
    prompt = torch.tensor([[stoi["\n"]]], device=device)
    small = throughput_sweep(model, prompt, lengths=(64, 128, 256), runs=args.runs)

    # 2) a larger untrained model (throughput does not need trained weights),
    #    with a bigger block_size so we can push to longer sequences
    torch.manual_seed(0)
    big_cfg = GPTConfig(vocab_size=65, block_size=2112, n_layer=8, n_head=8, n_embd=512)
    big = TinyGPT(big_cfg).to(device).eval()
    big_prompt = torch.zeros(1, 1, dtype=torch.long, device=device)
    large = throughput_sweep(big, big_prompt, lengths=(256, 512, 1024, 2048), runs=3)

    out = {
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "small_model_4L_256d": small,
        "large_model_8L_512d": large,
    }
    print(json.dumps(out, indent=2))
    save_json(out, "throughput.json")
