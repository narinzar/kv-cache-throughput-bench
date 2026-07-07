import argparse
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.train import train

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()
    t0 = time.perf_counter()
    ckpt, val = train(steps=args.steps, batch_size=args.batch_size)
    print(f"train wall time: {time.perf_counter() - t0:.1f}s")
