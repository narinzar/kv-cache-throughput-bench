import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data import download, load

path = download()
tok, train_data, val_data = load()
print(f"downloaded {path}")
print(f"chars: {len(train_data) + len(val_data)}  vocab: {tok.vocab_size}")
print(f"train tokens: {len(train_data)}  val tokens: {len(val_data)}")
