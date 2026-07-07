"""Tiny Shakespeare loader with a char-level tokenizer.

The corpus is the public-domain Karpathy char-rnn Shakespeare file. It is
downloaded on first use into data/ (gitignored).
"""
import os
import urllib.request

import truststore
import torch

truststore.inject_into_ssl()  # trust the OS cert store (needed behind TLS proxy)

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "tinyshakespeare.txt")


def download(path=DATA_PATH):
    path = os.path.abspath(path)
    if os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with urllib.request.urlopen(DATA_URL, timeout=60) as r:
        text = r.read().decode("utf-8")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path


class CharTokenizer:
    def __init__(self, text):
        chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for c, i in self.stoi.items()}
        self.vocab_size = len(chars)

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, ids):
        return "".join(self.itos[int(i)] for i in ids)


def load(path=DATA_PATH):
    path = download(path)
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    tok = CharTokenizer(text)
    data = torch.tensor(tok.encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    return tok, data[:n], data[n:]


def get_batch(data, block_size, batch_size, device):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix])
    return x.to(device), y.to(device)
