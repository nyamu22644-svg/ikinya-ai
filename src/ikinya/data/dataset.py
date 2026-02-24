import torch


class CharDataset:
    """
    Loads a text file, builds vocab, and samples random (x,y) batches.
    Returns:
      x: (B, T) int64
      y: (B, T) int64
    """

    def __init__(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            self.text = f.read()

        if len(self.text) < 1000:
            raise ValueError(f"Corpus too small ({len(self.text)} chars). Put more text in {path}")

        self.chars = sorted(list(set(self.text)))
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.vocab_size = len(self.chars)

        self.data = torch.tensor([self.stoi[ch] for ch in self.text], dtype=torch.long)

    def sample_batch(self, batch_size: int, seq_len: int, device: str):
        n = self.data.numel()
        if n < seq_len + 2:
            raise ValueError(f"Corpus too small for seq_len={seq_len}. Need at least {seq_len+2} chars.")

        ix = torch.randint(0, n - seq_len - 1, (batch_size,), device=device)
        x = torch.stack([self.data[i : i + seq_len] for i in ix]).to(device)          # (B,T)
        y = torch.stack([self.data[i + 1 : i + 1 + seq_len] for i in ix]).to(device)  # (B,T)
        return x, y

    def decode(self, ids):
        return "".join(self.itos[int(i)] for i in ids)