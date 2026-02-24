import torch
from ikinya.models.model import MiniTransformerLM

torch.manual_seed(0)

text = "hello ikinya ai"
chars = sorted(list(set(text)))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}

tokens = torch.tensor([stoi[ch] for ch in text], dtype=torch.long)

model = MiniTransformerLM(
    vocab_size=len(chars),
    d_model=16,
    num_heads=2,
    d_ff=64,
    num_layers=2,
    max_len=64
)

logits = model(tokens)  # (T, vocab_size)

print("tokens shape:", tokens.shape)
print("logits shape:", logits.shape)

# pretend we want to predict the next char after the last one
last_logits = logits[-1]
next_id = int(torch.argmax(last_logits).item())
print("predicted next token id:", next_id)
print("predicted next char:", itos[next_id])