import torch
import torch.nn.functional as F
import math

torch.manual_seed(0)

text = "hello ikinya ai"

# vocab + tokens
chars = sorted(list(set(text)))
stoi = {ch: i for i, ch in enumerate(chars)}
tokens = torch.tensor([stoi[ch] for ch in text])

vocab_size = len(chars)
d_model = 8
T = len(tokens)

token_emb = torch.nn.Embedding(vocab_size, d_model)
pos_emb = torch.nn.Embedding(T, d_model)

x = token_emb(tokens) + pos_emb(torch.arange(T))  # (T, d_model)

# Linear projections
Wq = torch.randn(d_model, d_model)
Wk = torch.randn(d_model, d_model)
Wv = torch.randn(d_model, d_model)

Q = x @ Wq
K = x @ Wk
V = x @ Wv

# ---- SCALED DOT-PRODUCT ATTENTION ----
scores = (Q @ K.T) / math.sqrt(d_model)

# Causal mask
mask = torch.tril(torch.ones(T, T))
scores = scores.masked_fill(mask == 0, float('-inf'))

weights = F.softmax(scores, dim=1)

out = weights @ V

print("Scaled attention running.")
print("Output shape:", out.shape)
print("Row 5 sum:", weights[5].sum().item())

top = torch.topk(weights[5], k=5)
print("\nTop positions:", top.indices.tolist())
print("Weights:", top.values.tolist())