import torch
import torch.nn.functional as F

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

# Attention scores
scores = Q @ K.T  # (T, T)

# ---- CAUSAL MASK ----
mask = torch.tril(torch.ones(T, T))  # lower triangle
scores = scores.masked_fill(mask == 0, float('-inf'))

# Softmax
weights = F.softmax(scores, dim=1)

out = weights @ V

print("scores shape:", scores.shape)
print("Row 5 attention weights sum:", weights[5].sum().item())

print("\nToken 5 can attend only to positions <= 5")
top = torch.topk(weights[5], k=5)
print("Top positions:", top.indices.tolist())
print("Weights:", top.values.tolist())