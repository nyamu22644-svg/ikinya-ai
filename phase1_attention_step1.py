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

x = token_emb(tokens) + pos_emb(torch.arange(T))   # (T, d_model)

# --- Self-attention (single head, simplest form) ---
# Create Q, K, V as linear projections of x
Wq = torch.randn(d_model, d_model)
Wk = torch.randn(d_model, d_model)
Wv = torch.randn(d_model, d_model)

Q = x @ Wq   # (T, d_model)
K = x @ Wk
V = x @ Wv

# Attention scores = QK^T
scores = Q @ K.T  # (T, T)

# Softmax to get weights
weights = F.softmax(scores, dim=1)  # each row sums to 1

# Weighted sum of V
out = weights @ V  # (T, d_model)

print("x shape:", x.shape)
print("scores shape:", scores.shape)
print("weights[0] sums to:", weights[0].sum().item())
print("out shape:", out.shape)

# show what token 0 attends to (top 5 positions)
top = torch.topk(weights[0], k=5)
print("\nToken 0 attends most to positions:", top.indices.tolist())
print("With weights:", top.values.tolist())