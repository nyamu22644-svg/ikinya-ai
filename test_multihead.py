import torch
from core.multihead import MultiHeadSelfAttention

torch.manual_seed(0)

T = 6
d_model = 8
heads = 2

x = torch.randn(T, d_model)

mha = MultiHeadSelfAttention(d_model=d_model, num_heads=heads)
out = mha(x, causal=True)

print("x shape:", x.shape)
print("out shape:", out.shape)