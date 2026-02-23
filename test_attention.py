import torch
from core.attention import scaled_dot_product_attention

torch.manual_seed(0)

T = 6
d_model = 8

x = torch.randn(T, d_model)

Wq = torch.randn(d_model, d_model)
Wk = torch.randn(d_model, d_model)
Wv = torch.randn(d_model, d_model)

Q = x @ Wq
K = x @ Wk
V = x @ Wv

mask = torch.tril(torch.ones(T, T))

out, weights = scaled_dot_product_attention(Q, K, V, mask)

print("Output shape:", out.shape)
print("Row 3 sum:", weights[3].sum().item())