import torch
from ikinya.models.transformer_block import TransformerBlock

torch.manual_seed(0)

T = 6
d_model = 8

x = torch.randn(T, d_model)

block = TransformerBlock(d_model=d_model, num_heads=2, d_ff=32)
out = block(x, causal=True)

print("x shape:", x.shape)
print("out shape:", out.shape)