import torch
from core.multihead import MultiHeadSelfAttention
from core.ffn import FeedForward


class TransformerBlock(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int):
        super().__init__()
        self.attn = MultiHeadSelfAttention(d_model=d_model, num_heads=num_heads)
        self.ffn = FeedForward(d_model=d_model, d_ff=d_ff)

        self.ln1 = torch.nn.LayerNorm(d_model)
        self.ln2 = torch.nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, causal: bool = True) -> torch.Tensor:
        # Pre-LN transformer (stable)
        x = x + self.attn(self.ln1(x), causal=causal)
        x = x + self.ffn(self.ln2(x))
        return x