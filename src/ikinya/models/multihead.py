import torch
import math
from .attention import scaled_dot_product_attention


class MultiHeadSelfAttention(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Linear layers to produce Q, K, V for all heads at once
        self.Wq = torch.nn.Linear(d_model, d_model, bias=False)
        self.Wk = torch.nn.Linear(d_model, d_model, bias=False)
        self.Wv = torch.nn.Linear(d_model, d_model, bias=False)

        # Output projection back to d_model
        self.Wo = torch.nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, causal: bool = True):
        """
        x: (T, d_model)
        returns: (T, d_model)
        """
        T, _ = x.shape

        # (T, d_model)
        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)

        # reshape into heads: (num_heads, T, head_dim)
        Q = Q.view(T, self.num_heads, self.head_dim).transpose(0, 1)
        K = K.view(T, self.num_heads, self.head_dim).transpose(0, 1)
        V = V.view(T, self.num_heads, self.head_dim).transpose(0, 1)

        mask = None
        if causal:
            mask = torch.tril(torch.ones(T, T, device=x.device))

        head_outputs = []
        for h in range(self.num_heads):
            out_h, _ = scaled_dot_product_attention(Q[h], K[h], V[h], mask)
            head_outputs.append(out_h)

        # concat heads: (T, d_model)
        out = torch.cat(head_outputs, dim=-1)

        # final projection
        return self.Wo(out)