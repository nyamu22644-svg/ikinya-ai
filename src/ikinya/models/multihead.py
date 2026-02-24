import torch
from .attention import scaled_dot_product_attention


class MultiHeadSelfAttention(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.Wq = torch.nn.Linear(d_model, d_model, bias=False)
        self.Wk = torch.nn.Linear(d_model, d_model, bias=False)
        self.Wv = torch.nn.Linear(d_model, d_model, bias=False)
        self.Wo = torch.nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, causal: bool = True):
        """
        x:
          - (T, d_model) OR
          - (B, T, d_model)

        returns:
          - (T, d_model) OR
          - (B, T, d_model)
        """
        if x.dim() == 2:
            # add batch dim
            x = x.unsqueeze(0)
            squeeze_back = True
        elif x.dim() == 3:
            squeeze_back = False
        else:
            raise ValueError(f"Expected x dim 2 or 3, got {x.dim()}")

        B, T, _ = x.shape

        Q = self.Wq(x)  # (B, T, d_model)
        K = self.Wk(x)
        V = self.Wv(x)

        # (B, num_heads, T, head_dim)
        Q = Q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        mask = None
        if causal:
            # (1, 1, T, T) broadcastable to (B, H, T, T)
            mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)

        # (B, H, T, head_dim)
        out, _ = scaled_dot_product_attention(Q, K, V, mask)

        # back to (B, T, d_model)
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)

        out = self.Wo(out)

        if squeeze_back:
            out = out.squeeze(0)  # (T, d_model)

        return out