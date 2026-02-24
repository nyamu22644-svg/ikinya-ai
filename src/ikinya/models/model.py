
import torch
from .transformer_block import TransformerBlock


class MiniTransformerLM(torch.nn.Module):
    def __init__(self, vocab_size: int, d_model: int, num_heads: int, d_ff: int, num_layers: int, max_len: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_len = max_len

        self.token_emb = torch.nn.Embedding(vocab_size, d_model)
        self.pos_emb = torch.nn.Embedding(max_len, d_model)

        self.blocks = torch.nn.ModuleList([
            TransformerBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff)
            for _ in range(num_layers)
        ])

        self.ln_f = torch.nn.LayerNorm(d_model)
        self.head = torch.nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, tokens: torch.Tensor):
        """
        tokens:
          - (T,) int64 -> returns (T, vocab_size)
          - (B, T) int64 -> returns (B, T, vocab_size)
        """
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
            squeeze_back = True
        elif tokens.dim() == 2:
            squeeze_back = False
        else:
            raise ValueError(f"Expected tokens dim 1 or 2, got {tokens.dim()}")

        B, T = tokens.shape
        assert T <= self.max_len, "Sequence too long for this model"

        positions = torch.arange(T, device=tokens.device).unsqueeze(0).expand(B, T)

        x = self.token_emb(tokens) + self.pos_emb(positions)  # (B, T, d_model)

        for block in self.blocks:
            x = block(x, causal=True)  # TransformerBlock already works with (B,T,d) via attention

        x = self.ln_f(x)
        logits = self.head(x)  # (B, T, vocab_size)

        if squeeze_back:
            logits = logits.squeeze(0)  # (T, vocab_size)

        return logits