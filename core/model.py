import torch
from core.transformer_block import TransformerBlock


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
        self.head = torch.nn.Linear(d_model, vocab_size, bias=False)  # output projection

    def forward(self, tokens: torch.Tensor):
        """
        tokens: (T,) int64
        returns logits: (T, vocab_size)
        """
        T = tokens.size(0)
        assert T <= self.max_len, "Sequence too long for this model"

        positions = torch.arange(T, device=tokens.device)

        x = self.token_emb(tokens) + self.pos_emb(positions)

        for block in self.blocks:
            x = block(x, causal=True)

        x = self.ln_f(x)
        logits = self.head(x)
        return logits