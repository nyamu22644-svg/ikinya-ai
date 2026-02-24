from __future__ import annotations

from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class SamplingConfig:
    max_new_tokens: int = 200
    temperature: float = 1.0
    top_k: int = 40
    greedy: bool = False  # if True, ignores temperature/top_k


@torch.no_grad()
def generate_text(model, prompt: str, stoi: dict, itos: dict, sampling: SamplingConfig) -> str:
    """
    Compatible with your current MiniTransformerLM:
    - model expects 1D token ids (T,)
    - model enforces max length via model.max_len
    We crop context to last max_len tokens so generation never asserts.
    """
    model.eval()
    device = next(model.parameters()).device
    max_len = int(getattr(model, "max_len"))

    # encode prompt
    idx = torch.tensor([stoi[ch] for ch in prompt], dtype=torch.long, device=device)

    for _ in range(int(sampling.max_new_tokens)):
        idx_cond = idx[-max_len:]  # enforce max_len
        logits = model(idx_cond)   # (T, V)
        logits_last = logits[-1]   # (V,)

        if sampling.greedy:
            next_id = torch.argmax(logits_last, dim=-1).view(1)
        else:
            temp = float(sampling.temperature)
            if temp <= 0:
                temp = 1.0

            probs = torch.softmax(logits_last / temp, dim=-1)

            k = int(sampling.top_k)
            if k > 0 and k < probs.numel():
                v, ix = torch.topk(probs, k=k)
                v = v / v.sum()
                next_id = ix[torch.multinomial(v, num_samples=1)]
            else:
                next_id = torch.multinomial(probs, num_samples=1)

        idx = torch.cat([idx, next_id], dim=0)

    return "".join(itos[int(i)] for i in idx.tolist())