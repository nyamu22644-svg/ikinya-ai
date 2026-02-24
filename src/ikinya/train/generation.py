import torch


@torch.no_grad()
def generate_text(
    model,
    prompt: str,
    stoi: dict,
    itos: dict,
    max_new_tokens: int = 128,
    temperature: float = 1.0,
    top_k: int = 0,
    greedy: bool = False,
):
    """
    Robust text generation that works whether your model forward accepts:
      - (T,) -> (T, V)
      - (B, T) -> (B, T, V)

    Args:
      prompt: starting string
      stoi/itos: vocab maps
      max_new_tokens: how many new tokens to generate
      temperature: sampling temperature (ignored if greedy=True)
      top_k: if >0, sample only from top_k logits
      greedy: if True, argmax decoding

    Returns:
      generated string (prompt + continuation)
    """
    model.eval()
    device = next(model.parameters()).device
    max_len = getattr(model, "max_len", 256)

    # Encode prompt
    if len(prompt) == 0:
        raise ValueError("prompt must not be empty")

    idx = torch.tensor([stoi[c] for c in prompt], dtype=torch.long, device=device)

    for _ in range(int(max_new_tokens)):
        # crop context to model max_len
        idx_cond = idx[-max_len:]

        # Always try batched call for safety: (1, T)
        x = idx_cond.view(1, -1)

        logits = model(x)

        # logits could be (1, T, V) or (T, V)
        if logits.dim() == 3:
            logits_last = logits[0, -1, :]  # (V,)
        elif logits.dim() == 2:
            logits_last = logits[-1, :]     # (V,)
        else:
            raise RuntimeError(f"Unexpected logits shape: {tuple(logits.shape)}")

        if greedy:
            next_id = torch.argmax(logits_last, dim=-1).view(1)
        else:
            # temperature
            t = float(temperature)
            if t <= 0:
                t = 1.0
            logits_last = logits_last / t

            # top-k filtering
            k = int(top_k) if top_k is not None else 0
            if k > 0:
                v, ix = torch.topk(logits_last, k)
                probs = torch.softmax(v, dim=-1)
                next_id = ix[torch.multinomial(probs, num_samples=1)]
            else:
                probs = torch.softmax(logits_last, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)

        idx = torch.cat([idx, next_id.to(device)], dim=0)

    return "".join(itos[int(i)] for i in idx.tolist())