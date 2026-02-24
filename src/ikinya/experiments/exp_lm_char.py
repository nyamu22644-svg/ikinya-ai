import os
import torch
import torch.nn.functional as F

from ikinya.core.checkpoint import save_checkpoint, load_checkpoint
from ikinya.core.logging import RunLogger
from ikinya.experiments.registry import register
from ikinya.models.model import MiniTransformerLM


def build_vocab(text: str):
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    return chars, stoi, itos


def encode(text: str, stoi: dict) -> torch.Tensor:
    return torch.tensor([stoi[ch] for ch in text], dtype=torch.long)


@torch.no_grad()
def generate(model: MiniTransformerLM, prompt: str, stoi: dict, itos: dict, max_new_tokens: int = 200) -> str:
    model.eval()
    device = next(model.parameters()).device
    max_len = model.max_len  # from your MiniTransformerLM

    # start with prompt
    idx = torch.tensor([stoi[ch] for ch in prompt], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        # crop context to model max_len
        idx_cond = idx[-max_len:]

        logits = model(idx_cond)  # (T, V)
        next_id = torch.argmax(logits[-1], dim=-1).view(1)
        idx = torch.cat([idx, next_id], dim=0)

    return "".join(itos[int(i)] for i in idx.tolist())


@register("lm_char")
def run_lm_char(cfg: dict, run_dir: str, logger: RunLogger):
    # ---- load corpus ----
    path = cfg["data"]["path"]
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    if len(text) < 1000:
        raise ValueError(f"Corpus too small ({len(text)} chars). Put more text in {path}")

    chars, stoi, itos = build_vocab(text)
    vocab_size = len(chars)

    data = encode(text, stoi)
    T = int(cfg["data"]["seq_len"])
    B = int(cfg["data"]["batch_size"])

    # ---- model ----
    mcfg = cfg["model"]
    model = MiniTransformerLM(
        vocab_size=vocab_size,
        d_model=int(mcfg["d_model"]),
        num_heads=int(mcfg["n_heads"]),
        d_ff=int(mcfg["d_ff"]),
        num_layers=int(mcfg["n_layers"]),
        max_len=T,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["trainer"]["lr"]))

    # ---- resume if last.pt exists ----
    ckpt_path = os.path.join(run_dir, "last.pt")
    start_step = 0
    if os.path.exists(ckpt_path):
        ckpt = load_checkpoint(ckpt_path, model=model, optimizer=opt, map_location=device)
        start_step = int(ckpt.get("step", 0)) + 1
        logger.log_event(start_step, "resume", {"checkpoint": ckpt_path})

    max_steps = int(cfg["trainer"]["max_steps"])
    log_every = int(cfg["trainer"]["log_every"])

    # ---- training loop ----
    model.train()
    n = data.numel()

    for step in range(start_step, max_steps + 1):
        # sample random batch
        ix = torch.randint(0, n - T - 1, (B,), device=device)
        x = torch.stack([data[i : i + T] for i in ix]).to(device)       # (B, T)
        y = torch.stack([data[i + 1 : i + 1 + T] for i in ix]).to(device)  # (B, T)

                # model expects (T,) not (B,T) -> run per sequence, then stack
        logits_list = []
        for b in range(B):
            logits_b = model(x[b])          # (T, V)
            logits_list.append(logits_b)
        logits = torch.stack(logits_list, dim=0)  # (B, T, V)

        loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        # log + sample + checkpoint
        if step % log_every == 0 or step == start_step or step == max_steps:
            logger.log_metrics(step, {"loss": float(loss.item())})

            prompt = "h" if "h" in stoi else chars[0]
            sample = generate(model, prompt=prompt, stoi=stoi, itos=itos, max_new_tokens=200)
            logger.log_event(step, "sample", {"prompt": prompt, "text": sample})

            save_checkpoint(ckpt_path, step=step, model=model, optimizer=opt, config=cfg)

    logger.log_event(max_steps, "train_done", {"vocab_size": vocab_size, "device": device})