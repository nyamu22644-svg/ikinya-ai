import os
import torch
import torch.nn.functional as F

from ikinya.core.checkpoint import save_checkpoint, load_checkpoint
from ikinya.core.logging import RunLogger
from ikinya.experiments.registry import register
from ikinya.models.model import MiniTransformerLM
from ikinya.train.generation import generate_text
from ikinya.data.dataset import CharDataset


def build_vocab(text: str):
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    return chars, stoi, itos


@register("lm_char")
def run_lm_char(cfg: dict, run_dir: str, logger: RunLogger):
    # ---- load corpus ----
    path = cfg["data"]["path"]
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    chars, stoi, itos = build_vocab(text)
    vocab_size = len(chars)

    # ---- config ----
    T = int(cfg["data"]["seq_len"])
    B = int(cfg["data"]["batch_size"])

    tr_cfg = cfg["trainer"]
    max_steps = int(tr_cfg["max_steps"])
    log_every = int(tr_cfg["log_every"])
    lr = float(tr_cfg["lr"])
    grad_accum = int(tr_cfg.get("grad_accum", 1))

    samp_cfg = cfg.get("sampling", {})
    samp_max_new = int(samp_cfg.get("max_new_tokens", 128))
    samp_temp = float(samp_cfg.get("temperature", 1.0))
    samp_top_k = int(samp_cfg.get("top_k", 0))
    samp_greedy = bool(samp_cfg.get("greedy", False))

    # ---- device ----
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(max(1, os.cpu_count() // 2))

    # ---- dataset ----
    ds = CharDataset(text=text, seq_len=T, stoi=stoi)

    # ---- model ----
    mcfg = cfg["model"]
    model = MiniTransformerLM(
        vocab_size=vocab_size,
        d_model=int(mcfg["d_model"]),
        num_heads=int(mcfg["n_heads"]),
        d_ff=int(mcfg["d_ff"]),
        num_layers=int(mcfg["n_layers"]),
        max_len=T,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    # ---- resume if last.pt exists ----
    ckpt_path = os.path.join(run_dir, "last.pt")
    start_step = 0
    if os.path.exists(ckpt_path):
        ckpt = load_checkpoint(ckpt_path, model=model, optimizer=opt, map_location=device)
        start_step = int(ckpt.get("step", 0)) + 1
        logger.log_event(start_step, "resume", {"checkpoint": ckpt_path})

    # ---- training loop ----
    model.train()
    opt.zero_grad(set_to_none=True)

    try:
        for step in range(start_step, max_steps + 1):
            # gradient accumulation
            total_loss = 0.0
            for _ in range(grad_accum):
                x, y = ds.sample_batch(batch_size=B, device=device)  # (B,T), (B,T)

                logits = model(x)  # (B,T,V) after your batched upgrade
                loss = F.cross_entropy(
                    logits.reshape(-1, vocab_size),
                    y.reshape(-1),
                )
                (loss / grad_accum).backward()
                total_loss += float(loss.item())

            opt.step()
            opt.zero_grad(set_to_none=True)

            # log + sample + checkpoint
            if step % log_every == 0 or step == start_step or step == max_steps:
                avg_loss = total_loss / grad_accum
                logger.log_metrics(step, {"loss": avg_loss})

                prompt = "h" if "h" in stoi else chars[0]
                sample = generate_text(
                    model=model,
                    prompt=prompt,
                    stoi=stoi,
                    itos=itos,
                    max_new_tokens=samp_max_new,
                    temperature=samp_temp,
                    top_k=samp_top_k,
                    greedy=samp_greedy,
                )
                logger.log_event(step, "sample", {"prompt": prompt, "text": sample, "sampling": {
                    "max_new_tokens": samp_max_new,
                    "temperature": samp_temp,
                    "top_k": samp_top_k,
                    "greedy": samp_greedy,
                }})

                save_checkpoint(ckpt_path, step=step, model=model, optimizer=opt, config=cfg)

        logger.log_event(max_steps, "train_done", {"vocab_size": vocab_size, "device": device})

    except KeyboardInterrupt:
        # ALWAYS save progress when you stop training
        logger.log_event(step, "interrupt", {"message": "KeyboardInterrupt - saving last.pt"})
        save_checkpoint(ckpt_path, step=step, model=model, optimizer=opt, config=cfg)
        raise