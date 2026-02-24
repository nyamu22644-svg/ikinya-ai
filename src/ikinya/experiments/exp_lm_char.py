import os
import torch
import torch.nn.functional as F

from ikinya.core.checkpoint import save_checkpoint, load_checkpoint
from ikinya.core.logging import RunLogger
from ikinya.experiments.registry import register
from ikinya.models.model import MiniTransformerLM
from ikinya.data.dataset import CharDataset
from ikinya.train.generation import generate_text


@register("lm_char")
def run_lm_char(cfg: dict, run_dir: str, logger: RunLogger):
    # ---- setup ----
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dcfg = cfg["data"]
    mcfg = cfg["model"]
    tcfg = cfg["trainer"]
    scfg = cfg.get("sampling", {})

    dataset = CharDataset(dcfg["path"])
    vocab_size = dataset.vocab_size

    T = int(dcfg["seq_len"])
    B = int(dcfg["batch_size"])

    # ---- model (translate config keys -> model args) ----
    model = MiniTransformerLM(
        vocab_size=vocab_size,
        d_model=int(mcfg["d_model"]),
        num_heads=int(mcfg["n_heads"]),
        d_ff=int(mcfg["d_ff"]),
        num_layers=int(mcfg["n_layers"]),
        max_len=T,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=float(tcfg["lr"]))

    # ---- resume ----
    ckpt_path = os.path.join(run_dir, "last.pt")
    start_step = 0
    if os.path.exists(ckpt_path):
        ckpt = load_checkpoint(ckpt_path, model=model, optimizer=opt, map_location=device)
        start_step = int(ckpt.get("step", 0)) + 1
        logger.log_event(start_step, "resume", {"checkpoint": ckpt_path})

    max_steps = int(tcfg["max_steps"])
    log_every = int(tcfg["log_every"])
    grad_accum = int(tcfg.get("grad_accum", 1))

    # ---- sampling params ----
    prompt = scfg.get("prompt", "h")
    if prompt and prompt[0] not in dataset.stoi:
        prompt = dataset.chars[0]

    temperature = float(scfg.get("temperature", 1.0))
    top_k = int(scfg.get("top_k", 0))
    greedy = bool(scfg.get("greedy", False))
    max_new_tokens = int(scfg.get("max_new_tokens", 128))

    # ---- training ----
    model.train()
    opt.zero_grad(set_to_none=True)

    try:
        for step in range(start_step, max_steps + 1):
            # gradient accumulation
            total_loss = 0.0
            for micro in range(grad_accum):
                x, y = dataset.sample_batch(B, T, device=device)  # (B,T), (B,T)

                logits = model(x)  # expected (B,T,V)
                loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))
                (loss / grad_accum).backward()
                total_loss += float(loss.item())

            opt.step()
            opt.zero_grad(set_to_none=True)

            # log/sample/ckpt
            if step % log_every == 0 or step == start_step or step == max_steps:
                avg_loss = total_loss / grad_accum
                logger.log_metrics(step, {"loss": avg_loss})

                sample = generate_text(
                    model=model,
                    stoi=dataset.stoi,
                    itos=dataset.itos,
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    greedy=greedy,
                )
                logger.log_event(step, "sample", {
                    "prompt": prompt,
                    "text": sample,
                    "sampling": {
                        "max_new_tokens": max_new_tokens,
                        "temperature": temperature,
                        "top_k": top_k,
                        "greedy": greedy,
                    }
                })

                save_checkpoint(ckpt_path, step=step, model=model, optimizer=opt, config=cfg)

        logger.log_event(max_steps, "train_done", {"vocab_size": vocab_size, "device": device})

    except KeyboardInterrupt:
        # interrupt-safe save
        logger.log_event(step, "interrupt", {"message": "KeyboardInterrupt: saving last.pt"})
        save_checkpoint(ckpt_path, step=step, model=model, optimizer=opt, config=cfg)
        raise