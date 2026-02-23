import torch
from core.model import MiniTransformerLM

ckpt = torch.load("checkpoints_phase1_charlm.pt", map_location="cpu", weights_only=True)

stoi = ckpt["stoi"]
itos = ckpt["itos"]
cfg = ckpt["config"]

model = MiniTransformerLM(
    vocab_size=cfg["vocab_size"],
    d_model=cfg["d_model"],
    num_heads=cfg["num_heads"],
    d_ff=cfg["d_ff"],
    num_layers=cfg["num_layers"],
    max_len=cfg["max_len"],
)

model.load_state_dict(ckpt["model_state"])
model.eval()

seq_len = cfg["seq_len"]

# start prompt
prompt = "h"
context = torch.tensor([stoi[ch] for ch in prompt], dtype=torch.long)

generated = context.tolist()

for _ in range(80):
    logits = model(torch.tensor(generated[-seq_len:], dtype=torch.long))
    probs = torch.softmax(logits[-1], dim=0)
    next_id = int(torch.multinomial(probs, 1).item())
    generated.append(next_id)

print("PROMPT:", prompt)
print("--- sample ---")
print("".join(itos[i] for i in generated))