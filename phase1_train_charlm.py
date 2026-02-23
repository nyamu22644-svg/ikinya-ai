import torch
import torch.nn.functional as F
from core.model import MiniTransformerLM

torch.manual_seed(0)

# Tiny dataset (character-level)
text = ("hello ikinya ai\n" * 200)

chars = sorted(list(set(text)))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}

data = torch.tensor([stoi[ch] for ch in text], dtype=torch.long)

# Hyperparams (small + CPU-friendly)
d_model = 32
num_heads = 4
d_ff = 128
num_layers = 2
max_len = 64
lr = 1e-3
steps = 1000
seq_len = 32

model = MiniTransformerLM(
    vocab_size=len(chars),
    d_model=d_model,
    num_heads=num_heads,
    d_ff=d_ff,
    num_layers=num_layers,
    max_len=max_len
)

opt = torch.optim.AdamW(model.parameters(), lr=lr)

def get_batch():
    i = torch.randint(0, len(data) - seq_len - 1, (1,)).item()
    x = data[i:i+seq_len]
    y = data[i+1:i+seq_len+1]
    return x, y

for step in range(1, steps + 1):
    x, y = get_batch()

    logits = model(x)                  # (T, vocab)
    loss = F.cross_entropy(logits, y)  # next-token loss

    opt.zero_grad()
    loss.backward()
    opt.step()

    if step % 100 == 0:
        print(f"step {step:4d} | loss={loss.item():.4f}")

# --- simple generation ---
model.eval()
context = torch.tensor([stoi["h"]], dtype=torch.long)

generated = [context[0].item()]

for _ in range(60):
    logits = model(context)
    probs = torch.softmax(logits[-1], dim=0)
    next_id = int(torch.multinomial(probs, 1).item())

    generated.append(next_id)
    context = torch.tensor(generated[-seq_len:], dtype=torch.long)

print("\n--- sample ---")
print("".join(itos[i] for i in generated))

torch.save({
    "model_state": model.state_dict(),
    "stoi": stoi,
    "itos": itos,
    "config": {
        "d_model": d_model,
        "num_heads": num_heads,
        "d_ff": d_ff,
        "num_layers": num_layers,
        "max_len": max_len,
        "seq_len": seq_len,
        "vocab_size": len(chars),
    }
}, "checkpoints_phase1_charlm.pt")

print("\nSaved checkpoint: checkpoints_phase1_charlm.pt")