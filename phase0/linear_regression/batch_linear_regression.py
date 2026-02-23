import torch

torch.manual_seed(0)

# Dataset: y = 2x + 3
x = torch.linspace(0, 10, 20).unsqueeze(1)  # (20,1)
y_true = 2 * x + 3

def train(lr: float, steps: int = 500):
    w = torch.randn(1, requires_grad=True)
    b = torch.randn(1, requires_grad=True)

    for step in range(1, steps + 1):
        y_pred = w * x + b
        loss = ((y_pred - y_true) ** 2).mean()

        # stop if it explodes
        if torch.isnan(loss) or torch.isinf(loss):
            return None

        loss.backward()

        with torch.no_grad():
            w -= lr * w.grad
            b -= lr * b.grad

        w.grad.zero_()
        b.grad.zero_()

    return loss.item(), w.item(), b.item()

# Try multiple learning rates (mini "experimentation lab")
lrs = [0.001, 0.003, 0.01, 0.03, 0.05]
results = []

for lr in lrs:
    out = train(lr, steps=500)
    if out is None:
        print(f"lr={lr}: exploded (nan/inf)")
    else:
        loss, w, b = out
        results.append((loss, lr, w, b))
        print(f"lr={lr}: loss={loss:.6f} w={w:.4f} b={b:.4f}")

best = min(results, key=lambda t: t[0])
print("\nBEST:")
print(f"lr={best[1]} loss={best[0]:.6f} w={best[2]:.4f} b={best[3]:.4f}")