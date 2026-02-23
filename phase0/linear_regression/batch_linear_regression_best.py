import torch

torch.manual_seed(0)

x = torch.linspace(0, 10, 20).unsqueeze(1)
y_true = 2 * x + 3

w = torch.randn(1, requires_grad=True)
b = torch.randn(1, requires_grad=True)

lr = 0.01

for step in range(1, 5001):
    y_pred = w * x + b
    loss = ((y_pred - y_true) ** 2).mean()

    loss.backward()

    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad

    w.grad.zero_()
    b.grad.zero_()

    if step in [1, 10, 100, 1000, 5000]:
        print(f"step {step:4d} | loss={loss.item():.8f} | w={w.item():.4f} | b={b.item():.4f}")

print("\nFinal:")
print("w:", w.item())
print("b:", b.item())