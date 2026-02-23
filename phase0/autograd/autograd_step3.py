import torch

w = torch.tensor(1.0, requires_grad=True)
x = torch.tensor(2.0)
target = torch.tensor(10.0)

lr = 0.1

for step in range(1, 21):
    # forward
    y = w * x
    loss = (y - target) ** 2

    # backward
    loss.backward()

    # update
    with torch.no_grad():
        w -= lr * w.grad
    w.grad.zero_()

    if step in [1, 2, 3, 5, 10, 20]:
        print(f"step {step:2d} | w={w.item():.4f} | y={y.item():.4f} | loss={loss.item():.6f}")

print("\nFinal w:", w.item())