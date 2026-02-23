import torch

w = torch.tensor(1.0, requires_grad=True)
x = torch.tensor(2.0)
target = torch.tensor(10.0)

lr = 0.1

# forward
y = w * x
loss = (y - target) ** 2

# backward
loss.backward()

print("Before update -> w:", w.item(), "loss:", loss.item(), "grad:", w.grad.item())

# gradient descent update (no tracking)
with torch.no_grad():
    w -= lr * w.grad

# IMPORTANT: clear gradient for next step
w.grad.zero_()

# forward again
y2 = w * x
loss2 = (y2 - target) ** 2

print("After update  -> w:", w.item(), "loss:", loss2.item())