import torch

# create a weight (requires gradient tracking)
w = torch.tensor(1.0, requires_grad=True)

# simple input + target
x = torch.tensor(2.0)
target = torch.tensor(10.0)

# forward pass
y = w * x

# loss (mean squared error)
loss = (y - target) ** 2

print("Output:", y.item())
print("Loss:", loss.item())

# backward pass (compute gradient)
loss.backward()

print("Gradient (dLoss/dw):", w.grad.item())