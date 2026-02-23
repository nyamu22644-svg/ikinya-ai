import torch

torch.manual_seed(0)

# XOR dataset
X = torch.tensor([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0],
])
y = torch.tensor([
    [0.0],
    [1.0],
    [1.0],
    [0.0],
])

# 2-layer network parameters
W1 = torch.randn(2, 4, requires_grad=True)  # 2 -> 4 hidden
b1 = torch.zeros(4, requires_grad=True)

W2 = torch.randn(4, 1, requires_grad=True)  # 4 -> 1
b2 = torch.zeros(1, requires_grad=True)

lr = 0.1

def sigmoid(z):
    return 1 / (1 + torch.exp(-z))

for step in range(1, 5001):
    # forward
    z1 = X @ W1 + b1
    a1 = torch.relu(z1)

    z2 = a1 @ W2 + b2
    y_hat = sigmoid(z2)

    # binary cross-entropy loss
    eps = 1e-7
    loss = -(y * torch.log(y_hat + eps) + (1 - y) * torch.log(1 - y_hat + eps)).mean()

    # backward
    loss.backward()

    # update
    with torch.no_grad():
        W1 -= lr * W1.grad
        b1 -= lr * b1.grad
        W2 -= lr * W2.grad
        b2 -= lr * b2.grad

    W1.grad.zero_()
    b1.grad.zero_()
    W2.grad.zero_()
    b2.grad.zero_()

    if step in [1, 10, 100, 1000, 5000]:
        print(f"step {step:4d} | loss={loss.item():.6f}")

# final predictions
with torch.no_grad():
    z1 = X @ W1 + b1
    a1 = torch.relu(z1)
    y_hat = sigmoid(a1 @ W2 + b2)

print("\nPredictions:")
print(y_hat.round())
print("\nRaw probs:")
print(y_hat)