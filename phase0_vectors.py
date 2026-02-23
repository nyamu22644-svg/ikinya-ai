import numpy as np

def relu(x):
    return np.maximum(0, x)

x = np.array([1, 2])
target = np.array([10, 5])

# We'll only tweak ONE weight to observe effect
w = 1.0  # this will replace W2[0,0]

def compute_loss(w_value):
    W1 = np.array([[2.0, 0.0],
                   [0.0, -3.0]])
    b1 = np.array([1.0, -1.0])

    W2 = np.array([[w_value, 1.0],
                   [2.0, 0.0]])
    b2 = np.array([0.0, 1.0])

    z1 = W1 @ x + b1
    a1 = relu(z1)

    z2 = W2 @ a1 + b2
    output = relu(z2)

    error = target - output
    return np.sum(error ** 2)

# Original loss
original_loss = compute_loss(w)

# Slightly increase w
loss_plus = compute_loss(w + 0.01)

# Slightly decrease w
loss_minus = compute_loss(w - 0.01)

print("Original loss:", original_loss)
print("Loss if w increases:", loss_plus)
print("Loss if w decreases:", loss_minus)

learning_rate = 0.1

# Approximate gradient
gradient = (loss_plus - loss_minus) / (2 * 0.01)

# Update weight
w_new = w - learning_rate * gradient

print("Estimated gradient:", gradient)
print("New weight after update:", w_new)
print("New loss:", compute_loss(w_new))