import torch
import torch.nn.functional as F
import math


def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Supports:
      - unbatched: Q,K,V (T, d)
      - batched:   Q,K,V (..., T, d)  e.g. (B, H, T, d)

    mask:
      - (T, T) or broadcastable to (..., T, T)
      - mask should be 1 for allowed, 0 for blocked
    """
    d = Q.size(-1)

    # (..., T, T)
    scores = (Q @ K.transpose(-2, -1)) / math.sqrt(d)

    if mask is not None:
        # broadcast mask to scores shape
        scores = scores.masked_fill(mask == 0, float("-inf"))

    weights = F.softmax(scores, dim=-1)  # (..., T, T)
    output = weights @ V                # (..., T, d)
    return output, weights