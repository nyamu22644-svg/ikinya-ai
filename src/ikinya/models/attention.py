import torch
import torch.nn.functional as F
import math


def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q, K, V: (T, d_model)
    mask: (T, T) lower triangular mask for causal attention
    """

    d_model = Q.size(-1)

    # Scaled dot product
    scores = (Q @ K.T) / math.sqrt(d_model)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    weights = F.softmax(scores, dim=-1)

    output = weights @ V

    return output, weights