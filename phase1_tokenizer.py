import torch

text = "hello ikinya ai"

# vocab
chars = sorted(list(set(text)))
stoi = {ch: i for i, ch in enumerate(chars)}
tokens = torch.tensor([stoi[ch] for ch in text])

vocab_size = len(chars)
embedding_dim = 8
sequence_length = len(tokens)

# token embedding
token_embedding = torch.nn.Embedding(vocab_size, embedding_dim)

# positional embedding
position_embedding = torch.nn.Embedding(sequence_length, embedding_dim)

positions = torch.arange(sequence_length)

# combine
x = token_embedding(tokens) + position_embedding(positions)

print("Final input shape:", x.shape)
print("First combined vector:\n", x[0])