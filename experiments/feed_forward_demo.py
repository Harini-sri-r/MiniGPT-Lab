import torch

from model.feed_forward import FeedForwardNetwork


batch_size = 1
sequence_length = 5
embedding_dim = 64
hidden_dim = 256

x = torch.randn(batch_size, sequence_length, embedding_dim)

feed_forward = FeedForwardNetwork(
    embedding_dim=embedding_dim,
    hidden_dim=hidden_dim,
)

output = feed_forward(x)

print("Input shape:")
print(x.shape)

print("\nOutput shape:")
print(output.shape)

print("\nEmbedding dimension:")
print(feed_forward.embedding_dim)

print("\nHidden dimension:")
print(feed_forward.hidden_dim)
