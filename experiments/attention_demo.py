import torch

from model.attention import SelfAttention


batch_size = 1
sequence_length = 5
embedding_dim = 4


x = torch.randn(
    batch_size,
    sequence_length,
    embedding_dim
)


attention = SelfAttention(
    embedding_dim
)


output = attention(x)


print("Input shape:")
print(x.shape)

print("\nOutput shape:")
print(output.shape)

print("\nOutput:")
print(output)