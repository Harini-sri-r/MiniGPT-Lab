import torch

from model.multi_head_attention import MultiHeadAttention


batch_size = 1
sequence_length = 5
embedding_dim = 8
num_heads = 2


x = torch.randn(
    batch_size,
    sequence_length,
    embedding_dim
)


attention = MultiHeadAttention(
    embedding_dim=embedding_dim,
    num_heads=num_heads
)


output = attention(x)


print("Input shape:")
print(x.shape)

print("\nOutput shape:")
print(output.shape)

print("\nNumber of heads:")
print(attention.num_heads)

print("\nHead dimension:")
print(attention.head_dim)
