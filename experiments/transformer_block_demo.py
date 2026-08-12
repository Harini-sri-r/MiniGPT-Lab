import torch

from model.transformer_block import TransformerBlock


batch_size = 2
sequence_length = 5
embedding_dim = 64
num_heads = 4
hidden_dim = 256
dropout = 0.1

x = torch.randn(batch_size, sequence_length, embedding_dim)
block = TransformerBlock(
    embedding_dim=embedding_dim,
    num_heads=num_heads,
    hidden_dim=hidden_dim,
    dropout=dropout,
)
output = block(x)

print("Input shape:")
print(x.shape)
print("\nOutput shape:")
print(output.shape)
print("\nEmbedding dimension:")
print(block.embedding_dim)
print("\nNumber of heads:")
print(block.num_heads)
print("\nHidden dimension:")
print(block.hidden_dim)
print("\nDropout probability:")
print(block.dropout_probability)
