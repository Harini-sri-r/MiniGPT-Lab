import torch

from model.transformer import Transformer


batch_size = 2
sequence_length = 10
embedding_dim = 64
num_heads = 4
hidden_dim = 256
num_layers = 4
dropout = 0.1
max_sequence_length = 128

x = torch.randn(batch_size, sequence_length, embedding_dim)
model = Transformer(
    embedding_dim=embedding_dim,
    num_heads=num_heads,
    hidden_dim=hidden_dim,
    num_layers=num_layers,
    dropout=dropout,
    max_sequence_length=max_sequence_length,
)
output = model(x)
trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

print("Input shape:")
print(x.shape)
print("\nOutput shape:")
print(output.shape)
print("\nNumber of layers:")
print(model.num_layers)
print("\nEmbedding dimension:")
print(model.embedding_dim)
print("\nNumber of heads:")
print(model.num_heads)
print("\nHidden dimension:")
print(model.hidden_dim)
print("\nTotal trainable parameters:")
print(trainable_parameters)
