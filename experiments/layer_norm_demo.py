import torch

from model.layer_norm import LayerNorm


batch_size = 2
sequence_length = 5
embedding_dim = 8

x = torch.randn(batch_size, sequence_length, embedding_dim)
layer_norm = LayerNorm(embedding_dim=embedding_dim)
output = layer_norm(x)

# Inspect the first token. Default gamma=1 and beta=0 mean the output is the
# directly normalized representation.
input_token = x[0, 0]
output_token = output[0, 0]

print("Input shape:")
print(x.shape)

print("\nOutput shape:")
print(output.shape)

print("\nFirst token statistics:")
print("Input mean:", input_token.mean().item())
print("Output mean:", output_token.mean().item())
print("Input variance:", input_token.var(unbiased=False).item())
print("Output variance:", output_token.var(unbiased=False).item())
