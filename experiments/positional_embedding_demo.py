import torch

from model.embeddings import TokenEmbedding
from model.positional_embeddings import PositionalEmbedding


vocab_size = 8
embedding_dim = 4
sequence_length = 5


token_embedding = TokenEmbedding(
    vocab_size,
    embedding_dim
)

position_embedding = PositionalEmbedding(
    sequence_length,
    embedding_dim
)


token_ids = torch.tensor([
    [3, 2, 4, 4, 5]
])


token_vectors = token_embedding(token_ids)

final_vectors = position_embedding(token_vectors)


print("Token IDs:")
print(token_ids)

print("\nToken embedding shape:")
print(token_vectors.shape)

print("\nFinal embedding shape:")
print(final_vectors.shape)

print("\nFinal embeddings:")
print(final_vectors)