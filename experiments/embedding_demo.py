import torch

from model.embeddings import TokenEmbedding


# Our vocabulary contains 8 characters
vocab_size = 8

# Each token will be represented by a vector of 4 numbers
embedding_dim = 4


embedding = TokenEmbedding(
    vocab_size,
    embedding_dim
)


# Token IDs from our tokenizer
token_ids = torch.tensor([
    3, 2, 4, 4, 5
])


vectors = embedding(token_ids)


print("Token IDs:")
print(token_ids)

print("\nEmbeddings:")
print(vectors)

print("\nShape:")
print(vectors.shape)