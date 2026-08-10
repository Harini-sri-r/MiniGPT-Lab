import torch
import torch.nn as nn


class PositionalEmbedding(nn.Module):

    def __init__(self, max_sequence_length, embedding_dim):
        super().__init__()

        self.position_embedding = nn.Embedding(
            max_sequence_length,
            embedding_dim
        )

    def forward(self, token_embeddings):
        sequence_length = token_embeddings.size(1)

        positions = torch.arange(
            sequence_length,
            device=token_embeddings.device
        )

        return token_embeddings + self.position_embedding(positions)