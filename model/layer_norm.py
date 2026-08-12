import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """Normalize each token representation across its embedding dimension."""

    def __init__(self, embedding_dim, epsilon=1e-5):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.epsilon = epsilon

        # These parameters let the model learn the most useful scale and
        # offset after normalization.
        self.gamma = nn.Parameter(torch.ones(embedding_dim))
        self.beta = nn.Parameter(torch.zeros(embedding_dim))

    def forward(self, x):
        # Compute separate statistics for every token, across its final
        # embedding dimension only. Shapes: (batch, sequence, 1).
        mean = x.mean(dim=-1, keepdim=True)
        variance = x.var(dim=-1, keepdim=True, unbiased=False)

        # Center and scale each token. Epsilon keeps the division numerically
        # stable even if every feature in a token has the same value.
        normalized = (x - mean) / torch.sqrt(variance + self.epsilon)

        # gamma and beta have shape (embedding_dim,) and broadcast across the
        # batch and sequence dimensions, preserving x's original shape.
        return self.gamma * normalized + self.beta
