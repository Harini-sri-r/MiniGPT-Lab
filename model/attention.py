import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):

    def __init__(self, embedding_dim):
        super().__init__()

        self.query = nn.Linear(
            embedding_dim,
            embedding_dim
        )

        self.key = nn.Linear(
            embedding_dim,
            embedding_dim
        )

        self.value = nn.Linear(
            embedding_dim,
            embedding_dim
        )

    def forward(self, x):

        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # Calculate attention scores
        scores = Q @ K.transpose(-2, -1)

        # Scale the scores
        scores = scores / (K.size(-1) ** 0.5)

        # Create causal mask
        sequence_length = x.size(1)

        mask = torch.triu(
            torch.ones(
                sequence_length,
                sequence_length,
                device=x.device
            ),
            diagonal=1
        ).bool()

        # Hide future tokens
        scores = scores.masked_fill(
            mask,
            float("-inf")
        )

        # Convert scores to probabilities
        attention_weights = F.softmax(
            scores,
            dim=-1
        )

        # Weighted combination of values
        output = attention_weights @ V

        return output