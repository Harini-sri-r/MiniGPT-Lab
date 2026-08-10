import torch.nn as nn


class FeedForwardNetwork(nn.Module):
    """Apply the same small neural network independently to each token."""

    def __init__(self, embedding_dim, hidden_dim):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        # Expand each token representation so the network can learn richer
        # feature combinations before returning to the original width.
        self.input_projection = nn.Linear(embedding_dim, hidden_dim)
        self.activation = nn.GELU()
        self.output_projection = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, x):
        # x has shape (batch_size, sequence_length, embedding_dim). nn.Linear
        # operates on the final dimension, so every sequence position is
        # processed independently with the same learned weights.
        hidden = self.input_projection(x)
        activated = self.activation(hidden)
        return self.output_projection(activated)
