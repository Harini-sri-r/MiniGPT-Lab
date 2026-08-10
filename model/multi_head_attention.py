import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):

    def __init__(self, embedding_dim, num_heads):
        super().__init__()

        if embedding_dim % num_heads != 0:
            raise ValueError(
                "embedding_dim must be divisible by num_heads "
                f"(got embedding_dim={embedding_dim}, num_heads={num_heads})."
            )

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads

        self.head_dim = embedding_dim // num_heads

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

        self.output_projection = nn.Linear(
            embedding_dim,
            embedding_dim
        )

    def forward(self, x, return_attention_weights=False):
        """Apply causal self-attention to ``x``.

        Set ``return_attention_weights`` to True when inspecting the causal
        mask in an experiment or test.
        """

        batch_size, sequence_length, _ = x.shape

        # Create one query, key, and value vector for every input token.
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # Split into multiple heads
        Q = Q.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim
        )

        K = K.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim
        )

        V = V.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim
        )

        # Move heads before sequence dimension
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Compare each query with every key inside each attention head.
        scores = Q @ K.transpose(-2, -1)

        # Scaling keeps score magnitudes stable as each head gets wider.
        scores = scores / (self.head_dim ** 0.5)

        # Block positions above the diagonal: those are future tokens.
        mask = torch.triu(
            torch.ones(
                sequence_length,
                sequence_length,
                device=x.device
            ),
            diagonal=1
        ).bool()

        scores = scores.masked_fill(
            mask,
            float("-inf")
        )

        # Convert each row of allowed scores into probabilities.
        attention_weights = F.softmax(
            scores,
            dim=-1
        )

        # Use the probabilities to take a weighted combination of values.
        output = attention_weights @ V

        # Move heads beside one another to restore embedding_dim.
        output = output.transpose(1, 2)

        output = output.contiguous().view(
            batch_size,
            sequence_length,
            self.embedding_dim
        )

        # Final projection
        output = self.output_projection(output)

        if return_attention_weights:
            return output, attention_weights

        return output
