import torch.nn as nn

from model.layer_norm import LayerNorm
from model.transformer_block import TransformerBlock


class Transformer(nn.Module):
    """A sequential stack of decoder-only Transformer blocks."""

    def __init__(
        self,
        embedding_dim,
        num_heads,
        hidden_dim,
        num_layers,
        dropout=0.1,
        max_sequence_length=128,
    ):
        super().__init__()

        if embedding_dim % num_heads != 0:
            raise ValueError(
                "embedding_dim must be divisible by num_heads "
                f"(got embedding_dim={embedding_dim}, num_heads={num_heads})."
            )
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1.")

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # ModuleList registers every block's parameters while letting forward
        # explicitly apply blocks one after another.
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                    max_sequence_length=max_sequence_length,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_layer_norm = LayerNorm(embedding_dim)

    def forward(self, x):
        # Each block refines the representation while preserving its shape.
        for block in self.blocks:
            x = block(x)

        # Normalize the final representation before it is passed to a future
        # language-model output head.
        return self.final_layer_norm(x)
