import torch.nn as nn

from model.feed_forward import FeedForwardNetwork
from model.layer_norm import LayerNorm
from model.multi_head_attention import MultiHeadAttention
from model.residual import ResidualConnection


class _PreNormSublayer(nn.Module):
    """Normalize, transform, then apply dropout before a residual addition."""

    def __init__(self, layer_norm, sublayer, dropout):
        super().__init__()
        self.layer_norm = layer_norm
        self.sublayer = sublayer
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        normalized = self.layer_norm(x)
        sublayer_output = self.sublayer(normalized)
        return self.dropout(sublayer_output)


class TransformerBlock(nn.Module):
    """One pre-norm decoder-only Transformer block."""

    def __init__(
        self,
        embedding_dim,
        num_heads,
        hidden_dim,
        dropout=0.1,
        max_sequence_length=128,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout_probability = dropout
        self.max_sequence_length = max_sequence_length

        # Each ResidualConnection performs x + its pre-normalized sublayer.
        self.attention_residual = ResidualConnection(
            _PreNormSublayer(
                LayerNorm(embedding_dim),
                MultiHeadAttention(embedding_dim, num_heads),
                dropout,
            )
        )
        self.feed_forward_residual = ResidualConnection(
            _PreNormSublayer(
                LayerNorm(embedding_dim),
                FeedForwardNetwork(embedding_dim, hidden_dim),
                dropout,
            )
        )

    @property
    def attention(self):
        """Expose causal attention for educational inspection and tests."""
        return self.attention_residual.sublayer.sublayer

    def forward(self, x):
        if x.size(1) > self.max_sequence_length:
            raise ValueError(
                "sequence length exceeds max_sequence_length "
                f"({x.size(1)} > {self.max_sequence_length})."
            )

        # x = x + Dropout(Attention(LayerNorm(x)))
        x = self.attention_residual(x)

        # x = x + Dropout(FeedForward(LayerNorm(x)))
        x = self.feed_forward_residual(x)

        return x
