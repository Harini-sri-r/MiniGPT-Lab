import torch.nn as nn

from model.embeddings import TokenEmbedding
from model.positional_embeddings import PositionalEmbedding
from model.transformer import Transformer


class MiniGPT(nn.Module):
    """A small decoder-only language model that returns vocabulary logits."""

    def __init__(
        self,
        vocab_size,
        embedding_dim,
        num_heads,
        hidden_dim,
        num_layers,
        max_sequence_length,
        dropout=0.1,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_sequence_length = max_sequence_length

        self.token_embedding = TokenEmbedding(vocab_size, embedding_dim)
        self.positional_embedding = PositionalEmbedding(
            max_sequence_length, embedding_dim
        )
        self.transformer = Transformer(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            max_sequence_length=max_sequence_length,
        )

        # Convert each final token representation to one score per vocabulary
        # token. Sharing this matrix with token embeddings reduces parameters
        # and connects input and output token representations.
        self.lm_head = nn.Linear(embedding_dim, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.embedding.weight

    def forward(self, input_ids):
        sequence_length = input_ids.size(1)
        if sequence_length > self.max_sequence_length:
            raise ValueError(
                "Sequence length exceeds the configured maximum sequence length."
            )

        # (batch, sequence) -> (batch, sequence, embedding_dim)
        x = self.token_embedding(input_ids)
        x = self.positional_embedding(x)
        x = self.transformer(x)

        # The result contains raw logits. Softmax belongs in loss calculation
        # or sampling later, not inside the model forward pass.
        return self.lm_head(x)
