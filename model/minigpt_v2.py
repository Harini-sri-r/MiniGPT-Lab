"""
MiniGPT v2 — BPE Token-Based Language Model

MiniGPT v2 extends the core MiniGPT architecture to work with subword tokens
from a Byte Pair Encoding (BPE) tokenizer instead of character-level tokens.

The architecture is identical to v1 but operates on:
- BPE token IDs (0 to vocab_size-1)
- Larger vocabulary (typically 256-1024 tokens vs. ~100 for characters)
- Domain-specific Q&A corpus

Architecture:

BPE Token IDs
    ↓
Token Embedding (vocab_size → embedding_dim)
    ↓
Positional Embedding (max_sequence_length → embedding_dim)
    ↓
Transformer Blocks (4 layers by default)
    ├─ LayerNorm → Multi-Head Causal Attention → Residual → Dropout
    └─ LayerNorm → Feed-Forward → Residual → Dropout
    ↓
Final LayerNorm
    ↓
LM Head (embedding_dim → vocab_size)
    ↓
Logits (scores for each BPE token)

Key design:
- Weight tying: embedding matrix is shared with LM head weights
- Causal masking: future tokens cannot attend to past
- Deterministic output: same input always produces same logits in eval mode
"""

import torch
import torch.nn as nn

from model.embeddings import TokenEmbedding
from model.positional_embeddings import PositionalEmbedding
from model.transformer import Transformer


class MiniGPTV2(nn.Module):
    """
    BPE token-based language model.
    
    Predicts next BPE token given a sequence of BPE token IDs.
    
    Args:
        vocab_size: Size of BPE vocabulary (typically 256-1024)
        embedding_dim: Dimension of token and positional embeddings
        num_heads: Number of attention heads (must divide embedding_dim)
        hidden_dim: Dimension of feed-forward intermediate layer
        num_layers: Number of Transformer blocks
        max_sequence_length: Maximum sequence length
        dropout: Dropout probability
    """

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
        self.hidden_dim = hidden_dim
        self.max_sequence_length = max_sequence_length
        self.dropout = dropout

        # Token embedding: maps discrete categorical token IDs to dense continuous
        # vector representations. Discrete IDs (e.g. 0, 1, 2... 511) have no inherent
        # numerical distance or ordering (token 50 is not "50 times token 1").
        # Embeddings place tokens in a continuous vector space where the model can learn
        # semantic geometric relationships between tokens.
        self.token_embedding = TokenEmbedding(vocab_size, embedding_dim)
        
        # Positional embedding: adds position information to token representations so
        # the Transformer model can understand word order and relative position.
        self.positional_embedding = PositionalEmbedding(
            max_sequence_length, embedding_dim
        )
        
        # Transformer: sequential stack of decoder blocks featuring causal self-attention
        # and feed-forward networks. Causal masking ensures position i can only attend
        # to positions <= i.
        self.transformer = Transformer(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            max_sequence_length=max_sequence_length,
        )

        # LM Head: projects final hidden vector of dimension embedding_dim to logits over
        # the entire BPE vocabulary.
        # Weight tying: share weight matrix between the input token embedding layer and
        # output projection layer (self.lm_head.weight = self.token_embedding.embedding.weight).
        # Benefits of Weight Tying:
        # 1. Reduces total model parameters significantly (by vocab_size * embedding_dim).
        # 2. Connects input and output representation spaces, enforcing linguistic symmetry
        #    where tokens with similar input embeddings receive similar prediction logits.
        self.lm_head = nn.Linear(embedding_dim, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.embedding.weight

    def forward(self, input_ids):
        """
        Forward pass.
        
        Args:
            input_ids: Tensor of shape (batch_size, sequence_length)
                       Each element is a BPE token ID in range [0, vocab_size)
        
        Returns:
            logits: Tensor of shape (batch_size, sequence_length, vocab_size)
                    Unnormalized raw scores for each token in the BPE vocabulary at each position.
                    (No softmax inside forward pass; CrossEntropyLoss computes softmax internally).
        
        Raises:
            ValueError: If sequence_length exceeds max_sequence_length
        """
        sequence_length = input_ids.size(1)
        if sequence_length > self.max_sequence_length:
            raise ValueError(
                f"Sequence length ({sequence_length}) exceeds maximum sequence length ({self.max_sequence_length})."
            )

        # Token embedding: (batch, seq) -> (batch, seq, embedding_dim)
        x = self.token_embedding(input_ids)
        
        # Add positional information
        x = self.positional_embedding(x)
        
        # Process through Transformer stack with causal masking & final LayerNorm
        x = self.transformer(x)

        # Project to unnormalized raw logits: (batch, seq, vocab_size)
        logits = self.lm_head(x)
        
        return logits

    def get_parameter_count(self) -> int:
        """
        Get total number of trainable parameters.
        
        Returns:
            Total parameter count
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_config(self) -> dict:
        """
        Get model configuration dictionary.
        
        Returns:
            Dictionary with all configuration parameters
        """
        return {
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim,
            'num_heads': self.num_heads,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'max_sequence_length': self.max_sequence_length,
            'dropout': self.dropout,
        }

