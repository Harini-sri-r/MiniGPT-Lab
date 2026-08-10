import pytest
import torch

from model.multi_head_attention import MultiHeadAttention


def test_output_shape_matches_input_shape():
    attention = MultiHeadAttention(embedding_dim=8, num_heads=2)
    x = torch.randn(1, 5, 8)

    output = attention(x)

    assert output.shape == x.shape


def test_rejects_embedding_dimension_not_divisible_by_heads():
    with pytest.raises(ValueError, match="divisible"):
        MultiHeadAttention(embedding_dim=7, num_heads=2)


def test_causal_mask_gives_future_tokens_zero_attention():
    attention = MultiHeadAttention(embedding_dim=8, num_heads=2)
    x = torch.randn(1, 4, 8)

    _, attention_weights = attention(x, return_attention_weights=True)

    # Entries above the diagonal represent attention to future tokens.
    future_positions = torch.triu(torch.ones(4, 4, dtype=torch.bool), diagonal=1)

    assert torch.equal(
        attention_weights[..., future_positions],
        torch.zeros_like(attention_weights[..., future_positions]),
    )
