import pytest
import torch

from model.layer_norm import LayerNorm
from model.transformer import Transformer


def create_transformer(num_layers=2):
    return Transformer(
        embedding_dim=8,
        num_heads=2,
        hidden_dim=32,
        num_layers=num_layers,
        dropout=0.0,
        max_sequence_length=16,
    )


@pytest.mark.parametrize("batch_size,sequence_length", [(1, 1), (2, 5), (4, 7)])
def test_output_shape_for_different_batch_and_sequence_sizes(
    batch_size, sequence_length
):
    model = create_transformer()
    x = torch.randn(batch_size, sequence_length, 8)

    assert model(x).shape == x.shape


@pytest.mark.parametrize("num_layers", [1, 2, 4])
def test_different_numbers_of_layers_work(num_layers):
    model = create_transformer(num_layers=num_layers)

    assert len(model.blocks) == num_layers
    assert model(torch.randn(2, 5, 8)).shape == (2, 5, 8)


def test_final_layer_norm_exists():
    assert isinstance(create_transformer().final_layer_norm, LayerNorm)


def test_output_contains_no_nan_values():
    output = create_transformer()(torch.randn(2, 5, 8))

    assert not torch.isnan(output).any()


def test_causal_masking_is_active_inside_every_block():
    model = create_transformer(num_layers=4)
    x = torch.randn(1, 4, 8)
    future_positions = torch.triu(torch.ones(4, 4, dtype=torch.bool), diagonal=1)

    for block in model.blocks:
        normalized = block.attention_residual.sublayer.layer_norm(x)
        _, weights = block.attention(normalized, return_attention_weights=True)
        assert torch.equal(
            weights[..., future_positions],
            torch.zeros_like(weights[..., future_positions]),
        )


def test_future_input_changes_do_not_affect_earlier_outputs():
    model = create_transformer(num_layers=2)
    model.eval()
    x = torch.randn(1, 5, 8)
    changed_future = x.clone()
    changed_future[:, 4] += 100

    original_output = model(x)
    changed_output = model(changed_future)

    assert torch.allclose(original_output[:, :4], changed_output[:, :4])


def test_invalid_embedding_and_head_dimensions_are_rejected():
    with pytest.raises(ValueError, match="divisible"):
        Transformer(7, 2, 32, 2)
