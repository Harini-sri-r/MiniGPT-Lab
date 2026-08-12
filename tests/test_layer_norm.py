import pytest
import torch

from model.layer_norm import LayerNorm


@pytest.mark.parametrize(
    ("batch_size", "sequence_length", "embedding_dim"),
    [(1, 1, 4), (2, 5, 8), (4, 3, 16)],
)
def test_preserves_shape_for_different_input_sizes(
    batch_size, sequence_length, embedding_dim
):
    layer_norm = LayerNorm(embedding_dim=embedding_dim)
    x = torch.randn(batch_size, sequence_length, embedding_dim)

    output = layer_norm(x)

    assert output.shape == x.shape


def test_output_contains_no_nan_values():
    layer_norm = LayerNorm(embedding_dim=8)
    x = torch.zeros(2, 5, 8)

    output = layer_norm(x)

    assert not torch.isnan(output).any()


def test_default_parameters_produce_zero_mean_per_token():
    layer_norm = LayerNorm(embedding_dim=8)
    x = torch.randn(2, 5, 8)

    output = layer_norm(x)

    assert torch.allclose(
        output.mean(dim=-1), torch.zeros(2, 5), atol=1e-6, rtol=0
    )


def test_default_parameters_produce_unit_variance_per_token():
    layer_norm = LayerNorm(embedding_dim=8)
    x = torch.randn(2, 5, 8)

    output = layer_norm(x)

    assert torch.allclose(
        output.var(dim=-1, unbiased=False),
        torch.ones(2, 5),
        atol=2e-4,
        rtol=0,
    )
