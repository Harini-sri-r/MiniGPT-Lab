import pytest
import torch
import torch.nn as nn

from model.residual import ResidualConnection


class DoubleSublayer(nn.Module):
    def forward(self, x):
        return 2 * x


class WrongShapeSublayer(nn.Module):
    def forward(self, x):
        return x[..., :-1]


@pytest.mark.parametrize("batch_size,sequence_length", [(1, 1), (2, 5), (4, 7)])
def test_output_shape_matches_input_for_batch_and_sequence_sizes(
    batch_size, sequence_length
):
    residual = ResidualConnection(DoubleSublayer())
    x = torch.randn(batch_size, sequence_length, 8)

    output = residual(x)

    assert output.shape == x.shape


def test_residual_output_is_input_plus_sublayer_output():
    residual = ResidualConnection(DoubleSublayer())
    x = torch.tensor([[[1.0, 2.0, 3.0]]])

    output = residual(x)

    assert torch.equal(output, torch.tensor([[[3.0, 6.0, 9.0]]]))


def test_shape_mismatch_raises_clear_error():
    residual = ResidualConnection(WrongShapeSublayer())
    x = torch.randn(2, 5, 8)

    with pytest.raises(ValueError, match="matching shapes"):
        residual(x)


def test_output_contains_no_nan_values():
    residual = ResidualConnection(nn.Linear(8, 8))
    x = torch.randn(2, 5, 8)

    output = residual(x)

    assert not torch.isnan(output).any()
