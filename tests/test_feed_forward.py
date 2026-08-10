import torch

from model.feed_forward import FeedForwardNetwork


def test_output_shape_matches_input_shape():
    network = FeedForwardNetwork(embedding_dim=8, hidden_dim=32)
    x = torch.randn(2, 5, 8)

    output = network(x)

    assert output.shape == x.shape


def test_multiple_sequence_positions_are_processed():
    network = FeedForwardNetwork(embedding_dim=8, hidden_dim=32)
    x = torch.randn(3, 7, 8)

    output = network(x)

    assert output.shape == (3, 7, 8)


def test_output_contains_no_nan_values():
    network = FeedForwardNetwork(embedding_dim=8, hidden_dim=32)
    x = torch.randn(2, 5, 8)

    output = network(x)

    assert not torch.isnan(output).any()
