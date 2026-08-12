import torch

from model.feed_forward import FeedForwardNetwork


def test_output_shape_matches_input_shape():
    network = FeedForwardNetwork(embedding_dim=8, hidden_dim=32)
    x = torch.randn(2, 5, 8)

    output = network(x)

    assert output.shape == x.shape


def test_supports_multiple_batch_sizes():
    network = FeedForwardNetwork(embedding_dim=8, hidden_dim=32)

    for batch_size in (1, 3, 5):
        x = torch.randn(batch_size, 5, 8)
        output = network(x)

        assert output.shape == (batch_size, 5, 8)


def test_supports_multiple_sequence_lengths():
    network = FeedForwardNetwork(embedding_dim=8, hidden_dim=32)

    for sequence_length in (1, 5, 7):
        x = torch.randn(2, sequence_length, 8)
        output = network(x)

        assert output.shape == (2, sequence_length, 8)


def test_output_contains_no_nan_values():
    network = FeedForwardNetwork(embedding_dim=8, hidden_dim=32)
    x = torch.randn(2, 5, 8)

    output = network(x)

    assert not torch.isnan(output).any()


def test_uses_embedding_dimension_for_input_and_output():
    network = FeedForwardNetwork(embedding_dim=8, hidden_dim=32)

    assert network.input_projection.in_features == 8
    assert network.input_projection.out_features == 32
    assert network.output_projection.in_features == 32
    assert network.output_projection.out_features == 8
