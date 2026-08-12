import torch

from model.transformer_block import TransformerBlock


def create_block(dropout=0.0):
    return TransformerBlock(
        embedding_dim=8,
        num_heads=2,
        hidden_dim=32,
        dropout=dropout,
        max_sequence_length=16,
    )


def test_output_shape_matches_input_shape_for_batch_and_sequence_sizes():
    for batch_size, sequence_length in [(1, 1), (2, 5), (4, 7)]:
        block = create_block()
        x = torch.randn(batch_size, sequence_length, 8)

        assert block(x).shape == x.shape


def test_configuration_is_stored_correctly():
    block = create_block()

    assert block.embedding_dim == 8
    assert block.num_heads == 2


def test_output_contains_no_nan_values():
    block = create_block()
    output = block(torch.randn(2, 5, 8))

    assert not torch.isnan(output).any()


def test_causal_attention_remains_active():
    block = create_block()
    x = torch.randn(1, 4, 8)
    normalized = block.attention_residual.sublayer.layer_norm(x)

    _, weights = block.attention(normalized, return_attention_weights=True)
    future_positions = torch.triu(torch.ones(4, 4, dtype=torch.bool), diagonal=1)

    assert torch.equal(
        weights[..., future_positions],
        torch.zeros_like(weights[..., future_positions]),
    )


def test_evaluation_mode_disables_dropout():
    block = create_block(dropout=0.5)
    x = torch.randn(2, 5, 8)
    block.eval()

    assert torch.equal(block(x), block(x))


def test_dropout_changes_training_outputs_when_appropriate():
    torch.manual_seed(0)
    block = create_block(dropout=0.5)
    x = torch.randn(2, 5, 8)
    block.train()

    first_output = block(x)
    second_output = block(x)

    assert not torch.equal(first_output, second_output)
