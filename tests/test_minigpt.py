import pytest
import torch

from model.minigpt import MiniGPT


VOCAB_SIZE = 13


def create_model(max_sequence_length=16):
    return MiniGPT(
        vocab_size=VOCAB_SIZE,
        embedding_dim=8,
        num_heads=2,
        hidden_dim=32,
        num_layers=2,
        max_sequence_length=max_sequence_length,
        dropout=0.0,
    )


@pytest.mark.parametrize("batch_size,sequence_length", [(1, 1), (2, 5), (4, 7)])
def test_logits_shape_for_different_batch_and_sequence_sizes(
    batch_size, sequence_length
):
    model = create_model()
    input_ids = torch.randint(VOCAB_SIZE, (batch_size, sequence_length))

    logits = model(input_ids)

    assert logits.shape == (batch_size, sequence_length, VOCAB_SIZE)


def test_sequence_too_long_raises_clear_error():
    model = create_model(max_sequence_length=4)
    input_ids = torch.randint(VOCAB_SIZE, (1, 5))

    with pytest.raises(ValueError, match="maximum sequence length"):
        model(input_ids)


def test_logits_contain_no_nan_values():
    logits = create_model()(torch.randint(VOCAB_SIZE, (2, 5)))

    assert not torch.isnan(logits).any()


def test_model_has_trainable_parameters_and_tied_weights():
    model = create_model()

    assert sum(p.numel() for p in model.parameters() if p.requires_grad) > 0
    assert model.lm_head.weight is model.token_embedding.embedding.weight


def test_future_tokens_do_not_change_earlier_logits():
    model = create_model()
    model.eval()
    input_a = torch.tensor([[1, 2, 3, 4]])
    input_b = torch.tensor([[1, 2, 9, 4]])

    logits_a = model(input_a)
    logits_b = model(input_b)

    assert torch.allclose(logits_a[:, :2], logits_b[:, :2])
