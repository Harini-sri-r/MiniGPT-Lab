import pytest
import torch
import torch.nn as nn

from generation.generate import generate


class PredictNextTokenModel(nn.Module):
    """Small deterministic model used to inspect generation behavior."""

    def __init__(self, vocab_size=5, max_sequence_length=3):
        super().__init__()
        self.max_sequence_length = max_sequence_length
        self.vocab_size = vocab_size
        self.seen_contexts = []
        self.parameter = nn.Parameter(torch.zeros(1))

    def forward(self, input_ids):
        self.seen_contexts.append(input_ids.detach().clone())
        logits = torch.full(
            (*input_ids.shape, self.vocab_size), -10.0, device=input_ids.device
        )
        next_ids = (input_ids + 1) % self.vocab_size
        logits.scatter_(-1, next_ids.unsqueeze(-1), 10.0)
        return logits


def test_greedy_generation_appends_expected_tokens():
    model = PredictNextTokenModel()

    output = generate(model, torch.tensor([[1, 2]]), max_new_tokens=3, do_sample=False)

    assert torch.equal(output, torch.tensor([[1, 2, 3, 4, 0]]))


def test_generation_crops_context_to_model_maximum_length():
    model = PredictNextTokenModel(max_sequence_length=3)

    generate(model, torch.tensor([[0, 1, 2, 3]]), max_new_tokens=2, do_sample=False)

    assert all(context.size(1) <= 3 for context in model.seen_contexts)
    assert torch.equal(model.seen_contexts[0], torch.tensor([[1, 2, 3]]))


def test_top_k_one_matches_greedy_selection():
    model = PredictNextTokenModel()
    prompt = torch.tensor([[1, 2]])

    output = generate(model, prompt, max_new_tokens=2, top_k=1, do_sample=True)

    assert torch.equal(output, torch.tensor([[1, 2, 3, 4]]))


def test_generation_restores_training_mode():
    model = PredictNextTokenModel()
    model.train()

    generate(model, torch.tensor([[1]]), max_new_tokens=1, do_sample=False)

    assert model.training


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_new_tokens": 1, "temperature": 0},
        {"max_new_tokens": 1, "top_k": 0},
        {"max_new_tokens": -1},
    ],
)
def test_invalid_generation_arguments_are_rejected(kwargs):
    with pytest.raises(ValueError):
        generate(PredictNextTokenModel(), torch.tensor([[1]]), **kwargs)
