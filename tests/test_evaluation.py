import math

import torch

from evaluation.evaluate import evaluate_loss, history_records, perplexity
from model.minigpt import MiniGPT
from training.dataset import create_dataloaders


def create_model_and_loader():
    model = MiniGPT(8, 8, 2, 16, 1, 4, dropout=0.0)
    train_loader, _ = create_dataloaders(
        torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]),
        torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]),
        block_size=4,
        batch_size=2,
    )
    return model, train_loader


def test_evaluation_is_finite_and_does_not_change_parameters_or_gradients():
    model, loader = create_model_and_loader()
    original_parameters = [parameter.detach().clone() for parameter in model.parameters()]

    loss = evaluate_loss(model, loader, torch.device("cpu"))

    assert math.isfinite(loss)
    assert all(parameter.grad is None for parameter in model.parameters())
    assert all(
        torch.equal(parameter, original)
        for parameter, original in zip(model.parameters(), original_parameters)
    )


def test_perplexity_equals_exponential_of_loss():
    loss = 1.5

    assert math.isfinite(perplexity(loss))
    assert math.isclose(perplexity(loss), math.exp(loss))


def test_history_records_have_expected_fields_for_old_and_new_formats():
    old_history = {"train_loss": [3.0], "validation_loss": [4.0]}
    new_history = [{"epoch": 1, "train_loss": 3.0, "validation_loss": 4.0}]

    for history in (old_history, new_history):
        record = history_records(history)[0]
        assert set(record) == {"epoch", "train_loss", "validation_loss"}
