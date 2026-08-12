import torch
import torch.nn as nn

from model.minigpt import MiniGPT
from training.dataset import NextTokenDataset, create_dataloaders
from training.train import calculate_loss, validate


def create_tiny_model():
    return MiniGPT(
        vocab_size=8,
        embedding_dim=8,
        num_heads=2,
        hidden_dim=16,
        num_layers=1,
        max_sequence_length=4,
        dropout=0.0,
    )


def test_model_and_dataset_produce_logits_and_loss():
    model = create_tiny_model()
    dataset = NextTokenDataset(torch.tensor([0, 1, 2, 3, 4, 5, 6]), block_size=4)
    input_ids, targets = dataset[0]

    logits, loss = calculate_loss(
        model, input_ids.unsqueeze(0), targets.unsqueeze(0), nn.CrossEntropyLoss()
    )

    assert logits.shape == (1, 4, 8)
    assert torch.isfinite(loss)


def test_one_training_step_produces_gradients_and_updates_parameters():
    model = create_tiny_model()
    input_ids = torch.tensor([[0, 1, 2, 3]])
    targets = torch.tensor([[1, 2, 3, 4]])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    original_weight = model.token_embedding.embedding.weight.detach().clone()

    optimizer.zero_grad()
    _, loss = calculate_loss(model, input_ids, targets, nn.CrossEntropyLoss())
    loss.backward()

    assert model.token_embedding.embedding.weight.grad is not None
    optimizer.step()
    assert not torch.equal(model.token_embedding.embedding.weight, original_weight)


def test_validation_runs_without_creating_gradients():
    model = create_tiny_model()
    loader, _ = create_dataloaders(
        torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]),
        torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]),
        block_size=4,
        batch_size=2,
    )

    loss = validate(model, loader, nn.CrossEntropyLoss(), torch.device("cpu"))

    assert torch.isfinite(torch.tensor(loss))
    assert all(parameter.grad is None for parameter in model.parameters())
