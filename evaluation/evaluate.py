from pathlib import Path
import math

import torch
import torch.nn as nn

from model.minigpt import MiniGPT
from tokenizer.tokenizer import SimpleTokenizer
from training.train import calculate_loss


def perplexity(loss):
    """Convert average next-token loss into perplexity."""
    return math.exp(loss)


def evaluate_loss(model, data_loader, device):
    """Calculate average cross-entropy without gradients or weight updates."""
    criterion = nn.CrossEntropyLoss()
    was_training = model.training
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for input_ids, targets in data_loader:
            _, loss = calculate_loss(
                model, input_ids.to(device), targets.to(device), criterion
            )
            total_loss += loss.item()

    if was_training:
        model.train()
    return total_loss / len(data_loader)


def load_trained_model(checkpoint_path, device):
    """Rebuild the trained MiniGPT and tokenizer from its saved checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    configuration = checkpoint["configuration"]
    tokenizer = SimpleTokenizer("".join(checkpoint["tokenizer_characters"]))
    model = MiniGPT(
        vocab_size=len(tokenizer.characters),
        embedding_dim=configuration["embedding_dim"],
        num_heads=configuration["num_heads"],
        hidden_dim=configuration["hidden_dim"],
        num_layers=configuration["num_layers"],
        max_sequence_length=configuration["block_size"],
        dropout=configuration["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, tokenizer, configuration, checkpoint


def model_report(model, device):
    """Return concise, serializable model facts for an experiment report."""
    parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return {
        "vocabulary_size": model.vocab_size,
        "embedding_dimension": model.embedding_dim,
        "heads": model.num_heads,
        "hidden_dimension": model.transformer.hidden_dim,
        "layers": model.num_layers,
        "max_sequence_length": model.max_sequence_length,
        "parameters": parameters,
        "trainable_parameters": trainable_parameters,
        "device": str(device),
    }


def history_records(history):
    """Normalize both current and earlier checkpoint history formats."""
    if isinstance(history, list):
        return history
    return [
        {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
        }
        for epoch, (train_loss, validation_loss) in enumerate(
            zip(history["train_loss"], history["validation_loss"])
        )
    ]
