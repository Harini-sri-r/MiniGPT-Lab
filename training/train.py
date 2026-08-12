from pathlib import Path

import torch
import torch.nn as nn


def get_device():
    """Use CUDA when available, otherwise run entirely on CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def calculate_loss(model, input_ids, targets, criterion):
    """Flatten batch and sequence positions for CrossEntropyLoss."""
    logits = model(input_ids)
    vocabulary_size = logits.size(-1)
    loss = criterion(logits.reshape(-1, vocabulary_size), targets.reshape(-1))
    return logits, loss


def validate(model, validation_loader, criterion, device):
    """Return average validation loss without constructing gradient graphs."""
    was_training = model.training
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for input_ids, targets in validation_loader:
            _, loss = calculate_loss(
                model, input_ids.to(device), targets.to(device), criterion
            )
            total_loss += loss.item()

    if was_training:
        model.train()
    return total_loss / len(validation_loader)


def train_model(model, train_loader, validation_loader, num_epochs, learning_rate, device):
    """Train a model for a small number of epochs and return loss history."""
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    history = []

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0

        for input_ids, targets in train_loader:
            input_ids = input_ids.to(device)
            targets = targets.to(device)

            # Clear old gradients, calculate new ones, then update weights.
            optimizer.zero_grad()
            _, loss = calculate_loss(model, input_ids, targets, criterion)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        train_loss = total_train_loss / len(train_loader)
        validation_loss = validate(model, validation_loader, criterion, device)
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
            }
        )
        print(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"Train loss: {train_loss:.4f} | Validation loss: {validation_loss:.4f}"
        )

    return history, optimizer


def save_checkpoint(path, model, optimizer, configuration, tokenizer, history):
    """Save enough state to inspect or resume this educational training run."""
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "configuration": configuration,
            "tokenizer_characters": tokenizer.characters,
            "training_history": history,
        },
        checkpoint_path,
    )
