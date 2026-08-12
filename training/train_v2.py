"""
MiniGPT v2 Q&A Training Pipeline

Handles training, validation, loss computation with padding masking,
gradient optimization, and checkpoint management for MiniGPT v2.

Key Features:
- Next-Token Prediction with Shifted Targets:
  Target tokens are shifted by 1 position relative to input tokens
  (Input: T_0..T_{k-1}, Target: T_1..T_k). This teaches the model to predict
  the next token at every sequence position.

- Loss Masking for Padding:
  Padding tokens (<PAD>, ID 0) are added to standardize batch tensor dimensions.
  Using `nn.CrossEntropyLoss(ignore_index=0)` ensures padding tokens are ignored
  during loss calculation and gradient backpropagation, preventing the model from
  wasting capacity learning padding patterns.

- Automatic Device Selection:
  Uses CUDA GPU if available, falling back to CPU.

- Checkpoint Management:
  Saves training history, model state, optimizer state, and configuration to
  `checkpoints/v2/minigpt_v2_best.pt`.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader


def get_device() -> torch.device:
    """
    Select CUDA GPU if available, otherwise CPU.
    
    Returns:
        torch.device object
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    return device


def calculate_loss(
    model: nn.Module,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    criterion: nn.Module,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute forward pass and loss with padding tokens ignored.
    
    Args:
        model: MiniGPTV2 model instance
        input_ids: Tensor of shape (batch_size, sequence_length)
        targets: Tensor of shape (batch_size, sequence_length)
        criterion: CrossEntropyLoss with ignore_index=0 (<PAD>)
        
    Returns:
        Tuple of (logits, loss)
    """
    # Forward pass: (batch, sequence) -> (batch, sequence, vocab_size)
    logits = model(input_ids)
    vocab_size = logits.size(-1)
    
    # Flatten batch and sequence dimensions for CrossEntropyLoss
    # Logits shape: (batch_size * sequence_length, vocab_size)
    # Targets shape: (batch_size * sequence_length)
    flat_logits = logits.view(-1, vocab_size)
    flat_targets = targets.view(-1)
    
    loss = criterion(flat_logits, flat_targets)
    return logits, loss


def validate_v2(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """
    Calculate average validation loss without updating model weights.
    
    Args:
        model: MiniGPTV2 model instance
        val_loader: Validation DataLoader
        criterion: Loss function
        device: PyTorch device
        
    Returns:
        Average validation loss
    """
    was_training = model.training
    model.eval()
    total_val_loss = 0.0
    num_batches = len(val_loader)
    
    if num_batches == 0:
        return 0.0

    with torch.no_grad():
        for input_ids, targets in val_loader:
            input_ids = input_ids.to(device)
            targets = targets.to(device)
            
            _, loss = calculate_loss(model, input_ids, targets, criterion)
            total_val_loss += loss.item()

    if was_training:
        model.train()
        
    return total_val_loss / num_batches


def save_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: AdamW,
    config: dict,
    tokenizer_info: dict,
    epoch: int,
    train_loss: float,
    val_loss: float,
    history: List[dict],
):
    """
    Save training checkpoint state to disk.
    
    Args:
        checkpoint_path: Target path for checkpoint file (.pt)
        model: MiniGPTV2 model instance
        optimizer: AdamW optimizer instance
        config: Model configuration dictionary
        tokenizer_info: Tokenizer metadata (vocab size, merges, etc.)
        epoch: Current epoch number
        train_loss: Current training loss
        val_loss: Current validation loss
        history: Complete training history list
    """
    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint_data = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "config": config,
        "tokenizer_info": tokenizer_info,
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "history": history,
    }
    
    torch.save(checkpoint_data, path)


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[AdamW] = None,
    device: Optional[torch.device] = None,
) -> dict:
    """
    Load model state and checkpoint metadata from disk.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: MiniGPTV2 model instance
        optimizer: Optional AdamW optimizer to restore state
        device: PyTorch device
        
    Returns:
        Loaded checkpoint dictionary
    """
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint file not found at: {checkpoint_path}")
        
    checkpoint_data = torch.load(path, map_location=device or "cpu")
    model.load_state_dict(checkpoint_data["model_state_dict"])
    
    if optimizer and checkpoint_data.get("optimizer_state_dict"):
        optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])
        
    return checkpoint_data


def train_v2_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 5,
    learning_rate: float = 1e-3,
    device: Optional[torch.device] = None,
    checkpoint_dir: str = "checkpoints/v2",
    pad_token_id: int = 0,
    tokenizer_info: Optional[dict] = None,
    max_grad_norm: float = 1.0,
) -> Tuple[List[dict], str]:
    """
    Train MiniGPT v2 model across epochs and save best-validation checkpoint.
    
    Args:
        model: MiniGPTV2 model
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        num_epochs: Number of training epochs
        learning_rate: Learning rate for AdamW optimizer
        device: PyTorch device
        checkpoint_dir: Directory to store model checkpoints
        pad_token_id: Token ID to ignore in loss calculation (default 0)
        tokenizer_info: Optional metadata for checkpoint metadata
        max_grad_norm: Maximum gradient norm for gradient clipping
        
    Returns:
        Tuple of (training_history, best_checkpoint_path)
    """
    if device is None:
        device = get_device()
        
    model.to(device)
    
    # Loss masking: ignore_index=pad_token_id so padding tokens don't affect loss
    criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    
    best_val_loss = float("inf")
    best_checkpoint_path = str(Path(checkpoint_dir) / "minigpt_v2_best.pt")
    history = []
    
    config = model.get_config() if hasattr(model, "get_config") else {}
    
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_train_loss = 0.0
        num_batches = len(train_loader)
        
        for input_ids, targets in train_loader:
            input_ids = input_ids.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            _, loss = calculate_loss(model, input_ids, targets, criterion)
            loss.backward()
            
            # Clip gradients for numerical stability
            if max_grad_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                
            optimizer.step()
            total_train_loss += loss.item()
            
        epoch_train_loss = total_train_loss / max(num_batches, 1)
        epoch_val_loss = validate_v2(model, val_loader, criterion, device)
        
        epoch_record = {
            "epoch": epoch,
            "train_loss": epoch_train_loss,
            "val_loss": epoch_val_loss,
        }
        history.append(epoch_record)
        
        print(
            f"Epoch {epoch:2d}/{num_epochs:2d} | "
            f"Train loss: {epoch_train_loss:.4f} | "
            f"Val loss: {epoch_val_loss:.4f}"
        )
        
        # Save best checkpoint
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            save_checkpoint(
                checkpoint_path=best_checkpoint_path,
                model=model,
                optimizer=optimizer,
                config=config,
                tokenizer_info=tokenizer_info or {},
                epoch=epoch,
                train_loss=epoch_train_loss,
                val_loss=epoch_val_loss,
                history=history,
            )
            
    return history, best_checkpoint_path
